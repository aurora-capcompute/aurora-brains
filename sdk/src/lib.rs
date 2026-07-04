//! The Aurora brain SDK: everything a guest brain needs to speak the syscall
//! boundary, so a brain crate contains only cognition. It owns the ABI v3
//! wire codec ([`wire`]), the single `extism:host/compute` syscall import, and
//! the dispatch protocol — result/failed observations, the yield sentinel,
//! [`savepoint`]s, and savepoint-bracketed "hard" calls ([`dispatch_hard`]).
//!
//! On top of that it owns the typed plumbing a brain would otherwise
//! re-implement by hand: [`input`]/[`output`] for the run's payloads, [`log`]
//! for progress, and the decoded [`Capability`] menu the host grants. What is
//! left for the brain is cognition.
//!
//! A brain is one cdylib crate under `brains/<name>/` that depends on this
//! SDK and exports its entrypoint with `#[plugin_fn]`.

use extism_pdk::Memory;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod wire;

#[link(wasm_import_module = "extism:host/compute")]
extern "C" {
    fn syscall(offset: u64) -> u64;
}

/// Syscall ABI this SDK speaks (sys.ABIVersion in capcompute); the host
/// rejects mismatches with code "bad_abi". Since v3 the envelope is protobuf
/// ([`wire`], mirroring capcompute's sys/wire); args and results stay JSON
/// payloads inside it.
pub const ABI_VERSION: u32 = 3;

/// Reserved savepoint markers (sys.SyscallBegin/sys.SyscallCommit in
/// capcompute). They carry no side effect; the host journals them and uses an
/// open sys.begin (one with no matching sys.commit) as the fork point when a
/// failed run is resumed. Brackets have stack semantics. [`dispatch_hard`]
/// wraps one call in them.
pub const SYS_BEGIN: &str = "sys.begin";
pub const SYS_COMMIT: &str = "sys.commit";

/// Reserved names for the guest↔host protocol plumbing the kernel handles
/// itself (not a dispatcher): fetch this run's input ([`input`]), publish its
/// result ([`output`]), and emit a progress line ([`log`]).
pub const SYS_INPUT: &str = "sys.input";
pub const SYS_OUTPUT: &str = "sys.output";
pub const SYS_LOG: &str = "sys.log";

/// Reserved name for rolling a critical section back: instead of finishing
/// with [`output`], [`abort`] asks the host to execute the compensations the
/// guest registered with [`compensate`], newest first, and then retry the
/// section after a delay (or stop). The backward counterpart of a crash
/// resume: a host failure re-drives a run; sys.abort deliberately undoes it.
pub const SYS_ABORT: &str = "sys.abort";

/// Reserved name for registering an effect's undo: a deferred syscall the host
/// journals (name + concrete args) but does not execute. Registered
/// compensations run — newest first — only if the section later aborts.
pub const SYS_COMPENSATE: &str = "sys.compensate";

/// Status of a [`HostResponse`]. The host reports "result" or "failed" — both
/// recoverable observations the brain can react to; "yield" never reaches the
/// caller as a response (it surfaces as [`YieldedError`]), and "unspecified"
/// covers a status the host left unset. These are the decoded-string mirror of
/// the wire status codes ([`wire::STATUS_RESULT`] and friends).
pub const STATUS_RESULT: &str = "result";
pub const STATUS_YIELD: &str = "yield";
pub const STATUS_FAILED: &str = "failed";
pub const STATUS_UNSPECIFIED: &str = "unspecified";

/// YieldedError is the yield sentinel: the host parked this run on external
/// work (an approval, a timer, a message). Bubble it up and return
/// `{"status":"yielded"}` from the entrypoint; the run resumes by replay.
#[derive(Debug)]
pub struct YieldedError;

impl std::fmt::Display for YieldedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "host yielded")
    }
}

impl std::error::Error for YieldedError {}

/// yielded reports whether an error (chain) is the yield sentinel.
pub fn yielded(err: &anyhow::Error) -> bool {
    err.downcast_ref::<YieldedError>().is_some()
}

/// Call is one syscall request: a capability (or reserved sys.*) name and its
/// JSON args.
pub struct Call {
    pub name: String,
    pub args: Option<Value>,
}

/// HostResponse is the brain's view of a syscall outcome with the JSON result
/// payload already parsed. Status is "result" or "failed" — a yield never
/// reaches the caller as a response (see [`YieldedError`]).
pub struct HostResponse {
    pub abi: u32,
    pub status: String,
    pub code: String,
    pub result: Option<Value>,
    pub message: String,
    /// Provenance labels of the result — where this data came from. Treat
    /// content labelled untrusted as untrusted.
    pub labels: Vec<String>,
}

/// dispatch sends one syscall and returns its outcome. Failures come back as
/// a response with status "failed" (recoverable by default: the brain can
/// react); a host yield is surfaced as [`YieldedError`].
pub fn dispatch(c: &Call) -> anyhow::Result<HostResponse> {
    let args = match &c.args {
        Some(value) => {
            serde_json::to_vec(value).map_err(|e| anyhow::anyhow!("encode call args: {}", e))?
        }
        None => Vec::new(),
    };
    let raw = wire::encode_syscall(&wire::Syscall {
        abi: ABI_VERSION,
        name: c.name.clone(),
        args,
    });
    let mem = Memory::from_bytes(&raw)?;
    let response_offset = unsafe { syscall(mem.offset()) };
    mem.free();
    let response_mem = Memory::find(response_offset)
        .ok_or_else(|| anyhow::anyhow!("decode host response: invalid offset"))?;
    let decoded = wire::decode_response(&response_mem.to_vec())
        .map_err(|e| anyhow::anyhow!("decode host response: {}", e))?;

    let status = match decoded.status {
        wire::STATUS_RESULT => STATUS_RESULT,
        wire::STATUS_YIELD => STATUS_YIELD,
        wire::STATUS_FAILED => STATUS_FAILED,
        _ => STATUS_UNSPECIFIED,
    };
    let result = if decoded.result.is_empty() {
        None
    } else {
        Some(
            serde_json::from_slice(&decoded.result)
                .map_err(|e| anyhow::anyhow!("decode result payload: {}", e))?,
        )
    };
    let response = HostResponse {
        abi: decoded.abi,
        status: status.to_string(),
        code: decoded.code,
        result,
        message: decoded.message,
        labels: decoded.labels,
    };
    match response.status.as_str() {
        STATUS_RESULT | STATUS_FAILED => Ok(response),
        STATUS_YIELD => Err(YieldedError.into()),
        other => Err(anyhow::anyhow!("unsupported host outcome: {}", other)),
    }
}

/// A Savepoint brackets a critical zone in a sys.begin/sys.commit pair. Open one
/// with [`savepoint`] and close it with [`Savepoint::commit`] once the zone has
/// succeeded; *dropping it without committing* leaves the sys.begin open, so a
/// resumed run forks right after it and re-executes the whole zone live. That
/// drop-aborts behavior is the point — propagate an error out of the zone (with
/// `?`) and the savepoint unwinds the run for you. Brackets have stack
/// semantics; [`dispatch_hard`] wraps a single call this way.
#[must_use = "a Savepoint aborts the run unless it is committed"]
#[non_exhaustive]
pub struct Savepoint {}

impl Savepoint {
    /// commit closes the savepoint with a sys.commit marker, keeping the zone's
    /// effects on the happy path.
    pub fn commit(self) -> anyhow::Result<()> {
        dispatch(&Call {
            name: SYS_COMMIT.into(),
            args: None,
        })?;
        Ok(())
    }
}

/// savepoint opens a sys.begin marker and returns the [`Savepoint`] guard for a
/// critical zone. See [`Savepoint`] for the commit-or-abort contract.
pub fn savepoint() -> anyhow::Result<Savepoint> {
    dispatch(&Call {
        name: SYS_BEGIN.into(),
        args: None,
    })?;
    Ok(Savepoint {})
}

/// dispatch_hard brackets a single call in a [`savepoint`]. On success it commits
/// and returns the result. On failure it leaves the begin open and returns an
/// error that aborts the run, so a later resume forks right after the begin and
/// re-executes the call under a new revision. A plain [`dispatch`] (the default,
/// "soft") instead records the failure for replay and lets the brain react to it.
pub fn dispatch_hard(c: &Call) -> anyhow::Result<HostResponse> {
    let sp = savepoint()?;
    let response = dispatch(c)?;
    if response.status == STATUS_FAILED {
        anyhow::bail!("hard capability {:?} failed: {}", c.name, response.message);
    }
    sp.commit()?;
    Ok(response)
}

/// input fetches this run's input payload with the sys.input syscall and
/// deserializes it into `T` — the typed front door a brain uses instead of
/// dispatching sys.input by hand.
pub fn input<T: DeserializeOwned>() -> anyhow::Result<T> {
    let response = dispatch(&Call {
        name: SYS_INPUT.into(),
        args: None,
    })?;
    if response.status != STATUS_RESULT {
        anyhow::bail!("host failed to provide input: {}", response.message);
    }
    let result = response
        .result
        .ok_or_else(|| anyhow::anyhow!("decode input: empty result"))?;
    serde_json::from_value(result).map_err(|e| anyhow::anyhow!("decode input: {}", e))
}

/// output publishes this run's result payload with the sys.output syscall.
pub fn output<T: Serialize>(value: &T) -> anyhow::Result<()> {
    let args = serde_json::to_value(value).map_err(|e| anyhow::anyhow!("encode output: {}", e))?;
    dispatch(&Call {
        name: SYS_OUTPUT.into(),
        args: Some(args),
    })?;
    Ok(())
}

/// compensate registers an effect's undo, to run only if the section later
/// aborts: `name` is the granted capability to dispatch and `args` its exact
/// arguments — built from the effect's result (e.g. the charge id a refund
/// needs), so the undo is concrete, not generic. Register immediately after
/// the effect succeeds. The host validates the name against the grant set and
/// journals the deferred call; a rejected registration comes back as a failed
/// response.
pub fn compensate(name: &str, args: Value) -> anyhow::Result<HostResponse> {
    dispatch(&Call {
        name: SYS_COMPENSATE.into(),
        args: Some(serde_json::json!({ "name": name, "args": args })),
    })
}

/// abort rolls the open critical section back instead of finishing with
/// [`output`]: the host executes the compensations registered with
/// [`compensate`] newest-first, journaling each. With `retry_seconds` the
/// section then retries after that delay — the journal forks at the section's
/// begin and the whole section re-executes fresh; without it the process stops
/// as compensated. The guest returns after calling abort.
pub fn abort(reason: &str, retry_seconds: Option<u64>) -> anyhow::Result<()> {
    let mut args = serde_json::json!({ "reason": reason });
    if let Some(delay) = retry_seconds {
        args["retry_seconds"] = delay.into();
    }
    dispatch(&Call {
        name: SYS_ABORT.into(),
        args: Some(args),
    })?;
    Ok(())
}

/// log emits a human-readable progress line with the sys.log syscall. Logging is
/// best-effort observability: any failure (or a host yield) is swallowed so it
/// never perturbs the run.
pub fn log(message: &str) {
    let args = serde_json::json!({ "message": message });
    let _ = dispatch(&Call {
        name: SYS_LOG.into(),
        args: Some(args),
    });
}

/// Capability is one tool the host has granted this run — the guest's decoded
/// view of capcompute's `sys.Capability`. A brain reads `name`/`description`/
/// `input_schema` to build its tool menu, and may consult `hidden` (keep a
/// dispatchable tool off that menu) and `labels`/`forbid` (the provenance a
/// result carries and the labels barred from its args). Decode-only: the host
/// owns the record. How an effect is undone is not capability metadata — the
/// guest registers concrete undos with [`compensate`].
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Capability {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
    /// Dispatchable, but excluded from the brain's discoverable tool menu.
    #[serde(default)]
    pub hidden: bool,
    /// Source classes this capability's results carry (taint labels).
    #[serde(default)]
    pub labels: Vec<String>,
    /// Labels that may not flow into this capability's args.
    #[serde(default)]
    pub forbid: Vec<String>,
}
