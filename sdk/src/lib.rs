//! The Aurora program SDK: everything a guest program needs to speak the
//! syscall boundary, so a program crate contains only cognition. It owns the
//! ABI v4 JSON envelope, the single `extism:host/compute` syscall import, and
//! the dispatch protocol — result/failed observations, the yield sentinel,
//! [`savepoint`]s, and savepoint-bracketed "hard" calls ([`dispatch_hard`]).
//!
//! On top of that it owns the typed plumbing a program would otherwise
//! re-implement by hand: [`input`]/[`output`] for the process's payloads, [`log`]
//! for progress, the decoded [`Capability`] menu the host grants, and [`llm`] —
//! the chat call and the model-reply decoding. What is left for the program is
//! cognition.
//!
//! A program is one cdylib crate under `programs/<name>/` that depends on this
//! SDK and exports its entrypoint with `#[plugin_fn]`. It ships with an
//! `interface.json` manifest — its description and input/output JSON Schemas —
//! that the host loads alongside the wasm.

pub mod llm;

use extism_pdk::Memory;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The guest-to-host envelope, mirroring capcompute's `sys.Syscall`.
#[derive(Serialize)]
struct SyscallEnvelope<'a> {
    abi: u32,
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<&'a Value>,
}

/// The host-to-guest envelope, mirroring capcompute's `sys.SyscallResult`.
/// Every field but `status` is omitted when empty, so all of them default.
#[derive(Deserialize, Default)]
struct ResponseEnvelope {
    #[serde(default)]
    status: String,
    #[serde(default)]
    code: String,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    message: String,
}

#[link(wasm_import_module = "extism:host/compute")]
extern "C" {
    fn syscall(offset: u64) -> u64;
}

/// Syscall ABI this SDK speaks (sys.ABIVersion in capcompute); the host
/// rejects mismatches with code "bad_abi". Since v4 the envelope is JSON, so
/// args and results nest directly inside it — v3's protobuf envelope needed a
/// hand-rolled codec on each side of the boundary to carry the same fields.
pub const ABI_VERSION: u32 = 4;

/// Reserved savepoint markers (sys.SyscallBegin/sys.SyscallCommit in
/// capcompute). They carry no side effect; the host journals them and uses an
/// open sys.begin (one with no matching sys.commit) as the fork point when a
/// failed run is resumed. Brackets have stack semantics. [`dispatch_hard`]
/// wraps one call in them.
pub const SYS_BEGIN: &str = "sys.begin";
pub const SYS_COMMIT: &str = "sys.commit";

/// Reserved names for the guest↔host protocol plumbing the processor handles
/// itself (not a dispatcher): fetch this run's input ([`input`]), publish its
/// result ([`output`]), and emit a progress line ([`log`]).
pub const SYS_INPUT: &str = "sys.input";
pub const SYS_OUTPUT: &str = "sys.output";
pub const SYS_LOG: &str = "sys.log";

/// Reserved name for abandoning the open critical section: instead of finishing
/// with [`output`], [`abort`] forks the journal at the section's sys.begin and
/// re-runs it from there, or stops. Nothing is undone — the section re-executes.
pub const SYS_ABORT: &str = "sys.abort";

/// Status of a [`HostResponse`]. The host reports "result" or "failed" — both
/// recoverable observations the program can react to; "yield" never reaches the
/// caller as a response (it surfaces as [`YieldedError`]), and "unspecified"
/// covers a status the host left unset. These are the status strings the
/// envelope carries verbatim (capcompute's sys.SyscallStatus).
pub const STATUS_RESULT: &str = "result";
pub const STATUS_YIELD: &str = "yield";
pub const STATUS_FAILED: &str = "failed";
pub const STATUS_UNSPECIFIED: &str = "unspecified";

/// YieldedError is the yield sentinel: the host parked this run on external
/// work (an approval, a timer, a message). Bubble it up and return
/// `{"status":"yielded"}` from the entrypoint; the process resumes by replay.
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

/// HostResponse is the program's view of a syscall outcome with the JSON result
/// payload already parsed. Status is "result" or "failed" — a yield never
/// reaches the caller as a response (see [`YieldedError`]).
pub struct HostResponse {
    pub status: String,
    pub code: String,
    pub result: Option<Value>,
    pub message: String,
}

/// dispatch sends one syscall and returns its outcome. Failures come back as
/// a response with status "failed" (recoverable by default: the program can
/// react); a host yield is surfaced as [`YieldedError`].
pub fn dispatch(c: &Call) -> anyhow::Result<HostResponse> {
    let raw = serde_json::to_vec(&SyscallEnvelope {
        abi: ABI_VERSION,
        name: &c.name,
        args: c.args.as_ref(),
    })
    .map_err(|e| anyhow::anyhow!("encode syscall: {}", e))?;
    let mem = Memory::from_bytes(&raw)?;
    let response_offset = unsafe { syscall(mem.offset()) };
    mem.free();
    let response_mem = Memory::find(response_offset)
        .ok_or_else(|| anyhow::anyhow!("decode host response: invalid offset"))?;
    // Copy the response bytes out, then free the host's response block. Memory
    // (unlike ManagedMemory) has no Drop, so without this the response block leaks
    // on every syscall — the mirror of the request `mem.free()` above.
    let response_bytes = response_mem.to_vec();
    response_mem.free();
    let decoded: ResponseEnvelope = serde_json::from_slice(&response_bytes)
        .map_err(|e| anyhow::anyhow!("decode host response: {}", e))?;

    let response = HostResponse {
        status: if decoded.status.is_empty() {
            STATUS_UNSPECIFIED.to_string()
        } else {
            decoded.status
        },
        code: decoded.code,
        result: decoded.result,
        message: decoded.message,
    };
    match response.status.as_str() {
        STATUS_RESULT | STATUS_FAILED => Ok(response),
        STATUS_YIELD => Err(YieldedError.into()),
        other => Err(anyhow::anyhow!("unsupported host outcome: {}", other)),
    }
}

/// A Savepoint brackets a critical zone in a sys.begin/sys.commit pair. Open one
/// with [`savepoint`] and close it with [`Savepoint::commit`] once the zone has
/// succeeded; *dropping it without committing* leaves the sys.begin open. A run
/// that then fails is first re-driven by replay — a transient failure simply
/// resumes and continues. A failure that re-drives without progress aborts the
/// zone: the journal forks at the begin and the whole zone re-executes live.
/// That drop-aborts behavior is the point — propagate an error out of the zone
/// (with `?`) and the savepoint unwinds the run for you. Brackets have stack
/// semantics; [`dispatch_hard`] wraps a single call this way.
#[must_use = "a Savepoint aborts the process unless it is committed"]
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
/// error that fails the process — a transient failure re-drives and re-executes
/// the call; a deterministic one aborts the section, and a retry forks right at
/// the begin, re-executing under a new revision. A plain [`dispatch`] (the
/// default, "soft") instead
/// records the failure for replay and lets the program react to it.
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
/// deserializes it into `T` — the typed front door a program uses instead of
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
/// The host validates the answer against the program's declared output schema
/// (its `interface.json` manifest); a rejected answer comes back as an error
/// the program can react to — correct the answer and publish again.
pub fn output<T: Serialize>(value: &T) -> anyhow::Result<()> {
    let args = serde_json::to_value(value).map_err(|e| anyhow::anyhow!("encode output: {}", e))?;
    let response = dispatch(&Call {
        name: SYS_OUTPUT.into(),
        args: Some(args),
    })?;
    if response.status == STATUS_FAILED {
        anyhow::bail!("publish output: {}", response.message);
    }
    Ok(())
}

/// abort abandons the open critical section instead of finishing with
/// [`output`]. The journal forks at the section's begin and the whole section
/// re-executes fresh; with no section open the process fails with `reason`.
/// The guest returns after calling abort.
pub fn abort(reason: &str) -> anyhow::Result<()> {
    let args = serde_json::json!({ "reason": reason });
    dispatch(&Call {
        name: SYS_ABORT.into(),
        args: Some(args),
    })?;
    Ok(())
}

/// log emits a human-readable progress line with the sys.log syscall. Logging is
/// best-effort observability: any failure (or a host yield) is swallowed so it
/// never perturbs the process.
pub fn log(message: &str) {
    let args = serde_json::json!({ "message": message });
    let _ = dispatch(&Call {
        name: SYS_LOG.into(),
        args: Some(args),
    });
}

/// Capability is one tool the host has granted this run — the guest's decoded
/// view of the host's capability descriptor. A program reads
/// `name`/`description`/`input_schema` to build its tool menu, and may consult
/// `hidden` (keep a dispatchable tool off that menu). A capability names an ADT
/// family whose operations are cases of `input_schema` (a `oneOf` discriminated
/// by an `operation`/`method` field), so a leaf grant is one tool, not several.
///
/// Data-flow provenance is deliberately absent. The host tracks which source
/// classes a run has observed and refuses the calls they bar, but it does not
/// tell the guest: the taxonomy is the map a compromised program would use to
/// find a route out the policy has not marked, so it stays host-side and a
/// refusal is all the guest ever sees. Decode-only: the host owns the record.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Capability {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
    /// Dispatchable, but excluded from the program's discoverable tool menu.
    #[serde(default)]
    pub hidden: bool,
}
