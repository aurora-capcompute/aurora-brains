//! The Aurora brain SDK: everything a guest brain needs to speak the syscall
//! boundary, so a brain crate contains only cognition. It owns the ABI v3
//! wire codec ([`wire`]), the single `extism:host/compute syscall` import,
//! and the dispatch protocol — result/failed observations, the yield
//! sentinel, and savepoint-bracketed "hard" calls.
//!
//! A brain is one cdylib crate under `brains/<name>/` that depends on this
//! SDK and exports its entrypoint with `#[plugin_fn]`.

use extism_pdk::Memory;
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
        wire::STATUS_RESULT => "result",
        wire::STATUS_YIELD => "yield",
        wire::STATUS_FAILED => "failed",
        _ => "unspecified",
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
        "result" | "failed" => Ok(response),
        "yield" => Err(YieldedError.into()),
        other => Err(anyhow::anyhow!("unsupported host outcome: {}", other)),
    }
}

/// dispatch_hard brackets a single call in a sys.begin/sys.commit savepoint.
/// On success it commits and returns the result. On failure it leaves the
/// begin open and returns an error that aborts the run, so a later resume
/// forks right after the begin and re-executes the call under a new revision.
/// A plain [`dispatch`] (the default, "soft") instead records the failure for
/// replay and lets the brain react to it.
pub fn dispatch_hard(c: &Call) -> anyhow::Result<HostResponse> {
    dispatch(&Call {
        name: SYS_BEGIN.into(),
        args: None,
    })?;
    let response = dispatch(c)?;
    if response.status == "failed" {
        anyhow::bail!("hard capability {:?} failed: {}", c.name, response.message);
    }
    dispatch(&Call {
        name: SYS_COMMIT.into(),
        args: None,
    })?;
    Ok(response)
}
