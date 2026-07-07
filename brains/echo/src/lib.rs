//! echo — the smallest possible Aurora brain. It reads its input through the
//! SDK, needs no LLM and no capabilities, and returns a deterministic answer:
//! the message it was given, or "pong" when none was. It exists to exercise the
//! multi-program path and to show how little a brain is once the SDK owns the
//! protocol — cognition here is a single `if`, and the plumbing is just
//! [`sdk::input`] → [`sdk::output`].

use aurora_brain_sdk as sdk;
use extism_pdk::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    message: String,
}

/// The run's result payload, published via sys.output — the same `{"answer":
/// ...}` shape the agent brain reports.
#[derive(Serialize)]
struct Answer {
    answer: String,
}

/// The entrypoint's return envelope: the process's terminal status, all the host
/// reads from a guest's return. The answer itself travels through sys.output.
#[derive(Serialize)]
struct Output {
    status: &'static str,
}

#[plugin_fn]
pub fn run(_: ()) -> FnResult<Json<Output>> {
    match echo() {
        Ok(()) => Ok(Json(Output {
            status: "completed",
        })),
        Err(e) if sdk::yielded(&e) => Ok(Json(Output { status: "yielded" })),
        Err(e) => Err(e.into()),
    }
}

fn echo() -> anyhow::Result<()> {
    let input: Input = sdk::input()?;
    let answer = if input.message.is_empty() {
        "pong".to_string()
    } else {
        input.message
    };
    sdk::output(&Answer { answer })
}

/// The program's bundled interface: what to pass and what comes back.
#[plugin_fn]
pub fn describe(_: ()) -> FnResult<Json<sdk::Interface>> {
    Ok(Json(sdk::Interface {
        description: "Echoes the message back; answers \"pong\" when the message is empty. \
                      Needs no LLM and no capabilities."
            .into(),
        input: serde_json::json!({"type": "string", "description": "The text to echo."}),
        output: serde_json::json!({"type": "string", "description": "The message verbatim, or \"pong\"."}),
    }))
}
