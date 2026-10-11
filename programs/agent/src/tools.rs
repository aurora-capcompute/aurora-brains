//! The model's tool calls: running them and turning their results into
//! observations.
//!
//! Every successful tool result is saved in session scratch with core.scratch
//! store, under an io/ key made from its content. A saved result goes into the
//! transcript only as its key and its size, so no read can flood the window. The
//! model reads it with core.scratch, or with the summarize action, which
//! summarizes a saved value in a separate chat of at most HARD_CEILING. A result
//! that is not saved has no key to read from, so it goes in whole up to
//! INLINE_LIMIT, else as its size and its first EXCERPT_BYTES.

use super::*;
use crate::html::html_to_text;
use std::collections::HashSet;

// The section calls the model may make directly. The runtime serves them for
// every program, so they are not on the tool menu.
const SECTION_CALLS: [&str; 2] = ["sys.begin", "sys.commit"];

/// What this run may call, and how its results are kept.
pub(crate) struct Tools {
    /// The tools on the menu. Hidden capabilities stay dispatchable but off it,
    /// so the model never learns of them.
    allowed: HashSet<String>,
    /// Whether results are saved in session scratch: core.scratch is on the
    /// menu with store, because the model reads saved results through it.
    pub(crate) saves: bool,
    /// Whether the summarize action exists: saving, plus get to read back.
    pub(crate) summarizes: bool,
    /// The run's input: the default focus of a summarize action.
    task: String,
}

impl Tools {
    pub(crate) fn new(capabilities: &[Capability], task: &str) -> Self {
        let scratch = scratch_operations(capabilities);
        let saves = scratch.iter().any(|op| op == "store");
        Tools {
            allowed: capabilities
                .iter()
                .filter(|c| !c.hidden)
                .map(|c| c.name.clone())
                .collect(),
            saves,
            summarizes: saves && scratch.iter().any(|op| op == "get"),
            task: task.to_string(),
        }
    }

    /// run runs one reply's tool calls in order and returns their observations.
    /// A call the agent cannot run is a failed observation, so the model can
    /// correct itself next turn.
    pub(crate) fn run(&self, envelopes: &[ModelEnvelope]) -> anyhow::Result<Vec<ToolObservation>> {
        let mut observations = Vec::with_capacity(envelopes.len());
        for envelope in envelopes {
            let action = envelope.action.as_str();
            let observation = match action {
                "final" => continue,
                "summarize" if self.summarizes => {
                    emit_progress(action);
                    summarize_saved(&envelope.content, &self.task)?
                }
                "sys.abort" => {
                    failed_observation(envelope, "sys.abort must be the only action in its reply")
                }
                _ if !self.allowed.contains(action) && !SECTION_CALLS.contains(&action) => {
                    failed_observation(
                        envelope,
                        format!(
                            "unknown tool {action:?}. Use only the tools in the system prompt."
                        ),
                    )
                }
                _ => {
                    emit_progress(action);
                    self.call(envelope)?
                }
            };
            observations.push(observation);
        }
        Ok(observations)
    }

    /// call dispatches one tool call and observes its result.
    fn call(&self, envelope: &ModelEnvelope) -> anyhow::Result<ToolObservation> {
        let action = envelope.action.as_str();
        let mut response = sdk::dispatch(&Call {
            name: envelope.action.clone(),
            args: (!envelope.content.is_null()).then(|| envelope.content.clone()),
        })?;
        if response.status == sdk::STATUS_FAILED {
            return Ok(failed_observation(envelope, response.message));
        }
        strip_internet_html(action, &envelope.content, &mut response.result);
        observe_result(action, &envelope.content, response.result, self.saves)
    }
}

#[derive(Serialize)]
pub(crate) struct ToolObservation {
    action: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// Every successful tool result is saved in session scratch with core.scratch
// store, under an io/ key made from its content. A saved result goes into the
// transcript only as its key and its size, so no read can flood the window. The
// model reads it with core.scratch, or with the summarize action, which
// summarizes a saved value in a separate chat of at most HARD_CEILING. A result
// that is not saved has no key to read from, so it goes in whole up to
// INLINE_LIMIT, else as its size and its first EXCERPT_BYTES.
const INLINE_LIMIT: usize = 48 * 1024;
const EXCERPT_BYTES: usize = 2 * 1024;

const SAVED_SUMMARY_PROMPT: &str = "You summarize a saved tool result for an AI agent. The agent gives a focus. Keep every fact, URL, identifier, number, quote, name and finding that serves the focus. Remove navigation, advertisements, boilerplate and repetition. Reply with prose only. Do not write JSON or tool calls.";

// failed_observation reports a model request that did not succeed.
fn failed_observation(envelope: &ModelEnvelope, error: impl Into<String>) -> ToolObservation {
    ToolObservation {
        action: envelope.action.clone(),
        status: sdk::STATUS_FAILED.into(),
        args: Some(envelope.content.clone()),
        content: None,
        error: Some(error.into()),
    }
}

// strip_internet_html rewrites a core.internet GET result's HTML body to plain
// text in place. Non-internet calls, non-GET methods, and non-HTML responses are
// left untouched, so JSON API payloads are never mangled.
fn strip_internet_html(action: &str, args: &Value, result: &mut Option<Value>) {
    if action != "core.internet" {
        return;
    }
    let method = args.get("method").and_then(|v| v.as_str()).unwrap_or("");
    if !method.eq_ignore_ascii_case("get") {
        return;
    }
    let Some(Value::Object(obj)) = result.as_mut() else {
        return;
    };
    let is_html = obj
        .get("headers")
        .and_then(Value::as_object)
        .is_some_and(|headers| {
            headers.iter().any(|(k, v)| {
                k.eq_ignore_ascii_case("content-type")
                    && v.as_str()
                        .is_some_and(|s| s.to_ascii_lowercase().contains("text/html"))
            })
        });
    if !is_html {
        return;
    }
    if let Some(Value::String(body)) = obj.get_mut("body") {
        *body = html_to_text(body);
    }
}

// observe_result builds the observation of a successful tool call. A saved
// result is shown only as its key and its size: the model reads it through
// core.scratch, or summarizes it. Two kinds of result are not saved and come
// back whole: scratch's own, because they are how the model reads, and the
// runtime's sys.* calls (a timer, a declassify), whose results are small and
// fixed. An unsaved result comes back whole up to INLINE_LIMIT, else as its
// size and head. The LLM's replies never pass through here: they are the
// conversation itself.
fn observe_result(
    action: &str,
    args: &Value,
    result: Option<Value>,
    saves: bool,
) -> anyhow::Result<ToolObservation> {
    let text = result.as_ref().map(Value::to_string).unwrap_or_default();
    let key = match &result {
        Some(value) if saves && action != "core.scratch" && !action.starts_with("sys.") => {
            save_result(value)?
        }
        _ => None,
    };
    let content = match key {
        Some(key) => Some(serde_json::json!({ "key": key, "bytes_length": text.len() })),
        None if text.len() <= INLINE_LIMIT => result,
        None => Some(serde_json::json!({
            "bytes_length": text.len(),
            "excerpt": head_excerpt(&text, EXCERPT_BYTES),
            "note": "The excerpt is the first part of the result. The host did not save the result.",
        })),
    };
    Ok(ToolObservation {
        action: action.to_string(),
        status: sdk::STATUS_RESULT.into(),
        args: Some(args.clone()),
        content,
        error: None,
    })
}

// save_result saves a tool result in session scratch and returns its key. A
// save that fails leaves the result unsaved, and its observation has no key.
fn save_result(value: &Value) -> anyhow::Result<Option<String>> {
    let response = sdk::dispatch(&Call {
        name: "core.scratch".into(),
        args: Some(serde_json::json!({ "operation": "store", "value": value })),
    })?;
    if response.status != sdk::STATUS_RESULT {
        return Ok(None);
    }
    Ok(response
        .result
        .as_ref()
        .and_then(|r| r.get("key"))
        .and_then(Value::as_str)
        .map(String::from))
}

// summarize_saved serves the summarize action. It reads a saved value with
// core.scratch get and summarizes it toward the focus in a separate chat, so the
// value never enters this conversation and only the summary does. Of a value
// larger than HARD_CEILING, the head is summarized, and the observation says how
// much.
fn summarize_saved(content: &Value, task: &str) -> anyhow::Result<ToolObservation> {
    let observation =
        |status: &str, content_out: Option<Value>, error: Option<String>| ToolObservation {
            action: "summarize".into(),
            status: status.into(),
            args: Some(content.clone()),
            content: content_out,
            error,
        };
    let failed = |error: String| observation(sdk::STATUS_FAILED, None, Some(error));

    let Some(key) = content.get("key").and_then(Value::as_str) else {
        return Ok(failed("summarize needs the key of a saved result".into()));
    };
    let focus = content
        .get("focus")
        .and_then(Value::as_str)
        .filter(|f| !f.trim().is_empty())
        .unwrap_or(task);
    let response = sdk::dispatch(&Call {
        name: "core.scratch".into(),
        args: Some(serde_json::json!({ "operation": "get", "key": key })),
    })?;
    if response.status != sdk::STATUS_RESULT {
        return Ok(failed(response.message));
    }
    let value = match response.result {
        Some(Value::Object(mut found)) if found.get("found") == Some(&Value::Bool(true)) => {
            found.remove("value").unwrap_or(Value::Null)
        }
        _ => {
            return Ok(failed(format!(
                "the host has no saved result with the key {key}"
            )))
        }
    };
    let text = match value {
        Value::String(text) => text,
        other => serde_json::to_string_pretty(&other)?,
    };
    let covered = &text[..text.floor_char_boundary(HARD_CEILING)];
    let mut msgs = [
        Message {
            role: "system".into(),
            content: SAVED_SUMMARY_PROMPT.into(),
        },
        Message {
            role: "user".into(),
            content: format!(
                "Focus:\n{}\n\nSaved result {}:\n\n{}",
                &focus[..focus.floor_char_boundary(512)],
                key,
                covered
            ),
        },
    ];
    // An LLM failure fails the process, as in the main chat: its section stays
    // open, so a resume calls the LLM again.
    let summary = chat_within_budget(&mut msgs)?;
    let mut out = serde_json::json!({ "key": key, "summary": summary });
    if covered.len() < text.len() {
        out["note"] = Value::String(format!(
            "The summary covers only the first {} of {} bytes. To find something in the rest, use core.scratch search with the key.",
            covered.len(),
            text.len()
        ));
    }
    Ok(observation(sdk::STATUS_RESULT, Some(out), None))
}

// scratch_operations lists the core.scratch operations on this run's menu, read
// from the entry's input schema: one branch per operation, each fixing
// "operation" with a const.
fn scratch_operations(capabilities: &[Capability]) -> Vec<String> {
    let Some(scratch) = capabilities
        .iter()
        .find(|c| c.name == "core.scratch" && !c.hidden)
    else {
        return Vec::new();
    };
    let schema = &scratch.input_schema;
    let branches: Vec<&Value> = match schema.get("oneOf").and_then(Value::as_array) {
        Some(branches) => branches.iter().collect(),
        None => vec![schema],
    };
    branches
        .iter()
        .filter_map(|b| {
            b.pointer("/properties/operation/const")
                .and_then(Value::as_str)
        })
        .map(String::from)
        .collect()
}

// head_excerpt returns the first limit bytes of s (on a char boundary) with a
// marker when it was cut, so the model can tell a head from a whole value.
fn head_excerpt(s: &str, limit: usize) -> String {
    let head = &s[..s.floor_char_boundary(limit)];
    if head.len() < s.len() {
        format!("{head}\n…[This is only the first part of the result.]")
    } else {
        head.to_string()
    }
}

fn emit_progress(action: &str) {
    sdk::log(&format!("⚙ {}", action));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // The excerpt is verbatim when it fits and carries a cut marker when it
    // doesn't, so the model can tell a head from a whole small value.
    #[test]
    fn head_excerpt_marks_truncation() {
        assert_eq!(head_excerpt("short", 100), "short");
        let cut = head_excerpt("abcdefghij", 4);
        assert!(cut.starts_with("abcd"));
        assert!(cut.contains("only the first part"));
    }

    // The granted scratch operations are read from the menu schema: a oneOf of
    // branches, or one branch alone. A hidden grant is not on the menu.
    #[test]
    fn scratch_operations_reads_the_menu_schema() {
        let branch = |op: &str| json!({"properties": {"operation": {"const": op}}});
        let scratch = |schema: Value, hidden: bool| Capability {
            name: "core.scratch".into(),
            input_schema: schema,
            hidden,
            ..Capability::default()
        };
        let both = scratch(json!({"oneOf": [branch("get"), branch("store")]}), false);
        assert_eq!(scratch_operations(&[both]), vec!["get", "store"]);
        assert_eq!(
            scratch_operations(&[scratch(branch("search"), false)]),
            vec!["search"]
        );
        assert!(scratch_operations(&[scratch(branch("store"), true)]).is_empty());
    }
}
