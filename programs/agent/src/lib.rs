use aurora_program_sdk as sdk;
use aurora_program_sdk::llm::{
    self, decode_model_envelopes, extract_json_region, looks_like_tool_batch, truncate_bytes,
    Message, ModelEnvelope,
};
use aurora_program_sdk::{Call, Capability};
use extism_pdk::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

const PROTOCOL_PROMPT: &str = "You are an Aurora agent running inside a Wasm guest.\n\
The host owns all side effects. Reply with exactly one compact JSON object containing an \"actions\" array.\n\
Use only the tools listed below. Match each tool's input JSON schema exactly.\n\
You may request multiple independent tool calls in one turn. The host executes them sequentially and returns one aggregated observation array.\n\
Each observation has status \"result\" with content or status \"failed\" with an error. A failed tool call is recoverable by default: use other sources, retry when appropriate, or explain the limitation.\n\
Add \"hard\": true to a tool call only when its failure must abort the process so a later resume re-executes it (for example, a state-changing step the process cannot meaningfully continue without). Omit \"hard\" for all normal, recoverable calls.\n\
After receiving observations, either request more tools or return exactly one terminal action:\n\
{\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"...\",\"reason\":\"...\"}}]} to finish, or\n\
{\"actions\":[{\"action\":\"abort\",\"content\":{\"reason\":\"...\"}}]} to give up: the process stops as failed with that reason, and nothing it already did is undone.\n\
Never combine a terminal action with tool calls in the same actions array.";

// Context-management bounds. The transcript's serialized byte length is a cheap
// proxy for tokens (~4 bytes/token); tune these to the model's real window.
//   MAX_STEPS         — hard cap on agentic turns; the last turn is forced final.
//   COMPACT_THRESHOLD — summarize the oldest messages once the transcript grows
//                       past this (keeping the system prompt and newest turns).
//   HARD_CEILING      — if it is still this large after compaction, force final.
//   KEEP_RECENT       — messages at the tail kept verbatim, never summarized.
//
// Both size bounds MUST stay below the core.openaiApi driver's max_request_bytes
// (default 1 MiB): the chat request it caps is essentially the transcript, so a
// summary or forced-final call at these sizes still has to fit under that cap.
// The margin (here ~256 KiB below 1 MiB) absorbs the request wrapper plus one
// turn's added observations before the next compaction runs.
const MAX_STEPS: u32 = 16;
const COMPACT_THRESHOLD: usize = 512 * 1024;
const HARD_CEILING: usize = 768 * 1024;
const KEEP_RECENT: usize = 6;

// Every successful tool result is saved in session scratch with core.scratch
// store, under an io/ key made from its content. A saved result goes into the
// transcript only as its key and its size, so no read can flood the window. The
// model reads it with core.scratch, or with the summarize action, which
// summarizes a saved value in a separate chat of at most HARD_CEILING. A result
// that is not saved has no key to read from, so it goes in whole up to
// INLINE_LIMIT, else as its size and its first EXCERPT_BYTES.
const INLINE_LIMIT: usize = 48 * 1024;
const EXCERPT_BYTES: usize = 2 * 1024;

const SUMMARY_PROMPT: &str = "You compress an AI agent's earlier working log. Preserve every fact, URL, identifier, number, finding, and decision needed to finish the task; drop repetition and chatter. Reply with prose only — no JSON, no tool calls.";

const SAVED_SUMMARY_PROMPT: &str = "You summarize a saved tool result for an AI agent, toward a focus. Keep every fact, URL, identifier, number, quote, name and finding that serves the focus. Drop navigation, ads, boilerplate and repetition. Reply with prose only, with no JSON and no tool calls.";

const FINAL_DIRECTIVE: &str = "You have reached this run's step or size limit. Reply now with EXACTLY one final action and no tool calls: {\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"...\",\"reason\":\"...\"}}]}. Base the answer on what you already have.";

// -- Data structures --

#[derive(Serialize)]
struct Output {
    status: &'static str,
    answer: String,
}

#[derive(Deserialize)]
struct Input {
    input: String,
    #[serde(default)]
    history: Vec<Message>,
    #[serde(default)]
    capabilities: Vec<Capability>,
    /// Which attempt this is (bumped by the host per retry) — lets the model
    /// know earlier attempts were rolled back and back off accordingly.
    #[serde(default)]
    attempt: u32,
}

#[derive(Deserialize)]
struct FinalAction {
    answer: String,
}

#[derive(Serialize)]
struct ToolObservation {
    action: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct FinishArgs {
    answer: String,
}

// -- Entry point --

#[plugin_fn]
pub fn run(_: ()) -> FnResult<Json<Output>> {
    match run_agent() {
        Ok(()) => Ok(Json(Output {
            status: "completed",
            answer: String::new(),
        })),
        Err(e) if sdk::yielded(&e) => Ok(Json(Output {
            status: "yielded",
            answer: String::new(),
        })),
        Err(e) => Err(e.into()),
    }
}

fn run_agent() -> anyhow::Result<()> {
    let inp: Input = sdk::input()?;
    if inp.input.is_empty() {
        anyhow::bail!("input is required");
    }

    // Saving needs core.scratch on the menu with store granted, because the
    // model reads saved results through it. Summarize reads them back with get.
    let scratch_ops = scratch_operations(&inp.capabilities);
    let saves = scratch_ops.iter().any(|op| op == "store");
    let summarizes = saves && scratch_ops.iter().any(|op| op == "get");

    let mut system_prompt = build_system_prompt(&inp.capabilities, saves, summarizes)?;
    if inp.attempt > 1 {
        system_prompt.push_str(&format!(
            "\nThis is attempt {} of this task; earlier attempts were rolled back.",
            inp.attempt
        ));
    }

    let mut messages: Vec<Message> = Vec::with_capacity(inp.history.len() + 2);
    messages.push(Message {
        role: "system".into(),
        content: system_prompt,
    });

    // Hidden capabilities stay dispatchable but off this program's menu, so the
    // model never sees them and never gets to request one.
    let mut allowed: HashSet<&str> = HashSet::with_capacity(inp.capabilities.len());
    for cap in inp.capabilities.iter().filter(|c| !c.hidden) {
        allowed.insert(cap.name.as_str());
    }
    for (i, msg) in inp.history.iter().enumerate() {
        if msg.role != "user" && msg.role != "assistant" {
            anyhow::bail!("history message {} has unsupported role {:?}", i, msg.role);
        }
        if msg.content.is_empty() {
            anyhow::bail!("history message {} has empty content", i);
        }
        messages.push(msg.clone());
    }

    // Capture the task before the move: it is the default focus of a
    // summarize action.
    let task = inp.input.clone();
    messages.push(Message {
        role: "user".into(),
        content: inp.input,
    });
    // The head compaction never touches: the system prompt, the session
    // history and this run's input.
    let head = messages.len();

    let mut step: u32 = 0;
    loop {
        step += 1;
        // Each agentic turn — one LLM call plus the tool calls it requests —
        // is a savepoint: sys.begin here, sys.commit at the turn's end. If the
        // turn breaks mid-way (a malformed model reply, an unavailable
        // capability), the savepoint is left open and a
        // resumed run forks right after it, re-executing the WHOLE turn live —
        // including the LLM call, giving the model a fresh chance — instead of
        // deterministically replaying the broken completion forever.
        let turn = sdk::savepoint()?;

        // Keep the transcript within the model's window: once it crosses
        // COMPACT_THRESHOLD, summarize the oldest of the run's own turns (never
        // the head or the newest turns) into one message. The summary is itself
        // a journaled LLM call, so a crash-resume rebuilds the same compacted
        // history by replay.
        maybe_compact(&mut messages, head)?;

        // Bound the run. At the step budget — or if the transcript is still near
        // the hard ceiling even after compaction — demand a final answer this
        // turn instead of looping (or blowing the context) further.
        if step >= MAX_STEPS || messages_bytes(&messages) >= HARD_CEILING {
            return finalize(turn, &mut messages);
        }

        let chat = chat_within_budget(&mut messages)?;
        let envelopes = match decode_model_envelopes(&chat) {
            Ok(envelopes) => envelopes,
            // The reply didn't parse as an action envelope. A prose answer, or a
            // truncated final carrying a recoverable "answer", is still the answer:
            // end the run with it via wrap_up rather than failing the process —
            // this turn's LLM call is journaled, so a resume replays the salvage.
            // But a botched/truncated TOOL batch (looks like a batch, no
            // recoverable answer) must NOT be salvaged: publishing its raw
            // {"actions":...} text would leak protocol as the answer. Leave the
            // savepoint open and fail so the turn re-drives and the model retries.
            Err(e) => {
                if looks_like_tool_batch(&chat) && recover_answer_field(&chat).is_none() {
                    return Err(e);
                }
                turn.commit()?;
                return wrap_up(&chat);
            }
        };

        let has_tool = envelopes
            .iter()
            .any(|e| e.action != "final" && e.action != "abort");
        let first_final_idx = envelopes.iter().position(|e| e.action == "final");
        let first_abort_idx = envelopes.iter().position(|e| e.action == "abort");

        if !has_tool {
            if let Some(idx) = first_abort_idx {
                // The model gave up. Commit the turn first so no section is left
                // open: with nothing to re-run, the host stops the process as
                // failed with the model's reason.
                turn.commit()?;
                return sdk::abort(&abort_reason(&envelopes[idx]));
            }
            if let Some(idx) = first_final_idx {
                turn.commit()?;
                return output_final(&envelopes[idx]);
            }
        }

        messages.push(Message {
            role: "assistant".into(),
            content: chat,
        });

        let mut observations: Vec<ToolObservation> = Vec::with_capacity(envelopes.len());
        for (i, envelope) in envelopes.iter().enumerate() {
            if envelope.action == "final" {
                continue;
            }
            if envelope.action == "summarize" && summarizes {
                emit_progress(&envelope.action);
                observations.push(summarize_saved(&envelope.content, &task)?);
                continue;
            }
            if !allowed.contains(envelope.action.as_str()) {
                anyhow::bail!(
                    "action {} requested unavailable capability {:?}",
                    i,
                    envelope.action
                );
            }
            if envelope.content.is_null() {
                anyhow::bail!("capability action {} missing content", i);
            }
            emit_progress(&envelope.action);
            let tool_call = Call {
                name: envelope.action.clone(),
                args: Some(envelope.content.clone()),
            };
            let mut response = if envelope.hard {
                sdk::dispatch_hard(&tool_call)?
            } else {
                sdk::dispatch(&tool_call)?
            };
            // A fetched web page comes back as raw HTML; strip it to readable
            // text before it is saved or shown. Guarded to GET responses whose
            // content-type is text/html, so JSON API responses are left
            // byte-for-byte intact.
            if response.status == sdk::STATUS_RESULT {
                strip_internet_html(&envelope.action, &envelope.content, &mut response.result);
            }
            let obs = if response.status == sdk::STATUS_FAILED {
                ToolObservation {
                    action: envelope.action.clone(),
                    status: sdk::STATUS_FAILED.into(),
                    args: Some(envelope.content.clone()),
                    content: None,
                    error: Some(response.message),
                }
            } else {
                observe_result(&envelope.action, &envelope.content, response.result, saves)?
            };
            observations.push(obs);
        }

        let raw_obs = serde_json::to_string(&observations)
            .map_err(|e| anyhow::anyhow!("encode tool observations: {}", e))?;
        // Feed observations back as a user message, matching the Go program. The
        // role "tool" is reserved by the OpenAI/DeepSeek API for native
        // function-call results and requires a tool_call_id referencing a prior
        // assistant tool_calls entry; this program uses a text (ReAct) protocol
        // with no native tool calls, so "tool" is rejected as malformed.
        messages.push(Message {
            role: "user".into(),
            content: raw_obs,
        });
        turn.commit()?;
    }
}

// chat_within_budget calls the LLM and, if the request is rejected for size
// (the core.openaiApi max_request_bytes cap, or a provider context-length
// error), sheds transcript bytes and retries until it fits or nothing more can
// be shed. It adapts to whatever request cap the host is configured with, below
// the proactive COMPACT_THRESHOLD/HARD_CEILING bounds. Each failed-then-retried
// call is journaled, so a crash-resume replays the same sequence. A host yield
// (e.g. an approval) is not a size error and propagates unchanged.
fn chat_within_budget(messages: &mut [Message]) -> anyhow::Result<String> {
    loop {
        match llm::chat(messages) {
            Ok(chat) => return Ok(chat),
            Err(e) => {
                if !is_size_error(&e.to_string()) || !shrink_messages(messages) {
                    return Err(e);
                }
            }
        }
    }
}

// is_size_error recognizes a request-too-large failure — the driver's byte cap
// or a provider's context-length rejection — versus a genuine host error.
fn is_size_error(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("exceed")
        || m.contains("too large")
        || m.contains("too long")
        || m.contains("too many tokens")
        || m.contains("context length")
        || m.contains("context_length")
        || m.contains("maximum context")
}

// shrink_messages halves the largest non-system message in place (head+tail
// around an elision marker) — the fastest way to bring an oversized request
// under the cap regardless of which observation blew it up. Returns false once
// every message is already small, so the retry loop terminates.
fn shrink_messages(messages: &mut [Message]) -> bool {
    let Some(i) = (1..messages.len()).max_by_key(|&i| messages[i].content.len()) else {
        return false;
    };
    let count = messages[i].content.chars().count();
    if count <= 2048 {
        return false;
    }
    let chars: Vec<char> = messages[i].content.chars().collect();
    let keep = count / 2;
    let head: String = chars[..keep / 2].iter().collect();
    let tail: String = chars[count - (keep - keep / 2)..].iter().collect();
    messages[i].content = format!("{head}\n…[truncated to fit the request limit]…\n{tail}");
    true
}

// messages_bytes is the serialized length of the transcript — the token proxy
// the compaction and ceiling thresholds compare against.
fn messages_bytes(messages: &[Message]) -> usize {
    serde_json::to_string(messages)
        .map(|s| s.len())
        .unwrap_or(0)
}

// maybe_compact summarizes the oldest of the run's own turns when the
// transcript grows past COMPACT_THRESHOLD. It keeps the first `head` messages
// verbatim — the system prompt, the session history and the run's input, which
// can carry instructions that must not be paraphrased — and the last
// KEEP_RECENT, and replaces the turns between them with one summary message.
// No-op below the threshold or when there is too little between them to gain
// anything.
fn maybe_compact(messages: &mut Vec<Message>, head: usize) -> anyhow::Result<()> {
    if messages_bytes(messages) < COMPACT_THRESHOLD || messages.len() <= head + KEEP_RECENT + 1 {
        return Ok(());
    }
    let split = messages.len() - KEEP_RECENT;
    let mut middle = String::new();
    for m in &messages[head..split] {
        middle.push_str(&m.role);
        middle.push_str(": ");
        middle.push_str(&m.content);
        middle.push_str("\n\n");
    }
    let summary = summarize(&middle)?;
    let mut rebuilt = Vec::with_capacity(head + 1 + KEEP_RECENT);
    rebuilt.extend_from_slice(&messages[..head]);
    rebuilt.push(Message {
        role: "user".into(),
        content: format!("[Earlier steps, compacted to save context]\n{}", summary),
    });
    rebuilt.extend_from_slice(&messages[split..]);
    *messages = rebuilt;
    Ok(())
}

// summarize asks the LLM to compress an excerpt of the transcript. It is a
// self-contained chat (its own system+user pair), not part of the agent's own
// message list, and returns the summary text.
fn summarize(excerpt: &str) -> anyhow::Result<String> {
    let msgs = [
        Message {
            role: "system".into(),
            content: SUMMARY_PROMPT.into(),
        },
        Message {
            role: "user".into(),
            content: format!("Compress these earlier steps:\n\n{}", excerpt),
        },
    ];
    llm::chat(&msgs)
}

// finalize forces the run to a close: it commits the current turn after asking
// the model for exactly one final action. If the model complies, that answer is
// published; if it still refuses (returns tools, or unparseable), its best text
// is salvaged so the run always terminates with an answer.
fn finalize(turn: sdk::Savepoint, messages: &mut Vec<Message>) -> anyhow::Result<()> {
    messages.push(Message {
        role: "user".into(),
        content: FINAL_DIRECTIVE.into(),
    });
    let chat = chat_within_budget(messages)?;
    turn.commit()?;
    wrap_up(&chat)
}

// wrap_up ends the run from a model reply that ought to carry a terminal
// action: use its final (or abort) when the reply parses, otherwise salvage a
// usable answer from it. Shared by the step/size-limit path (finalize) and the
// recovery when a mid-loop reply doesn't parse as the action envelope at all —
// so a model that simply answered in prose finishes the process instead of
// failing it.
fn wrap_up(chat: &str) -> anyhow::Result<()> {
    if let Ok(envelopes) = decode_model_envelopes(chat) {
        if let Some(idx) = envelopes.iter().position(|e| e.action == "final") {
            return output_final(&envelopes[idx]);
        }
        if let Some(idx) = envelopes.iter().position(|e| e.action == "abort") {
            return sdk::abort(&abort_reason(&envelopes[idx]));
        }
    }
    sdk::output(&FinishArgs {
        answer: salvage_answer(chat),
    })
}

// abort_reason reads the reason from an abort envelope.
fn abort_reason(envelope: &ModelEnvelope) -> String {
    envelope
        .content
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

// salvage_answer extracts a usable answer from a reply that would not parse as a
// final action — an "answer" field anywhere in its JSON, else the raw text.
fn salvage_answer(chat: &str) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(extract_json_region(chat)) {
        if let Some(answer) = find_answer(&value) {
            if !answer.is_empty() {
                return answer;
            }
        }
    }
    // The JSON did not parse as a whole — most often a long final answer
    // truncated by the model's token limit, leaving an unbalanced envelope.
    // Recover the answer field's text directly so the user sees the (partial)
    // answer prose, not the raw {"actions":...} protocol wrapper.
    if let Some(answer) = recover_answer_field(chat) {
        if !answer.trim().is_empty() {
            return answer;
        }
    }
    let trimmed = chat.trim();
    // A reply that still looks like a (botched/truncated) tool batch but yielded
    // no recoverable answer above must not be echoed verbatim — that would publish
    // raw {"actions":...} protocol text as the answer. Fall back to the note.
    if trimmed.is_empty() || looks_like_tool_batch(trimmed) {
        "Reached the step or size limit before completing the task.".into()
    } else {
        trimmed.to_string()
    }
}

// recover_answer_field pulls the text of an "answer" string field out of a reply
// whose JSON did not parse. It scans to the first "answer" key and decodes the
// JSON string that follows, stopping at the closing quote or at end-of-input — so
// a string truncated mid-value yields what was received rather than nothing.
fn recover_answer_field(chat: &str) -> Option<String> {
    let key = "\"answer\"";
    let after_key = &chat[chat.find(key)? + key.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let mut chars = after_colon.trim_start().strip_prefix('"')?.chars();
    let mut out = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out), // string closed cleanly
            '\\' => match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('b') => out.push('\u{8}'),
                Some('f') => out.push('\u{c}'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                        out.push(ch);
                    }
                }
                Some(other) => out.push(other), // unknown escape: keep the char
                None => break,                  // truncated mid-escape
            },
            _ => out.push(c),
        }
    }
    // Truncated before the closing quote — return the partial text.
    (!out.is_empty()).then_some(out)
}

// find_answer walks a JSON value for the first non-empty "answer" string.
fn find_answer(value: &Value) -> Option<String> {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(answer)) = map.get("answer") {
                return Some(answer.clone());
            }
            map.values().find_map(find_answer)
        }
        Value::Array(items) => items.iter().find_map(find_answer),
        _ => None,
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
        .and_then(|h| h.as_object())
        .map(|headers| {
            headers.iter().any(|(k, v)| {
                k.eq_ignore_ascii_case("content-type")
                    && v.as_str()
                        .map(|s| s.to_ascii_lowercase().contains("text/html"))
                        .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if !is_html {
        return;
    }
    if let Some(Value::String(body)) = obj.get_mut("body") {
        *body = html_to_text(body);
    }
}

// observe_result builds the observation of a successful tool call. A saved
// result is shown only as its key and its size: the model reads it through
// core.scratch, or summarizes it. Scratch's own results are not saved, because
// they are how the model reads: they come back whole. An unsaved result comes
// back whole up to INLINE_LIMIT, else as its size and head.
fn observe_result(
    action: &str,
    args: &Value,
    result: Option<Value>,
    saves: bool,
) -> anyhow::Result<ToolObservation> {
    let text = result.as_ref().map(Value::to_string).unwrap_or_default();
    let key = match &result {
        Some(value) if saves && action != "core.scratch" => save_result(value)?,
        _ => None,
    };
    let content = match key {
        Some(key) => Some(serde_json::json!({ "key": key, "bytes_length": text.len() })),
        None if text.len() <= INLINE_LIMIT => result,
        None => Some(serde_json::json!({
            "bytes_length": text.len(),
            "excerpt": head_excerpt(&text, EXCERPT_BYTES),
            "note": "The excerpt is the head of the result. The result is not saved.",
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
        return Ok(failed("summarize needs the key of a saved value".into()));
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
        _ => return Ok(failed(format!("no value is saved under {key}"))),
    };
    let text = match value {
        Value::String(text) => text,
        other => serde_json::to_string_pretty(&other)?,
    };
    let covered = truncate_bytes(&text, HARD_CEILING);
    let mut msgs = [
        Message {
            role: "system".into(),
            content: SAVED_SUMMARY_PROMPT.into(),
        },
        Message {
            role: "user".into(),
            content: format!(
                "Focus:\n{}\n\nSaved value {}:\n\n{}",
                truncate_bytes(focus, 512),
                key,
                covered
            ),
        },
    ];
    let summary = match chat_within_budget(&mut msgs) {
        Ok(summary) => summary,
        Err(e) if sdk::yielded(&e) => return Err(e),
        Err(e) => return Ok(failed(e.to_string())),
    };
    let mut out = serde_json::json!({ "key": key, "summary": summary });
    if covered.len() < text.len() {
        out["note"] = Value::String(format!(
            "The summary covers the first {} of {} bytes. Use core.scratch search on the key for the rest.",
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

// head_excerpt returns the first max bytes of s (on a char boundary) with a
// marker when it was cut, so the model can tell a head from a whole value.
fn head_excerpt(s: &str, max: usize) -> String {
    let head = truncate_bytes(s, max);
    if head.len() < s.len() {
        format!("{head}\n…[only the head is shown]")
    } else {
        head.to_string()
    }
}

// html_to_text extracts readable text from HTML with lol_html (Cloudflare's
// streaming rewriter): it keeps only ordinary `Data` text — so <script>/<style>/
// <title> content, which the parser tags as ScriptData/RawText, is dropped — and
// inserts a newline at block boundaries, then decodes common entities and
// collapses whitespace. On a rewriter error the raw input is returned unchanged.
fn html_to_text(html: &str) -> String {
    use lol_html::html_content::TextType;
    use lol_html::{element, rewrite_str, text, RewriteStrSettings};

    let out = Rc::new(RefCell::new(String::new()));
    let o_block = out.clone();
    let o_text = out.clone();

    let settings = RewriteStrSettings {
        element_content_handlers: vec![
            // Block-level elements: separate their text with a newline.
            element!(
                "p, div, br, li, tr, hr, section, article, header, footer, main, aside, nav, h1, h2, h3, h4, h5, h6, ul, ol, table, blockquote, pre, figure",
                move |_el| {
                    o_block.borrow_mut().push('\n');
                    Ok(())
                }
            ),
            // Keep only ordinary text; script/style/raw-text chunks are dropped.
            text!("*", move |t| {
                if t.text_type() == TextType::Data {
                    o_text.borrow_mut().push_str(t.as_str());
                }
                Ok(())
            }),
        ],
        ..RewriteStrSettings::default()
    };

    if rewrite_str(html, settings).is_err() {
        return html.to_string();
    }
    let raw = out.borrow().clone();
    collapse_ws(&decode_entities(&raw))
}

// decode_entities expands the handful of HTML entities lol_html leaves in text.
// "&amp;" is expanded last so an already-decoded "&lt;" is not re-decoded.
fn decode_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

// collapse_ws squeezes runs of spaces/tabs to one space and runs of newlines to
// at most a blank line, so extracted text reads as paragraphs without the
// original markup's whitespace.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    let mut pending_newlines = 0u32;
    for ch in s.chars() {
        if ch == '\n' || ch == '\r' {
            pending_newlines += 1;
            continue;
        }
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_newlines > 0 {
            if !out.is_empty() {
                out.push_str(if pending_newlines >= 2 { "\n\n" } else { "\n" });
            }
        } else if pending_space && !out.is_empty() {
            out.push(' ');
        }
        pending_space = false;
        pending_newlines = 0;
        out.push(ch);
    }
    out.trim().to_string()
}

fn build_system_prompt(
    capabilities: &[Capability],
    saves: bool,
    summarizes: bool,
) -> anyhow::Result<String> {
    let mut prompt = String::new();
    prompt.push_str(PROTOCOL_PROMPT);
    prompt.push_str("\n\nAvailable tools for this run:");
    // Hidden capabilities are dispatchable but kept off the discoverable menu.
    let visible: Vec<&Capability> = capabilities.iter().filter(|c| !c.hidden).collect();
    if visible.is_empty() {
        prompt.push_str("\nNone. Return a final action without attempting a tool call.");
        return Ok(prompt);
    }
    for (i, tool) in visible.iter().enumerate() {
        let name = tool.name.trim();
        if name.is_empty() {
            anyhow::bail!("capability {} name is required", i);
        }
        let schema = if tool.input_schema.is_null() {
            serde_json::json!({})
        } else {
            tool.input_schema.clone()
        };
        let compact_schema = serde_json::to_string(&schema).map_err(|e| {
            anyhow::anyhow!("capability {:?} has invalid input schema: {}", name, e)
        })?;
        prompt.push_str(&format!("\n\nTool {}\nName: {}", i + 1, name));
        let description = tool.description.trim();
        if !description.is_empty() {
            prompt.push_str(&format!("\nDescription: {}", description));
        }
        prompt.push_str(&format!("\nInput JSON schema: {}", compact_schema));
    }
    prompt.push_str("\n\nTool call shape:\n");
    prompt.push_str(
        r#"{"actions":[{"action":"<exact tool name>","content":<input matching that tool's schema>}]}"#,
    );
    if saves {
        prompt.push_str("\n\nEvery successful tool result, except core.scratch's own, is saved in this session's scratch memory. In place of the result, its observation gives {\"key\", \"bytes_length\"}. A saved result never changes, and it stays available in later messages of this session. Read a result with core.scratch: get returns it whole, and search finds lines in it.");
    }
    if summarizes {
        prompt.push_str(" To get an overview of a saved result without reading it here, use the summarize action. It summarizes the result in a separate conversation and returns only the summary:\n");
        prompt.push_str(
            r#"{"actions":[{"action":"summarize","content":{"key":"<key>","focus":"<what you need from it>"}}]}"#,
        );
    }
    Ok(prompt)
}

fn output_final(envelope: &ModelEnvelope) -> anyhow::Result<()> {
    if envelope.content.is_null() {
        anyhow::bail!("invalid final action: content is required");
    }
    let action: FinalAction = serde_json::from_value(envelope.content.clone())
        .map_err(|e| anyhow::anyhow!("invalid final action: {}", e))?;
    if action.answer.is_empty() {
        anyhow::bail!("final action missing answer");
    }
    sdk::output(&FinishArgs {
        answer: action.answer,
    })
}

fn emit_progress(action: &str) {
    sdk::log(&format!("⚙ {}", action));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Salvage returns the model's prose verbatim (trimmed) as the answer.
    #[test]
    fn salvage_returns_prose_verbatim() {
        let prose = "\n\nWarp is a modern terminal with AI features.\n";
        assert_eq!(
            salvage_answer(prose),
            "Warp is a modern terminal with AI features."
        );
    }

    // When the reply is JSON carrying an "answer" somewhere, salvage prefers it
    // over the raw text (a near-miss final that didn't decode as an envelope).
    #[test]
    fn salvage_prefers_an_embedded_answer_field() {
        let reply = r#"{"content":{"answer":"the moon is 384400 km away"}}"#;
        assert_eq!(salvage_answer(reply), "the moon is 384400 km away");
    }

    // An empty reply salvages to a stable placeholder, never an empty answer.
    #[test]
    fn salvage_of_empty_reply_is_nonempty() {
        assert!(!salvage_answer("   ").is_empty());
    }

    // A long final answer truncated by the token limit leaves an unbalanced
    // envelope that neither decodes nor whole-parses. Salvage must still surface
    // the answer prose, never the raw {"actions":...} wrapper.
    #[test]
    fn salvage_recovers_a_truncated_final_answer() {
        let truncated =
            r#"{"actions":[{"action":"final","content":{"answer":"**HWaaS** is Hardware as a Service, a model where"#;
        assert!(decode_model_envelopes(truncated).is_err());
        let got = salvage_answer(truncated);
        assert_eq!(got, "**HWaaS** is Hardware as a Service, a model where");
        assert!(!got.contains("\"actions\""), "raw envelope leaked: {got}");
    }

    // Recovery decodes JSON string escapes in the truncated answer.
    #[test]
    fn salvage_recovers_answer_with_escapes() {
        let truncated =
            r#"{"actions":[{"action":"final","content":{"answer":"line1\nline2 \"q\" tail"#;
        assert_eq!(salvage_answer(truncated), "line1\nline2 \"q\" tail");
    }

    // A botched/truncated TOOL batch carrying no recoverable "answer" must not be
    // salvaged verbatim — echoing its raw {"actions":...} text would leak protocol
    // as the answer. It is flagged as a tool batch and salvages to the note.
    #[test]
    fn salvage_does_not_leak_a_botched_tool_batch() {
        let botched =
            r#"{"actions":[{"action":"core.internet","content":{"method":"GET","url":"http://ex"#;
        assert!(decode_model_envelopes(botched).is_err());
        assert!(looks_like_tool_batch(botched));
        let got = salvage_answer(botched);
        assert!(!got.contains("\"action\""), "protocol leaked: {got}");
    }

    // -- large-read offload helpers --

    // The excerpt is verbatim when it fits and carries a cut marker when it
    // doesn't, so the model can tell a head from a whole small value.
    #[test]
    fn head_excerpt_marks_truncation() {
        assert_eq!(head_excerpt("short", 100), "short");
        let cut = head_excerpt("abcdefghij", 4);
        assert!(cut.starts_with("abcd"));
        assert!(cut.contains("only the head"));
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
