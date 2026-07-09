use aurora_program_sdk as sdk;
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
To make a completed side effect undoable, register its exact inverse right after observing its result: {\"action\":\"compensate\",\"content\":{\"name\":\"<tool>\",\"args\":{...}}}. The host only records it; registered inverses run, newest first, if you later abort.\n\
After receiving observations, either request more tools or return exactly one terminal action:\n\
{\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"...\",\"reason\":\"...\"}}]} to finish, or\n\
{\"actions\":[{\"action\":\"abort\",\"content\":{\"reason\":\"...\",\"retry_seconds\":60}}]} to undo the registered effects and retry the task after the delay (omit retry_seconds to undo and stop).\n\
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

const SUMMARY_PROMPT: &str = "You compress an AI agent's earlier working log. Preserve every fact, URL, identifier, number, finding, and decision needed to finish the task; drop repetition and chatter. Reply with prose only — no JSON, no tool calls.";

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

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct LlmRequest<'a> {
    // The ADT discriminator selecting the chat operation of the core.openaiApi
    // family; the host routes on it and strips it from the provider request.
    operation: &'static str,
    messages: &'a [Message],
}

#[derive(Deserialize)]
struct LlmResponse {
    choices: Vec<LlmChoice>,
}

#[derive(Deserialize)]
struct LlmChoice {
    message: LlmChoiceMessage,
}

#[derive(Deserialize)]
struct LlmChoiceMessage {
    content: String,
}

#[derive(Debug)]
struct ModelEnvelope {
    action: String,
    content: Value,
    // `hard` marks a call whose failure must abort the process (with its savepoint
    // left open) so a later resume re-executes it, instead of being reported back
    // as a recoverable observation. Default is the soft path.
    hard: bool,
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

    let mut system_prompt = build_system_prompt(&inp.capabilities)?;
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

    messages.push(Message {
        role: "user".into(),
        content: inp.input,
    });

    let mut step: u32 = 0;
    loop {
        step += 1;
        // Each agentic turn — one LLM call plus the tool calls it requests —
        // is a savepoint: sys.begin here, sys.commit at the turn's end. If the
        // turn breaks mid-way (a malformed model reply, an unavailable
        // capability, an aborted delegation), the savepoint is left open and a
        // resumed run forks right after it, re-executing the WHOLE turn live —
        // including the LLM call, giving the model a fresh chance — instead of
        // deterministically replaying the broken completion forever.
        let turn = sdk::savepoint()?;

        // Keep the transcript within the model's window: once it crosses
        // COMPACT_THRESHOLD, summarize the oldest messages (never the system
        // prompt or the newest turns) into one message. The summary is itself a
        // journaled LLM call, so a crash-resume rebuilds the same compacted
        // history by replay.
        maybe_compact(&mut messages)?;

        // Bound the run. At the step budget — or if the transcript is still near
        // the hard ceiling even after compaction — demand a final answer this
        // turn instead of looping (or blowing the context) further.
        if step >= MAX_STEPS || messages_bytes(&messages) >= HARD_CEILING {
            return finalize(turn, &mut messages);
        }

        let chat = chat_within_budget(&mut messages)?;
        let envelopes = match decode_model_envelopes(&chat) {
            Ok(envelopes) => envelopes,
            // The reply didn't parse as an action envelope — most often the
            // model answered directly in prose, judging no tool was needed
            // (occasionally a truncated batch). That reply is still the answer:
            // end the run with it rather than failing the process. This turn's
            // LLM call is already journaled, so a resume replays this salvage.
            Err(_) => {
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
                // The model gave up on the task and asked to roll it back: the
                // host executes the compensations this run registered, newest
                // first, then retries after the given delay or stops. Commit
                // the turn first so no section is left open — a model abort
                // rolls back (and retries) the whole task, not just this turn.
                let reason = envelopes[idx]
                    .content
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let retry = envelopes[idx]
                    .content
                    .get("retry_seconds")
                    .and_then(|v| v.as_u64());
                turn.commit()?;
                return sdk::abort(&reason, retry);
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
            // "compensate" is protocol, not a menu tool: it registers a deferred
            // undo with the host (validated there against the grant set).
            let is_compensate = envelope.action == "compensate";
            if !is_compensate && !allowed.contains(envelope.action.as_str()) {
                anyhow::bail!(
                    "action {} requested unavailable capability {:?}",
                    i,
                    envelope.action
                );
            }
            if envelope.content.is_null() {
                anyhow::bail!("capability action {} missing content", i);
            }
            emit_progress(&envelope.action, &envelope.content);
            let tool_call = Call {
                name: if is_compensate {
                    sdk::SYS_COMPENSATE.into()
                } else {
                    envelope.action.clone()
                },
                args: Some(envelope.content.clone()),
            };
            let mut response = if envelope.hard {
                sdk::dispatch_hard(&tool_call)?
            } else {
                sdk::dispatch(&tool_call)?
            };
            // A fetched web page comes back as raw HTML; strip it to readable
            // text so a single page can't flood the model's context. Guarded to
            // GET responses whose content-type is text/html, so JSON API
            // responses are left byte-for-byte intact.
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
                ToolObservation {
                    action: envelope.action.clone(),
                    status: response.status.clone(),
                    args: Some(envelope.content.clone()),
                    content: response.result,
                    error: None,
                }
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
        match llm_chat(messages) {
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

// maybe_compact summarizes the oldest messages when the transcript grows past
// COMPACT_THRESHOLD, keeping messages[0] (the system prompt: protocol + tool
// menu) and the last KEEP_RECENT turns verbatim, and replacing the middle with a
// single summary message. No-op below the threshold or when there is too little
// middle to gain anything.
fn maybe_compact(messages: &mut Vec<Message>) -> anyhow::Result<()> {
    if messages_bytes(messages) < COMPACT_THRESHOLD || messages.len() <= KEEP_RECENT + 2 {
        return Ok(());
    }
    let split = messages.len() - KEEP_RECENT;
    let mut middle = String::new();
    for m in &messages[1..split] {
        middle.push_str(&m.role);
        middle.push_str(": ");
        middle.push_str(&m.content);
        middle.push_str("\n\n");
    }
    let summary = summarize(&middle)?;
    let mut rebuilt = Vec::with_capacity(2 + KEEP_RECENT);
    rebuilt.push(messages[0].clone());
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
    llm_chat(&msgs)
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
            let (reason, retry) = abort_fields(&envelopes[idx]);
            return sdk::abort(&reason, retry);
        }
    }
    sdk::output(&FinishArgs {
        answer: salvage_answer(chat),
    })
}

// abort_fields reads the reason and optional retry delay from an abort envelope.
fn abort_fields(envelope: &ModelEnvelope) -> (String, Option<u64>) {
    let reason = envelope
        .content
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let retry = envelope
        .content
        .get("retry_seconds")
        .and_then(|v| v.as_u64());
    (reason, retry)
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
    let trimmed = chat.trim();
    if trimmed.is_empty() {
        "Reached the step or size limit before completing the task.".into()
    } else {
        trimmed.to_string()
    }
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

fn build_system_prompt(capabilities: &[Capability]) -> anyhow::Result<String> {
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
    Ok(prompt)
}

fn decode_model_envelopes(content: &str) -> anyhow::Result<Vec<ModelEnvelope>> {
    decode_model_envelope_stream(content, 0)
}

fn decode_model_envelope_stream(content: &str, depth: u32) -> anyhow::Result<Vec<ModelEnvelope>> {
    if depth > 1 {
        anyhow::bail!("nested encoded model JSON is not supported");
    }
    // Reasoning models (e.g. deepseek-reasoner) often wrap the JSON batch in
    // prose or a markdown code fence — "Let me look it up.\n\n{...}". Narrow to
    // the JSON region before parsing, and tolerate trailing commentary once a
    // valid batch has been decoded, rather than requiring the whole reply to be
    // bare JSON.
    let json_part = extract_json_region(content);
    let mut envelopes = Vec::new();
    let stream = serde_json::Deserializer::from_str(json_part).into_iter::<Value>();
    for result in stream {
        let value = match result {
            Ok(value) => value,
            // A parse error after we already have envelopes means trailing prose
            // after the batch; stop. Before any envelope it's a genuine failure.
            Err(_) if !envelopes.is_empty() => break,
            Err(e) => return Err(e.into()),
        };
        match &value {
            Value::Array(arr) => {
                for item in arr {
                    let decoded = decode_model_envelope_object(item.clone())?;
                    envelopes.extend(decoded);
                }
            }
            Value::Object(_) => {
                let decoded = decode_model_envelope_object(value)?;
                envelopes.extend(decoded);
            }
            Value::String(s) => {
                let nested = decode_model_envelope_stream(s, depth + 1)?;
                envelopes.extend(nested);
            }
            _ if !envelopes.is_empty() => break,
            _ => anyhow::bail!("expected action object or array"),
        }
    }
    if envelopes.is_empty() {
        anyhow::bail!("model action batch is empty");
    }
    Ok(envelopes)
}

// extract_json_region trims a model reply down to the JSON batch, dropping a
// natural-language preamble and/or a surrounding markdown code fence. It returns
// the slice starting at the first JSON value; trailing text after the value is
// handled by the caller's parse loop.
fn extract_json_region(content: &str) -> &str {
    let mut s = content.trim();
    if let Some(rest) = s.strip_prefix("```") {
        // Drop an optional language tag on the fence's first line (```json).
        let body = rest.find('\n').map(|i| &rest[i + 1..]).unwrap_or(rest);
        // Cut at the closing fence when present.
        s = body.find("```").map(|i| &body[..i]).unwrap_or(body).trim();
    }
    match s.find(['{', '[']) {
        Some(i) => &s[i..],
        None => s,
    }
}

fn decode_model_envelope_object(value: Value) -> anyhow::Result<Vec<ModelEnvelope>> {
    // Skip diagnostic objects (non-empty "error" field).
    if let Some(err_str) = value.get("error").and_then(|v| v.as_str()) {
        if !err_str.is_empty() {
            return Ok(vec![]);
        }
    }

    // Unwrap a batch wrapper that has an "actions" array.
    if let Some(actions_val) = value.get("actions") {
        let items: Vec<Value> = serde_json::from_value(actions_val.clone())
            .map_err(|e| anyhow::anyhow!("actions must be an array: {}", e))?;
        if items.is_empty() {
            anyhow::bail!("model action batch is empty");
        }
        let mut envelopes = Vec::new();
        for item in items {
            let decoded = decode_model_envelope_object(item)?;
            envelopes.extend(decoded);
        }
        return Ok(envelopes);
    }

    // Single envelope.
    let action = value
        .get("action")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "action is required in model object: {}",
                abbreviated_json(&value, 300)
            )
        })?;
    let content = value.get("content").cloned().unwrap_or(Value::Null);
    let hard = value.get("hard").and_then(|v| v.as_bool()).unwrap_or(false);
    Ok(vec![ModelEnvelope {
        action: action.to_string(),
        content,
        hard,
    }])
}

fn abbreviated_json(value: &Value, limit: usize) -> String {
    let s = serde_json::to_string(value).unwrap_or_default();
    if s.len() <= limit {
        s
    } else {
        format!("{}[...]", &s[..limit])
    }
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

fn llm_chat(messages: &[Message]) -> anyhow::Result<String> {
    let req = LlmRequest {
        operation: "chat",
        messages,
    };
    let args = serde_json::to_value(&req)?;
    let response = sdk::dispatch(&Call {
        name: "core.openaiApi".into(),
        args: Some(args),
    })?;
    if response.status != sdk::STATUS_RESULT {
        anyhow::bail!("host failure: {}", response.message);
    }
    let result = response
        .result
        .ok_or_else(|| anyhow::anyhow!("LLM returned empty result"))?;
    let chat: LlmResponse = serde_json::from_value(result)
        .map_err(|e| anyhow::anyhow!("decode llm response: {}", e))?;
    if chat.choices.is_empty() {
        anyhow::bail!("provider returned no choices");
    }
    Ok(chat.choices[0].message.content.clone())
}

fn emit_progress(action: &str, content: &Value) {
    sdk::log(&progress_summary(action, content));
}

fn progress_summary(action: &str, content: &Value) -> String {
    let fields = match content.as_object() {
        Some(f) => f,
        None => return format!("⚙ {}", action),
    };
    if action.starts_with("call.") {
        if let Some(s) = fields.get("message").and_then(|v| v.as_str()) {
            if !s.is_empty() {
                let truncated = if s.len() > 80 {
                    format!("{}…", &s[..80])
                } else {
                    s.to_string()
                };
                return format!("🔀 {}: {}", action, truncated);
            }
        }
        return format!("🔀 {}", action);
    }
    if action.starts_with("k8s.") || action.starts_with("helm.") {
        let mut parts: Vec<&str> = Vec::new();
        for key in &[
            "kind",
            "namespace",
            "name",
            "release",
            "chart",
            "api_version",
        ] {
            if let Some(s) = fields.get(*key).and_then(|v| v.as_str()) {
                if !s.is_empty() {
                    parts.push(s);
                }
            }
        }
        if !parts.is_empty() {
            return format!("⚙ {} {}", action, parts.join("/"));
        }
    }
    format!("⚙ {}", action)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A model that answers directly in prose — no action envelope — must not
    // parse as an envelope; wrap_up then salvages it (see below) rather than
    // failing the run, which is the bug this guards against.
    #[test]
    fn prose_reply_is_not_a_valid_envelope() {
        let prose = "\n\nWarp has real trade-offs. **Strengths:** fast, modern. \
                     **Weaknesses:** closed-source. Verdict: worth trying.";
        assert!(decode_model_envelopes(prose).is_err());
    }

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

    // A proper final envelope still decodes normally — salvage is a fallback,
    // not the primary path.
    #[test]
    fn well_formed_final_still_decodes() {
        let reply = r#"{"actions":[{"action":"final","content":{"answer":"hi"}}]}"#;
        let envelopes = decode_model_envelopes(reply).unwrap();
        assert_eq!(envelopes.len(), 1);
        assert_eq!(envelopes[0].action, "final");
    }
}
