use aurora_program_sdk as sdk;
use aurora_program_sdk::{Call, Capability};
use extism_pdk::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

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

    loop {
        // Each agentic turn — one LLM call plus the tool calls it requests —
        // is a savepoint: sys.begin here, sys.commit at the turn's end. If the
        // turn breaks mid-way (a malformed model reply, an unavailable
        // capability, an aborted delegation), the savepoint is left open and a
        // resumed run forks right after it, re-executing the WHOLE turn live —
        // including the LLM call, giving the model a fresh chance — instead of
        // deterministically replaying the broken completion forever.
        let turn = sdk::savepoint()?;
        let chat = llm_chat(&messages)?;
        let envelopes = decode_model_envelopes(&chat)
            .map_err(|e| anyhow::anyhow!("invalid model JSON: {}", e))?;

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
            let response = if envelope.hard {
                sdk::dispatch_hard(&tool_call)?
            } else {
                sdk::dispatch(&tool_call)?
            };
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
    let req = LlmRequest { messages };
    let args = serde_json::to_value(&req)?;
    let response = sdk::dispatch(&Call {
        name: "openai.chat".into(),
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
