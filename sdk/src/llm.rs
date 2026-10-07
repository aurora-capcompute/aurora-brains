//! The LLM plumbing every conversational program shares: one chat call to the
//! core.openaiApi capability, and the decoding of a model reply into the
//! actions it asks for — `{"actions":[{"action":...,"content":...,"hard":...}]}`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{dispatch, Call, STATUS_RESULT};

/// The LLM capability a program chats through.
pub const CAPABILITY: &str = "core.openaiApi";

/// MAX_COMPLETION_TOKENS caps each model reply so a long final answer is not
/// truncated by a smaller provider default (a truncated reply loses its closing
/// JSON and only salvages as partial text). Tool-call turns are short JSON and
/// stop well under it; only a verbose final answer approaches it.
pub const MAX_COMPLETION_TOKENS: u32 = 4096;

/// One chat message.
#[derive(Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    // The ADT discriminator selecting the chat operation of the core.openaiApi
    // family; the host routes on it and strips it from the provider request.
    operation: &'static str,
    messages: &'a [Message],
    // Forwarded verbatim to the provider by the driver.
    max_completion_tokens: u32,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

/// chat sends the conversation to the LLM and returns its reply text.
pub fn chat(messages: &[Message]) -> anyhow::Result<String> {
    let args = serde_json::to_value(ChatRequest {
        operation: "chat",
        messages,
        max_completion_tokens: MAX_COMPLETION_TOKENS,
    })?;
    let response = dispatch(&Call {
        name: CAPABILITY.into(),
        args: Some(args),
    })?;
    if response.status != STATUS_RESULT {
        anyhow::bail!("host failure: {}", response.message);
    }
    let result = response
        .result
        .ok_or_else(|| anyhow::anyhow!("LLM returned empty result"))?;
    let chat: ChatResponse = serde_json::from_value(result)
        .map_err(|e| anyhow::anyhow!("decode llm response: {}", e))?;
    let choice = chat
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("provider returned no choices"))?;
    Ok(choice.message.content)
}

/// One action a model reply asks for.
#[derive(Debug)]
pub struct ModelEnvelope {
    pub action: String,
    pub content: Value,
    /// `hard` marks a call whose failure must abort the process (with its
    /// savepoint left open) so a later resume re-executes it, instead of being
    /// reported back as a recoverable observation. Default is the soft path.
    pub hard: bool,
}

/// decode_model_envelopes reads the actions out of a model reply.
pub fn decode_model_envelopes(content: &str) -> anyhow::Result<Vec<ModelEnvelope>> {
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
                    envelopes.extend(decode_model_envelope_object(item.clone())?);
                }
            }
            Value::Object(_) => envelopes.extend(decode_model_envelope_object(value)?),
            Value::String(s) => envelopes.extend(decode_model_envelope_stream(s, depth + 1)?),
            _ if !envelopes.is_empty() => break,
            _ => anyhow::bail!("expected action object or array"),
        }
    }
    if envelopes.is_empty() {
        anyhow::bail!("model action batch is empty");
    }
    Ok(envelopes)
}

/// extract_json_region trims a model reply down to the JSON batch, dropping a
/// natural-language preamble and/or a surrounding markdown code fence. It
/// returns the slice starting at the first JSON value; trailing text after the
/// value is the caller's to handle.
pub fn extract_json_region(content: &str) -> &str {
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
            envelopes.extend(decode_model_envelope_object(item)?);
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
        // Clamp on a char boundary: a fixed byte slice would panic on multi-byte
        // UTF-8 in a model-authored value, and this runs while building an error.
        format!("{}[...]", truncate_bytes(&s, limit))
    }
}

/// looks_like_tool_batch reports whether a reply's JSON region still carries an
/// "action"/"actions" key — the shape of a tool-call batch rather than a prose
/// reply. A botched or truncated batch is kept out of a program's salvage path
/// so its raw {"actions":...} protocol text is never published as the answer.
pub fn looks_like_tool_batch(reply: &str) -> bool {
    let region = extract_json_region(reply);
    region.contains("\"action\"") || region.contains("\"actions\"")
}

/// truncate_bytes clamps s to at most max bytes, backing up to the nearest
/// char boundary so the result stays valid UTF-8.
pub fn truncate_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // A model that answers directly in prose — no action envelope — does not
    // parse as an envelope; a program salvages it rather than failing the run.
    #[test]
    fn prose_reply_is_not_a_valid_envelope() {
        let prose = "\n\nWarp has real trade-offs. **Strengths:** fast, modern. \
                     **Weaknesses:** closed-source. Verdict: worth trying.";
        assert!(decode_model_envelopes(prose).is_err());
    }

    #[test]
    fn decode_accepts_an_actions_batch_with_hard_flag() {
        let reply = r#"{"actions":[{"action":"db.write","content":{"k":"v"},"hard":true},{"action":"final","content":{"answer":"$1"}}]}"#;
        let envelopes = decode_model_envelopes(reply).unwrap();
        assert_eq!(envelopes.len(), 2);
        assert_eq!(envelopes[0].action, "db.write");
        assert!(envelopes[0].hard);
        assert_eq!(envelopes[1].action, "final");
        assert!(!envelopes[1].hard);
    }

    #[test]
    fn decode_unwraps_a_fenced_reply() {
        let reply = "Sure.\n```json\n{\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"done\"}}]}\n```";
        let envelopes = decode_model_envelopes(reply).unwrap();
        assert_eq!(envelopes.len(), 1);
        assert_eq!(envelopes[0].action, "final");
        assert_eq!(envelopes[0].content, json!({"answer": "done"}));
    }

    // A truncated or botched batch does not decode, and is still recognized as
    // a batch, so a program keeps its raw protocol text out of the answer.
    #[test]
    fn looks_like_tool_batch_flags_a_batch_not_prose() {
        let botched =
            r#"{"actions":[{"action":"core.internet","content":{"method":"GET","url":"http://ex"#;
        assert!(decode_model_envelopes(botched).is_err());
        assert!(looks_like_tool_batch(botched));
        assert!(!looks_like_tool_batch("Here is the answer in prose."));
    }

    // The chat request carries a completion-token cap so a long final answer is
    // not truncated by the provider default.
    #[test]
    fn chat_request_caps_completion_tokens() {
        let messages = vec![Message {
            role: "user".into(),
            content: "hi".into(),
        }];
        let value = serde_json::to_value(ChatRequest {
            operation: "chat",
            messages: &messages,
            max_completion_tokens: MAX_COMPLETION_TOKENS,
        })
        .unwrap();
        assert_eq!(value["operation"], "chat");
        assert_eq!(
            value["max_completion_tokens"].as_u64(),
            Some(MAX_COMPLETION_TOKENS as u64)
        );
    }

    #[test]
    fn truncate_bytes_respects_char_boundaries() {
        assert_eq!(truncate_bytes("hello", 100), "hello");
        assert_eq!(truncate_bytes("hello", 3), "hel");
        // "é" is two bytes; a 2-byte clamp of "aé" must drop it, not split it.
        assert_eq!(truncate_bytes("aé", 2), "a");
    }
}
