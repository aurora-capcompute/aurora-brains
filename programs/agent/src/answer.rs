//! Ending a run: publishing the model's final answer, or salvaging one from a
//! reply that did not parse, and telling the model when the host rejected it.

use super::*;

// publish_final publishes a final action's answer. A final with no answer, and
// an answer the host rejects against the program's output schema, come back as
// the reason, so the model can answer again.
pub(crate) fn publish_final(envelope: &ModelEnvelope) -> anyhow::Result<Result<(), String>> {
    match envelope.content.get("answer").and_then(Value::as_str) {
        Some(answer) if !answer.is_empty() => sdk::output(&FinishArgs {
            answer: answer.to_string(),
        }),
        _ => Ok(Err(
            "a final action needs a non-empty \"answer\" string".into()
        )),
    }
}

// wrap_up ends the run from a model reply that ought to carry a terminal
// action: use its final (or abort) when the reply parses, otherwise salvage a
// usable answer from it. Shared by the step/size-limit path (finalize) and the
// recovery when a mid-loop reply doesn't parse as the action envelope at all —
// so a model that simply answered in prose finishes the process instead of
// failing it. An answer the host rejects comes back as its reason.
pub(crate) fn wrap_up(chat: &str) -> anyhow::Result<Result<(), String>> {
    if let Ok(envelopes) = decode_model_envelopes(chat) {
        if let Some(envelope) = envelopes.iter().find(|e| e.action == "final") {
            return publish_final(envelope);
        }
        if let Some(envelope) = envelopes.iter().find(|e| e.action == "sys.abort") {
            return sdk::abort(&abort_reason(envelope)).map(Ok);
        }
    }
    sdk::output(&FinishArgs {
        answer: salvage_answer(chat),
    })
}

// abort_reason reads the reason from an abort envelope.
pub(crate) fn abort_reason(envelope: &ModelEnvelope) -> String {
    envelope
        .content
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

// reply_again puts the model's reply and a note on what was wrong with it into
// the transcript, so the model can reply again in its next turn.
pub(crate) fn reply_again(messages: &mut Vec<Message>, reply: String, note: String) {
    messages.push(Message {
        role: "assistant".into(),
        content: reply,
    });
    messages.push(Message {
        role: "user".into(),
        content: note,
    });
}

// answer_rejected is the note on a final answer that was not accepted.
pub(crate) fn answer_rejected(reason: &str) -> String {
    format!("The host did not accept your final answer: {reason}. Correct the answer. Then reply again with a final action.")
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
        "The run reached its step or size limit before it finished the task.".into()
    } else {
        trimmed.to_string()
    }
}

// recover_answer_field pulls the text of an "answer" string field out of a reply
// whose JSON did not parse. It scans to the first "answer" key and decodes the
// JSON string that follows, stopping at the closing quote or at end-of-input — so
// a string truncated mid-value yields what was received rather than nothing.
pub(crate) fn recover_answer_field(chat: &str) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let truncated = r#"{"actions":[{"action":"final","content":{"answer":"**HWaaS** is Hardware as a Service, a model where"#;
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
}
