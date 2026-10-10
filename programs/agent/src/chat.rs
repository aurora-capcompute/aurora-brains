//! Every LLM call, and keeping the transcript inside the model's window.
//!
//! Each call runs in its own section, so a failed call leaves the section open
//! and a resume calls the LLM again. A request too large for the driver sheds
//! bytes and is sent again. Past COMPACT_THRESHOLD, the oldest of the run's own
//! turns are summarized into one message.

use super::*;

const SUMMARY_PROMPT: &str = "You compress an AI agent's earlier working log. Preserve every fact, URL, identifier, number, finding, and decision needed to finish the task; drop repetition and chatter. Reply with prose only — no JSON, no tool calls.";

const FINAL_DIRECTIVE: &str = "You have reached this run's step or size limit. Reply now with EXACTLY one final action and no tool calls: {\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"...\"}}]}. Base the answer on what you already have.";

// chat_in_section makes one LLM call inside its own section. An LLM call has no
// effects, so it is safe to run again: when it fails, the section stays open
// and the process fails, and a resume forks at the outermost open section to
// call the LLM again. A size error is the exception: the caller sheds bytes and
// calls again, so the section is closed first.
fn chat_in_section(messages: &[Message]) -> anyhow::Result<String> {
    let section = sdk::savepoint()?;
    match llm::chat(messages) {
        Ok(reply) => {
            section.commit()?;
            Ok(reply)
        }
        Err(e) if is_size_error(&e.to_string()) => {
            section.commit()?;
            Err(e)
        }
        Err(e) => Err(e),
    }
}

// chat_within_budget calls the LLM and, if the request is rejected for size
// (the core.openaiApi max_request_bytes cap, or a provider context-length
// error), sheds transcript bytes and retries until it fits or nothing more can
// be shed. It adapts to whatever request cap the host is configured with, below
// the proactive COMPACT_THRESHOLD/HARD_CEILING bounds. Each failed-then-retried
// call is journaled, so a crash-resume replays the same sequence. A host yield
// (e.g. an approval) is not a size error and propagates unchanged.
pub(crate) fn chat_within_budget(messages: &mut [Message]) -> anyhow::Result<String> {
    loop {
        match chat_in_section(messages) {
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
pub(crate) fn messages_bytes(messages: &[Message]) -> usize {
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
pub(crate) fn maybe_compact(messages: &mut Vec<Message>, head: usize) -> anyhow::Result<()> {
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
    chat_in_section(&msgs)
}

// finalize forces the run to a close by asking the model for exactly one final
// action. If the model complies, that answer is published; if it still refuses
// (returns tools, or unparseable), its best text is salvaged so the run always
// terminates with an answer. At the limit there is no turn left to correct an
// answer the host rejects, so a rejection fails the process.
pub(crate) fn finalize(messages: &mut Vec<Message>) -> anyhow::Result<()> {
    messages.push(Message {
        role: "user".into(),
        content: FINAL_DIRECTIVE.into(),
    });
    let chat = chat_within_budget(messages)?;
    wrap_up(&chat)?.map_err(|reason| anyhow::anyhow!("the final answer was not accepted: {reason}"))
}
