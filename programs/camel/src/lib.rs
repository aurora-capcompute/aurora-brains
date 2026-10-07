//! camel — the plan/execute split program: the agent loop made
//! prompt-injection-resilient after CaMeL (Debenedetti et al. 2025,
//! "Defeating Prompt Injections by Design") and Willison's dual-LLM pattern.
//!
//! The processor already mediates every capability and tracks data-flow labels
//! host-side. This program supplies the missing guest half: the planning model
//! NEVER reads raw tool output. Each successful call's result is held in a
//! guest-side variable store ([`quarantine`]) as `$1`, `$2`, ...; the model
//! receives only a stub observation `{action, status, var}` — a failure is a
//! generic "failed" marker plus an optional machine code, never error text.
//! The model routes data it cannot read by writing the literal string `"$N"`
//! inside a later tool call's string arguments or its final answer, and the
//! guest substitutes the stored value only AFTER the model has chosen the
//! action. Injected text inside a tool result can therefore never name the
//! next action — it is data in a store, not words in the planner's context.
//!
//! Everything else mirrors `programs/agent`: the same conversation/action
//! protocol, per-turn savepoints, hard calls, and terminal final/abort
//! actions.

use aurora_program_sdk as sdk;
use aurora_program_sdk::llm::{self, decode_model_envelopes, looks_like_tool_batch, Message};
use aurora_program_sdk::{Call, Capability};
use extism_pdk::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

mod quarantine;
use quarantine::{substitute, StubObservation, VarStore};

const PROTOCOL_PROMPT: &str = "You are an Aurora agent running inside a Wasm guest, split into planner and executor: you plan; a quarantined executor runs the tools and keeps their output.\n\
The host owns all side effects. Reply with exactly one compact JSON object containing an \"actions\" array.\n\
Use only the tools listed below. Match each tool's input JSON schema exactly.\n\
You may request multiple independent tool calls in one turn. The host executes them sequentially and returns one aggregated observation array.\n\
You never see tool output. A successful call's result is stored as a guest-side variable and its observation is {\"action\":...,\"status\":\"result\",\"var\":\"$N\"}. A failed call's observation is {\"action\":...,\"status\":\"failed\",\"error\":\"failed\"}, plus a short machine \"code\" when one exists; error text is withheld. Plan from what you requested and where it came from, never from result content you expect to read.\n\
To use a stored result, write the literal string \"$N\" inside a later tool call's string arguments or in your final answer; the executor substitutes the stored value after you have chosen the action. A string that is exactly \"$N\" becomes the stored JSON value itself; \"$N\" inside a longer string becomes its text rendering (strings verbatim, other values compact JSON). Write \"$$\" for a literal \"$\". Referencing a variable that does not exist fails the turn.\n\
A failed tool call is recoverable by default: use other sources, retry when appropriate, or explain the limitation.\n\
Add \"hard\": true to a tool call only when its failure must abort the process so a later resume re-executes it (for example, a state-changing step the process cannot meaningfully continue without). Omit \"hard\" for all normal, recoverable calls.\n\
After receiving observations, either request more tools or return exactly one terminal action:\n\
{\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"...\",\"reason\":\"...\"}}]} to finish — the answer may reference $N and is substituted before delivery, or\n\
{\"actions\":[{\"action\":\"abort\",\"content\":{\"reason\":\"...\"}}]} to give up: the process stops as failed with that reason, and nothing it already did is undone; abort reasons are not substituted.\n\
Never combine a terminal action with tool calls in the same actions array.";

// MAX_STEPS is a hard cap on agentic turns; the last turn is forced to a final
// answer instead of looping forever. It mirrors programs/agent (camel keeps none
// of agent's compaction machinery).
const MAX_STEPS: u32 = 16;

const FINAL_DIRECTIVE: &str = "You have reached this run's step limit. Reply now with EXACTLY one final action and no tool calls: {\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"...\"}}]}. Base the answer on what you already have; the answer may reference $N.";

// -- Data structures --

#[derive(Serialize)]
struct Output {
    status: &'static str,
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

#[derive(Serialize)]
struct FinishArgs {
    answer: String,
}

// -- Entry point --

#[plugin_fn]
pub fn run(_: ()) -> FnResult<Json<Output>> {
    match run_camel() {
        Ok(()) => Ok(Json(Output {
            status: "completed",
        })),
        Err(e) if sdk::yielded(&e) => Ok(Json(Output { status: "yielded" })),
        Err(e) => Err(e.into()),
    }
}

fn run_camel() -> anyhow::Result<()> {
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

    // The quarantine: raw tool results live here, keyed $1, $2, ... — never
    // in `messages`.
    let mut store = VarStore::default();

    let mut step: u32 = 0;
    loop {
        step += 1;
        // Each agentic turn — one LLM call plus the tool calls it requests —
        // is a savepoint: sys.begin here, sys.commit at the turn's end. If the
        // turn breaks mid-way (a malformed model reply, an unknown variable
        // reference, an unavailable capability), the savepoint is left open
        // and a resumed run forks right after it, re-executing the WHOLE turn
        // live — including the LLM call, giving the model a fresh chance —
        // instead of deterministically replaying the broken completion forever.
        let turn = sdk::savepoint()?;
        // Bound the run: at the step budget, demand one final action this turn and
        // finish instead of looping forever. A minimal step cap mirroring
        // programs/agent (camel keeps none of agent's compaction machinery).
        if step >= MAX_STEPS {
            return finalize(turn, &mut messages, &store);
        }
        let chat = llm::chat(&messages)?;
        let envelopes = match decode_model_envelopes(&chat) {
            Ok(envelopes) => envelopes,
            // The reply didn't parse as an action envelope — usually the model
            // answered directly in prose instead of the required JSON. Take
            // that reply as the answer rather than failing the process, routing
            // it through the same $N substitution a proper final gets so the
            // quarantine still holds. The turn's LLM call is journaled, so a
            // resume replays this salvage. But a botched/truncated TOOL batch
            // (looks like a batch, no clean final) must NOT be salvaged:
            // publishing its raw {"actions":...} text would leak protocol as the
            // answer. Leave the savepoint open and fail so the turn re-drives and
            // the model retries.
            Err(e) => {
                if looks_like_tool_batch(&chat) {
                    return Err(e);
                }
                let answer = salvage(&chat, &store)?;
                turn.commit()?;
                return sdk::output(&FinishArgs { answer });
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
                // The reason is model-authored control metadata; it is passed
                // through unsubstituted so quarantined data never rides out on
                // the abort channel.
                let reason = envelopes[idx]
                    .content
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                turn.commit()?;
                return sdk::abort(&reason);
            }
            if let Some(idx) = first_final_idx {
                // Resolve $N references before committing, so a bad final
                // (missing answer, unknown variable) reopens the turn on
                // resume instead of wedging after the commit.
                let answer = final_answer(&envelopes[idx].content, &store)?;
                turn.commit()?;
                return sdk::output(&FinishArgs { answer });
            }
        }

        messages.push(Message {
            role: "assistant".into(),
            content: chat,
        });

        let mut observations: Vec<StubObservation> = Vec::with_capacity(envelopes.len());
        for (i, envelope) in envelopes.iter().enumerate() {
            if envelope.action == "final" {
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
            // The CaMeL move: the model has already chosen the action and
            // authored its args; only now does quarantined data flow in.
            let args = substitute(&envelope.content, &store)?;
            // Progress lines stay content-free: substituted args may carry
            // quarantined data.
            sdk::log(&format!("⚙ {}", envelope.action));
            let tool_call = Call {
                name: envelope.action.clone(),
                args: Some(args),
            };
            let response = if envelope.hard {
                sdk::dispatch_hard(&tool_call)?
            } else {
                sdk::dispatch(&tool_call)?
            };
            // Quarantine the outcome. The raw result (and the raw error
            // message) never reaches `messages` — the model gets a stub.
            let obs = if response.status == sdk::STATUS_FAILED {
                StubObservation::failed(&envelope.action, &response.code)
            } else {
                let var = store.insert(response.result.unwrap_or(Value::Null));
                StubObservation::result(&envelope.action, var)
            };
            observations.push(obs);
        }

        let raw_obs = serde_json::to_string(&observations)
            .map_err(|e| anyhow::anyhow!("encode stub observations: {}", e))?;
        // Feed the stubs back as a user message, matching the agent program (the
        // "tool" role is reserved by the OpenAI/DeepSeek API for native
        // function-call results; this program uses a text protocol). Only stubs
        // travel this way — never tool output.
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
    prompt.push_str("\nRouting a stored result into a later call:\n");
    prompt.push_str(r#"{"actions":[{"action":"<tool>","content":{"field":"$1","note":"see $2"}}]}"#);
    Ok(prompt)
}

/// final_answer resolves a model-authored final answer. The model must author
/// a non-empty answer string; $N references in it are substituted, and a
/// whole-string "$N" naming a non-string value is rendered as text — the
/// run's answer is a string.
fn final_answer(content: &Value, store: &VarStore) -> anyhow::Result<String> {
    let authored = content
        .get("answer")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("invalid final action: answer is required"))?;
    if authored.is_empty() {
        anyhow::bail!("final action missing answer");
    }
    let substituted = substitute(&Value::String(authored.to_string()), store)?;
    Ok(match substituted {
        Value::String(s) => s,
        other => other.to_string(),
    })
}

/// salvage turns a reply that didn't parse as an action envelope into an answer
/// — the model answered in prose instead of the JSON protocol. Its $N
/// references are substituted just like a proper final's, so the quarantine
/// still holds; an out-of-range reference (or empty reply) falls back to the raw
/// trimmed text rather than failing the run.
fn salvage(chat: &str, store: &VarStore) -> anyhow::Result<String> {
    let trimmed = chat.trim();
    if trimmed.is_empty() {
        return Ok("The model returned an empty reply.".into());
    }
    match substitute(&Value::String(trimmed.to_string()), store) {
        Ok(Value::String(s)) => Ok(s),
        Ok(other) => Ok(other.to_string()),
        Err(_) => Ok(trimmed.to_string()),
    }
}

/// finalize forces the run to a close at the step budget: it asks the model for
/// exactly one final action and publishes that answer. If the model instead
/// returns tools or a botched batch, a neutral note is published rather than its
/// raw {"actions":...} protocol text; a plain prose reply is salvaged. A minimal
/// mirror of programs/agent's finalize, without the compaction machinery.
fn finalize(
    turn: sdk::Savepoint,
    messages: &mut Vec<Message>,
    store: &VarStore,
) -> anyhow::Result<()> {
    messages.push(Message {
        role: "user".into(),
        content: FINAL_DIRECTIVE.into(),
    });
    let chat = llm::chat(messages)?;
    // Resolve any $N in a clean final before committing, so a bad final reopens
    // the turn on resume instead of wedging after the commit.
    if let Ok(envelopes) = decode_model_envelopes(&chat) {
        if let Some(idx) = envelopes.iter().position(|e| e.action == "final") {
            let answer = final_answer(&envelopes[idx].content, store)?;
            turn.commit()?;
            return sdk::output(&FinishArgs { answer });
        }
    }
    // No clean final: salvage a prose reply, but never echo a tool batch's raw
    // protocol text as the answer.
    let answer = if looks_like_tool_batch(&chat) {
        "Reached the step limit before completing the task.".to_string()
    } else {
        salvage(&chat, store)?
    };
    turn.commit()?;
    sdk::output(&FinishArgs { answer })
}

// -- Model reply decoding (mirrors programs/agent: same conversation/action
// protocol; kept in-program because it is cognition-level, not syscall
// plumbing) --

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store_with(values: &[Value]) -> VarStore {
        let mut store = VarStore::default();
        for v in values {
            store.insert(v.clone());
        }
        store
    }

    // -- final_answer --

    #[test]
    fn final_answer_substitutes_embedded_refs() {
        let store = store_with(&[json!("42 items")]);
        let got = final_answer(&json!({"answer": "Found: $1."}), &store).unwrap();
        assert_eq!(got, "Found: 42 items.");
    }

    #[test]
    fn final_answer_renders_whole_ref_non_string_as_text() {
        let store = store_with(&[json!({"total": 3})]);
        let got = final_answer(&json!({"answer": "$1"}), &store).unwrap();
        assert_eq!(got, r#"{"total":3}"#);
    }

    #[test]
    fn final_answer_passes_whole_ref_string_through() {
        let store = store_with(&[json!("just text")]);
        let got = final_answer(&json!({"answer": "$1"}), &store).unwrap();
        assert_eq!(got, "just text");
    }

    #[test]
    fn final_answer_requires_a_model_authored_answer() {
        let store = VarStore::default();
        assert!(final_answer(&json!({}), &store).is_err());
        assert!(final_answer(&json!({"answer": ""}), &store).is_err());
        assert!(final_answer(&json!({"answer": 7}), &store).is_err());
    }

    #[test]
    fn final_answer_fails_on_unknown_variable() {
        let store = VarStore::default();
        assert!(final_answer(&json!({"answer": "$1"}), &store).is_err());
    }

    // -- salvage (a prose reply that isn't the action envelope) --

    #[test]
    fn salvage_returns_prose_verbatim() {
        let store = VarStore::default();
        assert_eq!(salvage("\n\nHere is the answer.\n", &store).unwrap(), "Here is the answer.");
    }

    #[test]
    fn salvage_substitutes_quarantine_refs() {
        let store = store_with(&[json!("42 items")]);
        assert_eq!(salvage("Found: $1.", &store).unwrap(), "Found: 42 items.");
    }

    #[test]
    fn salvage_of_a_bad_ref_falls_back_to_raw_text() {
        let store = VarStore::default();
        assert_eq!(salvage("see $9 for details", &store).unwrap(), "see $9 for details");
    }

    #[test]
    fn salvage_of_empty_reply_is_nonempty() {
        let store = VarStore::default();
        assert!(!salvage("   ", &store).unwrap().is_empty());
    }

}
