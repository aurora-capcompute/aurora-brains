//! The Aurora agent: a tool-calling LLM loop over the capabilities its
//! manifest grants. run_agent is the loop. Each turn it keeps the transcript
//! within bounds, asks the model, and either ends the run or runs the tool calls
//! the model asked for and feeds the results back.
//!
//!   prompt  the system prompt and the tool menu
//!   chat    every LLM call, and the transcript's size
//!   tools   tool calls, saved results, and observations
//!   answer  ending a run: the final answer, or a salvaged one
//!   html    web pages reduced to text

mod answer;
mod chat;
mod html;
mod prompt;
mod tools;

use answer::{
    abort_reason, answer_rejected, publish_final, recover_answer_field, reply_again, wrap_up,
};
use aurora_program_sdk as sdk;
use aurora_program_sdk::llm::{
    self, decode_model_envelopes, extract_json_region, looks_like_tool_batch, Message,
    ModelEnvelope,
};
use aurora_program_sdk::{Call, Capability, Skill};
use chat::{chat_within_budget, finalize, maybe_compact, messages_bytes};
use extism_pdk::*;
use prompt::build_system_prompt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tools::Tools;

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
    #[serde(default)]
    skills: Vec<Skill>,
    /// Which attempt this is (bumped by the host per retry) — lets the model
    /// know earlier attempts were rolled back and back off accordingly.
    #[serde(default)]
    attempt: u32,
}

#[derive(Serialize)]
struct FinishArgs {
    answer: String,
}

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
    // The host checked the input against the program's interface, and it built
    // the history from completed turns: both are trusted as they come.
    let inp: Input = sdk::input()?;
    let tools = Tools::new(&inp.capabilities, &inp.input);

    let mut system_prompt = build_system_prompt(
        &inp.capabilities,
        &inp.skills,
        tools.saves,
        tools.summarizes,
    );
    if inp.attempt > 1 {
        system_prompt.push_str(&format!(
            "\nThis is attempt {} of this task. The earlier attempts stopped before they finished. The host did not undo their steps.",
            inp.attempt
        ));
    }
    let mut messages = vec![Message {
        role: "system".into(),
        content: system_prompt,
    }];
    messages.extend(inp.history);
    messages.push(Message {
        role: "user".into(),
        content: inp.input,
    });
    // Compaction never touches the head: the system prompt, the session
    // history and this run's input, which can carry instructions that must not
    // be paraphrased.
    let head = messages.len();

    for step in 1.. {
        // Keep the transcript within bounds. The summary is itself a journaled
        // LLM call, so a resume rebuilds the same transcript by replay.
        maybe_compact(&mut messages, head)?;
        // At the step budget, or still too large after compaction, demand a
        // final answer this turn.
        if step >= MAX_STEPS || messages_bytes(&messages) >= HARD_CEILING {
            return finalize(&mut messages);
        }

        let chat = chat_within_budget(&mut messages)?;
        let envelopes = match decode_model_envelopes(&chat) {
            Ok(envelopes) => envelopes,
            // A truncated or botched tool batch with no answer in it goes back
            // to the model: publishing it would leak protocol as the answer.
            Err(e) if looks_like_tool_batch(&chat) && recover_answer_field(&chat).is_none() => {
                reply_again(&mut messages, chat, format!("Your reply is not a valid actions object: {e}. Reply again with exactly one valid JSON object."));
                continue;
            }
            // Any other reply that does not parse — prose, or a truncated final
            // — is still the answer.
            Err(_) => match wrap_up(&chat)? {
                Ok(()) => return Ok(()),
                Err(reason) => {
                    reply_again(&mut messages, chat, answer_rejected(&reason));
                    continue;
                }
            },
        };

        // A reply with no tool call ends the run: an abort, or a final answer.
        // The runtime reads an abort off the journal's tail, so nothing may
        // follow it.
        let has_tool = envelopes
            .iter()
            .any(|e| e.action != "final" && e.action != "sys.abort");
        if !has_tool {
            if let Some(envelope) = envelopes.iter().find(|e| e.action == "sys.abort") {
                return sdk::abort(&abort_reason(envelope));
            }
            if let Some(envelope) = envelopes.iter().find(|e| e.action == "final") {
                match publish_final(envelope)? {
                    Ok(()) => return Ok(()),
                    Err(reason) => {
                        reply_again(&mut messages, chat, answer_rejected(&reason));
                        continue;
                    }
                }
            }
        }

        // Run the tool calls, and feed their observations back as a user
        // message. The "tool" role belongs to native function calling, which
        // this text protocol does not use, so a provider would refuse it.
        let observations = tools.run(&envelopes)?;
        messages.push(Message {
            role: "assistant".into(),
            content: chat,
        });
        messages.push(Message {
            role: "user".into(),
            content: serde_json::to_string(&observations)?,
        });
    }
    unreachable!("the step loop returns at MAX_STEPS")
}
