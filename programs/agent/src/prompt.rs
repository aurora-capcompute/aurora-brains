//! The system prompt: the protocol the model follows, and the tool menu.

use super::*;

const PROTOCOL_PROMPT: &str = r#"You are an Aurora agent. You run inside a Wasm sandbox. Only the host can do something outside the sandbox. You act through tool calls.

Reply format:
- Reply with exactly one JSON object, with no text before or after it. Write the object on one line.
- The object has one key, "actions". Its value is a list of actions.
- Each action has two keys: "action" is the tool name, and "content" is the tool input.
- Use only the tools in the list below. Each input must match the input schema of its tool exactly.

Tool calls:
- One reply can contain more than one tool call. The host runs the calls in order.
- The host then returns a list with one observation for each call.
- An observation has the status "result" and a "content", or the status "failed" and an "error".
- A failed call does not stop the run. If the call failed for a reason other than access, use a different source, try the call again, or explain the problem.

Access limits:
- A call can fail because you do not have the access. The tool is not granted, an argument is not allowed, or a policy refuses the call.
- Do not try to go around an access limit. Do not try other tools or other arguments to get the same access.
- Do the parts of the task that your access permits. Then finish the run with the information that you have.
- In the answer, tell the user which access you did not have. Tell the user which tools would help you do the task.
- sys.declassify asks a human to lift a label. It is not a way around a limit.

Sections:
- {"action":"sys.begin","content":{}} opens a section. {"action":"sys.commit","content":{}} closes it.
- Open a section only around steps that are safe to run again.
- A section can be inside another section.

End of the run:
- To finish, reply with only a final action:
  {"actions":[{"action":"final","content":{"answer":"..."}}]}
- To abandon the work, reply with only an abort action:
  {"actions":[{"action":"sys.abort","content":{"reason":"..."}}]}
- If a section is open, an abort sends the run back to the sys.begin of the outermost open section. Then you decide again. This can occur at most 10 times.
- If no section is open, an abort stops the process with the status "failed" and your reason.
- An abort does not undo the steps that ran.
- Do not put a final action or an abort action in a reply with tool calls."#;

pub(crate) fn build_system_prompt(
    capabilities: &[Capability],
    skills: &[Skill],
    saves: bool,
    summarizes: bool,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(PROTOCOL_PROMPT);
    prompt.push_str("\n\nTools for this run:");
    // Hidden capabilities are dispatchable but kept off the discoverable menu.
    let visible: Vec<&Capability> = capabilities.iter().filter(|c| !c.hidden).collect();
    if visible.is_empty() {
        prompt.push_str("\nNone. Do not call a tool. Reply with a final action.");
        return prompt;
    }
    for (i, tool) in visible.iter().enumerate() {
        let schema = if tool.input_schema.is_null() {
            serde_json::json!({})
        } else {
            tool.input_schema.clone()
        };
        prompt.push_str(&format!("\n\nTool {}\nName: {}", i + 1, tool.name));
        let description = tool.description.trim();
        if !description.is_empty() {
            prompt.push_str(&format!("\nDescription: {}", description));
        }
        prompt.push_str(&format!("\nInput JSON schema: {}", schema));
    }
    push_skills(&mut prompt, skills);
    prompt.push_str("\n\nThe shape of a tool call:\n");
    prompt.push_str(
        r#"{"actions":[{"action":"<exact tool name>","content":<input matching that tool's schema>}]}"#,
    );
    if saves {
        prompt.push_str(concat!(
            "\n\nSaved results:",
            "\n- The host saves each successful tool result in the scratch memory of this session. It does not save results of core.scratch or of sys.* calls.",
            "\n- For a saved result, the observation gives {\"key\", \"bytes_length\"} in place of the result.",
            "\n- A saved result does not change. It stays available for the rest of this session.",
            "\n- To read a saved result, use core.scratch. The operation \"get\" returns all of it. The operation \"search\" finds lines in it.",
        ));
    }
    if summarizes {
        prompt.push_str(concat!(
            "\n- To get a summary of a saved result, use the summarize action. The host makes the summary in a separate conversation. It returns only the summary to this conversation:",
            "\n  ",
            r#"{"actions":[{"action":"summarize","content":{"key":"<key>","focus":"<what you need from it>"}}]}"#,
        ));
    }
    prompt
}

fn push_skills(prompt: &mut String, skills: &[Skill]) {
    if skills.is_empty() {
        return;
    }
    prompt.push_str("\n\nSkills for this run. The operator wrote these skills. When a skill matches the task, follow its instructions. A skill gives no tools in addition to the list above.");
    for skill in skills {
        prompt.push_str(&format!("\n\nSkill: {}", skill.name.trim()));
        let description = skill.description.trim();
        if !description.is_empty() {
            prompt.push_str(&format!("\nDescription: {}", description));
        }
        let instructions = skill.instructions.trim();
        if !instructions.is_empty() {
            prompt.push_str(&format!("\nInstructions:\n{}", instructions));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Capability {
        Capability {
            name: name.into(),
            ..Default::default()
        }
    }

    #[test]
    fn skills_follow_the_tools() {
        let skills = [Skill {
            name: "status".into(),
            description: "check service status".into(),
            instructions: "Read the status page first.".into(),
        }];
        let prompt = build_system_prompt(&[tool("core.internet")], &skills, false, false);
        let tools = prompt.find("Name: core.internet").expect("tool listed");
        let skill = prompt.find("Skill: status").expect("skill listed");
        assert!(tools < skill, "skills come after the tool list");
        assert!(prompt.contains("Description: check service status"));
        assert!(prompt.contains("Instructions:\nRead the status page first."));
    }

    #[test]
    fn no_skills_no_section() {
        let prompt = build_system_prompt(&[tool("core.internet")], &[], false, false);
        assert!(!prompt.contains("Skills for this run"));
    }
}
