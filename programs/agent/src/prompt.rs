//! The system prompt: the protocol the model follows, and the tool menu.

use super::*;

const PROTOCOL_PROMPT: &str = "You are an Aurora agent running inside a Wasm guest.\n\
The host owns all side effects. Reply with exactly one compact JSON object containing an \"actions\" array.\n\
Use only the tools listed below. Match each tool's input JSON schema exactly.\n\
You may request multiple independent tool calls in one turn. The host executes them sequentially and returns one aggregated observation array.\n\
Each observation has status \"result\" with content or status \"failed\" with an error. A failed tool call is recoverable: use other sources, retry when appropriate, or explain the limitation.\n\
Sections: {\"action\":\"sys.begin\",\"content\":{}} opens a section and {\"action\":\"sys.commit\",\"content\":{}} closes it. Open a section only around steps that are safe to run again. Sections nest.\n\
After receiving observations, either request more tools or return exactly one terminal action:\n\
{\"actions\":[{\"action\":\"final\",\"content\":{\"answer\":\"...\"}}]} to finish, or\n\
{\"actions\":[{\"action\":\"sys.abort\",\"content\":{\"reason\":\"...\"}}]} to abandon. With a section open, the run goes back to the start of the outermost open section and you decide again from there, at most 10 times. With no section open, the process stops as failed with that reason. Nothing that already ran is undone.\n\
Never combine a terminal action with tool calls in the same actions array.";

pub(crate) fn build_system_prompt(
    capabilities: &[Capability],
    skills: &[Skill],
    saves: bool,
    summarizes: bool,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(PROTOCOL_PROMPT);
    prompt.push_str("\n\nAvailable tools for this run:");
    // Hidden capabilities are dispatchable but kept off the discoverable menu.
    let visible: Vec<&Capability> = capabilities.iter().filter(|c| !c.hidden).collect();
    if visible.is_empty() {
        prompt.push_str("\nNone. Return a final action without attempting a tool call.");
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
    prompt.push_str("\n\nTool call shape:\n");
    prompt.push_str(
        r#"{"actions":[{"action":"<exact tool name>","content":<input matching that tool's schema>}]}"#,
    );
    if saves {
        prompt.push_str("\n\nEvery successful tool result, except those of core.scratch and the sys.* calls, is saved in this session's scratch memory. In place of the result, its observation gives {\"key\", \"bytes_length\"}. A saved result never changes, and it stays available in later messages of this session. Read a result with core.scratch: get returns it whole, and search finds lines in it.");
    }
    if summarizes {
        prompt.push_str(" To get an overview of a saved result without reading it here, use the summarize action. It summarizes the result in a separate conversation and returns only the summary:\n");
        prompt.push_str(
            r#"{"actions":[{"action":"summarize","content":{"key":"<key>","focus":"<what you need from it>"}}]}"#,
        );
    }
    prompt
}

fn push_skills(prompt: &mut String, skills: &[Skill]) {
    if skills.is_empty() {
        return;
    }
    prompt.push_str("\n\nSkills for this run. The operator wrote them: follow them when the task matches. They grant no tools beyond the list above.");
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
