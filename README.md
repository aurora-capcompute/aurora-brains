# aurora-brains

The Aurora **agent** program and its SDK. The agent compiles to a WebAssembly
module (`wasm32-wasip1`) and runs as a process inside the
[capcompute](https://github.com/aurora-capcompute/capcompute) processor. (The
repo keeps its old name. Its contents are the agent, not blockchain code.)

The agent is a tool-calling LLM loop over whatever its manifest grants. It
handles several actions per turn, a savepoint per turn, web pages reduced to
text, and transcript compaction. Compaction summarizes only the run's own turns:
the system prompt, the session history and the run's input stay word for word.

When the manifest grants `core.scratch` with `store`, the agent saves every
successful tool result, except the results of `core.scratch` itself, in the
session's scratch memory, under a key made from the result's content
(`io/<hash>`). A saved result never changes. In place of each result, the model
gets only `{key, bytes_length}`, and it reads the result with `core.scratch`. With `get` also granted,
the model can call the `summarize` action. The agent then summarizes the saved
result in a separate LLM chat, and only the summary enters the model's
conversation.

It has no ambient authority. It cannot read the real clock, get entropy, open a
connection or call an LLM. It asks the host through **syscalls**, and the host
checks, journals and replays each one. So the agent can do only what its
manifest grants, and a crash does not lose its progress.

To run it, put `agent.wasm` and `agent.json` into
[aurora-dist](https://github.com/aurora-capcompute/aurora-dist)'s programs
directory and start the server. A `spawn` from
[aurora-cli](https://github.com/aurora-capcompute/aurora-cli) or the Slack
connector runs it.

## The SDK

`sdk/` (the `aurora-program-sdk` crate) is the syscall boundary:

- `dispatch`: call a granted capability. A failure comes back as a response
  that the agent can act on.
- `dispatch_hard`: the same call inside its own savepoint. A failure fails the
  process, and a retry runs the call again.
- `input`, `output`, `log`: read the input, record the answer, report progress.
- `savepoint()` / `commit()`: bracket a section. `abort(reason)` runs the open
  section again from its `savepoint()`. With no open section, the process
  fails with `reason`. Nothing is undone, so a section must be safe to run
  again.
- `yielded`: detects a yield (an approval or a timer), which the agent passes
  up so the process can pause.
- `llm`: the chat call and model-reply decoding.

The envelope is JSON (ABI v4). Guest and host encode the same fields with their
standard libraries. The aurora-capcompute and aurora-dist tests build the agent
and run it through the real Go host.

## The interface file

`programs/agent/interface.json` holds a one-line `description` and JSON Schemas
for the agent's `input` and its answer (`output`). Both are strings. The build
copies the file beside the Wasm as `agent.json`. The host reads it without
running the agent, shows it at `GET /v1/programs`, and checks every input and
answer against it.

## Build

Requirement: Rust (stable) with the `wasm32-wasip1` target.

```sh
rustup target add wasm32-wasip1
sh programs/agent/build.sh      # programs/agent/dist/{agent.wasm,agent.json}
cargo test --workspace
```

## Layout

```
Cargo.toml          the workspace: sdk, programs/agent
sdk/src/lib.rs      dispatch, savepoints, input/output/log, abort, the envelope
sdk/src/llm.rs      the chat call and model-reply decoding
programs/agent/     the agent crate (src/), interface.json, build.sh
```

Dependencies: `extism-pdk`, `serde`, `serde_json`, `anyhow`, and `lol_html` (to
reduce HTML to text).
