# aurora-brains

A Rust workspace of Aurora **programs**. Each program compiles to a
WebAssembly module (`wasm32-wasip1`) and runs as a process inside the
[capcompute](https://github.com/aurora-capcompute/capcompute) processor. (The
repo keeps its old name. Its contents are programs, not blockchain code.)

A program is one agent's logic: its LLM loop and its decisions. It has no
ambient authority. It cannot read the real clock, get entropy, open a
connection or call an LLM. It asks the host through **syscalls**, and the host
checks, journals and replays each one. So a program can do only what its
manifest grants, and a crash does not lose its progress.

To run a program, put its `<name>.wasm` and `<name>.json` into
[aurora-dist](https://github.com/aurora-capcompute/aurora-dist)'s programs
directory and start the server. A `spawn` from
[aurora-cli](https://github.com/aurora-capcompute/aurora-cli) or the Slack
connector runs it.

## Programs

| Program | What it is |
| --- | --- |
| `agent` | The general agent: a tool-calling LLM loop over whatever its manifest grants. It handles several actions per turn, a savepoint per turn, web pages reduced to text, transcript compaction, and large reads offloaded to `core.scratch`. |
| `echo` | The smallest program, with no LLM: it returns its input, or `"pong"` when the input is empty. Copy it to start a new program. |

## The SDK

`sdk/` (the `aurora-program-sdk` crate) is the syscall boundary that every
program uses:

- `dispatch`: call a granted capability. A failure comes back as a response
  that the program can act on.
- `dispatch_hard`: the same call inside its own savepoint. A failure fails the
  process, and a retry runs the call again.
- `input`, `output`, `log`: read the input, record the answer, report progress.
- `savepoint()` / `commit()`: bracket a section. `abort(reason)` runs the open
  section again from its `savepoint()`. With no open section, the process
  fails with `reason`. Nothing is undone, so a section must be safe to run
  again.
- `now`, `random`: real time and entropy as journaled syscalls, so replay
  sees the same values.
- `yielded`: detects a yield (an approval or a timer), which the program
  passes up so the process can pause.
- `llm`: the chat call and model-reply decoding.

The envelope is JSON (ABI v4). Guest and host encode the same fields with their
standard libraries. The aurora-capcompute and aurora-dist tests build `agent`
and run it through the real Go host.

## The interface file

Each program has an `interface.json`: a one-line `description` and JSON Schemas
for its `input` and its answer (`output`).

```json
{
  "description": "Echoes the message back; \"pong\" when empty.",
  "input":  {"type": "string"},
  "output": {"type": "string"}
}
```

A conversational program declares `{"type":"string"}`. A structured one declares
an object schema. The build copies the file beside the Wasm. The host reads it
without running the program, shows it at `GET /v1/programs`, and checks every
input and answer against it.

## Build

Requirement: Rust (stable) with the `wasm32-wasip1` target.

```sh
rustup target add wasm32-wasip1
sh programs/echo/build.sh                                 # programs/echo/dist/{echo.wasm,echo.json}
cargo build --release --target wasm32-wasip1 -p agent    # one program
cargo build --release --target wasm32-wasip1 --workspace # all
cargo test --workspace
```

## Write a new program

1. Copy `programs/echo` to `programs/<name>` and change the package name.
2. Add `"programs/<name>"` to `members` in the root `Cargo.toml`.
3. In `src/lib.rs`, read the input with `sdk::input`, do the work, and record
   the answer with `sdk::output`. Export the entrypoint with `#[plugin_fn]`.
   Return `{"status":"completed"}`, or `{"status":"yielded"}` when a syscall
   yielded.
4. Write `interface.json`.
5. Keep the program deterministic: no clock, randomness or I/O except through
   the SDK.

## Layout

```
Cargo.toml        the workspace: sdk, programs/agent, programs/echo
sdk/src/lib.rs    dispatch, savepoints, input/output/log, abort, now/random, the envelope
sdk/src/llm.rs    the chat call and model-reply decoding
programs/<name>/  the crate (src/), interface.json, build.sh
```

Dependencies: `extism-pdk`, `serde`, `serde_json`, `anyhow`, and `lol_html` (only
in `agent`, to reduce HTML to text).
