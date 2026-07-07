# aurora-brains

WebAssembly agent brains for Aurora, as a Rust workspace. A **brain** is the
cognition of one agent program: it runs as a deterministic wasm guest inside
the Aurora kernel ([capcompute](https://github.com/aurora-capcompute/capcompute)),
holds zero ambient authority, and reaches the world only through journaled
syscalls — every side effect it requests is validated, flow-checked,
replayable, and auditable by the host.

## Layout

```
Cargo.toml        the workspace: shared versions and dependencies
sdk/              aurora-brain-sdk — everything every brain needs:
                    the ABI v3 wire codec (proto3, hand-rolled, pinned to the
                    host by shared golden fixtures); the dispatch protocol
                    (result/failed observations, the yield sentinel,
                    savepoints, savepoint-bracketed "hard" calls); the
                    typed plumbing — input/output/log and the decoded
                    Capability menu the host grants; and rollback —
                    compensate(&Call) registers an effect's undo (deferred,
                    journaled, same shape as a dispatch) and abort(reason,
                    retry_seconds) undoes the registered effects newest-first,
                    then retries the section after the delay or stops
brains/
  agent/          the general-purpose agent: a tool-calling LLM loop over
                    whatever capabilities its manifest grants
  camel/          the plan/execute split brain (CaMeL; Debenedetti et al.
                    2025): the agent loop with the planner quarantined from
                    tool output — results live guest-side as $1, $2, ...; the
                    model sees only {action, status, var} stubs (failures: a
                    generic marker + machine code, no error text) and routes
                    data by writing "$N", substituted by the guest after the
                    action is chosen. Injected tool output can name no new
                    actions because the planner never reads it; limits: the
                    args the model authors are still model-chosen, and the
                    host's labels/capability policy remain the other half
  echo/           the smallest brain: no LLM, just input→output — the
                    multi-program path on the shared SDK
```

A brain crate contains only cognition; the boundary lives in the SDK.

## A program describes itself

Every brain exports a second, pure entrypoint next to `run`: `describe`, which
returns an `sdk::Interface` — a one-line `description` plus JSON Schemas for the
process's input `message` and its answer `output`. Conversational brains declare
`{"type":"string"}`; a structured program declares object schemas and callers
pass/receive JSON text. The host extracts the interface at registration (a pure
instantiation with syscalls stubbed out — a program that dispatches during
`describe` fails to register), so it travels inside the wasm and the program's
content digest covers it. From there the host publishes it to callers (the
`sys.spawn` menu a parent LLM reads, the program directory a user lists) and
validates every input message and answer against the schemas. That is how a
caller — model or human — knows what to pass a program without reading its code.

## Building a brain

```sh
sh brains/agent/build.sh     # → brains/agent/dist/agent-brain.wasm
```

or directly:

```sh
cargo build --release --target wasm32-wasip1 -p agent-brain
```

See `brains/echo` for the smallest possible brain — input→output with no LLM.

1. `mkdir -p brains/<name>/src` and copy `brains/echo/Cargo.toml`, changing
   the package name and description.
2. Add `"brains/<name>"` to the workspace members in the root `Cargo.toml`.
3. Write `src/lib.rs`: read the run's input with `sdk::input`, do the
   cognition, report the result with `sdk::output` (and `sdk::log` for
   progress), and export the entrypoint with `#[plugin_fn]`. Return
   `{"status":"completed"}` or bubble the yield sentinel (`sdk::yielded`) up
   and return `{"status":"yielded"}`. Also export a pure `describe`
   (`#[plugin_fn]`) returning `sdk::Interface` — the program's description and
   input/output schemas (see `brains/echo` for the minimal shape).
4. Keep the brain deterministic: no clocks, no randomness, no I/O outside the
   SDK's syscalls — the kernel pins the ambient sources and the journal
   replays the rest.

## Tests

```sh
cargo test --workspace
```

The SDK's wire tests include golden byte fixtures shared verbatim with the
host's codec (capcompute `sys/wire/wire_interop_test.go`) — the
cross-language pin for the ABI.
