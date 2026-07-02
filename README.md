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
                    host by shared golden fixtures) and the dispatch protocol
                    (result/failed observations, the yield sentinel,
                    savepoint-bracketed "hard" calls)
brains/
  agent/          the general-purpose agent: a tool-calling LLM loop over
                    whatever capabilities its manifest grants
```

A brain crate contains only cognition; the boundary lives in the SDK.

## Building a brain

```sh
sh brains/agent/build.sh     # → brains/agent/dist/agent-brain.wasm
```

or directly:

```sh
cargo build --release --target wasm32-wasip1 -p agent-brain
```

## Adding a new brain

1. `mkdir -p brains/<name>/src` and copy `brains/agent/Cargo.toml`, changing
   the package name and description.
2. Add `"brains/<name>"` to the workspace members in the root `Cargo.toml`.
3. Write `src/lib.rs`: depend on `aurora_brain_sdk::{dispatch, dispatch_hard,
   yielded, Call}` and export the entrypoint with `#[plugin_fn]`. Return
   `{"status":"completed",...}` or bubble the yield sentinel up and return
   `{"status":"yielded"}`.
4. Keep the brain deterministic: no clocks, no randomness, no I/O outside
   `dispatch` — the kernel pins the ambient sources and the journal replays
   the rest.

## Tests

```sh
cargo test --workspace
```

The SDK's wire tests include golden byte fixtures shared verbatim with the
host's codec (capcompute `sys/wire/wire_interop_test.go`) — the
cross-language pin for the ABI.
