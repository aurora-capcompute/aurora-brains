# aurora-brains

**The "brains" of Aurora agents — the actual decision‑making code.** This is a Rust
workspace of small programs that each compile to a WebAssembly (`wasm32-wasip1`)
module and run as a sandboxed agent inside the Aurora kernel.

> The repo is still named `aurora-brains`; its contents are Aurora **programs**
> (a rename to `aurora-programs` is pending).

> **Not blockchain.** Despite the `programs/` + `sdk/` layout, this has nothing to
> do with Solana/Anchor or any chain. "Programs" here are Wasm agent modules.

---

## What is this, in plain words?

A **program** is the cognition of one agent — its LLM loop and decision logic. It
runs as a deterministic Wasm guest inside the
[capcompute](https://github.com/aurora-capcompute/capcompute) kernel and holds
**zero ambient authority**: it can't read the clock, generate randomness, make a
network call, or even call an LLM on its own. It can only *ask* the host to do those
things through **journaled syscalls** — each one validated, permission‑checked,
recorded, replayable, and reversible by the host. A running program is a **process**.

Why bother? Because it lets you run LLM‑driven agent logic **safely, deterministically,
and auditably**, with crash‑recovery and undo built in — the agent literally cannot
do anything the host didn't grant.

## Where this fits in the Aurora system

```
        you (a human)
              │
   aurora-cli / aurora-slack-connector      ← clients you talk to
              │  HTTP /v1
         aurora-dist                         ← the server (one binary you run)
              │  assembled from…
   ┌──────────┴──────────┐
 aurora-capcompute    aurora-dispatchers     ← orchestration runtime + capability drivers
   └──────────┬──────────┘
              │  both built on
         capcompute                          ← the kernel (the foundation)

   aurora-brains  ◀── YOU ARE HERE (the Wasm agent "programs" that run inside)
```

You build a program here into a `<name>.wasm` + `<name>.json` pair, drop it into
[aurora-dist](https://github.com/aurora-capcompute/aurora-dist)'s programs directory,
and the running server loads it. When you `spawn` a process (via
[aurora-cli](https://github.com/aurora-capcompute/aurora-cli) or the Slack bot), *this*
is the code that runs.

## The programs (features)

| Program | What it is |
| --- | --- |
| **`echo`** | The smallest possible program — no LLM, just input → output (returns `"pong"` when empty). The minimal example to copy. |
| **`agent`** | The general‑purpose agent: a tool‑calling LLM loop over whatever capabilities its manifest grants. Handles multi‑action turns, per‑turn savepoints, rollback, stripping fetched web pages to text, transcript compaction, and offloading big reads to scratch memory. |
| **`camel`** | Same protocol as `agent`, but prompt‑injection‑resistant (the **CaMeL** pattern). The planner LLM never sees raw tool output — results live guest‑side as `$1, $2, …` and the model only sees `{action, status, var}` stubs, so injected text in a tool result can never choose the next action. |

Cross‑cutting features every program gets from the **SDK** (`sdk/`, the
`aurora-program-sdk` crate):

- **Capability‑based access** — a program only reaches the tools its manifest grants.
- **Deterministic replay & crash recovery** — clock (`sdk::now`) and randomness
  (`sdk::random`) are syscalls, journaled on first run and replayed verbatim.
- **Savepoints + transactional rollback** — `savepoint()` / `commit()`, plus
  `compensate(&Call)` to register undo actions and `abort(reason, retry_seconds)` to
  unwind them newest‑first and retry.
- **Yield / resume** — pause on outside work (approval, timer) via the yield sentinel.
- **Declarative interface manifests** — every program ships an `interface.json` the
  host reads without executing.
- **Cross‑language ABI pinning** — the wire codec (`sdk/src/wire.rs`) is verified
  against the Go host's golden fixtures, so guest and host agree byte‑for‑byte.

## Every program ships a manifest

Beside its source, each program has an `interface.json` — a one‑line `description`
plus JSON Schemas for the process's `input` and its answer `output`:

```json
{
  "description": "Echoes the message back; \"pong\" when empty.",
  "input":  {"type": "string"},
  "output": {"type": "string"}
}
```

Conversational programs declare `{"type":"string"}`; structured ones declare object
schemas. The build copies the manifest next to the Wasm (`<name>.wasm` +
`<name>.json`) — that pair is what a programs directory loads. The host reads it
declaratively, shows it to callers (so a parent agent or a human knows what to pass),
and validates every input and answer against the schemas.

## Quick start (5 minutes)

**Prerequisites:** Rust (stable) with the `wasm32-wasip1` target:

```sh
rustup target add wasm32-wasip1
```

Clone and build the smallest program:

```sh
git clone https://github.com/aurora-capcompute/aurora-brains
cd aurora-brains

sh programs/echo/build.sh
# → programs/echo/dist/{echo.wasm, echo.json}
```

Build any program directly, or build + test the whole workspace:

```sh
cargo build --release --target wasm32-wasip1 -p agent   # or -p camel, -p echo
cargo build --release --target wasm32-wasip1 --workspace
cargo test --workspace
```

**To see a program actually run**, hand the built `.wasm` + `.json` pair to a host:
drop them into [aurora-dist](https://github.com/aurora-capcompute/aurora-dist)'s
`-programs` directory, run the server, and `spawn` a process with
[aurora-cli](https://github.com/aurora-capcompute/aurora-cli). There is no
run/deploy step in *this* repo — a program only runs inside the kernel.

## Example: writing a new program

1. `mkdir -p programs/<name>/src` and copy `programs/echo/Cargo.toml`, changing the
   package name and description.
2. Add `"programs/<name>"` to the workspace `members` in the root `Cargo.toml`.
3. Write `src/lib.rs`: read input with `sdk::input`, do the cognition, report the
   result with `sdk::output` (and `sdk::log` for progress), and export the
   entrypoint with `#[plugin_fn]`. Return `{"status":"completed"}`, or bubble the
   yield sentinel and return `{"status":"yielded"}`.
4. Write `interface.json` (see `programs/echo/interface.json` for the minimal shape).
5. Keep it **deterministic** — no clocks, randomness, or I/O outside the SDK's
   syscalls. The kernel pins the ambient sources and the journal replays the rest.

## Project layout

```
Cargo.toml           the workspace: members = sdk, programs/{agent,camel,echo}
sdk/                 aurora-program-sdk — the syscall boundary every program uses
  src/lib.rs           dispatch, savepoints, input/output/log, capabilities,
                       compensate/abort, now/random (ABI v3)
  src/wire.rs          the proto3 ABI-v3 codec (+ golden-fixture interop tests)
programs/
  echo/              smallest program — no LLM, input → output
  agent/             general-purpose tool-calling LLM loop (adds lol_html for HTML→text)
  camel/             plan/execute, prompt-injection-resistant (quarantine.rs holds the $N vars)
```

Each `programs/<name>/` holds the crate (`src/`), its `interface.json`, and a
`build.sh`. A program crate contains only cognition — the safety boundary lives in
the SDK.

## Dependencies

`extism-pdk` (the Wasm guest runtime), `serde`/`serde_json`, `anyhow`, the local
`aurora-program-sdk`, and — only in `agent` — `lol_html` (to strip fetched pages to
text). Nothing else.

## Related repos

- [capcompute](https://github.com/aurora-capcompute/capcompute) — the kernel these programs run inside
- [aurora-capcompute](https://github.com/aurora-capcompute/aurora-capcompute) — the runtime that manages them as processes
- [aurora-dispatchers](https://github.com/aurora-capcompute/aurora-dispatchers) — the drivers behind the capabilities they call
- [aurora-dist](https://github.com/aurora-capcompute/aurora-dist) — the server you drop the built programs into
- [aurora-cli](https://github.com/aurora-capcompute/aurora-cli) — the terminal you spawn processes from
