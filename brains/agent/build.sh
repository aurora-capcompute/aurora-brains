#!/bin/sh
# Build this brain to its deployable wasm artifact.
set -e
cd "$(dirname "$0")/../.."
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 -p agent-brain
mkdir -p brains/agent/dist
cp target/wasm32-wasip1/release/agent_brain.wasm brains/agent/dist/agent-brain.wasm
echo "built brains/agent/dist/agent-brain.wasm"
