#!/bin/sh
# Build this program to its deployable artifacts: the wasm and its interface
# manifest (`<name>.wasm` + `<name>.json`, the pair a programs directory loads).
set -e
cd "$(dirname "$0")/../.."
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 -p agent
mkdir -p programs/agent/dist
cp target/wasm32-wasip1/release/agent.wasm programs/agent/dist/agent.wasm
cp programs/agent/interface.json programs/agent/dist/agent.json
echo "built programs/agent/dist/agent.{wasm,json}"
