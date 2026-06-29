#!/bin/sh
set -e
rustup target add wasm32-wasip1
cargo build --target wasm32-wasip1 --release
mkdir -p dist
cp target/wasm32-wasip1/release/aurora_agent.wasm dist/aurora-agent.wasm
