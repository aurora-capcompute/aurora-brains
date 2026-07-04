#!/bin/sh
# Build this brain to its deployable wasm artifact.
set -e
cd "$(dirname "$0")/../.."
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 -p echo-brain
mkdir -p brains/echo/dist
cp target/wasm32-wasip1/release/echo_brain.wasm brains/echo/dist/echo-brain.wasm
echo "built brains/echo/dist/echo-brain.wasm"
