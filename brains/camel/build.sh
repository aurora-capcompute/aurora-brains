#!/bin/sh
# Build this brain to its deployable wasm artifact.
set -e
cd "$(dirname "$0")/../.."
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 -p camel-brain
mkdir -p brains/camel/dist
cp target/wasm32-wasip1/release/camel_brain.wasm brains/camel/dist/camel-brain.wasm
echo "built brains/camel/dist/camel-brain.wasm"
