#!/bin/sh
# Build this program to its deployable artifacts: the wasm and its interface
# manifest (`<name>.wasm` + `<name>.json`, the pair a programs directory loads).
set -e
cd "$(dirname "$0")/../.."
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 -p camel
mkdir -p programs/camel/dist
cp target/wasm32-wasip1/release/camel.wasm programs/camel/dist/camel.wasm
cp programs/camel/interface.json programs/camel/dist/camel.json
echo "built programs/camel/dist/camel.{wasm,json}"
