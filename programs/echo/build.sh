#!/bin/sh
# Build this program to its deployable artifacts: the wasm and its interface
# manifest (`<name>.wasm` + `<name>.json`, the pair a programs directory loads).
set -e
cd "$(dirname "$0")/../.."
rustup target add wasm32-wasip1
cargo build --release --target wasm32-wasip1 -p echo
mkdir -p programs/echo/dist
cp target/wasm32-wasip1/release/echo.wasm programs/echo/dist/echo.wasm
cp programs/echo/interface.json programs/echo/dist/echo.json
echo "built programs/echo/dist/echo.{wasm,json}"
