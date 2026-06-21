#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

: "${GOCACHE:=/tmp/aurora-capcompute-go-build}"
: "${XDG_CACHE_HOME:=/tmp/aurora-capcompute-tinygo-cache}"
export GOCACHE XDG_CACHE_HOME

tinygo build \
	-target wasip1 \
	-buildmode=c-shared \
	-tags tinygo \
	-o agent/agent.wasm \
	./agent
