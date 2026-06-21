# aurora-brains

Versioned WebAssembly agent brains for Aurora.

Build the default TinyGo brain:

```sh
sh agent/build.sh
```

The guest ABI uses JSON input and returns either
`{"status":"completed","answer":"..."}` or `{"status":"yielded"}`.
