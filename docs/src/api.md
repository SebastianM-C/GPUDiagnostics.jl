# API reference

```@index
```

```@autodocs
Modules = [GPUDiagnostics]
Filter = t -> !(t isa Function && startswith(String(nameof(t)), "backend_"))
```

The `backend_*` hooks are documented on the [Porting a backend](porting.md) page.
