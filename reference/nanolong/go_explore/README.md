# NanoLong Go-Explore

`nanolong/go_explore` is now the canonical home for Go-Explore work.

This directory has two layers:

- the NanoLong-native Crafter-first local runner in the top-level package
- the migrated historical multi-environment stack under [`legacy/`](/Users/joshpurtell/Documents/GitHub/nanolong/go_explore/legacy)

The top-level package is the path to iterate on now:

- local config/result types
- filesystem-backed service lifecycle
- Crafter candidate search
- candidate-vs-baseline evaluation
- NanoLong-owned artifact writing

The legacy area preserves the older optimizer/plugin/archive/runtime code that
used to live outside NanoLong, including experimental containers, prompt
mutation logic, search/archive machinery, and old reference docs.

Use the NanoLong-native example first:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./examples/go_explore_crafter/run_smoke.sh
```

If you need the old experimental surfaces for reference, start in:

- [`legacy/README.md`](/Users/joshpurtell/Documents/GitHub/nanolong/go_explore/legacy/README.md)
- [`legacy/run_local_crafter.py`](/Users/joshpurtell/Documents/GitHub/nanolong/go_explore/legacy/run_local_crafter.py)
- [`legacy/optimizer.py`](/Users/joshpurtell/Documents/GitHub/nanolong/go_explore/legacy/optimizer.py)
