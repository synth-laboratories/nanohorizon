## Summary

This checked-in RLVR baseline is the first completed clustered Modal smoke run for the Craftax track.

- topology: one public Craftax web service plus one clustered learner-plus-inference runtime
- model: `Qwen/Qwen3.5-4B`
- budget config used in this validation run: `8` minutes
- result: final mean outcome reward `0.0`

## What This Record Proves

- Craftax transport is working through the public Synth-compatible service
- clustered learner and inference ownership matches the NanoLong-style runtime boundary
- LoRA adapter publish and reload worked for `iter_000`
- periodic eval and final eval both completed with structured rollouts

## What It Does Not Prove

- meaningful reward lift
- a scored 20-minute canonical track result

The longer 10-step / multi-checkpoint experiment was started separately after this reference smoke run.
