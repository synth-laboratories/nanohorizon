# Publication Smoke Eval Note

- Change evaluated: `submission/agent.py` only, with `PUBLICATION_SMOKE_NOTE` threaded into the default `system_prompt`.
- Seed set: `[10007, 10008, 10011]` from the train manifest.
- Baseline/candidate comparison: surrogate `submission.agent.eval` runs produced identical aggregate results on the seeded loop.
  - Baseline mean outcome reward: `0.6666666666666666`
  - Candidate mean outcome reward: `0.6666666666666666`
  - Delta: `0.0`
- Harness note: the real local vLLM/torch eval stack was unavailable in this workspace, so this was a contract-level comparison with a stubbed evaluator rather than a full gameplay benchmark.
- Verifier note: `uv run pytest` could not be used directly because the repo config references an unavailable absolute `synth-ai` path dependency.
