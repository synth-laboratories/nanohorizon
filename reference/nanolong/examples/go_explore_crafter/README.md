# Go-Explore Crafter

This example runs the NanoLong-local Go-Explore Crafter path.

Current scope:

- Crafter only
- local artifact storage only
- no Temporal or backend job plumbing
- reuses the existing local mock-policy and `crafter-rs` container flow

Commands:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./examples/go_explore_crafter/run_smoke.sh
./examples/go_explore_crafter/run_prototype.sh
./examples/go_explore_crafter/run_crafter.sh
./.venv/bin/python ./examples/go_explore_crafter/run_real_go_explore.py
./.venv/bin/python -m go_explore.run_full_go_explore
```

Artifacts land under:

- `examples/go_explore_crafter/artifacts/smoke/`
- `examples/go_explore_crafter/artifacts/prototype/`
- `examples/go_explore_crafter/artifacts/crafter/`

Main outputs:

- `status.json`
- `result.json`
- `artifacts/results/summary.txt`
- `artifacts/results/result.json`
- `artifacts/real/<system_id>/summary.json`
- `artifacts/full/<system_id>/full_go_explore_result.json`

This is a research-first local entrypoint. It is meant for iterating on the
Go-Explore algorithm shape in NanoLong before upstreaming a hardened version
back into the production backend.

`run_prototype.sh` is the most reliable local prototype today. It talks
directly to the native Crafter container and uses a deterministic local
`policy_config.action_cycle` rather than the mock inference server, so it is
useful for validating the NanoLong Go-Explore artifact flow and Crafter wiring
without depending on prompt parsing.

`run_real_go_explore.py` is the real Crafter acceptance path. It uses the
NanoLong-owned local RLM archive reasoner, scanner, verifier, checkpoints,
branch cohorts, and in-loop prompt mutation from `go_explore/legacy`, then
evaluates the optimizer-selected winning prompt on a fixed holdout seed set.
The output prints waypoint counts, checkpoint frontier, branch cohorts,
trajectory frontier, and the baseline-vs-winner holdout comparison.

Acceptance expectations for the default command:

- resumed checkpoint branches execute
- the winning prompt is a mutated candidate, not the baseline
- branch cohorts are non-empty
- holdout mean uplift is at least `+2.0`

For a faster local-dev loop, use a lighter budget and the nano model:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./.venv/bin/python ./examples/go_explore_crafter/run_real_go_explore.py \
  --model openai/gpt-4.1-nano \
  --iterations 1 \
  --fresh-queries 1 \
  --resumed-queries 1 \
  --local-trials-per-start-state 1 \
  --segment-steps 32 \
  --max-mutations 1 \
  --frontier-size 2 \
  --required-uplift 0
```

That local-dev profile is meant to validate the real runner and produce a real
artifact quickly. It is not the final acceptance bar.

Artifacts for the acceptance run include:

- `config.json`
- `optimizer_result.json`
- `summary.json`

`python -m go_explore.run_full_go_explore` is the full local checkpointing lane.
It runs fresh seeded rollouts, saves checkpoints, branches from the best
checkpoint states, switches resumed branches into `waypoint_planned` mode, and
writes the resulting rollout/checkpoint archive under `artifacts/full/`.
