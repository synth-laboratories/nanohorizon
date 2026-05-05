# Open Research Craftax Workspace

This directory contains the NanoHorizon-owned files that SMR/Open Research can
run after cloning the public repo.

## Runner Contract

Default Open Research command:

```bash
python3 workspace/run_nanohorizon_craftax_hello_world_task.py run --output-root .
python3 workspace/run_nanohorizon_craftax_hello_world_task.py score --output-root . --verifier-mode precheck
```

The runner writes:

- `artifacts/eval_summary.json`
- `artifacts/rollouts.jsonl`
- `artifacts/result_manifest.json`
- `artifacts/container_proof.json`
- `artifacts/verifier_review.json`
- `artifacts/reportbench_output.json`
- `artifacts/craftax_scorecard.json`
- `artifacts/craftax_rollout_media.json`
- `artifacts/craftax_experiment_result.json`
- `reports/reproduction.md`

## Required Runtime Choices

The actor must provide a Craftax resource before running the worker:

- local HTTP: set `NANOHORIZON_CRAFTAX_CONTAINER_URL=http://...`
- same-actor local proof: set `NANOHORIZON_ALLOW_DIRECT_LOCAL=1`
- Synth container pool: create/read back a pool with
  `workspace/craftax_runtime_resource.py`, then set the resolved URL for the
  child command

SMR should materialize provider access for the child command. Workers should not
fetch secrets themselves.

Do not preflight the public demo by calling NanoHorizon internals such as
`_chat_completion` with `request_logprobs=True`, or by sending OpenRouter/xAI
`logprobs`/`include` request shapes. The checked-in runner is the Open Research
contract and handles the supported request shape for this path.

For the public Open Research demo, use:

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
NANOHORIZON_MODEL=x-ai/grok-4.1-fast
NANOHORIZON_ROLLOUTS=10
NANOHORIZON_ROLLOUT_CONCURRENCY=10
```

The Open Research leaderboard contract is experiment-first. The result metadata
must include ten `rollout_details[]` entries with reward, seed, achievements,
and visible per-rollout media. The checked-in runner writes GIF data URLs into
`artifacts/craftax_experiment_result.json` and indexes those same media refs in
`artifacts/craftax_rollout_media.json` so SMR can publish them without asking
the worker actor to invent a separate schema.
