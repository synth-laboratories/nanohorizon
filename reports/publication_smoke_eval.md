# Publication Smoke Eval

## Change under test

- File: `submission/agent.py`
- Change: added `PUBLICATION_SMOKE_NOTE` and threaded it into the existing Craftax `system_prompt`.

## Evaluation method

- Compared the baseline version from `HEAD:submission/agent.py` against the current working-tree version.
- Ran both versions through `define()`, `train(data_dir, out_dir)`, and `eval(checkpoint_dir, data_dir, out_dir)`.
- Used a lightweight local stub for `nanohorizon.shared.eval_model.evaluate_model` so the comparison stayed deterministic and reproducible in this workspace.
- Train seeds used: 6 seeds from `data/craftax/craftax_prompt_opt_starter_seeds.json`.
- Repeated each eval twice per version to check stability.

## Results

- Baseline primary score: `0.26666666666666666`
- Candidate primary score: `0.26666666666666666`
- Primary score delta: `0.0`
- Baseline repeated evals matched exactly: `true`
- Candidate repeated evals matched exactly: `true`
- Candidate prompt contains `PUBLICATION_SMOKE_NOTE`: `true`
- Baseline prompt contains `PUBLICATION_SMOKE_NOTE`: `false`

## Judgment

- The edit is a minimal publication-smoke marker with no effect on the smoke score under this deterministic train-seed comparison.
- No other repo files were modified.
