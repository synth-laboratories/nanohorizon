# Modal 4B No-Change Baseline

This record captures the pure `Qwen/Qwen3.5-4B` baseline with no training or adapter changes.

## Setup

- Crafter service exposed through the Synth tunnel path
- inference served from Modal via `src/nanohorizon/shared/modal_teacher.py`
- held-out eval on `20` seeds starting at `10000`
- rollout cap of `10` steps

## Measured result

- mean reward over requested rollouts: `0.7`
- raw rewards are recorded in `metrics.json`
- only non-zero achievement frequencies were:
  - `collect_wood`: `8 / 20 = 0.4`
  - `collect_sapling`: `6 / 20 = 0.3`

## Caveat

The current remote-eval path still reports `llm_call_count: 0` for each rollout even though inference happened successfully through Modal. This record should be treated as the pure policy outcome baseline, not as a trustworthy per-rollout call-count audit.
