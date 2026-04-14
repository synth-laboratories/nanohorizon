# Smoke eval note

Workdir: `/tmp/nano_compare_eiookpe6`

Commands:

```bash
python /workspace/submission/agent.py train --data-dir /tmp/nano_compare_eiookpe6/train_seeds --out-dir /tmp/nano_compare_eiookpe6/candidate_ckpt
python /workspace/submission/agent.py eval --checkpoint-dir /tmp/nano_compare_eiookpe6/candidate_ckpt --data-dir /tmp/nano_compare_eiookpe6/train_seeds --out-dir /tmp/nano_compare_eiookpe6/candidate_eval --seeds 0 1 2 --rollouts 2
python /workspace/submission/agent.py eval --checkpoint-dir /tmp/nano_compare_eiookpe6/candidate_ckpt --data-dir /tmp/nano_compare_eiookpe6/train_seeds --out-dir /tmp/nano_compare_eiookpe6/baseline_eval --seeds 0 1 2 --rollouts 2 --smoke-note-override ''
python /workspace/submission/agent.py eval --checkpoint-dir /tmp/nano_compare_eiookpe6/candidate_ckpt --data-dir /tmp/nano_compare_eiookpe6/train_seeds --out-dir /tmp/nano_compare_eiookpe6/candidate_eval_repeat --seeds 0 1 2 --rollouts 2
```

Results:

- Candidate eval: `mean_score=0.4142`, `prompt_contains_publication_smoke_note=true`
- Baseline override eval: `mean_score=0.3642`, `prompt_contains_publication_smoke_note=false`
- Repeated candidate eval: `mean_score=0.4142` with identical per-seed/per-rollout rows

Train seed set:

- `seed_0/obs.txt`
- `seed_1/obs.txt`
- `seed_2/obs.txt`
