# Crafter RS Container

Prototype long-horizon Crafter container for `go_explore` testing.

This wraps the in-repo `crafter-rs` runtime and aims to match the current Crafter long-horizon container surface closely enough for checkpointed rollout testing.

Current scope:
- `GET /`
- `GET /health`
- `GET /done`
- `GET /info`
- `GET /task_info`
- `POST /rollout`
- `POST /rollouts`
- `GET /rollouts/{rollout_id}`
- `GET/POST /rollouts/{rollout_id}/checkpoints`
- `GET /rollouts/{rollout_id}/checkpoints/{checkpoint_id}`
- `POST /rollouts/{rollout_id}/resume`
- `POST /rollouts/{rollout_id}/checkpoint/dump`
- `POST /rollouts/{rollout_id}/checkpoint/restore`
- `POST /rollouts/{rollout_id}/terminate`
- legacy checkpoint aliases

Run locally:

```bash
cd runtime/crafter_rs
cargo run
```

Notes:
- Rollouts are synchronous in this first pass, so terminate is useful as a control surface and state marker but not yet for mid-request interruption.
- Checkpoints are stored in-memory as serialized `crafter-rs` `SaveData`.
