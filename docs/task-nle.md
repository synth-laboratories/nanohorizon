# NetHack/NLE Task

NanoHorizon supports NetHack through the NetHack Learning Environment as a peer runtime family beside Craftax.

## Runtime

The NLE runtime lives under `src/nanohorizon/nle_core/` and exposes an HTTP shim at `nanohorizon.nle_core.http_shim`.

The shim mirrors the Craftax runtime shape:

- `/health`
- `/info`
- `/task_info`
- `/rollout`
- `/rollouts`

The default local port is `8913`, controlled by `NANOHORIZON_NLE_BIND_PORT`.

## Reward

The canonical reward is `scout_score`.

Scout score increases when the player observes new non-blank NetHack tiles. The runtime tracks observed tile counts separately for each `(dungeon_num, dungeon_level)` pair and only rewards positive increases.

Native NetHack score is diagnostic metadata only and is not the primary objective.

## Action Surface

The v1 NLE track exposes the full primitive NLE action catalog through the `nle_interact` tool:

```json
{"actions_list": ["..."]}
```

Blog-post-style augmented actions for menus, inventory item selection, tokenized menu observations, and role/race/alignment extraction are intentionally out of scope for this first runtime pass.

## Rendering

Text observations include the message line, selected `blstats`, and the terminal screen when available.

Pixel observations render the terminal state from `tty_chars` and `tty_colors`, then use the shared media persistence path for frames/GIF/MP4 artifacts.

## Smoke Config

Use `configs/nle_scout_smoke.yaml` as the initial smoke target.
