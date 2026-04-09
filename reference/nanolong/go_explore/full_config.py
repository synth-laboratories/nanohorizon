from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_api_key(env_name: str) -> str:
    value = os.getenv(env_name, "").strip()
    if value:
        return value
    env_path = _repo_root().parent / "synth-ai" / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() == env_name:
                candidate = raw_value.strip().strip("'").strip('"')
                if candidate:
                    return candidate
    raise RuntimeError(f"{env_name} is required")


@dataclass(slots=True)
class FullGoExploreConfig:
    system_id: str = field(default_factory=lambda: f"full_go_explore_{uuid4().hex[:10]}")
    container_url: str = "http://127.0.0.1:8903"
    inference_url: str = "https://openrouter.ai/api/v1/chat/completions"
    policy_model: str = "openai/gpt-4.1-mini"
    api_key_env: str = "OPENROUTER_API_KEY"
    prompt_text: str = (
        "You are a Crafter agent. You receive a text state with a 9x9 local map "
        "and inventory/health summaries. Choose 2-5 actions to make progress toward "
        "crafting and survival. Use do whenever adjacent to a resource (tree, stone, "
        "cow, plant). Craft progression: wood -> table -> wood_pickaxe -> stone -> "
        "stone_pickaxe -> iron tools. If the observation includes current_waypoint, "
        "treat that waypoint as the top priority until it is completed. Avoid noop unless stuck."
    )
    seed_ids: list[int] = field(default_factory=lambda: [11, 29])
    max_iterations: int = 3
    fresh_queries_per_iteration: int = 2
    resumed_queries_per_iteration: int = 2
    segment_steps: int = 64
    output_dir: Path = field(
        default_factory=lambda: _repo_root()
        / "examples"
        / "go_explore_crafter"
        / "artifacts"
        / "full"
    )

    def api_key(self) -> str:
        return load_api_key(self.api_key_env)
