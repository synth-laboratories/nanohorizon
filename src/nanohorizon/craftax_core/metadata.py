"""Stable metadata for the Craftax prompt-opt candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from nanohorizon.baselines.prompt_opt import FULL_AUTO_E2E_SYSTEM_PROMPT, todo_scratchpad_directive

OBSERVATION_FIELD_ORDER = (
    "health",
    "food",
    "energy",
    "nearby_entities",
    "inventory",
)

PRIMARY_TOOL_NAME = "craftax_interact"

CANDIDATE_LABEL = "Full Auto E2E"
PRIMARY_STRATEGY = "Todo Tool"
TRACK_NAME = "prompt_opt_1usd_gpt54_family"
TASK_NAME = "craftax"
OUTPUT_ROOT = "records/prompt_opt_1usd_gpt54_family/2026-04-11_full_auto_e2e"
TODO_ITEMS = (
    "State the objective and verifier target in one line.",
    "Keep the change set as small as possible.",
    "Record the evidence before expanding scope.",
    "Check the candidate against the preserved harness surfaces.",
    "Stop only after the verifier pass is documented.",
)
PRESERVED_SURFACES = (
    "docs/task-craftax.md",
    "src/nanohorizon/craftax_core/http_shim.py",
    "src/nanohorizon/craftax_core/runner.py",
    "src/nanohorizon/craftax_core/metadata.py",
    "scripts/run_craftax_model_eval.sh",
)

DEFAULT_ACTION_NAMES = (
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "rest",
    "descend",
    "ascend",
    "make_diamond_pickaxe",
    "make_diamond_sword",
    "make_iron_armour",
    "make_diamond_armour",
    "shoot_arrow",
    "make_arrow",
    "cast_fireball",
    "cast_iceball",
    "place_torch",
    "drink_potion_red",
    "drink_potion_green",
    "drink_potion_blue",
    "drink_potion_pink",
    "drink_potion_cyan",
    "drink_potion_yellow",
    "read_book",
    "enchant_sword",
    "enchant_armour",
    "make_torch",
    "level_up_dexterity",
    "level_up_strength",
    "level_up_intelligence",
    "enchant_bow",
)

DEFAULT_ACHIEVEMENT_NAMES = (
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
)

REWARD_WINDOW_SIZE = 5


def _shorten_text(value: Any, limit: int = 140) -> str:
    text = repr(value) if not isinstance(value, str) else value
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def build_candidate_manifest() -> dict[str, Any]:
    return {
        "label": CANDIDATE_LABEL,
        "strategy": PRIMARY_STRATEGY,
        "todo_items": list(TODO_ITEMS),
        "preserved_surfaces": PRESERVED_SURFACES,
        "verification_modes": ["config_roundtrip_smoke", "prompt_render_smoke"],
        "track_name": TRACK_NAME,
        "task_name": TASK_NAME,
        "output_root": OUTPUT_ROOT,
        "rationale": (
            "The prompt-opt baseline centralizes the Todo Tool contract, keeps the "
            "candidate packaging explicit, and preserves the existing Craftax "
            "harness surfaces."
        ),
    }


def build_candidate_prompt() -> str:
    return (
        f"{FULL_AUTO_E2E_SYSTEM_PROMPT}\n\n"
        "Todo contract:\n"
        f"{todo_scratchpad_directive()}"
    )


@dataclass(slots=True)
class StructuredObservation:
    health: Any = None
    food: Any = None
    energy: Any = None
    nearby_entities: Any = None
    inventory: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_observation(cls, observation: Any) -> "StructuredObservation":
        if isinstance(observation, Mapping):
            known = {field: observation.get(field) for field in OBSERVATION_FIELD_ORDER}
            extras = {
                key: value
                for key, value in observation.items()
                if key not in OBSERVATION_FIELD_ORDER
            }
            return cls(**known, extras=extras)
        if hasattr(observation, "tolist") and not isinstance(observation, (str, bytes, bytearray)):
            try:
                observation = observation.tolist()
            except Exception:
                pass
        if isinstance(observation, Sequence) and not isinstance(observation, (str, bytes, bytearray)):
            values = list(observation)
            known = {
                field: values[idx] if idx < len(values) else None
                for idx, field in enumerate(OBSERVATION_FIELD_ORDER)
            }
            extras = {
                "remaining_features": {
                    f"feature_{idx}": value
                    for idx, value in enumerate(
                        values[len(OBSERVATION_FIELD_ORDER) :],
                        start=len(OBSERVATION_FIELD_ORDER),
                    )
                }
            }
            return cls(**known, extras=extras)
        return cls(extras={"raw_observation": observation})

    def to_prompt_payload(self) -> dict[str, Any]:
        payload = {field: getattr(self, field) for field in OBSERVATION_FIELD_ORDER}
        payload["extras"] = self.extras
        return payload

    def brief_summary(self) -> str:
        parts = []
        for field in OBSERVATION_FIELD_ORDER:
            value = getattr(self, field)
            if value is not None:
                parts.append(f"{field}={_shorten_text(value)}")
        if self.extras:
            parts.append(f"extras={_shorten_text(self.extras)}")
        return ", ".join(parts) if parts else "no structured observation available"


@dataclass(slots=True)
class RewardHistoryEntry:
    action: Any
    observation_summary: str
    reward_delta: float

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "observation_summary": self.observation_summary,
            "reward_delta": self.reward_delta,
        }


@dataclass(slots=True)
class RewardHistoryWindow:
    entries: list[RewardHistoryEntry] = field(default_factory=list)

    def append(self, entry: RewardHistoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > REWARD_WINDOW_SIZE:
            self.entries = self.entries[-REWARD_WINDOW_SIZE:]

    def to_prompt_payload(self) -> list[dict[str, Any]]:
        return [entry.to_prompt_payload() for entry in self.entries]


@dataclass(slots=True)
class PromptContext:
    observation: StructuredObservation
    reward_history: RewardHistoryWindow = field(default_factory=RewardHistoryWindow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "structured_observation": self.observation.to_prompt_payload(),
            "structured_observation_summary": self.observation.brief_summary(),
            "reward_history": self.reward_history.to_prompt_payload(),
            "metadata": dict(self.metadata),
        }

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)
