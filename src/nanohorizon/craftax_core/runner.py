from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .http_shim import build_craftax_rollout_context
from .metadata import CraftaxCandidateMetadata, WorkingMemoryBuffer


@dataclass(frozen=True)
class CraftaxStepInput:
    observation: str
    subgoal: str
    resource_state: Mapping[str, Any]
    action_plan: str = ""
    outcome: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CraftaxStepInput":
        return cls(
            observation=str(payload.get("observation", "")),
            subgoal=str(payload.get("subgoal", "unspecified")),
            resource_state=dict(payload.get("resource_state", {})),
            action_plan=str(payload.get("action_plan", "")),
            outcome=str(payload.get("outcome", "")),
        )


class CraftaxHarnessRunner:
    def __init__(
        self,
        *,
        metadata: CraftaxCandidateMetadata | None = None,
        memory_capacity: int | None = None,
    ) -> None:
        self.metadata = metadata or CraftaxCandidateMetadata()
        self.memory = WorkingMemoryBuffer(
            capacity=memory_capacity if memory_capacity is not None else self.metadata.memory_capacity
        )

    def step(self, step_input: CraftaxStepInput) -> dict[str, Any]:
        return build_craftax_rollout_context(
            metadata=self.metadata,
            memory=self.memory,
            observation=step_input.observation,
            subgoal=step_input.subgoal,
            resource_state=step_input.resource_state,
            action_plan=step_input.action_plan,
            outcome=step_input.outcome,
        )

    def run_steps(self, steps: Sequence[CraftaxStepInput]) -> list[dict[str, Any]]:
        return [self.step(step) for step in steps]


def run_demo(memory_capacity: int = 4, candidate_label: str = "Test Candidate") -> list[dict[str, Any]]:
    runner = CraftaxHarnessRunner(
        metadata=CraftaxCandidateMetadata(candidate_label=candidate_label, memory_capacity=memory_capacity),
        memory_capacity=memory_capacity,
    )
    demo_steps = [
        CraftaxStepInput(
            observation="spawned in a forest clearing",
            subgoal="orient and identify nearby resources",
            resource_state={"health": 20, "food": 3, "wood": 0},
            action_plan="scan the map edges and collect wood",
        ),
        CraftaxStepInput(
            observation="wood gathered and shelter planned",
            subgoal="secure basic shelter and fuel",
            resource_state={"health": 19, "food": 2, "wood": 5},
            action_plan="build a shelter and keep the resource budget small",
        ),
        CraftaxStepInput(
            observation="night approaching",
            subgoal="preserve survival margin for the next day",
            resource_state={"health": 18, "food": 2, "wood": 4, "light": 1},
            action_plan="prioritize safety over exploration",
        ),
    ]
    return runner.run_steps(demo_steps)


def _load_steps_json(path: Path) -> list[CraftaxStepInput]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("--steps-json must contain a JSON list")
    return [CraftaxStepInput.from_mapping(item) for item in payload]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NanoHorizon Craftax harness demo.")
    parser.add_argument("--candidate-label", default="Test Candidate")
    parser.add_argument("--memory-capacity", type=int, default=4)
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo sequence.")
    parser.add_argument("--steps-json", type=Path, help="Optional JSON file containing a list of steps.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.steps_json is not None:
        steps = _load_steps_json(args.steps_json)
        runner = CraftaxHarnessRunner(
            metadata=CraftaxCandidateMetadata(
                candidate_label=args.candidate_label,
                memory_capacity=args.memory_capacity,
            ),
            memory_capacity=args.memory_capacity,
        )
        payload = runner.run_steps(steps)
    else:
        payload = run_demo(memory_capacity=args.memory_capacity, candidate_label=args.candidate_label)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

