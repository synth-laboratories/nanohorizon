from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Mapping

CANDIDATE_LABEL = "Server Push E2E"

TODO_SCRATCHPAD_REQUIREMENTS = (
    "Keep a private scratchpad with at most three live todo items.",
    "Push completed items as server-pushed state: remove them immediately.",
    "If progress stalls, replace the stalest item with the next best action.",
)


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    objective: str
    scratchpad_contract: str
    seed_prompt: str
    reflection_prompt: str
    scratchpad_requirements: tuple[str, str, str]


def build_scratchpad_contract() -> str:
    requirements = "\n".join(f"- {item}" for item in TODO_SCRATCHPAD_REQUIREMENTS)
    return dedent(
        f"""
        The scratchpad contract is:

        {requirements}
        """
    ).strip()


def refresh_todo_items(
    todo_items: tuple[str, ...] | list[str],
    completed_items: tuple[str, ...] | list[str] = (),
    next_action: str | None = None,
) -> tuple[str, ...]:
    """Refresh a three-item scratchpad by removing completed work first."""

    completed = {item.strip() for item in completed_items if item and item.strip()}
    live_items = [item for item in todo_items if item not in completed]
    if next_action:
        live_items = [item for item in live_items if item != next_action]
        live_items.append(next_action)
    return tuple(live_items[-3:])


def build_seed_prompt() -> str:
    return dedent(
        f"""
        You are optimizing Craftax with the `{CANDIDATE_LABEL}` strategy.

        Maintain a private three-item todo scratchpad and keep it server-pushed.
        Use the same scratchpad contract below every turn.

        {build_scratchpad_contract()}
        """
    ).strip()


def build_reflection_prompt() -> str:
    return dedent(
        f"""
        Reflect on the last turn using the same scratchpad contract.
        Before proposing the next action, refresh the scratchpad so it stays
        server-pushed and still obeys the contract below.

        {build_scratchpad_contract()}
        """
    ).strip()


def build_candidate_spec() -> CandidateSpec:
    scratchpad_contract = build_scratchpad_contract()
    return CandidateSpec(
        label=CANDIDATE_LABEL,
        objective=(
            "Improve the Craftax approach with a compact todo/scratchpad "
            "contract that stays fresh across turns."
        ),
        scratchpad_contract=scratchpad_contract,
        seed_prompt=build_seed_prompt(),
        reflection_prompt=build_reflection_prompt(),
        scratchpad_requirements=TODO_SCRATCHPAD_REQUIREMENTS,
    )


def candidate_record() -> dict[str, object]:
    spec = build_candidate_spec()
    return {
        "candidate_label": spec.label,
        "objective": spec.objective,
        "scratchpad_contract": spec.scratchpad_contract,
        "seed_prompt": spec.seed_prompt,
        "reflection_prompt": spec.reflection_prompt,
        "scratchpad_requirements": list(spec.scratchpad_requirements),
        "scratchpad_refresh_example": refresh_todo_items(
            (
                "Confirm the task constraints and keep the change narrow.",
                "Render a compact server-pushed todo scratchpad.",
                "Surface the scratchpad through task info and the runner.",
            ),
            completed_items=("Render a compact server-pushed todo scratchpad.",),
            next_action="Validate the output with a local smoke verifier.",
        ),
    }


def validate_candidate_record(record: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if record.get("candidate_label") != CANDIDATE_LABEL:
        errors.append("candidate_label mismatch")
    requirements = record.get("scratchpad_requirements")
    if not isinstance(requirements, list) or len(requirements) != 3:
        errors.append("scratchpad_requirements must contain exactly three items")
    else:
        requirement_text = " ".join(str(item) for item in requirements)
        for token in ("three", "server-pushed", "stalest"):
            if token not in requirement_text:
                errors.append(f"scratchpad_requirements missing '{token}'")
                break
    refresh_example = record.get("scratchpad_refresh_example")
    if not isinstance(refresh_example, (list, tuple)) or len(refresh_example) > 3:
        errors.append("scratchpad_refresh_example must stay at or below three items")
    return errors

