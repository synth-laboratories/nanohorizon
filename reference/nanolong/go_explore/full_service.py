from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .full_config import FullGoExploreConfig
from .full_models import (
    BranchDecision,
    CheckpointSummary,
    FullGoExploreResult,
    RolloutSummary,
    WaypointSpec,
)
from .full_runtime import FullCrafterRuntime


_MILESTONE_CHAIN: list[WaypointSpec] = [
    WaypointSpec(description="Collect wood", achievement="collect_wood"),
    WaypointSpec(description="Place a table", achievement="place_table"),
    WaypointSpec(description="Craft a wood pickaxe", achievement="make_wood_pickaxe"),
    WaypointSpec(description="Collect stone", achievement="collect_stone"),
    WaypointSpec(description="Craft a stone pickaxe", achievement="make_stone_pickaxe"),
    WaypointSpec(description="Collect coal", achievement="collect_coal"),
    WaypointSpec(description="Collect iron", achievement="collect_iron"),
    WaypointSpec(description="Place a furnace", achievement="place_furnace"),
]


def _labels_from_rollout(rollout: RolloutSummary) -> list[str]:
    labels = [str(item) for item in rollout.achievements]
    if rollout.reward > 0:
        labels.append("positive_reward")
    if rollout.completed_waypoints:
        labels.extend(
            str(item.get("description") or item.get("reason") or "completed_waypoint")
            for item in rollout.completed_waypoints
            if isinstance(item, dict)
        )
    return sorted({item for item in labels if item})


def _next_waypoints(checkpoint: CheckpointSummary) -> list[WaypointSpec]:
    unlocked = set(checkpoint.achievements)
    for index, waypoint in enumerate(_MILESTONE_CHAIN):
        if waypoint.achievement and waypoint.achievement not in unlocked:
            next_items = [waypoint]
            if index + 1 < len(_MILESTONE_CHAIN):
                next_items.append(_MILESTONE_CHAIN[index + 1])
            return next_items
    return []


def _checkpoint_score(checkpoint: CheckpointSummary) -> float:
    next_targets = _next_waypoints(checkpoint)
    return (
        checkpoint.total_reward
        + (2.0 * len(checkpoint.achievements))
        + (0.05 * checkpoint.step_index)
        + (1.5 * len(next_targets))
        - (0.75 * checkpoint.branch_visits)
    )


@dataclass(slots=True)
class FullCrafterGoExploreService:
    output_root: Path | None = None

    def __post_init__(self) -> None:
        if self.output_root is None:
            self.output_root = (
                Path(__file__).resolve().parents[1]
                / "examples"
                / "go_explore_crafter"
                / "artifacts"
                / "full"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(self, config: FullGoExploreConfig) -> FullGoExploreResult:
        runtime = FullCrafterRuntime(
            container_url=config.container_url,
            inference_url=config.inference_url,
            policy_model=config.policy_model,
            api_key=config.api_key(),
        )
        if not runtime.healthcheck():
            raise RuntimeError("Crafter runtime healthcheck failed")
        task_info = runtime.task_info()
        task_metadata = dict(task_info.get("task_metadata") or {})
        if not task_metadata.get("supports_restart_from_seed_checkpoint"):
            raise RuntimeError("Crafter runtime does not advertise checkpoint support")

        output_dir = config.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_root = output_dir / config.system_id
        run_root.mkdir(parents=True, exist_ok=True)

        rollouts: list[RolloutSummary] = []
        checkpoints: dict[str, CheckpointSummary] = {}
        waypoint_counts: Counter[str] = Counter()
        branch_decisions: list[BranchDecision] = []
        iterations: list[dict[str, object]] = []

        seed_cursor = 0
        for iteration_idx in range(config.max_iterations):
            iteration_log: dict[str, object] = {
                "iteration_idx": iteration_idx,
                "fresh_rollout_ids": [],
                "resumed_rollout_ids": [],
                "branch_decisions": [],
            }
            for fresh_idx in range(config.fresh_queries_per_iteration):
                seed = config.seed_ids[(seed_cursor + fresh_idx) % len(config.seed_ids)]
                rollout_id = f"{config.system_id}_iter{iteration_idx}_fresh{fresh_idx}_{uuid4().hex[:6]}"
                rollout = runtime.rollout_from_seed(
                    rollout_id=rollout_id,
                    prompt_text=config.prompt_text,
                    seed=seed,
                    segment_steps=config.segment_steps,
                )
                rollout.target_waypoints = []
                rollouts.append(rollout)
                waypoint_counts.update(_labels_from_rollout(rollout))
                checkpoint = runtime.create_checkpoint(
                    rollout_id=rollout.rollout_id,
                    seed=seed,
                    label=f"iter{iteration_idx}_fresh",
                    metadata={
                        "seed": seed,
                        "candidate_prompt": config.prompt_text,
                        "source_kind": "fresh",
                    },
                    source_kind="fresh",
                )
                checkpoint.labels = _labels_from_rollout(rollout)
                checkpoints[checkpoint.checkpoint_id] = checkpoint
                rollout.checkpoint_ids = [checkpoint.checkpoint_id]
                rollout.checkpoint_count = 1
                iteration_log["fresh_rollout_ids"].append(rollout.rollout_id)
            seed_cursor += config.fresh_queries_per_iteration

            ranked = sorted(
                checkpoints.values(),
                key=_checkpoint_score,
                reverse=True,
            )
            selected = ranked[: config.resumed_queries_per_iteration]
            for resume_idx, checkpoint in enumerate(selected):
                target_waypoints = _next_waypoints(checkpoint)
                if not target_waypoints:
                    continue
                checkpoint.branch_visits += 1
                decision = BranchDecision(
                    iteration_idx=iteration_idx,
                    checkpoint_id=checkpoint.checkpoint_id,
                    rollout_id=checkpoint.rollout_id,
                    seed=checkpoint.seed,
                    score=_checkpoint_score(checkpoint),
                    target_waypoints=target_waypoints,
                    reason="resume highest-value checkpoint toward next missing Crafter milestone",
                )
                branch_decisions.append(decision)
                iteration_log["branch_decisions"].append(decision.to_dict())

                rollout_id = f"{config.system_id}_iter{iteration_idx}_resume{resume_idx}_{uuid4().hex[:6]}"
                rollout = runtime.resume_from_checkpoint(
                    parent_rollout_id=checkpoint.rollout_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    target_rollout_id=rollout_id,
                    prompt_text=config.prompt_text,
                    seed=checkpoint.seed,
                    segment_steps=config.segment_steps,
                    planner_mode="waypoint_planned",
                    waypoints=target_waypoints,
                )
                rollouts.append(rollout)
                waypoint_counts.update(_labels_from_rollout(rollout))
                resumed_checkpoint = runtime.create_checkpoint(
                    rollout_id=rollout.rollout_id,
                    seed=checkpoint.seed,
                    label=f"iter{iteration_idx}_resume",
                    metadata={
                        "seed": checkpoint.seed,
                        "source_kind": "resumed",
                        "parent_checkpoint_id": checkpoint.checkpoint_id,
                    },
                    source_kind="resumed",
                )
                resumed_checkpoint.labels = _labels_from_rollout(rollout)
                checkpoints[resumed_checkpoint.checkpoint_id] = resumed_checkpoint
                rollout.checkpoint_ids = [resumed_checkpoint.checkpoint_id]
                rollout.checkpoint_count = 1
                rollout.target_waypoints = [item.description for item in target_waypoints]
                iteration_log["resumed_rollout_ids"].append(rollout.rollout_id)

            iterations.append(iteration_log)

        best_rollout = max(rollouts, key=lambda item: item.reward, default=None)
        artifact_path = run_root / "full_go_explore_result.json"
        result = FullGoExploreResult(
            system_id=config.system_id,
            prompt_text=config.prompt_text,
            policy_model=config.policy_model,
            iterations=iterations,
            rollouts=rollouts,
            checkpoints=list(checkpoints.values()),
            branch_decisions=branch_decisions,
            waypoint_counts=dict(waypoint_counts),
            best_rollout_id=best_rollout.rollout_id if best_rollout is not None else None,
            best_reward=best_rollout.reward if best_rollout is not None else 0.0,
            total_rollouts=len(rollouts),
            total_checkpoints=len(checkpoints),
            artifact_path=str(artifact_path),
            metadata={
                "container_url": config.container_url,
                "inference_url": config.inference_url,
                "seed_ids": list(config.seed_ids),
                "max_iterations": config.max_iterations,
                "segment_steps": config.segment_steps,
            },
        )
        artifact_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result
