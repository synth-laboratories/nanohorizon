from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Sequence
from uuid import uuid4

from .legacy.config import GoExploreConfig
from .legacy.models import GoExploreResult, PromptCandidate, SearchQuery, StartStateRef, TaskSeedRef
from .legacy.optimizer import GoExploreOptimizer
from .legacy.runtime import ContainerSearchRuntime
from .legacy.runtime.crafter_client import CrafterRuntimeRequestError


DEFAULT_ACCEPTANCE_OUTPUT_DIR = Path(
    "/Users/joshpurtell/Documents/GitHub/nanolong/examples/go_explore_crafter/artifacts/real"
)
DEFAULT_BASELINE_PROMPT = (
    "Explore cautiously. Keep moving, avoid wasting actions, and gather whatever seems nearby."
)
DEFAULT_EXPECTED_WAYPOINTS = (
    "collect_wood",
    "collect_sapling",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
)
DEFAULT_TRAINING_SEEDS = (47, 53)
DEFAULT_HOLDOUT_SEEDS = (11, 29, 41, 43)
DEFAULT_REQUIRED_UPLIFT = 2.0
REAL_CRAFTER_PHASES = (
    "optimizer",
    "baseline_holdout",
    "winner_holdout",
    "summary",
)
RealCrafterPhaseName = Literal[
    "optimizer",
    "baseline_holdout",
    "winner_holdout",
    "summary",
]
RealCrafterCliPhase = Literal[
    "full",
    "optimizer-only",
    "baseline-holdout-only",
    "winner-holdout-only",
]


@dataclass(frozen=True)
class HoldoutEvaluation:
    candidate_id: str
    prompt_text: str
    mean_reward: float
    max_reward: float
    episodes: list[dict[str, Any]]
    achievement_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "prompt_text": self.prompt_text,
            "mean_reward": self.mean_reward,
            "max_reward": self.max_reward,
            "episodes": list(self.episodes),
            "achievement_counts": dict(self.achievement_counts),
        }


@dataclass(frozen=True)
class AcceptanceVerdict:
    accepted: bool
    required_uplift: float
    actual_uplift: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "required_uplift": self.required_uplift,
            "actual_uplift": self.actual_uplift,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RealCrafterExperimentResult:
    system_id: str
    artifact_dir: Path
    config: GoExploreConfig
    optimizer_result: GoExploreResult
    baseline_holdout: HoldoutEvaluation
    winning_holdout: HoldoutEvaluation
    checkpoint_frontier: list[dict[str, Any]]
    branch_cohorts: list[dict[str, Any]]
    trajectory_frontier: list[dict[str, Any]]
    waypoint_summary: dict[str, Any]
    prompt_trace_overview: dict[str, Any]
    acceptance: AcceptanceVerdict

    def to_dict(self) -> dict[str, Any]:
        best_candidate = self.optimizer_result.best_candidate
        return {
            "system_id": self.system_id,
            "artifact_dir": str(self.artifact_dir),
            "config": _sanitize_config_payload(self.config.to_dict()),
            "best_candidate_id": self.optimizer_result.best_candidate_id,
            "best_prompt": best_candidate.prompt_text if best_candidate is not None else None,
            "archive_summary": self.optimizer_result.archive_summary.to_dict(),
            "baseline_holdout": self.baseline_holdout.to_dict(),
            "winning_holdout": self.winning_holdout.to_dict(),
            "mean_uplift": self.winning_holdout.mean_reward - self.baseline_holdout.mean_reward,
            "checkpoint_frontier": list(self.checkpoint_frontier),
            "branch_cohorts": list(self.branch_cohorts),
            "trajectory_frontier": list(self.trajectory_frontier),
            "waypoint_summary": dict(self.waypoint_summary),
            "prompt_trace_overview": dict(self.prompt_trace_overview),
            "prompt_trace_path": str(self.artifact_dir / "prompt_trace.json"),
            "acceptance": self.acceptance.to_dict(),
            "optimizer_result": self.optimizer_result.to_dict(),
        }


@dataclass
class PhaseStatusRecord:
    status: str = "pending"
    failure: dict[str, Any] | None = None
    completed_seeds: list[int] = field(default_factory=list)
    pending_seeds: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure": dict(self.failure) if self.failure is not None else None,
            "completed_seeds": list(self.completed_seeds),
            "pending_seeds": list(self.pending_seeds),
        }


@dataclass
class RunStateSnapshot:
    system_id: str
    artifact_dir: Path
    current_phase: str
    selected_best_candidate_id: str | None = None
    phases: dict[str, PhaseStatusRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "artifact_dir": str(self.artifact_dir),
            "current_phase": self.current_phase,
            "selected_best_candidate_id": self.selected_best_candidate_id,
            "phases": {
                name: record.to_dict()
                for name, record in self.phases.items()
            },
        }


@dataclass
class HoldoutPhaseArtifact:
    phase_name: RealCrafterPhaseName
    candidate_id: str
    prompt_text: str
    phase_status: PhaseStatusRecord
    episodes: list[dict[str, Any]] = field(default_factory=list)
    achievement_counts: dict[str, int] = field(default_factory=dict)
    mean_reward: float | None = None
    max_reward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "candidate_id": self.candidate_id,
            "prompt_text": self.prompt_text,
            "phase_status": self.phase_status.to_dict(),
            "episodes": list(self.episodes),
            "achievement_counts": dict(self.achievement_counts),
            "mean_reward": self.mean_reward,
            "max_reward": self.max_reward,
        }

    def to_holdout_evaluation(self) -> HoldoutEvaluation:
        if self.phase_status.status != "completed":
            raise RuntimeError(
                f"holdout phase {self.phase_name} is not complete: {self.phase_status.status}"
            )
        return HoldoutEvaluation(
            candidate_id=self.candidate_id,
            prompt_text=self.prompt_text,
            mean_reward=float(self.mean_reward or 0.0),
            max_reward=float(self.max_reward or 0.0),
            episodes=list(self.episodes),
            achievement_counts=dict(self.achievement_counts),
        )


@dataclass(frozen=True)
class RealCrafterPhaseResult:
    system_id: str
    artifact_dir: Path
    phase: RealCrafterCliPhase
    run_state: RunStateSnapshot
    optimizer_result: GoExploreResult | None = None
    holdout_artifact: HoldoutPhaseArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "artifact_dir": str(self.artifact_dir),
            "phase": self.phase,
            "run_state": self.run_state.to_dict(),
            "optimizer_best_candidate_id": (
                self.optimizer_result.best_candidate_id
                if self.optimizer_result is not None
                else None
            ),
            "holdout_artifact": (
                self.holdout_artifact.to_dict()
                if self.holdout_artifact is not None
                else None
            ),
        }


def load_api_key(env_name: str) -> str:
    value = os.getenv(env_name, "").strip()
    if value:
        return value
    env_path = Path("/Users/joshpurtell/Documents/GitHub/synth-ai/.env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() == env_name:
                value = raw_value.strip().strip("'").strip('"')
                if value:
                    return value
    raise RuntimeError(f"{env_name} is required")


def build_real_crafter_config(
    *,
    api_key: str,
    system_id: str | None = None,
    container_url: str = "http://127.0.0.1:8903",
    inference_url: str = "https://openrouter.ai/api/v1/chat/completions",
    model: str = "openai/gpt-4.1-mini",
    baseline_prompt: str = DEFAULT_BASELINE_PROMPT,
    training_seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    iterations: int = 3,
    fresh_queries: int = 3,
    resumed_queries: int = 2,
    local_trials_per_start_state: int = 2,
    segment_steps: int = 64,
    frontier_size: int = 3,
    max_mutations: int = 2,
    candidate_exploration_slots: int = 1,
    reasoning_effort: str = "low",
    prompt_mutation_tool_profile: str = "compact",
) -> GoExploreConfig:
    resolved_system_id = system_id or f"real_goexp_{uuid4().hex[:10]}"
    payload = {
        "system_id": resolved_system_id,
        "runtime": {
            "base_url": container_url,
            "env_kind": "crafter",
            "runtime_kind": "container_http",
            "rollout_execution_mode": "request_response",
            "planner_mode": "direct",
            "model": model,
            "inference_url": inference_url,
            "inference_api_key": api_key,
            "timeout_seconds": 240.0,
            "env_config": {
                "segment_steps": segment_steps,
                "episode_max_steps": segment_steps,
                "max_steps": segment_steps,
            },
            "policy_config": {
                "temperature": 0.0,
                "max_tokens": 128,
            },
        },
        "seed_pool": [{"seed_id": str(seed)} for seed in training_seeds],
        "initial_candidates": [
            {
                "candidate_id": "baseline",
                "prompt_text": baseline_prompt,
                "generation": 0,
            }
        ],
        "budget": {
            "max_iterations": iterations,
            "fresh_queries_per_iteration": fresh_queries,
            "resumed_queries_per_iteration": resumed_queries,
            "local_trials_per_start_state": local_trials_per_start_state,
            "segment_steps": segment_steps,
            "trajectory_frontier_size": 8,
        },
        "scanner": {
            "model": model,
            "inference_url": inference_url,
            "inference_api_key": api_key,
            "reasoning_effort": reasoning_effort,
            "expected_waypoints": list(DEFAULT_EXPECTED_WAYPOINTS),
        },
        "prompt_plugin": {
            "frontier_size": frontier_size,
            "max_mutations_per_iteration": max_mutations,
            "candidate_exploration_slots": candidate_exploration_slots,
        },
        "proposer": {
            "model": model,
            "inference_url": inference_url,
            "inference_api_key": api_key,
            "reasoning_effort": reasoning_effort,
            "temperature": 0.2,
            "timeout_seconds": 240.0,
            "max_tool_round_trips": 10,
            "prompt_token_budget": 32_000,
            "prompt_mutation_tool_profile": prompt_mutation_tool_profile,
        },
    }
    return GoExploreConfig.from_dict(payload)


async def run_real_crafter_experiment(
    config: GoExploreConfig,
    *,
    holdout_seeds: Sequence[int] = DEFAULT_HOLDOUT_SEEDS,
    output_dir: Path | str = DEFAULT_ACCEPTANCE_OUTPUT_DIR,
    required_uplift: float = DEFAULT_REQUIRED_UPLIFT,
    holdout_segment_steps: int | None = None,
    runtime: ContainerSearchRuntime | None = None,
) -> RealCrafterExperimentResult:
    runtime_adapter = runtime or ContainerSearchRuntime(config.crafter)
    optimizer_phase = await run_real_crafter_optimizer_phase(
        config,
        output_dir=output_dir,
        runtime=runtime_adapter,
    )
    optimizer_result = optimizer_phase.optimizer_result
    if optimizer_result is None:
        raise RuntimeError("optimizer phase did not produce an optimizer_result")
    baseline_candidate = _baseline_candidate_from_config(config)
    best_candidate = optimizer_result.best_candidate
    if best_candidate is None:
        raise RuntimeError("optimizer result is missing a best candidate")
    baseline_phase = await _run_holdout_phase(
        artifact_root=optimizer_phase.artifact_dir,
        config=config,
        runtime=runtime_adapter,
        run_state=optimizer_phase.run_state,
        phase_name="baseline_holdout",
        candidate=PromptCandidate(
            candidate_id=f"{baseline_candidate.candidate_id}_holdout",
            prompt_text=baseline_candidate.prompt_text,
            generation=baseline_candidate.generation,
            metadata=dict(baseline_candidate.metadata),
        ),
        seeds=holdout_seeds,
        segment_steps=holdout_segment_steps or config.budget.segment_steps,
        label_prefix=f"{config.system_id}_baseline_holdout",
    )
    winner_phase = await _run_holdout_phase(
        artifact_root=optimizer_phase.artifact_dir,
        config=config,
        runtime=runtime_adapter,
        run_state=optimizer_phase.run_state,
        phase_name="winner_holdout",
        candidate=PromptCandidate(
            candidate_id=f"{best_candidate.candidate_id}_holdout",
            prompt_text=best_candidate.prompt_text,
            generation=best_candidate.generation,
            parent_candidate_id=best_candidate.parent_candidate_id,
            metadata=dict(best_candidate.metadata),
        ),
        seeds=holdout_seeds,
        segment_steps=holdout_segment_steps or config.budget.segment_steps,
        label_prefix=f"{config.system_id}_winner_holdout",
    )
    run_state = optimizer_phase.run_state
    _set_phase_status(run_state, "summary", "running")
    _write_run_state(run_state)
    checkpoint_frontier = _compact_checkpoint_summary(optimizer_result)
    branch_cohorts = _cohort_summary(optimizer_result)
    trajectory_frontier = _frontier_summary(optimizer_result)
    waypoint_summary = _label_summary(optimizer_result)
    prompt_trace_overview = _prompt_trace_overview(optimizer_result)
    bundle = RealCrafterExperimentResult(
        system_id=config.system_id,
        artifact_dir=optimizer_phase.artifact_dir,
        config=config,
        optimizer_result=optimizer_result,
        baseline_holdout=baseline_phase.to_holdout_evaluation(),
        winning_holdout=winner_phase.to_holdout_evaluation(),
        checkpoint_frontier=checkpoint_frontier,
        branch_cohorts=branch_cohorts,
        trajectory_frontier=trajectory_frontier,
        waypoint_summary=waypoint_summary,
        prompt_trace_overview=prompt_trace_overview,
        acceptance=_build_acceptance_verdict(
            result=optimizer_result,
            baseline_holdout=baseline_phase.to_holdout_evaluation(),
            winning_holdout=winner_phase.to_holdout_evaluation(),
            required_uplift=required_uplift,
            baseline_candidate_id=baseline_candidate.candidate_id,
        ),
    )
    _write_json(
        optimizer_phase.artifact_dir / "summary.json",
        bundle.to_dict(),
    )
    _set_phase_status(run_state, "summary", "completed")
    _write_run_state(run_state)
    return bundle


async def run_real_crafter_optimizer_phase(
    config: GoExploreConfig,
    *,
    output_dir: Path | str = DEFAULT_ACCEPTANCE_OUTPUT_DIR,
    runtime: ContainerSearchRuntime | None = None,
) -> RealCrafterPhaseResult:
    artifact_root = _artifact_root_for(output_dir, config.system_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_root / "config.json", _sanitize_config_payload(config.to_dict()))
    run_state = _load_or_init_run_state(config.system_id, artifact_root)
    _set_phase_status(run_state, "optimizer", "running")
    _write_run_state(run_state)
    runtime_adapter = runtime or ContainerSearchRuntime(config.crafter)
    optimizer = GoExploreOptimizer(config, runtime=runtime_adapter)
    try:
        result = await optimizer.optimize()
        best_candidate = result.best_candidate
        if best_candidate is None:
            raise RuntimeError("optimizer result is missing a best candidate")
        run_state.selected_best_candidate_id = best_candidate.candidate_id
        _write_optimizer_artifacts(
            artifact_root=artifact_root,
            optimizer_result=result,
        )
        _set_phase_status(run_state, "optimizer", "completed")
        _write_run_state(run_state)
        return RealCrafterPhaseResult(
            system_id=config.system_id,
            artifact_dir=artifact_root,
            phase="optimizer-only",
            run_state=run_state,
            optimizer_result=result,
        )
    except Exception as exc:
        _set_phase_status(
            run_state,
            "optimizer",
            "failed",
            failure=_serialize_exception(exc),
        )
        _write_run_state(run_state)
        raise


async def run_real_crafter_holdout_phase(
    *,
    artifact_dir: Path | str,
    api_key: str,
    phase: Literal["baseline-holdout-only", "winner-holdout-only"],
    holdout_seeds: Sequence[int] = DEFAULT_HOLDOUT_SEEDS,
    holdout_segment_steps: int | None = None,
    runtime: ContainerSearchRuntime | None = None,
) -> RealCrafterPhaseResult:
    artifact_root = Path(artifact_dir).expanduser().resolve()
    config = _load_config_from_artifact(artifact_root, api_key=api_key)
    runtime_adapter = runtime or ContainerSearchRuntime(config.crafter)
    run_state = _load_or_init_run_state(config.system_id, artifact_root)
    optimizer_result = _load_optimizer_result_from_artifact(artifact_root)
    run_state.selected_best_candidate_id = optimizer_result.best_candidate_id
    _set_phase_status(run_state, "optimizer", "completed")
    _write_run_state(run_state)
    segment_steps = holdout_segment_steps or config.budget.segment_steps
    if phase == "baseline-holdout-only":
        baseline_candidate = _baseline_candidate_from_config(config)
        holdout_artifact = await _run_holdout_phase(
            artifact_root=artifact_root,
            config=config,
            runtime=runtime_adapter,
            run_state=run_state,
            phase_name="baseline_holdout",
            candidate=PromptCandidate(
                candidate_id=f"{baseline_candidate.candidate_id}_holdout",
                prompt_text=baseline_candidate.prompt_text,
                generation=baseline_candidate.generation,
                metadata=dict(baseline_candidate.metadata),
            ),
            seeds=holdout_seeds,
            segment_steps=segment_steps,
            label_prefix=f"{config.system_id}_baseline_holdout",
        )
    else:
        winner = _load_best_candidate_from_optimizer_artifact(artifact_root)
        holdout_artifact = await _run_holdout_phase(
            artifact_root=artifact_root,
            config=config,
            runtime=runtime_adapter,
            run_state=run_state,
            phase_name="winner_holdout",
            candidate=PromptCandidate(
                candidate_id=f"{winner.candidate_id}_holdout",
                prompt_text=winner.prompt_text,
                generation=winner.generation,
                parent_candidate_id=winner.parent_candidate_id,
                metadata=dict(winner.metadata),
            ),
            seeds=holdout_seeds,
            segment_steps=segment_steps,
            label_prefix=f"{config.system_id}_winner_holdout",
        )
    return RealCrafterPhaseResult(
        system_id=config.system_id,
        artifact_dir=artifact_root,
        phase=phase,
        run_state=run_state,
        optimizer_result=optimizer_result,
        holdout_artifact=holdout_artifact,
    )


def summarize_real_crafter_result(result: RealCrafterExperimentResult) -> dict[str, Any]:
    best_candidate = result.optimizer_result.best_candidate
    return {
        "system_id": result.system_id,
        "best_candidate_id": result.optimizer_result.best_candidate_id,
        "best_prompt": best_candidate.prompt_text if best_candidate is not None else None,
        "baseline_holdout_mean_reward": result.baseline_holdout.mean_reward,
        "winning_holdout_mean_reward": result.winning_holdout.mean_reward,
        "mean_uplift": result.winning_holdout.mean_reward - result.baseline_holdout.mean_reward,
        "archive_summary": result.optimizer_result.archive_summary.to_dict(),
        "acceptance": result.acceptance.to_dict(),
        "waypoint_summary": dict(result.waypoint_summary),
        "checkpoint_frontier": list(result.checkpoint_frontier),
        "branch_cohorts": list(result.branch_cohorts),
        "trajectory_frontier": list(result.trajectory_frontier),
        "prompt_trace_overview": dict(result.prompt_trace_overview),
        "prompt_trace_path": str(result.artifact_dir / "prompt_trace.json"),
        "artifact_dir": str(result.artifact_dir),
    }


def _artifact_root_for(output_dir: Path | str, system_id: str) -> Path:
    return Path(output_dir).expanduser().resolve() / system_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _empty_run_state(system_id: str, artifact_dir: Path) -> RunStateSnapshot:
    return RunStateSnapshot(
        system_id=system_id,
        artifact_dir=artifact_dir,
        current_phase="optimizer",
        phases={name: PhaseStatusRecord() for name in REAL_CRAFTER_PHASES},
    )


def _load_or_init_run_state(system_id: str, artifact_dir: Path) -> RunStateSnapshot:
    run_state_path = artifact_dir / "run_state.json"
    if not run_state_path.exists():
        return _empty_run_state(system_id, artifact_dir)
    payload = json.loads(run_state_path.read_text(encoding="utf-8"))
    phases_payload = payload.get("phases", {})
    phases = {
        name: PhaseStatusRecord(
            status=str((phases_payload.get(name) or {}).get("status", "pending")),
            failure=(
                dict((phases_payload.get(name) or {}).get("failure", {}))
                if isinstance((phases_payload.get(name) or {}).get("failure"), dict)
                else None
            ),
            completed_seeds=[
                int(item)
                for item in (phases_payload.get(name) or {}).get("completed_seeds", [])
            ],
            pending_seeds=[
                int(item)
                for item in (phases_payload.get(name) or {}).get("pending_seeds", [])
            ],
        )
        for name in REAL_CRAFTER_PHASES
    }
    return RunStateSnapshot(
        system_id=str(payload.get("system_id", system_id)),
        artifact_dir=artifact_dir,
        current_phase=str(payload.get("current_phase", "optimizer")),
        selected_best_candidate_id=(
            str(payload["selected_best_candidate_id"])
            if payload.get("selected_best_candidate_id") is not None
            else None
        ),
        phases=phases,
    )


def _write_run_state(run_state: RunStateSnapshot) -> None:
    _write_json(run_state.artifact_dir / "run_state.json", run_state.to_dict())


def _set_phase_status(
    run_state: RunStateSnapshot,
    phase_name: RealCrafterPhaseName,
    status: str,
    *,
    failure: dict[str, Any] | None = None,
    completed_seeds: Sequence[int] | None = None,
    pending_seeds: Sequence[int] | None = None,
) -> None:
    record = run_state.phases.setdefault(phase_name, PhaseStatusRecord())
    record.status = status
    record.failure = dict(failure) if failure is not None else None
    if completed_seeds is not None:
        record.completed_seeds = [int(item) for item in completed_seeds]
    if pending_seeds is not None:
        record.pending_seeds = [int(item) for item in pending_seeds]
    run_state.current_phase = phase_name


def _serialize_exception(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, CrafterRuntimeRequestError):
        return exc.to_dict()
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _baseline_candidate_from_config(config: GoExploreConfig) -> PromptCandidate:
    if not config.initial_candidates:
        raise RuntimeError("config is missing an initial baseline candidate")
    return config.initial_candidates[0]


def _load_config_from_artifact(artifact_root: Path, *, api_key: str) -> GoExploreConfig:
    config_path = artifact_root / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"missing config artifact: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for section_name in ("runtime", "crafter", "scanner", "proposer"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            if "inference_api_key" in section or section_name != "crafter":
                section["inference_api_key"] = api_key
    return GoExploreConfig.from_dict(payload)


def _load_optimizer_result_payload(artifact_root: Path) -> dict[str, Any]:
    optimizer_path = artifact_root / "optimizer_result.json"
    if not optimizer_path.exists():
        raise RuntimeError(f"missing optimizer artifact: {optimizer_path}")
    payload = json.loads(optimizer_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"optimizer artifact is not a JSON object: {optimizer_path}")
    return payload


def _load_optimizer_result_from_artifact(artifact_root: Path) -> GoExploreResult:
    payload = _load_optimizer_result_payload(artifact_root)
    candidate_pool = payload.get("all_candidates") or payload.get("frontier") or []
    return GoExploreResult(
        system_id=str(payload.get("system_id", artifact_root.name)),
        best_candidate_id=(
            str(payload["best_candidate_id"])
            if payload.get("best_candidate_id") is not None
            else None
        ),
        frontier=[
            PromptCandidate.from_dict(item)
            for item in payload.get("frontier", [])
            if isinstance(item, dict)
        ],
        all_candidates=[
            PromptCandidate.from_dict(item)
            for item in candidate_pool
            if isinstance(item, dict)
        ],
        evaluations=[],
        archive_summary=_load_archive_summary(payload.get("archive_summary", {})),
    )


def _load_best_candidate_from_optimizer_artifact(artifact_root: Path) -> PromptCandidate:
    payload = _load_optimizer_result_payload(artifact_root)
    best_candidate_id = payload.get("best_candidate_id")
    if not isinstance(best_candidate_id, str) or not best_candidate_id.strip():
        raise RuntimeError("optimizer artifact is missing best_candidate_id")
    candidate_pool = payload.get("all_candidates") or payload.get("frontier") or []
    for item in candidate_pool:
        if not isinstance(item, dict):
            continue
        if str(item.get("candidate_id", "")).strip() == best_candidate_id:
            return PromptCandidate.from_dict(item)
    raise RuntimeError(
        f"optimizer artifact does not contain best candidate payload for {best_candidate_id}"
    )


def _load_archive_summary(payload: dict[str, Any]) -> Any:
    from .legacy.models import ArchiveSummary

    return ArchiveSummary(
        candidate_count=int(payload.get("candidate_count", 0)),
        rollout_count=int(payload.get("rollout_count", 0)),
        resumed_rollout_count=int(payload.get("resumed_rollout_count", 0)),
        checkpoint_count=int(payload.get("checkpoint_count", 0)),
        label_view_count=int(payload.get("label_view_count", 0)),
        verified_rollout_count=int(payload.get("verified_rollout_count", 0)),
        max_verified_depth=int(payload.get("max_verified_depth", 0)),
        seed_coverage=[str(item) for item in payload.get("seed_coverage", [])],
        waypoint_counts=dict(payload.get("waypoint_counts", {})),
        undercovered_waypoints=[str(item) for item in payload.get("undercovered_waypoints", [])],
        total_cost_by_category=dict(payload.get("total_cost_by_category", {})),
    )


def _write_optimizer_artifacts(
    *,
    artifact_root: Path,
    optimizer_result: GoExploreResult,
) -> None:
    _write_json(artifact_root / "optimizer_result.json", optimizer_result.to_dict())
    _write_json(artifact_root / "prompt_trace.json", _build_prompt_trace(optimizer_result))
    _write_json(artifact_root / "checkpoint_frontier.json", {"checkpoint_frontier": _compact_checkpoint_summary(optimizer_result)})
    _write_json(artifact_root / "branch_cohorts.json", {"branch_cohorts": _cohort_summary(optimizer_result)})
    _write_json(artifact_root / "trajectory_frontier.json", {"trajectory_frontier": _frontier_summary(optimizer_result)})
    _write_json(artifact_root / "waypoint_summary.json", _label_summary(optimizer_result))


def _holdout_artifact_path(artifact_root: Path, phase_name: RealCrafterPhaseName) -> Path:
    if phase_name not in {"baseline_holdout", "winner_holdout"}:
        raise ValueError(f"unsupported holdout phase: {phase_name}")
    return artifact_root / f"{phase_name}.json"


async def _run_holdout_phase(
    *,
    artifact_root: Path,
    config: GoExploreConfig,
    runtime: ContainerSearchRuntime,
    run_state: RunStateSnapshot,
    phase_name: Literal["baseline_holdout", "winner_holdout"],
    candidate: PromptCandidate,
    seeds: Sequence[int],
    segment_steps: int,
    label_prefix: str,
) -> HoldoutPhaseArtifact:
    phase_status = run_state.phases.setdefault(phase_name, PhaseStatusRecord())
    phase_status.status = "running"
    phase_status.failure = None
    phase_status.completed_seeds = []
    phase_status.pending_seeds = [int(seed) for seed in seeds]
    run_state.current_phase = phase_name
    holdout_artifact = HoldoutPhaseArtifact(
        phase_name=phase_name,
        candidate_id=candidate.candidate_id,
        prompt_text=candidate.prompt_text,
        phase_status=phase_status,
    )
    _write_run_state(run_state)
    _write_json(_holdout_artifact_path(artifact_root, phase_name), holdout_artifact.to_dict())

    for seed in seeds:
        query = SearchQuery(
            query_id=f"{label_prefix}_{seed}_{uuid4().hex[:8]}",
            candidate_id=candidate.candidate_id,
            start_state_ref=StartStateRef.from_seed(TaskSeedRef(seed_id=str(seed))),
            intent="holdout_evaluation",
            budget_hint_steps=segment_steps,
            metadata={
                "runtime_phase": phase_name,
                "request_kind": "holdout",
            },
        )
        try:
            rollout = await runtime.run_query(query, candidate)
        except Exception as exc:
            phase_status.status = "failed"
            phase_status.failure = _serialize_exception(exc)
            phase_status.pending_seeds = [
                int(item)
                for item in seeds
                if int(item) not in phase_status.completed_seeds
            ]
            _write_run_state(run_state)
            _write_json(_holdout_artifact_path(artifact_root, phase_name), holdout_artifact.to_dict())
            raise
        episode = {
            "seed": int(seed),
            "query_id": query.query_id,
            "reward": rollout.total_reward,
            "step_count": rollout.step_count,
            "status": rollout.status,
            "achievements": sorted(
                {
                    str(item)
                    for item in rollout.reward_details.get("achievements", []) or []
                }
            ),
            "checkpoint_ids": [item.checkpoint_id for item in rollout.checkpoints],
        }
        holdout_artifact.episodes.append(episode)
        phase_status.completed_seeds.append(int(seed))
        phase_status.pending_seeds = [
            int(item)
            for item in seeds
            if int(item) not in phase_status.completed_seeds
        ]
        rewards = [float(item["reward"]) for item in holdout_artifact.episodes]
        holdout_artifact.mean_reward = mean(rewards) if rewards else 0.0
        holdout_artifact.max_reward = max(rewards, default=0.0)
        holdout_artifact.achievement_counts = dict(
            Counter(
                achievement
                for item in holdout_artifact.episodes
                for achievement in item["achievements"]
            )
        )
        _write_run_state(run_state)
        _write_json(_holdout_artifact_path(artifact_root, phase_name), holdout_artifact.to_dict())

    phase_status.status = "completed"
    phase_status.failure = None
    _write_run_state(run_state)
    _write_json(_holdout_artifact_path(artifact_root, phase_name), holdout_artifact.to_dict())
    return holdout_artifact


def _sanitize_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(payload))
    runtime = sanitized.get("runtime")
    if isinstance(runtime, dict):
        if runtime.get("inference_api_key"):
            runtime["inference_api_key"] = "<redacted>"
        if runtime.get("container_api_key"):
            runtime["container_api_key"] = "<redacted>"
    crafter = sanitized.get("crafter")
    if isinstance(crafter, dict):
        if crafter.get("inference_api_key"):
            crafter["inference_api_key"] = "<redacted>"
        if crafter.get("container_api_key"):
            crafter["container_api_key"] = "<redacted>"
    proposer = sanitized.get("proposer")
    if isinstance(proposer, dict) and proposer.get("inference_api_key"):
        proposer["inference_api_key"] = "<redacted>"
    scanner = sanitized.get("scanner")
    if isinstance(scanner, dict) and scanner.get("inference_api_key"):
        scanner["inference_api_key"] = "<redacted>"
    return sanitized


def _build_acceptance_verdict(
    *,
    result: GoExploreResult,
    baseline_holdout: HoldoutEvaluation,
    winning_holdout: HoldoutEvaluation,
    required_uplift: float,
    baseline_candidate_id: str,
) -> AcceptanceVerdict:
    reasons: list[str] = []
    actual_uplift = winning_holdout.mean_reward - baseline_holdout.mean_reward
    if result.archive_summary.resumed_rollout_count <= 0:
        reasons.append("No resumed checkpoint branches executed.")
    if not result.branch_cohorts:
        reasons.append("No branch cohorts were recorded.")
    if result.best_candidate_id is None:
        reasons.append("Optimizer did not select a best candidate.")
    elif result.best_candidate_id == baseline_candidate_id:
        reasons.append("Optimizer winner remained the baseline candidate.")
    if actual_uplift < required_uplift:
        reasons.append(
            f"Holdout uplift {actual_uplift:.3f} is below required uplift {required_uplift:.3f}."
        )
    return AcceptanceVerdict(
        accepted=not reasons,
        required_uplift=required_uplift,
        actual_uplift=actual_uplift,
        reasons=reasons,
    )


async def _evaluate_prompt(
    *,
    runtime: ContainerSearchRuntime,
    prompt_text: str,
    candidate_id: str,
    seeds: Sequence[int],
    segment_steps: int,
    label_prefix: str,
) -> HoldoutEvaluation:
    candidate = PromptCandidate(
        candidate_id=candidate_id,
        prompt_text=prompt_text,
        generation=0,
    )
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        query = SearchQuery(
            query_id=f"{label_prefix}_{seed}_{uuid4().hex[:8]}",
            candidate_id=candidate_id,
            start_state_ref=StartStateRef.from_seed(TaskSeedRef(seed_id=str(seed))),
            intent="holdout_evaluation",
            budget_hint_steps=segment_steps,
        )
        rollout = await runtime.run_query(query, candidate)
        episodes.append(
            {
                "seed": seed,
                "reward": rollout.total_reward,
                "step_count": rollout.step_count,
                "status": rollout.status,
                "achievements": sorted(
                    {
                        str(item)
                        for item in rollout.reward_details.get("achievements", []) or []
                    }
                ),
                "checkpoint_ids": [item.checkpoint_id for item in rollout.checkpoints],
            }
        )
    rewards = [float(item["reward"]) for item in episodes]
    return HoldoutEvaluation(
        candidate_id=candidate_id,
        prompt_text=prompt_text,
        mean_reward=mean(rewards) if rewards else 0.0,
        max_reward=max(rewards, default=0.0),
        episodes=episodes,
        achievement_counts=dict(
            Counter(
                achievement
                for episode in episodes
                for achievement in episode["achievements"]
            )
        ),
    )


def _compact_checkpoint_summary(result: GoExploreResult) -> list[dict[str, Any]]:
    checkpoints = result.checkpoint_refs[:8]
    rollout_by_id = {item.rollout_id: item for item in result.rollouts}
    label_view_by_rollout = {item.rollout_id: item for item in result.label_views}
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        rollout = rollout_by_id.get(checkpoint.rollout_id)
        label_view = label_view_by_rollout.get(checkpoint.rollout_id)
        checkpoint_labels = (
            label_view.checkpoint_labels.get(checkpoint.checkpoint_id, [])
            if label_view is not None
            else []
        )
        rows.append(
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "rollout_id": checkpoint.rollout_id,
                "seed_id": checkpoint.seed_id,
                "step_index": checkpoint.step_index,
                "total_reward": checkpoint.total_reward,
                "candidate_id": rollout.candidate_id if rollout is not None else None,
                "checkpoint_labels": [item.waypoint_type for item in checkpoint_labels],
            }
        )
    return rows


def _frontier_summary(result: GoExploreResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in result.trajectory_frontier[:5]:
        rows.append(
            {
                "rollout_id": entry.rollout_id,
                "candidate_id": entry.candidate_id,
                "achieved_value": entry.achieved_value,
                "potential_value": entry.potential_value,
                "bottleneck_waypoints": list(entry.bottleneck_waypoints),
                "supporting_checkpoint_ids": list(entry.supporting_checkpoint_ids),
            }
        )
    return rows


def _prompt_trace_overview(result: GoExploreResult) -> dict[str, Any]:
    candidate_entries = _build_prompt_trace(result)["candidates"]
    return {
        "candidate_count": len(candidate_entries),
        "mutated_candidate_count": sum(
            1 for item in candidate_entries if item.get("parent_candidate_id")
        ),
        "candidates": candidate_entries[:5],
    }


def _build_prompt_trace(result: GoExploreResult) -> dict[str, Any]:
    candidate_pool = result.all_candidates or result.frontier
    evaluations_by_id = {
        evaluation.candidate_id: evaluation for evaluation in result.evaluations
    }
    label_view_by_rollout = {
        label_view.rollout_id: label_view for label_view in result.label_views
    }
    verification_by_rollout = {
        report.rollout_id: report for report in result.verification_reports
    }
    queries_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for event in result.events:
        if event.get("type") != "learning.policy.go_explore.query_selection.completed":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        phase = str(payload.get("phase", ""))
        iteration_idx = payload.get("iteration_idx")
        selected_queries = payload.get("selected_queries", [])
        if not isinstance(selected_queries, list):
            continue
        for item in selected_queries:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", "")).strip()
            if not candidate_id:
                continue
            queries_by_candidate.setdefault(candidate_id, []).append(
                {
                    "iteration_idx": iteration_idx,
                    "phase": phase,
                    "query_id": item.get("query_id"),
                    "intent": item.get("intent"),
                    "start_kind": item.get("start_kind"),
                    "checkpoint_id": item.get("checkpoint_id"),
                    "seed_id": item.get("seed_id"),
                    "target_waypoints": list(item.get("target_waypoints", []) or []),
                }
            )
    candidates: list[dict[str, Any]] = []
    for candidate in candidate_pool:
        evaluation = evaluations_by_id.get(candidate.candidate_id)
        rollouts = [
            rollout
            for rollout in result.rollouts
            if rollout.candidate_id == candidate.candidate_id
        ]
        rollout_rows: list[dict[str, Any]] = []
        for rollout in rollouts:
            label_view = label_view_by_rollout.get(rollout.rollout_id)
            verification = verification_by_rollout.get(rollout.rollout_id)
            checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in rollout.checkpoints]
            checkpoint_waypoints: list[str] = []
            if label_view is not None:
                for checkpoint_id in checkpoint_ids:
                    checkpoint_waypoints.extend(
                        label.waypoint_type
                        for label in label_view.checkpoint_labels.get(checkpoint_id, [])
                    )
            rollout_rows.append(
                {
                    "rollout_id": rollout.rollout_id,
                    "intent": rollout.metadata.get("intent"),
                    "start_state_kind": rollout.start_state_ref.kind,
                    "seed_id": rollout.seed_id,
                    "parent_rollout_id": rollout.parent_rollout_id,
                    "total_reward": rollout.total_reward,
                    "step_count": rollout.step_count,
                    "target_waypoints": list(rollout.metadata.get("target_waypoints", []) or []),
                    "checkpoint_ids": checkpoint_ids,
                    "completed_waypoints": (
                        list(label_view.completed_waypoint_types)
                        if label_view is not None
                        else []
                    ),
                    "checkpoint_waypoints": list(dict.fromkeys(checkpoint_waypoints)),
                    "verified_waypoints": (
                        list(verification.verified_waypoints)
                        if verification is not None
                        else []
                    ),
                    "branch_ready": (
                        verification.branch_ready if verification is not None else None
                    ),
                    "loop_detected": (
                        verification.loop_detected if verification is not None else None
                    ),
                }
            )
        candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "parent_candidate_id": candidate.parent_candidate_id,
                "generation": candidate.generation,
                "prompt_text": candidate.prompt_text,
                "proposal_summary": candidate.metadata.get("proposal_summary"),
                "proposal_rationale": candidate.metadata.get("proposal_rationale"),
                "proposal_mode": candidate.metadata.get("proposal_mode"),
                "target_waypoints": list(candidate.metadata.get("target_waypoints", []) or []),
                "supporting_rollout_ids": list(
                    candidate.metadata.get("supporting_rollout_ids", []) or []
                ),
                "supporting_checkpoint_ids": list(
                    candidate.metadata.get("supporting_checkpoint_ids", []) or []
                ),
                "scheduled_queries": queries_by_candidate.get(candidate.candidate_id, []),
                "evaluation": evaluation.to_dict() if evaluation is not None else None,
                "rollouts": rollout_rows,
            }
        )
    candidates.sort(
        key=lambda item: (
            0 if item["candidate_id"] == result.best_candidate_id else 1,
            item["generation"],
            item["candidate_id"],
        )
    )
    return {
        "system_id": result.system_id,
        "best_candidate_id": result.best_candidate_id,
        "candidates": candidates,
    }


def _cohort_summary(result: GoExploreResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in result.branch_cohorts[:5]:
        rows.append(
            {
                "cohort_id": cohort.cohort_id,
                "candidate_id": cohort.candidate_id,
                "intent": cohort.intent,
                "target_waypoints": list(cohort.target_waypoints),
                "rollout_ids": list(cohort.rollout_ids),
            }
        )
    return rows


def _label_summary(result: GoExploreResult) -> dict[str, Any]:
    rollout_waypoints: Counter[str] = Counter()
    checkpoint_waypoints: Counter[str] = Counter()
    for label_view in result.label_views:
        rollout_waypoints.update(label.waypoint_type for label in label_view.waypoint_labels)
        for labels in label_view.checkpoint_labels.values():
            checkpoint_waypoints.update(label.waypoint_type for label in labels)
    return {
        "rollout_waypoints": dict(rollout_waypoints),
        "checkpoint_waypoints": dict(checkpoint_waypoints),
    }
