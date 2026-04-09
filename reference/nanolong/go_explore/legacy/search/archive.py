"""In-memory archive for Go-Explore bring-up."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, Optional, Sequence, TYPE_CHECKING

from ..models import (
    ArchiveSummary,
    BranchCohort,
    CandidateEvaluation,
    CheckpointRef,
    ComparativeLabelView,
    PotentialEstimate,
    PromptCandidate,
    RolloutArtifact,
    ScannerLabelView,
    TrajectoryFrontierEntry,
    VerificationReport,
    WaypointDefinition,
)
from ..config import GoExploreScoringConfig

if TYPE_CHECKING:
    from ..progress.scanner import ModelBasedRolloutScanner
    from ..verifiers import GenericVerifierRegistry


_META_WAYPOINT_PREFIXES = ("meta_", "system_")
_META_WAYPOINT_EXACT: set[str] = set()
_DIAGNOSTIC_WAYPOINT_SUBSTRINGS = (
    "hazard",
    "loop",
    "backtrack",
    "empty",
    "avoid_",
    "death",
    "fail",
    "failure",
    "risk",
    "critical",
    "no_",
    "without_",
    "fragile",
    "diagnostic",
    "avoid",
    "stall",
)
_DIAGNOSTIC_DESCRIPTION_SUBSTRINGS = (
    "diagnostic",
    "failure-mode",
    "failure mode",
    "anti-target",
    "negative modifier",
)
_EXPLORATORY_DESCRIPTION_SUBSTRINGS = (
    "exploratory",
    "underexplored",
    "not yet durable",
    "not durable",
    "candidate waypoint",
    "single supporting rollout",
    "single-support",
    "track but do not prioritize",
    "frontier rather than durable cluster",
)


def is_meta_waypoint_type(waypoint_type: str) -> bool:
    normalized = str(waypoint_type).strip()
    if not normalized:
        return False
    if normalized in _META_WAYPOINT_EXACT:
        return True
    return normalized.startswith(_META_WAYPOINT_PREFIXES)


def is_diagnostic_waypoint_type(
    waypoint_type: str,
    *,
    definition: WaypointDefinition | None = None,
) -> bool:
    normalized = str(waypoint_type).strip().lower()
    if not normalized:
        return False
    if any(token in normalized for token in _DIAGNOSTIC_WAYPOINT_SUBSTRINGS):
        return True
    if definition is None:
        return False
    text = " ".join(
        [
            str(definition.description or ""),
            *[str(note) for note in definition.notes],
        ]
    ).lower()
    return any(token in text for token in _DIAGNOSTIC_DESCRIPTION_SUBSTRINGS)


def is_exploratory_waypoint_type(
    waypoint_type: str,
    *,
    definition: WaypointDefinition | None = None,
) -> bool:
    del waypoint_type
    if definition is None:
        return False
    text = " ".join(
        [
            str(definition.description or ""),
            *[str(note) for note in definition.notes],
        ]
    ).lower()
    return any(token in text for token in _EXPLORATORY_DESCRIPTION_SUBSTRINGS)


class InMemoryArchive:
    """Lightweight archive used for local algorithm bring-up."""

    def __init__(
        self,
        *,
        expected_waypoints: Iterable[str] = (),
        trajectory_frontier_size: int = 8,
        scoring: GoExploreScoringConfig | None = None,
    ) -> None:
        self.candidates: Dict[str, PromptCandidate] = {}
        self.rollouts: Dict[str, RolloutArtifact] = {}
        self.checkpoints: Dict[str, CheckpointRef] = {}
        self.label_views: Dict[str, ScannerLabelView] = {}
        self.rollout_to_label_view: Dict[str, str] = {}
        self.verification_reports: Dict[str, VerificationReport] = {}
        self.rollout_to_verification: Dict[str, str] = {}
        self.waypoint_definitions: Dict[str, WaypointDefinition] = {}
        self.total_cost_by_category: Dict[str, float] = defaultdict(float)
        self.branch_cohorts: Dict[str, BranchCohort] = {}
        self.comparative_views: Dict[str, ComparativeLabelView] = {}
        self.cohort_to_comparative_view: Dict[str, str] = {}
        self.trajectory_frontier: Dict[str, TrajectoryFrontierEntry] = {}
        self.potential_estimates: Dict[str, PotentialEstimate] = {}
        self.trajectory_frontier_size = trajectory_frontier_size
        self.scoring = scoring or GoExploreScoringConfig()
        for waypoint_type in expected_waypoints:
            self.register_waypoints(
                [
                    WaypointDefinition(
                        waypoint_type=str(waypoint_type),
                        description=f"Manual seed waypoint {waypoint_type}.",
                        matcher_kind="existing_label",
                        matcher={"waypoint_type": str(waypoint_type)},
                        proposed_by="manual_seed",
                    )
                ]
            )

    def register_candidates(self, candidates: Sequence[PromptCandidate]) -> None:
        for candidate in candidates:
            self.candidates[candidate.candidate_id] = candidate

    def ingest_rollout(
        self,
        rollout: RolloutArtifact,
        label_view: Optional[ScannerLabelView] = None,
        verification_report: Optional[VerificationReport] = None,
    ) -> None:
        self.rollouts[rollout.rollout_id] = rollout
        for checkpoint in rollout.checkpoints:
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        for category, amount in rollout.cost_by_category.items():
            self.total_cost_by_category[category] += amount
        if label_view is not None:
            self.record_label_view(label_view)
        if verification_report is not None:
            self.record_verification_report(verification_report)
        self._record_branch_cohort(rollout)
        self._update_potential_estimate(rollout, label_view, verification_report)
        self._update_trajectory_frontier(rollout, label_view, verification_report)
        cohort_id = rollout.metadata.get("cohort_id")
        if isinstance(cohort_id, str) and cohort_id:
            self._refresh_comparative_view(cohort_id)

    def record_label_view(self, label_view: ScannerLabelView) -> None:
        self.label_views[label_view.label_view_id] = label_view
        self.rollout_to_label_view[label_view.rollout_id] = label_view.label_view_id

    def record_verification_report(self, report: VerificationReport) -> None:
        self.verification_reports[report.report_id] = report
        self.rollout_to_verification[report.rollout_id] = report.report_id

    def latest_label_view_for_rollout(self, rollout_id: str) -> Optional[ScannerLabelView]:
        label_view_id = self.rollout_to_label_view.get(rollout_id)
        if label_view_id is None:
            return None
        return self.label_views.get(label_view_id)

    def verification_for_rollout(self, rollout_id: str) -> Optional[VerificationReport]:
        report_id = self.rollout_to_verification.get(rollout_id)
        if report_id is None:
            return None
        return self.verification_reports.get(report_id)

    def waypoint_counts(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for label_view in self.label_views.values():
            counts.update(label_view.completed_waypoint_types)
        return dict(counts)

    def label_support_counts(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for label_view in self.label_views.values():
            counts.update(
                {
                    label.waypoint_type
                    for label in label_view.waypoint_labels
                    if label.status in {"observed", "completed"}
                }
            )
        return dict(counts)

    def bottleneck_waypoint_counts(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for view in self.comparative_views.values():
            counts.update(view.bottleneck_waypoints)
        return dict(counts)

    def registered_waypoint_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.waypoint_definitions))

    def registered_waypoint_definitions(self) -> list[WaypointDefinition]:
        return [
            self.waypoint_definitions[waypoint_type]
            for waypoint_type in sorted(self.waypoint_definitions)
        ]

    def undercovered_waypoints(self) -> tuple[str, ...]:
        counts = self.waypoint_counts()
        undercovered = [
            waypoint for waypoint in self.registered_waypoint_types() if counts.get(waypoint, 0) == 0
        ]
        return tuple(undercovered)

    def active_search_targets(self, *, limit: int = 4) -> tuple[str, ...]:
        targets = sorted(
            (
                waypoint
                for waypoint in self.undercovered_waypoints()
                if not is_meta_waypoint_type(waypoint)
                and not is_diagnostic_waypoint_type(
                    waypoint,
                    definition=self.waypoint_definitions.get(waypoint),
                )
            ),
            key=self.search_target_priority,
            reverse=True,
        )
        if not targets:
            bottlenecks = sorted(
                (
                    waypoint
                    for waypoint, _ in sorted(
                        self.bottleneck_waypoint_counts().items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                    if not is_meta_waypoint_type(waypoint)
                    and not is_diagnostic_waypoint_type(
                        waypoint,
                        definition=self.waypoint_definitions.get(waypoint),
                    )
                ),
                key=self.search_target_priority,
                reverse=True,
            )
            targets.extend(waypoint for waypoint in bottlenecks if waypoint not in targets)
        if not targets:
            prioritized = sorted(
                (
                    waypoint
                    for waypoint, _ in sorted(
                        self.waypoint_counts().items(),
                        key=lambda item: (-self.learned_label_utility(item[0]), item[0]),
                    )
                    if not is_meta_waypoint_type(waypoint)
                    and not is_diagnostic_waypoint_type(
                        waypoint,
                        definition=self.waypoint_definitions.get(waypoint),
                    )
                ),
                key=self.search_target_priority,
                reverse=True,
            )
            targets.extend(waypoint for waypoint in prioritized if waypoint not in targets)
        return tuple(targets[:limit])

    def search_target_priority(self, waypoint_type: str) -> tuple[float, float, float, str]:
        definition = self.waypoint_definitions.get(waypoint_type)
        exploratory_penalty = -1.0 if is_exploratory_waypoint_type(
            waypoint_type,
            definition=definition,
        ) else 0.0
        return (
            exploratory_penalty,
            self.learned_label_utility(waypoint_type),
            float(self.label_support_counts().get(waypoint_type, 0)),
            waypoint_type,
        )

    def register_waypoints(
        self,
        definitions: Sequence[WaypointDefinition],
    ) -> list[WaypointDefinition]:
        added: list[WaypointDefinition] = []
        for definition in definitions:
            if definition.waypoint_type in self.waypoint_definitions:
                continue
            self.waypoint_definitions[definition.waypoint_type] = definition
            added.append(definition)
        return added

    def relabel_rollouts(
        self,
        *,
        scanner: Any,
        verifier: Any,
    ) -> None:
        rollouts = self.rollout_artifacts()
        self.label_views = {}
        self.rollout_to_label_view = {}
        self.verification_reports = {}
        self.rollout_to_verification = {}
        self.total_cost_by_category = defaultdict(float)
        self.branch_cohorts = {}
        self.comparative_views = {}
        self.cohort_to_comparative_view = {}
        self.trajectory_frontier = {}
        self.potential_estimates = {}
        for rollout in rollouts:
            for category, amount in rollout.cost_by_category.items():
                self.total_cost_by_category[category] += amount
            label_view = scanner.scan_rollout(rollout, archive=self)
            verification = verifier.verify_rollout(rollout, label_view=label_view)
            self.record_label_view(label_view)
            self.record_verification_report(verification)
            self._record_branch_cohort(rollout)
            self._update_potential_estimate(rollout, label_view, verification)
            self._update_trajectory_frontier(rollout, label_view, verification)
            cohort_id = rollout.metadata.get("cohort_id")
            if isinstance(cohort_id, str) and cohort_id:
                self._refresh_comparative_view(cohort_id)

    def potential_estimate_for_checkpoint(
        self,
        checkpoint_id: str,
    ) -> Optional[PotentialEstimate]:
        return self.potential_estimates.get(f"checkpoint:{checkpoint_id}")

    def comparative_views_for_checkpoint(
        self,
        checkpoint_id: str,
        *,
        candidate_id: Optional[str] = None,
    ) -> list[ComparativeLabelView]:
        views: list[ComparativeLabelView] = []
        for view in self.comparative_views.values():
            checkpoint = view.start_state_ref.checkpoint
            if checkpoint is None or checkpoint.checkpoint_id != checkpoint_id:
                continue
            if candidate_id is not None and view.candidate_id != candidate_id:
                continue
            views.append(view)
        return sorted(views, key=lambda item: item.comparative_view_id)

    def checkpoint_loop_risk(self, checkpoint: CheckpointRef) -> float:
        summary = checkpoint.state_summary
        same_seed = [
            other
            for other in self.checkpoints.values()
            if other.seed_id == checkpoint.seed_id and other.checkpoint_id != checkpoint.checkpoint_id
        ]
        if not same_seed:
            return 0.0
        duplicate_count = sum(
            1
            for other in same_seed
            if other.state_summary.dedupe_signature == summary.dedupe_signature
        )
        dominated = any(self._checkpoint_dominates(other, checkpoint) for other in same_seed)
        repeated_rollout_count = sum(
            1
            for rollout in self.rollouts.values()
            if rollout.start_state_ref.checkpoint_id == checkpoint.checkpoint_id
        )
        risk = 0.0
        if duplicate_count:
            risk += min(0.4, 0.1 * duplicate_count)
        if dominated:
            risk += 0.5
        if repeated_rollout_count > 1:
            risk += min(0.3, 0.1 * (repeated_rollout_count - 1))
        return min(1.0, risk)

    def branchable_checkpoints(self, *, limit: Optional[int] = None) -> list[CheckpointRef]:
        scored: list[tuple[float, CheckpointRef]] = []
        for checkpoint in self.checkpoints.values():
            verification = self.verification_for_rollout(checkpoint.rollout_id)
            if verification is not None and not verification.branch_ready:
                continue
            rollout = self.rollouts.get(checkpoint.rollout_id)
            label_view = self.latest_label_view_for_rollout(checkpoint.rollout_id)
            label_bonus = 0.0
            if label_view is not None:
                checkpoint_labels = label_view.checkpoint_labels.get(checkpoint.checkpoint_id, [])
                label_bonus = (
                    float(len(checkpoint_labels))
                    * self.scoring.checkpoint_ranking.label_bonus_per_waypoint
                )
            reward_bonus = (
                checkpoint.total_reward
                * self.scoring.checkpoint_ranking.reward_bonus_weight
            )
            resumed_bonus = (
                self.scoring.checkpoint_ranking.resumed_bonus
                if rollout is not None and rollout.is_resumed
                else 0.0
            )
            potential_bonus = 0.0
            estimate = self.potential_estimate_for_checkpoint(checkpoint.checkpoint_id)
            if estimate is not None:
                potential_bonus = max(
                    0.0,
                    estimate.potential_value - estimate.achieved_value - estimate.loop_risk,
                ) * self.scoring.checkpoint_ranking.potential_gap_weight
            loop_penalty = (
                self.checkpoint_loop_risk(checkpoint)
                * self.scoring.checkpoint_ranking.loop_penalty_weight
            )
            scored.append(
                (
                    label_bonus + reward_bonus + resumed_bonus + potential_bonus - loop_penalty,
                    checkpoint,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        checkpoints = [checkpoint for _, checkpoint in scored]
        if limit is not None:
            return checkpoints[:limit]
        return checkpoints

    def build_archive_summary(self) -> ArchiveSummary:
        seed_coverage = tuple(sorted({rollout.seed_id for rollout in self.rollouts.values()}))
        resumed_rollout_count = sum(1 for rollout in self.rollouts.values() if rollout.is_resumed)
        high_potential_checkpoint_count = sum(
            1
            for estimate in self.potential_estimates.values()
            if estimate.target_kind == "checkpoint"
            and estimate.potential_value > estimate.achieved_value
        )
        return ArchiveSummary(
            candidate_count=len(self.candidates),
            rollout_count=len(self.rollouts),
            resumed_rollout_count=resumed_rollout_count,
            checkpoint_count=len(self.checkpoints),
            label_view_count=len(self.label_views),
            verified_rollout_count=len(self.verification_reports),
            max_verified_depth=max(
                (report.verified_depth for report in self.verification_reports.values()),
                default=0,
            ),
            seed_coverage=seed_coverage,
            waypoint_counts=self.waypoint_counts(),
            undercovered_waypoints=self.undercovered_waypoints(),
            total_cost_by_category=dict(self.total_cost_by_category),
            trajectory_frontier_count=len(self.trajectory_frontier),
            cohort_count=len(self.branch_cohorts),
            comparative_view_count=len(self.comparative_views),
            high_potential_checkpoint_count=high_potential_checkpoint_count,
            bottleneck_waypoint_counts=self.bottleneck_waypoint_counts(),
        )

    def candidate_rollouts(self, candidate_id: str) -> list[RolloutArtifact]:
        return [
            rollout
            for rollout in self.rollouts.values()
            if rollout.candidate_id == candidate_id
        ]

    def candidate_comparative_views(self, candidate_id: str) -> list[ComparativeLabelView]:
        return [
            view
            for view in self.comparative_views.values()
            if view.candidate_id == candidate_id
        ]

    def candidate_frontier_entries(self, candidate_id: str) -> list[TrajectoryFrontierEntry]:
        return [
            entry
            for entry in self.trajectory_frontier.values()
            if entry.candidate_id == candidate_id
        ]

    def evaluate_candidates(
        self,
        candidate_ids: Optional[Sequence[str]] = None,
    ) -> list[CandidateEvaluation]:
        target_ids = list(candidate_ids or self.candidates.keys())
        evaluations: list[CandidateEvaluation] = []
        for candidate_id in target_ids:
            rollouts = self.candidate_rollouts(candidate_id)
            label_views = [
                label_view
                for rollout in rollouts
                if (label_view := self.latest_label_view_for_rollout(rollout.rollout_id)) is not None
            ]
            verification_reports = [
                report
                for rollout in rollouts
                if (report := self.verification_for_rollout(rollout.rollout_id)) is not None
            ]
            comparative_views = self.candidate_comparative_views(candidate_id)
            completed_waypoint_scores = [
                sum(self.learned_label_utility(waypoint) for waypoint in label_view.completed_waypoint_types)
                for label_view in label_views
            ]
            progress_score = mean(completed_waypoint_scores) if completed_waypoint_scores else 0.0
            novelty_score = mean(label_view.novelty_score for label_view in label_views) if label_views else 0.0
            learnability_score = (
                mean(label_view.learnability_by_plugin.get("prompt", 0.0) for label_view in label_views)
                if label_views
                else 0.0
            )

            fresh_progress = [
                sum(
                    self.learned_label_utility(waypoint)
                    for waypoint in self.latest_label_view_for_rollout(rollout.rollout_id).completed_waypoint_types
                )
                for rollout in rollouts
                if not rollout.is_resumed and self.latest_label_view_for_rollout(rollout.rollout_id) is not None
            ]
            resumed_progress = [
                sum(
                    self.learned_label_utility(waypoint)
                    for waypoint in self.latest_label_view_for_rollout(rollout.rollout_id).completed_waypoint_types
                )
                for rollout in rollouts
                if rollout.is_resumed and self.latest_label_view_for_rollout(rollout.rollout_id) is not None
            ]
            checkpoint_progress_gain = (
                mean(resumed_progress) - mean(fresh_progress)
                if resumed_progress and fresh_progress
                else (mean(resumed_progress) if resumed_progress else 0.0)
            )
            frontier_expansion = float(
                sum(
                    self.learned_label_utility(waypoint)
                    for waypoint in {
                        waypoint
                        for label_view in label_views
                        for waypoint in label_view.completed_waypoint_types
                    }
                )
            )
            verified_depth = (
                mean(report.verified_depth for report in verification_reports)
                if verification_reports
                else 0.0
            )
            fresh_verified_depth = [
                self.verification_for_rollout(rollout.rollout_id).verified_depth
                for rollout in rollouts
                if not rollout.is_resumed and self.verification_for_rollout(rollout.rollout_id) is not None
            ]
            resumed_verified_depth = [
                self.verification_for_rollout(rollout.rollout_id).verified_depth
                for rollout in rollouts
                if rollout.is_resumed and self.verification_for_rollout(rollout.rollout_id) is not None
            ]
            total_cost = sum(sum(rollout.cost_by_category.values()) for rollout in rollouts)
            trajectory_frontier_hits = sum(
                1 for entry in self.trajectory_frontier.values() if entry.candidate_id == candidate_id
            )
            potential_gaps = [
                max(0.0, estimate.potential_value - estimate.achieved_value - estimate.loop_risk)
                for estimate in self.potential_estimates.values()
                if estimate.target_kind == "checkpoint"
                and any(
                    rollout.start_state_ref.checkpoint_id == estimate.target_id
                    for rollout in rollouts
                    if rollout.start_state_ref.kind == "checkpoint"
                )
            ]
            average_potential_gap = mean(potential_gaps) if potential_gaps else 0.0
            average_loop_risk = mean(
                view.loop_risk for view in comparative_views
            ) if comparative_views else 0.0
            selection_score = (
                progress_score * self.scoring.candidate_evaluation.progress_weight
                + novelty_score * self.scoring.candidate_evaluation.novelty_weight
                + learnability_score * self.scoring.candidate_evaluation.learnability_weight
                + checkpoint_progress_gain
                * self.scoring.candidate_evaluation.checkpoint_progress_gain_weight
                + frontier_expansion
                * self.scoring.candidate_evaluation.frontier_expansion_weight
                + verified_depth * self.scoring.candidate_evaluation.verified_depth_weight
                - total_cost * self.scoring.candidate_evaluation.total_cost_penalty_weight
            )
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate_id,
                    progress_score=progress_score,
                    novelty_score=novelty_score,
                    learnability_score=learnability_score,
                    checkpoint_progress_gain=checkpoint_progress_gain,
                    frontier_expansion=frontier_expansion,
                    verified_depth=verified_depth,
                    total_cost=total_cost,
                    selection_score_value=selection_score,
                    metrics={
                        "rollout_count": len(rollouts),
                        "fresh_rollout_count": len([rollout for rollout in rollouts if not rollout.is_resumed]),
                        "resumed_rollout_count": len([rollout for rollout in rollouts if rollout.is_resumed]),
                        "verified_progress_gain": (
                            mean(resumed_verified_depth) - mean(fresh_verified_depth)
                            if resumed_verified_depth and fresh_verified_depth
                            else (mean(resumed_verified_depth) if resumed_verified_depth else 0.0)
                        ),
                        "cohort_count": len(comparative_views),
                        "trajectory_frontier_hits": trajectory_frontier_hits,
                        "average_potential_gap": average_potential_gap,
                        "average_loop_risk": average_loop_risk,
                    },
                )
            )
        evaluations.sort(key=lambda item: item.selection_score, reverse=True)
        return evaluations

    def learned_label_utility(self, waypoint_type: str) -> float:
        support_rollout_ids = self._supporting_rollout_ids_for_label(waypoint_type)
        if not support_rollout_ids:
            return 1.0
        outcome_scores = [
            self._rollout_outcome_score(rollout_id) for rollout_id in support_rollout_ids
        ]
        if not outcome_scores:
            return 1.0
        # Shrink sparse label histories toward a neutral prior until the archive has enough evidence.
        return (1.0 + sum(outcome_scores)) / (1.0 + len(outcome_scores))

    def _supporting_rollout_ids_for_label(self, waypoint_type: str) -> list[str]:
        return [
            label_view.rollout_id
            for label_view in self.label_views.values()
            if any(
                label.waypoint_type == waypoint_type
                and label.status in {"observed", "completed"}
                for label in label_view.waypoint_labels
            )
        ]

    def _rollout_outcome_score(self, rollout_id: str) -> float:
        rollout = self.rollouts.get(rollout_id)
        if rollout is None:
            return 0.0
        report = self.verification_for_rollout(rollout_id)
        label_view = self.latest_label_view_for_rollout(rollout_id)
        depth_component = (report.verified_depth / 16.0) if report is not None else 0.0
        verified_score_component = (
            max(0.0, report.score) * 0.25 if report is not None else 0.0
        )
        reward_component = max(0.0, rollout.total_reward) * 0.25
        checkpoint_gain_component = 0.0
        if rollout.is_resumed and rollout.start_state_ref.checkpoint is not None:
            checkpoint_gain_component = max(
                0.0,
                rollout.total_reward - rollout.start_state_ref.checkpoint.total_reward,
            ) * 0.2
        branch_ready_component = 0.1 if report is not None and report.branch_ready else 0.0
        novelty_component = (
            label_view.novelty_score * 0.15 if label_view is not None else 0.0
        )
        loop_penalty = 0.2 if report is not None and report.loop_detected else 0.0
        return max(
            0.0,
            depth_component
            + verified_score_component
            + reward_component
            + checkpoint_gain_component
            + branch_ready_component
            + novelty_component
            - loop_penalty,
        )

    def best_candidate_id(self, candidate_ids: Optional[Sequence[str]] = None) -> Optional[str]:
        evaluations = self.evaluate_candidates(candidate_ids)
        if not evaluations:
            return None
        return evaluations[0].candidate_id

    def checkpoint_refs(self) -> list[CheckpointRef]:
        return sorted(
            self.checkpoints.values(),
            key=lambda checkpoint: (
                checkpoint.seed_id,
                checkpoint.rollout_id,
                checkpoint.step_index,
                checkpoint.checkpoint_id,
            ),
        )

    def rollout_artifacts(self) -> list[RolloutArtifact]:
        return sorted(
            self.rollouts.values(),
            key=lambda rollout: (
                rollout.seed_id,
                rollout.rollout_id,
            ),
        )

    def label_view_artifacts(self) -> list[ScannerLabelView]:
        return sorted(
            self.label_views.values(),
            key=lambda label_view: (
                label_view.rollout_id,
                label_view.label_view_id,
            ),
        )

    def verification_report_artifacts(self) -> list[VerificationReport]:
        return sorted(
            self.verification_reports.values(),
            key=lambda report: (
                report.rollout_id,
                report.report_id,
            ),
        )

    def trajectory_frontier_artifacts(self) -> list[TrajectoryFrontierEntry]:
        return sorted(
            self.trajectory_frontier.values(),
            key=lambda entry: (
                -entry.selection_score,
                -entry.achieved_value,
                entry.rollout_id,
            ),
        )

    def branch_cohort_artifacts(self) -> list[BranchCohort]:
        return sorted(self.branch_cohorts.values(), key=lambda cohort: cohort.cohort_id)

    def comparative_view_artifacts(self) -> list[ComparativeLabelView]:
        return sorted(self.comparative_views.values(), key=lambda view: view.comparative_view_id)

    def potential_estimate_artifacts(self) -> list[PotentialEstimate]:
        return sorted(self.potential_estimates.values(), key=lambda estimate: estimate.estimate_id)

    def reward_reference_stats_for_rollout(
        self,
        rollout: RolloutArtifact,
    ) -> Dict[str, object]:
        return {
            "reference_step_count": rollout.step_count,
            **self.reward_reference_stats_for_start_state(
                rollout.start_state_ref,
                step_count=rollout.step_count,
            ),
        }

    def reward_reference_stats_for_start_state(
        self,
        start_state_ref,
        *,
        step_count: Optional[int] = None,
    ) -> Dict[str, object]:
        same_seed = self._matching_rollouts(seed_id=start_state_ref.seed_id)
        same_seed_same_step = (
            self._matching_rollouts(seed_id=start_state_ref.seed_id, step_count=step_count)
            if step_count is not None
            else []
        )
        result: Dict[str, object] = {
            "seed_id": start_state_ref.seed_id,
            "same_seed_any_steps": self._reward_distribution_summary(same_seed),
            "same_seed_same_step_count": (
                self._reward_distribution_summary(same_seed_same_step)
                if step_count is not None
                else None
            ),
        }
        checkpoint_id = start_state_ref.checkpoint_id
        if checkpoint_id is not None:
            same_checkpoint = self._matching_rollouts(start_checkpoint_id=checkpoint_id)
            same_checkpoint_same_step = (
                self._matching_rollouts(
                    start_checkpoint_id=checkpoint_id,
                    step_count=step_count,
                )
                if step_count is not None
                else []
            )
            result.update(
                {
                    "checkpoint_id": checkpoint_id,
                    "same_checkpoint_any_steps": self._reward_distribution_summary(
                        same_checkpoint
                    ),
                    "same_checkpoint_same_step_count": (
                        self._reward_distribution_summary(same_checkpoint_same_step)
                        if step_count is not None
                        else None
                    ),
                }
            )
        return result

    def reward_reference_stats_for_seed(
        self,
        seed_id: str,
        *,
        step_count: Optional[int] = None,
    ) -> Dict[str, object]:
        same_seed = self._matching_rollouts(seed_id=seed_id)
        same_seed_same_step = (
            self._matching_rollouts(seed_id=seed_id, step_count=step_count)
            if step_count is not None
            else []
        )
        return {
            "seed_id": seed_id,
            "reference_step_count": step_count,
            "same_seed_any_steps": self._reward_distribution_summary(same_seed),
            "same_seed_same_step_count": (
                self._reward_distribution_summary(same_seed_same_step)
                if step_count is not None
                else None
            ),
        }

    def _matching_rollouts(
        self,
        *,
        seed_id: Optional[str] = None,
        start_checkpoint_id: Optional[str] = None,
        step_count: Optional[int] = None,
    ) -> list[RolloutArtifact]:
        matches: list[RolloutArtifact] = []
        for rollout in self.rollouts.values():
            if seed_id is not None and rollout.seed_id != seed_id:
                continue
            if (
                start_checkpoint_id is not None
                and rollout.start_state_ref.checkpoint_id != start_checkpoint_id
            ):
                continue
            if step_count is not None and rollout.step_count != step_count:
                continue
            matches.append(rollout)
        return matches

    def _reward_distribution_summary(
        self,
        rollouts: Sequence[RolloutArtifact],
    ) -> Optional[Dict[str, object]]:
        if not rollouts:
            return None
        rewards = sorted(float(rollout.total_reward) for rollout in rollouts)
        step_counts = sorted(int(rollout.step_count) for rollout in rollouts)
        return {
            "count": len(rollouts),
            "mean_reward": round(mean(rewards), 4),
            "median_reward": round(self._percentile(rewards, 0.5), 4),
            "p90_reward": round(self._percentile(rewards, 0.9), 4),
            "min_reward": rewards[0],
            "max_reward": rewards[-1],
            "mean_step_count": round(mean(step_counts), 2),
            "min_step_count": step_counts[0],
            "max_step_count": step_counts[-1],
            "rollout_ids": sorted(rollout.rollout_id for rollout in rollouts),
        }

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        position = max(0.0, min(1.0, percentile)) * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        if lower == upper:
            return float(values[lower])
        weight = position - lower
        return float(values[lower] * (1.0 - weight) + values[upper] * weight)

    def _record_branch_cohort(self, rollout: RolloutArtifact) -> None:
        cohort_id = rollout.metadata.get("cohort_id")
        if not isinstance(cohort_id, str) or not cohort_id:
            return
        cohort = self.branch_cohorts.get(cohort_id)
        if cohort is None:
            cohort = BranchCohort(
                cohort_id=cohort_id,
                candidate_id=rollout.candidate_id,
                start_state_ref=rollout.start_state_ref,
                intent=str(rollout.metadata.get("intent", "")),
                target_waypoints=tuple(rollout.metadata.get("target_waypoints", ()) or ()),
                planned_trials=int(rollout.metadata.get("cohort_size", 1) or 1),
                rollout_ids=[rollout.rollout_id],
                metadata={
                    "query_ids": [rollout.metadata.get("query_id")],
                },
            )
            self.branch_cohorts[cohort_id] = cohort
            return
        if rollout.rollout_id not in cohort.rollout_ids:
            cohort.rollout_ids.append(rollout.rollout_id)
        query_id = rollout.metadata.get("query_id")
        if query_id is not None:
            query_ids = cohort.metadata.setdefault("query_ids", [])
            if isinstance(query_ids, list) and query_id not in query_ids:
                query_ids.append(query_id)

    def _refresh_comparative_view(self, cohort_id: str) -> None:
        cohort = self.branch_cohorts.get(cohort_id)
        if cohort is None:
            return
        rollout_ids = tuple(sorted(cohort.rollout_ids))
        if not rollout_ids:
            return
        waypoint_sets = []
        for rollout_id in rollout_ids:
            label_view = self.latest_label_view_for_rollout(rollout_id)
            waypoint_sets.append(set(label_view.completed_waypoint_types) if label_view is not None else set())
        union_waypoints = set().union(*waypoint_sets) if waypoint_sets else set()
        common_waypoints = (
            set(waypoint_sets[0]).intersection(*waypoint_sets[1:])
            if waypoint_sets
            else set()
        )
        divergent_waypoints = union_waypoints - common_waypoints
        expected = set(cohort.target_waypoints) or set(self.active_search_targets(limit=4))
        bottleneck_waypoints = tuple(sorted(expected - union_waypoints))[:4]

        def _score_rollout(rollout_id: str) -> tuple[float, float]:
            report = self.verification_for_rollout(rollout_id)
            rollout = self.rollouts[rollout_id]
            verified_depth = float(report.verified_depth) if report is not None else 0.0
            return verified_depth, rollout.total_reward

        best_rollout_id = max(rollout_ids, key=_score_rollout, default=None)
        worst_rollout_id = min(rollout_ids, key=_score_rollout, default=None)
        preferred_action_patterns, discouraged_action_patterns = self._compare_action_patterns(
            rollout_ids=rollout_ids,
            best_rollout_id=best_rollout_id,
            worst_rollout_id=worst_rollout_id,
        )
        loop_risk = 0.0
        if cohort.start_state_ref.kind == "checkpoint" and cohort.start_state_ref.checkpoint is not None:
            loop_risk = self.checkpoint_loop_risk(cohort.start_state_ref.checkpoint)
        dependency_hints = self._dependency_hints(
            cohort=cohort,
            common_waypoints=tuple(sorted(common_waypoints)),
            bottleneck_waypoints=bottleneck_waypoints,
            preferred_action_patterns=preferred_action_patterns,
        )
        comparative_view = ComparativeLabelView(
            comparative_view_id=f"cmp_{cohort_id}",
            cohort_id=cohort.cohort_id,
            candidate_id=cohort.candidate_id,
            start_state_ref=cohort.start_state_ref,
            rollout_ids=rollout_ids,
            common_waypoints=tuple(sorted(common_waypoints)),
            divergent_waypoints=tuple(sorted(divergent_waypoints)),
            bottleneck_waypoints=bottleneck_waypoints,
            preferred_action_patterns=preferred_action_patterns,
            discouraged_action_patterns=discouraged_action_patterns,
            dependency_hints=dependency_hints,
            loop_risk=loop_risk,
            best_rollout_id=best_rollout_id,
            worst_rollout_id=worst_rollout_id,
            notes=[
                f"planned_trials={cohort.planned_trials}",
                f"actual_trials={cohort.actual_trials}",
            ],
            metrics={
                "trial_count": len(rollout_ids),
                "completed_union_count": len(union_waypoints),
                "reward_span": (
                    max(self.rollouts[rollout_id].total_reward for rollout_id in rollout_ids)
                    - min(self.rollouts[rollout_id].total_reward for rollout_id in rollout_ids)
                ),
                "loop_risk": loop_risk,
            },
        )
        self.comparative_views[comparative_view.comparative_view_id] = comparative_view
        self.cohort_to_comparative_view[cohort_id] = comparative_view.comparative_view_id

    def _update_potential_estimate(
        self,
        rollout: RolloutArtifact,
        label_view: Optional[ScannerLabelView],
        verification_report: Optional[VerificationReport],
    ) -> None:
        if rollout.start_state_ref.kind != "checkpoint" or rollout.start_state_ref.checkpoint is None:
            return
        checkpoint = rollout.start_state_ref.checkpoint
        estimate_key = f"checkpoint:{checkpoint.checkpoint_id}"
        completed = set(label_view.completed_waypoint_types) if label_view is not None else set()
        bottlenecks = tuple(
            sorted(set(self.active_search_targets(limit=4)) - completed)
        )[:4]
        novelty = label_view.novelty_score if label_view is not None else 0.0
        learnability = (
            label_view.learnability_by_plugin.get("prompt", 0.0)
            if label_view is not None
            else 0.0
        )
        loop_risk = self.checkpoint_loop_risk(checkpoint)
        verified_bonus = (
            float(verification_report.verified_depth)
            * self.scoring.potential_estimate.verified_depth_weight
            if verification_report is not None
            else 0.0
        )
        achieved_value = rollout.total_reward
        potential_value = (
            achieved_value
            + novelty * self.scoring.potential_estimate.novelty_weight
            + learnability * self.scoring.potential_estimate.learnability_weight
            + verified_bonus
        )
        existing = self.potential_estimates.get(estimate_key)
        supporting_rollout_ids = [rollout.rollout_id]
        notes = [f"source_rollout={rollout.rollout_id}"]
        if existing is not None:
            supporting_rollout_ids = list(existing.supporting_rollout_ids)
            if rollout.rollout_id not in supporting_rollout_ids:
                supporting_rollout_ids.append(rollout.rollout_id)
            achieved_value = max(existing.achieved_value, achieved_value)
            potential_value = max(existing.potential_value, potential_value)
            loop_risk = max(existing.loop_risk, loop_risk)
            bottlenecks = tuple(sorted(set(existing.bottleneck_waypoints).union(bottlenecks)))[:4]
            notes = list(existing.notes)
            if f"source_rollout={rollout.rollout_id}" not in notes:
                notes.append(f"source_rollout={rollout.rollout_id}")
        self.potential_estimates[estimate_key] = PotentialEstimate(
            estimate_id=f"estimate_{checkpoint.checkpoint_id}",
            target_kind="checkpoint",
            target_id=checkpoint.checkpoint_id,
            achieved_value=achieved_value,
            potential_value=potential_value,
            loop_risk=loop_risk,
            bottleneck_waypoints=bottlenecks,
            supporting_rollout_ids=tuple(sorted(supporting_rollout_ids)),
            notes=notes,
        )

    def _update_trajectory_frontier(
        self,
        rollout: RolloutArtifact,
        label_view: Optional[ScannerLabelView],
        verification_report: Optional[VerificationReport],
    ) -> None:
        completed = set(label_view.completed_waypoint_types) if label_view is not None else set()
        bottlenecks = tuple(
            sorted(set(self.active_search_targets(limit=4)) - completed)
        )[:4]
        potential_value = rollout.total_reward
        if rollout.start_state_ref.kind == "checkpoint" and rollout.start_state_ref.checkpoint is not None:
            estimate = self.potential_estimate_for_checkpoint(rollout.start_state_ref.checkpoint.checkpoint_id)
            if estimate is not None:
                potential_value = max(
                    potential_value,
                    (
                        estimate.potential_value
                        - estimate.achieved_value
                    )
                    * self.scoring.trajectory_frontier.checkpoint_potential_gap_weight
                    + estimate.achieved_value
                    - estimate.loop_risk
                    * self.scoring.trajectory_frontier.checkpoint_loop_penalty_weight,
                )
        if label_view is not None:
            potential_value += (
                label_view.novelty_score
                * self.scoring.trajectory_frontier.novelty_weight
            )
        if verification_report is not None:
            potential_value += (
                verification_report.verified_depth
                * self.scoring.trajectory_frontier.verified_depth_weight
            )
        selection_score = (
            rollout.total_reward * self.scoring.trajectory_frontier.achieved_value_weight
            + potential_value
        )
        self.trajectory_frontier[rollout.rollout_id] = TrajectoryFrontierEntry(
            frontier_entry_id=f"traj_{rollout.rollout_id}",
            rollout_id=rollout.rollout_id,
            candidate_id=rollout.candidate_id,
            start_state_ref=rollout.start_state_ref,
            achieved_value=rollout.total_reward,
            potential_value=potential_value,
            stall_step=rollout.step_count,
            bottleneck_waypoints=bottlenecks,
            supporting_checkpoint_ids=tuple(
                checkpoint.checkpoint_id for checkpoint in rollout.checkpoints
            ),
            selection_score=selection_score,
            notes=[
                f"status={rollout.status}",
                f"is_resumed={rollout.is_resumed}",
            ],
        )
        retained = sorted(
            self.trajectory_frontier.values(),
            key=lambda entry: (
                entry.selection_score,
                entry.achieved_value,
                entry.potential_value,
            ),
            reverse=True,
        )[: self.trajectory_frontier_size]
        self.trajectory_frontier = {
            entry.rollout_id: entry for entry in retained
        }

    def _checkpoint_dominates(self, lhs: CheckpointRef, rhs: CheckpointRef) -> bool:
        lhs_summary = lhs.state_summary
        rhs_summary = rhs.state_summary
        lhs_markers = set(lhs_summary.progress_markers)
        rhs_markers = set(rhs_summary.progress_markers)
        if not rhs_markers.issubset(lhs_markers):
            return False
        rhs_resources = {
            key: value
            for key, value in rhs_summary.resources.items()
            if isinstance(value, (int, float))
        }
        lhs_resources = {
            key: value
            for key, value in lhs_summary.resources.items()
            if isinstance(value, (int, float))
        }
        resources_cover = all(lhs_resources.get(key, 0) >= value for key, value in rhs_resources.items())
        if not resources_cover:
            return False
        if lhs.total_reward < rhs.total_reward:
            return False
        return (
            lhs.total_reward > rhs.total_reward
            or lhs_markers != rhs_markers
            or lhs_summary.dedupe_signature != rhs_summary.dedupe_signature
        )

    def _compare_action_patterns(
        self,
        *,
        rollout_ids: Sequence[str],
        best_rollout_id: Optional[str],
        worst_rollout_id: Optional[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if best_rollout_id is None:
            return (), ()
        best_actions = self._action_signatures_for_rollout(best_rollout_id)
        if worst_rollout_id is None or worst_rollout_id == best_rollout_id:
            preferred = tuple(best_actions[:3])
            return preferred, ()
        worst_actions = self._action_signatures_for_rollout(worst_rollout_id)
        preferred = tuple(action for action in best_actions if action not in worst_actions)[:3]
        discouraged = tuple(action for action in worst_actions if action not in best_actions)[:3]
        if not preferred and best_actions:
            preferred = tuple(best_actions[:2])
        return preferred, discouraged

    def _action_signatures_for_rollout(self, rollout_id: str) -> list[str]:
        rollout = self.rollouts[rollout_id]
        trace = rollout.raw_trace if isinstance(rollout.raw_trace, dict) else {}
        event_history = trace.get("event_history", [])
        signatures: list[str] = []
        if not isinstance(event_history, list):
            return signatures
        for event in event_history:
            if not isinstance(event, dict):
                continue
            step_idx = int(event.get("step_idx", 0))
            actions = event.get("actions", [])
            if not isinstance(actions, list):
                continue
            for action in actions:
                if isinstance(action, str):
                    signatures.append(f"step{step_idx}:{action}")
        return signatures

    def _dependency_hints(
        self,
        *,
        cohort: BranchCohort,
        common_waypoints: tuple[str, ...],
        bottleneck_waypoints: tuple[str, ...],
        preferred_action_patterns: tuple[str, ...],
    ) -> tuple[str, ...]:
        hints: list[str] = []
        start_checkpoint = cohort.start_state_ref.checkpoint
        if start_checkpoint is not None:
            start_summary = start_checkpoint.state_summary
            if start_summary.progress_markers:
                hints.append(
                    "avoid re-solving already-satisfied progress markers: "
                    + ", ".join(start_summary.progress_markers[:3])
                )
            if start_summary.resources:
                resource_keys = [str(key) for key, value in sorted(start_summary.resources.items()) if value]
                if resource_keys:
                    hints.append(
                        "treat current resources as available context: "
                        + ", ".join(resource_keys[:3])
                    )
        if bottleneck_waypoints:
            hints.append(
                "branch repeatedly stalled before: " + ", ".join(bottleneck_waypoints[:3])
            )
        if preferred_action_patterns:
            hints.append(
                "better siblings used: " + ", ".join(preferred_action_patterns[:2])
            )
        if common_waypoints:
            hints.append(
                "all siblings repeated: " + ", ".join(common_waypoints[:2])
            )
        deduped: list[str] = []
        for hint in hints:
            if hint not in deduped:
                deduped.append(hint)
        return tuple(deduped[:4])
