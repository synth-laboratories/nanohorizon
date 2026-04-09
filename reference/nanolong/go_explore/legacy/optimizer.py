"""Executable local optimizer loop for Go-Explore + Prompt."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Dict, Sequence
from uuid import uuid4

from .config import GoExploreConfig
from .event_schemas import (
    GoExploreJobCompletedPayload,
    GoExplorePluginUpdatedPayload,
    GoExploreQuerySelectionPayload,
    GoExploreRolloutBatchPayload,
    GoExploreScanCompletedPayload,
    GoExploreVerifierCompletedPayload,
    GoExploreWaypointUpdatePayload,
)
from .models import (
    BatchAnalysis,
    BatchCheckpointBranchDecision,
    GoExploreResult,
    PromptCandidate,
    SearchQuery,
    StartStateRef,
)
from .plugins import DeterministicGoExplorePlugin, GoExplorePlugin
from .plugins.prompt import PromptCandidateMutator, PromptPlugin, RlmPromptCandidateMutator
from .proposers import RlmMcpBackend
from .reasoners import (
    ExplorationReasoner,
    RlmMcpExplorationReasoner,
)
from .runtime import SearchRuntime
from .search.archive import InMemoryArchive
from .verifiers import build_verifier_registry


class GoExploreCancelledError(RuntimeError):
    """Raised when a Go-Explore job is cancelled cooperatively."""


class GoExploreOptimizer:
    """Local algorithm core for Go-Explore + Prompt v0."""

    def __init__(
        self,
        config: GoExploreConfig,
        *,
        runtime: SearchRuntime,
        event_callback: Callable[[str, Dict[str, Any]], Awaitable[None] | None] | None = None,
        cancel_check: Callable[[], Awaitable[bool] | bool] | None = None,
        proposer_backend: RlmMcpBackend | None = None,
        reasoner: ExplorationReasoner | None = None,
        prompt_mutator: PromptCandidateMutator | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.archive = InMemoryArchive(
            expected_waypoints=config.scanner.expected_waypoints,
            trajectory_frontier_size=config.budget.trajectory_frontier_size,
            scoring=config.scoring,
        )
        self.reasoner = reasoner or self._build_reasoner(
            proposer_backend=proposer_backend,
        )
        resolved_prompt_mutator = prompt_mutator
        if resolved_prompt_mutator is None and reasoner is None:
            resolved_prompt_mutator = self._build_prompt_mutator(
                proposer_backend=proposer_backend
            )
        self.verifiers = build_verifier_registry(config.runtime.env_kind)
        self.plugin = self._build_plugin(
            prompt_mutator=resolved_prompt_mutator,
        )
        self.archive.register_candidates(self.plugin.all_candidates)
        self._events: list[Dict[str, object]] = []
        self._event_callback = event_callback
        self._cancel_check = cancel_check

    async def optimize(self) -> GoExploreResult:
        is_healthy = await self.runtime.healthcheck()
        if not is_healthy:
            raise RuntimeError(
                f"{self.config.runtime.env_kind} runtime healthcheck failed"
            )

        current_queries = self._build_initial_queries(
            self.plugin.frontier,
            iteration_idx=0,
        )
        for iteration_idx in range(self.config.budget.max_iterations):
            await self._ensure_not_cancelled()
            frontier = self.plugin.frontier
            await self._record_query_selection_events(
                iteration_idx,
                queries=current_queries,
            )
            completed_rollouts = await self._execute_queries(
                current_queries,
                frontier_by_id={item.candidate_id: item for item in frontier},
            )
            await self._ensure_not_cancelled()
            batch_analysis = await asyncio.to_thread(
                self.reasoner.analyze_batch,
                archive=self.archive,
                completed_rollouts=completed_rollouts,
                frontier=frontier,
                iteration_idx=iteration_idx,
                budget_context=self._batch_budget_context(),
            )
            self._ingest_batch_analysis(
                batch_analysis=batch_analysis,
                completed_rollouts=completed_rollouts,
            )
            await self._record_waypoint_update_event(
                iteration_idx,
                phase="after_batch",
                refresh_result={
                    "proposed_waypoint_types": [
                        item.waypoint_type for item in batch_analysis.waypoint_definitions
                    ],
                    "active_search_targets": list(batch_analysis.active_search_targets),
                    "waypoint_counts": self.archive.waypoint_counts(),
                    "relabeled_archive": False,
                },
            )
            await self._record_scan_event(iteration_idx)
            await self._record_verifier_event(iteration_idx)
            await self._record_rollout_batch_event(
                iteration_idx,
                rollout_count=len(completed_rollouts),
                resumed_rollout_count=sum(
                    1 for rollout in completed_rollouts if rollout.is_resumed
                ),
                candidate_ids=[item.candidate_id for item in current_queries],
            )

            archive_summary = self.archive.build_archive_summary()
            plugin_result = await asyncio.to_thread(
                self.plugin.apply_iteration,
                batch_analysis=batch_analysis,
                archive=self.archive,
                archive_summary=archive_summary,
            )
            await self._record_plugin_event(
                iteration_idx,
                plugin_kind=self.plugin.plugin_kind,
                frontier_ids=[candidate.candidate_id for candidate in plugin_result.frontier],
                mutated_candidate_ids=[
                    candidate.candidate_id for candidate in plugin_result.new_candidates
                ],
            )
            if iteration_idx + 1 < self.config.budget.max_iterations:
                current_queries = self._build_next_iteration_queries(
                    batch_analysis,
                    plugin_result=plugin_result,
                    frontier=self.plugin.frontier,
                    iteration_idx=iteration_idx + 1,
                )

        final_summary = self.archive.build_archive_summary()
        final_evaluations = self.archive.evaluate_candidates(
            [candidate.candidate_id for candidate in self.plugin.all_candidates]
        )
        best_candidate_id = self.archive.best_candidate_id(
            [candidate.candidate_id for candidate in self.plugin.frontier]
        )
        await self._record_completion_event(
            best_candidate_id=best_candidate_id,
            frontier_size=len(self.plugin.frontier),
            rollout_count=final_summary.rollout_count,
            checkpoint_count=final_summary.checkpoint_count,
        )
        return GoExploreResult(
            system_id=self.config.system_id,
            best_candidate_id=best_candidate_id,
            frontier=self.plugin.frontier,
            evaluations=final_evaluations,
            archive_summary=final_summary,
            all_candidates=self.plugin.all_candidates,
            checkpoint_refs=self.archive.checkpoint_refs(),
            events=list(self._events),
            trajectory_frontier=self.archive.trajectory_frontier_artifacts(),
            branch_cohorts=self.archive.branch_cohort_artifacts(),
            comparative_views=self.archive.comparative_view_artifacts(),
            potential_estimates=self.archive.potential_estimate_artifacts(),
            rollouts=self.archive.rollout_artifacts(),
            label_views=self.archive.label_view_artifacts(),
            verification_reports=self.archive.verification_report_artifacts(),
        )

    def _build_reasoner(
        self,
        *,
        proposer_backend: RlmMcpBackend | None,
    ) -> ExplorationReasoner:
        if self.config.proposer.kind == "rlm_mcp":
            return RlmMcpExplorationReasoner(
                self.config.proposer,
                scanner_config=self.config.scanner,
                runtime_config=self.config.runtime,
                backend=proposer_backend,
            )
        raise ValueError(
            f"Unsupported Go-Explore proposer kind: {self.config.proposer.kind}"
        )

    def _build_plugin(
        self,
        *,
        prompt_mutator: PromptCandidateMutator | None,
    ) -> GoExplorePlugin:
        if self.config.plugin_kind == "prompt":
            return PromptPlugin(
                config=self.config.prompt_plugin,
                seed_candidates=self.config.initial_candidates,
                mutator=prompt_mutator,
            )
        return DeterministicGoExplorePlugin(
            plugin_kind=self.config.plugin_kind,
            seed_candidates=self.config.initial_candidates,
        )

    def _build_prompt_mutator(
        self,
        *,
        proposer_backend: RlmMcpBackend | None,
    ) -> PromptCandidateMutator | None:
        if self.config.plugin_kind != "prompt":
            return None
        if self.config.prompt_plugin.max_mutations_per_iteration <= 0:
            return None
        if self.config.proposer.kind == "rlm_mcp":
            return RlmPromptCandidateMutator(
                self.config.proposer,
                runtime_config=self.config.runtime,
                backend=proposer_backend,
            )
        raise ValueError(
            f"Unsupported Go-Explore proposer kind: {self.config.proposer.kind}"
        )

    def _build_initial_queries(
        self,
        frontier: Sequence[PromptCandidate],
        *,
        iteration_idx: int,
    ) -> list[SearchQuery]:
        queries: list[SearchQuery] = []
        if not frontier:
            return queries
        frontier_cycle = list(frontier)
        seed_pool = list(self.config.seed_pool)
        query_budget = self.config.budget.fresh_queries_per_iteration
        for query_idx in range(query_budget):
            candidate = frontier_cycle[query_idx % len(frontier_cycle)]
            seed = seed_pool[query_idx % len(seed_pool)]
            query = SearchQuery(
                query_id=f"{self.config.system_id}_iter{iteration_idx}_fresh{query_idx}_{uuid4().hex[:6]}",
                candidate_id=candidate.candidate_id,
                start_state_ref=StartStateRef.from_seed(seed),
                intent="fresh_seed_exploration",
                target_waypoints=(),
                budget_hint_steps=self.config.budget.segment_steps,
                metadata={
                    "iteration_idx": iteration_idx,
                },
            )
            queries.append(query)
        return queries

    def _build_queries_from_batch_analysis(
        self,
        batch_analysis: BatchAnalysis,
        *,
        frontier: Sequence[PromptCandidate],
        iteration_idx: int,
    ) -> list[SearchQuery]:
        selected_candidate_ids = {candidate.candidate_id for candidate in frontier}
        queries: list[SearchQuery] = []
        fresh_budget = self.config.budget.fresh_queries_per_iteration
        resumed_budget = self.config.budget.resumed_queries_per_iteration
        for query_idx, decision in enumerate(batch_analysis.fresh_seed_plan[:fresh_budget]):
            if decision.candidate_id not in selected_candidate_ids:
                continue
            seed = next(
                (item for item in self.config.seed_pool if item.seed_id == decision.seed_id),
                None,
            )
            if seed is None:
                continue
            queries.append(
                SearchQuery(
                    query_id=(
                        f"{self.config.system_id}_iter{iteration_idx}_fresh{query_idx}_{uuid4().hex[:6]}"
                    ),
                    candidate_id=decision.candidate_id,
                    start_state_ref=StartStateRef.from_seed(seed),
                    intent="fresh_seed_exploration",
                    target_waypoints=decision.target_waypoints,
                    budget_hint_steps=self.config.budget.segment_steps,
                    priority=decision.priority,
                    metadata={
                        "iteration_idx": iteration_idx,
                        "planner_reason": decision.reason,
                        **dict(decision.metadata),
                    },
                )
            )
        for query_idx, decision in enumerate(batch_analysis.checkpoint_branch_plan[:resumed_budget]):
            if decision.candidate_id not in selected_candidate_ids:
                continue
            checkpoint = self.archive.checkpoints.get(decision.checkpoint_id)
            if checkpoint is None:
                continue
            cohort_id = f"{self.config.system_id}_iter{iteration_idx}_cohort_{query_idx}_{uuid4().hex[:6]}"
            for trial_idx in range(self.config.budget.local_trials_per_start_state):
                queries.append(
                    SearchQuery(
                        query_id=(
                            f"{self.config.system_id}_iter{iteration_idx}_resume{query_idx}"
                            f"_trial{trial_idx}_{uuid4().hex[:6]}"
                        ),
                        candidate_id=decision.candidate_id,
                        start_state_ref=StartStateRef.from_checkpoint(checkpoint),
                        intent="checkpoint_branch_exploration",
                        target_waypoints=decision.target_waypoints,
                        budget_hint_steps=self.config.budget.segment_steps,
                        cohort_id=cohort_id,
                        cohort_size=self.config.budget.local_trials_per_start_state,
                        cohort_trial_index=trial_idx,
                        priority=decision.priority,
                        metadata={
                            "iteration_idx": iteration_idx,
                            "parent_rollout_id": checkpoint.rollout_id,
                            "planner_reason": decision.reason,
                            **dict(decision.metadata),
                        },
                    )
                )
        return queries

    def _build_next_iteration_queries(
        self,
        batch_analysis: BatchAnalysis,
        *,
        plugin_result: Any,
        frontier: Sequence[PromptCandidate],
        iteration_idx: int,
    ) -> list[SearchQuery]:
        queries = self._build_queries_from_batch_analysis(
            batch_analysis,
            frontier=frontier,
            iteration_idx=iteration_idx,
        )
        plugin_queries = self._build_plugin_exploration_queries(
            plugin_result=plugin_result,
            existing_queries=queries,
            iteration_idx=iteration_idx,
        )
        plugin_resume_queries = self._build_plugin_resume_queries(
            plugin_result=plugin_result,
            existing_queries=queries,
            iteration_idx=iteration_idx,
        )
        if plugin_queries:
            queries.extend(plugin_queries)
        if plugin_resume_queries:
            queries.extend(plugin_resume_queries)
        forced_candidate_ids = self._forced_mutation_candidate_ids(
            plugin_result=plugin_result,
        )
        if forced_candidate_ids:
            scheduled_candidate_ids = {query.candidate_id for query in queries}
            missing_forced = [
                candidate_id for candidate_id in forced_candidate_ids if candidate_id not in scheduled_candidate_ids
            ]
            if missing_forced:
                raise RuntimeError(
                    "prompt plugin produced mutated candidates that were not scheduled for evaluation: "
                    + ", ".join(missing_forced)
                )
        if queries:
            return queries
        raise RuntimeError(
            "batch analysis and plugin iteration produced no executable next queries; "
            "refusing to fall back to fresh-seed initialization. Check fresh_seed_plan, "
            "checkpoint_branch_plan, plugin exploration slots, and candidate/checkpoint ids."
        )

    def _build_plugin_resume_queries(
        self,
        *,
        plugin_result: Any,
        existing_queries: Sequence[SearchQuery],
        iteration_idx: int,
    ) -> list[SearchQuery]:
        if self.config.plugin_kind != "prompt":
            return []
        if self.config.budget.resumed_queries_per_iteration <= 0:
            return []
        new_candidates = list(getattr(plugin_result, "new_candidates", []) or [])
        if not new_candidates:
            return []
        max_forced = self.config.prompt_plugin.candidate_exploration_slots
        if max_forced <= 0:
            return []
        resume_anchors = [
            query
            for query in existing_queries
            if query.start_state_ref.kind == "checkpoint"
            and query.start_state_ref.checkpoint is not None
        ]
        if not resume_anchors:
            return []
        forced_candidates = new_candidates[:max_forced]
        queries: list[SearchQuery] = []
        for query_idx, candidate in enumerate(forced_candidates):
            anchor = resume_anchors[query_idx % len(resume_anchors)]
            checkpoint = anchor.start_state_ref.checkpoint
            if checkpoint is None:
                continue
            candidate_targets = tuple(
                str(item)
                for item in candidate.metadata.get("target_waypoints", [])
                if str(item)
            )
            queries.append(
                SearchQuery(
                    query_id=(
                        f"{self.config.system_id}_iter{iteration_idx}_pluginresume{query_idx}_"
                        f"{uuid4().hex[:6]}"
                    ),
                    candidate_id=candidate.candidate_id,
                    start_state_ref=StartStateRef.from_checkpoint(checkpoint),
                    intent="plugin_candidate_branch_exploration",
                    target_waypoints=candidate_targets or anchor.target_waypoints,
                    budget_hint_steps=self.config.budget.segment_steps,
                    cohort_id=(
                        f"{self.config.system_id}_iter{iteration_idx}_plugincohort_{query_idx}_"
                        f"{uuid4().hex[:6]}"
                    ),
                    cohort_size=1,
                    cohort_trial_index=0,
                    priority=max(anchor.priority, 1.0),
                    metadata={
                        "iteration_idx": iteration_idx,
                        "parent_rollout_id": checkpoint.rollout_id,
                        "planner_reason": (
                            "prompt plugin requested checkpoint branch evaluation for new candidate"
                        ),
                        "source_query_id": anchor.query_id,
                        "source_candidate_id": anchor.candidate_id,
                    },
                )
            )
        return queries

    def _build_plugin_exploration_queries(
        self,
        *,
        plugin_result: Any,
        existing_queries: Sequence[SearchQuery],
        iteration_idx: int,
    ) -> list[SearchQuery]:
        if self.config.plugin_kind != "prompt":
            return []
        new_candidates = list(getattr(plugin_result, "new_candidates", []) or [])
        if not new_candidates:
            return []
        max_forced = self.config.prompt_plugin.candidate_exploration_slots
        if max_forced <= 0:
            return []
        existing_candidate_ids = {query.candidate_id for query in existing_queries}
        outstanding_candidates = [
            candidate
            for candidate in new_candidates
            if candidate.candidate_id not in existing_candidate_ids
        ]
        if not outstanding_candidates:
            return []
        existing_fresh_queries = [
            query for query in existing_queries if query.start_state_ref.kind == "task_seed"
        ]
        available_fresh_budget = self.config.budget.fresh_queries_per_iteration - len(existing_fresh_queries)
        forced_candidates = outstanding_candidates[:max_forced]
        if available_fresh_budget <= 0:
            return self._replacement_plugin_exploration_queries(
                forced_candidates=forced_candidates,
                existing_fresh_queries=existing_fresh_queries,
                iteration_idx=iteration_idx,
            )
        forced_candidates = forced_candidates[: min(len(forced_candidates), available_fresh_budget)]
        existing_candidate_ids = {query.candidate_id for query in existing_queries}
        queries: list[SearchQuery] = []
        for query_idx, candidate in enumerate(forced_candidates):
            if candidate.candidate_id in existing_candidate_ids:
                continue
            seed = self.config.seed_pool[query_idx % len(self.config.seed_pool)]
            candidate_targets = tuple(
                str(item).strip()
                for item in candidate.metadata.get("target_waypoints", [])
                if str(item).strip()
            )
            queries.append(
                SearchQuery(
                    query_id=(
                        f"{self.config.system_id}_iter{iteration_idx}_pluginfresh{query_idx}_"
                        f"{uuid4().hex[:6]}"
                    ),
                    candidate_id=candidate.candidate_id,
                    start_state_ref=StartStateRef.from_seed(seed),
                    intent="plugin_candidate_exploration",
                    target_waypoints=candidate_targets,
                    budget_hint_steps=self.config.budget.segment_steps,
                    metadata={
                        "iteration_idx": iteration_idx,
                        "planner_reason": "prompt plugin requested explicit exploration for new candidate",
                    },
                )
            )
        return queries

    def _replacement_plugin_exploration_queries(
        self,
        *,
        forced_candidates: Sequence[PromptCandidate],
        existing_fresh_queries: Sequence[SearchQuery],
        iteration_idx: int,
    ) -> list[SearchQuery]:
        replaceable_count = min(
            len(forced_candidates),
            self.config.prompt_plugin.candidate_exploration_slots,
            len(existing_fresh_queries),
        )
        if replaceable_count <= 0:
            return []
        queries: list[SearchQuery] = []
        for query_idx, candidate in enumerate(list(forced_candidates)[:replaceable_count]):
            replaced_query = sorted(
                existing_fresh_queries,
                key=lambda item: item.priority,
            )[query_idx]
            candidate_targets = tuple(
                str(item).strip()
                for item in candidate.metadata.get("target_waypoints", [])
                if str(item).strip()
            )
            seed = (
                replaced_query.start_state_ref.seed
                if replaced_query.start_state_ref.seed is not None
                else self.config.seed_pool[query_idx % len(self.config.seed_pool)]
            )
            queries.append(
                SearchQuery(
                    query_id=(
                        f"{self.config.system_id}_iter{iteration_idx}_pluginfresh{query_idx}_"
                        f"{uuid4().hex[:6]}"
                    ),
                    candidate_id=candidate.candidate_id,
                    start_state_ref=StartStateRef.from_seed(seed),
                    intent="plugin_candidate_exploration",
                    target_waypoints=candidate_targets,
                    budget_hint_steps=self.config.budget.segment_steps,
                    priority=max(replaced_query.priority, 1.0),
                    metadata={
                        "iteration_idx": iteration_idx,
                        "planner_reason": (
                            "prompt plugin reserved fresh evaluation slot for new candidate"
                        ),
                        "replaced_query_id": replaced_query.query_id,
                        "replaced_candidate_id": replaced_query.candidate_id,
                    },
                )
            )
        return queries

    def _forced_mutation_candidate_ids(
        self,
        *,
        plugin_result: Any,
    ) -> list[str]:
        if self.config.plugin_kind != "prompt":
            return []
        limit = self.config.prompt_plugin.candidate_exploration_slots
        if limit <= 0:
            return []
        new_candidates = list(getattr(plugin_result, "new_candidates", []) or [])
        return [candidate.candidate_id for candidate in new_candidates[:limit]]

    async def _execute_queries(
        self,
        queries: Sequence[SearchQuery],
        *,
        frontier_by_id: Dict[str, PromptCandidate],
    ) -> list[object]:
        if not queries:
            return []
        semaphore = asyncio.Semaphore(
            min(self.config.budget.max_concurrent_rollouts, len(queries))
        )

        async def _run_query(query: SearchQuery) -> tuple[SearchQuery, object]:
            async with semaphore:
                await self._ensure_not_cancelled()
                candidate = frontier_by_id[query.candidate_id]
                rollout = await self.runtime.run_query(query, candidate)
                return query, rollout

        results = await asyncio.gather(*[_run_query(query) for query in queries])
        rollouts = []
        for query, rollout in results:
            await self._ensure_not_cancelled()
            rollout.metadata.update(
                {
                    "query_id": query.query_id,
                    "intent": query.intent,
                    "target_waypoints": list(query.target_waypoints),
                    "cohort_id": query.cohort_id,
                    "cohort_size": query.cohort_size,
                    "cohort_trial_index": query.cohort_trial_index,
                    "priority": query.priority,
                    "verifier_profile": query.verifier_profile,
                }
            )
            rollouts.append(rollout)
        return rollouts

    def _batch_budget_context(self) -> Dict[str, Any]:
        return {
            "fresh_queries_per_iteration": self.config.budget.fresh_queries_per_iteration,
            "resumed_queries_per_iteration": self.config.budget.resumed_queries_per_iteration,
            "local_trials_per_start_state": self.config.budget.local_trials_per_start_state,
            "segment_steps": self.config.budget.segment_steps,
            "max_concurrent_rollouts": self.config.budget.max_concurrent_rollouts,
            "seed_pool": [seed.to_dict() for seed in self.config.seed_pool],
            "plugin_kind": self.config.plugin_kind,
        }

    def _ingest_batch_analysis(
        self,
        *,
        batch_analysis: BatchAnalysis,
        completed_rollouts: Sequence[Any],
    ) -> None:
        self.archive.register_waypoints(batch_analysis.waypoint_definitions)
        for rollout in completed_rollouts:
            label_view = batch_analysis.label_view_for_rollout(rollout.rollout_id)
            if label_view is None:
                raise RuntimeError(
                    "batch analysis missing label_view for rollout "
                    f"{rollout.rollout_id} in iteration {batch_analysis.iteration_idx}"
                )
            verification_report = self.verifiers.verify_rollout(
                rollout,
                label_view=label_view,
                verifier_profile=str(rollout.metadata.get("verifier_profile", "default")),
            )
            self.archive.ingest_rollout(
                rollout,
                label_view=label_view,
                verification_report=verification_report,
            )

    async def _record_scan_event(self, iteration_idx: int) -> None:
        payload = GoExploreScanCompletedPayload(
            iteration_idx=iteration_idx,
            label_view_count=len(self.archive.label_views),
            rollout_count=len(self.archive.rollouts),
            waypoint_count=len(self.archive.waypoint_counts()),
            verified_rollout_count=len(self.archive.verification_reports),
            max_verified_depth=max(
                (
                    report.verified_depth
                    for report in self.archive.verification_reports.values()
                ),
                default=0,
            ),
            comparative_view_count=len(self.archive.comparative_views),
            trajectory_frontier_count=len(self.archive.trajectory_frontier),
        )
        await self._emit_event(
            "learning.policy.go_explore.scan.completed",
            payload.model_dump(),
        )

    async def _record_verifier_event(self, iteration_idx: int) -> None:
        branch_ready_count = sum(
            1
            for report in self.archive.verification_reports.values()
            if report.branch_ready
        )
        payload = GoExploreVerifierCompletedPayload(
            iteration_idx=iteration_idx,
            verified_rollout_count=len(self.archive.verification_reports),
            max_verified_depth=max(
                (
                    report.verified_depth
                    for report in self.archive.verification_reports.values()
                ),
                default=0,
            ),
            branch_ready_count=branch_ready_count,
        )
        await self._emit_event(
            "learning.policy.go_explore.verifier.completed",
            payload.model_dump(),
        )

    async def _record_rollout_batch_event(
        self,
        iteration_idx: int,
        *,
        rollout_count: int,
        resumed_rollout_count: int,
        candidate_ids: list[str],
    ) -> None:
        payload = GoExploreRolloutBatchPayload(
            iteration_idx=iteration_idx,
            rollout_count=rollout_count,
            resumed_rollout_count=resumed_rollout_count,
            candidate_ids=candidate_ids,
        )
        await self._emit_event(
            "learning.policy.go_explore.rollout_batch.completed",
            payload.model_dump(),
        )

    async def _record_query_selection_event(
        self,
        iteration_idx: int,
        *,
        phase: str,
        queries: Sequence[SearchQuery],
    ) -> None:
        payload = GoExploreQuerySelectionPayload(
            iteration_idx=iteration_idx,
            phase=phase,
            query_count=len(queries),
            selected_queries=[
                {
                    "query_id": query.query_id,
                    "candidate_id": query.candidate_id,
                    "seed_id": query.start_state_ref.seed_id,
                    "start_kind": query.start_state_ref.kind,
                    "checkpoint_id": query.start_state_ref.checkpoint_id,
                    "intent": query.intent,
                    "target_waypoints": list(query.target_waypoints),
                    "cohort_id": query.cohort_id,
                    "cohort_trial_index": query.cohort_trial_index,
                    "forced_candidate_evaluation": bool(
                        query.metadata.get("forced_candidate_evaluation", False)
                    ),
                }
                for query in queries
            ],
        )
        await self._emit_event(
            "learning.policy.go_explore.query_selection.completed",
            payload.model_dump(),
        )

    async def _record_query_selection_events(
        self,
        iteration_idx: int,
        *,
        queries: Sequence[SearchQuery],
    ) -> None:
        await self._record_query_selection_event(
            iteration_idx,
            phase="batch",
            queries=queries,
        )
        fresh_queries = [query for query in queries if query.intent == "fresh_seed_exploration"]
        if fresh_queries:
            await self._record_query_selection_event(
                iteration_idx,
                phase="fresh",
                queries=fresh_queries,
            )
        resumed_queries = [
            query for query in queries if query.intent == "checkpoint_branch_exploration"
        ]
        if resumed_queries:
            await self._record_query_selection_event(
                iteration_idx,
                phase="resumed",
                queries=resumed_queries,
            )

    async def _record_waypoint_update_event(
        self,
        iteration_idx: int,
        *,
        phase: str,
        refresh_result: Dict[str, Any],
    ) -> None:
        payload = GoExploreWaypointUpdatePayload(
            iteration_idx=iteration_idx,
            phase=phase,
            proposed_waypoint_types=list(
                refresh_result.get("proposed_waypoint_types", [])
            ),
            active_search_targets=list(
                refresh_result.get("active_search_targets", [])
            ),
            waypoint_counts=dict(refresh_result.get("waypoint_counts", {})),
            relabeled_archive=bool(refresh_result.get("relabeled_archive", False)),
        )
        await self._emit_event(
            "learning.policy.go_explore.waypoints.updated",
            payload.model_dump(),
        )

    async def _record_plugin_event(
        self,
        iteration_idx: int,
        *,
        plugin_kind: str,
        frontier_ids: list[str],
        mutated_candidate_ids: list[str],
    ) -> None:
        payload = GoExplorePluginUpdatedPayload(
            iteration_idx=iteration_idx,
            plugin_kind=plugin_kind,
            frontier_ids=frontier_ids,
            mutated_candidate_ids=mutated_candidate_ids,
        )
        await self._emit_event(
            "learning.policy.go_explore.plugin.updated",
            payload.model_dump(),
        )

    async def _record_completion_event(
        self,
        *,
        best_candidate_id: str | None,
        frontier_size: int,
        rollout_count: int,
        checkpoint_count: int,
    ) -> None:
        payload = GoExploreJobCompletedPayload(
            best_candidate_id=best_candidate_id,
            frontier_size=frontier_size,
            rollout_count=rollout_count,
            checkpoint_count=checkpoint_count,
        )
        await self._emit_event(
            "learning.policy.go_explore.job.completed",
            payload.model_dump(),
        )

    async def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {"type": event_type, "payload": payload}
        self._events.append(event)
        if self._event_callback is None:
            return
        maybe_awaitable = self._event_callback(event_type, payload)
        if maybe_awaitable is not None:
            await maybe_awaitable

    async def _ensure_not_cancelled(self) -> None:
        if self._cancel_check is None:
            return
        cancelled = self._cancel_check()
        if isinstance(cancelled, Awaitable):
            cancelled = await cancelled
        if cancelled:
            raise GoExploreCancelledError("Go-Explore job cancelled")
