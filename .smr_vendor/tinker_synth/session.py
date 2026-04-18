"""Session-scoped Tinker training metering wrapper for SMR worker sandboxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from synth_budget import ProviderUsageAttribution
from synth_budget import ProviderUsageReport
from synth_budget import SessionBudget
from synth_budget import stable_usage_idempotency_key
from synth_budget.reporting import ProviderUsageReporter

DEFAULT_BUDGET_USD = Decimal("10.00")


@dataclass(slots=True)
class TinkerBudgetSession:
    run_id: str
    attribution: ProviderUsageAttribution | None = None
    project_id: str | None = None
    org_id: str | None = None
    task_id: str | None = None
    actor_id: str | None = None
    worker_id: str | None = None
    participant_session_id: str | None = None
    participant_role: str | None = None
    funding_source: str | None = None
    budget_usd: Decimal | float | int | str = DEFAULT_BUDGET_USD
    reporter: ProviderUsageReporter | None = None
    session_budget: SessionBudget = field(init=False)

    def __post_init__(self) -> None:
        (self.attribution or ProviderUsageAttribution.from_env()).apply_defaults(self)
        self.session_budget = SessionBudget(self.budget_usd)
        if self.reporter is None:
            self.reporter = ProviderUsageReporter()

    @classmethod
    def from_env(
        cls,
        *,
        budget_usd: Decimal | float | int | str = DEFAULT_BUDGET_USD,
        reporter: ProviderUsageReporter | None = None,
        funding_source: str | None = None,
        task_id: str | None = None,
        actor_id: str | None = None,
        worker_id: str | None = None,
        participant_session_id: str | None = None,
        participant_role: str | None = None,
    ) -> "TinkerBudgetSession":
        attribution = ProviderUsageAttribution.from_env()
        return cls(
            run_id=attribution.run_id or "",
            attribution=attribution,
            project_id=attribution.project_id,
            org_id=attribution.org_id,
            task_id=task_id or attribution.task_id,
            actor_id=actor_id or attribution.actor_id,
            worker_id=worker_id or attribution.worker_id,
            participant_session_id=participant_session_id
            or attribution.participant_session_id,
            participant_role=participant_role or attribution.participant_role,
            funding_source=funding_source,
            budget_usd=budget_usd,
            reporter=reporter,
        )

    def check_projected_cost(
        self,
        projected_cost_usd: Decimal | float | int | str,
        *,
        model: str | None = None,
        operation_kind: str = "training_job",
    ) -> Decimal:
        return self.session_budget.check_increment(
            projected_cost_usd,
            context=f"Tinker {operation_kind} {model or 'job'}",
        )

    def report_job(
        self,
        *,
        model: str | None,
        job_id: str | None,
        result_id: str | None = None,
        projected_cost_usd: Decimal | float | int | str | None = None,
        actual_cost_usd: Decimal | float | int | str | None = None,
        operation_kind: str = "training_job",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if projected_cost_usd is not None:
            self.check_projected_cost(
                projected_cost_usd, model=model, operation_kind=operation_kind
            )
        if actual_cost_usd is not None:
            self.session_budget.commit_increment(actual_cost_usd)
        report = ProviderUsageReport(
            provider="tinker",
            operation_kind=operation_kind,
            run_id=self.run_id,
            org_id=self.org_id,
            project_id=self.project_id,
            task_id=self.task_id,
            actor_id=self.actor_id,
            worker_id=self.worker_id,
            participant_session_id=self.participant_session_id,
            participant_role=self.participant_role,
            model=model,
            estimated_cost_usd=projected_cost_usd,
            actual_cost_usd=actual_cost_usd,
            quantity=1,
            quantity_unit="job",
            funding_source=self.funding_source,
            usage_category="metered_infra",
            source_type="third_party_infra",
            source_subtype="third_party_training",
            source_provider="tinker",
            pricing_policy="wrapper_reported_or_backend_priced",
            meter_kind="tinker_training_job",
            provider_result_id=result_id or job_id,
            request_id=job_id,
            require_smr_attribution=True,
            idempotency_key=stable_usage_idempotency_key(
                "smr",
                self.run_id,
                "tinker",
                job_id or result_id or model or operation_kind,
                payload={
                    "task_id": self.task_id,
                    "actor_id": self.actor_id,
                    "worker_id": self.worker_id,
                },
            ),
            metadata={
                **dict(metadata or {}),
                "job_id": job_id,
                "result_id": result_id,
            },
        )
        return self.reporter.report(report)
