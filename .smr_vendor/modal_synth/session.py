"""Observability-first Modal session wrapper for SMR worker sandboxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from .sandbox import create_sandbox
from synth_budget import ProviderUsageAttribution
from synth_budget import ProviderUsageReport
from synth_budget import SessionBudget
from synth_budget import stable_usage_idempotency_key
from synth_budget.reporting import ProviderUsageReporter

DEFAULT_BUDGET_USD = Decimal("10.00")


@dataclass(slots=True)
class ModalBudgetSession:
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
    ) -> "ModalBudgetSession":
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

    def create_sandbox(
        self, *sandbox_args: Any, **sandbox_kwargs: Any
    ) -> tuple[Any, Any]:
        sandbox, estimate = create_sandbox(
            *sandbox_args,
            budget_usd=self.budget_usd,
            **sandbox_kwargs,
        )
        self.report_usage(
            operation_kind="sandbox_launch",
            model=sandbox_kwargs.get("gpu") or "cpu",
            estimated_cost_usd=estimate.estimated_upper_bound_usd,
            quantity=float(estimate.timeout_seconds),
            quantity_unit="seconds",
            metadata={
                "gpu_type": estimate.gpu_type,
                "gpu_count": estimate.gpu_count,
                "timeout_seconds": estimate.timeout_seconds,
            },
        )
        return sandbox, estimate

    def report_usage(
        self,
        *,
        operation_kind: str,
        model: str | None = None,
        estimated_cost_usd: Decimal | float | int | str | None = None,
        actual_cost_usd: Decimal | float | int | str | None = None,
        quantity: float | int | Decimal | None = None,
        quantity_unit: str | None = None,
        provider_result_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if estimated_cost_usd is not None:
            self.session_budget.check_increment(
                estimated_cost_usd,
                context=f"Modal {operation_kind} {model or 'sandbox'}",
            )
        if actual_cost_usd is not None:
            self.session_budget.commit_increment(actual_cost_usd)
        report = ProviderUsageReport(
            provider="modal",
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
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=actual_cost_usd,
            quantity=quantity,
            quantity_unit=quantity_unit or "seconds",
            funding_source=self.funding_source,
            usage_category="metered_infra",
            source_type="third_party_infra",
            source_subtype="third_party_gpu",
            source_provider="modal",
            pricing_policy="observability_only_modal",
            meter_kind="modal_sandbox_seconds",
            provider_result_id=provider_result_id,
            require_smr_attribution=True,
            idempotency_key=stable_usage_idempotency_key(
                "smr",
                self.run_id,
                "modal",
                operation_kind,
                provider_result_id or model or self.task_id or self.actor_id,
                payload={
                    "worker_id": self.worker_id,
                    "participant_session_id": self.participant_session_id,
                },
            ),
            metadata={
                **dict(metadata or {}),
                "billing": {
                    "chargeable": False,
                    "billing_route": "none",
                    "chargeability_reason": "smr_modal_observability_only",
                },
            },
        )
        return self.reporter.report(report)
