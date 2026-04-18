"""Session-scoped OpenRouter metering wrapper for SMR worker sandboxes."""

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


def _int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass(slots=True)
class OpenRouterBudgetSession:
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
    ) -> "OpenRouterBudgetSession":
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
        operation_kind: str = "chat_completion",
    ) -> Decimal:
        return self.session_budget.check_increment(
            projected_cost_usd,
            context=f"OpenRouter {operation_kind} {model or 'request'}",
        )

    def report_response(
        self,
        *,
        model: str,
        request_id: str | None,
        usage: Mapping[str, Any] | None = None,
        response_id: str | None = None,
        projected_cost_usd: Decimal | float | int | str | None = None,
        actual_cost_usd: Decimal | float | int | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        operation_kind: str = "chat_completion",
    ) -> dict[str, Any]:
        usage_payload = dict(usage or {})
        input_tokens = _int(
            usage_payload.get("prompt_tokens") or usage_payload.get("input_tokens")
        )
        cached_input_tokens = _int(
            usage_payload.get("prompt_tokens_details", {}).get("cached_tokens")
        )
        output_tokens = _int(
            usage_payload.get("completion_tokens") or usage_payload.get("output_tokens")
        )
        reasoning_output_tokens = _int(
            usage_payload.get("completion_tokens_details", {}).get("reasoning_tokens")
        )
        total_tokens = _int(usage_payload.get("total_tokens"))
        if total_tokens is None:
            total_tokens = (
                sum(
                    value or 0
                    for value in (
                        input_tokens,
                        cached_input_tokens,
                        output_tokens,
                        reasoning_output_tokens,
                    )
                )
                or None
            )
        if projected_cost_usd is not None:
            self.check_projected_cost(
                projected_cost_usd, model=model, operation_kind=operation_kind
            )
        if actual_cost_usd is not None:
            self.session_budget.commit_increment(actual_cost_usd)
        report = ProviderUsageReport(
            provider="openrouter",
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
            quantity_unit="request",
            funding_source=self.funding_source,
            usage_category="inference",
            source_type="inference",
            source_subtype="user_code_inference",
            source_provider="openrouter",
            pricing_policy="wrapper_reported_or_backend_priced",
            meter_kind="openrouter_request",
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            total_tokens=total_tokens,
            provider_result_id=response_id,
            request_id=request_id,
            require_smr_attribution=True,
            idempotency_key=stable_usage_idempotency_key(
                "smr",
                self.run_id,
                "openrouter",
                request_id or response_id or model,
                payload={
                    "task_id": self.task_id,
                    "actor_id": self.actor_id,
                    "worker_id": self.worker_id,
                },
            ),
            metadata={
                **dict(metadata or {}),
                "provider_response_id": response_id,
                "usage": usage_payload,
            },
        )
        return self.reporter.report(report)
