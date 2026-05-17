"""Shared reporting helpers for provider wrapper usage ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from typing import Any, Mapping
from urllib.parse import quote


_DEFAULT_TIMEOUT_SECONDS = 20.0
_BACKEND_BASE_URL_KEYS = (
    "SYNTH_BACKEND_INTERNAL_URL",
    "BACKEND_INTERNAL_URL",
    "SMR_BACKEND_PUBLIC_URL",
    "SMR_RUNTIME_PUBLIC_BASE_URL",
    "SYNTH_BACKEND_URL",
    "BACKEND_URL",
)
_API_KEY_KEYS = (
    "SMR_WORKER_API_KEY",
    "SMR_WORKER_API_KEY_DEFAULT",
    "SMR_API_KEY",
    "SYNTH_API_KEY",
)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProviderUsageReport:
    provider: str
    operation_kind: str
    run_id: str
    org_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    actor_id: str | None = None
    worker_id: str | None = None
    participant_session_id: str | None = None
    participant_role: str | None = None
    model: str | None = None
    estimated_cost_usd: Decimal | float | int | str | None = None
    actual_cost_usd: Decimal | float | int | str | None = None
    quantity: float | int | Decimal | None = None
    quantity_unit: str | None = None
    funding_source: str | None = None
    execution_path: str | None = None
    usage_category: str | None = None
    source_type: str | None = None
    source_subtype: str | None = None
    source_provider: str | None = None
    pricing_policy: str | None = None
    meter_kind: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None
    provider_result_id: str | None = None
    request_id: str | None = None
    occurred_at: datetime | None = None
    idempotency_key: str | None = None
    require_smr_attribution: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "operation_kind": self.operation_kind,
            "run_id": self.run_id,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "actor_id": self.actor_id,
            "worker_id": self.worker_id,
            "participant_session_id": self.participant_session_id,
            "participant_role": self.participant_role,
            "model": self.model,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "quantity": self.quantity,
            "quantity_unit": self.quantity_unit,
            "funding_source": self.funding_source,
            "execution_path": self.execution_path,
            "usage_category": self.usage_category,
            "source_type": self.source_type,
            "source_subtype": self.source_subtype,
            "source_provider": self.source_provider,
            "pricing_policy": self.pricing_policy,
            "meter_kind": self.meter_kind,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
            "provider_result_id": self.provider_result_id,
            "request_id": self.request_id,
            "occurred_at": self.occurred_at or datetime.now(timezone.utc),
            "idempotency_key": self.idempotency_key,
            "require_smr_attribution": self.require_smr_attribution,
            "metadata": self.metadata,
        }
        return {
            key: _serialize_value(value)
            for key, value in payload.items()
            if value is not None and value != {}
        }


def stable_usage_idempotency_key(
    *parts: Any, payload: Mapping[str, Any] | None = None
) -> str:
    normalized_parts = [str(part).strip() for part in parts if str(part).strip()]
    payload_blob = json.dumps(
        _serialize_value(dict(payload or {})),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload_blob.encode("utf-8")).hexdigest()[:24]
    if normalized_parts:
        return ":".join([*normalized_parts, digest])
    return digest


def resolve_backend_base_url(
    environment: Mapping[str, str] | None = None,
) -> str | None:
    env = environment or os.environ
    for key in _BACKEND_BASE_URL_KEYS:
        value = str(env.get(key) or "").strip().rstrip("/")
        if value:
            return value
    return None


def resolve_reporting_api_key(
    environment: Mapping[str, str] | None = None,
) -> str | None:
    env = environment or os.environ
    for key in _API_KEY_KEYS:
        value = str(env.get(key) or "").strip()
        if value:
            return value
    return None


def _provider_usage_url(base_url: str, run_id: str) -> str:
    return f"{base_url}/smr/internal/runs/{quote(str(run_id).strip(), safe='')}/provider-usage"


class ProviderUsageReporter:
    def __init__(
        self,
        *,
        backend_base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        # Resolution order for the ingest URL:
        #   1. explicit constructor arg (caller pins it)
        #   2. ProviderUsageAttribution.provider_usage_ingest_base_url (file)
        #   3. environment via resolve_backend_base_url (compat fallback)
        # The attribution file is the authority when present, and the env
        # path emits a WARNING so we can eventually delete the fallback.
        if backend_base_url:
            self._backend_base_url = backend_base_url
        else:
            self._backend_base_url = _resolve_backend_base_url_via_attribution()
        self._api_key = api_key or resolve_reporting_api_key()
        self._timeout_seconds = timeout_seconds

    def report(self, report: ProviderUsageReport | Mapping[str, Any]) -> dict[str, Any]:
        payload = (
            report.to_payload()
            if isinstance(report, ProviderUsageReport)
            else dict(report)
        )
        if not self._backend_base_url:
            raise RuntimeError(
                "No backend base URL configured for provider usage reporting."
            )
        if not self._api_key:
            raise RuntimeError("No API key configured for provider usage reporting.")
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            raise RuntimeError("Provider usage reporting requires run_id.")
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "httpx is required for provider usage reporting."
            ) from exc
        response = httpx.post(
            _provider_usage_url(self._backend_base_url, run_id),
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "X-API-Key": self._api_key,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return dict(response.json())


def _resolve_backend_base_url_via_attribution() -> str | None:
    """Resolve the ingest URL via attribution file → env fallback.

    Delegates to the typed attribution authority. If the attribution file
    declares ``provider_usage_ingest_base_url``, use it; otherwise fall
    back to the env lookup that existed before file-first landed. Logged
    at WARNING when we fall back to env so the migration can be tracked
    and the fallback eventually removed.
    """
    import logging

    try:
        from synth_budget.attribution import ProviderUsageAttribution
    except Exception:
        return resolve_backend_base_url()
    attribution = ProviderUsageAttribution.resolve()
    if attribution.provider_usage_ingest_base_url:
        return attribution.provider_usage_ingest_base_url
    resolved = resolve_backend_base_url()
    if resolved:
        logging.getLogger(__name__).warning(
            "provider_usage_reporter.base_url_from_env resolved=%s "
            "(attribution file did not declare provider_usage_ingest_base_url)",
            resolved,
        )
    return resolved


def report_provider_usage(
    report: ProviderUsageReport | Mapping[str, Any],
    *,
    backend_base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return ProviderUsageReporter(
        backend_base_url=backend_base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    ).report(report)
