"""Shared SMR runtime attribution for provider wrappers.

Problem this solves
-------------------
Before this module was made file-first, attribution was read exclusively
from environment variables (``SMR_RUN_ID``, ``SMR_ORG_ID``, etc.). That
worked when the runtime invoked the worker script directly, but broke
whenever a subprocess layer (``uv run``, a subagent spawn, Codex's
``shell --sanitize``) stripped the env. On 2026-04-16 the
``nanohorizon_responses_api_message_policy`` lane made 21 real OpenRouter
calls; every one recorded ``status=skipped`` because the helper's
subprocess environment had lost ``SMR_RUN_ID``.

Design (synth-style: one configuration authority layer)
-------------------------------------------------------
Attribution lives in a typed JSON file that the runtime writes into the
workspace before any worker subprocess runs. The file is the authority.
Environment variables remain a *compat* fallback for older code paths,
with a loud warning when we resolve from env — so the fallback can be
deleted once every runtime writes the file.

Schema (``<workspace>/.smr_attribution.json``)::

    {
      "schema_version": 1,
      "org_id": "...",
      "project_id": "...",
      "run_id": "...",
      "task_id": "...",           # optional
      "actor_id": "...",          # optional
      "worker_id": "...",         # optional
      "participant_session_id": "...",  # optional
      "participant_role": "worker"|"orchestrator"|"reviewer"  # optional
    }

Required fields: ``schema_version``, ``run_id``. Missing required fields
cause :func:`ProviderUsageAttribution.resolve_required` to raise with a
typed error so a caller that declared metering does not silently degrade.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)


ATTRIBUTION_SCHEMA_VERSION = 1
ATTRIBUTION_FILENAME = ".smr_attribution.json"


def _first_env(environment: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        value = str(environment.get(key) or "").strip()
        if value:
            return value
    return None


class ProviderUsageAttributionMissing(RuntimeError):
    """Raised when attribution is required but cannot be resolved.

    Carries a typed reason so operators can distinguish "no file, no env"
    from "file present but missing required field" from "file present but
    malformed". The body is also included in logs for postmortem triage.
    """


@dataclass(frozen=True, slots=True)
class ProviderUsageAttribution:
    org_id: str | None = None
    project_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    actor_id: str | None = None
    worker_id: str | None = None
    participant_session_id: str | None = None
    participant_role: str | None = None
    # Ingest routing: where the ProviderUsageReporter should POST usage
    # facts, and the URL path suffix. Auth stays in the environment
    # (SMR_WORKER_API_KEY etc.) because tokens are credentials, not
    # routing config — but the *URL* is routing config and belongs in
    # the attribution authority, not scattered os.getenv calls across
    # subprocess layers.
    provider_usage_ingest_base_url: str | None = None

    # --- new file-first resolvers -----------------------------------

    @classmethod
    def from_attribution_file(
        cls, path: str | Path
    ) -> "ProviderUsageAttribution | None":
        """Read a typed attribution file. Returns None when the file is absent.

        Malformed files (bad JSON, wrong schema version, missing required
        ``run_id``) raise :class:`ProviderUsageAttributionMissing` with a
        specific reason — we never silently degrade to partial attribution.
        """
        p = Path(path)
        if not p.is_file():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderUsageAttributionMissing(
                f"attribution file at {p} is not valid JSON: {exc.msg} "
                f"(offset {exc.pos})"
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderUsageAttributionMissing(
                f"attribution file at {p} must be a JSON object, "
                f"got {type(payload).__name__}"
            )
        schema_version = payload.get("schema_version")
        if schema_version != ATTRIBUTION_SCHEMA_VERSION:
            raise ProviderUsageAttributionMissing(
                f"attribution file at {p} has schema_version={schema_version!r}; "
                f"expected {ATTRIBUTION_SCHEMA_VERSION}"
            )
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            raise ProviderUsageAttributionMissing(
                f"attribution file at {p} is missing required field run_id"
            )
        return cls(
            org_id=str(payload.get("org_id") or "").strip() or None,
            project_id=str(payload.get("project_id") or "").strip() or None,
            run_id=run_id,
            task_id=str(payload.get("task_id") or "").strip() or None,
            actor_id=str(payload.get("actor_id") or "").strip() or None,
            worker_id=str(payload.get("worker_id") or "").strip() or None,
            participant_session_id=(
                str(payload.get("participant_session_id") or "").strip() or None
            ),
            participant_role=(
                str(payload.get("participant_role") or "").strip() or None
            ),
            provider_usage_ingest_base_url=(
                str(payload.get("provider_usage_ingest_base_url") or "")
                .strip()
                .rstrip("/")
                or None
            ),
        )

    @classmethod
    def resolve(
        cls,
        *,
        workspace_root: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> "ProviderUsageAttribution":
        """Resolve attribution: file first, env fallback.

        Searches for an attribution file at:
          1. ``<workspace_root>/.smr_attribution.json`` if ``workspace_root``
             was provided.
          2. ``./.smr_attribution.json`` relative to the current working
             directory, as a convenience for subprocesses that inherit cwd
             but may have lost env.

        Falls back to :meth:`from_env` with a WARNING when neither file
        exists. Callers that *require* attribution should use
        :meth:`resolve_required`.
        """
        candidate_paths: list[Path] = []
        if workspace_root is not None:
            candidate_paths.append(Path(workspace_root) / ATTRIBUTION_FILENAME)
        # cwd-relative fallback lets /workspace-mounted containers find it
        # without the caller knowing the mount path explicitly.
        candidate_paths.append(Path.cwd() / ATTRIBUTION_FILENAME)
        for candidate in candidate_paths:
            result = cls.from_attribution_file(candidate)
            if result is not None:
                return result
        logger.warning(
            "provider_usage_attribution.file_missing candidates=%s falling_back_to=env",
            [str(p) for p in candidate_paths],
        )
        return cls.from_env(environment=environment)

    @classmethod
    def resolve_required(
        cls,
        *,
        workspace_root: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> "ProviderUsageAttribution":
        """Like :meth:`resolve` but raises if ``run_id`` is not established.

        Use this from metering helpers whose contract says attribution is
        load-bearing. Silent degradation to an empty-run-id session is the
        bug we are closing.
        """
        attribution = cls.resolve(
            workspace_root=workspace_root, environment=environment
        )
        if not str(attribution.run_id or "").strip():
            raise ProviderUsageAttributionMissing(
                "Provider usage attribution is required but could not be "
                "resolved from either the workspace .smr_attribution.json "
                "file or the process environment. A lane helper that "
                "declares metering must see a populated run_id; if you are "
                "running locally, ensure the SMR runtime wrote the "
                "attribution file to the workspace before invoking the lane."
            )
        return attribution

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "ProviderUsageAttribution":
        env = environment or os.environ
        return cls(
            org_id=_first_env(env, "SMR_ORG_ID", "SYNTH_ORG_ID", "ORG_ID"),
            project_id=_first_env(
                env, "SMR_PROJECT_ID", "SYNTH_PROJECT_ID", "PROJECT_ID"
            ),
            run_id=_first_env(env, "SMR_RUN_ID", "SYNTH_RUN_ID", "RUN_ID"),
            task_id=_first_env(env, "SMR_TASK_ID", "SMR_TASK_KEY", "TASK_ID"),
            actor_id=_first_env(env, "SMR_ACTOR_ID", "SMR_ACTOR_KEY", "ACTOR_ID"),
            worker_id=_first_env(env, "SMR_WORKER_ID", "WORKER_ID"),
            participant_session_id=_first_env(
                env,
                "SMR_PARTICIPANT_SESSION_ID",
                "SMR_SESSION_ID",
                "PARTICIPANT_SESSION_ID",
            ),
            participant_role=_first_env(
                env, "SMR_PARTICIPANT_ROLE", "PARTICIPANT_ROLE"
            ),
            provider_usage_ingest_base_url=_first_env(
                env,
                "SMR_PROVIDER_USAGE_INGEST_BASE_URL",
                "SYNTH_BACKEND_INTERNAL_URL",
                "SYNTH_BACKEND_URL",
                "BACKEND_URL",
            ),
        )

    def apply_defaults(self, target: object) -> None:
        # Only push identity fields onto the session target. The ingest URL
        # is reporter routing, not session identity — it's consumed by
        # ``ProviderUsageReporter.__init__`` directly via the attribution
        # resolver, never attached to the session. Pushing it onto a
        # ``slots=True`` session dataclass raises AttributeError.
        for field_name in (
            "org_id",
            "project_id",
            "run_id",
            "task_id",
            "actor_id",
            "worker_id",
            "participant_session_id",
            "participant_role",
        ):
            current = getattr(target, field_name, None)
            if str(current or "").strip():
                continue
            resolved = getattr(self, field_name)
            if str(resolved or "").strip():
                setattr(target, field_name, resolved)

    def to_file_payload(self) -> dict[str, object]:
        """Render for :meth:`write_attribution_file`. Only non-empty fields."""
        payload: dict[str, object] = {
            "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        }
        for field_name in (
            "org_id",
            "project_id",
            "run_id",
            "task_id",
            "actor_id",
            "worker_id",
            "participant_session_id",
            "participant_role",
            "provider_usage_ingest_base_url",
        ):
            value = getattr(self, field_name)
            if str(value or "").strip():
                payload[field_name] = value
        return payload


def write_attribution_file(
    workspace_root: str | Path,
    attribution: ProviderUsageAttribution,
) -> Path:
    """Atomically write the attribution file into ``workspace_root``.

    The runtime calls this immediately after seeding a worker's workspace.
    Writes through a ``.tmp`` + ``os.replace`` so concurrent readers
    never see a half-written JSON blob.
    """
    if not str(attribution.run_id or "").strip():
        raise ValueError("write_attribution_file requires attribution.run_id to be set")
    target = Path(workspace_root) / ATTRIBUTION_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(attribution.to_file_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


__all__ = [
    "ATTRIBUTION_FILENAME",
    "ATTRIBUTION_SCHEMA_VERSION",
    "ProviderUsageAttribution",
    "ProviderUsageAttributionMissing",
    "write_attribution_file",
]
