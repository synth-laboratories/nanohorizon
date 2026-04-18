"""Canonical outcome vocabulary for provider-usage report rows.

Problem this solves
-------------------
Before this enum landed, every lane invented its own ``status`` strings
in its metering-report JSONL: ``ok``, ``skipped``, ``error``,
``report_failed``, ``local_measurement``, … The submission gate validated
against a hard-coded subset of those strings. Lanes drifted — nano
emitted ``local_measurement`` (correct, descriptive) but the gate didn't
know it; the 2026-04-16 incident had the gate accepting a file full of
``skipped`` rows as "published successfully" because the file existed.

Style basis (specifications/tanha/references/synthstyle.md)
-----------------------------------------------------------
- "If the system exposes a concept like readiness, status, phase,
  eligibility, completion, or authority, it must have a single
  authoritative meaning derived from one canonical model. Do not
  maintain parallel definitions of the same concept across persistence,
  APIs, summaries, logs, or control flow."
- "do not reconstruct meaning through heuristic string matching, alias
  tables, or payload-shape probing when a typed contract can exist"

This module is the one source of truth. Every helper that writes a
row must call :meth:`ProviderUsageReportOutcome.value` on an enum
member. Every gate that reads rows must call
:meth:`ProviderUsageReportOutcome.coerce` and treat anything outside
the enum as :attr:`ProviderUsageReportOutcome.UNKNOWN` (loud, typed,
visible in the rejection body).
"""

from __future__ import annotations

import enum
from typing import Any


class ProviderUsageReportOutcome(str, enum.Enum):
    """Canonical outcome for a single metering-report row.

    Ordering (top to bottom) reflects *decreasing* guarantee that the
    backend's usage ledger has a matching ``smr_usage_facts`` row. Gates
    and dashboards should treat :attr:`REPORTED` as the sole success
    outcome. All other values are degradations with distinct remediation
    paths:

    - :attr:`REPORTED`       — provider call happened, reporter POSTed
      successfully, backend persisted the usage fact. This is the only
      state that produces a verifiable entry in the usage ledger.
    - :attr:`REPORT_FAILED`  — provider call happened, reporter attempted
      the POST, backend rejected or was unreachable. Measurement exists
      locally but is not in the ledger. Remediation: fix the reporter
      URL / auth / backend health.
    - :attr:`MEASURED_LOCAL` — provider call happened, the helper chose
      not to POST (usually because routing config wasn't present).
      Semantically the same as report_failed from the ledger's
      perspective, but distinguishes "didn't try" from "tried and
      failed". Remediation: populate
      ``ProviderUsageAttribution.provider_usage_ingest_base_url``.
    - :attr:`SKIPPED`        — provider call did not happen. Upstream
      guard (e.g. missing attribution) refused before the network call.
      Usually accompanied by a ``reason`` field.
    - :attr:`ERROR`          — provider call raised during the helper's
      execution. ``reason`` should carry the exception type/message.
    - :attr:`UNKNOWN`        — row's ``status`` field was absent, or a
      string not in this enum. Gates treat UNKNOWN as failure and
      surface the raw value so the operator can decide whether to
      extend the enum or fix the producer.
    """

    REPORTED = "reported"
    REPORT_FAILED = "report_failed"
    MEASURED_LOCAL = "local_measurement"
    SKIPPED = "skipped"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, raw: Any) -> "ProviderUsageReportOutcome":
        """Parse an incoming row's ``status`` into the canonical enum.

        Never raises. Unknown inputs map to :attr:`UNKNOWN` so the gate
        can surface the offending string in the rejection body rather
        than crashing at the validator.
        """
        if isinstance(raw, cls):
            return raw
        value = str(raw or "").strip().lower()
        # Historical synonym: some older lanes wrote "ok" when they meant
        # "reported". Treat the legacy alias as REPORTED so older runs
        # don't retroactively fail, but never emit "ok" from new code.
        if value == "ok":
            return cls.REPORTED
        for member in cls:
            if member.value == value:
                return member
        return cls.UNKNOWN

    @classmethod
    def default_accept(cls) -> frozenset["ProviderUsageReportOutcome"]:
        """The strict default: only REPORTED rows satisfy a metering gate."""
        return frozenset({cls.REPORTED})


__all__ = [
    "ProviderUsageReportOutcome",
]
