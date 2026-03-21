from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, cast

try:
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
except Exception:
    BatchUpdate = cast(Any, object)
    LogitsProcessor = cast(Any, object)

from .runtime import (
    DEFAULT_QWEN_THINK_CLOSE_ID,
    DEFAULT_QWEN_THINK_OPEN_ID,
    THINK_BUDGET_EXTRA_ARG,
    THINK_CLOSE_ID_ENV,
    THINK_OPEN_ID_ENV,
    THINK_STARTS_OPEN_EXTRA_ARG,
    normalize_thinking_budget,
)

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _load_token_id(env_name: str, fallback: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        return int(fallback)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(fallback)


def _token_sequence_view(token_ids: Any) -> Sequence[Any] | None:
    if isinstance(token_ids, (str, bytes, bytearray)):
        return None
    if isinstance(token_ids, Sequence):
        return token_ids
    if hasattr(token_ids, "__len__") and hasattr(token_ids, "__getitem__"):
        return cast(Sequence[Any], token_ids)
    return None


class QwenThinkingBudgetLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory: bool) -> None:
        del vllm_config, device, is_pin_memory
        self._open_id = _load_token_id(THINK_OPEN_ID_ENV, DEFAULT_QWEN_THINK_OPEN_ID)
        self._close_id = _load_token_id(THINK_CLOSE_ID_ENV, DEFAULT_QWEN_THINK_CLOSE_ID)
        self._state: dict[int, dict[str, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    @classmethod
    def validate_params(cls, params: Any) -> None:
        extra_args = getattr(params, "extra_args", None) or {}
        budget = extra_args.get(THINK_BUDGET_EXTRA_ARG)
        if budget is None:
            return
        if normalize_thinking_budget(budget) is None:
            raise ValueError(
                f"{THINK_BUDGET_EXTRA_ARG} must be a positive integer, got {budget!r}"
            )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            self._sync_output_growth()
            return

        for index in getattr(batch_update, "removed", []) or []:
            self._state.pop(index, None)

        for index, params, prompt_tok_ids, output_tok_ids in getattr(batch_update, "added", []) or []:
            extra_args = getattr(params, "extra_args", None) or {}
            budget = normalize_thinking_budget(extra_args.get(THINK_BUDGET_EXTRA_ARG))
            if budget is None:
                self._state.pop(index, None)
                continue
            starts_open = _coerce_bool(
                extra_args.get(THINK_STARTS_OPEN_EXTRA_ARG),
                default=self._prompt_has_unclosed_think(prompt_tok_ids),
            )
            output_sequence = _token_sequence_view(output_tok_ids)
            self._state[index] = {
                "budget": budget,
                "observed_think_tokens": 0,
                "predicted_think_tokens": 0,
                "in_think": starts_open,
                "forcing_close": False,
                "budget_exhausted": False,
                "output_tok_ids": output_sequence,
                "last_len": len(output_sequence) if output_sequence is not None else 0,
            }

        for source_idx, target_idx, _direction in getattr(batch_update, "moved", []) or []:
            if source_idx in self._state and source_idx != target_idx:
                self._state[target_idx] = self._state.pop(source_idx)

        self._sync_output_growth()

    def apply(self, logits) -> Any:
        self._sync_output_growth()
        vocab_size = logits.shape[-1]

        for row_idx, state in list(self._state.items()):
            if state.get("forcing_close"):
                self._mask_row_to_token(
                    logits=logits,
                    row_idx=row_idx,
                    token_id=self._close_id,
                    vocab_size=vocab_size,
                )
                state["forcing_close"] = False
                state["in_think"] = False
                state["budget_exhausted"] = True
                continue
            if state.get("in_think") and not state.get("budget_exhausted"):
                budget = int(state.get("budget", 0))
                observed = int(state.get("observed_think_tokens", 0))
                predicted = int(state.get("predicted_think_tokens", observed))
                effective_used = observed if observed > predicted else predicted
                if effective_used >= budget:
                    self._mask_row_to_token(
                        logits=logits,
                        row_idx=row_idx,
                        token_id=self._close_id,
                        vocab_size=vocab_size,
                    )
                    state["forcing_close"] = False
                    state["in_think"] = False
                    state["budget_exhausted"] = True
                    continue
                state["predicted_think_tokens"] = effective_used + 1
            if state.get("budget_exhausted") and not state.get("in_think"):
                self._mask_token(logits=logits, row_idx=row_idx, token_id=self._open_id, vocab_size=vocab_size)
                self._mask_token(logits=logits, row_idx=row_idx, token_id=self._close_id, vocab_size=vocab_size)

        return logits

    def _sync_output_growth(self) -> None:
        for state in self._state.values():
            output_tok_ids = _token_sequence_view(state.get("output_tok_ids"))
            if output_tok_ids is None:
                continue
            last_len = int(state.get("last_len", 0))
            current_len = len(output_tok_ids)
            if current_len <= last_len:
                continue
            for token_id in output_tok_ids[last_len:current_len]:
                self._consume_token(state=state, token_id=token_id)
            state["last_len"] = current_len

    def _consume_token(self, *, state: dict[str, Any], token_id: Any) -> None:
        try:
            token_int = int(token_id)
        except (TypeError, ValueError):
            return

        if state.get("forcing_close"):
            if token_int == self._close_id:
                state["forcing_close"] = False
                state["in_think"] = False
                state["budget_exhausted"] = True
            return

        if not state.get("in_think"):
            if token_int == self._open_id and not state.get("budget_exhausted"):
                state["in_think"] = True
            return

        if token_int == self._close_id:
            state["in_think"] = False
            return

        state["observed_think_tokens"] = int(state.get("observed_think_tokens", 0)) + 1
        state["predicted_think_tokens"] = max(
            int(state.get("predicted_think_tokens", 0)),
            int(state["observed_think_tokens"]),
        )
        if int(state["observed_think_tokens"]) >= int(state.get("budget", 0)):
            state["forcing_close"] = True

    def _prompt_has_unclosed_think(self, prompt_tok_ids: Any) -> bool:
        prompt_tokens = _token_sequence_view(prompt_tok_ids)
        if prompt_tokens is None:
            return False
        last_open = -1
        last_close = -1
        for idx, token_id in enumerate(prompt_tokens):
            try:
                token_int = int(token_id)
            except (TypeError, ValueError):
                continue
            if token_int == self._open_id:
                last_open = idx
            elif token_int == self._close_id:
                last_close = idx
        return last_open > last_close

    def _mask_row_to_token(self, *, logits, row_idx: int, token_id: int, vocab_size: int) -> None:
        if not (0 <= token_id < vocab_size):
            logger.warning("thinking close token %s is outside vocab size %s", token_id, vocab_size)
            return
        if logits.dim() == 1:
            logits.fill_(float("-inf"))
            logits[token_id] = 0.0
            return
        if not (0 <= row_idx < logits.shape[0]):
            return
        logits[row_idx, :].fill_(float("-inf"))
        logits[row_idx, token_id] = 0.0

    def _mask_token(self, *, logits, row_idx: int, token_id: int, vocab_size: int) -> None:
        if not (0 <= token_id < vocab_size):
            return
        if logits.dim() == 1:
            logits[token_id] = float("-inf")
            return
        if 0 <= row_idx < logits.shape[0]:
            logits[row_idx, token_id] = float("-inf")
