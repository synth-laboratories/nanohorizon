from __future__ import annotations

from typing import Any

THINK_BUDGET_EXTRA_ARG = "think_budget"
THINK_STARTS_OPEN_EXTRA_ARG = "think_starts_open"
THINKING_BUDGET_PROCESSOR_FQCN = (
    "nanohorizon.custom_vllm.thinking_budget:QwenThinkingBudgetLogitsProcessor"
)
THINK_OPEN_ID_ENV = "NANOHORIZON_THINK_OPEN_ID"
THINK_CLOSE_ID_ENV = "NANOHORIZON_THINK_CLOSE_ID"
QWEN3_THINK_OPEN_ID = 151667
QWEN3_THINK_CLOSE_ID = 151668
QWEN35_THINK_OPEN_ID = 248068
QWEN35_THINK_CLOSE_ID = 248069
DEFAULT_QWEN_THINK_OPEN_ID = QWEN35_THINK_OPEN_ID
DEFAULT_QWEN_THINK_CLOSE_ID = QWEN35_THINK_CLOSE_ID


def normalize_thinking_budget(value: Any) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _default_think_token_ids(model_ref: str) -> tuple[int, int]:
    lowered = str(model_ref).strip().lower()
    if "qwen3.5" in lowered or "qwen3_5" in lowered:
        return (QWEN35_THINK_OPEN_ID, QWEN35_THINK_CLOSE_ID)
    return (QWEN3_THINK_OPEN_ID, QWEN3_THINK_CLOSE_ID)


def _resolve_token_id(tokenizer: Any, token: str) -> int | None:
    for resolver in (
        lambda: tokenizer.convert_tokens_to_ids(token),
        lambda: (tokenizer.get_vocab() or {}).get(token),
        lambda: tokenizer.encode(token, add_special_tokens=False),
    ):
        try:
            resolved = resolver()
        except Exception:
            continue
        if isinstance(resolved, int) and resolved >= 0:
            return resolved
        if isinstance(resolved, list) and len(resolved) == 1:
            try:
                token_id = int(resolved[0])
            except (TypeError, ValueError):
                continue
            if token_id >= 0:
                return token_id
    return None


def build_thinking_budget_request_overrides(
    *,
    enable_thinking: bool | None,
    thinking_budget: Any = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    chat_template_kwargs: dict[str, Any] = {}
    normalized_budget = normalize_thinking_budget(thinking_budget)

    if enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = bool(enable_thinking)

    if normalized_budget is not None:
        chat_template_kwargs["enable_thinking"] = True
        overrides["vllm_xargs"] = {
            THINK_BUDGET_EXTRA_ARG: normalized_budget,
            THINK_STARTS_OPEN_EXTRA_ARG: True,
        }

    if chat_template_kwargs:
        overrides["chat_template_kwargs"] = chat_template_kwargs

    return overrides


def _prime_think_token_ids_env(
    *,
    env: dict[str, str],
    model_ref: str,
) -> None:
    default_open_id, default_close_id = _default_think_token_ids(model_ref)
    env.setdefault(THINK_OPEN_ID_ENV, str(default_open_id))
    env.setdefault(THINK_CLOSE_ID_ENV, str(default_close_id))

    try:
        from transformers import AutoTokenizer
    except Exception:
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        open_id = _resolve_token_id(tokenizer, "<think>")
        close_id = _resolve_token_id(tokenizer, "</think>")
        if open_id is not None:
            env[THINK_OPEN_ID_ENV] = str(open_id)
        if close_id is not None:
            env[THINK_CLOSE_ID_ENV] = str(close_id)
    except Exception as exc:
        print(
            f"warning: failed to resolve thinking token ids for {model_ref}: {exc!r}",
            flush=True,
        )


def enable_thinking_budget_support(
    *,
    cmd: list[str],
    env: dict[str, str],
    model_ref: str,
    resolve_token_ids: bool = True,
) -> tuple[list[str], dict[str, str]]:
    updated_cmd = list(cmd)
    updated_env = dict(env)

    if "--logits-processors" not in updated_cmd:
        updated_cmd += ["--logits-processors", THINKING_BUDGET_PROCESSOR_FQCN]

    updated_env.setdefault("VLLM_USE_V1", "1")

    if resolve_token_ids:
        _prime_think_token_ids_env(env=updated_env, model_ref=model_ref)

    return updated_cmd, updated_env
