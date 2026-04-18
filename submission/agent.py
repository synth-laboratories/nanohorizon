from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.shared.common import write_json
from nanohorizon.shared.craftax_data import (
    collect_rollouts_concurrently_with_summary,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
)
from nanohorizon.shared.openai_compat import create_chat_completion
from nanohorizon.shared.vllm_eval import LocalVLLMEvalConfig, local_vllm_server

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_DEFAULT_POLICY_VERSION = "deterministic_resource_controller_v2"
_DEFAULT_HONEST_EVAL_LIMIT = 0

_RESOURCE_KEYS = (
    "wood",
    "stone",
    "coal",
    "iron",
    "sapling",
    "drink",
    "food",
    "health",
    "energy",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
)
_VISIBLE_OBJECT_KEYS = (
    "tree",
    "stone",
    "table",
    "furnace",
    "plant",
    "cow",
    "water",
    "zombie",
    "skeleton",
    "log",
    "trunk",
)
_OBSERVATION_PREFIX = "Current Craftax long-horizon observation:"
_OBSERVATION_SUFFIX = "Plan a short useful macro-action."


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_str(name: str, default: str) -> str:
    raw = str(os.getenv(name, "")).strip()
    return raw or default


def _default_train_seeds() -> list[int]:
    if _SEED_MANIFEST_PATH.exists():
        payload = json.loads(_SEED_MANIFEST_PATH.read_text(encoding="utf-8"))
        values = payload.get("train_seeds") if isinstance(payload, dict) else None
        if isinstance(values, list) and values:
            return [int(item) for item in values]
    return [seed for seed in range(0, 20)]


def _controller_system_prompt() -> str:
    return (
        "You are a Craftax policy.\n"
        "A deterministic pre-policy handles obvious tree, table, and wood-pickaxe states.\n"
        "Saplings and plants are traps: if they are the only visible cue, search for trees with movement only.\n"
        "If the observation only exposes player_position and achievements, use deterministic exploration instead of asking the LLM.\n"
        "Once wood is available, place a table immediately, then make a wood pickaxe immediately.\n"
        "Only fall back to the LLM when the observation is genuinely ungrounded or empty.\n"
        f"Use the provided {PRIMARY_TOOL_NAME!r} tool exactly once for the final answer.\n"
        "Return a short valid full-Craftax macro-action.\n"
        "Do not return JSON or plain text actions."
    )


def _honest_eval_limit() -> int:
    limit = _env_int("NANOHORIZON_SUBMISSION_HONEST_EVAL_LIMIT", _DEFAULT_HONEST_EVAL_LIMIT)
    return max(0, int(limit))


def _format_eval_report(result: dict[str, Any], *, config: dict[str, Any], seeds: list[int]) -> str:
    details = result.get("details") if isinstance(result, dict) else None
    detail_rows = [row for row in details if isinstance(row, dict)] if isinstance(details, list) else []
    lines = [
        "# NanoHorizon honest train-seed eval",
        "",
        f"- policy_version: `{config.get('policy_version', _DEFAULT_POLICY_VERSION)}`",
        f"- base_model: `{config.get('base_model', '')}`",
        f"- max_steps: `{config.get('max_steps', '')}`",
        f"- action_batch: `{config.get('min_action_batch_size', '')}-{config.get('target_action_batch_size', '')}`",
        f"- seeds_evaluated: `{len(seeds)}`",
        f"- eval_limit: `{_honest_eval_limit()}`",
        f"- mean_outcome_reward: `{float(result.get('mean_outcome_reward', 0.0)):.3f}`",
        f"- mean_outcome_reward_over_requested_rollouts: `{float(result.get('mean_outcome_reward_over_requested_rollouts', 0.0)):.3f}`",
        f"- max_outcome_reward: `{float(result.get('max_outcome_reward', 0.0)):.3f}`",
        f"- mean_llm_calls_per_rollout: `{float(result.get('mean_llm_calls_per_rollout', 0.0)):.3f}`",
        f"- controller_hits: `{int(result.get('controller_hits', 0))}`",
        f"- fallback_hits: `{int(result.get('fallback_hits', 0))}`",
        "",
        "## Rollout evidence",
    ]
    if not detail_rows:
        lines.append("- no rollout details were returned")
    else:
        for row in detail_rows:
            seed = row.get("seed", "")
            reward = float(row.get("outcome_reward", 0.0) or 0.0)
            achievements = row.get("achievements") if isinstance(row.get("achievements"), list) else []
            achievement_text = ", ".join(str(item) for item in achievements) if achievements else "none"
            error = str(row.get("error") or "").strip()
            if error:
                lines.append(f"- seed {seed}: error={error}")
            else:
                lines.append(f"- seed {seed}: reward={reward:.3f}, achievements={achievement_text}")
    return "\n".join(lines).rstrip() + "\n"


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon submission surface for a deterministic-plus-LLM Craftax controller.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 128),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 0),
        "enable_thinking": False,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 1),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 1),
        "policy_version": _DEFAULT_POLICY_VERSION,
        "system_prompt": _controller_system_prompt(),
    }


def train(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = define()
    checkpoint = {
        "define": config,
        "train_data_dir": str(data_dir),
        "trained": False,
        "policy_version": config["policy_version"],
    }
    write_json(out_dir / "checkpoint.json", checkpoint)


def _resolve_seeds(data_dir: Path, config: dict[str, Any]) -> list[int]:
    seeds_path = data_dir / "seeds.json"
    if seeds_path.exists():
        payload = json.loads(seeds_path.read_text(encoding="utf-8"))
        values = payload.get("seeds") if isinstance(payload, dict) else payload
        if isinstance(values, list):
            return [int(item) for item in values]
    return [int(item) for item in config.get("train_seeds", [])]


def _maybe_limit_seeds(seeds: list[int]) -> list[int]:
    raw_limit = str(os.getenv("NANOHORIZON_SUBMISSION_EVAL_SEED_LIMIT", "")).strip()
    if not raw_limit:
        return seeds
    try:
        limit = max(1, int(raw_limit))
    except ValueError:
        return seeds
    return seeds[:limit]


def _extract_user_observation(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
            text = "\n".join(parts).strip()
        else:
            text = str(content or "").strip()
        if not text:
            continue
        match = re.search(
            rf"{re.escape(_OBSERVATION_PREFIX)}\s*(.*?)(?:\n\s*\n{re.escape(_OBSERVATION_SUFFIX)}|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return text
    return ""


def _parse_inventory_from_text(text: str) -> dict[str, int]:
    normalized = str(text or "").lower()
    inventory: dict[str, int] = {}
    for key in _RESOURCE_KEYS:
        match = re.search(rf"\b{re.escape(key)}\s*=\s*(-?\d+)\b", normalized)
        if match:
            inventory[key] = max(0, int(match.group(1)))
    return inventory


def _parse_achievements_from_text(text: str) -> list[str]:
    match = re.search(r"achievements:\s*(.+)", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw or raw.lower() == "none":
        return []
    return [item.strip().lower().replace(" ", "_") for item in raw.split(",") if item.strip()]


def _visible_objects_from_text(text: str) -> list[str]:
    normalized = str(text or "").lower()
    return [name for name in _VISIBLE_OBJECT_KEYS if name in normalized]


def _parse_player_position_from_text(text: str) -> tuple[int, int] | None:
    normalized = str(text or "").lower()
    match = re.search(r"player_position:\s*[\[(]?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[\])]?", normalized)
    if not match:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except ValueError:
        return None


def _search_macro() -> list[str]:
    return ["move_right", "move_up", "move_left", "move_down"]


def _deterministic_controller_actions(state_text: str) -> tuple[list[str], str]:
    normalized = str(state_text or "").lower()
    inventory = _parse_inventory_from_text(state_text)
    achievements = set(_parse_achievements_from_text(state_text))
    visible = set(_visible_objects_from_text(state_text))
    player_position = _parse_player_position_from_text(state_text)
    grounded = bool(inventory) or bool(visible)

    if not grounded:
        if player_position is not None:
            x, y = player_position
            phase = (abs(x) + abs(y)) % 4
            move = ("move_right", "move_up", "move_left", "move_down")[phase]
            return [move], "sparse state -> deterministic exploration"
        return [], "ungrounded state -> fallback to LLM"

    if "collect_wood" not in achievements:
        tree_cues = {"tree", "log", "trunk"}
        if visible & tree_cues or any(token in normalized for token in ("tree nearby", "tree adjacent", "tree to the", "tree on the")):
            return ["do"], "visible tree cue -> collect wood"
        if "sapling" in normalized or "plant" in normalized:
            return _search_macro(), "saplings are ignored -> search for trees"
        return [], "no tree cue grounded -> fallback to LLM"

    if inventory.get("wood", 0) > 0 and "place_table" not in achievements:
        return ["place_table"], "wood available -> place a table"

    if inventory.get("wood", 0) > 0 and "make_wood_pickaxe" not in achievements:
        return ["make_wood_pickaxe"], "table available -> craft a wood pickaxe"

    return [], "fallback to LLM"


def _exploratory_fallback_action(state_text: str) -> tuple[list[str], str]:
    normalized = str(state_text or "").lower()
    if "tree" in normalized or "log" in normalized or "trunk" in normalized:
        return ["do"], "tree cue -> exploit"
    if "table" in normalized and "wood" in normalized:
        return ["place_table"], "table cue -> place table"
    return ["move_right"], "unknown state -> lightweight exploration"


def _tool_call_payload(*, model: str, actions: list[str], reasoning_text: str) -> dict[str, Any]:
    safe_actions = [str(item).strip().lower() for item in actions if str(item).strip()]
    return {
        "id": f"chatcmpl_{os.urandom(8).hex()}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning_text,
                    "tool_calls": [
                        {
                            "id": f"call_{os.urandom(8).hex()}",
                            "type": "function",
                            "function": {
                                "name": PRIMARY_TOOL_NAME,
                                "arguments": {
                                    "actions_list": safe_actions,
                                },
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@contextmanager
def _craftax_policy_proxy(
    *,
    fallback_chat_url: str,
    fallback_api_key: str,
    fallback_model: str,
    model_name: str,
) -> Iterator[tuple[str, dict[str, int]]]:
    stats = {"controller_hits": 0, "fallback_hits": 0}
    resolved_fallback_url = str(fallback_chat_url or "").rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - noisy server logs
            del format, args

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                content_length = 0
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
            return payload if isinstance(payload, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in {"/health", "/v1/health"}:
                self._send_json(200, {"ok": True, "mode": "craftax-proxy"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path not in {"/chat/completions", "/v1/chat/completions"}:
                self._send_json(404, {"error": "not found"})
                return
            payload = self._read_body()
            messages = payload.get("messages")
            if not isinstance(messages, list):
                messages = []
            state_text = _extract_user_observation([item for item in messages if isinstance(item, dict)])
            actions, reasoning_text = _deterministic_controller_actions(state_text)
            if actions:
                stats["controller_hits"] += 1
                self._send_json(
                    200,
                    _tool_call_payload(
                        model=str(payload.get("model") or model_name or "craftax-proxy"),
                        actions=actions,
                        reasoning_text=reasoning_text,
                    ),
                )
                return

            stats["fallback_hits"] += 1
            try:
                if not str(fallback_model or "").strip() or not resolved_fallback_url:
                    actions, reasoning_text = _exploratory_fallback_action(state_text)
                    self._send_json(
                        200,
                        _tool_call_payload(
                            model=str(payload.get("model") or model_name or "craftax-proxy"),
                            actions=actions,
                            reasoning_text=reasoning_text,
                        ),
                    )
                    return
                fallback_payload = dict(payload)
                standard_keys = {
                    "model",
                    "messages",
                    "temperature",
                    "max_tokens",
                    "tools",
                    "tool_choice",
                    "logprobs",
                    "seed",
                    "top_p",
                    "presence_penalty",
                    "frequency_penalty",
                    "response_format",
                    "stop",
                    "n",
                    "user",
                }
                extra_body = {
                    key: value
                    for key, value in fallback_payload.items()
                    if key not in standard_keys and key not in {"chat_template_kwargs", "vllm_xargs"}
                }
                response = create_chat_completion(
                    model=str(fallback_model or payload.get("model") or model_name or "craftax-proxy"),
                    messages=[item for item in messages if isinstance(item, dict)],
                    max_tokens=int(fallback_payload.get("max_tokens") or fallback_payload.get("max_completion_tokens") or 128),
                    temperature=float(fallback_payload.get("temperature") or 0.0),
                    base_url=resolved_fallback_url,
                    api_key=fallback_api_key,
                    timeout_seconds=300.0,
                    tools=fallback_payload.get("tools") if isinstance(fallback_payload.get("tools"), list) else None,
                    tool_choice=fallback_payload.get("tool_choice"),
                    extra_body=extra_body or None,
                )
                self._send_json(200, response if isinstance(response, dict) else {"error": "fallback payload not an object"})
            except Exception as exc:  # pragma: no cover - depends on local model availability
                self._send_json(
                    500,
                    {
                        "error": f"proxy fallback failed: {type(exc).__name__}",
                        "detail": str(exc)[:2000],
                    },
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="craftax-policy-proxy", daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions", stats)
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def _write_eval_report(*, out_dir: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    report_lines = [
        "# NanoHorizon honest train-seed eval",
        "",
        f"- policy_version: `{config.get('policy_version', _DEFAULT_POLICY_VERSION)}`",
        f"- base_model: `{summary.get('base_model')}`",
        f"- seeds_evaluated: `{summary.get('requested_num_eval_rollouts')}`",
        f"- mean_outcome_reward: `{summary.get('mean_outcome_reward')}`",
        f"- max_outcome_reward: `{summary.get('max_outcome_reward')}`",
        f"- controller_hits: `{summary.get('controller_hits', 0)}`",
        f"- fallback_hits: `{summary.get('fallback_hits', 0)}`",
        "",
        "Per-rollout evidence:",
    ]
    for detail in summary.get("details") or []:
        if not isinstance(detail, dict):
            continue
        if detail.get("error"):
            report_lines.append(f"- seed {detail.get('seed')}: error={detail.get('error')}")
            continue
        report_lines.append(
            f"- seed {detail.get('seed')}: reward={detail.get('outcome_reward')}, "
            f"llm_calls={detail.get('llm_call_count')}, achievements={detail.get('achievements') or []}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")


def eval(checkpoint_dir: Path, data_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {"define": define()}
    )
    config = checkpoint.get("define") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict):
        config = define()

    seeds = _maybe_limit_seeds(_resolve_seeds(data_dir, config))
    base_model = str(config.get("base_model", "Qwen/Qwen3.5-4B"))
    max_steps = int(config.get("max_steps", 10))
    max_new_tokens = int(config.get("max_new_tokens", 128))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", 0))
    enable_thinking = bool(config.get("enable_thinking", False))
    target_action_batch_size = int(config.get("target_action_batch_size", 1))
    min_action_batch_size = int(config.get("min_action_batch_size", 1))
    system_prompt = str(config.get("system_prompt", ""))

    rollout_root = out_dir / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)

    vllm_config = LocalVLLMEvalConfig(
        model=base_model,
        served_model_name=base_model,
        max_model_len=int(config.get("max_length", 8192)),
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        enforce_eager=False,
    )
    fallback_base_url = str(
        os.getenv("NANOHORIZON_SUBMISSION_FALLBACK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    fallback_api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    fallback_model = str(
        os.getenv("NANOHORIZON_SUBMISSION_FALLBACK_MODEL")
        or "gpt-4.1-mini"
    ).strip()
    local_vllm_bin = str(vllm_config.vllm_bin or "").strip()
    use_local_vllm = bool(local_vllm_bin and (Path(local_vllm_bin).exists() or shutil.which(local_vllm_bin)))

    if use_local_vllm:
        with local_vllm_server(config=vllm_config, log_path=out_dir / "vllm_eval_server.log") as server:
            fallback_chat_url = str(server["base_url"])
            proxy_fallback_api_key = ""
            proxy_fallback_model = base_model
            with _craftax_policy_proxy(
                fallback_chat_url=fallback_chat_url,
                fallback_api_key=proxy_fallback_api_key,
                fallback_model=proxy_fallback_model,
                model_name=base_model,
            ) as (proxy_url, proxy_stats):
                results, rollout_summary = asyncio.run(
                    collect_rollouts_concurrently_with_summary(
                        container_url="direct://local",
                        inference_url=proxy_url,
                        model=base_model,
                        api_key="",
                        seeds=seeds,
                        max_steps=max_steps,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        max_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                        thinking_budget_tokens=thinking_budget_tokens,
                        policy_version=str(config.get("policy_version", _DEFAULT_POLICY_VERSION)),
                        target_action_batch_size=target_action_batch_size,
                        min_action_batch_size=min_action_batch_size,
                        request_timeout_seconds=120.0,
                        max_concurrent_rollouts=1,
                        trace_prefix="submission_eval",
                        request_logprobs=False,
                    )
                )
    else:
        with _craftax_policy_proxy(
            fallback_chat_url=fallback_base_url,
            fallback_api_key=fallback_api_key,
            fallback_model=fallback_model,
            model_name=base_model,
        ) as (proxy_url, proxy_stats):
            results, rollout_summary = asyncio.run(
                collect_rollouts_concurrently_with_summary(
                    container_url="direct://local",
                    inference_url=proxy_url,
                    model=base_model,
                    api_key="",
                    seeds=seeds,
                    max_steps=max_steps,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget_tokens,
                    policy_version=str(config.get("policy_version", _DEFAULT_POLICY_VERSION)),
                    target_action_batch_size=target_action_batch_size,
                    min_action_batch_size=min_action_batch_size,
                    request_timeout_seconds=120.0,
                    max_concurrent_rollouts=1,
                    trace_prefix="submission_eval",
                    request_logprobs=False,
                )
            )

    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_counts: dict[str, int] = {}
    achievement_names: set[str] = set()
    for index, rollout in enumerate(results):
        rollout_seed = seeds[index] if index < len(seeds) else index
        rollout_dir = rollout_root / f"{index:05d}_{rollout_seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        detail = {
            "seed": int(rollout.get("metadata", {}).get("seed") or rollout.get("_request_seed") or rollout_seed)
            if isinstance(rollout, dict)
            else int(rollout_seed),
            "rollout_id": str(rollout.get("rollout_id") or f"rollout_{index:05d}") if isinstance(rollout, dict) else f"rollout_{index:05d}",
        }
        if isinstance(rollout, dict) and not rollout.get("error"):
            detail["outcome_reward"] = rollout_outcome_reward(rollout)
            detail["llm_call_count"] = rollout_llm_call_count(rollout)
            detail["achievements"] = rollout_achievements(rollout)
            rewards.append(float(detail["outcome_reward"]))
            llm_calls.append(float(detail["llm_call_count"]))
            for achievement in detail["achievements"]:
                achievement_names.add(achievement)
                achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
        else:
            detail["error"] = rollout.get("error") if isinstance(rollout, dict) else "missing rollout"
        details.append(detail)
        write_json(rollout_dir / "rollout.json", rollout if isinstance(rollout, dict) else {"error": "missing rollout"})

    requested = len(seeds)
    result = {
        "base_model": base_model,
        "policy_version": str(config.get("policy_version", _DEFAULT_POLICY_VERSION)),
        "requested_num_eval_rollouts": requested,
        "num_eval_rollouts": len([detail for detail in details if not detail.get("error")]),
        "num_rollout_errors": len([detail for detail in details if detail.get("error")]),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "mean_outcome_reward_over_requested_rollouts": (sum(rewards) / float(requested)) if requested else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "mean_llm_calls_per_rollout": mean(llm_calls) if llm_calls else 0.0,
        "achievement_names": sorted(achievement_names),
        "achievement_frequencies": {
            name: {
                "count": int(achievement_counts.get(name, 0)),
                "frequency": (float(achievement_counts.get(name, 0)) / float(requested)) if requested else 0.0,
            }
            for name in sorted(achievement_names)
        },
        "details": details,
        "seeds": seeds,
        "checkpoint": checkpoint,
        "controller_hits": int(proxy_stats.get("controller_hits", 0)),
        "fallback_hits": int(proxy_stats.get("fallback_hits", 0)),
        "proxy_mode": "deterministic-plus-fallback-llm",
        "rollout_summary": rollout_summary,
    }
    write_json(out_dir / "result.json", result)
    write_json(out_dir / "eval_summary.json", result)
    _write_eval_report(out_dir=out_dir, summary=result, config=config)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["define", "train", "eval"])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "out")
    args = parser.parse_args()
    if args.phase == "define":
        print(json.dumps(define(), indent=2, sort_keys=True))
        return 0
    if args.phase == "train":
        train(args.data_dir, args.out_dir)
        return 0
    print(json.dumps(eval(args.checkpoint_dir, args.data_dir, args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
