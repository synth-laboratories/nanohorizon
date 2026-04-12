from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse

import httpx

from nanohorizon.shared.common import ensure_dir
from nanohorizon.custom_vllm.runtime import build_thinking_budget_request_overrides

from .media import persist_media
from .metadata import (
    DEFAULT_ACHIEVEMENT_NAMES,
    DEFAULT_ACTION_NAMES,
    PRIMARY_TOOL_NAME,
)
from .modalities import RenderMode
from .upstream import achievement_names_from_state, action_name_to_index, make_runner

ACTION_NAME_TO_INDEX = action_name_to_index()


def _sanitize_actions(values: list[object]) -> list[str]:
    sanitized: list[str] = []
    for value in values:
        raw = str(value).strip().lower()
        if not raw:
            continue
        if raw in ACTION_NAME_TO_INDEX and raw not in sanitized:
            sanitized.append(raw)
            continue
        for token in re.findall(r"[a-z_]+", raw):
            if token in ACTION_NAME_TO_INDEX and token not in sanitized:
                sanitized.append(token)
    return sanitized


def _extract_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            return message
    return {}


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    if message.get("reasoning_content") is not None:
        return str(message.get("reasoning_content") or "").strip()
    if message.get("reasoning") is not None:
        return str(message.get("reasoning") or "").strip()
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                chunks.append(str(item.get("text") or ""))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return ""


def _extract_sequence_logprob(payload: dict[str, Any]) -> float | None:
    try:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        logprobs = choices[0].get("logprobs")
        if isinstance(logprobs, dict):
            value = logprobs.get("sequence_logprob")
            if value is not None:
                return float(value)
    except Exception:
        return None
    return None


def _extract_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = _extract_message(payload)
    parsed: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name != PRIMARY_TOOL_NAME:
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if isinstance(arguments, dict):
                parsed.append({"name": name, "arguments": arguments})
    if parsed:
        return parsed
    content = message.get("content")
    if isinstance(content, str):
        for raw_block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL):
            try:
                parsed_block = json.loads(raw_block.strip())
            except Exception:
                continue
            if not isinstance(parsed_block, dict):
                continue
            name = str(parsed_block.get("name") or "").strip()
            arguments = parsed_block.get("arguments", {})
            if name == PRIMARY_TOOL_NAME and isinstance(arguments, dict):
                parsed.append({"name": name, "arguments": arguments})
    return parsed


def _extract_actions(payload: dict[str, Any]) -> list[str]:
    for tool_call in _extract_tool_calls(payload):
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        values = arguments.get("actions_list")
        if isinstance(values, list):
            actions = _sanitize_actions(values)
            if actions:
                return actions
    return []


def _tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": PRIMARY_TOOL_NAME,
                "description": "Choose the next short Craftax macro-action sequence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions_list": {
                            "type": "array",
                            "items": {"type": "string", "enum": DEFAULT_ACTION_NAMES},
                            "minItems": 1,
                            "maxItems": 10,
                        }
                    },
                    "required": ["actions_list"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _chat_completion(
    *,
    inference_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    thinking_budget_tokens: int,
    timeout_s: int,
    request_logprobs: bool = True,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    parsed = urlparse(inference_url)
    hostname = str(parsed.hostname or "").lower()
    supports_vllm_request_overrides = hostname in {"127.0.0.1", "localhost"}
    uses_proxy_edge = (
        hostname.endswith(".modal.run")
        or hostname.endswith(".w.modal.host")
        or hostname.endswith(".trycloudflare.com")
    )
    if uses_proxy_edge:
        headers["Connection"] = "close"
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "tools": _tool_schema(),
        "tool_choice": "required" if supports_vllm_request_overrides else "auto",
    }
    if "api.openai.com" in inference_url:
        request_body["max_completion_tokens"] = int(max_tokens)
    else:
        request_body["max_tokens"] = int(max_tokens)
    thinking_overrides = build_thinking_budget_request_overrides(
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget_tokens,
    )
    if not supports_vllm_request_overrides:
        # Remote OpenAI-compatible teacher endpoints in our Modal flow do not
        # accept vLLM-specific request extensions such as chat_template_kwargs.
        thinking_overrides.pop("chat_template_kwargs", None)
        thinking_overrides.pop("vllm_xargs", None)
    request_body.update(thinking_overrides)
    if request_logprobs:
        request_body["logprobs"] = True
    timeout = httpx.Timeout(float(timeout_s), connect=min(30.0, float(timeout_s)))
    last_error: Exception | None = None
    retryable_statuses = {429, 500, 502, 503, 504}
    if uses_proxy_edge:
        retryable_statuses.add(404)
    for attempt in range(1, 7):
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"Connection": "close"} if uses_proxy_edge else None,
            ) as client:
                response = client.post(inference_url, headers=headers, json=request_body)
            if response.status_code in retryable_statuses and attempt < 6:
                time.sleep(min(5.0, float(attempt)))
                continue
            if response.status_code >= 400:
                body_preview = response.text[:2000]
                raise RuntimeError(
                    f"inference request failed status={response.status_code} body={body_preview}"
                )
            payload = response.json()
            break
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt >= 6:
                raise RuntimeError(f"inference request transport failed after retries: {exc!r}") from exc
            time.sleep(min(5.0, float(attempt)))
    else:
        if last_error is not None:
            raise RuntimeError(f"inference request failed after retries: {last_error!r}") from last_error
        raise RuntimeError("inference request failed after retries")
    if not isinstance(payload, dict):
        raise RuntimeError("inference response was not an object")
    return payload


def _recent_turns_prompt(turns: list[dict[str, Any]], *, max_turns: int = 3) -> str:
    recent_turns = [turn for turn in turns if isinstance(turn, dict)][-max_turns:]
    if not recent_turns:
        return ""

    lines = ["Recent trajectory:"]
    for turn in recent_turns:
        turn_index = turn.get("turn_index")
        actions = turn.get("actions")
        if isinstance(actions, list) and actions:
            actions_text = ", ".join(str(action) for action in actions)
        else:
            actions_text = "none"
        reward = turn.get("decision_reward")
        try:
            reward_text = f"{float(reward):.2f}"
        except (TypeError, ValueError):
            reward_text = "n/a"
        try:
            rtg_text = f"{float(turn.get('return_to_go')):.2f}"
        except (TypeError, ValueError):
            rtg_text = "n/a"
        lines.append(
            f"- turn {turn_index}: actions={actions_text}; reward={reward_text}; return_to_go={rtg_text}"
        )
    return "\n".join(lines)


def _observation_prompt(
    *,
    observation_text: str,
    target_action_batch_size: int,
    recent_turns: list[dict[str, Any]] | None = None,
) -> str:
    parts = [
        "Current Craftax long-horizon observation:",
        observation_text,
    ]
    recent_turns_text = _recent_turns_prompt(list(recent_turns or ()))
    if recent_turns_text:
        parts.extend([recent_turns_text])
    parts.append(
        (
            "Plan a short useful macro-action. "
            "If the observation names a nearby resource, machine, or other actionable target, "
            "prefer the matching interact/craft action; otherwise explore with non-repeating movement. "
            "If the recent trajectory shows no progress, do not mirror the same movement pattern again. "
            f"Use the {PRIMARY_TOOL_NAME} tool exactly once. "
            f"Return exactly {target_action_batch_size} actions unless the environment is already done. "
            "Use only valid full-Craftax actions. Do not return JSON or plain text actions."
        )
    )
    return "\n\n".join(parts)


def run_rollout(
    *,
    inference_url: str,
    model: str,
    api_key: str,
    seed: int,
    max_steps: int,
    trace_correlation_id: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 180,
    enable_thinking: bool = False,
    thinking_budget_tokens: int = 0,
    policy_version: str = "bootstrap",
    target_action_batch_size: int = 8,
    min_action_batch_size: int = 5,
    timeout_s: int = 45,
    render_mode: RenderMode = RenderMode.TEXT,
    media: Mapping[str, Any] | None = None,
    env_kind: str = "full",
    request_logprobs: bool = True,
) -> dict[str, Any]:
    runner = make_runner(kind="full" if env_kind == "full" else "classic", seed=int(seed), render_mode=render_mode)
    start = runner.reset()
    current = start
    unique_achievements: set[str] = set()
    total_reward = 0.0
    llm_call_count = 0
    frames: list[Any] = []
    if current.render.pixels is not None:
        frames.append(current.render.pixels)
    turns: list[dict[str, Any]] = []
    for turn_index in range(max(1, int(max_steps))):
        if current.done:
            break
        observation_text = str(current.render.text or "").strip() or "No text renderer available."
        user_prompt = _observation_prompt(
            observation_text=observation_text,
            target_action_batch_size=max(1, int(target_action_batch_size)),
            recent_turns=turns,
        )
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = _chat_completion(
            inference_url=inference_url,
            model=model,
            api_key=api_key,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            timeout_s=timeout_s,
            request_logprobs=request_logprobs,
        )
        llm_call_count += 1
        message = _extract_message(payload)
        actions = _extract_actions(payload)
        invalid_parse = False
        if len(actions) < int(min_action_batch_size):
            invalid_parse = True
            repair_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid. Use {PRIMARY_TOOL_NAME} exactly once and "
                        f"return exactly {int(target_action_batch_size)} valid full-Craftax actions."
                    ),
                },
            ]
            payload = _chat_completion(
                inference_url=inference_url,
                model=model,
                api_key=api_key,
                messages=repair_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                timeout_s=timeout_s,
                request_logprobs=request_logprobs,
            )
            llm_call_count += 1
            message = _extract_message(payload)
            actions = _extract_actions(payload)
            prompt_messages = repair_messages
        if not actions:
            actions = ["noop"]
        step_outputs = runner.step_many([ACTION_NAME_TO_INDEX[action] for action in actions if action in ACTION_NAME_TO_INDEX])
        if not step_outputs:
            step_outputs = runner.step_many([ACTION_NAME_TO_INDEX["noop"]])
        decision_reward = float(sum(item.reward for item in step_outputs))
        total_reward += decision_reward
        current = step_outputs[-1]
        if current.render.pixels is not None:
            frames.append(current.render.pixels)
        achievements = achievement_names_from_state(runner.state)
        unique_achievements.update(achievements)
        turns.append(
            {
                "turn_index": turn_index,
                "prompt_messages": prompt_messages,
                "assistant_text": str(message.get("content") or ""),
                "reasoning_text": _extract_reasoning_text(message),
                "actions": actions,
                "decision_reward": decision_reward,
                "return_to_go": float(len(unique_achievements)),
                "trainable": not invalid_parse,
                "invalid_parse": invalid_parse,
                "behavior_sequence_logprob": _extract_sequence_logprob(payload) or 0.0,
            }
        )
        if current.done:
            break
    media_payload: dict[str, Any] = {}
    if media and frames:
        output_dir = str(media.get("output_dir") or Path("/tmp") / f"craftax_media_{uuid.uuid4().hex[:8]}")
        artifacts = persist_media(
            frames=frames,
            output_dir=output_dir,
            fps=int(media.get("fps", 6)),
            write_mp4=True,
        )
        media_payload = {
            "frames_dir": artifacts.frames_dir,
            "gif_path": artifacts.gif_path,
            "mp4_path": artifacts.mp4_path,
        }
    rollout_id = f"rollout_{uuid.uuid4().hex}"
    return {
        "rollout_id": rollout_id,
        "trace_correlation_id": trace_correlation_id,
        "trial_id": trace_correlation_id,
        "policy_version": policy_version,
        "success_status": "success",
        "reward_info": {
            "outcome_reward": float(len(unique_achievements)),
            "outcome_objectives": {
                "unique_achievements": float(len(unique_achievements)),
                "reward": float(len(unique_achievements)),
                "native_env_reward_total": float(total_reward),
            },
            "details": {
                "achievements": sorted(unique_achievements),
                "native_env_reward_total": float(total_reward),
                "llm_call_count": int(llm_call_count),
            },
        },
        "trace": {"inference": {"turns": turns}},
        "metadata": {
            "llm_call_count": int(llm_call_count),
            "achievements": sorted(unique_achievements),
            "action_history": list(runner.action_history),
            "seed": int(seed),
            "render_mode": render_mode.value,
            "env_kind": env_kind,
        },
        "artifact": [{"turns": turns}],
        "media": media_payload,
    }


def run_rollout_request(request: Mapping[str, Any]) -> dict[str, Any]:
    env = request.get("env", {})
    policy = request.get("policy", {})
    policy_config = policy.get("config", {}) if isinstance(policy, Mapping) else {}
    env_config = env.get("config", {}) if isinstance(env, Mapping) else {}
    media = request.get("media")
    return run_rollout(
        inference_url=str(policy_config.get("inference_url") or "").strip(),
        model=str(policy_config.get("model") or "").strip(),
        api_key=str(policy_config.get("api_key") or "").strip(),
        seed=int(env.get("seed") or 0),
        max_steps=int(env_config.get("episode_max_steps") or env_config.get("max_steps") or 1),
        trace_correlation_id=str(request.get("trace_correlation_id") or request.get("trial_id") or uuid.uuid4().hex),
        system_prompt=str(policy_config.get("system_prompt") or "").strip(),
        temperature=float(policy_config.get("temperature") or 0.0),
        max_tokens=int(policy_config.get("max_tokens") or 180),
        enable_thinking=bool(policy_config.get("enable_thinking", False)),
        thinking_budget_tokens=int(policy_config.get("thinking_budget_tokens") or 0),
        policy_version=str(policy_config.get("policy_version") or "bootstrap"),
        target_action_batch_size=int(policy_config.get("target_action_batch_size") or 8),
        min_action_batch_size=int(policy_config.get("min_action_batch_size") or 5),
        timeout_s=int(policy_config.get("timeout_s") or 45),
        render_mode=RenderMode.BOTH if media else RenderMode.TEXT,
        media=media if isinstance(media, Mapping) else None,
        env_kind=str(env_config.get("env_kind") or "full"),
        request_logprobs=bool(policy_config.get("request_logprobs", True)),
    )


def collect_rollouts(
    *,
    requests: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.perf_counter()
    results = [run_rollout_request(request) for request in requests]
    rewards = [float(item.get("reward_info", {}).get("outcome_reward", 0.0)) for item in results]
    elapsed_s = max(1e-9, time.perf_counter() - started_at)
    summary = {
        "requested_rollouts": len(requests),
        "completed_rollouts": len(results),
        "num_errors": 0,
        "num_structured_rollouts": len(results),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "elapsed_s": elapsed_s,
        "rollouts_per_minute": len(results) / (elapsed_s / 60.0),
    }
    return results, summary
