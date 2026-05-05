from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def prewarm_one_step(
    *,
    env_kind: str = "full",
    render_mode: RenderMode = RenderMode.TEXT,
    block_pixel_size: int | None = None,
) -> None:
    runner = make_runner(
        kind="full" if env_kind == "full" else "classic",
        seed=0,
        render_mode=render_mode,
        block_pixel_size=block_pixel_size,
    )
    runner.reset()
    runner.step(ACTION_NAME_TO_INDEX["noop"])


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
                trust_env=False,
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


def _observation_prompt(*, observation_text: str, target_action_batch_size: int) -> str:
    return (
        "Current Craftax long-horizon observation:\n"
        f"{observation_text}\n\n"
        "Plan a short useful macro-action. "
        f"Use the {PRIMARY_TOOL_NAME} tool exactly once. "
        f"Return exactly {target_action_batch_size} actions unless the environment is already done. "
        "Use only valid full-Craftax actions. Do not return JSON or plain text actions."
    )


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
            write_mp4=bool(media.get("write_mp4", True)),
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


def persist_rollout_media_from_history(
    *,
    rollout: Mapping[str, Any],
    media: Mapping[str, Any],
    env_kind: str = "full",
) -> dict[str, Any]:
    metadata = rollout.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    seed = int(metadata.get("seed") or rollout.get("_request_seed") or 0)
    raw_actions = metadata.get("action_history")
    actions = [int(action) for action in raw_actions] if isinstance(raw_actions, list) else []
    runner = make_runner(
        kind="full" if env_kind == "full" else "classic",
        seed=seed,
        render_mode=RenderMode.BOTH,
        block_pixel_size=int(media.get("tile_size") or 16),
    )
    current = runner.reset()
    frames: list[Any] = []
    if current.render.pixels is not None:
        frames.append(current.render.pixels)
    for action in actions:
        current = runner.step(int(action))
        if current.render.pixels is not None:
            frames.append(current.render.pixels)
        if current.done:
            break
    if not frames:
        return {}
    output_dir = str(media.get("output_dir") or Path("/tmp") / f"craftax_media_{uuid.uuid4().hex[:8]}")
    artifacts = persist_media(
        frames=frames,
        output_dir=output_dir,
        fps=int(media.get("fps", 6)),
        write_mp4=bool(media.get("write_mp4", True)),
    )
    return {
        "frames_dir": artifacts.frames_dir,
        "gif_path": artifacts.gif_path,
        "mp4_path": artifacts.mp4_path,
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


def _rollout_request_context(request: Mapping[str, Any]) -> dict[str, Any]:
    env = request.get("env", {})
    policy = request.get("policy", {})
    policy_config = policy.get("config", {}) if isinstance(policy, Mapping) else {}
    env_config = env.get("config", {}) if isinstance(env, Mapping) else {}
    return {
        "inference_url": str(policy_config.get("inference_url") or "").strip(),
        "model": str(policy_config.get("model") or "").strip(),
        "api_key": str(policy_config.get("api_key") or "").strip(),
        "seed": int(env.get("seed") or 0),
        "max_steps": int(env_config.get("episode_max_steps") or env_config.get("max_steps") or 1),
        "trace_correlation_id": str(request.get("trace_correlation_id") or request.get("trial_id") or uuid.uuid4().hex),
        "system_prompt": str(policy_config.get("system_prompt") or "").strip(),
        "temperature": float(policy_config.get("temperature") or 0.0),
        "max_tokens": int(policy_config.get("max_tokens") or 180),
        "enable_thinking": bool(policy_config.get("enable_thinking", False)),
        "thinking_budget_tokens": int(policy_config.get("thinking_budget_tokens") or 0),
        "policy_version": str(policy_config.get("policy_version") or "bootstrap"),
        "target_action_batch_size": int(policy_config.get("target_action_batch_size") or 8),
        "min_action_batch_size": int(policy_config.get("min_action_batch_size") or 5),
        "timeout_s": int(policy_config.get("timeout_s") or 45),
        "env_kind": str(env_config.get("env_kind") or "full"),
        "request_logprobs": bool(policy_config.get("request_logprobs", True)),
    }


def _initial_observation_context(context: Mapping[str, Any]) -> dict[str, Any]:
    use_static_initial_observation = (
        str(os.getenv("NANOHORIZON_CRAFTAX_STATIC_INITIAL_OBSERVATION") or "1").strip().lower()
        not in {"0", "false", "no"}
    )
    if use_static_initial_observation:
        observation_text = (
            "Fresh full-Craftax episode at the starting state. Choose a short safe opening "
            "macro-action sequence that begins collecting basic resources and avoids invalid actions."
        )
    else:
        runner = make_runner(
            kind="full" if context["env_kind"] == "full" else "classic",
            seed=int(context["seed"]),
            render_mode=RenderMode.TEXT,
        )
        current = runner.reset()
        observation_text = str(current.render.text or "").strip() or "No text renderer available."
    user_prompt = _observation_prompt(
        observation_text=observation_text,
        target_action_batch_size=max(1, int(context["target_action_batch_size"])),
    )
    return {
        **dict(context),
        "prompt_messages": [
            {"role": "system", "content": str(context["system_prompt"])},
            {"role": "user", "content": user_prompt},
        ],
        "user_prompt": user_prompt,
    }


def _complete_policy_call(context: Mapping[str, Any]) -> dict[str, Any]:
    payload = _chat_completion(
        inference_url=str(context["inference_url"]),
        model=str(context["model"]),
        api_key=str(context["api_key"]),
        messages=list(context["prompt_messages"]),
        temperature=float(context["temperature"]),
        max_tokens=int(context["max_tokens"]),
        enable_thinking=bool(context["enable_thinking"]),
        thinking_budget_tokens=int(context["thinking_budget_tokens"]),
        timeout_s=int(context["timeout_s"]),
        request_logprobs=bool(context["request_logprobs"]),
    )
    llm_call_count = 1
    message = _extract_message(payload)
    actions = _extract_actions(payload)
    invalid_parse = False
    prompt_messages = list(context["prompt_messages"])
    if len(actions) < int(context["min_action_batch_size"]):
        invalid_parse = True
        repair_messages = [
            {"role": "system", "content": str(context["system_prompt"])},
            {"role": "user", "content": str(context["user_prompt"])},
            {
                "role": "user",
                "content": (
                    f"Your previous response was invalid. Use {PRIMARY_TOOL_NAME} exactly once and "
                    f"return exactly {int(context['target_action_batch_size'])} valid full-Craftax actions."
                ),
            },
        ]
        payload = _chat_completion(
            inference_url=str(context["inference_url"]),
            model=str(context["model"]),
            api_key=str(context["api_key"]),
            messages=repair_messages,
            temperature=float(context["temperature"]),
            max_tokens=int(context["max_tokens"]),
            enable_thinking=bool(context["enable_thinking"]),
            thinking_budget_tokens=int(context["thinking_budget_tokens"]),
            timeout_s=int(context["timeout_s"]),
            request_logprobs=bool(context["request_logprobs"]),
        )
        llm_call_count += 1
        message = _extract_message(payload)
        actions = _extract_actions(payload)
        prompt_messages = repair_messages
    if not actions:
        actions = ["noop"]
    return {
        **dict(context),
        "payload": payload,
        "message": message,
        "actions": actions,
        "invalid_parse": invalid_parse,
        "prompt_messages": prompt_messages,
        "llm_call_count": llm_call_count,
    }


def _replay_one_step_rollout(context: Mapping[str, Any]) -> dict[str, Any]:
    runner = make_runner(
        kind="full" if context["env_kind"] == "full" else "classic",
        seed=int(context["seed"]),
        render_mode=RenderMode.TEXT,
    )
    runner.reset()
    actions = [str(action) for action in context["actions"]]
    step_outputs = runner.step_many([ACTION_NAME_TO_INDEX[action] for action in actions if action in ACTION_NAME_TO_INDEX])
    if not step_outputs:
        step_outputs = runner.step_many([ACTION_NAME_TO_INDEX["noop"]])
    decision_reward = float(sum(item.reward for item in step_outputs))
    achievements = achievement_names_from_state(runner.state)
    message = context["message"]
    payload = context["payload"]
    turn = {
        "turn_index": 0,
        "prompt_messages": list(context["prompt_messages"]),
        "assistant_text": str(message.get("content") or ""),
        "reasoning_text": _extract_reasoning_text(message),
        "actions": actions,
        "decision_reward": decision_reward,
        "return_to_go": float(len(achievements)),
        "trainable": not bool(context["invalid_parse"]),
        "invalid_parse": bool(context["invalid_parse"]),
        "behavior_sequence_logprob": _extract_sequence_logprob(payload) or 0.0,
    }
    return {
        "rollout_id": f"rollout_{uuid.uuid4().hex}",
        "trace_correlation_id": str(context["trace_correlation_id"]),
        "trial_id": str(context["trace_correlation_id"]),
        "policy_version": str(context["policy_version"]),
        "success_status": "success",
        "reward_info": {
            "outcome_reward": float(len(achievements)),
            "outcome_objectives": {
                "unique_achievements": float(len(achievements)),
                "reward": float(len(achievements)),
                "native_env_reward_total": decision_reward,
            },
            "details": {
                "achievements": sorted(achievements),
                "native_env_reward_total": decision_reward,
                "llm_call_count": int(context["llm_call_count"]),
            },
        },
        "trace": {"inference": {"turns": [turn]}},
        "metadata": {
            "llm_call_count": int(context["llm_call_count"]),
            "achievements": sorted(achievements),
            "action_history": list(runner.action_history),
            "seed": int(context["seed"]),
            "render_mode": RenderMode.TEXT.value,
            "env_kind": str(context["env_kind"]),
        },
        "artifact": [{"turns": [turn]}],
    }


def collect_one_step_rollouts_with_parallel_inference(
    *,
    requests: list[Mapping[str, Any]],
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.perf_counter()
    contexts = [_initial_observation_context(_rollout_request_context(request)) for request in requests]
    policy_results: list[dict[str, Any] | None] = [None] * len(contexts)
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_to_index = {
            executor.submit(_complete_policy_call, context): index for index, context in enumerate(contexts)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                policy_results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                policy_results[index] = {
                    "error": str(exc).strip() or f"{type(exc).__name__}: no detail",
                    "seed": contexts[index]["seed"],
                    "trace_correlation_id": contexts[index]["trace_correlation_id"],
                }
    results: list[dict[str, Any]] = []
    for item in policy_results:
        if not isinstance(item, dict) or item.get("error"):
            results.append(item if isinstance(item, dict) else {"error": "missing policy result"})
            continue
        try:
            payload = _replay_one_step_rollout(item)
            payload.setdefault("_request_seed", int(item["seed"]))
            results.append(payload)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "error": str(exc).strip() or f"{type(exc).__name__}: no detail",
                    "seed": item["seed"],
                    "trace_correlation_id": item["trace_correlation_id"],
                }
            )
    valid_results = [item for item in results if not item.get("error")]
    rewards = [float(item.get("reward_info", {}).get("outcome_reward", 0.0)) for item in valid_results]
    elapsed_s = max(1e-9, time.perf_counter() - started_at)
    summary = {
        "requested_rollouts": len(requests),
        "completed_rollouts": len(results),
        "num_errors": len(results) - len(valid_results),
        "num_structured_rollouts": len(valid_results),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "elapsed_s": elapsed_s,
        "rollouts_per_minute": len(valid_results) / (elapsed_s / 60.0),
        "rollout_concurrency": max(1, int(max_workers)),
        "rollout_semaphore_limit": max(1, int(max_workers)),
        "rollout_requests_started": len(requests),
        "rollout_requests_finished": len(results),
        "active_rollout_high_watermark": min(len(requests), max(1, int(max_workers))),
    }
    return results, summary


def collect_rollouts(
    *,
    requests: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contexts = [_rollout_request_context(request) for request in requests]
    if contexts and all(int(context["max_steps"]) == 1 for context in contexts):
        return collect_one_step_rollouts_with_parallel_inference(
            requests=requests,
            max_workers=len(requests),
        )
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
