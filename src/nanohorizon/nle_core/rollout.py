from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from nanohorizon.craftax_core.media import persist_media
from nanohorizon.craftax_core.modalities import RenderMode
from nanohorizon.custom_vllm.runtime import build_thinking_budget_request_overrides

from .metadata import (
    ACTION_NAME_TO_VALUE,
    DEFAULT_ACTION_NAMES,
    PRIMARY_TOOL_NAME,
    safe_fallback_action_name,
)
from .runner import make_runner, native_score_from_observation


def sanitize_nle_actions(values: list[object]) -> list[str]:
    sanitized: list[str] = []
    for value in values:
        raw = str(value).strip().lower()
        if not raw:
            continue
        if raw in ACTION_NAME_TO_VALUE and raw not in sanitized:
            sanitized.append(raw)
            continue
        for token in re.findall(r"[a-z0-9_]+", raw):
            if token in ACTION_NAME_TO_VALUE and token not in sanitized:
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
        for key in ("actions_list", "actions"):
            values = arguments.get(key)
            if isinstance(values, list):
                actions = sanitize_nle_actions(values)
                if actions:
                    return actions
    return []


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


def _tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": PRIMARY_TOOL_NAME,
                "description": "Choose the next short primitive NetHack/NLE action sequence.",
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
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "tools": _tool_schema(),
        "tool_choice": "auto",
        "max_tokens": int(max_tokens),
    }
    request_body.update(
        build_thinking_budget_request_overrides(
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget_tokens,
        )
    )
    if request_logprobs:
        request_body["logprobs"] = True
    timeout = httpx.Timeout(float(timeout_s), connect=min(30.0, float(timeout_s)))
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.post(inference_url, headers=headers, json=request_body)
    if response.status_code >= 400:
        raise RuntimeError(f"inference request failed status={response.status_code} body={response.text[:2000]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("inference response was not an object")
    return payload


def _observation_prompt(*, observation_text: str, target_action_batch_size: int) -> str:
    return (
        "Current NetHack/NLE observation:\n"
        f"{observation_text}\n\n"
        "Plan a short useful primitive action sequence. "
        f"Use the {PRIMARY_TOOL_NAME} tool exactly once. "
        f"Return exactly {target_action_batch_size} actions unless the episode is already done. "
        "Use only valid primitive NLE action names. Do not return JSON or plain text actions."
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
    target_action_batch_size: int = 4,
    min_action_batch_size: int = 1,
    timeout_s: int = 45,
    render_mode: RenderMode = RenderMode.TEXT,
    media: Mapping[str, Any] | None = None,
    env_id: str = "NetHackChallenge-v0",
    max_episode_steps: int | None = None,
    savedir: str | None = None,
    request_logprobs: bool = True,
) -> dict[str, Any]:
    runner = make_runner(
        seed=int(seed),
        render_mode=render_mode,
        env_id=env_id,
        max_episode_steps=max_episode_steps,
        savedir=savedir,
    )
    current = runner.reset()
    total_reward = 0.0
    llm_call_count = 0
    frames: list[Any] = []
    if current.render.pixels is not None:
        frames.append(current.render.pixels)
    turns: list[dict[str, Any]] = []
    fallback = safe_fallback_action_name()

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
        invalid_parse = len(actions) < int(min_action_batch_size)
        if invalid_parse:
            repair_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid. Use {PRIMARY_TOOL_NAME} exactly once and "
                        f"return valid primitive NLE actions."
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
            actions = [fallback]
        step_outputs = runner.step_many([ACTION_NAME_TO_VALUE[action] for action in actions if action in ACTION_NAME_TO_VALUE])
        if not step_outputs:
            step_outputs = runner.step_many([ACTION_NAME_TO_VALUE[fallback]])
        decision_reward = float(sum(item.reward for item in step_outputs))
        total_reward += decision_reward
        current = step_outputs[-1]
        if current.render.pixels is not None:
            frames.append(current.render.pixels)
        turns.append(
            {
                "turn_index": turn_index,
                "prompt_messages": prompt_messages,
                "assistant_text": str(message.get("content") or ""),
                "reasoning_text": _extract_reasoning_text(message),
                "actions": actions,
                "decision_reward": decision_reward,
                "return_to_go": float(total_reward),
                "trainable": not invalid_parse,
                "invalid_parse": invalid_parse,
                "behavior_sequence_logprob": _extract_sequence_logprob(payload) or 0.0,
            }
        )
        if current.done:
            break

    media_payload: dict[str, Any] = {}
    if media and frames:
        output_dir = str(media.get("output_dir") or Path("/tmp") / f"nle_media_{uuid.uuid4().hex[:8]}")
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
    native_score = native_score_from_observation(runner.last_observation)
    return {
        "rollout_id": rollout_id,
        "trace_correlation_id": trace_correlation_id,
        "trial_id": trace_correlation_id,
        "policy_version": policy_version,
        "success_status": "success",
        "reward_info": {
            "outcome_reward": float(total_reward),
            "outcome_objectives": {
                "scout_score": float(total_reward),
                "reward": float(total_reward),
                "native_nle_score": float(native_score),
            },
            "details": {
                "scout_score": float(total_reward),
                "native_nle_score": float(native_score),
                "llm_call_count": int(llm_call_count),
            },
        },
        "trace": {"inference": {"turns": turns}},
        "metadata": {
            "environment_family": "nle",
            "task_id": "nethack_scout",
            "llm_call_count": int(llm_call_count),
            "action_history": list(runner.action_history),
            "seed": int(seed),
            "render_mode": render_mode.value,
            "env_id": env_id,
            "scout_score": float(total_reward),
            "native_nle_score": float(native_score),
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
    max_episode_steps = env_config.get("max_episode_steps")
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
        target_action_batch_size=int(policy_config.get("target_action_batch_size") or 4),
        min_action_batch_size=int(policy_config.get("min_action_batch_size") or 1),
        timeout_s=int(policy_config.get("timeout_s") or 45),
        render_mode=RenderMode.BOTH if media else RenderMode.TEXT,
        media=media if isinstance(media, Mapping) else None,
        env_id=str(env_config.get("env_id") or "NetHackChallenge-v0"),
        max_episode_steps=int(max_episode_steps) if max_episode_steps is not None else None,
        savedir=str(env_config.get("savedir") or "").strip() or None,
        request_logprobs=bool(policy_config.get("request_logprobs", True)),
    )


def collect_rollouts(*, requests: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
