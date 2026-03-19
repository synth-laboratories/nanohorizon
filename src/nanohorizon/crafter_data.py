from __future__ import annotations

from typing import Any


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        content = message.get("content")
        if isinstance(content, list):
            content_text = "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        else:
            content_text = str(content or "")
        rendered.append(f"{role}: {content_text}")
    return "\n".join(rendered).strip()


def rollout_turns(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = rollout.get("artifact")
    if isinstance(artifact, list):
        for entry in artifact:
            if isinstance(entry, dict) and isinstance(entry.get("turns"), list):
                return [turn for turn in entry["turns"] if isinstance(turn, dict)]
    trace = rollout.get("trace")
    if isinstance(trace, dict):
        inference = trace.get("inference")
        if isinstance(inference, dict) and isinstance(inference.get("turns"), list):
            return [turn for turn in inference["turns"] if isinstance(turn, dict)]
    return []


def rollout_outcome_reward(rollout: dict[str, Any]) -> float:
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        try:
            return float(reward_info.get("outcome_reward", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def build_sft_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            examples.append(
                {
                    "prompt": flatten_messages(messages[:-1]) if len(messages) > 1 else "",
                    "response": str(messages[-1].get("content") if isinstance(messages[-1], dict) else ""),
                    "weight": 1.0,
                    "metadata": row.get("metadata", {}),
                }
            )
            continue
        prompt = str(row.get("prompt") or "")
        response = str(row.get("response") or "")
        if prompt and response:
            examples.append({"prompt": prompt, "response": response, "weight": 1.0, "metadata": row.get("metadata", {})})
    return [example for example in examples if example["response"].strip()]


def build_rlvr_examples(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    rewards = [rollout_outcome_reward(rollout) for rollout in rollouts]
    min_reward = min(rewards) if rewards else 0.0
    max_reward = max(rewards) if rewards else 1.0
    denom = max(max_reward - min_reward, 1e-6)
    for rollout in rollouts:
        outcome_reward = rollout_outcome_reward(rollout)
        normalized = 0.1 + 0.9 * ((outcome_reward - min_reward) / denom)
        for turn in rollout_turns(rollout):
            messages = turn.get("prompt_messages")
            assistant_text = str(turn.get("assistant_text") or "")
            if not isinstance(messages, list) or not assistant_text.strip():
                continue
            examples.append(
                {
                    "prompt": flatten_messages(messages),
                    "response": assistant_text,
                    "weight": float(turn.get("decision_reward", normalized) or normalized),
                    "metadata": {
                        "rollout_id": rollout.get("rollout_id"),
                        "trace_correlation_id": rollout.get("trace_correlation_id"),
                        "outcome_reward": outcome_reward,
                    },
                }
            )
    return examples
