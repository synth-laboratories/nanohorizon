from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import re
import shlex
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import yaml

from nanohorizon.custom_vllm.runtime import build_thinking_budget_request_overrides
from nanohorizon.shared import train_lora as train_lora_lib
from nanohorizon.shared.common import (
    Timer,
    ensure_dir,
    load_config,
    now_utc_iso,
    read_jsonl,
    resolve_path,
    system_info,
    write_json,
    write_text,
)
from nanohorizon.shared.craftax_data import (
    CRAFTAX_INTERACT_TOOL,
    flatten_messages,
    is_rollout_payload,
    rollout_outcome_reward,
    rollout_turns,
)
from nanohorizon.shared.openai_compat import create_chat_completion, extract_openai_tool_calls
from nanohorizon.shared.train_lora import (
    release_cuda_memory,
    train_sft_with_trl,
    train_weighted_lora,
)
from nanohorizon.shared.vllm_eval import LocalVLLMEvalConfig, local_vllm_server

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TRACK_ID = "pivot_verifier_qwen35_4b"
DEFAULT_METHOD_NAME = "spct_qwen35_4b_baseline"
DEFAULT_VERIFIER_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-9B"
RESOURCE_KEYS = (
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
VISIBLE_OBJECT_KEYS = (
    "tree",
    "stone",
    "table",
    "furnace",
    "plant",
    "cow",
    "water",
    "zombie",
    "skeleton",
)
PASSIVE_ACTIONS = {"sleep", "noop"}
VERIFIER_TOOL_NAME = "pivot_verifier_report"
VERIFIER_SYSTEM_DIRECTIVE = (
    "You are a Craftax pivot verifier.\n"
    f"Use the provided `{VERIFIER_TOOL_NAME}` tool exactly once for the final answer.\n"
    "Do not answer in plain text.\n"
    "Do not output JSON in assistant content.\n"
    "Keep private reasoning concise and reserve tokens for the final tool call.\n"
    "After reasoning, your final assistant action must be a tool call."
)
VERIFIER_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": VERIFIER_TOOL_NAME,
        "description": "Return a structured pivot-verifier report with principles, critique, and calibrated rewards.",
        "parameters": {
            "type": "object",
            "properties": {
                "principles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "critique": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 6,
                },
                "scores": {
                    "type": "object",
                    "properties": {
                        "process_reward": {"type": "number"},
                        "progress_reward": {"type": "number"},
                        "total_reward": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "process_reward",
                        "progress_reward",
                        "total_reward",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
                "verdict": {"type": "string"},
            },
            "required": ["principles", "critique", "scores", "verdict"],
            "additionalProperties": False,
        },
    },
}

_PIVOTRL_CORE: Any | None = None


def pivotrl_core() -> Any:
    global _PIVOTRL_CORE
    if _PIVOTRL_CORE is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        _PIVOTRL_CORE = importlib.import_module("submissions.synth.pivotrl_core")
    return _PIVOTRL_CORE


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def default_output_dir(*, track_id: str, method_name: str) -> Path:
    stamp = Path(now_utc_iso().replace(":", "").replace("+00:00", "Z").replace("-", "")).name
    return PROJECT_ROOT / "records" / track_id / f"{stamp}_{method_name}"


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
            return "\n".join(parts).strip()
        return str(content or "").strip()
    return flatten_messages([item for item in messages if isinstance(item, dict)])


def parse_inventory_from_text(text: str) -> dict[str, int]:
    normalized = str(text or "").lower()
    inventory: dict[str, int] = {}
    for key in RESOURCE_KEYS:
        match = re.search(rf"\b{re.escape(key)}\s*=\s*(-?\d+)\b", normalized)
        if match:
            inventory[key] = max(0, int(match.group(1)))
    return inventory


def parse_achievements_from_text(text: str) -> list[str]:
    match = re.search(r"achievements:\s*(.+)", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw or raw.lower() == "none":
        return []
    return [item.strip().lower().replace(" ", "_") for item in raw.split(",") if item.strip()]


def visible_objects_from_text(text: str) -> list[str]:
    normalized = str(text or "").lower()
    return [name for name in VISIBLE_OBJECT_KEYS if name in normalized]


def normalize_actions(actions: Any, assistant_text: str) -> list[str]:
    if isinstance(actions, list):
        normalized = [str(item).strip().lower() for item in actions if str(item).strip()]
        if normalized:
            return normalized
    return pivotrl_core().extract_actions_from_text(assistant_text)


def _normalize_state_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _state_target_keywords(
    *,
    state_text: str,
    next_state_text: str,
    inventory_delta: dict[str, int],
    new_achievements: list[str],
    actions: list[str],
) -> list[str]:
    keywords = set(visible_objects_from_text(state_text))
    keywords.update(key for key, delta in inventory_delta.items() if delta > 0)
    keywords.update(item.replace("_", " ") for item in new_achievements)
    keywords.update(action.replace("_", " ") for action in actions)
    if "do" in actions:
        keywords.add("do")
    if next_state_text and next_state_text != state_text:
        keywords.add("next state")
    return sorted(keyword for keyword in keywords if keyword)


def score_process_reward(
    *,
    thinking_text: str,
    actions: list[str],
    state_text: str,
    next_state_text: str,
    inventory_delta: dict[str, int],
    new_achievements: list[str],
    invalid_parse: bool,
) -> tuple[float, list[str]]:
    normalized = str(thinking_text or "").strip().lower()
    evidence: list[str] = []
    score = 0.05
    if normalized:
        score += 0.20
        evidence.append("reasoning is present")
    else:
        evidence.append("reasoning is missing")
    if any(token in normalized for token in ("because", "need", "should", "so that", "first")):
        score += 0.10
        evidence.append("reasoning includes an explicit plan or justification")
    keywords = _state_target_keywords(
        state_text=state_text,
        next_state_text=next_state_text,
        inventory_delta=inventory_delta,
        new_achievements=new_achievements,
        actions=actions,
    )
    grounded = [keyword for keyword in keywords if keyword in normalized]
    if grounded:
        score += min(0.30, 0.10 * len(grounded))
        evidence.append(f"reasoning is grounded in pivot evidence: {', '.join(grounded[:4])}")
    if any(action.replace("_", " ") in normalized or action in normalized for action in actions):
        score += 0.15
        evidence.append("reasoning references the proposed action")
    if len(normalized.split()) >= 12:
        score += 0.05
        evidence.append("reasoning is detailed enough to critique")
    if invalid_parse:
        score -= 0.20
        evidence.append("the original turn had an invalid parse")
    if normalized and len(normalized.split()) <= 2:
        score -= 0.05
        evidence.append("reasoning is too short to be very informative")
    return clamp(score, 0.0, 1.0), evidence


def score_progress_reward(
    *,
    actions: list[str],
    state_text: str,
    next_state_text: str,
    inventory_delta: dict[str, int],
    new_achievements: list[str],
    decision_reward: float,
    current_return_to_go: float,
    next_return_to_go: float,
    invalid_parse: bool,
) -> tuple[float, list[str]]:
    evidence: list[str] = []
    score = 0.05
    if new_achievements:
        score += min(0.45, 0.25 + 0.15 * len(new_achievements))
        evidence.append(f"next state unlocks achievements: {', '.join(new_achievements[:4])}")
    positive_delta = {key: delta for key, delta in inventory_delta.items() if delta > 0}
    if positive_delta:
        score += min(0.25, 0.08 * sum(positive_delta.values()))
        summary = ", ".join(f"{key} +{delta}" for key, delta in sorted(positive_delta.items())[:4])
        evidence.append(f"next state improves resources: {summary}")
    if next_return_to_go > current_return_to_go:
        score += 0.15
        evidence.append("return-to-go increases after the pivot")
    if decision_reward > 0.0:
        score += 0.10
        evidence.append("environment decision reward is positive")
    visible_objects = visible_objects_from_text(state_text)
    if "do" in actions and visible_objects:
        score += 0.05
        evidence.append(f"the action interacts with visible affordances: {', '.join(visible_objects[:3])}")
    if _normalize_state_signature(state_text) == _normalize_state_signature(next_state_text):
        score -= 0.10
        evidence.append("pre and post state summaries are nearly identical")
    if actions and all(action in PASSIVE_ACTIONS for action in actions):
        score -= 0.15
        evidence.append("the pivot action is passive")
    if invalid_parse:
        score -= 0.20
        evidence.append("the original turn had an invalid parse")
    if not evidence:
        evidence.append("the pivot shows limited measurable local progress")
    return clamp(score, 0.0, 1.0), evidence


def build_principles() -> list[str]:
    return [
        "Prefer reasoning that is grounded in the observed state, inventory, and nearby affordances.",
        "Prefer actions that unlock achievements, increase prerequisite resources, or advance tool progression.",
        "Penalize passive, invalid, or strategically disconnected pivots even when they look superficially plausible.",
    ]


def rubric_reward_to_progress_score(rubric_reward: float) -> float:
    if rubric_reward >= 0.999:
        return 1.0
    if rubric_reward >= 0.49:
        return 0.65
    if rubric_reward <= -0.999:
        return 0.0
    if rubric_reward <= -0.49:
        return 0.1
    return 0.35


def infer_rubric_target(
    *,
    state_text: str,
    next_state_text: str,
    inventory_delta: dict[str, int],
    new_achievements: list[str],
    actions: list[str],
) -> str | None:
    for achievement in new_achievements:
        if achievement in {"collect_wood", "collect_stone", "place_table", "make_wood_pickaxe", "make_stone_pickaxe"}:
            return achievement
    if inventory_delta.get("wood", 0) > 0 or ("do" in actions and "tree" in state_text.lower()):
        return "collect_wood"
    if inventory_delta.get("stone", 0) > 0 or ("do" in actions and "stone" in state_text.lower()):
        return "collect_stone"
    if "place_table" in actions or "table" in next_state_text.lower():
        return "place_table"
    if "make_wood_pickaxe" in actions or inventory_delta.get("wood_pickaxe", 0) > 0:
        return "make_wood_pickaxe"
    if "make_stone_pickaxe" in actions or inventory_delta.get("stone_pickaxe", 0) > 0:
        return "make_stone_pickaxe"
    return None


def build_rubric_payload(
    *,
    target_achievement: str | None,
    state_text: str,
    inventory: dict[str, int],
) -> dict[str, Any] | None:
    if not target_achievement:
        return None
    try:
        return pivotrl_core().build_rubric(
            target_achievement=target_achievement,
            state_text=state_text,
            inventory=inventory,
        )
    except Exception:
        return None


def render_rubric_text(rubric: dict[str, Any] | None) -> str:
    if not isinstance(rubric, dict):
        return ""
    return (
        f"Target achievement: {rubric.get('target_achievement') or 'unknown'}\n"
        f"Strong accept actions: {', '.join(str(item) for item in rubric.get('accept_actions', [])) or 'none'}\n"
        f"Weak accept actions: {', '.join(str(item) for item in rubric.get('weak_accept_actions', [])) or 'none'}\n"
        f"Reject actions: {', '.join(str(item) for item in rubric.get('reject_actions', [])) or 'none'}\n"
        f"Forbidden actions: {', '.join(str(item) for item in rubric.get('forbidden_actions', [])) or 'none'}\n"
        f"Inventory requirements: {json.dumps(rubric.get('inventory_requirements', {}), sort_keys=True)}\n"
        f"Position requirements: {', '.join(str(item) for item in rubric.get('position_or_object_requirements', [])) or 'none'}\n"
        f"Notes: {str(rubric.get('notes') or '').strip() or 'none'}"
    )


def build_spct_target(
    *,
    process_reward: float,
    progress_reward: float,
    confidence: float,
    critique_points: list[str],
) -> dict[str, Any]:
    total_reward = clamp(0.4 * process_reward + 0.6 * progress_reward, 0.0, 1.0)
    verdict = "strong" if total_reward >= 0.75 else "good" if total_reward >= 0.55 else "mixed" if total_reward >= 0.35 else "weak"
    return {
        "principles": build_principles(),
        "critique": critique_points[:4],
        "scores": {
            "process_reward": round(process_reward, 4),
            "progress_reward": round(progress_reward, 4),
            "total_reward": round(total_reward, 4),
            "confidence": round(confidence, 4),
        },
        "verdict": verdict,
    }


def build_verifier_prompt_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    state_text = str(example["state_text"])
    next_state_text = str(example["next_state_text"])
    thinking_text = str(example["thinking_text"])
    actions = example["actions"]
    rubric_text = str(example.get("rubric_text") or "").strip()
    user_prompt = (
        "Evaluate this Craftax pivot.\n\n"
        "Think briefly about reasoning quality and local progress, then call the verifier tool exactly once.\n"
        "Do not answer with plain JSON or prose outside the tool call.\n\n"
        f"s_t:\n{state_text}\n\n"
        f"thinking_t:\n{thinking_text or '[empty reasoning]'}\n\n"
        f"a_t:\n{json.dumps(actions)}\n\n"
        f"s_t_plus_1:\n{next_state_text}\n"
    )
    if rubric_text:
        user_prompt += f"\nProgress rubric:\n{rubric_text}\n"
    return [
        {
            "role": "system",
            "content": VERIFIER_SYSTEM_DIRECTIVE
            + "\nGenerate a principled critique and calibrated reward scores for a single pivot.\n"
            + "Reward grounded reasoning and real local progress.",
        },
        {"role": "user", "content": user_prompt},
    ]


def build_verifier_example(
    *,
    rollout: dict[str, Any],
    current_turn: dict[str, Any],
    next_turn: dict[str, Any],
    trace_id: str,
    rollout_id: str,
    seed: int,
    turn_index: int,
) -> dict[str, Any] | None:
    prompt_messages = current_turn.get("prompt_messages")
    next_prompt_messages = next_turn.get("prompt_messages")
    if not isinstance(prompt_messages, list) or not isinstance(next_prompt_messages, list):
        return None
    safe_prompt_messages = [item for item in prompt_messages if isinstance(item, dict)]
    safe_next_prompt_messages = [item for item in next_prompt_messages if isinstance(item, dict)]
    state_text = latest_user_text(safe_prompt_messages)
    next_state_text = latest_user_text(safe_next_prompt_messages)
    assistant_text = str(current_turn.get("assistant_text") or "").strip()
    thinking_text = str(current_turn.get("reasoning_text") or assistant_text).strip()
    actions = normalize_actions(current_turn.get("actions"), assistant_text)
    if not state_text or not next_state_text or not actions:
        return None
    current_inventory = parse_inventory_from_text(state_text)
    next_inventory = parse_inventory_from_text(next_state_text)
    inventory_delta = {
        key: int(next_inventory.get(key, 0) - current_inventory.get(key, 0))
        for key in sorted(set(current_inventory) | set(next_inventory))
        if int(next_inventory.get(key, 0) - current_inventory.get(key, 0)) != 0
    }
    current_achievements = parse_achievements_from_text(state_text)
    next_achievements = parse_achievements_from_text(next_state_text)
    new_achievements = sorted(set(next_achievements) - set(current_achievements))
    current_return_to_go = float(current_turn.get("return_to_go") or 0.0)
    next_return_to_go = float(next_turn.get("return_to_go") or current_return_to_go)
    decision_reward = float(current_turn.get("decision_reward") or 0.0)
    invalid_parse = bool(current_turn.get("invalid_parse"))
    process_reward, process_evidence = score_process_reward(
        thinking_text=thinking_text,
        actions=actions,
        state_text=state_text,
        next_state_text=next_state_text,
        inventory_delta=inventory_delta,
        new_achievements=new_achievements,
        invalid_parse=invalid_parse,
    )
    progress_reward, progress_evidence = score_progress_reward(
        actions=actions,
        state_text=state_text,
        next_state_text=next_state_text,
        inventory_delta=inventory_delta,
        new_achievements=new_achievements,
        decision_reward=decision_reward,
        current_return_to_go=current_return_to_go,
        next_return_to_go=next_return_to_go,
        invalid_parse=invalid_parse,
    )
    evidence_count = 0
    if new_achievements:
        evidence_count += 1
    if inventory_delta:
        evidence_count += 1
    if thinking_text:
        evidence_count += 1
    if decision_reward > 0.0 or next_return_to_go > current_return_to_go:
        evidence_count += 1
    confidence = clamp(0.35 + 0.15 * evidence_count, 0.35, 0.95)
    rubric_target = infer_rubric_target(
        state_text=state_text,
        next_state_text=next_state_text,
        inventory_delta=inventory_delta,
        new_achievements=new_achievements,
        actions=actions,
    )
    rubric = build_rubric_payload(
        target_achievement=rubric_target,
        state_text=state_text,
        inventory=current_inventory,
    )
    rubric_text = render_rubric_text(rubric)
    spct_target = build_spct_target(
        process_reward=process_reward,
        progress_reward=progress_reward,
        confidence=confidence,
        critique_points=[*process_evidence, *progress_evidence],
    )
    total_reward = float(spct_target["scores"]["total_reward"])
    training_weight = round(0.5 + total_reward + 0.25 * confidence, 4)
    example = {
        "example_id": f"{trace_id or rollout_id}_turn_{turn_index:04d}",
        "trace_id": trace_id,
        "rollout_id": rollout_id,
        "seed": seed,
        "turn_index": turn_index,
        "state_text": state_text,
        "next_state_text": next_state_text,
        "thinking_text": thinking_text,
        "actions": actions,
        "prompt_messages": build_verifier_prompt_messages(
            {
                "state_text": state_text,
                "next_state_text": next_state_text,
                "thinking_text": thinking_text,
                "actions": actions,
                "rubric_text": rubric_text,
            }
        ),
        "response": json.dumps(spct_target, indent=2, sort_keys=True),
        "tools": [VERIFIER_REPORT_TOOL],
        "messages": [
            *build_verifier_prompt_messages(
                {
                    "state_text": state_text,
                    "next_state_text": next_state_text,
                    "thinking_text": thinking_text,
                    "actions": actions,
                    "rubric_text": rubric_text,
                }
            ),
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "\n".join([*process_evidence, *progress_evidence][:4]),
                "tool_calls": [
                    {
                        "id": f"call_{trace_id or rollout_id}_{turn_index}",
                        "type": "function",
                        "function": {
                            "name": VERIFIER_TOOL_NAME,
                            "arguments": spct_target,
                        },
                    }
                ],
            },
        ],
        "spct_target": spct_target,
        "labels": {
            "process_reward": process_reward,
            "progress_reward": progress_reward,
            "total_reward": total_reward,
            "confidence": confidence,
        },
        "metadata": {
            "state_inventory": current_inventory,
            "next_state_inventory": next_inventory,
            "inventory_delta": inventory_delta,
            "state_achievements": current_achievements,
            "next_state_achievements": next_achievements,
            "new_achievements": new_achievements,
            "decision_reward": decision_reward,
            "current_return_to_go": current_return_to_go,
            "next_return_to_go": next_return_to_go,
            "invalid_parse": invalid_parse,
            "outcome_reward": float(rollout_outcome_reward(rollout)),
            "rubric_target_achievement": rubric_target,
            "rubric": rubric,
        },
        "weight": training_weight,
    }
    return example


def load_golden_eval_examples(*, config: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation_cfg = config.get("evaluation", {}) if isinstance(config.get("evaluation"), dict) else {}
    golden_eval_path = str(evaluation_cfg.get("golden_eval_path") or "").strip()
    if not golden_eval_path:
        return []
    resolved = resolve_path(golden_eval_path, base_dir=PROJECT_ROOT)
    rows = read_jsonl(resolved)
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        state_text = str(row.get("state_text") or "").strip()
        next_state_text = str(row.get("next_state_text") or "").strip()
        thinking_text = str(row.get("thinking_text") or "").strip()
        actions = normalize_actions(row.get("actions"), str(row.get("assistant_text") or ""))
        if not state_text or not next_state_text or not actions:
            continue
        inventory = parse_inventory_from_text(state_text)
        target_achievement = str(row.get("target_achievement") or "").strip() or None
        rubric = row.get("rubric") if isinstance(row.get("rubric"), dict) else build_rubric_payload(
            target_achievement=target_achievement,
            state_text=state_text,
            inventory=inventory,
        )
        rubric_reward = row.get("rubric_reward")
        if rubric_reward is None and rubric:
            try:
                rubric_reward, _ = pivotrl_core().score_actions_against_rubric(
                    actions=actions,
                    rubric=rubric,
                    state_text=state_text,
                    inventory=inventory,
                )
            except Exception:
                rubric_reward = None
        progress_reward = row.get("gold_progress_reward")
        if progress_reward is None and rubric_reward is not None:
            progress_reward = rubric_reward_to_progress_score(float(rubric_reward))
        process_reward = float(row.get("gold_process_reward", 0.5) or 0.5)
        confidence = float(row.get("gold_confidence", 0.9) or 0.9)
        if progress_reward is None:
            continue
        spct_target = build_spct_target(
            process_reward=process_reward,
            progress_reward=float(progress_reward),
            confidence=confidence,
            critique_points=[str(item) for item in row.get("gold_critique", []) if str(item).strip()],
        )
        rubric_text = render_rubric_text(rubric if isinstance(rubric, dict) else None)
        examples.append(
            {
                "example_id": str(row.get("example_id") or f"golden_eval_{index:04d}"),
                "state_text": state_text,
                "next_state_text": next_state_text,
                "thinking_text": thinking_text,
                "actions": actions,
                "prompt_messages": build_verifier_prompt_messages(
                    {
                        "state_text": state_text,
                        "next_state_text": next_state_text,
                        "thinking_text": thinking_text,
                        "actions": actions,
                        "rubric_text": rubric_text,
                    }
                ),
                "labels": {
                    "process_reward": float(spct_target["scores"]["process_reward"]),
                    "progress_reward": float(spct_target["scores"]["progress_reward"]),
                    "total_reward": float(spct_target["scores"]["total_reward"]),
                    "confidence": float(spct_target["scores"]["confidence"]),
                },
                "metadata": {
                    "rubric_target_achievement": target_achievement,
                    "rubric": rubric,
                    "rubric_reward": rubric_reward,
                    "gold_source": str(row.get("source") or "golden_eval"),
                },
            }
        )
    return examples


def build_verifier_examples_from_rollouts(
    *,
    rollouts: list[dict[str, Any]],
    lookahead: int,
    max_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    skipped_missing_transition = 0
    for rollout in rollouts:
        if not isinstance(rollout, dict) or rollout.get("error") or not is_rollout_payload(rollout):
            continue
        turns = rollout_turns(rollout)
        if len(turns) <= lookahead:
            continue
        trace_id = str(rollout.get("trace_correlation_id") or rollout.get("trial_id") or "")
        rollout_id = str(rollout.get("rollout_id") or trace_id)
        seed = int(rollout.get("_request_seed") or 0)
        for turn_index in range(0, len(turns) - lookahead):
            current_turn = turns[turn_index]
            next_turn = turns[turn_index + lookahead]
            example = build_verifier_example(
                rollout=rollout,
                current_turn=current_turn,
                next_turn=next_turn,
                trace_id=trace_id,
                rollout_id=rollout_id,
                seed=seed,
                turn_index=int(current_turn.get("turn_index") or turn_index),
            )
            if example is None:
                skipped_missing_transition += 1
                continue
            examples.append(example)
            if max_examples > 0 and len(examples) >= max_examples:
                break
        if max_examples > 0 and len(examples) >= max_examples:
            break
    rewards = [float(item["labels"]["total_reward"]) for item in examples]
    summary = {
        "example_count": len(examples),
        "skipped_missing_transition": skipped_missing_transition,
        "mean_total_reward": mean(rewards) if rewards else 0.0,
        "max_total_reward": max(rewards) if rewards else 0.0,
        "min_total_reward": min(rewards) if rewards else 0.0,
    }
    return examples, summary


def split_examples(
    examples: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) <= 1 or holdout_fraction <= 0.0:
        return list(examples), []
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    holdout_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * holdout_fraction))))
    holdout_rows = shuffled[:holdout_count]
    train_rows = shuffled[holdout_count:]
    return train_rows, holdout_rows


def _bootstrap_args_from_config(config: dict[str, Any]) -> SimpleNamespace:
    bootstrap_cfg = config.get("bootstrap", {}) if isinstance(config.get("bootstrap"), dict) else {}
    return SimpleNamespace(
        container_url=str(bootstrap_cfg.get("container_url") or "").strip(),
        container_worker_token=str(bootstrap_cfg.get("container_worker_token") or "").strip(),
        teacher_inference_url=str(bootstrap_cfg.get("teacher_inference_url") or "").strip(),
        teacher_api_key=str(bootstrap_cfg.get("teacher_api_key") or "").strip(),
        teacher_model=str(bootstrap_cfg.get("teacher_model") or DEFAULT_TEACHER_MODEL),
        bootstrap_seed_start=int(bootstrap_cfg.get("seed_start", 0)),
        bootstrap_seed_count=int(bootstrap_cfg.get("seed_count", 16)),
        bootstrap_max_steps=int(bootstrap_cfg.get("max_steps", 32)),
        bootstrap_temperature=float(bootstrap_cfg.get("temperature", 0.2)),
        bootstrap_max_new_tokens=int(bootstrap_cfg.get("max_new_tokens", 2048)),
        bootstrap_thinking_budget_tokens=int(bootstrap_cfg.get("thinking_budget_tokens", 1500)),
        bootstrap_rollout_concurrency=int(bootstrap_cfg.get("rollout_concurrency", 4)),
        bootstrap_rollout_semaphore_limit=int(bootstrap_cfg.get("rollout_semaphore_limit", 4)),
        bootstrap_target_action_batch_size=int(bootstrap_cfg.get("target_action_batch_size", 4)),
        bootstrap_min_action_batch_size=int(bootstrap_cfg.get("min_action_batch_size", 3)),
        enable_thinking=bool(bootstrap_cfg.get("enable_thinking", True)),
        request_timeout_seconds=float(bootstrap_cfg.get("request_timeout_seconds", 300.0)),
    )


def collect_or_load_rollouts(*, config: dict[str, Any], output_root: Path) -> tuple[Path, dict[str, Any]]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    configured_path = str(data_cfg.get("bootstrap_rollouts_path") or "").strip()
    artifacts_dir = ensure_dir(output_root / "artifacts")
    if configured_path:
        resolved = resolve_path(configured_path, base_dir=PROJECT_ROOT)
        if not resolved.exists():
            raise FileNotFoundError(f"bootstrap rollouts not found: {resolved}")
        rows = read_jsonl(resolved)
        successful = [row for row in rows if isinstance(row, dict) and is_rollout_payload(row)]
        summary = {
            "bootstrap_rollouts_path": str(resolved),
            "requested_rollouts": len(rows),
            "successful_rollouts": len(successful),
            "skipped_live_bootstrap": True,
        }
        write_json(artifacts_dir / "bootstrap_summary.json", summary)
        if not successful:
            raise RuntimeError(f"bootstrap rollout input contains no successful rollout payloads: {resolved}")
        return resolved, summary

    args = _bootstrap_args_from_config(config)
    if not str(args.teacher_inference_url).strip():
        raise RuntimeError("teacher_inference_url is required when bootstrap_rollouts_path is not provided")
    runtime_log = artifacts_dir / "bootstrap_craftax_runtime.log"
    if str(args.container_url).strip():
        rollouts_path, summary = pivotrl_core().run_bootstrap(args, output_root=output_root)
        return rollouts_path, summary
    with pivotrl_core().local_craftax_runtime(log_path=runtime_log) as container_url:
        args.container_url = container_url
        rollouts_path, summary = pivotrl_core().run_bootstrap(args, output_root=output_root)
        return rollouts_path, summary


def build_dataset_artifacts(*, config: dict[str, Any], output_root: Path, rollouts_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset"), dict) else {}
    rollouts = read_jsonl(rollouts_path)
    examples, dataset_summary = build_verifier_examples_from_rollouts(
        rollouts=rollouts,
        lookahead=max(1, int(dataset_cfg.get("lookahead", 1))),
        max_examples=max(0, int(dataset_cfg.get("max_examples", 256))),
    )
    if not examples:
        raise RuntimeError(f"no verifier examples were built from {rollouts_path}")
    train_rows, holdout_rows = split_examples(
        examples,
        holdout_fraction=float(dataset_cfg.get("holdout_fraction", 0.15)),
        seed=int(dataset_cfg.get("split_seed", 13)),
    )
    artifacts_dir = ensure_dir(output_root / "artifacts")
    write_jsonl(artifacts_dir / "pivot_verifier_dataset.jsonl", examples)
    write_jsonl(artifacts_dir / "pivot_verifier_train.jsonl", train_rows)
    write_jsonl(artifacts_dir / "pivot_verifier_holdout.jsonl", holdout_rows)
    summary = {
        **dataset_summary,
        "bootstrap_rollouts_path": str(rollouts_path),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
    }
    write_json(artifacts_dir / "pivot_verifier_dataset_summary.json", summary)
    return train_rows, holdout_rows, summary


def train_verifier_model(*, config: dict[str, Any], output_root: Path, train_rows: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    training_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    verifier_cfg = config.get("verifier", {}) if isinstance(config.get("verifier"), dict) else {}
    artifacts_dir = ensure_dir(output_root / "artifacts")
    adapter_dir = artifacts_dir / "pivot_verifier_adapter"
    examples = [
        {
            "prompt_messages": row["prompt_messages"],
            "messages": row["messages"],
            "tools": row["tools"],
            "response": row["response"],
            "weight": float(row.get("weight", 1.0)),
        }
        for row in train_rows
    ]
    use_native_tool_calling = bool(training_cfg.get("use_native_tool_calling", True))
    if use_native_tool_calling:
        result = train_sft_with_trl(
            base_model=str(verifier_cfg.get("base_model") or DEFAULT_VERIFIER_MODEL),
            examples=examples,
            output_dir=adapter_dir,
            learning_rate=float(training_cfg.get("learning_rate", 1e-5)),
            epochs=max(1, int(training_cfg.get("epochs", 1))),
            max_length=int(verifier_cfg.get("max_length", 4096)),
            max_steps=int(training_cfg.get("max_steps", 32)),
            lora_rank=int(training_cfg.get("lora_rank", 16)),
        )
    else:
        result = train_weighted_lora(
            base_model=str(verifier_cfg.get("base_model") or DEFAULT_VERIFIER_MODEL),
            examples=examples,
            output_dir=adapter_dir,
            learning_rate=float(training_cfg.get("learning_rate", 1e-5)),
            epochs=max(1, int(training_cfg.get("epochs", 1))),
            max_length=int(verifier_cfg.get("max_length", 4096)),
            max_steps=int(training_cfg.get("max_steps", 32)),
            lora_rank=int(training_cfg.get("lora_rank", 16)),
        )
    summary = {
        "adapter_dir": str(adapter_dir),
        "base_model": str(verifier_cfg.get("base_model") or DEFAULT_VERIFIER_MODEL),
        "examples_seen": int(result.examples_seen),
        "optimizer_steps": int(result.optimizer_steps),
        "mean_loss": float(result.mean_loss),
    }
    write_json(artifacts_dir / "training_summary.json", summary)
    return adapter_dir, summary


def _render_prompt_text(tokenizer: Any, prompt_messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> str:
    safe_messages = [{"role": str(item["role"]), "content": str(item["content"])} for item in prompt_messages]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            safe_messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str):
            return rendered
    rendered_lines = [f"<|{item['role']}|>\n{item['content']}" for item in safe_messages]
    rendered_lines.append("<|assistant|>\n")
    return "\n".join(rendered_lines)


def _extract_verifier_tool_arguments_from_text(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, flags=re.DOTALL):
        try:
            payload = json.loads(block.strip())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or "").strip()
        if name != VERIFIER_TOOL_NAME:
            continue
        arguments = payload.get("arguments")
        if isinstance(arguments, dict):
            return arguments
    parsed = _extract_first_json_object(raw)
    if isinstance(parsed, dict):
        if str(parsed.get("name") or "").strip() == VERIFIER_TOOL_NAME and isinstance(parsed.get("arguments"), dict):
            return parsed["arguments"]
        scores = parsed.get("scores")
        if isinstance(scores, dict) and parsed.get("verdict") is not None:
            return parsed
    return None


def _normalize_verifier_request_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized_messages = [
        {
            "role": str(item.get("role") or "user"),
            "content": str(item.get("content") or ""),
        }
        for item in messages
        if isinstance(item, dict)
    ]
    if normalized_messages and normalized_messages[0]["role"] == "system":
        normalized_messages[0]["content"] = VERIFIER_SYSTEM_DIRECTIVE
    else:
        normalized_messages.insert(0, {"role": "system", "content": VERIFIER_SYSTEM_DIRECTIVE})
    return normalized_messages


def _extract_verifier_prediction(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, str, str, str | None]:
    tool_calls = extract_openai_tool_calls(payload, tool_name=VERIFIER_TOOL_NAME)
    parsed_payload = None
    parse_source = None
    for tool_call in tool_calls:
        arguments = tool_call.get("arguments")
        if isinstance(arguments, dict):
            parsed_payload = arguments
            parse_source = "tool_call"
            break
    message = ((payload.get("choices") or [{}])[0] or {}).get("message", {})
    response_text = str(message.get("content") or "")
    reasoning_text = str(message.get("reasoning") or message.get("reasoning_content") or "")
    if parsed_payload is None:
        parsed_payload = _extract_verifier_tool_arguments_from_text(response_text)
        if parsed_payload is None and reasoning_text:
            parsed_payload = _extract_verifier_tool_arguments_from_text(reasoning_text)
        if parsed_payload is not None:
            parse_source = "content_fallback"
    finish_reason = str(((payload.get("choices") or [{}])[0] or {}).get("finish_reason") or "").strip() or None
    return parsed_payload, parse_source, response_text, reasoning_text, finish_reason


def request_verifier_completion(
    *,
    server_base_url: str,
    prompt_messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    thinking_budget_tokens: int,
    guided_decoding_backend: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, str, str]:
    normalized_messages = _normalize_verifier_request_messages(prompt_messages)
    attempts = [
        {
            "enable_thinking": enable_thinking,
            "thinking_budget_tokens": thinking_budget_tokens,
            "max_tokens": max_tokens,
        }
    ]
    if enable_thinking:
        attempts.append(
            {
                "enable_thinking": False,
                "thinking_budget_tokens": 0,
                "max_tokens": max(max_tokens, 384),
            }
        )
    last_payload: dict[str, Any] | None = None
    last_response_text = ""
    last_reasoning_text = ""
    for attempt in attempts:
        extra_body = build_thinking_budget_request_overrides(
            enable_thinking=bool(attempt["enable_thinking"]),
            thinking_budget=int(attempt["thinking_budget_tokens"]),
        )
        extra_body["guided_decoding_backend"] = guided_decoding_backend
        payload = create_chat_completion(
            model="pivot-verifier-lora",
            messages=normalized_messages,
            max_tokens=int(attempt["max_tokens"]),
            temperature=temperature,
            base_url=server_base_url,
            api_key="",
            timeout_seconds=300.0,
            tools=[VERIFIER_REPORT_TOOL],
            tool_choice="auto",
            extra_body=extra_body,
        )
        last_payload = payload
        parsed_payload, parse_source, response_text, reasoning_text, finish_reason = _extract_verifier_prediction(payload)
        last_response_text = response_text
        last_reasoning_text = reasoning_text
        if parsed_payload is not None:
            return payload, parsed_payload, parse_source, response_text, reasoning_text
        if finish_reason != "length":
            return payload, None, None, response_text, reasoning_text
    assert last_payload is not None
    return last_payload, None, None, last_response_text, last_reasoning_text


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def evaluate_verifier_model(
    *,
    config: dict[str, Any],
    output_root: Path,
    adapter_dir: Path,
    holdout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluation_cfg = config.get("evaluation", {}) if isinstance(config.get("evaluation"), dict) else {}
    verifier_cfg = config.get("verifier", {}) if isinstance(config.get("verifier"), dict) else {}
    golden_eval_rows = load_golden_eval_examples(config=config)
    eval_limit = max(0, int(evaluation_cfg.get("max_examples", 24)))
    source_rows = golden_eval_rows if golden_eval_rows else holdout_rows
    evaluation_rows = list(source_rows[:eval_limit] if eval_limit > 0 else source_rows)
    if not evaluation_rows:
        summary = {"evaluated_examples": 0, "skipped": True, "reason": "no holdout rows"}
        write_json(output_root / "artifacts" / "eval_summary.json", summary)
        return summary

    base_model = str(verifier_cfg.get("base_model") or DEFAULT_VERIFIER_MODEL)
    parsed_rows = 0
    tool_call_parsed_rows = 0
    content_fallback_parsed_rows = 0
    mae_total_values: list[float] = []
    mae_progress_values: list[float] = []
    pairwise_total: list[tuple[float, float]] = []
    predictions: list[dict[str, Any]] = []
    generation_max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 384))
    thinking_budget_tokens = int(evaluation_cfg.get("thinking_budget_tokens", 512))
    enable_thinking = bool(evaluation_cfg.get("enable_thinking", True))
    vllm_config = LocalVLLMEvalConfig(
        model=base_model,
        served_model_name=base_model,
        lora_name="pivot-verifier-lora",
        lora_path=str(adapter_dir),
        max_lora_rank=max(1, int(evaluation_cfg.get("max_lora_rank", 16))),
        max_model_len=max(int(verifier_cfg.get("max_length", 4096)), generation_max_new_tokens + thinking_budget_tokens + 256),
        max_new_tokens=generation_max_new_tokens,
        enable_thinking=enable_thinking,
        enforce_eager=bool(evaluation_cfg.get("enforce_eager", True)),
        port=int(evaluation_cfg.get("vllm_port", 8013)),
        gpu_memory_utilization=float(evaluation_cfg.get("gpu_memory_utilization", 0.90)),
        max_num_seqs=int(evaluation_cfg.get("max_num_seqs", 8)),
        max_num_batched_tokens=int(evaluation_cfg.get("max_num_batched_tokens", 4096)),
    )
    server_log_path = output_root / "artifacts" / "verifier_eval_vllm.log"
    release_cuda_memory()
    with local_vllm_server(config=vllm_config, log_path=server_log_path) as server:
        for row in evaluation_rows:
            payload, parsed_payload, parse_source, response_text, reasoning_text = request_verifier_completion(
                server_base_url=str(server["base_url"]),
                prompt_messages=row["prompt_messages"],
                max_tokens=generation_max_new_tokens,
                temperature=0.0,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                guided_decoding_backend=vllm_config.guided_decoding_backend,
            )
            total_pred = None
            progress_pred = None
            if isinstance(parsed_payload, dict):
                scores = parsed_payload.get("scores")
                if isinstance(scores, dict):
                    try:
                        total_pred = float(scores.get("total_reward"))
                    except (TypeError, ValueError):
                        total_pred = None
                    try:
                        progress_pred = float(scores.get("progress_reward"))
                    except (TypeError, ValueError):
                        progress_pred = None
            target_total = float(row["labels"]["total_reward"])
            target_progress = float(row["labels"]["progress_reward"])
            if total_pred is not None:
                parsed_rows += 1
                if parse_source == "tool_call":
                    tool_call_parsed_rows += 1
                elif parse_source == "content_fallback":
                    content_fallback_parsed_rows += 1
                mae_total_values.append(abs(total_pred - target_total))
                pairwise_total.append((total_pred, target_total))
            if progress_pred is not None:
                mae_progress_values.append(abs(progress_pred - target_progress))
            predictions.append(
                {
                    "example_id": row["example_id"],
                    "response_text": response_text,
                    "reasoning_text": reasoning_text,
                    "raw_payload": payload,
                    "parsed_payload": parsed_payload,
                    "parse_source": parse_source if parsed_payload is not None else None,
                    "target_total_reward": target_total,
                    "predicted_total_reward": total_pred,
                    "target_progress_reward": target_progress,
                    "predicted_progress_reward": progress_pred,
                }
            )
    write_jsonl(output_root / "artifacts" / "eval_predictions.jsonl", predictions)
    pairwise_correct = 0
    pairwise_count = 0
    for left_index in range(len(pairwise_total)):
        for right_index in range(left_index + 1, len(pairwise_total)):
            left_pred, left_target = pairwise_total[left_index]
            right_pred, right_target = pairwise_total[right_index]
            if math.isclose(left_target, right_target, abs_tol=1e-6):
                continue
            pairwise_count += 1
            if (left_pred - right_pred) * (left_target - right_target) > 0:
                pairwise_correct += 1
    summary = {
        "evaluated_examples": len(evaluation_rows),
        "parsed_examples": parsed_rows,
        "tool_call_parsed_examples": tool_call_parsed_rows,
        "content_fallback_parsed_examples": content_fallback_parsed_rows,
        "json_parse_rate": (float(parsed_rows) / float(len(evaluation_rows))) if evaluation_rows else 0.0,
        "mae_total_reward": mean(mae_total_values) if mae_total_values else None,
        "mae_progress_reward": mean(mae_progress_values) if mae_progress_values else None,
        "pairwise_accuracy": (float(pairwise_correct) / float(pairwise_count)) if pairwise_count else None,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "thinking_budget_tokens": thinking_budget_tokens,
        "native_tool_calling": True,
        "eval_source": "golden_eval" if golden_eval_rows else "holdout",
        "skipped": False,
    }
    write_json(output_root / "artifacts" / "eval_summary.json", summary)
    return summary


def _score_predicted_verifier_payload(
    *,
    predicted: dict[str, Any],
    target_labels: dict[str, Any],
) -> float:
    scores = predicted.get("scores")
    if not isinstance(scores, dict):
        return 0.0
    try:
        process_pred = float(scores.get("process_reward"))
        progress_pred = float(scores.get("progress_reward"))
        total_pred = float(scores.get("total_reward"))
    except (TypeError, ValueError):
        return 0.0
    process_target = float(target_labels.get("process_reward", 0.0))
    progress_target = float(target_labels.get("progress_reward", 0.0))
    total_target = float(target_labels.get("total_reward", 0.0))
    error = (
        0.25 * abs(process_pred - process_target)
        + 0.40 * abs(progress_pred - progress_target)
        + 0.35 * abs(total_pred - total_target)
    )
    return clamp(1.0 - error, 0.0, 1.0)


def refine_verifier_with_sampled_rewards(
    *,
    config: dict[str, Any],
    output_root: Path,
    adapter_dir: Path,
    train_rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]] | tuple[None, None]:
    training_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    refinement_cfg = (
        training_cfg.get("refinement", {})
        if isinstance(training_cfg.get("refinement"), dict)
        else {}
    )
    if not bool(refinement_cfg.get("enabled", False)):
        return None, None
    verifier_cfg = config.get("verifier", {}) if isinstance(config.get("verifier"), dict) else {}
    sample_rows = list(train_rows[: max(1, int(refinement_cfg.get("max_examples", 8)))])
    if not sample_rows:
        return None, None

    artifacts_dir = ensure_dir(output_root / "artifacts")
    refinement_rows: list[dict[str, Any]] = []
    sample_count = max(1, int(refinement_cfg.get("samples_per_example", 3)))
    generation_max_new_tokens = int(refinement_cfg.get("max_new_tokens", 256))
    thinking_budget_tokens = int(refinement_cfg.get("thinking_budget_tokens", 512))
    enable_thinking = bool(refinement_cfg.get("enable_thinking", True))
    base_model = str(verifier_cfg.get("base_model") or DEFAULT_VERIFIER_MODEL)
    vllm_config = LocalVLLMEvalConfig(
        model=base_model,
        served_model_name=base_model,
        lora_name="pivot-verifier-lora",
        lora_path=str(adapter_dir),
        max_lora_rank=max(1, int(refinement_cfg.get("max_lora_rank", 16))),
        max_model_len=max(
            int(verifier_cfg.get("max_length", 4096)),
            generation_max_new_tokens + thinking_budget_tokens + 256,
        ),
        max_new_tokens=generation_max_new_tokens,
        enable_thinking=enable_thinking,
        enforce_eager=bool(refinement_cfg.get("enforce_eager", True)),
        port=int(refinement_cfg.get("vllm_port", 8014)),
        gpu_memory_utilization=float(refinement_cfg.get("gpu_memory_utilization", 0.90)),
        max_num_seqs=int(refinement_cfg.get("max_num_seqs", 8)),
        max_num_batched_tokens=int(refinement_cfg.get("max_num_batched_tokens", 4096)),
    )
    server_log_path = artifacts_dir / "verifier_refinement_vllm.log"
    release_cuda_memory()
    with local_vllm_server(config=vllm_config, log_path=server_log_path) as server:
        for row in sample_rows:
            best_payload = None
            best_reward = -1.0
            for _ in range(sample_count):
                _payload, parsed_payload, _parse_source, _response_text, _reasoning_text = request_verifier_completion(
                    server_base_url=str(server["base_url"]),
                    prompt_messages=row["prompt_messages"],
                    max_tokens=generation_max_new_tokens,
                    temperature=float(refinement_cfg.get("temperature", 0.7)),
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget_tokens,
                    guided_decoding_backend=vllm_config.guided_decoding_backend,
                )
                if not isinstance(parsed_payload, dict):
                    continue
                reward = _score_predicted_verifier_payload(
                    predicted=parsed_payload,
                    target_labels=row["labels"],
                )
                if reward > best_reward:
                    best_reward = reward
                    best_payload = parsed_payload
            if not isinstance(best_payload, dict):
                continue
            refinement_rows.append(
                {
                    "prompt_messages": row["prompt_messages"],
                    "messages": [
                        *row["prompt_messages"],
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(
                                str(item) for item in best_payload.get("critique", [])[:4]
                            ),
                            "tool_calls": [
                                {
                                    "id": f"refine_{row['example_id']}",
                                    "type": "function",
                                    "function": {
                                        "name": VERIFIER_TOOL_NAME,
                                        "arguments": best_payload,
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": [VERIFIER_REPORT_TOOL],
                    "response": json.dumps(best_payload, indent=2, sort_keys=True),
                    "weight": round(0.5 + best_reward, 4),
                }
            )
    write_jsonl(artifacts_dir / "pivot_verifier_refinement_rows.jsonl", refinement_rows)
    if not refinement_rows:
        summary = {
            "enabled": True,
            "refinement_examples": 0,
            "skipped": True,
            "reason": "no parseable refinement samples",
        }
        write_json(artifacts_dir / "verifier_refinement_summary.json", summary)
        return None, summary

    merged_examples = [
        {
            "prompt_messages": row["prompt_messages"],
            "messages": row["messages"],
            "tools": row["tools"],
            "response": row["response"],
            "weight": float(row.get("weight", 1.0)),
        }
        for row in train_rows
    ] + refinement_rows
    refined_adapter_dir = artifacts_dir / "pivot_verifier_refined_adapter"
    result = train_sft_with_trl(
        base_model=base_model,
        examples=merged_examples,
        output_dir=refined_adapter_dir,
        learning_rate=float(refinement_cfg.get("learning_rate", training_cfg.get("learning_rate", 1e-5))),
        epochs=max(1, int(refinement_cfg.get("epochs", 1))),
        max_length=int(verifier_cfg.get("max_length", 4096)),
        max_steps=int(refinement_cfg.get("max_steps", max(1, len(merged_examples)))),
        lora_rank=int(training_cfg.get("lora_rank", 16)),
    )
    summary = {
        "enabled": True,
        "adapter_dir": str(refined_adapter_dir),
        "base_model": base_model,
        "refinement_examples": len(refinement_rows),
        "examples_seen": int(result.examples_seen),
        "optimizer_steps": int(result.optimizer_steps),
        "mean_loss": float(result.mean_loss),
        "stage": "reward_weighted_refinement",
    }
    write_json(artifacts_dir / "verifier_refinement_summary.json", summary)
    return refined_adapter_dir, summary


def export_downstream_verifier_artifacts(
    *,
    output_root: Path,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    artifacts_dir = ensure_dir(output_root / "artifacts")
    reward_rows: list[dict[str, Any]] = []
    candidate_group_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in examples:
        state_key = _normalize_state_signature(str(row.get("state_text") or ""))
        target = str((row.get("metadata") or {}).get("rubric_target_achievement") or "unknown")
        reward_rows.append(
            {
                "example_id": row.get("example_id"),
                "state_text": row.get("state_text"),
                "thinking_text": row.get("thinking_text"),
                "actions": row.get("actions"),
                "next_state_text": row.get("next_state_text"),
                "process_reward": float((row.get("labels") or {}).get("process_reward", 0.0) or 0.0),
                "progress_reward": float((row.get("labels") or {}).get("progress_reward", 0.0) or 0.0),
                "total_reward": float((row.get("labels") or {}).get("total_reward", 0.0) or 0.0),
                "rubric_target_achievement": target,
            }
        )
        groups.setdefault((state_key, target), []).append(row)
    preference_rows: list[dict[str, Any]] = []
    for group_index, ((_state_key, target), rows) in enumerate(groups.items()):
        ranked = sorted(
            rows,
            key=lambda item: float((item.get("labels") or {}).get("total_reward", 0.0) or 0.0),
            reverse=True,
        )
        group_id = f"group_{group_index:05d}"
        candidate_group_rows.append(
            {
                "group_id": group_id,
                "rubric_target_achievement": target,
                "state_text": ranked[0].get("state_text") if ranked else "",
                "candidate_count": len(ranked),
                "candidates": [
                    {
                        "example_id": item.get("example_id"),
                        "thinking_text": item.get("thinking_text"),
                        "actions": item.get("actions"),
                        "next_state_text": item.get("next_state_text"),
                        "process_reward": float((item.get("labels") or {}).get("process_reward", 0.0) or 0.0),
                        "progress_reward": float((item.get("labels") or {}).get("progress_reward", 0.0) or 0.0),
                        "total_reward": float((item.get("labels") or {}).get("total_reward", 0.0) or 0.0),
                    }
                    for item in ranked
                ],
            }
        )
        if len(rows) < 2:
            continue
        rejected = ranked[-1]
        chosen = ranked[0]
        chosen_score = float((chosen.get("labels") or {}).get("total_reward", 0.0) or 0.0)
        rejected_score = float((rejected.get("labels") or {}).get("total_reward", 0.0) or 0.0)
        if chosen_score - rejected_score < 0.15:
            continue
        preference_rows.append(
            {
                "group_id": group_id,
                "pair_id": f"{chosen.get('example_id')}__vs__{rejected.get('example_id')}",
                "rubric_target_achievement": target,
                "state_text": chosen.get("state_text"),
                "prompt_messages": build_policy_prompt_messages(str(chosen.get("state_text") or "")),
                "chosen": {
                    "thinking_text": chosen.get("thinking_text"),
                    "actions": chosen.get("actions"),
                    "next_state_text": chosen.get("next_state_text"),
                    "score": chosen_score,
                },
                "rejected": {
                    "thinking_text": rejected.get("thinking_text"),
                    "actions": rejected.get("actions"),
                    "next_state_text": rejected.get("next_state_text"),
                    "score": rejected_score,
                },
            }
        )
    write_jsonl(artifacts_dir / "pivot_verifier_reward_rows.jsonl", reward_rows)
    write_jsonl(artifacts_dir / "pivot_verifier_candidate_groups.jsonl", candidate_group_rows)
    write_jsonl(artifacts_dir / "pivot_verifier_preference_pairs.jsonl", preference_rows)
    summary = {
        "reward_row_count": len(reward_rows),
        "candidate_group_count": len(candidate_group_rows),
        "preference_pair_count": len(preference_rows),
    }
    write_json(artifacts_dir / "pivot_verifier_downstream_summary.json", summary)
    return summary


POLICY_SYSTEM_PROMPT = (
    "You are a Craftax student policy.\n"
    "Use the provided `craftax_interact` tool exactly once for the final answer.\n"
    "Return a short useful macro-action for the current state.\n"
    "Do not return plain text actions or JSON.\n"
    "After reasoning, your final assistant action must be a tool call."
)


def build_policy_prompt_messages(state_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Current Craftax state summary:\n"
                f"{str(state_text).strip()}\n\n"
                "Choose the best next short action sequence."
            ),
        },
    ]


def build_policy_completion_messages(*, thinking_text: str, actions: list[str], example_id: str) -> list[dict[str, Any]]:
    safe_actions = [str(item).strip() for item in actions if str(item).strip()]
    return [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": str(thinking_text or "").strip(),
            "tool_calls": [
                {
                    "id": f"policy_call_{example_id}",
                    "type": "function",
                    "function": {
                        "name": "craftax_interact",
                        "arguments": {
                            "actions_list": safe_actions,
                        },
                    },
                }
            ],
        }
    ]


def render_policy_preference_text(
    *,
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    completion_messages: list[dict[str, Any]],
) -> tuple[str, str]:
    normalized_prompt_messages = [
        {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
        for item in prompt_messages
        if isinstance(item, dict)
    ]
    prompt_text = tokenizer.apply_chat_template(
        normalized_prompt_messages,
        tools=[CRAFTAX_INTERACT_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [*normalized_prompt_messages, *completion_messages],
        tools=[CRAFTAX_INTERACT_TOOL],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(prompt_text, str) or not isinstance(full_text, str):
        raise ValueError("policy chat template render failed")
    if not full_text.startswith(prompt_text):
        raise ValueError("full policy render does not share prompt prefix")
    return prompt_text, full_text[len(prompt_text) :]


def _tokenize_policy_preference_example(
    *,
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    max_length: int,
) -> dict[str, Any]:
    import torch

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    eos_tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    completion_budget = min(len(completion_ids) + len(eos_tokens), max_length)
    prompt_budget = max(0, max_length - completion_budget)
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
    input_ids = prompt_ids + completion_ids + eos_tokens
    labels = ([-100] * len(prompt_ids)) + completion_ids + eos_tokens
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def _mean_completion_logprob(*, model: Any, batch: dict[str, Any]) -> Any:
    import torch
    import torch.nn.functional as F

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_log_probs = torch.gather(
        F.log_softmax(shift_logits, dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    masked_log_probs = token_log_probs * valid_mask.to(token_log_probs.dtype)
    token_counts = valid_mask.sum(dim=-1).clamp(min=1)
    return masked_log_probs.sum(dim=-1) / token_counts


def train_downstream_policy_with_preferences(
    *,
    config: dict[str, Any],
    output_root: Path,
    preference_rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]] | tuple[None, None]:
    downstream_cfg = (
        config.get("downstream_policy", {})
        if isinstance(config.get("downstream_policy"), dict)
        else {}
    )
    if not bool(downstream_cfg.get("enabled", False)):
        return None, None
    if not preference_rows:
        summary = {"enabled": True, "skipped": True, "reason": "no preference rows"}
        write_json(output_root / "artifacts" / "downstream_policy_summary.json", summary)
        return None, summary

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    artifacts_dir = ensure_dir(output_root / "artifacts")
    base_model = str(downstream_cfg.get("base_model") or "Qwen/Qwen3.5-4B")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_rows: list[dict[str, Any]] = []
    limit = max(1, int(downstream_cfg.get("max_examples", len(preference_rows))))
    for row in preference_rows[:limit]:
        prompt_messages = row.get("prompt_messages")
        chosen = row.get("chosen") if isinstance(row.get("chosen"), dict) else {}
        rejected = row.get("rejected") if isinstance(row.get("rejected"), dict) else {}
        if not isinstance(prompt_messages, list):
            continue
        try:
            prompt_text, chosen_text = render_policy_preference_text(
                tokenizer=tokenizer,
                prompt_messages=[item for item in prompt_messages if isinstance(item, dict)],
                completion_messages=build_policy_completion_messages(
                    thinking_text=str(chosen.get("thinking_text") or ""),
                    actions=[str(item) for item in chosen.get("actions", []) if str(item).strip()],
                    example_id=str(row.get("pair_id") or "chosen"),
                ),
            )
            _prompt_text_check, rejected_text = render_policy_preference_text(
                tokenizer=tokenizer,
                prompt_messages=[item for item in prompt_messages if isinstance(item, dict)],
                completion_messages=build_policy_completion_messages(
                    thinking_text=str(rejected.get("thinking_text") or ""),
                    actions=[str(item) for item in rejected.get("actions", []) if str(item).strip()],
                    example_id=str(row.get("pair_id") or "rejected"),
                ),
            )
        except Exception:
            continue
        dataset_rows.append(
            {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "pair_id": str(row.get("pair_id") or ""),
                "score_margin": max(
                    0.05,
                    float(chosen.get("score", 0.0) or 0.0) - float(rejected.get("score", 0.0) or 0.0),
                ),
            }
        )
    write_jsonl(artifacts_dir / "pivot_policy_preference_dataset.jsonl", dataset_rows)
    if not dataset_rows:
        summary = {"enabled": True, "skipped": True, "reason": "no renderable preference rows"}
        write_json(artifacts_dir / "downstream_policy_summary.json", summary)
        return None, summary

    peft_config = LoraConfig(
        r=int(downstream_cfg.get("lora_rank", 8)),
        lora_alpha=int(downstream_cfg.get("lora_rank", 8)) * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=train_lora_lib.DEFAULT_TARGET_MODULES,
    )
    adapter_dir = artifacts_dir / "pivot_policy_preference_adapter"
    model = train_lora_lib._load_text_only_causal_lm(
        base_model=base_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=False,
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(downstream_cfg.get("learning_rate", 5e-6)))
    max_steps = int(downstream_cfg.get("max_steps", max(1, len(dataset_rows))))
    epochs = max(1, int(downstream_cfg.get("epochs", 1)))
    gradient_accumulation_steps = max(1, int(downstream_cfg.get("gradient_accumulation_steps", 1)))
    max_length = int(downstream_cfg.get("max_length", 1024))
    beta = float(downstream_cfg.get("beta", 0.2))
    chosen_distill_weight = float(downstream_cfg.get("chosen_distill_weight", 0.05))
    losses: list[float] = []
    optimizer_steps = 0

    for _ in range(epochs):
        for row in dataset_rows:
            chosen_batch = _tokenize_policy_preference_example(
                tokenizer=tokenizer,
                prompt_text=str(row["prompt"]),
                completion_text=str(row["chosen"]),
                max_length=max_length,
            )
            rejected_batch = _tokenize_policy_preference_example(
                tokenizer=tokenizer,
                prompt_text=str(row["prompt"]),
                completion_text=str(row["rejected"]),
                max_length=max_length,
            )
            chosen_batch = {key: value.to(model.device) for key, value in chosen_batch.items()}
            rejected_batch = {key: value.to(model.device) for key, value in rejected_batch.items()}
            chosen_logprob = _mean_completion_logprob(model=model, batch=chosen_batch)
            rejected_logprob = _mean_completion_logprob(model=model, batch=rejected_batch)
            margin = chosen_logprob - rejected_logprob
            weight = float(row.get("score_margin", 1.0) or 1.0)
            pairwise_loss = -torch.nn.functional.logsigmoid(beta * margin).mean()
            loss = pairwise_loss * weight
            if chosen_distill_weight > 0.0:
                loss = loss + (chosen_distill_weight * (-chosen_logprob.mean()))
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (optimizer_steps + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu().item()))
            if max_steps > 0 and optimizer_steps >= max_steps:
                break
        if max_steps > 0 and optimizer_steps >= max_steps:
            break

    if optimizer_steps % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    ensure_dir(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    del model
    release_cuda_memory()
    summary = {
        "enabled": True,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "preference_examples": len(dataset_rows),
        "max_steps": max_steps,
        "objective": "reference_free_pairwise_logsigmoid",
        "mean_loss": float(mean(losses) if losses else 0.0),
    }
    write_json(artifacts_dir / "downstream_policy_summary.json", summary)
    return adapter_dir, summary


def write_record_bundle(
    *,
    config: dict[str, Any],
    output_root: Path,
    bootstrap_summary: dict[str, Any],
    dataset_summary: dict[str, Any],
    training_summary: dict[str, Any] | None,
    refinement_summary: dict[str, Any] | None,
    eval_summary: dict[str, Any] | None,
    downstream_summary: dict[str, Any] | None,
    downstream_policy_summary: dict[str, Any] | None,
    command: str,
) -> None:
    task_cfg = config.get("task", {}) if isinstance(config.get("task"), dict) else {}
    verifier_cfg = config.get("verifier", {}) if isinstance(config.get("verifier"), dict) else {}
    metrics = {
        "dataset_example_count": int(dataset_summary.get("example_count", 0)),
        "dataset_train_count": int(dataset_summary.get("train_count", 0)),
        "dataset_holdout_count": int(dataset_summary.get("holdout_count", 0)),
        "dataset_mean_total_reward": float(dataset_summary.get("mean_total_reward", 0.0) or 0.0),
        "bootstrap_successful_rollouts": int(bootstrap_summary.get("successful_rollouts", 0)),
        "verifier_base_model": str(verifier_cfg.get("base_model") or DEFAULT_VERIFIER_MODEL),
        "method_name": str(task_cfg.get("method_name") or DEFAULT_METHOD_NAME),
    }
    if training_summary:
        metrics["training_mean_loss"] = float(training_summary.get("mean_loss", 0.0) or 0.0)
        metrics["training_optimizer_steps"] = int(training_summary.get("optimizer_steps", 0))
    if refinement_summary and not bool(refinement_summary.get("skipped", False)):
        metrics["refinement_mean_loss"] = float(refinement_summary.get("mean_loss", 0.0) or 0.0)
        metrics["refinement_optimizer_steps"] = int(refinement_summary.get("optimizer_steps", 0))
        metrics["refinement_examples"] = int(refinement_summary.get("refinement_examples", 0))
    if eval_summary:
        if eval_summary.get("mae_total_reward") is not None:
            metrics["eval_mae_total_reward"] = float(eval_summary["mae_total_reward"])
        if eval_summary.get("mae_progress_reward") is not None:
            metrics["eval_mae_progress_reward"] = float(eval_summary["mae_progress_reward"])
        if eval_summary.get("pairwise_accuracy") is not None:
            metrics["eval_pairwise_accuracy"] = float(eval_summary["pairwise_accuracy"])
        metrics["eval_json_parse_rate"] = float(eval_summary.get("json_parse_rate", 0.0) or 0.0)
    if downstream_summary:
        metrics["downstream_reward_row_count"] = int(downstream_summary.get("reward_row_count", 0))
        metrics["downstream_candidate_group_count"] = int(downstream_summary.get("candidate_group_count", 0))
        metrics["downstream_preference_pair_count"] = int(downstream_summary.get("preference_pair_count", 0))
    if downstream_policy_summary and not bool(downstream_policy_summary.get("skipped", False)):
        metrics["downstream_policy_preference_examples"] = int(downstream_policy_summary.get("preference_examples", 0))
        metrics["downstream_policy_mean_loss"] = float(downstream_policy_summary.get("mean_loss", 0.0) or 0.0)
    write_json(
        output_root / "metadata.json",
        {
            "track": str(task_cfg.get("track") or DEFAULT_TRACK_ID),
            "method_name": str(task_cfg.get("method_name") or DEFAULT_METHOD_NAME),
            "created_at": now_utc_iso(),
            "baseline_type": "spct_style_generative_verifier",
            "verifier_base_model": str(verifier_cfg.get("base_model") or DEFAULT_VERIFIER_MODEL),
        },
    )
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "system_info.json", system_info())
    write_text(output_root / "command.txt", command + "\n")
    write_text(output_root / "run_config.yaml", yaml.safe_dump(config, sort_keys=False))


def run_pipeline(*, config: dict[str, Any], output_root: Path, command: str, skip_train: bool, skip_eval: bool) -> dict[str, Any]:
    timer = Timer()
    output_root = ensure_dir(output_root)
    bootstrap_rollouts_path, bootstrap_summary = collect_or_load_rollouts(config=config, output_root=output_root)
    train_rows, holdout_rows, dataset_summary = build_dataset_artifacts(
        config=config,
        output_root=output_root,
        rollouts_path=bootstrap_rollouts_path,
    )
    training_summary = None
    refinement_summary = None
    eval_summary = None
    golden_eval_rows = load_golden_eval_examples(config=config)
    downstream_examples = golden_eval_rows if golden_eval_rows else [*train_rows, *holdout_rows]
    downstream_summary = export_downstream_verifier_artifacts(output_root=output_root, examples=downstream_examples)
    downstream_policy_summary = None
    adapter_dir = None
    if not skip_train:
        adapter_dir, training_summary = train_verifier_model(
            config=config,
            output_root=output_root,
            train_rows=train_rows,
        )
        refined_adapter_dir, refinement_summary = refine_verifier_with_sampled_rewards(
            config=config,
            output_root=output_root,
            adapter_dir=adapter_dir,
            train_rows=train_rows,
        )
        if refined_adapter_dir is not None:
            adapter_dir = refined_adapter_dir
        preference_rows = read_jsonl(output_root / "artifacts" / "pivot_verifier_preference_pairs.jsonl")
        _downstream_adapter_dir, downstream_policy_summary = train_downstream_policy_with_preferences(
            config=config,
            output_root=output_root,
            preference_rows=[row for row in preference_rows if isinstance(row, dict)],
        )
    if adapter_dir is not None and not skip_eval:
        eval_summary = evaluate_verifier_model(
            config=config,
            output_root=output_root,
            adapter_dir=adapter_dir,
            holdout_rows=holdout_rows,
        )
    elif skip_eval:
        eval_summary = {"skipped": True, "reason": "skip_eval flag set"}
        write_json(output_root / "artifacts" / "eval_summary.json", eval_summary)
    write_record_bundle(
        config=config,
        output_root=output_root,
        bootstrap_summary=bootstrap_summary,
        dataset_summary=dataset_summary,
        training_summary=training_summary,
        refinement_summary=refinement_summary,
        eval_summary=eval_summary,
        downstream_summary=downstream_summary,
        downstream_policy_summary=downstream_policy_summary,
        command=command,
    )
    result = {
        "output_root": str(output_root),
        "bootstrap_summary": bootstrap_summary,
        "dataset_summary": dataset_summary,
        "training_summary": training_summary,
        "refinement_summary": refinement_summary,
        "eval_summary": eval_summary,
        "downstream_summary": downstream_summary,
        "downstream_policy_summary": downstream_policy_summary,
        "elapsed_minutes": timer.elapsed_minutes,
        "completed_at": timer.ended_at,
    }
    write_json(output_root / "artifacts" / "pivot_verifier_result.json", result)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NanoHorizon pivot verifier baseline.")
    parser.add_argument(
        "--config",
        default="configs/pivot_verifier_qwen35_4b_spct.yaml",
        help="Path to the verifier baseline config.",
    )
    parser.add_argument("--output-dir", default="", help="Override the output record directory.")
    parser.add_argument("--skip-train", action="store_true", help="Build the dataset but skip LoRA training.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip held-out model evaluation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    config = load_config(resolve_path(args.config, base_dir=PROJECT_ROOT))
    task_cfg = config.get("task", {}) if isinstance(config.get("task"), dict) else {}
    track_id = str(task_cfg.get("track") or DEFAULT_TRACK_ID)
    method_name = str(task_cfg.get("method_name") or DEFAULT_METHOD_NAME)
    output_root = (
        resolve_path(args.output_dir, base_dir=PROJECT_ROOT)
        if str(args.output_dir).strip()
        else default_output_dir(track_id=track_id, method_name=method_name)
    )
    command = " ".join(shlex.quote(part) for part in [sys.executable, "-m", "nanohorizon.baselines.pivot_verifier", *list(argv or sys.argv[1:])])
    result = run_pipeline(
        config=config,
        output_root=Path(output_root),
        command=command,
        skip_train=bool(args.skip_train),
        skip_eval=bool(args.skip_eval or args.skip_train),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
