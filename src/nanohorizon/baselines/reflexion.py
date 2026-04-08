from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml

from nanohorizon.custom_vllm.runtime import build_thinking_budget_request_overrides
from nanohorizon.shared.openai_compat import create_chat_completion

from .prompt_opt import (
    TRACK_ID,
    CraftaxPromptOptAdapter,
    PromptOptExample,
    Timer,
    _achievement_summary,
    _actions_summary,
    _assistant_summary,
    _chat_base_url_from_rollout_inference_url,
    _feedback_for_rollout,
    _first_user_observation,
    _reasoning_summary,
    _summarize_eval,
    _truncate,
    _write_jsonl,
    rollout_turns,
    now_utc_iso,
    system_info,
    write_json,
    write_text,
)

def _extract_fenced_or_raw(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    block_match = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    if block_match:
        return str(block_match.group(1) or "").strip()
    return value


def _token_set(text: str) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9_]+", str(text or "").lower())
    stopwords = {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "into",
        "when",
        "only",
        "then",
        "next",
        "nearby",
        "early",
        "target",
        "action",
        "actions",
        "use",
        "using",
    }
    return {token for token in raw_tokens if len(token) >= 3 and token not in stopwords}


def _jaccard_similarity(lhs: set[str], rhs: set[str]) -> float:
    if not lhs or not rhs:
        return 0.0
    union = lhs | rhs
    if not union:
        return 0.0
    return float(len(lhs & rhs) / len(union))


def _flatten_rollout_actions(rollout: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for turn in rollout_turns(rollout):
        turn_actions = turn.get("actions")
        if isinstance(turn_actions, list):
            actions.extend(str(item) for item in turn_actions if str(item).strip())
    return actions


def _rollout_behavior_tags(
    rollout: dict[str, Any],
    *,
    outcome_reward: float | None = None,
    search_score: float | None = None,
) -> list[str]:
    actions = _flatten_rollout_actions(rollout)
    achievements = _achievement_summary(rollout)
    tags: list[str] = []
    if not actions:
        tags.append("no_actions")
    if actions:
        do_count = sum(1 for action in actions if action == "do")
        move_count = sum(1 for action in actions if action.startswith("move_"))
        unique_ratio = len(set(actions)) / max(1, len(actions))
        if do_count / max(1, len(actions)) >= 0.5:
            tags.append("do_heavy")
        if unique_ratio <= 0.5:
            tags.append("low_action_diversity")
        if move_count + do_count == len(actions):
            tags.append("movement_do_only")
        if len(actions) >= 4:
            tail = actions[-4:]
            if len(set(tail)) <= 2:
                tags.append("loop_risk")
    if achievements:
        tags.append("progress_signal")
    if (outcome_reward is not None and float(outcome_reward) <= 0.0) and not achievements:
        tags.append("no_progress_signal")
    if search_score is not None and float(search_score) >= 0.25:
        tags.append("strong_search_score")
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = str(tag or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _rollout_episode_card(
    *,
    seed: int,
    rollout: dict[str, Any],
    outcome_reward: float,
    search_score: float,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "outcome_reward": float(outcome_reward),
        "search_score": float(search_score),
        "achievements": _achievement_summary(rollout)[:6],
        "actions": _actions_summary(rollout),
        "behavior_tags": _rollout_behavior_tags(
            rollout,
            outcome_reward=outcome_reward,
            search_score=search_score,
        ),
        "feedback": _feedback_for_rollout(rollout, float(outcome_reward)),
        "observation": _truncate(_first_user_observation(rollout), 500),
    }


def _summarize_episode_cards(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return ""
    mean_reward = float(
        sum(float(card.get("outcome_reward", 0.0)) for card in cards) / max(1, len(cards))
    )
    mean_search = float(
        sum(float(card.get("search_score", 0.0)) for card in cards) / max(1, len(cards))
    )
    tag_counts: Counter[str] = Counter()
    achievement_counts: Counter[str] = Counter()
    for card in cards:
        tags = card.get("behavior_tags")
        if isinstance(tags, list):
            tag_counts.update(str(tag) for tag in tags if str(tag).strip())
        achievements = card.get("achievements")
        if isinstance(achievements, list):
            achievement_counts.update(str(item) for item in achievements if str(item).strip())
    top_tags = ", ".join(f"{tag}:{count}" for tag, count in tag_counts.most_common(6)) or "none"
    top_achievements = (
        ", ".join(f"{name}:{count}" for name, count in achievement_counts.most_common(6)) or "none"
    )
    return (
        f"episodes={len(cards)}; mean_outcome_reward={mean_reward:.3f}; "
        f"mean_search_score={mean_search:.3f}; top_behavior_tags={top_tags}; "
        f"top_achievements={top_achievements}"
    )


def _rollout_context_text(rollout: dict[str, Any]) -> str:
    parts = [
        _first_user_observation(rollout),
        _actions_summary(rollout),
        " ".join(_achievement_summary(rollout)),
        _assistant_summary(rollout),
    ]
    tags = _rollout_behavior_tags(rollout)
    if tags:
        parts.append("behavior_tags: " + ", ".join(tags))
    return "\n".join(part for part in parts if str(part or "").strip()).strip()


def _memory_similarity_to_rollout(
    *,
    rollout: dict[str, Any],
    memory_entry: dict[str, Any],
) -> float:
    rollout_tokens = _token_set(_rollout_context_text(rollout))
    memory_text = "\n".join(
        [
            str(memory_entry.get("lesson") or ""),
            str(memory_entry.get("context") or ""),
        ]
    )
    memory_tokens = _token_set(memory_text)
    return _jaccard_similarity(rollout_tokens, memory_tokens)


def _max_memory_similarity_for_rollout(
    *,
    rollout: dict[str, Any],
    memory_entries: list[dict[str, Any]],
) -> float:
    best_similarity, _ = _best_memory_match_for_rollout(
        rollout=rollout,
        memory_entries=memory_entries,
    )
    return best_similarity


def _best_memory_match_for_rollout(
    *,
    rollout: dict[str, Any],
    memory_entries: list[dict[str, Any]],
) -> tuple[float, str]:
    if not memory_entries:
        return 0.0, ""
    best_similarity = -1.0
    best_lesson = ""
    for entry in memory_entries:
        similarity = _memory_similarity_to_rollout(rollout=rollout, memory_entry=entry)
        if similarity > best_similarity:
            best_similarity = float(similarity)
            best_lesson = str(entry.get("lesson") or "")
    if best_similarity < 0.0:
        return 0.0, ""
    return best_similarity, best_lesson


def _parse_reflexion_lessons(text: str, *, max_items: int) -> list[str]:
    cleaned = _extract_fenced_or_raw(text)
    lessons: list[str] = []
    parsed: Any = None
    if cleaned:
        try:
            parsed = json.loads(cleaned)
        except Exception:
            object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if object_match:
                try:
                    parsed = json.loads(object_match.group(0))
                except Exception:
                    parsed = None
    if isinstance(parsed, dict):
        candidate_items = parsed.get("lessons")
        if isinstance(candidate_items, list):
            for item in candidate_items:
                lesson = str(item or "").strip()
                if lesson:
                    lessons.append(lesson)
    elif isinstance(parsed, list):
        for item in parsed:
            lesson = str(item or "").strip()
            if lesson:
                lessons.append(lesson)
    if not lessons:
        for line in cleaned.splitlines():
            stripped = line.strip()
            stripped = re.sub(r"^[\-\*\d\.\)\s]+", "", stripped).strip()
            if stripped:
                lessons.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for lesson in lessons:
        compact = " ".join(lesson.split())
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(_truncate(compact, 220))
        if len(deduped) >= max_items:
            break
    return deduped


def _render_reflexion_memory_block(
    memory_entries: list[dict[str, Any]],
    *,
    max_items: int,
    section_title: str,
) -> str:
    if not memory_entries:
        return ""
    recent = memory_entries[-max_items:]
    lines = [f"{section_title.strip() or 'Reflexion Memory'}:"]
    for entry in recent:
        lesson = str(entry.get("lesson") or "").strip()
        if lesson:
            lines.append(f"- {lesson}")
    lines.append(
        "Use these lessons as heuristics. If a memory conflicts with the current observation, trust the current observation."
    )
    return "\n".join(lines).strip()


def _compose_prompt_with_reflexion_memory(
    *,
    seed_prompt: str,
    memory_entries: list[dict[str, Any]],
    max_items: int,
    section_title: str,
) -> str:
    memory_block = _render_reflexion_memory_block(
        memory_entries,
        max_items=max_items,
        section_title=section_title,
    )
    if not memory_block:
        return seed_prompt.strip()
    return f"{seed_prompt.strip()}\n\n{memory_block}"


def _resolve_reflection_backend(
    *,
    requested_backend: str,
    requested_model: str,
    request_model: str,
) -> tuple[str, str]:
    backend = requested_backend.strip().lower() or "auto"
    openai_available = bool(str(os.getenv("OPENAI_API_KEY") or "").strip())
    if backend == "openai":
        return "openai", requested_model
    if backend == "policy_inference":
        return "policy_inference", request_model
    if backend == "auto" and openai_available and requested_model.startswith("gpt-"):
        return "openai", requested_model
    return "policy_inference", request_model


def _request_reflexion_lessons(
    *,
    reflection_backend: str,
    reflection_model: str,
    rollout_inference_url: str,
    inference_api_key: str,
    reflection_temperature: float,
    reflection_max_tokens: int,
    max_lessons_per_episode: int,
    current_memory: list[dict[str, Any]],
    memory_section_title: str,
    rollout: dict[str, Any],
    outcome_reward: float,
    search_score: float,
) -> tuple[list[str], str]:
    memory_block = _render_reflexion_memory_block(
        current_memory,
        max_items=max(1, min(len(current_memory), 8)),
        section_title=memory_section_title,
    )
    reflection_input = {
        "observation": _first_user_observation(rollout),
        "actions": _actions_summary(rollout),
        "assistant_response": _assistant_summary(rollout),
        "reasoning": _reasoning_summary(rollout),
        "achievements": _achievement_summary(rollout),
        "outcome_reward": float(outcome_reward),
        "search_score": float(search_score),
        "feedback": _feedback_for_rollout(rollout, outcome_reward),
    }
    system_message = (
        "You are a Reflexion memory writer for a Craftax policy. "
        "Write compact, actionable lessons from one episode. "
        "Each lesson should be a short imperative, not a paragraph."
    )
    user_message = (
        "Given the episode summary below, return JSON only with key `lessons` containing "
        f"1 to {max_lessons_per_episode} short strings. "
        "Keep lessons concrete and action-oriented for the next episode. "
        "Prioritize loop-breaking, resource progress, and strict tool-calling validity.\n\n"
        f"Current memory:\n{memory_block or '(none)'}\n\n"
        f"Episode summary:\n{json.dumps(reflection_input, indent=2, sort_keys=True)}"
    )
    base_url: str | None = None
    api_key: str | None = None
    extra_body: dict[str, Any] | None = None
    if reflection_backend == "policy_inference":
        base_url = _chat_base_url_from_rollout_inference_url(rollout_inference_url)
        api_key = inference_api_key
        extra_body = build_thinking_budget_request_overrides(enable_thinking=False, thinking_budget=0)
    payload = create_chat_completion(
        model=reflection_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max(128, int(reflection_max_tokens)),
        temperature=float(reflection_temperature),
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=180.0,
        extra_body=extra_body,
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return [], ""
    message = choices[0].get("message", {})
    content = str(message.get("content") or "").strip()
    lessons = _parse_reflexion_lessons(content, max_items=max(1, int(max_lessons_per_episode)))
    return lessons, content


def _request_reflexion_high_level_lessons(
    *,
    reflection_backend: str,
    reflection_model: str,
    rollout_inference_url: str,
    inference_api_key: str,
    reflection_temperature: float,
    reflection_max_tokens: int,
    max_lessons: int,
    current_memory: list[dict[str, Any]],
    memory_section_title: str,
    episode_cards: list[dict[str, Any]],
) -> tuple[list[str], str]:
    memory_block = _render_reflexion_memory_block(
        current_memory,
        max_items=max(1, min(len(current_memory), 8)),
        section_title=memory_section_title,
    )
    aggregate = _summarize_episode_cards(episode_cards)
    system_message = (
        "You are a Reflexion strategist for a Craftax policy. "
        "Synthesize high-level lessons that generalize across many episodes. "
        "Prefer trigger-action rules in the form `If <pattern>, then <action>`."
    )
    user_message = (
        "Given multiple rollout summaries from one training pass, return JSON only with key `lessons` "
        f"containing 1 to {max_lessons} strings. "
        "Each lesson must be robust and reusable, not tied to one exact seed.\n\n"
        f"Current memory:\n{memory_block or '(none)'}\n\n"
        f"Pass aggregate:\n{aggregate or '(none)'}\n\n"
        f"Episode cards:\n{json.dumps(episode_cards, indent=2, sort_keys=True)}"
    )
    base_url: str | None = None
    api_key: str | None = None
    extra_body: dict[str, Any] | None = None
    if reflection_backend == "policy_inference":
        base_url = _chat_base_url_from_rollout_inference_url(rollout_inference_url)
        api_key = inference_api_key
        extra_body = build_thinking_budget_request_overrides(enable_thinking=False, thinking_budget=0)
    payload = create_chat_completion(
        model=reflection_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max(128, int(reflection_max_tokens)),
        temperature=float(reflection_temperature),
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=180.0,
        extra_body=extra_body,
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return [], ""
    message = choices[0].get("message", {})
    content = str(message.get("content") or "").strip()
    lessons = _parse_reflexion_lessons(content, max_items=max(1, int(max_lessons)))
    return lessons, content


def _update_reflexion_memory(
    memory_entries: list[dict[str, Any]],
    *,
    new_lessons: list[str],
    max_memory_items: int,
    seed: int,
    outcome_reward: float,
    search_score: float,
    source_context: str = "",
    lesson_similarity_threshold: float = 0.65,
) -> list[dict[str, Any]]:
    if not new_lessons:
        return memory_entries
    updated = list(memory_entries)
    existing_signatures = [
        _token_set(
            "\n".join(
                [
                    str(entry.get("lesson") or ""),
                    str(entry.get("context") or ""),
                ]
            )
        )
        for entry in updated
    ]
    for lesson in new_lessons:
        normalized = " ".join(str(lesson or "").split()).strip()
        if not normalized:
            continue
        candidate_signature = _token_set("\n".join([normalized, source_context]))
        similarity_scores = [
            _jaccard_similarity(candidate_signature, signature) for signature in existing_signatures
        ]
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        best_match_idx = similarity_scores.index(max_similarity) if similarity_scores else -1
        if best_match_idx >= 0 and max_similarity >= float(lesson_similarity_threshold):
            existing = dict(updated[best_match_idx])
            previous_support = max(1, int(existing.get("support_count", 1)))
            next_support = previous_support + 1
            existing["support_count"] = int(next_support)
            existing["outcome_reward_mean"] = float(
                (
                    float(existing.get("outcome_reward_mean", existing.get("outcome_reward", 0.0)))
                    * previous_support
                    + float(outcome_reward)
                )
                / next_support
            )
            existing["search_score_mean"] = float(
                (
                    float(existing.get("search_score_mean", existing.get("search_score", 0.0)))
                    * previous_support
                    + float(search_score)
                )
                / next_support
            )
            if float(outcome_reward) >= float(existing.get("outcome_reward", float("-inf"))):
                existing["outcome_reward"] = float(outcome_reward)
                existing["search_score"] = float(search_score)
                existing["context"] = _truncate(source_context, 280)
                if seed >= 0:
                    existing["seed"] = int(seed)
            supporting_seeds = existing.get("supporting_seeds")
            seed_values = (
                [int(item) for item in supporting_seeds if isinstance(item, int)]
                if isinstance(supporting_seeds, list)
                else []
            )
            if seed >= 0 and seed not in seed_values:
                seed_values.append(int(seed))
                seed_values = seed_values[-24:]
            existing["supporting_seeds"] = seed_values
            existing["max_similarity_to_existing"] = float(
                max(float(existing.get("max_similarity_to_existing", 0.0)), float(max_similarity))
            )
            existing["timestamp_utc"] = now_utc_iso()
            updated[best_match_idx] = existing
            continue
        updated.append(
            {
                "lesson": normalized,
                "seed": int(seed),
                "outcome_reward": float(outcome_reward),
                "outcome_reward_mean": float(outcome_reward),
                "search_score": float(search_score),
                "search_score_mean": float(search_score),
                "context": _truncate(source_context, 280),
                "support_count": 1,
                "supporting_seeds": [int(seed)] if seed >= 0 else [],
                "max_similarity_to_existing": float(max_similarity),
                "timestamp_utc": now_utc_iso(),
            }
        )
        existing_signatures.append(candidate_signature)
    if len(updated) > max_memory_items:
        updated = updated[-max_memory_items:]
    return updated


def _mean_outcome_reward_for_candidate(
    *,
    dataset: list[PromptOptExample],
    candidate: dict[str, str],
    adapter: CraftaxPromptOptAdapter,
) -> float:
    if not dataset:
        return 0.0
    batch = adapter.evaluate(dataset, candidate, capture_traces=False)
    objective_scores = batch.objective_scores or []
    rewards = [
        float(score_map.get("outcome_reward", 0.0))
        for score_map in objective_scores
        if isinstance(score_map, dict)
    ]
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


def _greedy_select_memory_entries(
    *,
    seed_prompt: str,
    component_name: str,
    memory_entries: list[dict[str, Any]],
    max_memory_items: int,
    memory_section_title: str,
    trainset: list[PromptOptExample],
    adapter: CraftaxPromptOptAdapter,
    min_improvement: float,
) -> tuple[list[dict[str, Any]], float, float, list[dict[str, Any]]]:
    if not memory_entries:
        base_candidate = {component_name: seed_prompt}
        base_score = _mean_outcome_reward_for_candidate(
            dataset=trainset,
            candidate=base_candidate,
            adapter=adapter,
        )
        return [], base_score, base_score, []

    ranked_candidates = sorted(
        memory_entries,
        key=lambda entry: (
            int(entry.get("support_count", 1)),
            float(entry.get("outcome_reward_mean", entry.get("outcome_reward", 0.0))),
            float(entry.get("search_score_mean", entry.get("search_score", 0.0))),
            float(entry.get("outcome_reward", 0.0)),
            float(entry.get("search_score", 0.0)),
            str(entry.get("timestamp_utc") or ""),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    base_candidate = {component_name: seed_prompt}
    current_score = _mean_outcome_reward_for_candidate(
        dataset=trainset,
        candidate=base_candidate,
        adapter=adapter,
    )
    base_score = current_score
    trace_rows: list[dict[str, Any]] = []
    for entry in ranked_candidates:
        trial_selected = [*selected, entry]
        trial_prompt = _compose_prompt_with_reflexion_memory(
            seed_prompt=seed_prompt,
            memory_entries=trial_selected,
            max_items=max_memory_items,
            section_title=memory_section_title,
        )
        trial_candidate = {component_name: trial_prompt}
        trial_score = _mean_outcome_reward_for_candidate(
            dataset=trainset,
            candidate=trial_candidate,
            adapter=adapter,
        )
        improvement = float(trial_score - current_score)
        keep = bool(improvement >= float(min_improvement))
        trace_rows.append(
            {
                "candidate_lesson": str(entry.get("lesson") or ""),
                "candidate_seed": int(entry.get("seed") or 0),
                "candidate_outcome_reward": float(entry.get("outcome_reward", 0.0)),
                "candidate_outcome_reward_mean": float(
                    entry.get("outcome_reward_mean", entry.get("outcome_reward", 0.0))
                ),
                "candidate_search_score": float(entry.get("search_score", 0.0)),
                "candidate_search_score_mean": float(
                    entry.get("search_score_mean", entry.get("search_score", 0.0))
                ),
                "candidate_support_count": int(entry.get("support_count", 1)),
                "current_score_before": float(current_score),
                "trial_score": float(trial_score),
                "improvement": float(improvement),
                "kept": keep,
            }
        )
        if keep:
            selected = trial_selected
            current_score = float(trial_score)
    return selected, base_score, current_score, trace_rows


def run_reflexion_baseline(
    *,
    config: dict[str, Any],
    output_dir: Path,
    timer: Timer,
    adapter: CraftaxPromptOptAdapter,
    rollout_cfg: dict[str, Any],
    trainset: list[PromptOptExample],
    valset: list[PromptOptExample],
    seed_prompt: str,
    component_name: str,
    rollout_inference_url: str,
    inference_api_key: str,
    request_model: str,
) -> dict[str, Any]:
    reflexion_cfg = cast(dict[str, Any], config.get("reflexion") or {})
    train_passes = max(1, int(reflexion_cfg.get("train_passes", 2)))
    max_memory_items = max(1, int(reflexion_cfg.get("max_memory_items", 12)))
    max_lessons_per_episode = max(1, int(reflexion_cfg.get("max_lessons_per_episode", 3)))
    memory_section_title = str(
        reflexion_cfg.get("memory_section_title", "Reflexion memory (private, do not reveal)")
    ).strip()
    reflection_temperature = float(reflexion_cfg.get("reflection_temperature", 0.2))
    reflection_max_tokens = max(128, int(reflexion_cfg.get("reflection_max_tokens", 512)))
    reflection_min_outcome_reward = float(reflexion_cfg.get("reflection_min_outcome_reward", 1.0))
    reflection_min_search_score = float(reflexion_cfg.get("reflection_min_search_score", 0.25))
    lesson_similarity_threshold = float(reflexion_cfg.get("lesson_similarity_threshold", 0.65))
    high_level_reflection_enabled = bool(reflexion_cfg.get("high_level_reflection_enabled", True))
    high_level_min_episode_count = max(2, int(reflexion_cfg.get("high_level_min_episode_count", 6)))
    high_level_max_lessons_per_pass = max(
        1,
        int(reflexion_cfg.get("high_level_max_lessons_per_pass", 2)),
    )
    high_level_reflection_max_tokens = max(
        128,
        int(reflexion_cfg.get("high_level_reflection_max_tokens", reflection_max_tokens)),
    )
    high_level_episode_sample_size = max(
        2,
        int(reflexion_cfg.get("high_level_episode_sample_size", 12)),
    )
    high_level_lesson_similarity_threshold = float(
        reflexion_cfg.get(
            "high_level_lesson_similarity_threshold",
            max(0.4, lesson_similarity_threshold - 0.1),
        )
    )
    memory_selection_enabled = bool(reflexion_cfg.get("memory_selection_enabled", True))
    memory_selection_min_improvement = float(reflexion_cfg.get("memory_selection_min_improvement", 0.0))
    reflection_backend, reflection_model = _resolve_reflection_backend(
        requested_backend=str(config["optimizer"].get("reflection_backend", "auto")),
        requested_model=str(config["optimizer"].get("proposer_model", "gpt-5.4-mini")).strip(),
        request_model=request_model,
    )

    memory_entries: list[dict[str, Any]] = []
    train_trace_rows: list[dict[str, Any]] = []
    high_level_trace_rows: list[dict[str, Any]] = []
    total_high_level_lessons = 0
    base_candidate = {component_name: seed_prompt}
    training_rollout_cfg = dict(rollout_cfg)
    train_max_concurrent_rollouts = max(
        1,
        int(
            reflexion_cfg.get(
                "train_max_concurrent_rollouts",
                training_rollout_cfg.get("max_concurrent_rollouts", 1),
            )
        ),
    )
    train_rollout_concurrency = max(
        1,
        int(
            reflexion_cfg.get(
                "train_rollout_concurrency",
                training_rollout_cfg.get("rollout_concurrency", train_max_concurrent_rollouts),
            )
        ),
    )
    train_rollout_semaphore_limit = max(
        1,
        int(
            reflexion_cfg.get(
                "train_rollout_semaphore_limit",
                training_rollout_cfg.get("rollout_semaphore_limit", train_rollout_concurrency),
            )
        ),
    )
    training_rollout_cfg["max_concurrent_rollouts"] = train_max_concurrent_rollouts
    training_rollout_cfg["rollout_concurrency"] = train_rollout_concurrency
    training_rollout_cfg["rollout_semaphore_limit"] = train_rollout_semaphore_limit
    training_adapter = CraftaxPromptOptAdapter(
        container_url=adapter.container_url,
        inference_url=adapter.inference_url,
        inference_api_key=adapter.inference_api_key,
        request_model=adapter.request_model,
        rollout_cfg=training_rollout_cfg,
    )
    for pass_idx in range(train_passes):
        pass_episode_cards: list[dict[str, Any]] = []
        prompt_with_memory = _compose_prompt_with_reflexion_memory(
            seed_prompt=seed_prompt,
            memory_entries=memory_entries,
            max_items=max_memory_items,
            section_title=memory_section_title,
        )
        candidate = {component_name: prompt_with_memory}
        batch = training_adapter.evaluate(trainset, candidate, capture_traces=True)
        objective_scores = batch.objective_scores or []
        for index, example in enumerate(trainset):
            rollout = batch.outputs[index] if index < len(batch.outputs) and isinstance(batch.outputs[index], dict) else {}
            score_map = (
                objective_scores[index]
                if index < len(objective_scores) and isinstance(objective_scores[index], dict)
                else {}
            )
            outcome_reward = float(score_map.get("outcome_reward", 0.0))
            search_score = float(batch.scores[index]) if index < len(batch.scores) else 0.0
            parsed_lessons: list[str] = []
            reflection_text = ""
            reflection_error = ""
            reflection_skipped_reason = ""
            max_memory_similarity_before_update = 0.0
            best_matching_memory_lesson_before_update = ""
            if isinstance(rollout, dict) and rollout and memory_entries:
                (
                    max_memory_similarity_before_update,
                    best_matching_memory_lesson_before_update,
                ) = _best_memory_match_for_rollout(
                    rollout=rollout,
                    memory_entries=memory_entries,
                )
            if (
                isinstance(rollout, dict)
                and rollout
                and not rollout.get("error")
                and str(rollout.get("success_status") or "").strip().lower() == "success"
            ):
                pass_episode_cards.append(
                    _rollout_episode_card(
                        seed=example.seed,
                        rollout=rollout,
                        outcome_reward=outcome_reward,
                        search_score=search_score,
                    )
                )
            should_reflect = (
                isinstance(rollout, dict)
                and rollout
                and not rollout.get("error")
                and str(rollout.get("success_status") or "").strip().lower() == "success"
                and (
                    float(outcome_reward) >= float(reflection_min_outcome_reward)
                    or float(search_score) >= float(reflection_min_search_score)
                )
            )
            if should_reflect:
                try:
                    parsed_lessons, reflection_text = _request_reflexion_lessons(
                        reflection_backend=reflection_backend,
                        reflection_model=reflection_model,
                        rollout_inference_url=rollout_inference_url,
                        inference_api_key=inference_api_key,
                        reflection_temperature=reflection_temperature,
                        reflection_max_tokens=reflection_max_tokens,
                        max_lessons_per_episode=max_lessons_per_episode,
                        current_memory=memory_entries,
                        memory_section_title=memory_section_title,
                        rollout=rollout,
                        outcome_reward=outcome_reward,
                        search_score=search_score,
                    )
                except Exception as exc:
                    reflection_error = str(exc)
            else:
                if not isinstance(rollout, dict) or not rollout:
                    reflection_skipped_reason = "missing_rollout"
                elif rollout.get("error"):
                    reflection_skipped_reason = "rollout_error"
                elif str(rollout.get("success_status") or "").strip().lower() != "success":
                    reflection_skipped_reason = "unsuccessful_rollout"
                elif float(outcome_reward) < float(reflection_min_outcome_reward) and float(
                    search_score
                ) < float(reflection_min_search_score):
                    reflection_skipped_reason = "below_reflection_thresholds"
                else:
                    reflection_skipped_reason = "unknown"
            memory_entries = _update_reflexion_memory(
                memory_entries,
                new_lessons=parsed_lessons,
                max_memory_items=max_memory_items,
                seed=example.seed,
                outcome_reward=outcome_reward,
                search_score=search_score,
                source_context=_rollout_context_text(rollout) if isinstance(rollout, dict) else "",
                lesson_similarity_threshold=lesson_similarity_threshold,
            )
            train_trace_rows.append(
                {
                    "pass_index": int(pass_idx),
                    "seed": int(example.seed),
                    "split": example.split,
                    "outcome_reward": float(outcome_reward),
                    "search_score": float(search_score),
                    "memory_size_after_update": int(len(memory_entries)),
                    "new_lessons": parsed_lessons,
                    "reflection_backend": reflection_backend,
                    "reflection_model": reflection_model,
                    "reflection_error": reflection_error,
                    "reflection_skipped_reason": reflection_skipped_reason,
                    "max_memory_similarity_before_update": float(max_memory_similarity_before_update),
                    "best_matching_memory_lesson_before_update": best_matching_memory_lesson_before_update,
                    "reflection_raw_text": reflection_text,
                    "rollout_id": str(rollout.get("rollout_id") or "") if isinstance(rollout, dict) else "",
                }
            )
        sampled_episode_cards = sorted(
            pass_episode_cards,
            key=lambda card: (
                float(card.get("outcome_reward", 0.0)),
                float(card.get("search_score", 0.0)),
            ),
            reverse=True,
        )[:high_level_episode_sample_size]
        high_level_lessons: list[str] = []
        high_level_reflection_text = ""
        high_level_reflection_error = ""
        high_level_reflection_skipped_reason = ""
        if high_level_reflection_enabled:
            if len(sampled_episode_cards) >= high_level_min_episode_count:
                try:
                    high_level_lessons, high_level_reflection_text = _request_reflexion_high_level_lessons(
                        reflection_backend=reflection_backend,
                        reflection_model=reflection_model,
                        rollout_inference_url=rollout_inference_url,
                        inference_api_key=inference_api_key,
                        reflection_temperature=reflection_temperature,
                        reflection_max_tokens=high_level_reflection_max_tokens,
                        max_lessons=high_level_max_lessons_per_pass,
                        current_memory=memory_entries,
                        memory_section_title=memory_section_title,
                        episode_cards=sampled_episode_cards,
                    )
                except Exception as exc:
                    high_level_reflection_error = str(exc)
            else:
                high_level_reflection_skipped_reason = "insufficient_episode_cards"
        else:
            high_level_reflection_skipped_reason = "disabled"
        if high_level_lessons:
            mean_pass_reward = float(
                sum(float(card.get("outcome_reward", 0.0)) for card in sampled_episode_cards)
                / max(1, len(sampled_episode_cards))
            )
            mean_pass_search = float(
                sum(float(card.get("search_score", 0.0)) for card in sampled_episode_cards)
                / max(1, len(sampled_episode_cards))
            )
            memory_entries = _update_reflexion_memory(
                memory_entries,
                new_lessons=high_level_lessons,
                max_memory_items=max_memory_items,
                seed=-(pass_idx + 1),
                outcome_reward=mean_pass_reward,
                search_score=mean_pass_search,
                source_context=_summarize_episode_cards(sampled_episode_cards),
                lesson_similarity_threshold=high_level_lesson_similarity_threshold,
            )
            total_high_level_lessons += len(high_level_lessons)
        high_level_trace_rows.append(
            {
                "pass_index": int(pass_idx),
                "num_episode_cards": int(len(pass_episode_cards)),
                "num_sampled_episode_cards": int(len(sampled_episode_cards)),
                "high_level_reflection_enabled": bool(high_level_reflection_enabled),
                "high_level_reflection_skipped_reason": high_level_reflection_skipped_reason,
                "high_level_reflection_error": high_level_reflection_error,
                "high_level_lessons": high_level_lessons,
                "high_level_reflection_raw_text": high_level_reflection_text,
                "memory_size_after_high_level_update": int(len(memory_entries)),
                "pass_rollout_summary": _summarize_episode_cards(sampled_episode_cards),
            }
        )
    selected_memory_entries = list(memory_entries)
    memory_selection_trace: list[dict[str, Any]] = []
    memory_selection_train_base_score: float | None = None
    memory_selection_train_selected_score: float | None = None
    if memory_selection_enabled and memory_entries:
        (
            selected_memory_entries,
            memory_selection_train_base_score,
            memory_selection_train_selected_score,
            memory_selection_trace,
        ) = _greedy_select_memory_entries(
            seed_prompt=seed_prompt,
            component_name=component_name,
            memory_entries=memory_entries,
            max_memory_items=max_memory_items,
            memory_section_title=memory_section_title,
            trainset=trainset,
            adapter=training_adapter,
            min_improvement=memory_selection_min_improvement,
        )
    best_prompt = _compose_prompt_with_reflexion_memory(
        seed_prompt=seed_prompt,
        memory_entries=selected_memory_entries,
        max_items=max_memory_items,
        section_title=memory_section_title,
    )
    best_candidate = {component_name: best_prompt}

    base_eval = _summarize_eval(
        dataset=valset,
        candidate=base_candidate,
        adapter=adapter,
        name="base_eval",
        output_dir=output_dir,
    )
    best_eval = _summarize_eval(
        dataset=valset,
        candidate=best_candidate,
        adapter=adapter,
        name="best_eval",
        output_dir=output_dir,
    )
    base_mean = float(base_eval["mean_outcome_reward"])
    memory_mean = float(best_eval["mean_outcome_reward"])
    selected_is_memory = memory_mean >= base_mean
    selected_candidate = best_candidate if selected_is_memory else base_candidate
    selected_eval = best_eval if selected_is_memory else base_eval

    _write_jsonl(output_dir / "reflexion_train_trace.jsonl", train_trace_rows)
    _write_jsonl(output_dir / "reflexion_high_level_trace.jsonl", high_level_trace_rows)
    _write_jsonl(output_dir / "reflexion_memory_selection_trace.jsonl", memory_selection_trace)
    write_json(
        output_dir / "reflexion_memory.json",
        {
            "memory_section_title": memory_section_title,
            "max_memory_items": max_memory_items,
            "train_passes": train_passes,
            "entries": memory_entries,
            "selected_entries": selected_memory_entries,
            "lesson_similarity_threshold": lesson_similarity_threshold,
            "high_level_reflection_enabled": high_level_reflection_enabled,
            "high_level_min_episode_count": high_level_min_episode_count,
            "high_level_max_lessons_per_pass": high_level_max_lessons_per_pass,
            "high_level_reflection_max_tokens": high_level_reflection_max_tokens,
            "high_level_episode_sample_size": high_level_episode_sample_size,
            "high_level_lesson_similarity_threshold": high_level_lesson_similarity_threshold,
            "num_high_level_lessons": total_high_level_lessons,
            "memory_selection_enabled": memory_selection_enabled,
            "memory_selection_min_improvement": memory_selection_min_improvement,
            "memory_selection_train_base_score": memory_selection_train_base_score,
            "memory_selection_train_selected_score": memory_selection_train_selected_score,
        },
    )
    prompt_bundle = {
        "seed_candidate": base_candidate,
        "best_candidate": selected_candidate,
        "best_candidate_idx": 1 if selected_is_memory else 0,
        "candidates": [base_candidate, best_candidate],
        "val_aggregate_scores": [
            base_mean,
            memory_mean,
        ],
        "total_metric_calls": int(train_passes * len(trainset)),
        "method": "reflexion_baseline",
        "selected_prompt_source": "memory_prompt" if selected_is_memory else "seed_prompt",
    }
    write_json(output_dir / "prompt_bundle.json", prompt_bundle)
    run_config = {
        "track": str(config.get("track") or TRACK_ID),
        "task": str(config["task"]["name"]),
        "base_model": str(config["policy"]["model"]),
        "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
        "optimizer_models": list(config["optimizer"]["allowed_models"]),
        "rollout": rollout_cfg,
        "train_seeds": [example.seed for example in trainset],
        "eval_seeds": [example.seed for example in valset],
        "algorithm": "reflexion",
        "reflexion": {
            "train_passes": train_passes,
            "max_memory_items": max_memory_items,
            "max_lessons_per_episode": max_lessons_per_episode,
            "reflection_backend": reflection_backend,
            "reflection_model": reflection_model,
            "reflection_temperature": reflection_temperature,
            "reflection_max_tokens": reflection_max_tokens,
            "reflection_min_outcome_reward": reflection_min_outcome_reward,
            "reflection_min_search_score": reflection_min_search_score,
            "lesson_similarity_threshold": lesson_similarity_threshold,
            "high_level_reflection_enabled": high_level_reflection_enabled,
            "high_level_min_episode_count": high_level_min_episode_count,
            "high_level_max_lessons_per_pass": high_level_max_lessons_per_pass,
            "high_level_reflection_max_tokens": high_level_reflection_max_tokens,
            "high_level_episode_sample_size": high_level_episode_sample_size,
            "high_level_lesson_similarity_threshold": high_level_lesson_similarity_threshold,
            "num_high_level_lessons": total_high_level_lessons,
            "memory_selection_enabled": memory_selection_enabled,
            "memory_selection_min_improvement": memory_selection_min_improvement,
            "memory_selection_train_base_score": memory_selection_train_base_score,
            "memory_selection_train_selected_score": memory_selection_train_selected_score,
            "memory_section_title": memory_section_title,
            "selected_prompt_source": "memory_prompt" if selected_is_memory else "seed_prompt",
        },
    }
    write_text(output_dir / "run_config.yaml", yaml.safe_dump(run_config, sort_keys=True))
    write_json(
        output_dir / "metadata.json",
        {
            "name": os.environ.get("NANOHORIZON_RECORD_NAME", "reflexion_baseline"),
            "track": str(config.get("track") or TRACK_ID),
            "task": str(config["task"]["name"]),
            "base_model": str(config["policy"]["model"]),
            "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
            "optimizer_models": list(config["optimizer"]["allowed_models"]),
            "created_at": now_utc_iso()[:10],
            "algorithm": "reflexion",
        },
    )
    metrics = {
        "track": str(config.get("track") or TRACK_ID),
        "baseline": "reflexion_prompt_memory",
        "status": "success",
        "submission_mean_outcome_reward": float(
            selected_eval.get(
                "mean_outcome_reward_over_requested_rollouts",
                selected_eval["mean_outcome_reward"],
            )
        ),
        "submission_achievement_frequencies": selected_eval.get("achievement_frequencies", {}),
        "primary_score": float(selected_eval["mean_outcome_reward"]),
        "primary_achievement_frequencies": selected_eval.get("achievement_frequencies", {}),
        "bootstrap_score": base_mean,
        "bootstrap_achievement_frequencies": base_eval.get("achievement_frequencies", {}),
        "score_delta": float(selected_eval["mean_outcome_reward"] - base_mean),
        "memory_prompt_score": memory_mean,
        "memory_prompt_delta": float(memory_mean - base_mean),
        "num_train_episodes": int(train_passes * len(trainset)),
        "num_memory_entries": int(len(memory_entries)),
        "num_selected_memory_entries": int(len(selected_memory_entries)),
        "lesson_similarity_threshold": float(lesson_similarity_threshold),
        "high_level_reflection_enabled": bool(high_level_reflection_enabled),
        "high_level_min_episode_count": int(high_level_min_episode_count),
        "high_level_max_lessons_per_pass": int(high_level_max_lessons_per_pass),
        "high_level_reflection_max_tokens": int(high_level_reflection_max_tokens),
        "high_level_episode_sample_size": int(high_level_episode_sample_size),
        "high_level_lesson_similarity_threshold": float(high_level_lesson_similarity_threshold),
        "num_high_level_lessons": int(total_high_level_lessons),
        "memory_selection_enabled": bool(memory_selection_enabled),
        "memory_selection_min_improvement": float(memory_selection_min_improvement),
        "memory_selection_train_base_score": memory_selection_train_base_score,
        "memory_selection_train_selected_score": memory_selection_train_selected_score,
        "elapsed_minutes": timer.elapsed_minutes,
        "policy_model": str(config["policy"]["model"]),
        "reflection_model": reflection_model,
        "request_model": request_model,
        "rollout_inference_url": rollout_inference_url,
        "reflection_backend": reflection_backend,
        "method": "reflexion",
        "selected_prompt_source": "memory_prompt" if selected_is_memory else "seed_prompt",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "system_info.json", system_info())
    write_text(
        output_dir / "command.txt",
        "./scripts/run_craftax_reflexion_nano_baseline.sh\n",
    )
    write_text(
        output_dir / "notes.md",
        (
            f"Bootstrap held-out reward: {base_eval['mean_outcome_reward']:.3f}\n"
            f"Reflexion memory-prompt held-out reward: {best_eval['mean_outcome_reward']:.3f}\n"
            f"Selected held-out reward: {selected_eval['mean_outcome_reward']:.3f}\n"
            f"Selected delta vs bootstrap: {selected_eval['mean_outcome_reward'] - base_eval['mean_outcome_reward']:.3f}\n"
            f"Selected prompt source: {'memory_prompt' if selected_is_memory else 'seed_prompt'}\n"
            f"Memory entries: {len(memory_entries)}\n"
            f"High-level lessons added: {total_high_level_lessons}\n"
        ),
    )
    return {
        "output_dir": str(output_dir),
        "best_prompt": str(selected_candidate[component_name]),
        "metrics": metrics,
        "base_eval": base_eval,
        "best_eval": best_eval,
        "reflexion_memory_entries": memory_entries,
    }
