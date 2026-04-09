#!/usr/bin/env python3
"""Go-Explore prompt optimization for Craftax.

Self-contained script — only needs craftax, jax, and httpx (all on PyPI).
Searches for system prompts that maximize average reward on Craftax
using no more than 500 training rollouts.

Install: uv pip install "craftax>=1.5" "jax>=0.5" httpx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse

import httpx

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# ---------------------------------------------------------------------------
# Craftax environment setup (inlined from nanohorizon.craftax_core)
# ---------------------------------------------------------------------------

TOOL_NAME = "craftax_interact"

FULL_ACTIONS = {
    "noop": 0, "move_left": 1, "move_right": 2, "move_up": 3, "move_down": 4,
    "do": 5, "sleep": 6, "place_stone": 7, "place_table": 8, "place_furnace": 9,
    "place_plant": 10, "make_wood_pickaxe": 11, "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13, "make_wood_sword": 14, "make_stone_sword": 15,
    "make_iron_sword": 16, "rest": 17, "descend": 18, "ascend": 19,
    "make_diamond_pickaxe": 20, "make_diamond_sword": 21, "make_iron_armour": 22,
    "make_diamond_armour": 23, "shoot_arrow": 24, "make_arrow": 25,
    "cast_fireball": 26, "cast_iceball": 27, "place_torch": 28,
    "drink_potion_red": 29, "drink_potion_green": 30, "drink_potion_blue": 31,
    "drink_potion_pink": 32, "drink_potion_cyan": 33, "drink_potion_yellow": 34,
    "read_book": 35, "enchant_sword": 36, "enchant_armour": 37, "make_torch": 38,
    "level_up_dexterity": 39, "level_up_strength": 40, "level_up_intelligence": 41,
    "enchant_bow": 42,
}
ACTION_NAMES = sorted(FULL_ACTIONS.keys())

DEFAULT_SEED_SPLIT = Path("data/craftax/craftax_prompt_opt_eval20_seeds.json")
TRAINING_SEED_COUNT = 6
HOLDOUT_SEED_COUNT = 20
DEFAULT_VARIANTS_PER_ITERATION = 2
DEFAULT_MAX_TURNS = 10
TARGET_ACTION_BATCH_SIZE = 1

FULL_ACHIEVEMENTS = {
    0: "collect_wood", 1: "place_table", 2: "eat_cow", 3: "collect_sapling",
    4: "collect_drink", 5: "make_wood_pickaxe", 6: "make_wood_sword",
    7: "place_plant", 8: "defeat_zombie", 9: "collect_stone", 10: "place_stone",
    11: "eat_plant", 12: "defeat_skeleton", 13: "make_stone_pickaxe",
    14: "make_stone_sword", 15: "wake_up", 16: "place_furnace", 17: "collect_coal",
    18: "collect_iron", 19: "collect_diamond", 20: "make_iron_pickaxe",
    21: "make_iron_sword", 22: "make_arrow", 23: "make_torch", 24: "place_torch",
    25: "make_diamond_sword", 26: "make_iron_armour", 27: "make_diamond_armour",
}


def _make_runner(seed: int):
    """Create a Craftax full env runner."""
    import jax
    from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

    env = CraftaxSymbolicEnv()
    rng = jax.random.PRNGKey(seed)
    params = env.default_params
    obs, state = env.reset(rng, params)
    return env, params, state, obs, rng


def _render_text(state) -> str:
    """Simple text representation of craftax state for LLM consumption."""
    lines = ["Craftax State:"]
    inv = state.inventory
    scalar_items = [
        ("wood", inv.wood), ("stone", inv.stone), ("coal", inv.coal),
        ("iron", inv.iron), ("diamond", inv.diamond), ("sapling", inv.sapling),
        ("pickaxe_level", inv.pickaxe), ("sword_level", inv.sword),
        ("arrows", inv.arrows), ("torches", inv.torches),
        ("bow", inv.bow), ("books", inv.books),
    ]
    inv_str = ", ".join(f"{name}={int(val)}" for name, val in scalar_items if int(val) > 0)
    lines.append(f"Inventory: {inv_str or 'empty'}")
    lines.append(f"Health: {int(state.player_health)}")
    lines.append(f"Food: {int(state.player_food)}")
    lines.append(f"Drink: {int(state.player_drink)}")
    lines.append(f"Energy: {int(state.player_energy)}")
    lines.append(f"Position: ({int(state.player_position[0])}, {int(state.player_position[1])})")
    lines.append(f"Level: {int(state.player_level)}")
    achievements = []
    for idx, name in sorted(FULL_ACHIEVEMENTS.items()):
        if idx < len(state.achievements) and bool(state.achievements[idx]):
            achievements.append(name)
    if achievements:
        lines.append(f"Achievements: {', '.join(achievements)}")
    lines.append(f"Sleeping: {bool(state.is_sleeping)}")
    return "\n".join(lines)


def _load_seed_split(base_dir: Path) -> tuple[list[int], list[int]]:
    seed_file = (base_dir / DEFAULT_SEED_SPLIT).resolve()
    payload = json.loads(seed_file.read_text(encoding="utf-8"))
    train_seeds = [int(item) for item in payload.get("train_seeds", [])[:TRAINING_SEED_COUNT]]
    eval_seeds = [int(item) for item in payload.get("eval_seeds", [])[:HOLDOUT_SEED_COUNT]]
    if len(train_seeds) < TRAINING_SEED_COUNT or len(eval_seeds) < HOLDOUT_SEED_COUNT:
        raise RuntimeError(f"seed split in {seed_file} is too small for the task")
    return train_seeds, eval_seeds


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _tool_schema() -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Choose the next short Craftax macro-action sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions_list": {
                        "type": "array",
                        "items": {"type": "string", "enum": ACTION_NAMES},
                        "minItems": TARGET_ACTION_BATCH_SIZE,
                        "maxItems": TARGET_ACTION_BATCH_SIZE,
                    }
                },
                "required": ["actions_list"],
                "additionalProperties": False,
            },
        },
    }]


def _sanitize_actions(values: list[object]) -> list[str]:
    sanitized = []
    for value in values:
        raw = str(value).strip().lower()
        if raw in FULL_ACTIONS and raw not in sanitized:
            sanitized.append(raw)
            continue
        for token in re.findall(r"[a-z_]+", raw):
            if token in FULL_ACTIONS and token not in sanitized:
                sanitized.append(token)
    return sanitized


def _extract_actions(payload: dict[str, Any]) -> list[str]:
    message = payload.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            func = item.get("function", {})
            name = str(func.get("name", "")).strip()
            if name != TOOL_NAME:
                continue
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict):
                values = args.get("actions_list", [])
                if isinstance(values, list):
                    actions = _sanitize_actions(values)
                    if actions:
                        return actions
    return []


def _chat_completion(
    *, inference_url: str, model: str, api_key: str,
    messages: list[dict[str, Any]], temperature: float, max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": _tool_schema(),
        "tool_choice": "auto",
    }
    url = inference_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    timeout = httpx.Timeout(float(timeout_s), connect=30.0)
    for attempt in range(1, 6):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=body)
            if resp.status_code == 429 and attempt < 5:
                time.sleep(min(5.0, float(attempt)))
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"LLM request failed status={resp.status_code} body={resp.text[:500]}")
            return resp.json()
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt >= 5:
                raise RuntimeError(f"LLM request failed after retries: {exc!r}") from exc
            time.sleep(min(5.0, float(attempt)))
    raise RuntimeError("LLM request failed after retries")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    *, inference_url: str, model: str, api_key: str,
    seed: int, max_steps: int, system_prompt: str,
    target_action_batch_size: int = TARGET_ACTION_BATCH_SIZE,
) -> dict[str, Any]:
    """Run one Craftax episode with an LLM agent. Returns reward info."""
    import jax
    import jax.numpy as jnp

    env, params, state, obs, rng = _make_runner(seed)
    unique_achievements: set[str] = set()
    total_reward = 0.0
    llm_call_count = 0
    done = False

    for turn in range(max_steps):
        if done:
            break
        obs_text = _render_text(state)
        user_prompt = (
            f"Current Craftax observation:\n{obs_text}\n\n"
            f"Plan a short useful macro-action. Use the {TOOL_NAME} tool exactly once. "
            f"Return exactly {target_action_batch_size} actions. "
            "Use only valid Craftax actions."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = _chat_completion(
            inference_url=inference_url, model=model, api_key=api_key,
            messages=messages, temperature=0.0, max_tokens=180, timeout_s=45,
        )
        llm_call_count += 1
        actions = _extract_actions(payload)
        if not actions:
            actions = ["noop"]
        for action_name in actions:
            if done:
                break
            action_idx = FULL_ACTIONS.get(action_name, 0)
            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done_flag, info = env.step(step_rng, state, action_idx, params)
            total_reward += float(reward)
            done = bool(done_flag)
            for idx, name in FULL_ACHIEVEMENTS.items():
                if idx < len(state.achievements) and bool(state.achievements[idx]):
                    unique_achievements.add(name)

    return {
        "seed": seed,
        "total_reward": total_reward,
        "unique_achievements": len(unique_achievements),
        "achievement_names": sorted(unique_achievements),
        "llm_calls": llm_call_count,
        "reward": float(len(unique_achievements)),  # primary metric = achievement count
    }


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

def evaluate_prompt(
    *, prompt: str, seeds: list[int], max_steps: int,
    inference_url: str, model: str, api_key: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Evaluate a prompt across seeds sequentially (JAX JIT doesn't parallelize well)."""
    results = []
    for seed in seeds:
        try:
            result = run_rollout(
                inference_url=inference_url, model=model, api_key=api_key,
                seed=seed, max_steps=max_steps, system_prompt=prompt,
            )
            results.append(result)
        except Exception as exc:
            results.append({"seed": seed, "reward": 0.0, "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Go-Explore search
# ---------------------------------------------------------------------------

BASELINE_PROMPT = (
    "You are a Craftax policy agent. Think carefully, then use the "
    "`craftax_interact` tool exactly once. Return 1 valid full-Craftax action "
    "unless the episode is already done. Use only the tool call as the final "
    "answer. Do not output JSON, prose, or a plain-text action list."
)

MUTATION_SEEDS = [
    "Focus on food and water first — stable sustenance before exploration.",
    "Build tools in order: wooden pickaxe → stone pickaxe → iron pickaxe.",
    "Avoid combat until you have a sword and full health.",
    "Harvest every resource patch completely before moving on.",
    "Explore in expanding rings from spawn — don't wander randomly.",
    "Craft a workbench immediately, then prioritize tool upgrades.",
    "Keep inventory organized: always carry wood, stone, and food.",
    "When low on health, retreat and eat; never fight hungry.",
    "Prioritize achievements: each new achievement gives bonus reward.",
    "Mine downward for iron and diamond after securing surface resources.",
    "Place a table first — it unlocks all crafting recipes.",
    "Plant saplings after cutting trees to ensure wood supply.",
    "Use do action on the correct tile — standing on the resource matters.",
    "Plan 3-4 actions ahead as a batch — avoid single reactive moves.",
    "Alternate between resource gathering and crafting in short bursts.",
]


def _mutate(base: str, rng: random.Random) -> str:
    advice = rng.choice(MUTATION_SEEDS)
    strategy = rng.choice(["append", "prepend", "blend"])
    if strategy == "append":
        return f"{base} {advice}"
    if strategy == "prepend":
        return f"{advice} {base}"
    sentences = base.split(". ")
    keep = max(1, len(sentences) // 2)
    return ". ".join(sentences[:keep]) + ". " + advice


def _crossover(a: str, b: str, rng: random.Random) -> str:
    a_parts = [s.strip() for s in a.split(". ") if s.strip()]
    b_parts = [s.strip() for s in b.split(". ") if s.strip()]
    selected = [p for p in a_parts if rng.random() < 0.5]
    selected += [p for p in b_parts if rng.random() < 0.3]
    if not selected:
        selected = a_parts[:2] + b_parts[:1]
    return ". ".join(selected) + "."


def run_go_explore(
    *, output_dir: Path, inference_url: str, model: str, api_key: str,
    max_training_rollouts: int = 500,
    training_seeds: list[int], heldout_seeds: list[int],
    max_steps: int = 10, concurrency: int = 5,
    variants_per_iteration: int = 4,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    experiment_id = f"go_explore_{uuid.uuid4().hex[:8]}"

    archive: list[dict[str, Any]] = []
    iteration_log: list[dict[str, Any]] = []
    total_rollouts = 0
    best_prompt = BASELINE_PROMPT
    best_reward = -float("inf")

    _print = lambda *a, **kw: print(*a, **kw, flush=True)
    _print(f"[go-explore] Experiment {experiment_id}")
    _print(f"[go-explore] Budget: {max_training_rollouts} rollouts, {len(training_seeds)} train seeds, {len(heldout_seeds)} heldout seeds")

    # Phase 1: Baseline
    eval_subset = rng.sample(training_seeds, min(TRAINING_SEED_COUNT, len(training_seeds)))
    _print(f"[go-explore] Phase 1: Baseline on {len(eval_subset)} seeds...")
    baseline_results = evaluate_prompt(
        prompt=BASELINE_PROMPT, seeds=eval_subset, max_steps=max_steps,
        inference_url=inference_url, model=model, api_key=api_key,
        concurrency=concurrency,
    )
    baseline_mean = mean([r["reward"] for r in baseline_results])
    total_rollouts += len(baseline_results)
    archive.append({"prompt": BASELINE_PROMPT, "mean_reward": baseline_mean, "source": "baseline"})
    best_reward = baseline_mean
    _print(f"[go-explore] Baseline: {baseline_mean:.3f} ({total_rollouts} rollouts)")
    iteration_log.append({"iteration": 0, "phase": "baseline", "best_reward": baseline_mean, "rollouts": total_rollouts})

    # Phase 2: Search
    iteration = 0
    while total_rollouts + len(eval_subset) <= max_training_rollouts:
        iteration += 1
        _print(f"\n[go-explore] Iteration {iteration} ({total_rollouts}/{max_training_rollouts} rollouts)")
        iter_results = []
        for vi in range(variants_per_iteration):
            if total_rollouts + len(eval_subset) > max_training_rollouts:
                break
            if len(archive) >= 2 and rng.random() < 0.3:
                parents = rng.sample(archive, 2)
                variant = _crossover(parents[0]["prompt"], parents[1]["prompt"], rng)
            else:
                parent = max(archive, key=lambda x: x["mean_reward"])
                variant = _mutate(parent["prompt"], rng)
            if len(variant) > 600:
                variant = variant[:597] + "..."

            results = evaluate_prompt(
                prompt=variant, seeds=eval_subset, max_steps=max_steps,
                inference_url=inference_url, model=model, api_key=api_key,
                concurrency=concurrency,
            )
            v_mean = mean([r["reward"] for r in results])
            total_rollouts += len(results)
            iter_results.append({"variant": vi, "mean_reward": v_mean, "preview": variant[:100]})

            if v_mean > baseline_mean or len(archive) < 10:
                archive.append({"prompt": variant, "mean_reward": v_mean, "source": f"iter_{iteration}"})
            if v_mean > best_reward:
                best_reward = v_mean
                best_prompt = variant
                _print(f"[go-explore]   NEW BEST: {v_mean:.3f} (variant {vi})")
            else:
                _print(f"[go-explore]   variant {vi}: {v_mean:.3f}")

        iteration_log.append({
            "iteration": iteration, "phase": "search", "best_reward": best_reward,
            "rollouts": total_rollouts, "archive_size": len(archive), "results": iter_results,
        })

    # Phase 3: Held-out evaluation
    _print(f"\n[go-explore] Phase 3: Held-out ({len(heldout_seeds)} seeds)")
    _print("[go-explore] Evaluating baseline...")
    baseline_heldout = evaluate_prompt(
        prompt=BASELINE_PROMPT, seeds=heldout_seeds, max_steps=max_steps,
        inference_url=inference_url, model=model, api_key=api_key,
        concurrency=concurrency,
    )
    baseline_ho_mean = mean([r["reward"] for r in baseline_heldout])
    baseline_achievement_mean = mean([r.get("unique_achievements", 0.0) for r in baseline_heldout])

    _print("[go-explore] Evaluating best prompt...")
    best_heldout = evaluate_prompt(
        prompt=best_prompt, seeds=heldout_seeds, max_steps=max_steps,
        inference_url=inference_url, model=model, api_key=api_key,
        concurrency=concurrency,
    )
    best_ho_mean = mean([r["reward"] for r in best_heldout])
    best_achievement_mean = mean([r.get("unique_achievements", 0.0) for r in best_heldout])

    uplift = best_ho_mean - baseline_ho_mean
    rel_uplift = (uplift / baseline_ho_mean * 100) if baseline_ho_mean > 0 else 0.0
    achievement_uplift = best_achievement_mean - baseline_achievement_mean

    _print(f"\n[go-explore] RESULTS:")
    _print(f"  Baseline:  {baseline_ho_mean:.3f}")
    _print(f"  Best:      {best_ho_mean:.3f}")
    _print(f"  Uplift:    {uplift:+.3f} ({rel_uplift:+.1f}%)")
    _print(f"  Rollouts:  {total_rollouts}")

    archive.sort(key=lambda x: x["mean_reward"], reverse=True)

    result = {
        "experiment_id": experiment_id,
        "model": model,
        "max_steps_per_rollout": max_steps,
        "training_rollouts_used": total_rollouts,
        "iterations_completed": iteration,
        "archive_size": len(archive),
        "baseline": {
            "prompt": BASELINE_PROMPT,
            "training_mean_reward": baseline_mean,
            "heldout_mean_reward": baseline_ho_mean,
            "heldout_mean_achievements": baseline_achievement_mean,
            "heldout_per_seed": baseline_heldout,
        },
        "best": {
            "prompt": best_prompt,
            "training_best_reward": best_reward,
            "heldout_mean_reward": best_ho_mean,
            "heldout_mean_achievements": best_achievement_mean,
            "heldout_per_seed": best_heldout,
        },
        "uplift": uplift,
        "relative_uplift_percent": rel_uplift,
        "achievement_uplift": achievement_uplift,
        "archive_top_5": archive[:5],
        "iteration_log": iteration_log,
    }
    (output_dir / "go_explore_result.json").write_text(json.dumps(result, indent=2) + "\n")
    _print(f"[go-explore] Written to {output_dir / 'go_explore_result.json'}")
    return result


def parse_args() -> argparse.Namespace:
    default_base_url = (
        os.environ.get("SMR_METERED_INFERENCE_BASE_URL")
        or os.environ.get("NANOHORIZON_PROMPT_OPT_INFERENCE_BASE_URL")
        or "https://api.groq.com/openai/v1"
    )
    default_model = (
        os.environ.get("NANOHORIZON_PROMPT_OPT_MODEL")
        or os.environ.get("SMR_METERED_INFERENCE_MODEL")
        or "openai/gpt-oss-20b"
    )
    parser = argparse.ArgumentParser(description="Go-Explore prompt optimization for Craftax")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--max-training-rollouts", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--variants-per-iteration", type=int, default=DEFAULT_VARIANTS_PER_ITERATION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = (
        os.environ.get("SYNTH_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise SystemExit("SYNTH_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY required")

    output_dir = Path(args.output_dir).resolve()
    repo_root = output_dir.parent
    training_seeds, heldout_seeds = _load_seed_split(repo_root)
    started = time.time()
    result = run_go_explore(
        output_dir=output_dir,
        inference_url=args.base_url,
        model=args.model,
        api_key=api_key,
        max_training_rollouts=args.max_training_rollouts,
        training_seeds=training_seeds,
        heldout_seeds=heldout_seeds,
        max_steps=args.max_steps,
        concurrency=args.concurrency,
        variants_per_iteration=args.variants_per_iteration,
    )
    elapsed = time.time() - started

    summary = {
        "experiment_id": result["experiment_id"],
        "model": result["model"],
        "baseline_reward": result["baseline"]["heldout_mean_reward"],
        "best_reward": result["best"]["heldout_mean_reward"],
        "baseline_mean_achievements": result["baseline"]["heldout_mean_achievements"],
        "best_mean_achievements": result["best"]["heldout_mean_achievements"],
        "uplift": result["uplift"],
        "relative_uplift_percent": result["relative_uplift_percent"],
        "achievement_uplift": result["achievement_uplift"],
        "training_rollouts_used": result["training_rollouts_used"],
        "held_out_seeds": len(result["best"]["heldout_per_seed"]),
        "held_out_rollout_steps": result["max_steps_per_rollout"],
        "archive_size": result["archive_size"],
        "iterations_completed": result["iterations_completed"],
        "best_prompt_text": result["best"]["prompt"],
        "elapsed_seconds": round(elapsed, 1),
        "uplift_per_minute": round(
            result["uplift"] / max(elapsed / 60.0, 1e-9),
            6,
        ),
        "total_cost_usd": None,
        "estimated_inference_cost_usd": None,
        "termination_reason": "completed_search_budget",
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
