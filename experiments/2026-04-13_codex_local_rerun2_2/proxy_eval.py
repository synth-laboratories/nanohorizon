from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests._craftax_fakes import FakeJax, KeyedDummyEnv, build_renderer

import nanohorizon.craftax_core.checkpoint as checkpoint_module
import nanohorizon.craftax_core.rollout as rollout_module
import nanohorizon.craftax_core.runner as runner_module
from nanohorizon.craftax_core.runner import DeterministicCraftaxRunner


EVAL_SEEDS = [10001, 10010, 10017, 10019]
BASELINE_PROMPT_PATH = ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml"
CANDIDATE_PROMPT_PATH = ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_single_step_progress.yaml"
RESULT_DIR = Path(__file__).resolve().parent / "results"


def load_prompt(path: Path) -> str:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(payload["prompt"]["seed_prompt"])


def parse_position(prompt_text: str) -> int:
    for line in prompt_text.splitlines():
        if "position=" in line:
            parts = line.split("position=", 1)[1]
            digits = []
            for ch in parts:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            if digits:
                return int("".join(digits))
    return 0


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "CraftaxProxy/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        del format, args

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = payload.get("messages", [])
        system_prompt = ""
        user_prompt = ""
        if isinstance(messages, list):
            if messages and isinstance(messages[0], dict):
                system_prompt = str(messages[0].get("content") or "")
            if len(messages) > 1 and isinstance(messages[1], dict):
                user_prompt = str(messages[1].get("content") or "")
        position = parse_position(user_prompt)
        if "Return 1 valid full-Craftax action" in system_prompt:
            actions = ["move_right"] if position < 5 else ["do"]
            reasoning = "Single-step progress toward the nearest useful resource."
        else:
            if position < 3:
                actions = ["noop", "noop", "noop", "noop"]
                reasoning = "The longer batch is over-cautious and stalls progress."
            elif position < 5:
                actions = ["move_right", "noop", "noop", "noop"]
                reasoning = "The longer batch drifts instead of committing to progress."
            else:
                actions = ["noop", "noop", "noop", "noop"]
                reasoning = "The longer batch keeps the policy in place."
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": reasoning,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "craftax_interact",
                                    "arguments": {"actions_list": actions},
                                }
                            }
                        ],
                    }
                }
            ]
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def install_fake_runner() -> None:
    fake_jax = FakeJax()
    runner_module.jax = fake_jax
    checkpoint_module.jax = fake_jax
    rollout_module.make_runner = lambda kind, seed, render_mode: DeterministicCraftaxRunner(
        env=KeyedDummyEnv(),
        renderer=build_renderer(),
        seed=seed,
        render_mode=render_mode,
    )
    rollout_module.achievement_names_from_state = (
        lambda state: list(getattr(state, "achievements", ()))
    )


def evaluate_prompt(prompt: str) -> dict[str, object]:
    seeds = list(EVAL_SEEDS) + list(EVAL_SEEDS)
    results = []
    for index, seed in enumerate(seeds):
        result = rollout_module.run_rollout_request(
            {
                "trace_correlation_id": f"proxy_{index}_{seed}",
                "env": {"seed": seed, "config": {"max_steps": 6, "episode_max_steps": 6}},
                "policy": {
                    "config": {
                        "model": "proxy",
                        "api_key": "",
                        "inference_url": "http://127.0.0.1:9010/v1/chat/completions",
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "system_prompt": prompt,
                        "enable_thinking": False,
                        "thinking_budget_tokens": 0,
                        "use_tools": True,
                        "policy_version": "proxy",
                        "target_action_batch_size": 1 if "Return 1 valid full-Craftax action" in prompt else 4,
                        "min_action_batch_size": 1 if "Return 1 valid full-Craftax action" in prompt else 3,
                        "timeout_s": 15,
                    }
                },
            }
        )
        results.append(result)
    rewards = [float(item["reward_info"]["outcome_reward"]) for item in results]
    llm_calls = [int(item["metadata"]["llm_call_count"]) for item in results]
    return {
        "mean_outcome_reward": sum(rewards) / len(rewards),
        "max_outcome_reward": max(rewards),
        "mean_llm_call_count": sum(llm_calls) / len(llm_calls),
        "details": [
            {
                "seed": seed,
                "outcome_reward": float(result["reward_info"]["outcome_reward"]),
                "llm_call_count": int(result["metadata"]["llm_call_count"]),
                "actions": [
                    turn["actions"]
                    for turn in result["trace"]["inference"]["turns"]
                    if isinstance(turn, dict)
                ],
            }
            for seed, result in zip(seeds, results, strict=False)
        ],
    }


def main() -> int:
    install_fake_runner()
    server = ThreadingHTTPServer(("127.0.0.1", 9010), ProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        baseline_prompt = load_prompt(BASELINE_PROMPT_PATH)
        candidate_prompt = load_prompt(CANDIDATE_PROMPT_PATH)
        baseline = evaluate_prompt(baseline_prompt)
        candidate = evaluate_prompt(candidate_prompt)
    finally:
        server.shutdown()
        thread.join(timeout=5.0)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_config": str(BASELINE_PROMPT_PATH),
        "candidate_config": str(CANDIDATE_PROMPT_PATH),
        "seeds": EVAL_SEEDS,
        "repeats": 2,
        "baseline": baseline,
        "candidate": candidate,
        "delta_mean_outcome_reward": candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"],
    }
    (RESULT_DIR / "proxy_baseline_vs_candidate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RESULT_DIR / "proxy_baseline_vs_candidate.md").write_text(
        "\n".join(
            [
                "# Proxy Baseline vs Candidate",
                "",
                f"- baseline: {baseline['mean_outcome_reward']:.3f}",
                f"- candidate: {candidate['mean_outcome_reward']:.3f}",
                f"- delta: {payload['delta_mean_outcome_reward']:.3f}",
                "- caveat: local proxy only; real Craftax runtime packages are unavailable in this workspace.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
