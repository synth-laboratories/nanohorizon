from __future__ import annotations

import json
import socketserver
import subprocess
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

import pytest

from submissions.synth import pivotrl_core
from nanohorizon.shared import craftax_data


def _successful_rollout(seed: int = 0) -> dict[str, object]:
    return {
        "success_status": "success",
        "reward_info": {"outcome_reward": 1.0},
        "trace": {"inference": {"turns": []}},
        "_request_seed": seed,
        "trace_correlation_id": f"trace-{seed}",
        "metadata": {"achievements": ["collect_wood"]},
    }


def _sparse_successful_rollout(seed: int = 0) -> dict[str, object]:
    turns = [
        {
            "turn_index": index,
            "actions": ["noop"],
            "assistant_text": "",
            "invalid_parse": True,
            "trainable": False,
            "prompt_messages": [
                {
                    "role": "system",
                    "content": "system",
                },
                {
                    "role": "user",
                    "content": (
                        "Current Craftax long-horizon observation:\n"
                        "Craftax state summary\n"
                        "achievements: none\n"
                        f"player_position: [{24 + index} {24 + index}]"
                    ),
                },
            ],
        }
        for index in range(3)
    ]
    return {
        "success_status": "success",
        "reward_info": {"outcome_reward": 0.0, "details": {"achievements": []}},
        "trace": {"inference": {"turns": turns}},
        "_request_seed": seed,
        "trace_correlation_id": f"trace-sparse-{seed}",
        "metadata": {"achievements": [], "seed": seed},
    }


def test_modal_train_pipeline_requires_teacher_url_for_live_bootstrap(tmp_path):
    args = SimpleNamespace(
        bootstrap_rollouts_path="",
        teacher_inference_url="",
        teacher_api_key="",
        container_url="http://127.0.0.1:8903",
    )
    with pytest.raises(RuntimeError, match="teacher_inference_url"):
        pivotrl_core.run_modal_train_pipeline(args, output_root=tmp_path)


def test_build_local_craftax_env_is_cpu_only():
    env = pivotrl_core.build_local_craftax_env()
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["JAX_PLATFORM_NAME"] == "cpu"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.0"


def test_build_container_probe_headers_supports_synthtunnel():
    headers = pivotrl_core.build_container_probe_headers(
        container_url="https://infra-api.usesynth.ai/s/rt_example",
        container_worker_token="worker-token",
    )
    assert headers["Authorization"] == "Bearer worker-token"


def test_craftax_container_headers_force_connection_close_for_tunnel_edges():
    headers = craftax_data._container_headers(
        container_url="https://infra-api.usesynth.ai/s/rt_example",
        container_worker_token="worker-token",
        environment_api_key="",
    )
    assert headers["Authorization"] == "Bearer worker-token"
    assert headers["Connection"] == "close"


def test_modal_train_pipeline_writes_result_contract_from_saved_bootstrap(monkeypatch, tmp_path):
    bootstrap_path = tmp_path / "bootstrap_successful_rollouts.jsonl"
    bootstrap_path.write_text(json.dumps(_successful_rollout()) + "\n", encoding="utf-8")

    dataset_path = tmp_path / "artifacts" / "pivot_dataset.jsonl"
    adapter_dir = tmp_path / "artifacts" / "pivotrl_adapter"

    def fake_run_build_dataset(args, *, output_root, bootstrap_rollouts_path):  # type: ignore[no-untyped-def]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(json.dumps({"pivot_id": "pivot-1"}) + "\n", encoding="utf-8")
        return dataset_path, {"candidate_count": 1}, {"profiled_count": 1, "kept_count": 1, "dropped_count": 0}

    def fake_run_train(args, *, output_root, dataset_path):  # type: ignore[no-untyped-def]
        adapter_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "iterations_completed": 1,
            "optimizer_steps_total": 1,
            "adapter_dir": str(adapter_dir),
        }
        (output_root / "artifacts" / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return adapter_dir, summary

    monkeypatch.setattr(pivotrl_core, "run_build_dataset", fake_run_build_dataset)
    monkeypatch.setattr(pivotrl_core, "run_train", fake_run_train)

    args = SimpleNamespace(bootstrap_rollouts_path=str(bootstrap_path))
    result = pivotrl_core.run_modal_train_pipeline(args, output_root=tmp_path)

    modal_result_path = tmp_path / "artifacts" / "modal_train_result.json"
    assert modal_result_path.exists()
    payload = json.loads(modal_result_path.read_text(encoding="utf-8"))
    assert payload["adapter_dir"] == str(adapter_dir)
    assert payload["artifacts"]["bootstrap_rollouts"].endswith("bootstrap_rollouts.jsonl")
    assert payload["artifacts"]["bootstrap_successful_rollouts"].endswith("bootstrap_successful_rollouts.jsonl")
    assert payload["artifacts"]["pivot_dataset"].endswith("pivot_dataset.jsonl")
    assert payload["artifacts"]["training_summary"].endswith("training_summary.json")
    assert result["output_root"] == str(tmp_path)


def test_bootstrap_and_preflight_disable_rollout_logprobs(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []

    async def fake_collect_rollouts_concurrently_with_summary(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(dict(kwargs))
        return ([_successful_rollout()], {"requested_rollouts": 1, "completed_rollouts": 1})

    monkeypatch.setattr(
        pivotrl_core,
        "collect_rollouts_concurrently_with_summary",
        fake_collect_rollouts_concurrently_with_summary,
    )
    monkeypatch.setattr(
        pivotrl_core,
        "wait_for_http_health",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        pivotrl_core,
        "_wait_for_task_info",
        lambda *args, **kwargs: {"task": "craftax"},
    )
    monkeypatch.setattr(
        pivotrl_core,
        "_probe_inference_tool_call",
        lambda *args, **kwargs: {"status": "ok"},
    )

    args = SimpleNamespace(
        container_url="http://127.0.0.1:8903",
        container_worker_token="",
        teacher_inference_url="http://127.0.0.1:8000/v1",
        teacher_api_key="",
        teacher_model="Qwen/Qwen3.5-9B",
        bootstrap_seed_start=0,
        bootstrap_seed_count=1,
        bootstrap_max_steps=1,
        bootstrap_temperature=0.0,
        bootstrap_max_new_tokens=64,
        enable_thinking=False,
        bootstrap_thinking_budget_tokens=0,
        bootstrap_target_action_batch_size=1,
        bootstrap_min_action_batch_size=1,
        bootstrap_rollout_concurrency=1,
        bootstrap_rollout_semaphore_limit=1,
        request_timeout_seconds=5.0,
    )

    pivotrl_core.run_bootstrap(args, output_root=tmp_path)

    assert len(calls) == 2
    assert calls[0]["request_logprobs"] is False
    assert calls[1]["request_logprobs"] is False


def test_sparse_successful_rollouts_get_default_bootstrap_fallback_candidate():
    candidates, summary = pivotrl_core.build_candidate_pivots(
        rollouts=[_sparse_successful_rollout()],
        lookback=1,
    )
    assert len(candidates) == 1
    assert candidates[0]["target_achievement"] == "collect_wood"
    assert summary["fallback_count"] == 1
    assert summary["candidate_count"] == 1


def test_profile_pivots_backfills_near_misses_to_minimum(monkeypatch):
    pivots = [
        {
            "pivot_id": f"pivot-{index}",
            "state_text": f"state-{index}",
            "pre_achievement_inventory_or_summary": {"rank": index},
            "training_messages": [{"role": "user", "content": f"pivot-{index}"}],
            "rubric": {"target_achievement": "collect_wood"},
        }
        for index in range(3)
    ]

    monkeypatch.setattr(pivotrl_core, "initialize_reference_policy", lambda base_model: (object(), object()))
    monkeypatch.setattr(pivotrl_core, "sample_completion_text", lambda **kwargs: "noop")
    monkeypatch.setattr(pivotrl_core, "extract_actions_from_text", lambda text: ["noop"])
    monkeypatch.setattr(
        pivotrl_core,
        "score_actions_against_rubric",
        lambda *, inventory, **kwargs: (float(inventory["rank"]), "ranked reward"),
    )
    monkeypatch.setattr(pivotrl_core, "release_cuda_memory", lambda: None)

    kept, summary = pivotrl_core.profile_pivots(
        pivots=pivots,
        base_model="Qwen/Qwen3.5-4B",
        profile_k=1,
        lambda_diff=0.75,
        max_new_tokens=32,
        temperature=0.8,
        top_p=0.95,
        max_pivots=3,
        min_kept_pivots=2,
    )

    assert len(kept) == 2
    assert summary["kept_count"] == 2
    assert summary["backfill_kept_count"] == 2
    assert summary["min_kept_pivots"] == 2
    assert [item["pivot_id"] for item in kept] == ["pivot-0", "pivot-1"]
    assert all(item["profile_stats"].get("backfill_kept") for item in kept)


def test_pivotrl_worker_sources_ban_old_runtime_patterns():
    repo_root = Path(__file__).resolve().parents[1]
    for target in (
        repo_root / "submissions" / "synth" / "pivotrl.py",
        repo_root / "submissions" / "synth" / "pivotrl_core.py",
    ):
        text = target.read_text(encoding="utf-8")
        assert "modal.forward" not in text
        assert "local_vllm_server" not in text
        assert "shared_local_vllm_server" not in text
        assert "@modal.web_server" not in text


def test_pivotrl_script_uses_craftax_tunnel_for_remote_bootstrap():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_craftax_pivotrl_qwen35_4b_1xa100_20min.sh"
    text = script_path.read_text(encoding="utf-8")
    assert 'nanohorizon_open_craftax_tunnel_if_needed "$ROOT"' in text
    assert 'BOOTSTRAP_CONTAINER_URL="${BOOTSTRAP_CONTAINER_URL_OVERRIDE:-${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}}"' in text
    assert '="$(nanohorizon_start_modal_endpoint' not in text
    assert '--bootstrap-max-new-tokens "$BOOTSTRAP_MAX_NEW_TOKENS"' in text
    assert '--bootstrap-thinking-budget-tokens "$BOOTSTRAP_THINKING_BUDGET_TOKENS"' in text
    assert '--profile-max-new-tokens "$PROFILE_MAX_NEW_TOKENS"' in text
    assert '--min-kept-pivots "$MIN_KEPT_PIVOTS"' in text
    assert '--sample-max-new-tokens "$SAMPLE_MAX_NEW_TOKENS"' in text
    assert 'wait_for_inference_chat_url "$NANOHORIZON_TEACHER_INFERENCE_URL" "$NANOHORIZON_TEACHER_API_KEY"' in text


def test_shell_helper_extracts_modal_endpoint_url(tmp_path):
    log_path = tmp_path / "teacher.stdout.log"
    log_path.write_text("hello\nvLLM endpoint: https://abc.modal.run\n", encoding="utf-8")
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lib_craftax_tunnel.sh"
    command = f"source {script_path}; nanohorizon_extract_modal_endpoint_url {log_path}"
    completed = subprocess.run(["bash", "-lc", command], check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "https://abc.modal.run"


def test_shell_helper_waits_for_openai_compatible_endpoint(tmp_path):
    del tmp_path
    token = "test-key"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "Qwen/Qwen3.5-9B"}]}).encode("utf-8"))

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": "chatcmpl-test", "choices": []}).encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            del format, args

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "lib_craftax_tunnel.sh"
        command = (
            f"source {script_path}; "
            f"nanohorizon_wait_for_openai_compat_endpoint http://127.0.0.1:{port} {token} 3 1"
        )
        completed = subprocess.run(["bash", "-lc", command], check=False, capture_output=True, text=True)
        server.shutdown()
        thread.join(timeout=5)
    assert completed.returncode == 0, completed.stderr


def test_shell_helper_accepts_base_url_that_already_ends_with_v1(tmp_path):
    del tmp_path
    token = "test-key"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "Qwen/Qwen3.5-9B"}]}).encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            del format, args

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "lib_craftax_tunnel.sh"
        command = (
            f"source {script_path}; "
            f"nanohorizon_wait_for_openai_compat_endpoint http://127.0.0.1:{port}/v1 {token} 3 1 1"
        )
        completed = subprocess.run(["bash", "-lc", command], check=False, capture_output=True, text=True)
        server.shutdown()
        thread.join(timeout=5)
    assert completed.returncode == 0, completed.stderr
