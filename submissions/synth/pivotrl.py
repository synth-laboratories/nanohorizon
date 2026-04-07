from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import modal

REMOTE_ROOT = Path("/root/nanohorizon")
REMOTE_SRC = REMOTE_ROOT / "src"
_THIS_FILE = Path(__file__).resolve()
if len(_THIS_FILE.parents) >= 3:
    LOCAL_ROOT = _THIS_FILE.parents[2]
elif REMOTE_ROOT.exists():
    LOCAL_ROOT = REMOTE_ROOT
else:
    LOCAL_ROOT = Path.cwd().resolve()
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))
LOCAL_SRC = LOCAL_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))
if REMOTE_ROOT.exists():
    sys.path.insert(0, str(REMOTE_ROOT))
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.shared.common import ensure_dir, write_json
from nanohorizon.shared.modal_common import ARTIFACT_DIR, GPU_OFFLINE, PROJECT_ROOT, REMOTE_ROOT as SHARED_REMOTE_ROOT, offline_worker_image, volume_mounts
from submissions.synth import pivotrl_core

APP_NAME = os.getenv("NANOHORIZON_MODAL_PIVOTRL_APP_NAME", "nanohorizon-craftax-pivotrl").strip() or "nanohorizon-craftax-pivotrl"
image = offline_worker_image().add_local_dir(
    (PROJECT_ROOT / "submissions").as_posix(),
    remote_path=f"{SHARED_REMOTE_ROOT}/submissions",
    copy=True,
)
app = modal.App(APP_NAME)


def _default_output_dir() -> str:
    return pivotrl_core.default_modal_output_dir()


def _build_args_namespace(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


@app.function(
    image=image,
    gpu=GPU_OFFLINE,
    timeout=60 * 60 * 4,
    volumes=volume_mounts(),
)
def run(
    *,
    output_dir: str = "",
    base_model: str = pivotrl_core.DEFAULT_BASE_MODEL,
    teacher_model: str = pivotrl_core.DEFAULT_TEACHER_MODEL,
    teacher_inference_url: str = "",
    teacher_api_key: str = "",
    craftax_container_url: str = "",
    craftax_container_worker_token: str = "",
    bootstrap_rollouts_path: str = "",
    max_length: int = 8192,
    request_timeout_seconds: float = 300.0,
    bootstrap_seed_start: int = 0,
    bootstrap_seed_count: int = 32,
    bootstrap_max_steps: int = 48,
    bootstrap_temperature: float = 0.2,
    bootstrap_max_new_tokens: int = 3072,
    bootstrap_thinking_budget_tokens: int = 2000,
    bootstrap_rollout_concurrency: int = 4,
    bootstrap_rollout_semaphore_limit: int = 4,
    bootstrap_target_action_batch_size: int = 4,
    bootstrap_min_action_batch_size: int = 3,
    enable_thinking: bool = True,
    lookback: int = 1,
    profile_k: int = 4,
    lambda_diff: float = 0.75,
    profile_temperature: float = 0.8,
    profile_top_p: float = 0.95,
    profile_max_new_tokens: int = 96,
    max_pivots: int = 128,
    min_kept_pivots: int = 4,
    group_size: int = 4,
    train_iterations: int = 2,
    pivots_per_iteration: int = 16,
    max_train_steps: int = 16,
    train_steps_per_iteration: int = 8,
    sample_max_new_tokens: int = 96,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.02,
) -> dict[str, object]:
    os.chdir(str(REMOTE_ROOT))
    destination = ensure_dir(output_dir or _default_output_dir())
    args = _build_args_namespace(
        output_root=str(destination),
        base_model=base_model,
        teacher_model=teacher_model,
        teacher_inference_url=teacher_inference_url,
        teacher_api_key=teacher_api_key,
        container_url=craftax_container_url,
        container_worker_token=craftax_container_worker_token,
        bootstrap_rollouts_path=bootstrap_rollouts_path,
        max_length=max_length,
        request_timeout_seconds=request_timeout_seconds,
        bootstrap_seed_start=bootstrap_seed_start,
        bootstrap_seed_count=bootstrap_seed_count,
        bootstrap_max_steps=bootstrap_max_steps,
        bootstrap_temperature=bootstrap_temperature,
        bootstrap_max_new_tokens=bootstrap_max_new_tokens,
        bootstrap_thinking_budget_tokens=bootstrap_thinking_budget_tokens,
        bootstrap_rollout_concurrency=bootstrap_rollout_concurrency,
        bootstrap_rollout_semaphore_limit=bootstrap_rollout_semaphore_limit,
        bootstrap_target_action_batch_size=bootstrap_target_action_batch_size,
        bootstrap_min_action_batch_size=bootstrap_min_action_batch_size,
        enable_thinking=enable_thinking,
        lookback=lookback,
        profile_k=profile_k,
        lambda_diff=lambda_diff,
        profile_temperature=profile_temperature,
        profile_top_p=profile_top_p,
        profile_max_new_tokens=profile_max_new_tokens,
        max_pivots=max_pivots,
        min_kept_pivots=min_kept_pivots,
        group_size=group_size,
        train_iterations=train_iterations,
        pivots_per_iteration=pivots_per_iteration,
        max_train_steps=max_train_steps,
        train_steps_per_iteration=train_steps_per_iteration,
        sample_max_new_tokens=sample_max_new_tokens,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        clip_epsilon=clip_epsilon,
        kl_coef=kl_coef,
    )
    if craftax_container_url:
        return pivotrl_core.run_modal_train_pipeline(args, output_root=destination)
    logs_dir = ensure_dir(destination / "logs")
    with pivotrl_core.local_craftax_runtime(log_path=logs_dir / "craftax_runtime.log") as local_container_url:
        setattr(args, "container_url", local_container_url)
        return pivotrl_core.run_modal_train_pipeline(args, output_root=destination)


@app.local_entrypoint()
def main(
    output_dir: str = "",
    base_model: str = pivotrl_core.DEFAULT_BASE_MODEL,
    teacher_model: str = pivotrl_core.DEFAULT_TEACHER_MODEL,
    teacher_inference_url: str = "",
    teacher_api_key: str = "",
    craftax_container_url: str = "",
    craftax_container_worker_token: str = "",
    bootstrap_rollouts_path: str = "",
    max_length: int = 8192,
    request_timeout_seconds: float = 300.0,
    bootstrap_seed_start: int = 0,
    bootstrap_seed_count: int = 32,
    bootstrap_max_steps: int = 48,
    bootstrap_temperature: float = 0.2,
    bootstrap_max_new_tokens: int = 3072,
    bootstrap_thinking_budget_tokens: int = 2000,
    bootstrap_rollout_concurrency: int = 4,
    bootstrap_rollout_semaphore_limit: int = 4,
    bootstrap_target_action_batch_size: int = 4,
    bootstrap_min_action_batch_size: int = 3,
    enable_thinking: bool = True,
    lookback: int = 1,
    profile_k: int = 4,
    lambda_diff: float = 0.75,
    profile_temperature: float = 0.8,
    profile_top_p: float = 0.95,
    profile_max_new_tokens: int = 96,
    max_pivots: int = 128,
    min_kept_pivots: int = 4,
    group_size: int = 4,
    train_iterations: int = 2,
    pivots_per_iteration: int = 16,
    max_train_steps: int = 16,
    train_steps_per_iteration: int = 8,
    sample_max_new_tokens: int = 96,
    sample_temperature: float = 0.8,
    sample_top_p: float = 0.95,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.02,
    local_result_path: str = "",
) -> None:
    result = run.remote(
        output_dir=output_dir,
        base_model=base_model,
        teacher_model=teacher_model,
        teacher_inference_url=teacher_inference_url,
        teacher_api_key=teacher_api_key,
        craftax_container_url=craftax_container_url,
        craftax_container_worker_token=craftax_container_worker_token,
        bootstrap_rollouts_path=bootstrap_rollouts_path,
        max_length=max_length,
        request_timeout_seconds=request_timeout_seconds,
        bootstrap_seed_start=bootstrap_seed_start,
        bootstrap_seed_count=bootstrap_seed_count,
        bootstrap_max_steps=bootstrap_max_steps,
        bootstrap_temperature=bootstrap_temperature,
        bootstrap_max_new_tokens=bootstrap_max_new_tokens,
        bootstrap_thinking_budget_tokens=bootstrap_thinking_budget_tokens,
        bootstrap_rollout_concurrency=bootstrap_rollout_concurrency,
        bootstrap_rollout_semaphore_limit=bootstrap_rollout_semaphore_limit,
        bootstrap_target_action_batch_size=bootstrap_target_action_batch_size,
        bootstrap_min_action_batch_size=bootstrap_min_action_batch_size,
        enable_thinking=enable_thinking,
        lookback=lookback,
        profile_k=profile_k,
        lambda_diff=lambda_diff,
        profile_temperature=profile_temperature,
        profile_top_p=profile_top_p,
        profile_max_new_tokens=profile_max_new_tokens,
        max_pivots=max_pivots,
        min_kept_pivots=min_kept_pivots,
        group_size=group_size,
        train_iterations=train_iterations,
        pivots_per_iteration=pivots_per_iteration,
        max_train_steps=max_train_steps,
        train_steps_per_iteration=train_steps_per_iteration,
        sample_max_new_tokens=sample_max_new_tokens,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        clip_epsilon=clip_epsilon,
        kl_coef=kl_coef,
    )
    if str(local_result_path or "").strip():
        write_json(local_result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
