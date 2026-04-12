"""Craftax core helpers for interface-shaped prompts."""

from .checkpoint import Checkpoint, CheckpointCodec, state_digest
from .metadata import (
    DEFAULT_ACHIEVEMENT_NAMES,
    DEFAULT_ACTION_NAMES,
    OBSERVATION_FIELD_ORDER,
    PRIMARY_TOOL_NAME,
    PromptContext,
    RewardHistoryEntry,
    RewardHistoryWindow,
    StructuredObservation,
)
from .http_shim import build_prompt_context, create_app, render_prompt_turn, summarize_history
from .modalities import CallableRenderer, RenderBundle, RenderMode
from .rollout import collect_rollouts, run_rollout, run_rollout_request
from .runner import DeterministicCraftaxRunner, StepOutput
from .texture_cache import ensure_texture_cache

summarize_reward_history = summarize_history

__all__ = [
    "DEFAULT_ACHIEVEMENT_NAMES",
    "DEFAULT_ACTION_NAMES",
    "OBSERVATION_FIELD_ORDER",
    "PRIMARY_TOOL_NAME",
    "Checkpoint",
    "CheckpointCodec",
    "CallableRenderer",
    "DeterministicCraftaxRunner",
    "PromptContext",
    "RenderBundle",
    "RenderMode",
    "RewardHistoryEntry",
    "RewardHistoryWindow",
    "StepOutput",
    "StructuredObservation",
    "build_prompt_context",
    "create_app",
    "collect_rollouts",
    "ensure_texture_cache",
    "render_prompt_turn",
    "run_rollout",
    "run_rollout_request",
    "state_digest",
    "summarize_history",
    "summarize_reward_history",
]
