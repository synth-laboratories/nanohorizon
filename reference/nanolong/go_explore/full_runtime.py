from __future__ import annotations

from typing import Any

import requests

from .full_models import CheckpointSummary, RolloutSummary, WaypointSpec


class FullCrafterRuntime:
    def __init__(
        self,
        *,
        container_url: str,
        inference_url: str,
        policy_model: str,
        api_key: str,
    ) -> None:
        self.container_url = container_url.rstrip("/")
        self.inference_url = inference_url
        self.policy_model = policy_model
        self.api_key = api_key

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.container_url}{path}", json=payload, timeout=240)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"Expected object JSON from {path}")
        return body

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self.container_url}{path}", timeout=30)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"Expected object JSON from {path}")
        return body

    def healthcheck(self) -> bool:
        payload = self._get("/health")
        return payload.get("status") == "ok"

    def task_info(self) -> dict[str, Any]:
        return self._get("/task_info")

    def rollout_from_seed(
        self,
        *,
        rollout_id: str,
        prompt_text: str,
        seed: int,
        segment_steps: int,
        planner_mode: str = "direct",
        waypoints: list[WaypointSpec] | None = None,
    ) -> RolloutSummary:
        payload: dict[str, Any] = {
            "trace_correlation_id": rollout_id,
            "planner_mode": planner_mode,
            "env": {
                "seed": int(seed),
                "config": {
                    "max_steps": int(segment_steps),
                    "episode_max_steps": int(segment_steps),
                    "segment_steps": int(segment_steps),
                },
            },
            "policy": {
                "config": {
                    "model": self.policy_model,
                    "api_key": self.api_key,
                    "inference_url": self.inference_url,
                    "temperature": 0.0,
                    "max_tokens": 128,
                    "system_prompt": prompt_text,
                }
            },
        }
        if waypoints:
            payload["waypoint_plan_request"] = {
                "waypoints": [item.to_container_payload() for item in waypoints]
            }
        response = self._post("/rollout", payload)
        return self._parse_rollout(
            response,
            seed=seed,
            source_kind="fresh",
            planner_mode=planner_mode,
            prompt_text=prompt_text,
        )

    def resume_from_checkpoint(
        self,
        *,
        parent_rollout_id: str,
        checkpoint_id: str,
        target_rollout_id: str,
        prompt_text: str,
        seed: int,
        segment_steps: int,
        planner_mode: str = "waypoint_planned",
        waypoints: list[WaypointSpec] | None = None,
    ) -> RolloutSummary:
        policy_config: dict[str, Any] = {
            "model": self.policy_model,
            "api_key": self.api_key,
            "inference_url": self.inference_url,
            "temperature": 0.0,
            "max_tokens": 128,
            "system_prompt": prompt_text,
        }
        if waypoints:
            policy_config["policy_version"] = "full_go_explore_waypoint"
        payload: dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            "target_rollout_id": target_rollout_id,
            "mode": "new_rollout",
            "overrides": {
                "planner_mode": planner_mode,
                "segment_steps": int(segment_steps),
                "policy_config": policy_config,
            },
        }
        payload["overrides"]["env_config"] = {"segment_steps": int(segment_steps)}
        if waypoints:
            payload["overrides"]["policy_config"]["planner_hint"] = [item.description for item in waypoints]
            payload["overrides"]["waypoint_plan_request"] = {
                "waypoints": [item.to_container_payload() for item in waypoints]
            }
        response = self._post(f"/rollouts/{parent_rollout_id}/resume", payload)
        rollout = self._parse_rollout(
            response,
            seed=seed,
            source_kind="resumed",
            planner_mode=planner_mode,
            prompt_text=prompt_text,
        )
        rollout.parent_rollout_id = parent_rollout_id
        rollout.parent_checkpoint_id = checkpoint_id
        rollout.target_waypoints = [item.description for item in (waypoints or [])]
        return rollout

    def create_checkpoint(
        self,
        *,
        rollout_id: str,
        seed: int,
        label: str,
        metadata: dict[str, Any] | None = None,
        source_kind: str,
    ) -> CheckpointSummary:
        payload = {
            "label": label,
            "source": "go_explore",
            "metadata": dict(metadata or {}),
        }
        response = self._post(f"/rollouts/{rollout_id}/checkpoints", payload)
        metadata_payload = dict(response.get("metadata") or {})
        return CheckpointSummary(
            checkpoint_id=str(response["checkpoint_id"]),
            rollout_id=str(response["rollout_id"]),
            seed=int(metadata_payload.get("seed") or seed),
            step_index=int(response.get("step_index") or metadata_payload.get("step") or 0),
            total_reward=float(
                response.get("total_reward")
                or metadata_payload.get("total_reward")
                or 0.0
            ),
            achievements=list(metadata_payload.get("achievements") or []),
            inventory=dict(metadata_payload.get("inventory") or {}),
            player_pos=list(metadata_payload.get("player_pos") or []) or None,
            source_kind=source_kind,
            raw_payload=response,
        )

    def _parse_rollout(
        self,
        response: dict[str, Any],
        *,
        seed: int,
        source_kind: str,
        planner_mode: str,
        prompt_text: str,
    ) -> RolloutSummary:
        details = dict(response.get("reward_info", {}).get("details") or {})
        rollout_id = str(response.get("rollout_id") or response.get("trace_correlation_id") or "")
        return RolloutSummary(
            rollout_id=rollout_id,
            seed=seed,
            source_kind=source_kind,
            planner_mode=str(details.get("planner_mode") or planner_mode),
            prompt_text=prompt_text,
            reward=float(response.get("reward_info", {}).get("outcome_reward") or 0.0),
            step_count=int(details.get("step_count") or 0),
            success_status=str(response.get("success_status") or "unknown"),
            status=str(response.get("status") or response.get("success_status") or "completed"),
            achievements=list(details.get("achievements") or []),
            completed_waypoints=list(details.get("completed_waypoints") or []),
            checkpoint_count=int(details.get("checkpoint_count") or 0),
            llm_call_count=int(details.get("llm_call_count") or 0),
            raw_response=response,
        )
