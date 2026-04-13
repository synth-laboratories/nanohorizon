from __future__ import annotations

import unittest
from pathlib import Path

from workspace.actorbench_parent_lane import run_parent_action


class ActorBenchParentLaneTests(unittest.TestCase):
    def test_run_parent_action_injects_repo_root_env(self):
        captured: dict[str, object] = {}

        class FakeCompletedProcess:
            returncode = 0

        def fake_run(command, cwd=None, env=None, check=None):  # type: ignore[no-untyped-def]
            captured["command"] = command
            captured["cwd"] = cwd
            captured["env"] = env or {}
            captured["check"] = check
            return FakeCompletedProcess()

        import workspace.actorbench_parent_lane as lane

        original_run = lane.subprocess.run
        lane.subprocess.run = fake_run
        try:
            wrapper_path = Path("/workspace/workspace/run_actorbench_nh_craftax_hello_world_worker_v1.py")
            output_root = Path("/workspace") / "tmp" / "slot_a"
            output_root.mkdir(parents=True, exist_ok=True)

            rc = run_parent_action(
                {
                    "task_id": "actorbench_nh__craftax_hello_world__worker_v1",
                    "parent_task_id": "nanohorizon_craftax_hello_world",
                    "host_parent_runner": "run_nanohorizon_craftax_hello_world_task.py",
                    "bundled_parent_runner": "parent_runner.py",
                },
                wrapper_path,
                "run",
                output_root,
            )
        finally:
            lane.subprocess.run = original_run

        self.assertEqual(rc, 0)
        self.assertEqual(captured["cwd"], str(output_root))
        self.assertEqual(captured["env"]["NANOHORIZON_REPO_ROOT"], "/workspace")
        self.assertIn("python", Path(captured["command"][0]).name)
