from __future__ import annotations

from pathlib import Path

from nanohorizon.baselines.cpt import (
    build_export_command,
    resolve_latest_megatron_checkpoint_dir,
    truncate_rows_to_token_budget,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids)


def test_truncate_rows_to_token_budget_stops_at_exact_budget() -> None:
    rows = [
        {"text": "abcd"},
        {"text": "efgh"},
        {"text": "ijkl"},
    ]

    prepared, summary = truncate_rows_to_token_budget(
        rows,
        text_field="text",
        token_budget=10,
        tokenizer=FakeTokenizer(),
    )

    assert [row.text for row in prepared] == ["abcd", "efgh", "ij"]
    assert summary.total_tokens == 10
    assert summary.rows_written == 3
    assert summary.truncated_rows == 1
    assert summary.stopped_early is True


def test_truncate_rows_to_token_budget_skips_empty_rows() -> None:
    rows = [
        {"text": ""},
        {"text": "abc"},
        {"other": "ignored"},
        "xyz",
    ]

    prepared, summary = truncate_rows_to_token_budget(
        rows,
        text_field="text",
        token_budget=6,
        tokenizer=FakeTokenizer(),
    )

    assert [row.text for row in prepared] == ["abc", "xyz"]
    assert summary.total_tokens == 6
    assert summary.rows_written == 2
    assert summary.source_rows_used == 2
    assert summary.truncated_rows == 0


def test_build_export_command_has_expected_shape() -> None:
    command = build_export_command(
        conversion_script_path="/tmp/convert_checkpoints.py",
        hf_model="Qwen/Qwen3.5-0.8B",
        megatron_path="/tmp/megatron/iter_0000010",
        hf_output_path="/tmp/hf_export",
    )

    assert command == [
        command[0],
        "/private/tmp/convert_checkpoints.py",
        "export",
        "--hf-model",
        "Qwen/Qwen3.5-0.8B",
        "--megatron-path",
        "/private/tmp/megatron/iter_0000010",
        "--hf-path",
        "/private/tmp/hf_export",
    ]


def test_resolve_latest_megatron_checkpoint_dir_prefers_latest_iter(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "iter_0000005").mkdir(parents=True)
    (checkpoints / "iter_0000010").mkdir(parents=True)

    latest = resolve_latest_megatron_checkpoint_dir(checkpoints)

    assert latest.name == "iter_0000010"
