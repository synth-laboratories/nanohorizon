from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml


def now_utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must decode to an object: {config_path}")
    return payload


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target.resolve()
    anchor = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    return (anchor / target).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str: ...


@dataclass
class PreparedTextRow:
    text: str
    source_row_index: int
    source_token_count: int
    kept_token_count: int
    truncated: bool


@dataclass
class PreparedShardSummary:
    token_budget: int
    total_tokens: int
    rows_written: int
    source_rows_read: int
    source_rows_used: int
    truncated_rows: int
    stopped_early: bool


def _extract_text(row: Any, *, text_field: str) -> str:
    if isinstance(row, str):
        return row.strip()
    if isinstance(row, dict):
        raw = row.get(text_field)
        if isinstance(raw, str):
            return raw.strip()
    return ""


def read_jsonl_rows(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    for raw in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def truncate_rows_to_token_budget(
    rows: list[Any],
    *,
    text_field: str,
    token_budget: int,
    tokenizer: TokenizerLike,
) -> tuple[list[PreparedTextRow], PreparedShardSummary]:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    prepared: list[PreparedTextRow] = []
    total_tokens = 0
    source_rows_used = 0
    truncated_rows = 0
    stopped_early = False

    for row_index, row in enumerate(rows):
        text = _extract_text(row, text_field=text_field)
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        remaining = token_budget - total_tokens
        if remaining <= 0:
            stopped_early = True
            break
        kept_ids = token_ids[:remaining]
        kept_text = tokenizer.decode(kept_ids, skip_special_tokens=False).strip()
        if not kept_text:
            stopped_early = True
            break
        truncated = len(kept_ids) < len(token_ids)
        prepared.append(
            PreparedTextRow(
                text=kept_text,
                source_row_index=row_index,
                source_token_count=len(token_ids),
                kept_token_count=len(kept_ids),
                truncated=truncated,
            )
        )
        total_tokens += len(kept_ids)
        source_rows_used += 1
        if truncated:
            truncated_rows += 1
            stopped_early = True
            break

    return prepared, PreparedShardSummary(
        token_budget=token_budget,
        total_tokens=total_tokens,
        rows_written=len(prepared),
        source_rows_read=len(rows),
        source_rows_used=source_rows_used,
        truncated_rows=truncated_rows,
        stopped_early=stopped_early,
    )


def _load_tokenizer(model_name: str) -> TokenizerLike:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_bootstrap_shard(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    tokenizer_model: str,
    token_budget: int,
    text_field: str,
) -> PreparedShardSummary:
    rows = read_jsonl_rows(input_jsonl)
    tokenizer = _load_tokenizer(tokenizer_model)
    prepared_rows, summary = truncate_rows_to_token_budget(
        rows,
        text_field=text_field,
        token_budget=token_budget,
        tokenizer=tokenizer,
    )
    destination = Path(output_jsonl).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in prepared_rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    return summary


# ---------------------------------------------------------------------------
# HuggingFace Trainer CPT (no Megatron dependency)
# ---------------------------------------------------------------------------

def _rename_safetensors_keys(export_path: Path) -> None:
    """Rename weight keys in safetensors to match vLLM's expected naming.

    Transformers saves Qwen3_5ForConditionalGeneration weights as:
        model.language_model.layers.X  (model → Qwen3_5Model → language_model → TextModel)
    But vLLM's weight loader for Qwen3_5 expects the HF Hub naming:
        language_model.model.layers.X

    This function rewrites the safetensors files with the correct key names.
    """
    import safetensors.torch as st
    import torch

    shard_files = sorted(export_path.glob("*.safetensors"))
    if not shard_files:
        return

    renamed_count = 0
    for shard_path in shard_files:
        tensors = st.load_file(str(shard_path))
        new_tensors = {}
        for key, value in tensors.items():
            new_key = key
            if key.startswith("model.language_model."):
                # model.language_model.X → language_model.model.X
                suffix = key[len("model.language_model."):]
                new_key = f"language_model.model.{suffix}"
                renamed_count += 1
            elif key.startswith("model.") and not key.startswith("model.language_model"):
                # model.X (other top-level model attrs) → keep as-is for now
                pass
            new_tensors[new_key] = value
        st.save_file(new_tensors, str(shard_path))

    if renamed_count:
        # Also update the model.safetensors.index.json if present
        index_path = export_path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map", {})
            new_weight_map = {}
            for key, filename in weight_map.items():
                new_key = key
                if key.startswith("model.language_model."):
                    suffix = key[len("model.language_model."):]
                    new_key = f"language_model.model.{suffix}"
                new_weight_map[new_key] = filename
            index["weight_map"] = new_weight_map
            index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

        print(f"[cpt] Renamed {renamed_count} safetensors keys for vLLM compatibility", flush=True)


def _push_cpt_to_hub_preserving_format(
    *,
    trained_model: Any,
    base_model_name: str,
    hf_repo_id: str,
    hf_token: str,
) -> None:
    """Push CPT model to HF Hub with the EXACT same checkpoint format as the
    base model. This ensures vLLM can load it identically to the original.

    Strategy: download the base model's safetensors, replace text-model weights
    with the trained ones (same keys), upload back.
    """
    import tempfile
    import torch
    import safetensors.torch as st
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    api.create_repo(hf_repo_id, exist_ok=True, private=False, token=hf_token)

    # Download the base model checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(snapshot_download(base_model_name, local_dir=Path(tmpdir) / "base"))

        # Get the trained text-model state dict
        trained_state = trained_model.state_dict()

        # Map trained keys (model.X) to base-model keys (model.language_model.X)
        key_map = {}
        for key in trained_state:
            if key.startswith("model."):
                alt = "model.language_model." + key[len("model."):]
                key_map[key] = alt
            else:
                key_map[key] = key

        # Replace weights in the base model's safetensors shards
        shard_files = sorted(base_dir.glob("*.safetensors"))
        updated_total = 0
        for shard_path in shard_files:
            tensors = st.load_file(str(shard_path))
            updated = 0
            for trained_key, base_key in key_map.items():
                if base_key in tensors and tensors[base_key].shape == trained_state[trained_key].shape:
                    tensors[base_key] = trained_state[trained_key].cpu()
                    updated += 1
            if updated:
                st.save_file(tensors, str(shard_path))
                updated_total += updated

        print(f"[cpt] Replaced {updated_total}/{len(trained_state)} weights in base checkpoint", flush=True)

        # Upload everything
        api.upload_folder(
            folder_path=str(base_dir),
            repo_id=hf_repo_id,
            token=hf_token,
            commit_message="CPT bootstrap: Qwen3.5-0.8B fine-tuned on 100k Craftax tokens",
        )


def _build_full_model_with_trained_weights(
    *,
    trained_model: Any,
    base_model_name: str,
    trust_remote_code: bool,
) -> Any:
    """Build a full Qwen3_5ForConditionalGeneration with trained text weights.

    Returns the full model, or None if building fails.
    """
    import torch

    trained_state = trained_model.state_dict()

    try:
        from transformers import Qwen3_5ForConditionalGeneration
        full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
    except Exception as exc:
        print(f"[cpt] Warning: could not load full model: {exc}", flush=True)
        return None

    full_state = full_model.state_dict()
    updated = 0
    for key, value in trained_state.items():
        if key in full_state and full_state[key].shape == value.shape:
            full_state[key] = value
            updated += 1
        elif key.startswith("model."):
            # CausalLM: model.X → ConditionalGeneration: model.language_model.X
            alt_key = "model.language_model." + key[len("model."):]
            if alt_key in full_state and full_state[alt_key].shape == value.shape:
                full_state[alt_key] = value
                updated += 1

    full_model.load_state_dict(full_state)
    print(f"[cpt] Built full model with {updated}/{len(trained_state)} trained weights", flush=True)
    return full_model


def _save_as_full_model(
    *,
    trained_model: Any,
    base_model_name: str,
    export_path: Path,
    trust_remote_code: bool,
) -> None:
    """Save the trained text-only model weights inside the full conditional
    generation model structure that vLLM expects.

    AutoModelForCausalLM loads Qwen3.5 as Qwen3_5ForCausalLM (text-only, 320 params).
    vLLM's --language-model-only expects Qwen3_5ForConditionalGeneration (full, 473 params).
    We load the full model, copy trained weights into its language_model submodule, and save.
    """
    import torch

    trained_state = trained_model.state_dict()

    # Load the full conditional generation model
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        # Fallback: just save the trained model as-is
        trained_model.save_pretrained(str(export_path))
        return

    # Map text-only weights into the full model.
    # Qwen3_5ForCausalLM has: model.* and lm_head.*
    # Qwen3_5ForConditionalGeneration has: model.language_model.model.* and lm_head.*
    # But both also use: model.* at the top level.
    full_state = full_model.state_dict()
    updated = 0
    for key, value in trained_state.items():
        if key in full_state and full_state[key].shape == value.shape:
            full_state[key] = value
            updated += 1
        elif key.startswith("model."):
            # CausalLM: model.X → ConditionalGeneration: model.language_model.X
            alt_key = "model.language_model." + key[len("model."):]
            if alt_key in full_state and full_state[alt_key].shape == value.shape:
                full_state[alt_key] = value
                updated += 1

    full_model.load_state_dict(full_state)
    full_model.save_pretrained(str(export_path))
    print(f"[cpt] Saved full model ({updated}/{len(trained_state)} weights copied)", flush=True)


def _patch_config_for_vllm(export_path: Path) -> None:
    """Patch the saved config.json so vLLM with transformers 4.57.6 can load it.

    Transformers git-main (5.x) saves Qwen3.5 text-only models with:
      model_type='qwen3_5_text', architectures=['Qwen3_5ForCausalLM']
    But vLLM's transformers==4.57.6 only knows:
      model_type='qwen3_5', architectures=['Qwen3_5ForConditionalGeneration']
    With --language-model-only, vLLM handles the text extraction internally.
    """
    config_path = export_path / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    patched = False
    # Transformers 5.x saves text-only Qwen3.5 as qwen3_5_text/Qwen3_5ForCausalLM.
    # vLLM's transformers==4.57.6 only knows qwen3_5/Qwen3_5ForConditionalGeneration.
    # Patch to the full-model names so vLLM can resolve the architecture, then
    # it loads the text-only weights (which are a strict subset of the full model).
    if config.get("model_type") == "qwen3_5_text":
        config["model_type"] = "qwen3_5"
        patched = True
    if config.get("architectures") == ["Qwen3_5ForCausalLM"]:
        config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        patched = True
    config["use_cache"] = True
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if patched:
        print(f"[cpt] Patched config.json for vLLM: model_type=qwen3_5, arch=Qwen3_5ForConditionalGeneration", flush=True)
    else:
        print(f"[cpt] Patched config.json: use_cache=True", flush=True)

    # Patch tokenizer_config.json: transformers 5.x saves tokenizer_class as
    # "TokenizersBackend" which doesn't exist in 4.57.6.
    tokenizer_config_path = export_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tok_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if tok_config.get("tokenizer_class") == "TokenizersBackend":
            tok_config["tokenizer_class"] = "Qwen2Tokenizer"
            tok_config.pop("backend", None)
            tokenizer_config_path.write_text(json.dumps(tok_config, indent=2) + "\n", encoding="utf-8")
            print("[cpt] Patched tokenizer_config.json: tokenizer_class=Qwen2Tokenizer", flush=True)


def train_with_hf_trainer(
    *,
    base_model: str,
    prepared_jsonl: str | Path,
    output_dir: str | Path,
    text_field: str = "text",
    seq_length: int = 2048,
    train_iters: int = 10,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    learning_rate: float = 3.0e-5,
    lr_warmup_iters: int = 1,
    save_interval: int = 10,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    output_path = ensure_dir(output_dir)
    hf_export_path = output_path / "hf_export"
    hf_export_path.mkdir(parents=True, exist_ok=True)

    print(f"[cpt] Loading tokenizer: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[cpt] Loading model: {base_model}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    print(f"[cpt] Loading data: {prepared_jsonl}", flush=True)
    rows = read_jsonl_rows(prepared_jsonl)
    texts = [_extract_text(row, text_field=text_field) for row in rows]
    texts = [t for t in texts if t]

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_length,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    gradient_accumulation = max(1, global_batch_size // micro_batch_size)
    total_steps = max(train_iters, 1)

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=max(1, (total_steps * gradient_accumulation * micro_batch_size) // max(len(tokenized), 1) + 1),
        max_steps=total_steps,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=lr_warmup_iters,
        weight_decay=0.01,
        logging_steps=1,
        save_steps=save_interval,
        save_total_limit=2,
        bf16=device == "cuda",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print(f"[cpt] Starting training: {total_steps} steps, lr={learning_rate}", flush=True)
    train_result = trainer.train()

    # Save as the full Qwen3_5ForConditionalGeneration model (with trained
    # text weights copied in) so vLLM can load with --language-model-only.
    hf_repo_id = os.environ.get("NANOHORIZON_CPT_HF_REPO", "JoshPurtell/qwen35-0p8b-craftax-cpt")
    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"[cpt] Building full model with trained weights...", flush=True)
    full_model = _build_full_model_with_trained_weights(
        trained_model=model,
        base_model_name=base_model,
        trust_remote_code=trust_remote_code,
    )

    if hf_token:
        print(f"[cpt] Cloning base repo and replacing weights on HF Hub: {hf_repo_id}", flush=True)
        _push_cpt_to_hub_preserving_format(
            trained_model=model,
            base_model_name=base_model,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
        )
        print(f"[cpt] Pushed to {hf_repo_id}", flush=True)
        hf_export_path = Path(hf_repo_id)
    else:
        print(f"[cpt] Saving model to {hf_export_path}", flush=True)
        trainer.save_model(str(hf_export_path))
        tokenizer.save_pretrained(str(hf_export_path))
        _patch_config_for_vllm(hf_export_path)

    summary = {
        "base_model": base_model,
        "hf_export_path": str(hf_export_path),
        "train_loss": float(train_result.training_loss),
        "total_steps": int(train_result.global_step),
        "total_tokens_trained": sum(row.kept_token_count if hasattr(row, "kept_token_count") else 0 for row in rows) if rows else 0,
        "num_texts": len(texts),
        "seq_length": seq_length,
        "backend": "hf_trainer",
    }
    write_json(output_path / "train_summary.json", summary)
    print(f"[cpt] Training complete: loss={train_result.training_loss:.4f}", flush=True)
    return summary


def run_hf_pipeline(
    *,
    config_path: str | Path,
    output_dir: str | Path = "",
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    output_root = resolve_path(
        output_dir
        or config.get("output", {}).get("root_dir")
        or (Path("artifacts") / f"cpt_bootstrap_{now_utc_stamp()}"),
        base_dir=config_dir,
    )
    out_dir = ensure_dir(output_root)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    base_model = str(model_cfg.get("model") or "").strip()
    if not base_model:
        raise ValueError("config.model.model is required")

    input_jsonl = resolve_path(str(data_cfg.get("source_jsonl") or ""), base_dir=config_dir)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"missing CPT source_jsonl: {input_jsonl}")
    text_field = str(data_cfg.get("text_field", "text") or "text")
    token_budget = int(data_cfg.get("token_budget", 100_000))

    prepared_jsonl = out_dir / "prepared_bootstrap_text.jsonl"
    prepared_summary = prepare_bootstrap_shard(
        input_jsonl=input_jsonl,
        output_jsonl=prepared_jsonl,
        tokenizer_model=base_model,
        token_budget=token_budget,
        text_field=text_field,
    )
    write_json(out_dir / "prepared_bootstrap_summary.json", asdict(prepared_summary))

    train_summary = train_with_hf_trainer(
        base_model=base_model,
        prepared_jsonl=prepared_jsonl,
        output_dir=out_dir,
        text_field="text",
        seq_length=int(training_cfg.get("seq_length", 2048)),
        train_iters=int(training_cfg.get("train_iters", 10)),
        global_batch_size=int(training_cfg.get("global_batch_size", 8)),
        micro_batch_size=int(training_cfg.get("micro_batch_size", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 3.0e-5)),
        lr_warmup_iters=int(training_cfg.get("lr_warmup_iters", 1)),
        save_interval=int(training_cfg.get("save_interval", 10)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    result = {
        "output_dir": str(out_dir),
        "hf_export_path": train_summary["hf_export_path"],
        "prepared_summary": asdict(prepared_summary),
        "train_summary": train_summary,
        "backend": "hf_trainer",
    }
    write_json(out_dir / "pipeline_summary.json", result)
    return result


# ---------------------------------------------------------------------------
# Megatron Bridge CPT (legacy, requires megatron.bridge)
# ---------------------------------------------------------------------------

def _load_megatron_bridge() -> None:
    try:
        import megatron.bridge  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Megatron Bridge is required for CPT import/train steps. "
            "Install it in the active environment before executing those stages."
        ) from exc


def import_hf_checkpoint(*, base_model: str, megatron_path: str | Path, trust_remote_code: bool) -> dict[str, Any]:
    _load_megatron_bridge()
    from megatron.bridge import AutoBridge

    destination = ensure_dir(megatron_path)
    AutoBridge.import_ckpt(
        hf_model_id=base_model,
        megatron_path=str(destination),
        trust_remote_code=trust_remote_code,
    )
    return {
        "base_model": base_model,
        "megatron_path": str(destination),
        "trust_remote_code": trust_remote_code,
    }


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def build_preprocess_command(
    *,
    script_path: str | Path,
    input_jsonl: str | Path,
    output_prefix: str | Path,
    tokenizer_model: str,
    workers: int,
    text_field: str,
    append_eod: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(script_path).expanduser().resolve()),
        "--input",
        str(Path(input_jsonl).expanduser().resolve()),
        "--output-prefix",
        str(Path(output_prefix).expanduser().resolve()),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        tokenizer_model,
        "--json-keys",
        text_field,
        "--workers",
        str(max(1, int(workers))),
    ]
    if append_eod:
        command.append("--append-eod")
    return command


def build_torchrun_command(*, runtime_config_path: str | Path, nproc_per_node: int) -> list[str]:
    return [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(max(1, int(nproc_per_node))),
        "-m",
        "nanohorizon.baselines.cpt",
        "bridge-train",
        "--runtime-config",
        str(Path(runtime_config_path).expanduser().resolve()),
    ]


def build_export_command(
    *,
    conversion_script_path: str | Path,
    hf_model: str,
    megatron_path: str | Path,
    hf_output_path: str | Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(conversion_script_path).expanduser().resolve()),
        "export",
        "--hf-model",
        hf_model,
        "--megatron-path",
        str(Path(megatron_path).expanduser().resolve()),
        "--hf-path",
        str(Path(hf_output_path).expanduser().resolve()),
    ]


def run_command(command: list[str], *, cwd: str | Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


def _recipe_hint_for_model(model_name: str) -> str:
    lowered = model_name.lower()
    if "1.7b" in lowered or "1p7b" in lowered:
        return "qwen3_1p7b"
    return "qwen3_600m"


def _build_runtime_payload(
    *,
    config: dict[str, Any],
    prepared_jsonl: Path,
    preprocessed_prefix: Path,
    imported_checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tb_logs"
    return {
        "model": {
            "base_model": str(model_cfg.get("model") or ""),
            "recipe_hint": str(model_cfg.get("recipe_hint") or _recipe_hint_for_model(str(model_cfg.get("model") or ""))),
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        },
        "data": {
            "prepared_jsonl": str(prepared_jsonl),
            "indexed_dataset_prefix": str(preprocessed_prefix),
            "seq_length": int(training_cfg.get("seq_length", 2048)),
        },
        "checkpoint": {
            "pretrained_checkpoint": str(imported_checkpoint_dir),
            "save_dir": str(checkpoint_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "save_interval": int(training_cfg.get("save_interval", 10)),
        },
        "training": {
            "train_iters": int(training_cfg.get("train_iters", 10)),
            "global_batch_size": int(training_cfg.get("global_batch_size", 8)),
            "micro_batch_size": int(training_cfg.get("micro_batch_size", 1)),
            "learning_rate": float(training_cfg.get("learning_rate", 3.0e-5)),
            "lr_warmup_iters": int(training_cfg.get("lr_warmup_iters", 1)),
            "tensor_model_parallel_size": int(training_cfg.get("tensor_model_parallel_size", 1)),
            "pipeline_model_parallel_size": int(training_cfg.get("pipeline_model_parallel_size", 1)),
            "context_parallel_size": int(training_cfg.get("context_parallel_size", 1)),
        },
    }


def _apply_if_present(obj: Any, attr: str, value: Any) -> None:
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def resolve_latest_megatron_checkpoint_dir(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {root}")
    iter_dirs = sorted(
        [candidate for candidate in root.glob("iter_*") if candidate.is_dir()],
        key=lambda candidate: candidate.name,
    )
    if iter_dirs:
        return iter_dirs[-1]
    return root


def train_with_megatron_bridge(runtime_config: dict[str, Any]) -> None:
    _load_megatron_bridge()
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.qwen import qwen3_1p7b_pretrain_config, qwen3_600m_pretrain_config
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    model_cfg = runtime_config["model"]
    data_cfg = runtime_config["data"]
    checkpoint_cfg = runtime_config["checkpoint"]
    training_cfg = runtime_config["training"]

    recipe_hint = str(model_cfg.get("recipe_hint") or "qwen3_600m")
    if recipe_hint == "qwen3_1p7b":
        cfg = qwen3_1p7b_pretrain_config()
    else:
        cfg = qwen3_600m_pretrain_config()

    base_model = str(model_cfg["base_model"])
    bridge = AutoBridge.from_hf_pretrained(
        base_model,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    cfg.model = bridge.to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = base_model

    cfg.dataset.data_path = str(data_cfg["indexed_dataset_prefix"])
    cfg.dataset.blend = None
    cfg.dataset.split = "100,0,0"
    _apply_if_present(cfg.dataset, "seq_length", int(data_cfg["seq_length"]))
    _apply_if_present(cfg.dataset, "num_dataset_builder_threads", 1)

    cfg.checkpoint.pretrained_checkpoint = str(checkpoint_cfg["pretrained_checkpoint"])
    cfg.checkpoint.save = str(checkpoint_cfg["save_dir"])
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = int(checkpoint_cfg["save_interval"])
    cfg.checkpoint.finetune = True

    cfg.logger.tensorboard_dir = str(checkpoint_cfg["tensorboard_dir"])

    cfg.train.train_iters = int(training_cfg["train_iters"])
    cfg.train.global_batch_size = int(training_cfg["global_batch_size"])
    cfg.train.micro_batch_size = int(training_cfg["micro_batch_size"])

    _apply_if_present(cfg.optimizer, "lr", float(training_cfg["learning_rate"]))
    _apply_if_present(cfg.scheduler, "max_lr", float(training_cfg["learning_rate"]))
    _apply_if_present(cfg.scheduler, "lr_warmup_iters", int(training_cfg["lr_warmup_iters"]))
    _apply_if_present(cfg.scheduler, "lr_decay_iters", int(training_cfg["train_iters"]))

    _apply_if_present(cfg.model, "tensor_model_parallel_size", int(training_cfg["tensor_model_parallel_size"]))
    _apply_if_present(cfg.model, "pipeline_model_parallel_size", int(training_cfg["pipeline_model_parallel_size"]))
    _apply_if_present(cfg.model, "context_parallel_size", int(training_cfg["context_parallel_size"]))

    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0

    pretrain(config=cfg, forward_step_func=forward_step)


def run_pipeline(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    output_root = resolve_path(
        args.output_dir
        or config.get("output", {}).get("root_dir")
        or (Path("artifacts") / f"cpt_bootstrap_{now_utc_stamp()}"),
        base_dir=config_dir,
    )
    output_dir = ensure_dir(output_root)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    preprocess_cfg = config.get("preprocess", {})
    training_cfg = config.get("training", {})

    base_model = str(model_cfg.get("model") or "").strip()
    if not base_model:
        raise ValueError("config.model.model is required")

    input_jsonl = resolve_path(str(data_cfg.get("source_jsonl") or ""), base_dir=config_dir)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"missing CPT source_jsonl: {input_jsonl}")
    text_field = str(data_cfg.get("text_field", "text") or "text")
    token_budget = int(data_cfg.get("token_budget", 100_000))

    prepared_jsonl = output_dir / "prepared_bootstrap_text.jsonl"
    prepared_summary = prepare_bootstrap_shard(
        input_jsonl=input_jsonl,
        output_jsonl=prepared_jsonl,
        tokenizer_model=base_model,
        token_budget=token_budget,
        text_field=text_field,
    )
    write_json(output_dir / "prepared_bootstrap_summary.json", asdict(prepared_summary))

    imported_checkpoint_dir = output_dir / "imported_megatron_checkpoint"
    if args.skip_import:
        import_summary = {
            "skipped": True,
            "expected_megatron_path": str(imported_checkpoint_dir),
        }
    else:
        import_summary = import_hf_checkpoint(
            base_model=base_model,
            megatron_path=imported_checkpoint_dir,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
    write_json(output_dir / "import_summary.json", import_summary)

    preprocessed_prefix = output_dir / "megatron_data" / "craftax_cpt_bootstrap"
    preprocess_script = str(
        preprocess_cfg.get("script_path")
        or os.getenv("NANOHORIZON_MEGATRON_PREPROCESS_SCRIPT", "")
    ).strip()
    preprocess_command: list[str] | None = None
    if preprocess_script:
        preprocess_command = build_preprocess_command(
            script_path=preprocess_script,
            input_jsonl=prepared_jsonl,
            output_prefix=preprocessed_prefix,
            tokenizer_model=base_model,
            workers=int(preprocess_cfg.get("workers", 1)),
            text_field="text",
            append_eod=bool(preprocess_cfg.get("append_eod", True)),
        )
        write_json(output_dir / "preprocess_command.json", preprocess_command)
        if not args.skip_preprocess:
            ensure_dir(preprocessed_prefix.parent)
            run_command(preprocess_command, cwd=config_dir)
    elif not args.skip_preprocess:
        raise RuntimeError(
            "Megatron preprocess script is required for the preprocess stage. "
            "Set preprocess.script_path in config or NANOHORIZON_MEGATRON_PREPROCESS_SCRIPT."
        )

    runtime_payload = _build_runtime_payload(
        config=config,
        prepared_jsonl=prepared_jsonl,
        preprocessed_prefix=preprocessed_prefix,
        imported_checkpoint_dir=imported_checkpoint_dir,
        output_dir=output_dir,
    )
    runtime_config_path = output_dir / "bridge_runtime_config.json"
    write_json(runtime_config_path, runtime_payload)

    train_command = build_torchrun_command(
        runtime_config_path=runtime_config_path,
        nproc_per_node=int(training_cfg.get("gpus_per_node", 1)),
    )
    write_json(output_dir / "train_command.json", train_command)
    write_text(
        output_dir / "command.txt",
        " ".join(train_command) + "\n",
    )

    if not args.skip_train:
        run_command(train_command, cwd=config_dir)

    export_cfg = config.get("export", {})
    if bool(export_cfg.get("enabled", False)):
        conversion_script_path = str(
            export_cfg.get("conversion_script_path")
            or os.getenv("NANOHORIZON_MEGATRON_CONVERT_SCRIPT", "")
        ).strip()
        if not conversion_script_path:
            raise RuntimeError(
                "Megatron Bridge export is enabled but no conversion script path was provided. "
                "Set export.conversion_script_path or NANOHORIZON_MEGATRON_CONVERT_SCRIPT."
            )
        latest_ckpt_dir = resolve_latest_megatron_checkpoint_dir(output_dir / "checkpoints")
        export_hf_path = resolve_path(
            export_cfg.get("hf_output_path") or (output_dir / "hf_export"),
            base_dir=config_dir,
        )
        export_command = build_export_command(
            conversion_script_path=conversion_script_path,
            hf_model=base_model,
            megatron_path=latest_ckpt_dir,
            hf_output_path=export_hf_path,
        )
        write_json(output_dir / "export_command.json", export_command)
        write_json(
            output_dir / "export_plan.json",
            {
                "enabled": True,
                "latest_checkpoint_dir": str(latest_ckpt_dir),
                "hf_output_path": str(export_hf_path),
            },
        )
        if not args.skip_export:
            ensure_dir(export_hf_path)
            run_command(export_command, cwd=config_dir)
    else:
        write_json(output_dir / "export_plan.json", {"enabled": False, "skipped": True})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon dense CPT bootstrap utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Prepare and optionally execute the full bootstrap CPT pipeline.")
    run_parser.add_argument("--config", required=True, help="Path to the CPT config YAML/JSON.")
    run_parser.add_argument("--output-dir", default="", help="Optional override for the output directory.")
    run_parser.add_argument("--skip-import", action="store_true", help="Skip HF->Megatron checkpoint import.")
    run_parser.add_argument("--skip-preprocess", action="store_true", help="Skip Megatron dataset preprocessing.")
    run_parser.add_argument("--skip-train", action="store_true", help="Skip Megatron Bridge training execution.")
    run_parser.add_argument("--skip-export", action="store_true", help="Skip Megatron->HF export.")

    hf_run_parser = subparsers.add_parser("hf-run", help="Run CPT with HuggingFace Trainer (no Megatron needed).")
    hf_run_parser.add_argument("--config", required=True, help="Path to the CPT config YAML/JSON.")
    hf_run_parser.add_argument("--output-dir", default="", help="Optional override for the output directory.")

    train_parser = subparsers.add_parser("bridge-train", help="Internal Megatron Bridge train entrypoint.")
    train_parser.add_argument("--runtime-config", required=True, help="Path to the bridge runtime JSON payload.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "run":
        run_pipeline(args)
        return
    if args.command == "hf-run":
        result = run_hf_pipeline(config_path=args.config, output_dir=args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "bridge-train":
        runtime_config = load_config(args.runtime_config)
        train_with_megatron_bridge(runtime_config)
        return
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Modal entrypoint for CPT on GPU
# ---------------------------------------------------------------------------
REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

import modal

from nanohorizon.shared.modal_common import (
    GPU_OFFLINE,
    RECORDS_DIR,
    RECORDS_VOLUME,
    ARTIFACT_DIR,
    ARTIFACT_VOLUME,
    HF_CACHE_DIR,
    HF_CACHE_VOLUME,
    REMOTE_ROOT,
    training_image,
    volume_mounts,
)

CPT_APP_NAME = "nanohorizon-craftax-cpt"
cpt_app = modal.App(CPT_APP_NAME)

_cpt_image = training_image()


GPU_CPT = "A100-40GB"

@cpt_app.function(
    image=_cpt_image,
    gpu=GPU_CPT,
    volumes=volume_mounts(),
    timeout=60 * 30,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_cpt_on_modal(
    config_yaml: str,
    source_jsonl_content: str,
    output_label: str = "",
) -> dict[str, Any]:
    import os
    import json
    import yaml as _yaml

    os.environ["PYTHONPATH"] = f"{REMOTE_ROOT}/src"
    sys.path.insert(0, f"{REMOTE_ROOT}/src")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = output_label or "cpt_bootstrap"
    output_dir = Path(f"{RECORDS_DIR}/cpt_rlvr_qwen35_0p8b/{stamp}_{label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _yaml.safe_load(config_yaml)

    source_jsonl_path = output_dir / "source_cpt_rollouts_text.jsonl"
    source_jsonl_path.write_text(source_jsonl_content, encoding="utf-8")

    result = run_hf_pipeline_from_config(
        config=config,
        source_jsonl_path=source_jsonl_path,
        output_dir=output_dir,
    )

    RECORDS_VOLUME.commit()
    return result


@cpt_app.local_entrypoint()
def cpt_modal_main(
    config: str = "configs/craftax_cpt_qwen35_0p8b_100k.yaml",
    output_dir: str = "",
) -> None:
    from nanohorizon.shared.modal_common import PROJECT_ROOT

    config_path = (PROJECT_ROOT / config).expanduser().resolve()
    config_payload = load_config(config_path)

    source_jsonl_rel = str(config_payload.get("data", {}).get("source_jsonl") or "")
    source_jsonl_path = resolve_path(source_jsonl_rel, base_dir=config_path.parent)
    if not source_jsonl_path.exists():
        raise FileNotFoundError(f"CPT source JSONL not found: {source_jsonl_path}")

    source_content = source_jsonl_path.read_text(encoding="utf-8")
    config_yaml = config_path.read_text(encoding="utf-8")
    label = str(output_dir or "cpt_bootstrap").strip()

    print(f"[cpt-modal] Submitting CPT job to Modal (source={source_jsonl_path}, {len(source_content)} bytes)", flush=True)
    result = run_cpt_on_modal.remote(
        config_yaml=config_yaml,
        source_jsonl_content=source_content,
        output_label=label,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def run_hf_pipeline_from_config(
    *,
    config: dict[str, Any],
    source_jsonl_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    base_model = str(model_cfg.get("model") or "").strip()
    if not base_model:
        raise ValueError("config.model.model is required")

    text_field = str(data_cfg.get("text_field", "text") or "text")
    token_budget = int(data_cfg.get("token_budget", 100_000))

    out_dir = ensure_dir(output_dir)
    prepared_jsonl = out_dir / "prepared_bootstrap_text.jsonl"
    prepared_summary = prepare_bootstrap_shard(
        input_jsonl=source_jsonl_path,
        output_jsonl=prepared_jsonl,
        tokenizer_model=base_model,
        token_budget=token_budget,
        text_field=text_field,
    )
    write_json(out_dir / "prepared_bootstrap_summary.json", asdict(prepared_summary))

    train_summary = train_with_hf_trainer(
        base_model=base_model,
        prepared_jsonl=prepared_jsonl,
        output_dir=out_dir,
        text_field="text",
        seq_length=int(training_cfg.get("seq_length", 2048)),
        train_iters=int(training_cfg.get("train_iters", 10)),
        global_batch_size=int(training_cfg.get("global_batch_size", 8)),
        micro_batch_size=int(training_cfg.get("micro_batch_size", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 3.0e-5)),
        lr_warmup_iters=int(training_cfg.get("lr_warmup_iters", 1)),
        save_interval=int(training_cfg.get("save_interval", 10)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    result = {
        "output_dir": str(out_dir),
        "hf_export_path": train_summary["hf_export_path"],
        "prepared_summary": asdict(prepared_summary),
        "train_summary": train_summary,
        "backend": "hf_trainer",
        "base_model": base_model,
    }
    write_json(out_dir / "pipeline_summary.json", result)
    return result
