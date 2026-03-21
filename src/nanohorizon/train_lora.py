from __future__ import annotations

import json
import math
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _load_text_only_causal_lm(*, base_model: str, device: str, use_cache: bool = True) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception:
        # Fallback when transformers' AutoModel mapping lags behind Qwen3.5 checkpoints.
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(base_model, **model_kwargs)

    model.config.use_cache = use_cache
    return model


@dataclass
class TrainResult:
    output_dir: str
    examples_seen: int
    optimizer_steps: int
    mean_loss: float


def _tokenize_pair(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, Any]:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])
    labels = ([-100] * len(prompt_ids)) + response_ids + ([eos_id] if eos_id is not None else [])
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def _render_prompt(tokenizer: Any, prompt: Any) -> str:
    if isinstance(prompt, list):
        messages = _normalize_messages_for_chat_template([item for item in prompt if isinstance(item, dict)])
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str):
                return rendered
        rendered_lines = []
        for message in messages:
            rendered_lines.append(
                f"<|{message.get('role', 'user')}|>\n{message.get('content', '')}"
            )
        rendered_lines.append("<|assistant|>")
        return "\n".join(rendered_lines)
    return str(prompt or "")


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(tool_calls):
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, Any], item)
        raw_function = item_dict.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except Exception:
                parsed_arguments = arguments
            arguments = parsed_arguments
        if not isinstance(arguments, (dict, list, str, int, float, bool)) and arguments is not None:
            arguments = str(arguments)
        normalized.append(
            {
                "id": str(item_dict.get("id", f"call_{index}") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _normalize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip() or "user"
        entry: dict[str, Any] = {
            "role": role,
            "content": item.get("content") if item.get("content") is not None else "",
        }
        tool_calls = _normalize_tool_calls(item.get("tool_calls"))
        if tool_calls:
            entry["tool_calls"] = tool_calls
            if role == "assistant" and not isinstance(entry["content"], str):
                entry["content"] = ""
        reasoning_content = item.get("reasoning_content")
        if reasoning_content is not None:
            entry["reasoning_content"] = str(reasoning_content)
        normalized.append(entry)
    return normalized


def _render_messages(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    safe_messages = _normalize_messages_for_chat_template([item for item in messages if isinstance(item, dict)])
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            safe_messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(rendered, str):
            return rendered
    return _render_prompt(tokenizer, safe_messages)


def _tokenize_messages_with_assistant_mask(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    import torch

    normalized_prompt_messages = _normalize_messages_for_chat_template(prompt_messages)
    normalized_full_messages = _normalize_messages_for_chat_template(full_messages)
    prompt_text = tokenizer.apply_chat_template(
        normalized_prompt_messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = _render_messages(tokenizer, normalized_full_messages, tools)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) < len(prompt_ids):
        raise ValueError("full chat template render shorter than prompt render")

    input_ids = full_ids[:max_length]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def release_cuda_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def train_weighted_lora(
    *,
    base_model: str,
    examples: list[dict[str, Any]],
    output_dir: str | Path,
    learning_rate: float,
    epochs: int,
    max_length: int,
    max_steps: int,
    lora_rank: int,
) -> TrainResult:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    if not examples:
        raise ValueError("no examples provided for training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=True)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    optimizer_steps = 0

    for _ in range(max(1, epochs)):
        for example in examples:
            if max_steps > 0 and optimizer_steps >= max_steps:
                break
            batch = _tokenize_pair(
                tokenizer,
                prompt=_render_prompt(tokenizer, example.get("prompt_messages") or example["prompt"]),
                response=str(example["response"]),
                max_length=max_length,
            )
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            weight = float(example.get("weight", 1.0))
            scaled_loss = loss * max(weight, 0.01)
            scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu().item()))
        if max_steps > 0 and optimizer_steps >= max_steps:
            break

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    del model
    release_cuda_memory()

    mean_loss = sum(losses) / max(len(losses), 1)
    if math.isnan(mean_loss):
        mean_loss = 0.0
    return TrainResult(
        output_dir=str(destination),
        examples_seen=len(examples),
        optimizer_steps=optimizer_steps,
        mean_loss=mean_loss,
    )


def train_sft_with_trl(
    *,
    base_model: str,
    examples: list[dict[str, Any]],
    output_dir: str | Path,
    learning_rate: float,
    epochs: int,
    max_length: int,
    max_steps: int,
    lora_rank: int,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
) -> TrainResult:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    if not examples:
        raise ValueError("no examples provided for training")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=False)

    rows: list[dict[str, list[int]]] = []
    for example in examples:
        prompt = str(example.get("prompt") or "").strip()
        prompt_messages = example.get("prompt_messages")
        full_messages = example.get("messages")
        tools = example.get("tools") if isinstance(example.get("tools"), list) else []
        response = str(example.get("response") or "").strip()
        if isinstance(prompt_messages, list) and isinstance(full_messages, list) and full_messages:
            safe_tools = [item for item in tools if isinstance(item, dict)] if isinstance(tools, list) else []
            tokenized = _tokenize_messages_with_assistant_mask(
                tokenizer,
                [item for item in prompt_messages if isinstance(item, dict)],
                [item for item in full_messages if isinstance(item, dict)],
                safe_tools,
                max_length,
            )
        else:
            if not response:
                continue
            tokenized = _tokenize_pair(
                tokenizer,
                _render_prompt(tokenizer, prompt_messages or prompt),
                response,
                max_length,
            )
        rows.append(
            {
                "input_ids": tokenized["input_ids"][0].tolist(),
                "labels": tokenized["labels"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist(),
            }
        )
    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    def data_collator(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_feature_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        input_ids = []
        labels = []
        attention_mask = []
        for feature in features:
            pad_len = max_feature_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + ([pad_id] * pad_len))
            labels.append(feature["labels"] + ([-100] * pad_len))
            attention_mask.append(feature["attention_mask"] + ([0] * pad_len))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(destination),
        learning_rate=learning_rate,
        num_train_epochs=float(max(1, epochs)),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    train_output = trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    del trainer
    del model
    release_cuda_memory()

    mean_loss = float(getattr(train_output, "training_loss", 0.0) or 0.0)
    return TrainResult(
        output_dir=str(destination),
        examples_seen=len(rows),
        optimizer_steps=int(max_steps if max_steps > 0 else len(rows)),
        mean_loss=mean_loss,
    )


def generate_with_adapter(
    *,
    base_model: str,
    adapter_dir: str | Path,
    prompts: list[Any],
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    if not prompts:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=True)
    model = PeftModel.from_pretrained(model, str(Path(adapter_dir).expanduser().resolve()))
    model.eval()

    outputs: list[str] = []
    for prompt in prompts:
        rendered_prompt = _render_prompt(tokenizer, prompt)
        batch = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = batch["input_ids"].shape[1]
        new_tokens = generated[0][prompt_len:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def generate_with_model(
    *,
    base_model: str,
    prompts: list[Any],
    max_length: int,
    max_new_tokens: int,
    adapter_dir: str | Path | None = None,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer

    if not prompts:
        return []

    model_path = base_model
    load_adapter = adapter_dir is not None
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_text_only_causal_lm(
        base_model=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=True,
    )
    if load_adapter:
        from peft import PeftModel

        assert adapter_dir is not None
        model = PeftModel.from_pretrained(model, str(Path(adapter_dir).expanduser().resolve()))
    model.eval()

    outputs: list[str] = []
    for prompt in prompts:
        rendered_prompt = _render_prompt(tokenizer, prompt)
        batch = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = batch["input_ids"].shape[1]
        new_tokens = generated[0][prompt_len:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def merge_lora_adapter(
    *,
    base_model: str,
    adapter_dir: str | Path,
    output_dir: str | Path | None = None,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    destination: Path
    if output_dir is None:
        destination = Path(tempfile.mkdtemp(prefix="nanohorizon_merged_model_"))
    else:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

    model = _load_text_only_causal_lm(
        base_model=base_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=True,
    )
    model = PeftModel.from_pretrained(model, str(Path(adapter_dir).expanduser().resolve()))
    merged = model.merge_and_unload()
    merged.save_pretrained(destination)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(destination)
    del merged
    del model
    release_cuda_memory()
    return str(destination)
