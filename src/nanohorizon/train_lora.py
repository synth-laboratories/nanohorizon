from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainResult:
    output_dir: str
    examples_seen: int
    optimizer_steps: int
    mean_loss: float


def _tokenize_pair(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, torch.Tensor]:
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not examples:
        raise ValueError("no examples provided for training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
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
                prompt=str(example["prompt"]),
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
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not examples:
        raise ValueError("no examples provided for training")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for example in examples:
        prompt = str(example.get("prompt") or "").strip()
        response = str(example.get("response") or "").strip()
        if not response:
            continue
        text = prompt + ("\n" if prompt else "") + response
        rows.append({"text": text})
    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    training_args = SFTConfig(
        output_dir=str(destination),
        learning_rate=learning_rate,
        num_train_epochs=max(1, epochs),
        max_steps=max_steps,
        max_length=max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_output = trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)

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
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not prompts:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(model, str(Path(adapter_dir).expanduser().resolve()))
    model.eval()

    outputs: list[str] = []
    for prompt in prompts:
        batch = tokenizer(
            prompt,
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
