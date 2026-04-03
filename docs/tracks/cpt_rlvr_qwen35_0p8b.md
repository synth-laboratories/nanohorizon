# CPT -> RLVR Qwen3.5 0.8B

This is the planned dense-CPT-then-RLVR track family for NanoHorizon.

**Track ID:** `cpt_rlvr_qwen35_0p8b`

## Contract

- dense CPT base model: `Qwen/Qwen3.5-0.8B`
- CPT framework: Megatron Bridge
- downstream RL: LoRA RLVR in `nanohorizon`
- environment focus: Crafter / Craftax rollout text for CPT, then Craftax RLVR

## Current Bootstrap Path

The first implemented slice is the dense CPT bootstrap lane:

```bash
./scripts/run_craftax_cpt_qwen35_0p8b_100k.sh --skip-import --skip-preprocess --skip-train
```

The first implemented data-generation slice is:

```bash
./scripts/run_craftax_cpt_generate_qwen35_27b_100k.sh
```

The current stitched wrapper for the full flow is:

```bash
./scripts/run_craftax_cpt_then_rlvr_qwen35_0p8b.sh
```

The RLVR stage expects:

- `NANOHORIZON_CPT_RLVR_MODEL_REF` to point at a remote-accessible exported CPT
  checkpoint, such as a Hugging Face model id

That path currently owns:

- reading raw CPT text rows from JSONL
- truncating to the first `100k` tokens using the target tokenizer
- preparing the Megatron Bridge runtime payload
- wiring an HF-to-Megatron checkpoint import step
- wiring a Megatron preprocess step for indexed GPT datasets
- wiring a Megatron Bridge pretraining entrypoint

## Expected First Real Run

When the external dependencies are installed, the intended first real bootstrap run is:

1. point `preprocess.script_path` or `NANOHORIZON_MEGATRON_PREPROCESS_SCRIPT`
   at Megatron-LM `tools/preprocess_data.py`
2. provide a raw CPT source dataset at `data.source_jsonl`
3. run:

```bash
./scripts/run_craftax_cpt_qwen35_0p8b_100k.sh
```

## Current Files

- `src/nanohorizon/baselines/cpt.py`
- `src/nanohorizon/baselines/cpt_data.py`
- `configs/craftax_cpt_qwen35_0p8b_100k.yaml`
- `configs/craftax_cpt_data_qwen35_27b_100k.yaml`
- `configs/craftax_rlvr_qwen35_0p8b_2xa100_20min.yaml`
- `scripts/run_craftax_cpt_qwen35_0p8b_100k.sh`
- `scripts/run_craftax_cpt_generate_qwen35_27b_100k.sh`
- `scripts/run_craftax_rlvr_qwen35_0p8b_2xa100_20min.sh`
- `scripts/run_craftax_cpt_then_rlvr_qwen35_0p8b.sh`
- `docs/tracks/cpt_rlvr_qwen35_0p8b_plan.md`

## Next Steps

- generate the first `100k` tokens of Crafter rollout text
- run the first real dense CPT bootstrap checkpoint
- export the CPT checkpoint to a remote-accessible HF model ref
- run RLVR from that exported CPT checkpoint
