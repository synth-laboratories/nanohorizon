"""Monkey-patch vLLM's MergedColumnParallelLinearWithLoRA.set_lora to
bounds-check the lora_a / lora_b lists against n_slices.

vLLM 0.18 crashes on Qwen3.5 (hybrid Mamba-attention) during CUDA-graph
profiling because the dummy LoRA created for packed GDN modules
(in_proj_qkvz, in_proj_ba) may have fewer tensors than the module's
n_slices.  The fix is trivial: iterate min(n_slices, len(lora_a))
instead of n_slices alone.

Apply once at process start (before the engine core initializes):

    from nanohorizon.custom_vllm.lora_patch import apply
    apply()
"""
from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    try:
        import torch
        from vllm.lora.layers.column_parallel_linear import (
            MergedColumnParallelLinearWithLoRA,
        )
    except ImportError:
        return

    _orig_set_lora = MergedColumnParallelLinearWithLoRA.set_lora

    def _patched_set_lora(
        self,
        index: int,
        lora_a: "torch.Tensor | list[torch.Tensor]",
        lora_b: "torch.Tensor | list[torch.Tensor]",
    ) -> None:
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        # --- patched: clamp to the shorter of n_slices vs list length ---
        n = self.n_slices
        if isinstance(lora_a, list):
            n = min(n, len(lora_a))
        if isinstance(lora_b, list):
            n = min(n, len(lora_b))

        for i in range(n):
            if (lora_a_i := lora_a[i]) is not None:
                self.lora_a_stacked[i][
                    index, 0, : lora_a_i.shape[0], : lora_a_i.shape[1]
                ].copy_(lora_a_i, non_blocking=True)
            if (lora_b_i := lora_b[i]) is not None:
                self.lora_b_stacked[i][
                    index, 0, : lora_b_i.shape[0], : lora_b_i.shape[1]
                ].copy_(lora_b_i, non_blocking=True)

    MergedColumnParallelLinearWithLoRA.set_lora = _patched_set_lora  # type: ignore[assignment]
    _APPLIED = True
