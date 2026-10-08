# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Block-scaled FP8 fake quantization for BF16 training (quantization-aware training).

The GEMMs of the model keep running in BF16, but their operands are rounded through
FP8 (E4M3) the way a block-scaled FP8 inference engine rounds them, so the policy
the optimizer sees is the policy the inference engine serves. Every rounding is
wrapped in a straight-through estimator: the forward uses the rounded value, the
backward passes the gradient to the unrounded tensor unchanged.

The recipe is the one DeepSeek checkpoints are stored and served in:

* weights: one scale per 128x128 tile, powers of two (``scale_fmt: ue8m0``);
* activations: one scale per 128 values along the reduction axis, recomputed on
  every forward, plain float32 (power-of-two optional, for DeepGEMM-style kernels);
* KV cache (DeepSeek-V4): E4M3 with one power-of-two scale per 64 NoPE channels
  of the KV latent, RoPE channels kept in BF16;
* indexer (DeepSeek-V4): queries and keys in E4M3 with one power-of-two scale per
  vector.

Everything is driven by ``TransformerConfig`` (``fp8_fake_quant``,
``fp8_fake_quant_kv_cache``, ``fp8_fake_quant_indexer_qk`` and their block sizes).
The linear-layer part hooks the Transformer Engine wrappers in
``megatron.core.extensions.transformer_engine``; the attention-side parts are called
from the DeepSeek-V4 attention and CSA modules.

Combined with FP8 training (``fp8`` set) the TE linear layers are left to Transformer
Engine and ``fp8_fake_quant`` only rounds the DeepSeek-V4 grouped output projection, so
the same switches give the rollout-matching forward in both training modes.
"""

from typing import List

import torch
import torch.nn.functional as F

from megatron.core.model_parallel_config import ModelParallelConfig

E4M3_MAX = 448.0

_EXCLUDED_ATTR = "_fp8_fake_quant_excluded"


def _scale_from_amax(
    amax: torch.Tensor, pow2_scale: bool, amax_floor: float | None = None
) -> torch.Tensor:
    """Per-block scale from the block's absolute maximum.

    ``amax_floor`` reproduces kernels that clamp the amax before taking the
    exponent; the remaining ``tiny`` clamp only keeps an all-zero block (which
    rounds to zero regardless) away from a division by zero.
    """
    if amax_floor is not None:
        amax = amax.clamp(min=amax_floor)
    scale = (amax / E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    if pow2_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return scale


def _round_to_e4m3(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``values`` (float32) to E4M3 with ``scale`` and dequantize again."""
    return (values / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * scale


def _straight_through(x: torch.Tensor, rounded: torch.Tensor) -> torch.Tensor:
    """Forward ``rounded``, backward the identity on ``x``."""
    return x + (rounded.to(x.dtype) - x).detach()


def fake_quant_weight_blockwise(
    weight: torch.Tensor, block_size: int = 128, pow2_scale: bool = True
) -> torch.Tensor:
    """Round a 2D weight through E4M3 with one scale per ``block_size`` square tile.

    Tiles start at the top-left corner, as in the checkpoint layout; a dimension
    that is not a multiple of ``block_size`` gets a partial tile at its end.
    """
    if weight.dim() != 2:
        raise ValueError(f"fake_quant_weight_blockwise expects a 2D weight, got {weight.shape}")
    rows, cols = weight.shape
    pad_r, pad_c = (-rows) % block_size, (-cols) % block_size

    values = weight.detach().float()
    if pad_r or pad_c:
        values = F.pad(values, (0, pad_c, 0, pad_r))
    tiled = values.reshape(
        values.shape[0] // block_size, block_size, values.shape[1] // block_size, block_size
    )
    scale = _scale_from_amax(tiled.abs().amax(dim=(1, 3), keepdim=True), pow2_scale)
    rounded = _round_to_e4m3(tiled, scale).reshape(values.shape)
    if pad_r or pad_c:
        rounded = rounded[:rows, :cols]
    return _straight_through(weight, rounded)


def fake_quant_activation_groupwise(
    x: torch.Tensor, group_size: int = 128, pow2_scale: bool = False
) -> torch.Tensor:
    """Round an activation through E4M3 with one scale per ``group_size`` values of its last dim."""
    last = x.shape[-1]
    pad = (-last) % group_size

    values = x.detach().float()
    if pad:
        values = F.pad(values, (0, pad))
    grouped = values.reshape(*values.shape[:-1], values.shape[-1] // group_size, group_size)
    scale = _scale_from_amax(grouped.abs().amax(dim=-1, keepdim=True), pow2_scale)
    rounded = _round_to_e4m3(grouped, scale).reshape(values.shape)
    if pad:
        rounded = rounded[..., :last]
    return _straight_through(x, rounded)


def fake_quant_kv_latent(kv: torch.Tensor, rope_dim: int, block_size: int = 64) -> torch.Tensor:
    """Round a KV latent ``[..., head_dim]`` the way an FP8 KV cache stores it.

    The leading ``head_dim - rope_dim`` (NoPE) channels become E4M3 with one
    power-of-two scale per ``block_size`` channels; the trailing ``rope_dim``
    channels are returned untouched. The amax is floored at 1e-4 before the
    exponent is taken, as the cache-write kernel does.
    """
    nope = kv.shape[-1] - rope_dim
    if nope <= 0 or nope % block_size:
        raise ValueError(
            f"fake_quant_kv_latent: {kv.shape[-1]} channels minus {rope_dim} RoPE channels is not "
            f"a multiple of the KV block size {block_size}"
        )
    head, tail = kv[..., :nope], kv[..., nope:]

    grouped = head.detach().float().reshape(*head.shape[:-1], nope // block_size, block_size)
    scale = _scale_from_amax(
        grouped.abs().amax(dim=-1, keepdim=True), pow2_scale=True, amax_floor=1e-4
    )
    rounded = _round_to_e4m3(grouped, scale).reshape(head.shape)
    return torch.cat((_straight_through(head, rounded), tail), dim=-1)


def fake_quant_per_vector(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` through E4M3 with one power-of-two scale over its whole last dim."""
    values = x.detach().float()
    scale = _scale_from_amax(
        values.abs().amax(dim=-1, keepdim=True), pow2_scale=True, amax_floor=1e-10
    )
    return _straight_through(x, _round_to_e4m3(values, scale))


def exclude_from_fp8_fake_quant(module: torch.nn.Module) -> torch.nn.Module:
    """Keep a linear module out of ``fp8_fake_quant``.

    For the projections an FP8 checkpoint stores in BF16 and the inference engine
    therefore runs in BF16 (for DeepSeek-V4: the CSA compressor and the indexer
    weights projection).
    """
    setattr(module, _EXCLUDED_ATTR, True)
    return module


def fp8_fake_quant_active(module: torch.nn.Module) -> bool:
    """Whether ``module`` (a TE linear wrapper carrying ``.config``) fake-quantizes.

    Under FP8 training (``config.fp8`` set) the TE linear layers already compute in FP8,
    so they are left alone and ``fp8_fake_quant`` only covers the tensors outside the TE
    GEMMs (the DeepSeek-V4 grouped output projection, KV cache and indexer).
    """
    config = getattr(module, "config", None)
    if not getattr(config, "fp8_fake_quant", False) or getattr(config, "fp8", None):
        return False
    return not getattr(module, _EXCLUDED_ATTR, False)


def _validate_fp8_fake_quant_support(
    config: ModelParallelConfig, delay_wgrad_compute: bool, weight_tensors: List[torch.Tensor]
) -> None:
    """Reject paths that replace TE weights with STE tensors unsafely (as the int4 fake QAT does)."""
    if config.gradient_accumulation_fusion:
        raise RuntimeError(
            "fp8_fake_quant is not supported with gradient_accumulation_fusion because TE fused "
            "wgrad accumulation mutates Python attributes on the original weight tensors."
        )
    if delay_wgrad_compute:
        raise RuntimeError(
            "fp8_fake_quant is not supported with delayed wgrad compute because the delayed TE "
            "path mutates Python attributes on the original weight tensors."
        )
    if any(
        hasattr(weight, "__fsdp_param__") or hasattr(weight, "get_main_grad")
        for weight in weight_tensors
    ):
        raise RuntimeError(
            "fp8_fake_quant is not supported with Megatron FSDP because FSDP patches weight "
            "tensors with main-gradient attributes and methods."
        )


def maybe_fake_quantize_linear_weights(
    module: torch.nn.Module, weight_tensors: List[torch.Tensor], delay_wgrad_compute: bool
) -> List[torch.Tensor]:
    """Fake-quantize the weight tensors of a TE linear wrapper under ``fp8_fake_quant``."""
    if not fp8_fake_quant_active(module):
        return weight_tensors
    config = module.config
    _validate_fp8_fake_quant_support(config, delay_wgrad_compute, weight_tensors)

    rounded_tensors = []
    for weight in weight_tensors:
        rounded = fake_quant_weight_blockwise(
            weight, config.fp8_fake_quant_weight_block_size, config.fp8_fake_quant_weight_pow2_scale
        )
        if hasattr(weight, "main_grad"):
            rounded.main_grad = weight.main_grad
        rounded_tensors.append(rounded)
    return rounded_tensors


def maybe_fake_quantize_linear_input(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Fake-quantize the input of a TE linear wrapper under ``fp8_fake_quant``."""
    if not fp8_fake_quant_active(module):
        return x
    config = module.config
    return fake_quant_activation_groupwise(
        x, config.fp8_fake_quant_act_group_size, config.fp8_fake_quant_act_pow2_scale
    )
