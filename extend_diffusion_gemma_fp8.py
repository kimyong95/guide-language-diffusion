"""Load the FP8 (compressed-tensors W8A8) DiffusionGemma checkpoint in transformers.

`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` stores its quantized weights in two layouts the
stock ``DiffusionGemmaForBlockDiffusion`` can't load as-is:

* the MoE **experts** *per expert* (``...experts.{E}.{gate,up,down}_proj.weight`` + ``.weight_scale``),
  whereas the stock ``DiffusionGemmaTextExperts`` keeps them as fused 3D bf16 Parameters
  (``experts.gate_up_proj`` / ``experts.down_proj``); and
* the **dense** linears (``self_attn.{q,k,v,o}_proj``, dense ``mlp.{gate,up,down}_proj``) as fp8
  ``weight`` + per-output-channel bf16 ``weight_scale``, which the stock model has as plain bf16
  ``nn.Linear``.

We only care about *memory*, not fp8 compute throughput (the matmuls upcast to bf16 anyway), so
both layouts are handled with one uniform recipe -- keep the weights in fp8 (e4m3) with bf16 scales
and dequantize on the fly in ``forward`` -- and compressed-tensors is dropped entirely.

``patch_diffusion_gemma_fp8()`` (idempotent; call once before ``from_pretrained``) does five things:
  1. swap the experts module for :class:`FP8DiffusionGemmaTextExperts` (fused fp8 experts;
     dequant only the active top-k on the fly, preserving the ~22 GB expert footprint),
  2. guard ``_init_weights`` so it doesn't ``init.normal_`` the fp8 expert tensors,
  3. register a checkpoint conversion merging the per-expert fp8 weights/scales into the fused fp8
     buffers (mirrors the qwen2_moe / glm4_moe entries in transformers' ``conversion_mapping.py``),
  4. swap the quantized dense ``nn.Linear``s for :class:`FP8Linear` (fp8 ``weight`` + bf16
     ``weight_scale``, dequant in ``forward``). Done in each text class's ``__init__`` -- before
     ``post_init`` -- so the encoder's dense ``weight_scale`` ties to the decoder via the existing
     ``*scale`` tie rule, with no post-load re-tie, and
  5. strip ``config.quantization_config`` so no compressed-tensors quantizer runs (we hold every
     quantized weight ourselves). Without a quantizer the loader keeps each param at its declared
     dtype, so the fp8 weights stay fp8 under ``dtype="auto"``.

See ``test-vllm-transformers.py``. Follows the ``extend_*.py`` convention in this repo
(``extend_diffusion_gemma_layer.py``, ``extend_llada_block.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, DiffusionGemmaForBlockDiffusion
from transformers.activations import ACT2FN
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
from transformers.models.diffusion_gemma import modeling_diffusion_gemma as _mdg


class FP8DiffusionGemmaTextExperts(nn.Module):
    """Drop-in replacement for ``DiffusionGemmaTextExperts`` holding fp8 experts.

    Same ``__init__(config)`` / ``forward(hidden_states, top_k_index, top_k_weights)``
    interface as the stock module, with matching parameter names (``gate_up_proj`` /
    ``down_proj``) plus the FP8 scales (``gate_up_proj_scale`` / ``down_proj_scale``).

    Deliberately NOT a subclass of ``DiffusionGemmaTextExperts``: keeping it a plain
    ``nn.Module`` avoids the ``@use_experts_implementation`` forward dispatch on the base
    class. (The ``_init_weights`` branch that would call ``init.normal_`` on the fp8
    ``gate_up_proj`` buffer is skipped via the guard installed in
    ``patch_diffusion_gemma_fp8()``.)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_activation]

        gate_up_out = 2 * self.intermediate_dim
        # Registered as Parameters (NOT buffers) so they ride `device_map`, appear in the
        # state dict for the converter to fill, and — critically — are reachable via
        # `model.get_parameter(...)`, which the tied-weights resolver
        # (`mark_tied_weights_as_initialized`, `tie_weights`) calls on every tied key.
        # A buffer makes that raise `AttributeError: ... is not an nn.Parameter`. The stock
        # `gate_up_proj`/`down_proj` are Parameters, and the `*_scale` tensors are tied too
        # (via the `gate_up_proj`/`down_proj` prefix tie rules), so all four must be
        # Parameters. `requires_grad=False`; the loader re-wraps fp8 with `requires_grad=True`
        # (harmless, matches stock). Weights are fp8; the checkpoint's per-output-channel
        # scales are bf16 with a trailing singleton dim so they broadcast over the input dim.
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, gate_up_out, self.hidden_dim, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.gate_up_proj_scale = nn.Parameter(
            torch.empty(self.num_experts, gate_up_out, 1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.down_proj_scale = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, 1, dtype=torch.bfloat16),
            requires_grad=False,
        )

    @staticmethod
    def _dequant(proj: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # proj: (out, in) fp8, scale: (out, 1) bf16  ->  (out, in) in compute dtype.
        return proj.to(dtype) * scale.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors DiffusionGemmaTextExperts.forward (modeling_diffusion_gemma.py), only the
        # two expert matmuls change: the hit expert's weight slice is dequantized first.
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up_w = self._dequant(
                self.gate_up_proj[expert_idx], self.gate_up_proj_scale[expert_idx], current_state.dtype
            )
            gate, up = F.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down_w = self._dequant(
                self.down_proj[expert_idx], self.down_proj_scale[expert_idx], current_hidden_states.dtype
            )
            current_hidden_states = F.linear(current_hidden_states, down_w)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class FP8Linear(nn.Module):
    """Drop-in fp8 replacement for the quantized dense ``nn.Linear``s.

    Holds the W8A8-FP8 weight exactly as the checkpoint stores it -- ``weight`` fp8 (e4m3)
    ``[out, in]`` + per-output-channel ``weight_scale`` bf16 ``[out, 1]`` -- and dequantizes on the
    fly in ``forward`` (weight-only; activations stay bf16). The 2D analogue of
    :class:`FP8DiffusionGemmaTextExperts`. Parameter names match the checkpoint
    (``weight`` / ``weight_scale``), so the dense linears load by name with no converter.

    Built *during model construction* (before ``post_init``), so ``weight_scale`` exists when the
    encoder<->decoder tie map is frozen -- the existing ``*scale`` tie rule then shares it with the
    decoder automatically, with no post-load re-tie.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # `requires_grad=False`; the loader re-wraps floating params with `requires_grad=True`
        # (harmless). Created on `device` (meta under the load context) to avoid real allocation.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        return cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight: (out, in) fp8, weight_scale: (out, 1) bf16 -> (out, in) in x's dtype.
        weight = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
        return F.linear(x, weight, self.bias)


def _expert_converters() -> list[WeightConverter]:
    """Merge per-expert fp8 weights/scales into the fused fp8 buffers.

    The per-expert weight ``[out, in]`` and per-channel scale ``[out, 1]`` stack over
    experts (dim 0) and, for gate/up, concat over the output dim (dim 1), landing directly
    in the fused buffers with no transpose. Harmless for the original (already-fused)
    google checkpoint: the per-expert source patterns simply never match.
    """
    return [
        WeightConverter(
            source_patterns=["experts.*.gate_proj.weight", "experts.*.up_proj.weight"],
            target_patterns="experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns=["experts.*.gate_proj.weight_scale", "experts.*.up_proj.weight_scale"],
            target_patterns="experts.gate_up_proj_scale",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns="experts.*.down_proj.weight",
            target_patterns="experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
        WeightConverter(
            source_patterns="experts.*.down_proj.weight_scale",
            target_patterns="experts.down_proj_scale",
            operations=[MergeModulelist(dim=0)],
        ),
    ]


_PATCHED = False


def patch_diffusion_gemma_fp8() -> None:
    """Make ``DiffusionGemmaForBlockDiffusion.from_pretrained`` load the FP8 checkpoint.

    Idempotent; call once before ``from_pretrained``. Both the expert and the dense
    ``weight_scale`` params are created in their modules' ``__init__`` (before ``post_init``), so
    the encoder's scales tie to the decoder via the existing ``*scale`` (and
    ``gate_up_proj``/``down_proj``) tie rules with no ``_tied_weights_keys`` changes and no re-tie.
    """
    global _PATCHED
    if _PATCHED:
        return

    # 1. Swap the experts module. The layers resolve `DiffusionGemmaTextExperts` from the
    #    modeling module's globals at construction time, so this takes effect for every
    #    newly built layer (and for the `build_device_map` skeleton).
    _mdg.DiffusionGemmaTextExperts = FP8DiffusionGemmaTextExperts

    # 2. Guard `_init_weights`: the swap above also rebinds the name used by the
    #    `isinstance(module, DiffusionGemmaTextExperts)` branch (modeling_diffusion_gemma.py
    #    :834), which would call `init.normal_` on the fp8 `gate_up_proj` buffer. Skip our
    #    experts there (their buffers come from the checkpoint).
    _orig_init_weights = _mdg.DiffusionGemmaPreTrainedModel._init_weights

    def _init_weights(self, module):
        if isinstance(module, FP8DiffusionGemmaTextExperts):
            return
        return _orig_init_weights(self, module)

    _mdg.DiffusionGemmaPreTrainedModel._init_weights = _init_weights

    # 3. Register the per-expert -> fused FP8 converter.
    register_checkpoint_conversion_mapping("diffusion_gemma", _expert_converters(), overwrite=True)

    # 4. Swap the quantized dense `nn.Linear`s for `FP8Linear`. The three text classes below hold
    #    *only* quantized linears -- the checkpoint's quant `ignore` list (router.proj,
    #    self_conditioning, embed_vision, lm_head, vision tower) lives in other classes -- so every
    #    `nn.Linear` child is swapped unconditionally; `v_proj` is `None` on full-attention layers
    #    and is skipped (it isn't a child). The swap runs inside each class's `__init__`, i.e. during
    #    model construction *before* `post_init`, so the new `weight_scale` params exist when the
    #    encoder->decoder tie map is frozen and the existing `*scale` tie rule shares them with the
    #    decoder automatically (no post-load re-tie needed).
    def _wrap_init_swap_linears(cls):
        _orig_cls_init = cls.__init__

        def _init(self, *a, **kw):
            _orig_cls_init(self, *a, **kw)
            for name, child in list(self.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(self, name, FP8Linear.from_linear(child))

        cls.__init__ = _init

    for _cls in (
        _mdg.DiffusionGemmaEncoderTextAttention,
        _mdg.DiffusionGemmaDecoderTextAttention,
        _mdg.DiffusionGemmaText4MLP,
    ):
        _wrap_init_swap_linears(_cls)

    # 5. Strip the compressed-tensors quantizer. We now hold every quantized weight ourselves
    #    (experts + dense, both fp8), so there is nothing for compressed-tensors to do. Setting
    #    `config.quantization_config = None` makes `get_hf_quantizer` return None
    #    (transformers/quantizers/auto.py) -> no COMPRESSED-status linears, no decompress, no
    #    `bf16 x fp8` crash. Without a quantizer the loader materializes each param at its own
    #    declared dtype (core_model_loading.py), so the fp8 `weight`s stay fp8 under `dtype="auto"`.
    _orig_from_pretrained = DiffusionGemmaForBlockDiffusion.from_pretrained

    def _from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            kwargs["config"] = config
        config.quantization_config = None
        return _orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    DiffusionGemmaForBlockDiffusion.from_pretrained = staticmethod(_from_pretrained)

    _PATCHED = True
