"""Linearised Feature Trajectories (LiFT)."""
import os
import sys
from collections import defaultdict
import math

import torch
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import lightning as L

# from adapt4change.playground.layers import (
#     TransformerEncoderLayer,
#     TransformerEncoder,
# )
# from adapt4change.scripts.feat_utils import (
#     get_linear_probe_accuracy,
# )
# import shared.utils as su

import copy
from typing import Optional, Any, Union, Callable
import warnings

import torch
from torch.nn.modules.container import ModuleList
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, Dropout, Linear, LayerNorm, MultiheadAttention


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            need_weights: bool = False,
        ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ''
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            if need_weights:
                _x, attn_map = self._sa_block(
                    x=self.norm1(x),
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                )
            else:
                _x = self._sa_block(
                    x=self.norm1(x),
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False,
                )
            x  = x + _x
            x = x + self._ff_block(self.norm2(x))
        else:
            if need_weights:
                _x, attn_map = self._sa_block(
                    x=x,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                )
            else:
                _x = self._sa_block(
                    x=x,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False,
                )
            x = self.norm1(x + _x)
            x = self.norm2(x + self._ff_block(x))

        if need_weights:
            return x, attn_map
        else:
            return x

    # self-attention block
    def _sa_block(
        self, x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tensor:
        if need_weights:
            x, attn_map = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=True, is_causal=is_causal)
            return self.dropout1(x), attn_map
        else:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False, is_causal=is_causal)[0]
            return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None,
            need_weights: bool = False,
        ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            need_weights: output attn_weights of all layers. Default: ``False``.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        attn_maps = []
        mod = self.layers[0]
        x = mod(
            src=output,
            need_weights=True,
        )
        for mod in self.layers:
            if need_weights:
                output, attn_map = mod(
                    output,
                    src_mask=mask,
                    is_causal=is_causal,
                    src_key_padding_mask=src_key_padding_mask_for_layers,
                    need_weights=True,
                )
                attn_maps.append(attn_map)
            else:
                output = mod(
                    output,
                    src_mask=mask,
                    is_causal=is_causal,
                    src_key_padding_mask=src_key_padding_mask_for_layers,
                    need_weights=False,
                )

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            return output, attn_maps
        else:
            return output



class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input
    """
    def __init__(self, d_model, max_len=16):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * \
                (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        t = x.size(1)
        if t == self.max_len:
            return x + self.pe
        elif t > self.max_len:
            # Need to interpolate
            pe = self.pe
            pe = F.interpolate(pe, size=(t,), mode='linear', align_corners=False)
            return x + pe
        else:
            # Need to truncate
            pe = self.pe[:, :t]
            return x + pe


class DualCLSTransformer(nn.Module):
    """
    Transformer encoder with two CLS tokens: one for static and one for dynamic information
    """
    def __init__(self, feature_dim, hidden_dim, nhead=8, num_layers=4, dropout=0.1, max_len=16):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        
        # CLS tokens (static and dynamic)
        self.cls_static = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cls_dynamic = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        
        # Output projections for latent codes
        self.static_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization for outputs
        self.static_norm = nn.LayerNorm(hidden_dim)
        self.dynamic_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, need_weights=False):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, feature_dim]
            need_weights: Whether to return attention weights
        Returns:
            z_s: Static latent code [batch_size, hidden_dim]
            z_d: Dynamic latent code [batch_size, hidden_dim]
        """
        # TODO: add sinusoidal position encoding to represent time
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Expand CLS tokens to batch size
        cls_s = self.cls_static.expand(batch_size, -1, -1)
        cls_d = self.cls_dynamic.expand(batch_size, -1, -1)
        
        # Concatenate CLS tokens with input sequence
        # [batch_size, 2+seq_len, hidden_dim]
        x_with_cls = torch.cat([cls_s, cls_d, x], dim=1)
        
        # Pass through transformer
        if need_weights:
            output, attn_maps = self.transformer(x_with_cls, need_weights=True)
        else:
            output = self.transformer(x_with_cls, need_weights=False)
        
        # Extract and process the CLS token outputs
        z_s = self.static_norm(self.static_proj(output[:, 0]))
        z_d = self.dynamic_norm(self.dynamic_proj(output[:, 1]))
        
        if not need_weights:
            return z_s, z_d
        else:
            return z_s, z_d, attn_maps


class DualCLSTransformerWithoutProjection(nn.Module):
    """
    Transformer encoder with two CLS tokens: one for static and one for dynamic information
    """
    def __init__(self, feature_dim, hidden_dim=None, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        # Positional encoding
        hidden_dim = feature_dim
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # CLS tokens (static and dynamic)
        self.cls_static = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cls_dynamic = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Layer normalization for outputs
        self.static_norm = nn.LayerNorm(hidden_dim)
        self.dynamic_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, need_weights=False):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, feature_dim]
            need_weights: Whether to return attention weights
        Returns:
            z_s: Static latent code [batch_size, hidden_dim]
            z_d: Dynamic latent code [batch_size, hidden_dim]
        """
        # TODO: add sinusoidal position encoding to represent time
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        x = self.pos_encoder(x)

        # Expand CLS tokens to batch size
        cls_s = self.cls_static.expand(batch_size, -1, -1)
        cls_d = self.cls_dynamic.expand(batch_size, -1, -1)

        # Concatenate CLS tokens with input sequence
        # [batch_size, 2+seq_len, hidden_dim]
        x_with_cls = torch.cat([cls_s, cls_d, x], dim=1)

        # Pass through transformer
        if need_weights:
            output, attn_maps = self.transformer(x_with_cls, need_weights=True)
        else:
            output = self.transformer(x_with_cls, need_weights=False)
        
        # Extract and process the CLS token outputs
        z_s = self.static_norm(output[:, 0])
        z_d = self.dynamic_norm(output[:, 1])

        if not need_weights:
            return z_s, z_d
        else:
            return z_s, z_d, attn_maps


class Decoder(nn.Module):
    """
    Simple decoder that reconstructs features from the latent representation
    """
    def __init__(self, latent_dim, feature_dim, hidden_dim=512):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent vector at time t [batch_size, latent_dim]
        Returns:
            x_hat: Reconstructed feature [batch_size, feature_dim]
        """
        return self.model(z)


class LinearisedFeatureTrajectories(nn.Module):
    """
    Complete model for video linearization
    """
    def __init__(
        self,
        encoder="DualCLSTransformer",
        feature_dim=384,
        latent_dim=256,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_len=16,
    ):
        super().__init__()
        
        self.encoder = eval(encoder)(
            feature_dim=feature_dim,
            hidden_dim=latent_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.feature_dim = feature_dim
        self.embed_dim = latent_dim * 2

        self.decoder = Decoder(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            hidden_dim=latent_dim * 2
        )
    
    def encode(self, x, need_weights=False, cat=True):
        """
        Encode sequence into static and dynamic components
        
        Args:
            x: Sequence of features [batch_size, seq_len, feature_dim]
            need_weights: Whether to return attention weights
        Returns:
            z_s: Static component [batch_size, latent_dim]
            z_d: Dynamic component [batch_size, latent_dim]
        """
        z_s, z_d = self.encoder(x, need_weights=need_weights)
        if cat:
            z = torch.cat([z_s, z_d], dim=-1)
            return z
        else:
            return z_s, z_d
    
    def compute_latents(self, x):
        """
        Compute static and dynamic latents
        
        Args:
            x: Sequence of features [batch_size, seq_len, feature_dim]
        Returns:
            z_s: Static component [batch_size, latent_dim]
            z_d: Dynamic component [batch_size, latent_dim]
        """
        z_s, z_d = self.encode(x, need_weights=False)

        # Concatenate static and dynamic components
        z = torch.cat([z_s, z_d], dim=-1)

        return z_s, z_d, z
    
    def interpolate(self, z_s, z_d, t, T=1.0):
        """
        Interpolate in latent space using the linear model
        
        Args:
            z_s: Static component [batch_size, latent_dim]
            z_d: Dynamic component [batch_size, latent_dim]
            t: Normalized time value or tensor of time values [batch_size]
            T: Total sequence length for normalization (default=1.0)
        Returns:
            z_t: Interpolated latent [batch_size, latent_dim]
        """
        # Handle scalar t
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=z_s.device)
        
        # Handle scalar t with batched inputs
        if t.dim() == 0 and z_s.dim() > 1:
            t = t.expand(z_s.size(0))
        
        # Normalize t and reshape for broadcasting
        t_norm = (t / T).view(-1, 1)
        
        # Linear interpolation
        z_t = z_s + t_norm * z_d
        
        return z_t
    
    def decode(self, z_t):
        """
        Decode latent vector to feature
        
        Args:
            z_t: Latent vector at time t [batch_size, latent_dim]
        Returns:
            x_hat: Reconstructed feature [batch_size, feature_dim]
        """
        return self.decoder(z_t)
    
    def forward(self, x, times=None, need_weights=False):
        """
        Full forward pass
        
        Args:
            x: Sequence of features [batch_size, seq_len, feature_dim]
            times: Optional specific time points to reconstruct.
                   If None, reconstructs all input frames.
            need_weights: Whether to return attention weights
        Returns:
            x_hat: Reconstructed features
            z_s: Static latent code
            z_d: Dynamic latent code
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Encode sequence
        if need_weights:
            z_s, z_d, attn_maps = self.encode(x, need_weights=True)
        else:
            z_s, z_d = self.encode(x, need_weights=False)
        
        # Determine which time points to reconstruct
        if times is None:
            times = torch.arange(seq_len, device=x.device).float()
            times = times.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
            times_flat = times.flatten()  # [batch_size * seq_len]
            
            # Repeat z_s and z_d for each time point
            z_s_rep = z_s.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, z_s.size(-1))
            z_d_rep = z_d.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, z_d.size(-1))
            
            # Interpolate and decode
            z_t = self.interpolate(z_s_rep, z_d_rep, times_flat, T=seq_len-1)
            x_hat = self.decode(z_t)
            
            # Reshape output
            x_hat = x_hat.view(batch_size, seq_len, feature_dim)
        else:
            # Handle custom time points
            if not torch.is_tensor(times):
                times = torch.tensor(times, device=x.device).float()
            
            # Ensure times has the right shape
            if times.dim() == 1:
                times = times.unsqueeze(0).expand(batch_size, -1)
            
            times_flat = times.flatten()
            num_times = times.size(1)
            
            # Repeat z_s and z_d for each time point
            z_s_rep = z_s.unsqueeze(1).expand(-1, num_times, -1).reshape(-1, z_s.size(-1))
            z_d_rep = z_d.unsqueeze(1).expand(-1, num_times, -1).reshape(-1, z_d.size(-1))
            
            # Interpolate and decode
            z_t = self.interpolate(z_s_rep, z_d_rep, times_flat, T=seq_len-1)
            x_hat = self.decode(z_t)
            
            # Reshape output
            x_hat = x_hat.view(batch_size, num_times, feature_dim)
        
        if need_weights:
            return x_hat, z_s, z_d, attn_maps
        else:
            return x_hat, z_s, z_d


# class SimplifiedLoss(nn.Module):
#     """
#     Simplified loss function with just reconstruction and orthogonality losses
#     """
#     def __init__(self, recon_weight=1.0, ortho_weight=0.1, fotd_weight=0.):
#         super().__init__()
#         self.recon_weight = recon_weight
#         self.ortho_weight = ortho_weight
#         self.fotd_weight = fotd_weight
#         print("Loss weights:")
#         print(f"  Reconstruction: {self.recon_weight}")
#         print(f"  Orthogonality: {self.ortho_weight}")
#         print(f"  First-order temporal difference: {self.fotd_weight}")
        
#     def reconstruction_loss(self, x_hat, x):
#         """Mean squared error reconstruction loss"""
#         return F.mse_loss(x_hat, x)
    
#     def first_order_temporal_difference_loss(self, x_hat, x):
#         """
#         Applies a first-order temporal difference loss.

#             delta_X = X[1:, :] - X[:-1, :] (Shape: (T-1) x D)
#             delta_X_hat = X_hat[1:, :] - X_hat[:-1, :] (Shape: (T-1) x D)
#             loss = MSE(delta_X_hat, delta_X)
#         """
#         delta_x = x[1:, :] - x[:-1, :]
#         delta_x_hat = x_hat[1:, :] - x_hat[:-1, :]
#         return F.mse_loss(delta_x_hat, delta_x)
    
#     def orthogonality_loss(self, z_s, z_d):
#         """Normalized dot product between static and dynamic components"""
#         z_s_norm = F.normalize(z_s, dim=1)
#         z_d_norm = F.normalize(z_d, dim=1)
#         cos_sim = torch.abs(torch.sum(z_s_norm * z_d_norm, dim=1)).mean()
#         return cos_sim
    
#     def forward(self, x, x_hat, z_s, z_d):
#         """
#         Calculate the combined loss
        
#         Args:
#             x: Original features [batch_size, seq_len, feature_dim]
#             x_hat: Reconstructed features [batch_size, seq_len, feature_dim]
#             z_s: Static latent code [batch_size, latent_dim]
#             z_d: Dynamic latent code [batch_size, latent_dim]
#         Returns:
#             total_loss: Combined weighted loss
#             loss_dict: Dictionary with individual loss components
#         """
#         recon_loss = self.reconstruction_loss(x_hat, x)
#         ortho_loss = self.orthogonality_loss(z_s, z_d)
#         fotd_loss = self.first_order_temporal_difference_loss(x_hat, x)
        
#         # Combine losses
#         total_loss = self.recon_weight * recon_loss + \
#             self.ortho_weight * ortho_loss + \
#             self.fotd_weight * fotd_loss
        
#         # Create loss dictionary for logging
#         loss_dict = {
#             'total': total_loss.item(),
#             'recon': recon_loss.item(),
#             'ortho': ortho_loss.item(),
#             'fotd': fotd_loss.item(),
#         }
        
#         return total_loss, loss_dict


# class LiFTLightningModule(L.LightningModule):
#     def __init__(
#             self,
#             model,
#             opt_name="adam",
#             lr=1e-4,
#             sched_name="plateau",
#             loss_weights=dict(
#                 recon_weight=1.0,
#                 ortho_weight=0.1,
#                 fotd_weight=0.,
#             ),
#             show_traj=True,
#             no_wandb=False,
#             loss="SimplifiedLoss",
#         ):
#         super().__init__()

#         self.model = model
#         self.opt_name = opt_name
#         self.sched_name = sched_name
#         self.lr = lr
#         self.criterion = eval(loss)(
#             **loss_weights,
#         )
#         self.show_traj = show_traj
#         self.no_wandb = no_wandb
    
#         # To store outputs
#         # NOTE: no need to store train outputs since the 
#         # train and eval for linear probe is all in the 
#         # validation set itself.
#         # self.train_step_outputs = defaultdict(list)
#         self.valid_step_outputs = defaultdict(list)

#     def configure_optimizers(self):

#         # Tried other optimizers like SGD, but Adam works best
#         if self.opt_name == "adam":
#             optimizer = torch.optim.Adam(
#                 self.model.parameters(), lr=self.lr, weight_decay=1e-5,
#             )
#         else:
#             raise ValueError(f"Unknown optimizer: {self.opt_name}")

#         if self.sched_name != "none":
        
#             # Learning rate scheduler
#             # Tried other schedulers like CosineAnnealingLR, but
#             # ReduceLROnPlateau works best
#             if self.sched_name == "plateau":
#                 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                     optimizer, mode='min', factor=0.5, patience=5, verbose=True,
#                 )
#             else:
#                 raise ValueError(f"Unknown scheduler: {self.sched_name}")

#             return {
#                 "optimizer": optimizer,
#                 "lr_scheduler": scheduler,
#                 "monitor": "valid/loss",
#             }

#         elif self.sched_name == "none":
#             return optimizer

#     def process_batch(self, batch, return_data=False):

#         # Get inputs
#         x = batch['features']

#         # Forward pass
#         x_hat, z_s, z_d = self.model(x)

#         # Calculate loss
#         loss, loss_dict = self.criterion(x, x_hat, z_s, z_d)

#         # Return outputs
#         output = dict(
#             latents=torch.cat([z_s, z_d], dim=-1),
#             loss=loss,
#             chiral_triplet_ids=batch["chiral_triplet_id"],
#             chiral_labels=batch["chiral_label"],
#             linprobe_split=batch["linprobe_split"],
#             **loss_dict,
#         )

#         # If return_data is True, return the data as well
#         if return_data:
#             output["original"] = x
#             output["reconstructed"] = x_hat

#         return output

#     def training_step(self, batch, batch_idx):
#         outputs = self.process_batch(batch)
#         loss = outputs['loss']

#         # Log loss
#         self.log(f"train/loss", loss, prog_bar=True)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         outputs = self.process_batch(batch, return_data=(batch_idx == 0))
#         loss = outputs['loss']

#         # Log loss
#         self.log(f"valid/loss", outputs['loss'], prog_bar=True)

#         # Log visualisation of trajectories for few videos
#         # only for the first batch
#         if (batch_idx == 0) and self.show_traj:
#             from adapt4change.playground.expt_utils import (
#                 show_feature_trajectories,
#             )
#             n_show = 4
#             images = [
#                 show_feature_trajectories(
#                     video_path=batch['video_path'][i],
#                     x=outputs["original"][i].detach().cpu().numpy(),
#                     x_hat=outputs["reconstructed"][i].detach().cpu().numpy(),
#                 ) for i in range(n_show)
#             ]
#             canvas = su.visualize.concat_images_with_border(images)

#             # Log image
#             if not self.no_wandb:
#                 import wandb
#                 self.logger.experiment.log(
#                     {
#                         "valid/feature_trajectories": wandb.Image(canvas),
#                     },
#             )
                
#             del outputs["original"]
#             del outputs["reconstructed"]

#         # Add to validation outputs
#         del outputs["loss"]
#         del outputs['total']
#         del outputs["recon"]
#         del outputs["ortho"]
#         del outputs["fotd"]
#         for k, v in outputs.items():
#             if isinstance(v, torch.Tensor):
#                 outputs[k] = v.detach().cpu().numpy()
#             self.valid_step_outputs[k].append(outputs[k])

#         return loss

#     def on_validation_epoch_end(self):

#         # Concatenate all outputs
#         valid_step_outputs_all = {
#             k: np.concatenate(v, axis=0) for k, v in self.valid_step_outputs.items()
#         }

#         # Get train and eval indices
#         split_info = np.array(valid_step_outputs_all['linprobe_split'])
#         train_indices = np.where(split_info == "train")[0]
#         valid_indices = np.where(split_info == "validation")[0]

#         train_step_outputs_all = {
#             k: v[train_indices] for k, v in valid_step_outputs_all.items()
#         }
#         valid_step_outputs_all = {
#             k: v[valid_indices] for k, v in valid_step_outputs_all.items()
#         }

#         # Run linear probe on each triplet subset
#         triplets_train = train_step_outputs_all["chiral_triplet_ids"]
#         triplets_valid = valid_step_outputs_all["chiral_triplet_ids"]
#         triplets_common = np.intersect1d(
#             np.unique(triplets_train), np.unique(triplets_valid)
#         )
#         val_accs = []
#         for tid in triplets_common:
#             idx_train = np.where(triplets_train == tid)[0]
#             idx_valid = np.where(triplets_valid == tid)[0]
#             Z_train = train_step_outputs_all["latents"][idx_train]
#             Z_valid = valid_step_outputs_all["latents"][idx_valid]
#             Y_train = train_step_outputs_all["chiral_labels"][idx_train]
#             Y_valid = valid_step_outputs_all["chiral_labels"][idx_valid]
#             val_acc = get_linear_probe_accuracy(
#                 Z_train, Y_train, Z_valid, Y_valid, verbose=False,
#             )
#             val_accs.append(val_acc)
#         val_acc = np.mean(val_accs)
#         self.log(
#             "valid/linear_probe_acc", val_acc,
#             prog_bar=True,
#             on_step=False,
#             on_epoch=True,
#         )

#         # Free up memory
#         del self.valid_step_outputs
#         self.valid_step_outputs = defaultdict(list)


def load_checkpoint(
        litmodule,
        ckpt_name="ggwirp95/checkpoints/epoch=458-step=834003.ckpt",
        ckpt_root="/work/piyush/experiments/TimeBound.v1/time-antonyms/",
    ):
        """
        Load a checkpoint into a LightningModule.
        """
        ckpt_path = os.path.join(ckpt_root, ckpt_name)
        assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}."
        print("::: Loading checkpoint from: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        msg = litmodule.load_state_dict(ckpt["state_dict"])
        print(msg)
        return litmodule


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_pretrained_model(
        ckpt_name="ggwirp95/checkpoints/epoch=458-step=834003.ckpt",
        latent_dim=384,
        feature_dim=384,
        hidden_dim=128,
        load_pretrained=True,
        max_len=16,
        device=None,
    ):
    """
    Get video embedding from a sequence compressor model.

    Args:
        ckpt_name: str
            (sub) Path to the checkpoint file.
        latent_dim: int
            Dimension of the latent space.
    """

    args = dict(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=1e-3,
    )
    args = AttrDict(args)

    model = LinearisedFeatureTrajectories(
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_len=getattr(args, "max_len", max_len),
    )
    num_trainable_params(model)
    import ipdb; ipdb.set_trace()

    # Initialise Lightning Module
    litmodule = LiFTLightningModule(
        model=model, lr=args.lr, no_wandb=True,
    )

    # Load pre-trained weights
    if load_pretrained:
        ckpt_root = "/work/piyush/experiments/TimeBound.v1/time-antonyms/"
        ckpt_path = os.path.join(ckpt_root, ckpt_name)
        assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}."
        print("::: Loading checkpoint from: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        msg = litmodule.load_state_dict(ckpt["state_dict"])
        print(msg)

    # Port to GPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    litmodule = litmodule.to(device).eval()

    return litmodule


from tqdm import tqdm
def tqdm_iterator(items, desc=None, bar_format=None, **kwargs):
    tqdm._instances.clear()
    iterator = tqdm(
        items,
        desc=desc,
        # bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        **kwargs,
    )
    tqdm._instances.clear()
    return iterator


# def gather_latents(litmodule, X, batch_size=1024, reconstruct=False, verbose=True, device=None, to_cpu=True):
#     """Helper to run feature computation for litmodule."""
#     if device is None:
#         device = next(litmodule.model.parameters()).device
#     else:
#         litmodule = litmodule.to(device).eval()
#     start_indices = np.arange(0, len(X), batch_size)
#     if verbose:
#         iterator = tqdm_iterator(start_indices, desc="Gathering features")
#     else:
#         iterator = start_indices
#     latents = {
#         "static": [],
#         "dynamic": [],
#         "concat": [],
#     }
#     if reconstruct:
#         latents["reconstructed"] = []
#     for si in iterator:
#         ei = min(si + batch_size, len(X))
#         with torch.no_grad():

#             # Reconstruction as well as latents
#             x_hat, zs, zd = litmodule.model(X[si:ei].to(device))
#             z = torch.cat([zs, zd], dim=-1)

#             # # Only latents
#             # zs, zd, z = litmodule.model.compute_latents(X[si:ei].to(device))

#             if to_cpu:
#                 zs = zs.cpu()
#                 zd = zd.cpu()
#                 z = z.cpu()
#                 x_hat = x_hat.cpu()

#         latents["static"].append(zs)
#         latents["dynamic"].append(zd)
#         latents["concat"].append(z)
#         if reconstruct:
#             latents["reconstructed"].append(x_hat)
#     latents = {k: torch.cat(v) for k, v in latents.items()}
#     return latents


def get_terminal_width():
    import shutil
    return shutil.get_terminal_size().columns


def print_update(update, fillchar=".", color="yellow", pos="left", **kwargs):
    from termcolor import colored
    # add ::: to the beginning and end of the update s.t. the total length of the
    # update spans the whole terminal
    try:
        terminal_width = get_terminal_width()
    except:
        terminal_width = 98
    if pos == "center":
        update = update.center(len(update) + 2, " ")
        update = update.center(terminal_width, fillchar)
    elif pos == "left":
        update = update.ljust(terminal_width, fillchar)
        update = update.ljust(len(update) + 2, " ")
    elif pos == "right":
        update = update.rjust(terminal_width, fillchar)
        update = update.rjust(len(update) + 2, " ")
    else:
        raise ValueError("pos must be one of 'center', 'left', 'right'")
    print(colored(update, color, **kwargs))


def num_trainable_params(model, round=3, verbose=True, return_count=False):
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    model_name = model.__class__.__name__
    if round is not None:
        value = np.round(n_params / 1e6, round)
        unit = "M"
    else:
        value = n_params
        unit = ""
    if verbose:
        print(f"::: Number of trainable parameters in {model_name}: {value} {unit}")
    if return_count:
        return n_params


def num_params(model, round=3):
    n_params = sum([p.numel() for p in model.parameters()])
    model_name = model.__class__.__name__
    if round is not None:
        value = np.round(n_params / 1e6, round)
        unit = "M"
    else:
        value = n_params
        unit = ""
    print(f"::: Number of total parameters in {model_name}: {value}{unit}")


def prepare_model_lift(
    args=None,
    ckpt_path="/work/piyush/experiments/TimeBound.v1/time-antonyms/"\
        "ggwirp95/checkpoints/epoch=458-step=834003.ckpt",
):
    print_update("Building LiFT model", color="cyan")
    if args is None:
        # Default setting
        args = AttrDict(
            feature_dim=384,
            latent_dim=384,
            hidden_dim=128,
            max_len=5000,
            lr=1e-3,
            encoder="DualCLSTransformer",
        )
    print("NOTE: Using max_len from args", getattr(args, "max_len", 16))
    model = LinearisedFeatureTrajectories(
        encoder=args.encoder,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_len=getattr(args, "max_len", 16),
    )
    num_trainable_params(model)
    
    # Load checkpoint
    assert os.path.exists(ckpt_path), \
        f"Checkpoint not found at {ckpt_path}."
    print("::: Loading checkpoint from: ", ckpt_path)
    ckpt = torch.load(ckpt_path, weights_only=True)
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for k in state_dict.keys():
        if k.startswith("model."):
            new_k = k[6:]
            new_state_dict[new_k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    msg = model.load_state_dict(new_state_dict)
    print(msg)
    print_update(".", color="cyan")

    return model


if __name__ == "__main__":
    # Load the LIFT model
    lift = prepare_model_lift()
    
    # Encode a random sample
    x = torch.randn(4, 16, 384)
    z = lift.encode(x)
    print("Input shape:", x.shape)
    print("Output shape:", z.shape)
