# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    has_cross_attention: bool = False
    context_dim: int = 0
    uvit_skip_connection: bool = False
    time_as_token: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        # self.head_dim = self.dim // self.n_head

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, use_kv_cache=False):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.head_dim,
                                              self.config.rope_base, dtype).to(device)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).to(device)
        self.use_kv_cache = use_kv_cache
        self.uvit_skip_connection = self.config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.layers_emit_skip = [i for i in range(self.config.n_layer) if i < self.config.n_layer // 2]
            self.layers_receive_skip = [i for i in range(self.config.n_layer) if i > self.config.n_layer // 2]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def forward(self,
                x: Tensor,
                c: Tensor,
                input_pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                context: Optional[Tensor] = None,
                context_input_pos: Optional[Tensor] = None,
                cross_attention_mask: Optional[Tensor] = None,
                ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if mask is None: # in case of non-causal model
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        if context is not None:
            context_freqs_cis = self.freqs_cis[context_input_pos]
        else:
            context_freqs_cis = None
        skip_in_x_list = []
        for i, layer in enumerate(self.layers):
            if self.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None
            x = layer(x, c, input_pos, freqs_cis, mask, context, context_freqs_cis, cross_attention_mask, skip_in_x)
            if self.uvit_skip_connection and i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        x = self.norm(x, c)
        return x

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        if config.has_cross_attention:
            self.has_cross_attention = True
            self.cross_attention = Attention(config, is_cross_attention=True)
            self.cross_attention_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        else:
            self.has_cross_attention = False

        if config.uvit_skip_connection:
            self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)
            self.uvit_skip_connection = True
        else:
            self.uvit_skip_connection = False

        self.time_as_token = config.time_as_token

    def forward(self,
                x: Tensor,
                c: Tensor,
                input_pos: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                context: Optional[Tensor] = None,
                context_freqs_cis: Optional[Tensor] = None,
                cross_attention_mask: Optional[Tensor] = None,
                skip_in_x: Optional[Tensor] = None,
                ) -> Tensor:
        c = None if self.time_as_token else c
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask, input_pos)
        if self.has_cross_attention:
            h = h + self.cross_attention(self.cross_attention_norm(h, c), freqs_cis, cross_attention_mask, input_pos, context, context_freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h, c))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, is_cross_attention: bool = False):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        if is_cross_attention:
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wkv = nn.Linear(config.context_dim, 2 * config.n_local_heads * config.head_dim, bias=False)
        else:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        # self._register_load_state_dict_pre_hook(self.load_hook)

    # def load_hook(self, state_dict, prefix, *args):
    #     if prefix + "wq.weight" in state_dict:
    #         wq = state_dict.pop(prefix + "wq.weight")
    #         wk = state_dict.pop(prefix + "wk.weight")
    #         wv = state_dict.pop(prefix + "wv.weight")
    #         state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self,
                x: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                input_pos: Optional[Tensor] = None,
                context: Optional[Tensor] = None,
                context_freqs_cis: Optional[Tensor] = None,
                ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if context is None:
            q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
            context_seqlen = seqlen
        else:
            q = self.wq(x)
            k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
            context_seqlen = context.shape[1]

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, context_freqs_cis if context_freqs_cis is not None else freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
        seq_len: int, n_elem: int, base: int = 10000,
        dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
