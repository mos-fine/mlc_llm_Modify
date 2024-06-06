"""
Implementation for Mistral architecture.
"""

import dataclasses
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

from .vit_model import ViT, ViTConfig  # 确保导入ViT相关类

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class MistralConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Mistral model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    position_embedding_base: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    sliding_window_size: int = 4096
    prefill_chunk_size: int = 0
    attention_sink_size: int = 4
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size
        assert self.attention_sink_size >= 0
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.sliding_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.sliding_window_size, 2048)


# pylint: disable=invalid-name,missing-docstring


class MistralMLP(nn.Module):
    """Same as in Llama architecture (LlamaFFN)."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


class MistralAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Same as LlamaAttention, but with sliding window attention using a rolling buffer cache."""

    def __init__(self, config: MistralConfig):
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        # QKV Projection
        qkv = self.qkv_proj(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        # Attention
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_q_heads),
            (b, s, h_q * d),
        )
        return self.o_proj(output)


class MistralDecoderLayer(nn.Module):
    """Exact same as LlamaDecoderLayer."""

    def __init__(self, config: MistralConfig):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = MistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.mlp.gate_up_proj, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0))
            _set(self.mlp.down_proj, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class MistralModel(nn.Module):
    """Exact same as LlamaModel."""

    def __init__(self, config: MistralConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = nn.Embedding("vocab_size", config.hidden_size)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MistralForCasualLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Same as LlamaForCausalLM, except for the use of sliding window attention."""

    def __init__(self, config: MistralConfig):
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, "vocab_size", bias=False)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.sliding_window_size = config.sliding_window_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class ResamplerAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config):
        self.hidden_size = config.output_dim
        self.head_dim = 128
        self.num_heads = self.hidden_size // self.head_dim // config.tensor_parallel_shards
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor,
    ):
        """Forward pass of MistralAttention, performing QKV."""
        d, h = self.head_dim, self.num_heads
        b, sq, _ = q.shape
        _, sk, _ = k.shape
        assert b == 1, "Only support batch size 1 at this moment."
        q = self.q_proj(q).reshape(b, sq, h, d)
        k = self.k_proj(k).reshape(b, sk, h, d)
        v = self.v_proj(v).reshape(b, sk, h, d)
        output = op_ext.attention(q, k, v, attention_mask)
        return self.out_proj(output)


class Resampler(nn.Module):
    def __init__(self, config: ViTConfig):
        self.num_queries = config.num_query
        self.embed_dim = config.output_dim
        self.kv_dim = config.hidden_size
        self.image_len = config.image_len

        self.pos_embed = nn.Parameter((self.num_queries, self.embed_dim))
        #self.pos_embed_k = nn.Parameter((self.image_len, self.embed_dim))
        self.pos_embed_k = nn.Parameter((448*448//14//14, self.embed_dim))
        self.query = nn.Parameter((self.num_queries, self.embed_dim))
        self.kv_proj = nn.Linear(self.kv_dim, self.embed_dim, bias=False)
        self.ln_q = nn.LayerNorm(self.embed_dim, config.norm_eps)
        self.ln_kv = nn.LayerNorm(self.embed_dim, config.norm_eps)
        self.attn = ResamplerAttention(config)
        self.ln_post = nn.LayerNorm(self.embed_dim, config.norm_eps)
        self.proj = nn.Parameter((self.embed_dim, self.embed_dim))
        self.dtype = 'float32'


    def forward(
        self, 
        x : Tensor,
        tgt_size,
    ):
        pos_embed = self.pos_embed_k.reshape(32, 32, self.embed_dim)
        indices_x = nn.core.wrap_nested(tvm.relax.op.arange(0, tgt_size[0], dtype="int32"), "arange")
        indices_y = nn.core.wrap_nested(tvm.relax.op.arange(0, tgt_size[1], dtype="int32"), "arange")
        pos_embed = op.take(pos_embed, indices_x, axis=0)
        pos_embed = op.take(pos_embed, indices_y, axis=1).reshape(tgt_size[0] * tgt_size[1], self.embed_dim)

        x = self.kv_proj(x)
        x = self.ln_kv(x)

        q = self.ln_q(self.query)

        def _attention_mask(
            batch_size, q_len, k_len,
        ):
            # See `tests/legacy-python/test_sliding_window_mask.py` for its behavior
            return te.compute(
                (batch_size, 1, q_len, k_len),
                lambda b, _, i, j: tir.max_value(self.dtype),
                name="_attention_mask",
            )

        attention_mask = op.tensor_expr_op(
            _attention_mask,
            name_hint="_attention_mask",
            args=[
                1,
                self.num_queries,
                self.image_len,
            ],
        )

        x = self.attn(
            (q + self.pos_embed).reshape(1, q.shape[0], q.shape[1]),
            x + pos_embed.reshape(1, pos_embed.shape[0], pos_embed.shape[1]),
            x,
            attention_mask
        )

        x = self.ln_post(x)

        x = op.matmul(x, self.proj)
        return x

class VisMiniCPM(nn.Module):
    def __init__(self, config: MistralConfig):
        vit_config = ViTConfig()
        self.llm = MistralForCasualLM(config)
        self.vpm = ViT(vit_config)
        self.resampler = Resampler(vit_config)
        self.dtype = "float32"

    def image(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        inputs = (inputs.astype(self.dtype) / 255. - 0.5) / 0.5
        shape = inputs.shape
        inputs = self.vpm(inputs)
        inputs = self.resampler(inputs, ((shape[-2] + 13) // 14, (shape[-1] + 13) // 14))
        return self.llm.prefill_embed(inputs, rolling_cache_len, kv_seq_len, cache_offset)

    def prefill(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        return self.llm.prefill(inputs, rolling_cache_len, kv_seq_len, cache_offset)

    def decode(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        return self.llm.decode(inputs, rolling_cache_len, kv_seq_len, cache_offset)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        """Softmax."""
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1
        image_size = 448
        mod_spec = {
            "image": {
                "inputs": nn.spec.Tensor([batch_size, 3, image_size, image_size], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
