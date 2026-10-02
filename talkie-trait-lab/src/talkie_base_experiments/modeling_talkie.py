"""Logit-compatible Transformers implementation of the original Talkie model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_talkie import TalkieConfig


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"RoPE expects rank 4, got {tuple(x.shape)}")
    half = x.shape[3] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones([n_head]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = self.head_g
        local_heads = x.shape[-2]
        if local_heads != gain.numel():
            if gain.numel() % local_heads != 0:
                raise ValueError(
                    f"Cannot shard {gain.numel()} head gains over {local_heads} heads"
                )
            if not torch.distributed.is_initialized():
                raise RuntimeError("Local head gains require initialized distributed state")
            partitions = gain.numel() // local_heads
            partition = torch.distributed.get_rank() % partitions
            gain = gain.narrow(0, partition * local_heads, local_heads)
        return x * gain.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * self.w_g.type_as(weight)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TalkieConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.is_causal = True
        hidden = config.n_embd
        self.attn_query = nn.Linear(hidden, hidden, bias=False)
        self.attn_key = nn.Linear(hidden, hidden, bias=False)
        self.attn_value = nn.Linear(hidden, hidden, bias=False)
        self.attn_resid = nn.Linear(hidden, hidden, bias=False)
        self.head_gain = HeadGain(config.n_head)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.size()
        q, k, v = self.attn_query(x), self.attn_key(x), self.attn_value(x)
        local_width = q.shape[-1]
        if local_width % self.head_dim != 0:
            raise ValueError(
                f"Attention width {local_width} is not divisible by head dim {self.head_dim}"
            )
        local_heads = local_width // self.head_dim
        q = q.view(batch_size, sequence_length, local_heads, self.head_dim)
        k = k.view(batch_size, sequence_length, local_heads, self.head_dim)
        v = v.view(batch_size, sequence_length, local_heads, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        past_length = (
            0
            if past_key_values is None
            else past_key_values.get_seq_length(self.layer_idx)
        )
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        implementation = getattr(self.config, "_attn_implementation", "sdpa")
        if implementation == "sdpa":
            # Keep the ordinary unpadded path identical to upstream Talkie.
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=attention_mask is None and past_length == 0,
            )
            y = y.transpose(1, 2).contiguous()
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[implementation]
            y, _ = attention_interface(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
                **kwargs,
            )
        y = y.reshape(batch_size, sequence_length, -1)
        return self.attn_resid(y)


class MLP(nn.Module):
    def __init__(self, config: TalkieConfig):
        super().__init__()
        hidden = config.n_embd
        intermediate = int(round(((8 / 3) * hidden) / 128) * 128)
        self.mlp_gate = nn.Linear(hidden, intermediate, bias=False)
        self.mlp_linear = nn.Linear(hidden, intermediate, bias=False)
        self.mlp_resid = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.mlp_gate(x)) * self.mlp_linear(x)
        return self.mlp_resid(x)


class Block(nn.Module):
    def __init__(self, config: TalkieConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.attn_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.mlp = MLP(config)
        self.mlp_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def forward(
        self,
        embedded: torch.Tensor,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        normalized = F.rms_norm(hidden_states, (hidden_states.shape[-1],))
        hidden_states = hidden_states + self.attn_gain(
            self.attn(
                normalized,
                cos_sin,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        )
        hidden_states = hidden_states + self.mlp_gain(
            self.mlp(F.rms_norm(hidden_states, (hidden_states.shape[-1],)))
        )
        return hidden_states + self.embed_skip(embedded)


class TalkieForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TalkieConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True
    _skip_keys_device_placement = ["past_key_values"]
    _no_split_modules = ["Block"]
    _keys_to_ignore_on_load_unexpected = [r"model\.lm_head\.weight"]

    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        self.lm_head = nn.Parameter(torch.zeros(config.vocab_size, config.n_embd))
        self.lm_head_gain = WeightGain()
        cos, sin = self._precompute_rotary_embeddings(
            config.max_position_embeddings, config.head_dim, config.rope_theta
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _precompute_rotary_embeddings(
        self, sequence_length: int, head_dim: int, base: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        if device.type == "meta":
            device = torch.device("cpu")
        channels = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channels / head_dim))
        positions = torch.arange(sequence_length, dtype=torch.float32, device=device)
        frequencies = torch.outer(positions, inv_freq)
        cos, sin = frequencies.cos().bfloat16(), frequencies.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed = value

    def get_output_embeddings(self) -> nn.Module | None:
        # Talkie's output head is a raw parameter, not an nn.Module. Returning
        # None keeps generic tooling (notably FSDP2's tail wrapping) from
        # treating that parameter as a module; the root FSDP unit still shards it.
        return None

    def save_pretrained(self, save_directory: str, *args: object, **kwargs: object) -> None:
        """Save a vLLM-compatible alias for the external language-model head.

        The Transformers backend in vLLM owns its output head outside the custom
        backbone and looks for this conventional key.  Keeping the original
        ``lm_head`` key preserves exact compatibility with Talkie checkpoints.
        """
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
        if "lm_head" in state_dict and "model.lm_head.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["model.lm_head.weight"] = state_dict["lm_head"].clone()
        super().save_pretrained(
            save_directory,
            *args,
            state_dict=state_dict,
            **kwargs,
        )

    def _forward_hidden(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        use_cache: bool | None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, Cache | None, bool]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        assert source is not None
        batch_size, sequence_length = source.shape[:2]
        device = source.device

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
                raise ValueError("attention_mask must have shape [batch, sequence]")
            if attention_mask.shape[1] != past_length + sequence_length:
                raise ValueError(
                    "attention_mask length must equal cached plus current sequence length"
                )
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(
                    past_length,
                    past_length + sequence_length,
                    device=device,
                ).unsqueeze(0)
            else:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(~attention_mask, 0)
                position_ids = position_ids[:, -sequence_length:]
        elif position_ids.shape[-1] != sequence_length:
            position_ids = position_ids[:, -sequence_length:]
        position_ids = position_ids.to(device=device, dtype=torch.long)

        rotary_cos, rotary_sin = self.cos, self.sin
        if rotary_cos.device.type == "meta" or rotary_cos.device != device:
            rotary_cos, rotary_sin = self._precompute_rotary_embeddings(
                self.config.max_position_embeddings,
                self.config.head_dim,
                self.config.rope_theta,
            )
            # vLLM may construct custom Transformers models with non-persistent
            # buffers left on CPU/meta. Keep the fallback local: mutating module
            # buffers in forward is rejected by vLLM's compiled CUDA-graph path.
            rotary_cos = rotary_cos.to(device)
            rotary_sin = rotary_sin.to(device)
        # Keep the bounds checks on-device. Calling Tensor.item() here creates a
        # data-dependent graph break that prevents vLLM's torch.compile path.
        torch._assert_async(
            torch.all(position_ids >= 0),
            "position_ids cannot contain negative positions",
        )
        torch._assert_async(
            torch.all(position_ids < rotary_cos.shape[1]),
            "position_ids exceed max_position_embeddings",
        )
        cos_sin = rotary_cos[0][position_ids], rotary_sin[0][position_ids]
        implementation = getattr(self.config, "_attn_implementation", "sdpa")
        prepared_mask = attention_mask
        if implementation == "sdpa":
            prepared_mask = self._prepare_sdpa_mask(
                attention_mask,
                position_ids,
                batch_size,
                sequence_length,
                past_length,
                device,
            )

        hidden_states = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],))
        embedded = hidden_states
        for block in self.blocks:
            hidden_states = block(
                embedded,
                hidden_states,
                cos_sin,
                attention_mask=prepared_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],))
        return hidden_states, past_key_values, use_cache

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        hidden_states, past_key_values, use_cache = self._forward_hidden(
            input_ids,
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            **kwargs,
        )
        selected = hidden_states if logits_to_keep == 0 else hidden_states[:, -logits_to_keep:, :]
        logits = F.linear(selected, self.lm_head_gain(self.lm_head)).float()
        loss = None
        if labels is not None:
            if logits_to_keep != 0:
                raise ValueError("labels require logits_to_keep=0")
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
            )
        if return_dict is False:
            values = (logits, past_key_values) if use_cache else (logits,)
            return ((loss,) + values if loss is not None else values)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
        )

    @staticmethod
    def _prepare_sdpa_mask(
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        batch_size: int,
        sequence_length: int,
        past_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        total_length = past_length + sequence_length
        has_padding = attention_mask is not None and not bool(torch.all(attention_mask))
        needs_explicit_causal_mask = past_length > 0 and sequence_length > 1
        has_packed_boundaries = (
            attention_mask is None
            and past_length == 0
            and sequence_length > 1
            and bool(torch.any(position_ids[:, 1:] == 0))
        )
        if not has_padding and not needs_explicit_causal_mask and not has_packed_boundaries:
            return None

        query_positions = torch.arange(
            past_length, total_length, device=device
        ).unsqueeze(1)
        key_positions = torch.arange(total_length, device=device).unsqueeze(0)
        allowed = key_positions <= query_positions
        allowed = allowed.view(1, 1, sequence_length, total_length)
        if attention_mask is not None:
            allowed = allowed & attention_mask.view(batch_size, 1, 1, total_length)
        if has_packed_boundaries:
            segment_ids = position_ids.eq(0).long().cumsum(-1)
            same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
            allowed = allowed & same_segment.view(batch_size, 1, sequence_length, total_length)
        return allowed

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        if past_key_values is not None:
            if cache_position is not None:
                input_ids = input_ids[:, cache_position]
            else:
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": self.config.use_cache if use_cache is None else use_cache,
        }


class TalkieModel(TalkieForCausalLM):
    """Backbone view used by AutoModel and the vLLM Transformers backend."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> BaseModelOutputWithPast | tuple[torch.Tensor, ...]:
        hidden_states, past_key_values, use_cache = self._forward_hidden(
            input_ids,
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            **kwargs,
        )
        if return_dict is False:
            return (
                (hidden_states, past_key_values)
                if use_cache
                else (hidden_states,)
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


TalkieModel.register_for_auto_class("AutoModel")



TalkieForCausalLM.register_for_auto_class("AutoModelForCausalLM")
