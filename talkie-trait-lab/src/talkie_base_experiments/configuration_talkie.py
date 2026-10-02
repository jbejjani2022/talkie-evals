"""Transformers configuration for the original Talkie architecture."""

from transformers import PretrainedConfig


class TalkieConfig(PretrainedConfig):
    model_type = "talkie"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "blocks.*.attn.attn_query": "colwise",
        "blocks.*.attn.attn_key": "colwise",
        "blocks.*.attn.attn_value": "colwise",
        "blocks.*.attn.attn_resid": "rowwise",
        "blocks.*.mlp.mlp_gate": "colwise",
        "blocks.*.mlp.mlp_linear": "colwise",
        "blocks.*.mlp.mlp_resid": "rowwise",
    }

    def __init__(
        self,
        vocab_size: int = 65536,
        n_layer: int = 40,
        n_head: int = 40,
        n_embd: int = 5120,
        head_dim: int = 128,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1_000_000.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int | None = None,
        eos_token_id: int = 65535,
        pad_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = head_dim
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.intermediate_size = int(round(((8 / 3) * n_embd) / 128) * 128)
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


TalkieConfig.register_for_auto_class()
