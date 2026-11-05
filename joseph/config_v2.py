"""Configuration and size estimates for Hu Tao's from-scratch Llama-style model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


# This applies only to the final, complete inference export. It never caps
# training checkpoints, corpora, optimizer state or evaluation files.
INFERENCE_BUNDLE_MAX_BYTES = 1_500_000_000


@dataclass(frozen=True)
class HuTaoLlamaConfig:
    vocab_size: int = 32_768
    hidden_size: int = 2_560
    intermediate_size: int = 7_168
    num_hidden_layers: int = 24
    num_attention_heads: int = 20
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4_096
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def validate(self) -> None:
        if self.vocab_size < 260:
            raise ValueError("vocab_size must cover byte fallback and special tokens")
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def key_value_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def estimated_parameter_count(self) -> int:
        """Exact count for this implementation (bias-free, tied/untied output as configured)."""
        hidden = self.hidden_size
        attention = (
            2 * hidden * hidden
            + 2 * hidden * self.key_value_dim
            + 2 * hidden
        )
        mlp = 3 * hidden * self.intermediate_size
        layer_norms = 2 * hidden
        per_layer = attention + mlp + layer_norms
        token_embedding = self.vocab_size * hidden
        output_head = 0 if self.tie_word_embeddings else self.vocab_size * hidden
        final_norm = hidden
        return (
            token_embedding
            + self.num_hidden_layers * per_layer
            + final_norm
            + output_head
        )

    def estimated_q4_bundle_bytes(
        self,
        *,
        bytes_per_parameter: float = 0.56,
        auxiliary_bytes: int = 32_000_000,
    ) -> int:
        """Planning estimate only; actual GGUF bytes must be measured after export."""
        if bytes_per_parameter <= 0 or auxiliary_bytes < 0:
            raise ValueError("quantization and auxiliary size estimates must be non-negative")
        return int(self.estimated_parameter_count() * bytes_per_parameter) + auxiliary_bytes

    def estimated_kv_cache_bytes(
        self,
        *,
        sequence_length: int | None = None,
        batch_size: int = 1,
        bytes_per_value: int = 2,
    ) -> int:
        """K+V cache size before allocator/workspace overhead."""
        length = sequence_length or self.max_position_embeddings
        if length <= 0 or batch_size <= 0 or bytes_per_value <= 0:
            raise ValueError("KV-cache dimensions must be positive")
        return (
            2
            * self.num_hidden_layers
            * batch_size
            * length
            * self.key_value_dim
            * bytes_per_value
        )

    def validate_inference_estimate(
        self,
        *,
        max_bytes: int = INFERENCE_BUNDLE_MAX_BYTES,
        bytes_per_parameter: float = 0.56,
        auxiliary_bytes: int = 32_000_000,
    ) -> int:
        if max_bytes <= 0 or max_bytes > INFERENCE_BUNDLE_MAX_BYTES:
            raise ValueError("max_bytes must be within the project inference-bundle hard cap")
        estimate = self.estimated_q4_bundle_bytes(
            bytes_per_parameter=bytes_per_parameter,
            auxiliary_bytes=auxiliary_bytes,
        )
        if estimate > max_bytes:
            raise ValueError(
                f"estimated Q4 bundle is {estimate:,} bytes, above the {max_bytes:,}-byte cap"
            )
        return estimate

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "HuTaoLlamaConfig":
        allowed = cls.__dataclass_fields__.keys()
        config = cls(**{key: value for key, value in values.items() if key in allowed})
        config.validate()
        return config
