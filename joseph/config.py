"""Configuration for Joseph's from-scratch decoder-only language model."""

from dataclasses import asdict, dataclass

# Hard ceiling for the complete distributable AI payload, in decimal bytes.
AI_PACKAGE_MAX_BYTES = 1_500_000_000


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    block_size: int = 256
    dropout: float = 0.1
    max_weight_bytes: int = AI_PACKAGE_MAX_BYTES

    def validate(self) -> None:
        if self.vocab_size < 260:
            raise ValueError("vocab_size must be at least 260 (special tokens + 256 bytes)")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_weight_bytes <= 0:
            raise ValueError("max_weight_bytes must be positive")
        if self.max_weight_bytes > AI_PACKAGE_MAX_BYTES:
            raise ValueError(
                "checkpoint ceiling cannot exceed the complete AI package ceiling "
                f"of {AI_PACKAGE_MAX_BYTES} bytes"
            )
        if self.estimated_weight_bytes() > self.max_weight_bytes:
            raise ValueError("estimated FP32 model weights exceed max_weight_bytes")

    def estimated_parameter_count(self) -> int:
        d = self.d_model
        embedding = self.vocab_size * d + self.block_size * d
        attention = 4 * d * d + 4 * d
        mlp = 8 * d * d + 5 * d
        norms = 4 * d
        blocks = self.n_layers * (attention + mlp + norms)
        final_norm = 2 * d
        return embedding + blocks + final_norm

    def estimated_weight_bytes(self, bytes_per_parameter: int = 4) -> int:
        if bytes_per_parameter <= 0:
            raise ValueError("bytes_per_parameter must be positive")
        return self.estimated_parameter_count() * bytes_per_parameter

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        config = cls(**{key: value for key, value in data.items() if key in allowed})
        config.validate()
        return config
