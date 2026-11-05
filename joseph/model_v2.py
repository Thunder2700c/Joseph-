"""From-scratch Llama-style decoder for the Hu Tao model."""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Hu Tao's neural model requires PyTorch. Install it with: "
        "python -m pip install -e '.[train]'"
    ) from exc

from joseph.config_v2 import HuTaoLlamaConfig


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by Llama-family models."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return value.to(input_dtype) * self.weight.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE with dynamic position tables and no learned position embeddings."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        if head_dim % 2:
            raise ValueError("rotary head_dim must be even")
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        positions = torch.arange(
            sequence_length,
            device=query.device,
            dtype=self.inverse_frequency.dtype,
        )
        frequencies = torch.outer(positions, self.inverse_frequency)
        phases = torch.cat((frequencies, frequencies), dim=-1)
        cosine = phases.cos().to(dtype=query.dtype)[None, None, :, :]
        sine = phases.sin().to(dtype=query.dtype)[None, None, :, :]
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        return query, key


class GroupedQueryAttention(nn.Module):
    """Causal attention with fewer K/V heads than query heads."""

    def __init__(self, config: HuTaoLlamaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, config.key_value_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.key_value_dim, bias=False
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, hidden_size = x.shape
        query = self.q_proj(x).view(
            batch, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(x).view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(x).view(
            batch, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query, key = self.rotary(query, key)
        if self.groups > 1:
            key = key.repeat_interleave(self.groups, dim=1)
            value = value.repeat_interleave(self.groups, dim=1)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence_length, hidden_size
        )
        return self.residual_dropout(self.o_proj(attended))


class SwiGLU(nn.Module):
    """Gated feed-forward network used by Llama-family models."""

    def __init__(self, config: HuTaoLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class HuTaoTransformerBlock(nn.Module):
    def __init__(self, config: HuTaoLlamaConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))


class HuTaoLlamaModel(nn.Module):
    """Random-initialized Llama-style causal language model.

    Returns the same (logits, loss) tuple shape as the earlier reference model
    to ease integration with the existing masked trainer.
    """

    def __init__(self, config: HuTaoLlamaConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            HuTaoTransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.apply(self._initialize_weights)
        residual_std = config.initializer_range / math.sqrt(
            2 * config.num_hidden_layers
        )
        for layer in self.layers:
            nn.init.normal_(
                layer.attention.o_proj.weight,
                mean=0.0,
                std=residual_std,
            )
            nn.init.normal_(
                layer.mlp.down_proj.weight,
                mean=0.0,
                std=residual_std,
            )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, sequence_length = token_ids.shape
        if sequence_length <= 0:
            raise ValueError("input sequence cannot be empty")
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds max_position_embeddings")
        if targets is not None and tuple(targets.shape) != (batch, sequence_length):
            raise ValueError("targets must match token_ids shape [batch, sequence]")
        if loss_mask is not None:
            if targets is None:
                raise ValueError("loss_mask requires targets")
            if tuple(loss_mask.shape) != (batch, sequence_length):
                raise ValueError("loss_mask must match targets shape [batch, sequence]")

        hidden = self.embedding_dropout(self.token_embedding(token_ids))
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)

        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            token_losses = F.cross_entropy(
                logits.reshape(batch * sequence_length, self.config.vocab_size),
                targets.reshape(batch * sequence_length),
                reduction="none",
            ).view(batch, sequence_length)
            if loss_mask is None:
                loss = token_losses.mean()
            else:
                mask = loss_mask.to(
                    device=token_losses.device,
                    dtype=token_losses.dtype,
                )
                denominator = mask.sum()
                if not torch.isfinite(denominator) or denominator.item() <= 0:
                    raise ValueError("loss_mask must select at least one target token")
                loss = (token_losses * mask).sum() / denominator
        return (logits if return_logits else None), loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
        stop_token_ids: list[int] | None = None,
        forbidden_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        stop_ids = set(stop_token_ids or [])
        forbidden_ids = set(forbidden_token_ids or [])
        for token_id in stop_ids | forbidden_ids:
            if not 0 <= token_id < self.config.vocab_size:
                raise ValueError("generation control token is outside model vocabulary")
        if stop_ids & forbidden_ids:
            raise ValueError("a token cannot be both a stop token and forbidden")

        finished = torch.zeros(
            token_ids.size(0), dtype=torch.bool, device=token_ids.device
        )
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.max_position_embeddings:]
            logits, _ = self(context, return_logits=True)
            next_logits = logits[:, -1, :]
            if forbidden_ids:
                indices = torch.tensor(
                    sorted(forbidden_ids), dtype=torch.long, device=next_logits.device
                )
                next_logits = next_logits.clone()
                next_logits.index_fill_(
                    1, indices, torch.finfo(next_logits.dtype).min
                )
            if temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and 0 < top_k < next_logits.size(-1):
                    cutoff = torch.topk(next_logits, top_k, dim=-1).values[:, -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < cutoff,
                        torch.finfo(next_logits.dtype).min,
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            if finished.any():
                next_token = torch.where(
                    finished[:, None],
                    torch.zeros_like(next_token),
                    next_token,
                )
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if stop_ids:
                stopped = torch.zeros_like(finished)
                for token_id in stop_ids:
                    stopped |= next_token.squeeze(-1).eq(token_id)
                finished |= stopped
                if bool(finished.all()):
                    break
        return token_ids
