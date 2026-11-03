"""Small decoder-only Transformer used as Joseph's reference model."""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Joseph's neural model requires PyTorch. Install it with: "
        "python -m pip install -e '.[train]'"
    ) from exc

from joseph.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # PyTorch's fused scaled-dot-product attention applies the causal
        # mask without materialising a full attention matrix where supported.
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.residual_dropout(self.projection(output))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = 4 * config.d_model
        self.network = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class JosephLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

        # Residual branches start smaller so adding many blocks does not
        # produce excessively large activations at initialization.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.network[2].weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use GPT-style small initialization instead of PyTorch's unit-scale embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError("input sequence exceeds configured block_size")
        if targets is not None and tuple(targets.shape) != (batch, length):
            raise ValueError("targets must have the same [batch, sequence] shape as token_ids")
        if loss_mask is not None:
            if targets is None:
                raise ValueError("loss_mask requires targets")
            if tuple(loss_mask.shape) != (batch, length):
                raise ValueError("loss_mask must have the same [batch, sequence] shape as targets")

        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            token_losses = F.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size),
                targets.reshape(batch * length),
                reduction="none",
            ).view(batch, length)
            if loss_mask is None:
                loss = token_losses.mean()
            else:
                mask = loss_mask.to(device=token_losses.device, dtype=token_losses.dtype)
                denominator = mask.sum()
                if not torch.isfinite(denominator) or denominator.item() <= 0:
                    raise ValueError("loss_mask must select at least one target token")
                loss = (token_losses * mask).sum() / denominator
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = 40,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
        forbidden_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id is outside the model vocabulary")
        if pad_token_id is not None and not 0 <= pad_token_id < self.config.vocab_size:
            raise ValueError("pad_token_id is outside the model vocabulary")
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)
        forbidden_ids = set(forbidden_token_ids or [])
        for token_id in stop_ids | forbidden_ids:
            if not 0 <= token_id < self.config.vocab_size:
                raise ValueError("generation control token is outside the model vocabulary")
        if stop_ids & forbidden_ids:
            raise ValueError("a token cannot be both a stop token and forbidden")

        finished = torch.zeros(token_ids.size(0), dtype=torch.bool, device=token_ids.device)
        finished_pad_id = (
            pad_token_id if pad_token_id is not None
            else eos_token_id if eos_token_id is not None
            else 0
        )
        for _ in range(max_new_tokens):
            logits, _ = self(token_ids[:, -self.config.block_size :])
            next_logits = logits[:, -1, :]
            if forbidden_ids:
                forbidden_index = torch.tensor(
                    sorted(forbidden_ids), dtype=torch.long, device=next_logits.device
                )
                next_logits = next_logits.clone()
                next_logits.index_fill_(1, forbidden_index, torch.finfo(next_logits.dtype).min)
            if not torch.isfinite(next_logits).any(dim=-1).all():
                raise RuntimeError("generation constraints removed every possible next token")
            if temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and 0 < top_k < next_logits.size(-1):
                    threshold = torch.topk(next_logits, top_k, dim=-1).values[:, -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, torch.finfo(next_logits.dtype).min
                    )
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)

            # Do not continue emitting arbitrary text after an EOS. For a
            # mixed-length batch, pad rows that have already finished.
            was_finished = finished.clone()
            if was_finished.any():
                fill = torch.full_like(next_token, finished_pad_id)
                next_token = torch.where(was_finished[:, None], fill, next_token)
            token_ids = torch.cat((token_ids, next_token), dim=1)

            if stop_ids:
                token_is_stop = torch.zeros_like(finished)
                for token_id in stop_ids:
                    token_is_stop |= next_token.squeeze(-1).eq(token_id)
                finished |= token_is_stop
                if bool(finished.all()):
                    break
        return token_ids
