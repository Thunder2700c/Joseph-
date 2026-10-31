"""From-scratch decoder-only LM trainer using disk-backed token streams."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from joseph.config import ModelConfig
from joseph.model import JosephLanguageModel
from joseph.tokenizer import ByteBPETokenizer
from training.checkpoint import save_checkpoint
from training.tokenize_corpus import tokenize_text_file


def open_token_stream(path: str | Path, token_count: int) -> torch.Tensor:
    if token_count < 1:
        raise ValueError("token_count must be positive")
    expected_bytes = token_count * 4
    actual_bytes = Path(path).stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"token cache size mismatch for {path}: expected {expected_bytes} bytes, got {actual_bytes}"
        )
    return torch.from_file(
        str(path), shared=False, size=token_count, dtype=torch.int32
    )


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= seq_len + 1:
        raise ValueError(f"token stream needs more than {seq_len + 1} tokens")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[int(start):int(start) + seq_len] for start in starts])
    y = torch.stack([
        data[int(start) + 1:int(start) + seq_len + 1] for start in starts
    ])
    return x.to(device=device, dtype=torch.long), y.to(device=device, dtype=torch.long)


@torch.no_grad()
def estimate_loss(
    model: JosephLanguageModel,
    train_data: torch.Tensor,
    validation_data: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    for name, data in (("train", train_data), ("validation", validation_data)):
        if data is None:
            continue
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, batch_size, seq_len, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        mean_loss = sum(losses) / len(losses)
        metrics[f"{name}_loss"] = mean_loss
        metrics[f"{name}_perplexity"] = math.exp(min(mean_loss, 20.0))
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="training split UTF-8 text")
    parser.add_argument("--validation", help="held-out validation split UTF-8 text")
    parser.add_argument("--tokenizer", required=True, help="trained tokenizer JSON")
    parser.add_argument("--output", required=True, help="checkpoint output path")
    parser.add_argument("--token-cache-dir", help="directory for disk-backed token caches")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.seq_len < 8:
        parser.error("steps/batch-size must be positive and seq-len must be >= 8")
    if args.eval_interval < 1 or args.eval_batches < 1:
        parser.error("eval-interval and eval-batches must be positive")
    if args.d_model < 1 or args.n_layers < 1 or args.n_heads < 1:
        parser.error("model dimensions must be positive")
    if args.d_model % args.n_heads:
        parser.error("--d-model must be divisible by --n-heads")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    cache_dir = Path(args.token_cache_dir) if args.token_cache_dir else Path(args.output).parent / "token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train.bin"
    train_manifest = tokenize_text_file(args.input, args.tokenizer, train_cache)
    validation_data = None
    if args.validation:
        validation_cache = cache_dir / "validation.bin"
        validation_manifest = tokenize_text_file(
            args.validation, args.tokenizer, validation_cache
        )
        validation_data = open_token_stream(
            validation_cache, validation_manifest["token_count"]
        )

    train_data = open_token_stream(train_cache, train_manifest["token_count"])
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.seq_len,
        dropout=args.dropout,
    )
    config.validate()
    model = JosephLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(
        f"Starting from random initialization: {config.estimated_parameter_count():,} parameters; "
        f"device={device}; train_tokens={train_data.numel():,}; "
        f"validation_tokens={validation_data.numel() if validation_data is not None else 0:,}"
    )
    if validation_data is None:
        print("WARNING: no validation split supplied; this run is not eligible for model acceptance.")

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            metrics = estimate_loss(
                model, train_data, validation_data,
                args.batch_size, args.seq_len, args.eval_batches, device,
            )
            metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"step {step}/{args.steps} - train_loss={loss.item():.4f} {metric_text}")

    size = save_checkpoint(args.output, model, config, args.steps)
    print(
        f"Saved from-scratch experimental checkpoint ({size} bytes) to {args.output}. "
        "A valid checkpoint or lower loss is not proof of conversational competence."
    )


if __name__ == "__main__":
    main()
