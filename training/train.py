"""From-scratch decoder-only LM trainer with auditable run telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

from joseph.config import ModelConfig
from joseph.model import JosephLanguageModel
from joseph.tokenizer import ByteBPETokenizer
from training.checkpoint import save_checkpoint
from training.tokenize_corpus import tokenize_text_file


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser = argparse.ArgumentParser(
        description="Train a randomly initialized decoder-only language model."
    )
    parser.add_argument("--input", required=True, help="training split UTF-8 text")
    parser.add_argument("--validation", required=True, help="held-out validation split UTF-8 text")
    parser.add_argument("--tokenizer", required=True, help="project-trained tokenizer JSON")
    parser.add_argument("--output", required=True, help="checkpoint output path")
    parser.add_argument("--token-cache-dir", help="directory for disk-backed token caches")
    parser.add_argument("--metrics-file", help="JSONL telemetry output; defaults beside checkpoint")
    parser.add_argument("--run-metadata", help="JSON run metadata output; defaults beside checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-threads", type=int, help="limit CPU threads for reproducible hosted runs")
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
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay cannot be negative")
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("--num-threads must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    input_path = Path(args.input)
    validation_path = Path(args.validation)
    tokenizer_path = Path(args.tokenizer)
    for path in (input_path, validation_path, tokenizer_path):
        if not path.is_file():
            parser.error(f"required input file does not exist: {path}")

    train_hash = sha256_file(input_path)
    validation_hash = sha256_file(validation_path)
    tokenizer_hash = sha256_file(tokenizer_path)
    if train_hash == validation_hash:
        parser.error("training and validation files have identical SHA-256 hashes")

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    cache_dir = Path(args.token_cache_dir) if args.token_cache_dir else Path(args.output).parent / "token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train.bin"
    validation_cache = cache_dir / "validation.bin"
    train_manifest = tokenize_text_file(input_path, tokenizer_path, train_cache)
    validation_manifest = tokenize_text_file(
        validation_path, tokenizer_path, validation_cache
    )
    train_data = open_token_stream(train_cache, train_manifest["token_count"])
    validation_data = open_token_stream(
        validation_cache, validation_manifest["token_count"]
    )

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file) if args.metrics_file else Path(str(output_path) + ".metrics.jsonl")
    metadata_path = Path(args.run_metadata) if args.run_metadata else Path(str(output_path) + ".run.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")

    metadata = {
        "format": "joseph-training-run-v1",
        "status": "running",
        "initialization": "random",
        "seed": args.seed,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": device,
        "cpu_threads": torch.get_num_threads(),
        "training_file_sha256": train_hash,
        "validation_file_sha256": validation_hash,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "train_token_count": train_manifest["token_count"],
        "validation_token_count": validation_manifest["token_count"],
        "model_config": config.to_dict(),
        "estimated_parameters": config.estimated_parameter_count(),
        "estimated_fp32_weight_bytes": config.estimated_weight_bytes(),
        "batch_size": args.batch_size,
        "sequence_length": args.seq_len,
        "steps_requested": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "checkpoint_path": str(output_path),
        "metrics_path": str(metrics_path),
        "started_unix": time.time(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Starting from random initialization: {config.estimated_parameter_count():,} parameters; "
        f"vocab={config.vocab_size}; device={device}; "
        f"train_tokens={train_data.numel():,}; validation_tokens={validation_data.numel():,}"
    )
    print(
        f"Input SHA-256: train={train_hash} validation={validation_hash} "
        f"tokenizer={tokenizer_hash}"
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    last_log_time = start_time
    last_logged_step = 0
    model.train()

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            eval_metrics = estimate_loss(
                model, train_data, validation_data,
                args.batch_size, args.seq_len, args.eval_batches, device,
            )
            now = time.perf_counter()
            elapsed_interval = max(now - last_log_time, 1e-9)
            interval_tokens = (step - last_logged_step) * args.batch_size * args.seq_len
            record = {
                "step": step,
                "steps_total": args.steps,
                "last_train_loss": float(loss.item()),
                **eval_metrics,
                "interval_seconds": elapsed_interval,
                "interval_tokens_per_second": interval_tokens / elapsed_interval,
                "tokens_seen_by_optimizer": step * args.batch_size * args.seq_len,
                "elapsed_seconds": now - start_time,
            }
            if device == "cuda":
                record["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            metric_text = " ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in eval_metrics.items()
            )
            print(
                f"step {step}/{args.steps} - train_loss={loss.item():.4f} "
                f"{metric_text} tokens_per_second={record['interval_tokens_per_second']:.1f} "
                f"elapsed_seconds={record['elapsed_seconds']:.1f}",
                flush=True,
            )
            last_log_time = now
            last_logged_step = step

    elapsed = time.perf_counter() - start_time
    checkpoint_size = save_checkpoint(output_path, model, config, args.steps)
    metadata.update({
        "status": "completed",
        "steps_completed": args.steps,
        "tokens_seen_by_optimizer": args.steps * args.batch_size * args.seq_len,
        "elapsed_seconds": elapsed,
        "average_training_tokens_per_second": (
            args.steps * args.batch_size * args.seq_len / max(elapsed, 1e-9)
        ),
        "checkpoint_bytes": checkpoint_size,
        "finished_unix": time.time(),
    })
    if device == "cuda":
        metadata["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved from-scratch experimental checkpoint ({checkpoint_size} bytes) to {output_path}. "
        f"Training tokens={metadata['tokens_seen_by_optimizer']:,}; "
        f"elapsed={elapsed:.1f}s; average_tokens_per_second="
        f"{metadata['average_training_tokens_per_second']:.1f}. "
        "A valid checkpoint or lower loss is not proof of conversational competence."
    )


if __name__ == "__main__":
    main()
