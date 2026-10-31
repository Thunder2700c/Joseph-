"""Minimal CPU/GPU language-model training loop from random initialization."""

import argparse
from pathlib import Path

import torch

from joseph.config import ModelConfig
from joseph.model import JosephLanguageModel
from joseph.tokenizer import ByteBPETokenizer
from training.checkpoint import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="permitted UTF-8 text corpus")
    parser.add_argument("--tokenizer", required=True, help="trained tokenizer JSON")
    parser.add_argument("--output", required=True, help="checkpoint output path")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.seq_len < 8:
        parser.error("steps/batch-size must be positive and seq-len must be >= 8")

    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    corpus = Path(args.input).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(corpus, add_bos=True, add_eos=True)
    if len(token_ids) <= args.seq_len + 1:
        parser.error("corpus is too small for the selected seq-len")

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.seq_len,
    )
    config.validate()
    model = JosephLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    data = torch.tensor(token_ids, dtype=torch.long, device=device)

    model.train()
    for step in range(1, args.steps + 1):
        starts = torch.randint(
            0, data.numel() - args.seq_len - 1, (args.batch_size,), device=device
        )
        x = torch.stack([data[i.item():i.item() + args.seq_len] for i in starts])
        y = torch.stack([
            data[i.item() + 1:i.item() + args.seq_len + 1] for i in starts
        ])
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step {step}/{args.steps} - loss {loss.item():.4f}")

    size = save_checkpoint(args.output, model, config, args.steps)
    print(
        f"Saved fresh-from-random-init checkpoint ({size} bytes) to {args.output}. "
        "This is not yet a chat-tuned assistant."
    )


if __name__ == "__main__":
    main()
