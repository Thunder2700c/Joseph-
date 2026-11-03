"""Generate deterministic qualitative samples from a Joseph checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from joseph.config import ModelConfig
from joseph.model import JosephLanguageModel
from joseph.tokenizer import ByteBPETokenizer
from training.checkpoint import load_checkpoint


DEFAULT_PROMPTS = (
    "<|user|> Hello! How has your day been? <|assistant|>",
    "<|user|> I made a mistake and feel embarrassed. What should I do? <|assistant|>",
    "<|user|> Explain why the sky looks blue in simple words. <|assistant|>",
    "<|user|> I disagree with your answer. Can you reconsider it? <|assistant|>",
    "<|user|> Tell me a short story about a traveller who gets lost. <|assistant|>",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = load_checkpoint(args.checkpoint, map_location=device)
    config = ModelConfig.from_dict(payload["config"])
    model = JosephLanguageModel(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    tokenizer = ByteBPETokenizer.load(args.tokenizer)

    results = [
        "# Qualitative base-language samples",
        "",
        f"Checkpoint: {Path(args.checkpoint).name}",
        f"Checkpoint step: {payload['step']}",
        f"Device: {device}",
        f"Sampling: temperature={args.temperature}, top_k={args.top_k}, "
        f"max_new_tokens={args.max_new_tokens}, seed={args.seed}",
        "",
        "These are generic conversation diagnostics, not a scored benchmark or Hu Tao identity test.",
        "",
    ]

    for index, prompt in enumerate(DEFAULT_PROMPTS):
        torch.manual_seed(args.seed + index)
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        if len(prompt_ids) > config.block_size:
            prompt_ids = prompt_ids[-config.block_size:]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=tokenizer.special_token_id("<eos>"),
                pad_token_id=tokenizer.special_token_id("<pad>"),
                stop_token_ids=[tokenizer.special_token_id("<|eot|>")],
                forbidden_token_ids=[
                    tokenizer.special_token_id("<bos>"),
                    tokenizer.special_token_id("<|system|>"),
                    tokenizer.special_token_id("<|user|>"),
                    tokenizer.special_token_id("<|assistant|>"),
                    tokenizer.special_token_id("<|hutao|>"),
                    tokenizer.special_token_id("<|memory|>"),
                    tokenizer.special_token_id("<|report|>"),
                ],
            )[0].tolist()
        continuation = tokenizer.decode(generated[len(prompt_ids):])
        results.extend((f"## Prompt {index + 1}", prompt, "", continuation, ""))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(results).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {len(DEFAULT_PROMPTS)} qualitative samples to {output}")


if __name__ == "__main__":
    main()
