"""Train Joseph's byte-level BPE tokenizer from permitted plain text."""

import argparse
from pathlib import Path

from joseph.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UTF-8 plain-text corpus")
    parser.add_argument("--output", required=True, help="where tokenizer JSON is saved")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    tokenizer = ByteBPETokenizer.train(
        [text], vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    tokenizer.save(args.output)
    print(f"Saved tokenizer: vocab_size={tokenizer.vocab_size}, path={args.output}")


if __name__ == "__main__":
    main()
