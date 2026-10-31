"""Fit the byte-level BPE tokenizer on the training split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

from joseph.tokenizer import ByteBPETokenizer, SPECIAL_TOKENS


def iter_text_samples(path: str | Path, max_samples: int | None = None) -> Iterator[str]:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            sample = line.strip()
            if not sample:
                continue
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a byte-level BPE tokenizer using only the supplied training text."
    )
    parser.add_argument("--input", required=True, help="training text file; never pass validation/test text")
    parser.add_argument("--output", required=True, help="tokenizer JSON output path")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        parser.error(f"training text file does not exist: {source}")

    tokenizer = ByteBPETokenizer.train(
        iter_text_samples(source, args.max_samples),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    target = Path(args.output)
    tokenizer.save(target)

    manifest = {
        "format": "joseph-tokenizer-training-v1",
        "input_file": str(source),
        "input_sha256": sha256_file(source),
        "tokenizer_file": str(target),
        "tokenizer_sha256": sha256_file(target),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "tokenizers_library_version": version("tokenizers"),
        "validation_or_test_text_used_for_fitting": False,
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Tokenizer saved: vocab={tokenizer.vocab_size}, "
        f"sha256={manifest['tokenizer_sha256']}, path={target}"
    )
    print(f"Tokenizer manifest: {manifest_path}")


if __name__ == "__main__":
    main()
