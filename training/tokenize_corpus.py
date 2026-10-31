"""Encode line-oriented text into a disk-backed int32 token stream."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
from pathlib import Path

from joseph.tokenizer import ByteBPETokenizer


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_text_file(
    input_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Tokenize one text sample per line, adding BOS once and EOS per sample."""
    source = Path(input_path)
    tokenizer_file = Path(tokenizer_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")

    if array.array("I").itemsize != 4:
        raise RuntimeError("this token cache format requires 32-bit unsigned integers")

    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    buffer = array.array("I")
    token_count = 0
    sample_count = 0
    bos_id = tokenizer.special_token_id("<bos>")
    eos_id = tokenizer.special_token_id("<eos>")

    try:
        with source.open("r", encoding="utf-8") as input_file, temporary.open("wb") as output_file:
            buffer.append(bos_id)
            token_count += 1
            for line in input_file:
                sample = line.strip()
                if not sample:
                    continue
                ids = tokenizer.encode(sample)
                buffer.extend(ids)
                buffer.append(eos_id)
                token_count += len(ids) + 1
                sample_count += 1
                if len(buffer) >= 262_144:
                    buffer.tofile(output_file)
                    buffer = array.array("I")
            if buffer:
                buffer.tofile(output_file)
            output_file.flush()
            os.fsync(output_file.fileno())

        if sample_count == 0:
            raise ValueError(f"no non-empty text samples found in {source}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = {
        "format": "joseph-token-cache-v1",
        "input_file": str(source),
        "input_sha256": sha256_file(source),
        "tokenizer_file": str(tokenizer_file),
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "output_file": str(target),
        "output_sha256": sha256_file(target),
        "token_count": token_count,
        "sample_count": sample_count,
        "dtype": "int32",
        "byte_order": "native",
        "bos_id": bos_id,
        "eos_id": eos_id,
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a disk-backed int32 token cache for a line-oriented corpus."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = tokenize_text_file(args.input, args.tokenizer, args.output)
    print(
        f"Encoded {manifest['sample_count']:,} samples into "
        f"{manifest['token_count']:,} tokens: {manifest['output_file']}"
    )


if __name__ == "__main__":
    main()
