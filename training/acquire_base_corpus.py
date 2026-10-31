"""Stream a capped pilot sample from the SmolLM Cosmopedia v2 corpus.

Only the generated text field is used. Prompt and seed_data are deliberately omitted.
The acquired corpus is stored locally and is not copied into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

DATASET_ID = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "cosmopedia-v2"
DATASET_LICENSE = "ODC-By-1.0"
DATASET_REVISION = "3ba9d60"
DATASET_CARD = "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_corpus(
    rows: Iterable[dict],
    output_dir: str | Path,
    *,
    max_samples: int = 100_000,
    max_characters: int = 200_000_000,
    min_characters: int = 200,
    max_document_characters: int = 100_000,
    validation_percent: int = 5,
) -> dict:
    if max_samples < 2 or max_characters < 2:
        raise ValueError("max_samples and max_characters must be at least 2")
    if min_characters < 1 or max_document_characters < min_characters:
        raise ValueError("invalid document character limits")
    if not 1 <= validation_percent <= 49:
        raise ValueError("validation_percent must be between 1 and 49")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_path = target / "train.txt"
    validation_path = target / "validation.txt"

    seen: set[str] = set()
    train_samples = validation_samples = total_characters = duplicates = rejected = 0

    with train_path.open("w", encoding="utf-8", newline="\n") as train_file, \
            validation_path.open("w", encoding="utf-8", newline="\n") as validation_file:
        for row in rows:
            if train_samples + validation_samples >= max_samples:
                break
            if total_characters >= max_characters:
                break
            raw = row.get("text") if isinstance(row, dict) else None
            if not isinstance(raw, str):
                rejected += 1
                continue
            sample = normalize(raw)
            if len(sample) < min_characters or len(sample) > max_document_characters:
                rejected += 1
                continue
            if total_characters + len(sample) > max_characters:
                continue

            digest = hashlib.sha256(sample.casefold().encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)

            if int(digest[:8], 16) % 100 < validation_percent:
                validation_file.write(sample + "\n")
                validation_samples += 1
            else:
                train_file.write(sample + "\n")
                train_samples += 1
            total_characters += len(sample)

    if train_samples == 0 or validation_samples == 0:
        raise ValueError(
            "sample did not produce both train and validation splits; "
            "increase max_samples or adjust the sample selection"
        )

    manifest = {
        "format": "joseph-base-language-corpus-v1",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_card": DATASET_CARD,
        "dataset_license": DATASET_LICENSE,
        "selected_field": "text",
        "excluded_fields": ["prompt", "seed_data"],
        "selection": {
            "max_samples": max_samples,
            "max_characters": max_characters,
            "min_characters": min_characters,
            "max_document_characters": max_document_characters,
            "validation_percent": validation_percent,
            "split_method": "sha256(normalized_text) modulo 100",
            "deduplication": "sha256(normalized_text.casefold())",
        },
        "samples": {
            "train": train_samples,
            "validation": validation_samples,
            "duplicates_removed": duplicates,
            "rejected": rejected,
            "total_characters": total_characters,
            "approximate_tokens_at_4_chars_per_token": round(total_characters / 4),
        },
        "files": {
            "train.txt": {
                "sha256": file_sha256(train_path),
                "bytes": train_path.stat().st_size,
            },
            "validation.txt": {
                "sha256": file_sha256(validation_path),
                "bytes": validation_path.stat().st_size,
            },
        },
        "private_user_conversations_included": False,
        "traveler_private_story_information_included": False,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a capped Cosmopedia v2 sample into local train/validation text files."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--max-characters", type=int, default=200_000_000)
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--max-document-characters", type=int, default=100_000)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    args = parser.parse_args()

    if args.shuffle_buffer < 1:
        parser.error("--shuffle-buffer must be positive")
    try:
        from datasets import load_dataset
    except ImportError:
        parser.error(
            "optional data dependencies are missing; install with "
            "python -m pip install -e '.[data]'"
        )

    stream = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=args.revision,
    )
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    manifest = write_corpus(
        stream,
        args.output_dir,
        max_samples=args.max_samples,
        max_characters=args.max_characters,
        min_characters=args.min_characters,
        max_document_characters=args.max_document_characters,
        validation_percent=args.validation_percent,
    )
    manifest["dataset_revision"] = args.revision
    manifest["shuffle_seed"] = args.seed
    manifest["shuffle_buffer"] = args.shuffle_buffer
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["samples"], ensure_ascii=False, indent=2))
    print(f"Corpus manifest: {manifest_path}")


if __name__ == "__main__":
    main()
