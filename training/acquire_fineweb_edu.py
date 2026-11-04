"""Stream audited English FineWeb-Edu documents into deterministic corpus shards.

FineWeb-Edu's dataset card declares ODC-By 1.0 and notes that the underlying
Common Crawl terms also apply. This tool preserves that attribution caveat.
It trains on the text field only and uses the language/quality metadata solely
for filtering and audit reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import re
from pathlib import Path
from typing import Iterable

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATASET_LICENSE = "ODC-By-1.0"
DATASET_CARD = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/"
    "tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
)
EMAIL_PATTERN = re.compile(r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", re.IGNORECASE)
US_SSN_PATTERN = re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b")
LICENSE_NOTE = (
    "The dataset card declares ODC-By 1.0. FineWeb-Edu is filtered from Common Crawl; "
    "the dataset card also states that Common Crawl terms of use apply. This project "
    "records the dataset-level licence but does not claim to have independently "
    "cleared every underlying webpage's copyright or reuse rights."
)


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
    shard_index: int,
    shard_count: int,
    max_samples: int = 100_000,
    max_characters: int = 100_000_000,
    min_characters: int = 200,
    max_document_characters: int = 100_000,
    min_language_score: float = 0.90,
    min_educational_score: int = 4,
    min_source_tokens: int = 100,
    max_source_tokens: int = 30_000,
    validation_percent: int = 5,
) -> dict:
    if shard_count < 2 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    if max_samples < 2 or max_characters < 2:
        raise ValueError("max_samples and max_characters must be at least 2")
    if min_characters < 1 or max_document_characters < min_characters:
        raise ValueError("invalid document character limits")
    if not 0.0 <= min_language_score <= 1.0:
        raise ValueError("min_language_score must be in [0, 1]")
    if min_educational_score < 0:
        raise ValueError("min_educational_score cannot be negative")
    if not 1 <= validation_percent <= 99:
        raise ValueError("validation_percent must be between 1 and 99")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_path = target / "train.txt"
    validation_path = target / "validation.txt"

    seen: set[str] = set()
    train_samples = validation_samples = total_characters = duplicates = 0
    rejected: dict[str, int] = {}
    rows_seen = 0

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    with train_path.open("w", encoding="utf-8", newline="\n") as train_file, \
            validation_path.open("w", encoding="utf-8", newline="\n") as validation_file:
        for row in rows:
            rows_seen += 1
            if train_samples + validation_samples >= max_samples:
                break
            if total_characters >= max_characters:
                break
            if not isinstance(row, dict):
                reject("row_not_mapping")
                continue

            if row.get("language") != "en":
                reject("not_english")
                continue
            language_score = row.get("language_score")
            if not isinstance(language_score, numbers.Real) or float(language_score) < min_language_score:
                reject("language_score_below_threshold")
                continue
            educational_score = row.get("int_score")
            if not isinstance(educational_score, numbers.Integral) or int(educational_score) < min_educational_score:
                reject("educational_score_below_threshold")
                continue
            source_tokens = row.get("token_count")
            if not isinstance(source_tokens, numbers.Integral) or not min_source_tokens <= int(source_tokens) <= max_source_tokens:
                reject("source_token_count_out_of_range")
                continue

            raw = row.get("text")
            if not isinstance(raw, str):
                reject("missing_text")
                continue
            sample = normalize(raw)
            if len(sample) < min_characters or len(sample) > max_document_characters:
                reject("document_character_length_out_of_range")
                continue
            # A lightweight precaution against obvious contact/identifier
            # strings. It is not a comprehensive PII detector or guarantee.
            digit_count = sum(character.isdigit() for character in sample)
            if EMAIL_PATTERN.search(sample) or US_SSN_PATTERN.search(sample) or digit_count >= 10:
                reject("obvious_pii_pattern")
                continue
            if total_characters + len(sample) > max_characters:
                reject("would_exceed_character_cap")
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
            f"shard {shard_index} produced train={train_samples}, validation={validation_samples}; "
            "increase its cap or check the filtering thresholds"
        )

    manifest = {
        "format": "hutao-fineweb-edu-corpus-shard-v1",
        "shard": {
            "count": shard_count,
            "index": shard_index,
            "method": "IterableDataset.shard before per-shard shuffle",
        },
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_card": DATASET_CARD,
        "dataset_license": DATASET_LICENSE,
        "license_note": LICENSE_NOTE,
        "selected_field": "text",
        "filter_fields": ["language", "language_score", "int_score", "token_count"],
        "excluded_fields": ["id", "dump", "url", "date", "file_path"],
        "selection": {
            "shard_count": shard_count,
            "shard_index": shard_index,
            "max_samples": max_samples,
            "max_characters": max_characters,
            "min_characters": min_characters,
            "max_document_characters": max_document_characters,
            "min_language_score": min_language_score,
            "min_educational_score": min_educational_score,
            "min_source_tokens": min_source_tokens,
            "max_source_tokens": max_source_tokens,
            "obvious_pii_patterns_filtered": ["email", "US SSN", "10-or-more digits per document"],
            "validation_percent": validation_percent,
            "split_method": "sha256(normalized_text.casefold()) modulo 100",
            "deduplication": "sha256(normalized_text.casefold()) within each source shard; global deduplication at merge",
            "source_partitioning": "IterableDataset.shard before deterministic per-shard shuffle",
        },
        "rows_seen": rows_seen,
        "filter_rejections": dict(sorted(rejected.items())),
        "samples": {
            "train": train_samples,
            "validation": validation_samples,
            "duplicates_removed_within_shard": duplicates,
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
        "hu_tao_identity_material_included": False,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=20)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--max-characters", type=int, default=100_000_000)
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--max-document-characters", type=int, default=100_000)
    parser.add_argument("--min-language-score", type=float, default=0.90)
    parser.add_argument("--min-educational-score", type=int, default=4)
    parser.add_argument("--min-source-tokens", type=int, default=100)
    parser.add_argument("--max-source-tokens", type=int, default=30_000)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=2_000)
    args = parser.parse_args()

    if args.shuffle_buffer < 1:
        parser.error("--shuffle-buffer must be positive")
    if args.num_shards < 2:
        parser.error("--num-shards must be at least 2")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")

    try:
        from datasets import load_dataset
    except ImportError:
        parser.error("install datasets with: python -m pip install 'datasets>=3.3,<5'")

    stream = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    source_shard_count = getattr(stream, "num_shards", None)
    if source_shard_count is None or source_shard_count < args.num_shards:
        parser.error(
            f"source has {source_shard_count!r} shards; cannot partition into {args.num_shards}"
        )

    stream = stream.shard(num_shards=args.num_shards, index=args.shard_index)
    stream = stream.shuffle(
        seed=args.seed + args.shard_index,
        buffer_size=args.shuffle_buffer,
    )
    manifest = write_corpus(
        stream,
        args.output_dir,
        shard_index=args.shard_index,
        shard_count=args.num_shards,
        max_samples=args.max_samples,
        max_characters=args.max_characters,
        min_characters=args.min_characters,
        max_document_characters=args.max_document_characters,
        min_language_score=args.min_language_score,
        min_educational_score=args.min_educational_score,
        min_source_tokens=args.min_source_tokens,
        max_source_tokens=args.max_source_tokens,
        validation_percent=args.validation_percent,
    )
    manifest["source_shards_available"] = source_shard_count
    manifest["shard"] = {
        "count": args.num_shards,
        "index": args.shard_index,
        "method": "IterableDataset.shard before per-shard shuffle",
    }
    manifest["shuffle_seed"] = args.seed + args.shard_index
    manifest["shuffle_buffer"] = args.shuffle_buffer
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "source_shard": {"index": args.shard_index, "count": args.num_shards},
        "rows_seen": manifest["rows_seen"],
        "samples": manifest["samples"],
        "filter_rejections": manifest["filter_rejections"],
    }, ensure_ascii=False, indent=2))
    print(f"Corpus manifest: {manifest_path}")


if __name__ == "__main__":
    main()
