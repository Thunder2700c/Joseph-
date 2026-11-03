"""Merge disjoint Cosmopedia acquisition shards into one audited corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_corpus_shards(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    expected_shards: int = 20,
    max_characters: int = 200_000_000,
    shard_prefix: str = "cosmopedia-shard-",
) -> dict:
    """Verify, globally deduplicate, and merge all expected shard outputs.

    The content-hash split is checked again during merging so an item can never
    leak between train and validation, even if it appears in multiple shards.
    """
    if expected_shards < 2:
        raise ValueError("expected_shards must be at least 2")
    if max_characters < 2:
        raise ValueError("max_characters must be at least 2")

    root = Path(input_root)
    output = Path(output_dir)
    if not shard_prefix or "/" in shard_prefix or "\\" in shard_prefix:
        raise ValueError("shard_prefix must be a non-empty simple filename prefix")
    folders = sorted(path for path in root.glob(f"{shard_prefix}*") if path.is_dir())
    if len(folders) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} shard artifact directories, found {len(folders)}"
        )

    records: list[tuple[int, Path, dict, str]] = []
    seen_indexes: set[int] = set()
    common_fields: dict[str, object] | None = None
    source_shard_count: int | None = None
    validation_percent: int | None = None

    for folder in folders:
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing shard manifest: {manifest_path}")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        shard = manifest.get("shard")
        if not isinstance(shard, dict):
            raise ValueError(f"{folder} is not a source-sharded corpus output")
        if shard.get("count") != expected_shards:
            raise ValueError(f"{folder} declares an unexpected shard count")
        index = shard.get("index")
        if not isinstance(index, int) or not 0 <= index < expected_shards:
            raise ValueError(f"{folder} has an invalid shard index: {index!r}")
        if index in seen_indexes:
            raise ValueError(f"duplicate shard index: {index}")
        seen_indexes.add(index)

        available = manifest.get("source_shards_available")
        if not isinstance(available, int) or available < expected_shards:
            raise ValueError(f"{folder} has invalid source shard metadata")
        if source_shard_count is None:
            source_shard_count = available
        elif source_shard_count != available:
            raise ValueError("source shard count differs between acquisition shards")

        selection = manifest.get("selection", {})
        current_validation_percent = selection.get("validation_percent")
        if not isinstance(current_validation_percent, int):
            raise ValueError(f"{folder} is missing its validation split setting")
        if validation_percent is None:
            validation_percent = current_validation_percent
        elif validation_percent != current_validation_percent:
            raise ValueError("validation percentage differs between acquisition shards")

        selection_fields = manifest.get("selection", {})
        filter_policy = {
            key: selection_fields.get(key)
            for key in (
                "max_samples",
                "max_characters",
                "min_characters",
                "max_document_characters",
                "min_language_score",
                "min_educational_score",
                "min_source_tokens",
                "max_source_tokens",
                "validation_percent",
            )
            if key in selection_fields
        }
        fields = {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_config": manifest.get("dataset_config"),
            "dataset_revision": manifest.get("dataset_revision"),
            "dataset_license": manifest.get("dataset_license"),
            "dataset_card": manifest.get("dataset_card"),
            "license_note": manifest.get("license_note"),
            "selected_field": manifest.get("selected_field"),
            "filter_fields": manifest.get("filter_fields"),
            "excluded_fields": manifest.get("excluded_fields"),
            "filter_policy": filter_policy,
        }
        if common_fields is None:
            common_fields = fields
        elif common_fields != fields:
            raise ValueError("dataset provenance differs between acquisition shards")

        for filename in ("train.txt", "validation.txt"):
            file_path = folder / filename
            if not file_path.is_file():
                raise FileNotFoundError(f"missing shard split: {file_path}")
            expected_hash = manifest.get("files", {}).get(filename, {}).get("sha256")
            if expected_hash != sha256_file(file_path):
                raise ValueError(f"SHA-256 mismatch for {file_path}")

        records.append(
            (index, folder, manifest, hashlib.sha256(manifest_bytes).hexdigest())
        )

    if seen_indexes != set(range(expected_shards)):
        missing = sorted(set(range(expected_shards)) - seen_indexes)
        raise ValueError(f"missing source shard indexes: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        "train": output / "train.txt.tmp",
        "validation": output / "validation.txt.tmp",
    }
    final_paths = {
        "train": output / "train.txt",
        "validation": output / "validation.txt",
    }
    seen_hashes: set[str] = set()
    sample_counts = {"train": 0, "validation": 0}
    character_counts = {"train": 0, "validation": 0}
    duplicates = 0
    total_characters = 0

    try:
        with (
            temporary_paths["train"].open("w", encoding="utf-8", newline="\n") as train_out,
            temporary_paths["validation"].open("w", encoding="utf-8", newline="\n") as validation_out,
        ):
            writers = {"train": train_out, "validation": validation_out}
            for index, folder, _, _ in sorted(records, key=lambda row: row[0]):
                for split in ("train", "validation"):
                    with (folder / f"{split}.txt").open("r", encoding="utf-8") as source:
                        for raw_line in source:
                            sample = " ".join(raw_line.split())
                            if not sample:
                                continue
                            digest = hashlib.sha256(sample.casefold().encode("utf-8")).hexdigest()
                            assigned_split = (
                                "validation"
                                if int(digest[:8], 16) % 100 < validation_percent
                                else "train"
                            )
                            if assigned_split != split:
                                raise ValueError(
                                    f"split leakage detected in shard {index}: "
                                    f"sample belongs in {assigned_split}, not {split}"
                                )
                            if digest in seen_hashes:
                                duplicates += 1
                                continue
                            if total_characters + len(sample) > max_characters:
                                raise ValueError(
                                    f"merged corpus exceeds {max_characters} characters; "
                                    "raise the explicit cap or reduce per-shard acquisition"
                                )
                            seen_hashes.add(digest)
                            writers[split].write(sample + "\n")
                            sample_counts[split] += 1
                            character_counts[split] += len(sample)
                            total_characters += len(sample)

        if not sample_counts["train"] or not sample_counts["validation"]:
            raise ValueError("merged corpus must contain both training and validation samples")

        for split in ("train", "validation"):
            temporary_paths[split].replace(final_paths[split])

    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()

    shard_summaries = [
        {
            "index": index,
            "manifest_sha256": manifest_hash,
            "samples": manifest.get("samples", {}),
        }
        for index, _, manifest, manifest_hash in sorted(records, key=lambda row: row[0])
    ]

    manifest = {
        "format": "joseph-base-language-corpus-merged-v1",
        **(common_fields or {}),
        "selection": {
            "expected_shards": expected_shards,
            "source_shards_available": source_shard_count,
            "max_characters": max_characters,
            "validation_percent": validation_percent,
            "global_deduplication": "sha256(normalized_text.casefold())",
            "split_method": "sha256(normalized_text.casefold()) modulo 100",
            "shard_partitioning": "disjoint IterableDataset.shard partitions before shuffle",
            "source_shard_prefix": shard_prefix,
        },
        "samples": {
            "train": sample_counts["train"],
            "validation": sample_counts["validation"],
            "duplicates_removed_across_shards": duplicates,
            "total_characters": total_characters,
            "approximate_tokens_at_4_chars_per_token": round(total_characters / 4),
        },
        "files": {
            f"{split}.txt": {
                "sha256": sha256_file(final_paths[split]),
                "bytes": final_paths[split].stat().st_size,
                "characters": character_counts[split],
                "samples": sample_counts[split],
            }
            for split in ("train", "validation")
        },
        "shards": shard_summaries,
        "private_user_conversations_included": False,
        "traveler_private_story_information_included": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and globally deduplicate disjoint Cosmopedia acquisition shards."
    )
    parser.add_argument("--input-root", required=True, help="directory containing downloaded shard artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-shards", type=int, default=20)
    parser.add_argument("--max-characters", type=int, default=200_000_000)
    parser.add_argument("--shard-prefix", default="cosmopedia-shard-")
    args = parser.parse_args()
    manifest = merge_corpus_shards(
        args.input_root,
        args.output_dir,
        expected_shards=args.expected_shards,
        max_characters=args.max_characters,
        shard_prefix=args.shard_prefix,
    )
    print(json.dumps(manifest["samples"], ensure_ascii=False, indent=2))
    print(f"Merged corpus manifest: {Path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
