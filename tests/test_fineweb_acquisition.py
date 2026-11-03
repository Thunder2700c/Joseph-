import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.acquire_fineweb_edu import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    write_corpus,
)
from training.merge_corpus_shards import merge_corpus_shards


def fineweb_row(index, **overrides):
    row = {
        "text": f"Educational source document number {index}. " + ("Detailed explanation of a scientific topic. " * 30),
        "language": "en",
        "language_score": 0.95,
        "int_score": 4,
        "token_count": 240,
        "id": f"record-{index}",
        "url": f"https://example.org/{index}",
        "dump": "CC-MAIN-2025-05",
        "date": "2025-01-01",
        "file_path": "commoncrawl/path",
        "score": 4.5,
    }
    row.update(overrides)
    return row


def make_shard(root: Path, index: int, train_text: str, validation_text: str) -> None:
    folder = root / f"fineweb-shard-{index}"
    folder.mkdir(parents=True)
    (folder / "train.txt").write_text(train_text + "\n", encoding="utf-8")
    (folder / "validation.txt").write_text(validation_text + "\n", encoding="utf-8")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "format": "hutao-fineweb-edu-corpus-shard-v1",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_card": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        "dataset_license": DATASET_LICENSE,
        "license_note": "Common Crawl terms also apply.",
        "selected_field": "text",
        "filter_fields": ["language", "language_score", "int_score", "token_count"],
        "excluded_fields": ["id", "dump", "url", "date", "file_path"],
        "selection": {
            "shard_count": 2,
            "shard_index": index,
            "max_samples": 100,
            "max_characters": 10000,
            "min_characters": 200,
            "max_document_characters": 100000,
            "min_language_score": 0.90,
            "min_educational_score": 4,
            "min_source_tokens": 100,
            "max_source_tokens": 30000,
            "validation_percent": 50,
        },
        "source_shards_available": 20,
        "shard": {"count": 2, "index": index},
        "files": {
            "train.txt": {"sha256": digest(folder / "train.txt")},
            "validation.txt": {"sha256": digest(folder / "validation.txt")},
        },
        "samples": {"train": 1, "validation": 1, "total_characters": len(train_text) + len(validation_text)},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class FineWebAcquisitionTests(unittest.TestCase):
    def test_filters_english_and_educational_quality_and_records_rejections(self):
        rows = [fineweb_row(i) for i in range(100)]
        rows += [
            fineweb_row(100, language="fr"),
            fineweb_row(101, language_score=0.5),
            fineweb_row(102, int_score=3),
            fineweb_row(103, token_count=50),
            fineweb_row(104, text="Contact the author at private.person@example.org for more information."),
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_corpus(
                rows,
                directory,
                shard_index=0,
                shard_count=20,
                max_samples=200,
                max_characters=2_000_000,
                validation_percent=50,
            )
            self.assertEqual(manifest["dataset_id"], DATASET_ID)
            self.assertEqual(manifest["dataset_revision"], DATASET_REVISION)
            self.assertEqual(manifest["dataset_license"], "ODC-By-1.0")
            self.assertEqual(manifest["shard"], {
                "count": 20,
                "index": 0,
                "method": "IterableDataset.shard before per-shard shuffle",
            })
            self.assertEqual(manifest["samples"]["train"] + manifest["samples"]["validation"], 100)
            self.assertEqual(manifest["filter_rejections"]["not_english"], 1)
            self.assertEqual(manifest["filter_rejections"]["language_score_below_threshold"], 1)
            self.assertEqual(manifest["filter_rejections"]["educational_score_below_threshold"], 1)
            self.assertEqual(manifest["filter_rejections"]["source_token_count_out_of_range"], 1)
            self.assertEqual(manifest["filter_rejections"]["obvious_pii_pattern"], 1)
            self.assertNotIn("https://example.org/0", (Path(directory) / "train.txt").read_text() + (Path(directory) / "validation.txt").read_text())

    def test_duplicate_documents_are_removed_within_shard(self):
        duplicate = fineweb_row(1)
        rows = [fineweb_row(i) for i in range(100)] + [dict(duplicate)]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_corpus(
                rows,
                directory,
                shard_index=0,
                shard_count=20,
                max_samples=200,
                max_characters=2_000_000,
                validation_percent=50,
            )
            self.assertEqual(manifest["samples"]["duplicates_removed_within_shard"], 1)
            self.assertEqual(manifest["samples"]["train"] + manifest["samples"]["validation"], 100)

    def test_generic_merger_keeps_fineweb_filter_and_license_provenance(self):
        def sample_for(split, prefix):
            for index in range(1000):
                text = f"{prefix} {index}. " + ("A sufficiently long educational content phrase. " * 8)
                bucket = int(hashlib.sha256(text.casefold().encode()).hexdigest()[:8], 16) % 100
                assigned = "validation" if bucket < 50 else "train"
                if assigned == split:
                    return text
            raise AssertionError(f"could not find deterministic {split} test sample")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "incoming"
            root.mkdir()
            for index in range(2):
                train_text = sample_for("train", f"Training shard {index}")
                validation_text = sample_for("validation", f"Validation shard {index}")
                make_shard(root, index, train_text, validation_text)

            merged = merge_corpus_shards(
                root,
                Path(directory) / "merged",
                expected_shards=2,
                max_characters=10000,
                shard_prefix="fineweb-shard-",
            )
            self.assertEqual(merged["dataset_id"], DATASET_ID)
            self.assertEqual(merged["dataset_revision"], DATASET_REVISION)
            self.assertEqual(merged["license_note"], "Common Crawl terms also apply.")
            self.assertEqual(merged["filter_policy"]["min_educational_score"], 4)
            self.assertEqual(merged["selection"]["source_shard_prefix"], "fineweb-shard-")
            self.assertEqual(merged["samples"]["train"], 2)
            self.assertEqual(merged["samples"]["validation"], 2)



if __name__ == "__main__":
    unittest.main()
