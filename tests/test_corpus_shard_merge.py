from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.acquire_base_corpus import write_corpus
from training.merge_corpus_shards import merge_corpus_shards


def make_samples(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {"text": f"{prefix} document {index:04d}: " + ("A useful explanation about the world. " * 8)}
        for index in range(count)
    ]


def build_test_shard(
    folder: Path,
    *,
    index: int,
    rows: list[dict[str, str]],
    validation_percent: int = 20,
) -> None:
    manifest = write_corpus(
        rows,
        folder,
        max_samples=len(rows) + 1,
        max_characters=100_000,
        min_characters=1,
        max_document_characters=10_000,
        validation_percent=validation_percent,
    )
    manifest["source_shards_available"] = 104
    manifest["shard"] = {
        "count": 2,
        "index": index,
        "method": "IterableDataset.shard before per-shard shuffle",
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class CorpusShardMergeTests(unittest.TestCase):
    def test_merges_shards_and_removes_cross_shard_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "incoming"
            output = root / "merged"
            shared = {"text": "A shared document appears in both input shards. " * 4}
            build_test_shard(
                incoming / "cosmopedia-shard-0",
                index=0,
                rows=make_samples("first", 100) + [shared],
            )
            build_test_shard(
                incoming / "cosmopedia-shard-1",
                index=1,
                rows=make_samples("second", 100) + [shared],
            )

            manifest = merge_corpus_shards(
                incoming, output, expected_shards=2, max_characters=100_000
            )

            self.assertEqual(manifest["samples"]["duplicates_removed_across_shards"], 1)
            self.assertEqual(len(manifest["shards"]), 2)
            self.assertGreater(manifest["samples"]["train"], 0)
            self.assertGreater(manifest["samples"]["validation"], 0)
            self.assertTrue((output / "manifest.json").is_file())

            all_text = (
                (output / "train.txt").read_text(encoding="utf-8")
                + (output / "validation.txt").read_text(encoding="utf-8")
            )
            self.assertEqual(all_text.count(shared["text"].strip()), 1)

    def test_rejects_missing_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "incoming"
            build_test_shard(
                incoming / "cosmopedia-shard-0",
                index=0,
                rows=make_samples("first", 100),
            )
            with self.assertRaisesRegex(ValueError, "expected 2 shard artifact"):
                merge_corpus_shards(
                    incoming, root / "merged", expected_shards=2
                )

    def test_rejects_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "incoming"
            folder = incoming / "cosmopedia-shard-0"
            rows = make_samples("leakage", 100)
            build_test_shard(folder, index=0, rows=rows)

            train_path = folder / "train.txt"
            validation_path = folder / "validation.txt"
            train_lines = train_path.read_text(encoding="utf-8").splitlines()
            validation_lines = validation_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(train_lines)
            moved = train_lines.pop()
            validation_lines.append(moved)
            train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
            validation_path.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            from training.acquire_base_corpus import file_sha256

            manifest["files"]["train.txt"]["sha256"] = file_sha256(train_path)
            manifest["files"]["validation.txt"]["sha256"] = file_sha256(validation_path)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            # A second, valid partition is required so the completeness check passes.
            build_test_shard(
                incoming / "cosmopedia-shard-1",
                index=1,
                rows=make_samples("second", 100),
            )
            with self.assertRaisesRegex(ValueError, "split leakage detected"):
                merge_corpus_shards(
                    incoming, root / "merged", expected_shards=2
                )


if __name__ == "__main__":
    unittest.main()
