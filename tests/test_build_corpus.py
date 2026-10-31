import json
import tempfile
import unittest
from pathlib import Path

from training.build_corpus import build_corpus


class CorpusBuilderTests(unittest.TestCase):
    def test_deterministic_split_and_duplicate_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.jsonl"
            rows = [
                {"prompt": "What happened in Mondstadt?", "completion": "The city was saved by its defenders."},
                {"prompt": "What happened in Liyue?", "completion": "The harbour was defended by its people."},
                {"text": "A sufficiently long general language sample for the corpus builder."},
                {"text": "A duplicate sample used for testing corpus deduplication."},
                {"text": "A duplicate sample used for testing corpus deduplication."},
            ]
            source.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            first = build_corpus([str(source)], str(root / "first"), validation_percent=20)
            second = build_corpus([str(source)], str(root / "second"), validation_percent=20)
            self.assertEqual(first["unique_samples"], 4)
            self.assertEqual(first["duplicates_removed"], 1)
            self.assertGreater(first["train_samples"], 0)
            self.assertGreater(first["validation_samples"], 0)
            self.assertEqual(first["train_sha256"], second["train_sha256"])
            self.assertEqual(first["validation_sha256"], second["validation_sha256"])
            self.assertFalse(first["private_chat_history_included"])

    def test_rejects_single_sample_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "one.txt"
            source.write_text("Only one sufficiently long paragraph is here.", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_corpus([str(source)], str(root / "out"))


if __name__ == "__main__":
    unittest.main()
