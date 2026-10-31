import tempfile
import unittest
from pathlib import Path

from training.acquire_base_corpus import write_corpus


class BaseLanguageCorpusTests(unittest.TestCase):
    def test_streamed_rows_are_deduplicated_and_split(self):
        rows = [
            {"text": f"This is a sufficiently long synthetic educational sample number {i}."}
            for i in range(80)
        ]
        rows += [{"text": "This is a sufficiently long synthetic educational sample number 1."}]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_corpus(
                rows,
                directory,
                max_samples=100,
                max_characters=100_000,
                min_characters=20,
                validation_percent=20,
            )
            root = Path(directory)
            self.assertTrue((root / "train.txt").is_file())
            self.assertTrue((root / "validation.txt").is_file())
            self.assertGreater(manifest["samples"]["train"], 0)
            self.assertGreater(manifest["samples"]["validation"], 0)
            self.assertEqual(manifest["samples"]["duplicates_removed"], 1)
            self.assertEqual(manifest["selected_field"], "text")
            self.assertFalse(manifest["private_user_conversations_included"])
            self.assertNotIn("seed_data", (root / "train.txt").read_text(encoding="utf-8"))
            self.assertTrue((root / "manifest.json").is_file())

    def test_rejects_missing_split_or_invalid_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_corpus([{"text": "only one long enough item"}], directory)
            with self.assertRaises(ValueError):
                write_corpus([], directory, max_samples=1)


if __name__ == "__main__":
    unittest.main()
