import tempfile
import unittest
from pathlib import Path

from joseph.config import AI_PACKAGE_MAX_BYTES
from training.validate_ai_bundle import validate_ai_bundle


class AIBudgetTests(unittest.TestCase):
    def test_export_limit_cannot_be_raised_above_project_hard_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "project hard cap"):
                validate_ai_bundle(directory, max_bytes=AI_PACKAGE_MAX_BYTES + 1)

    def test_bundle_counts_all_files_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "model" / "weights.bin").write_bytes(b"x" * 60)
            (root / "tokenizer.json").write_bytes(b"y" * 25)
            (root / "config.json").write_bytes(b"z" * 15)

            report = validate_ai_bundle(root, max_bytes=100)

            self.assertEqual(report["total_bytes"], 100)
            self.assertEqual(report["file_count"], 3)
            self.assertEqual(report["remaining_bytes"], 0)
            self.assertTrue(report["within_limit"])

    def test_bundle_rejects_oversize_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"x" * 101)
            with self.assertRaisesRegex(ValueError, "above the hard limit"):
                validate_ai_bundle(root, max_bytes=100)

    def test_bundle_limit_cannot_be_configured_above_one_point_five_gigabytes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "project hard cap"):
                validate_ai_bundle(directory, max_bytes=AI_PACKAGE_MAX_BYTES + 1)


if __name__ == "__main__":
    unittest.main()
