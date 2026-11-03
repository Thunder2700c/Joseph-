import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from joseph.tokenizer import ByteBPETokenizer
from training.checkpoint import load_checkpoint


class TrainingCliSmokeTests(unittest.TestCase):
    def test_cli_runs_a_tiny_from_scratch_train_and_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                "A small language model learns patterns from text.",
                "Hu Tao speaks with a distinct first-person voice.",
                "A merchant carries public news to Liyue Harbor.",
                "A held out sample checks whether validation runs.",
            ] * 4
            train_file = root / "train.txt"
            validation_file = root / "validation.txt"
            train_file.write_text("\n".join(samples[:12]) + "\n", encoding="utf-8")
            validation_file.write_text("\n".join(samples[12:]) + "\n", encoding="utf-8")

            tokenizer = ByteBPETokenizer.train(samples, vocab_size=320)
            tokenizer_file = root / "tokenizer.json"
            tokenizer.save(tokenizer_file)
            output_file = root / "tiny.pt"

            result = subprocess.run(
                [
                    sys.executable, "-m", "training.train",
                    "--input", str(train_file),
                    "--validation", str(validation_file),
                    "--tokenizer", str(tokenizer_file),
                    "--output", str(output_file),
                    "--steps", "2",
                    "--batch-size", "2",
                    "--seq-len", "8",
                    "--d-model", "32",
                    "--n-layers", "1",
                    "--n-heads", "4",
                    "--eval-interval", "1",
                    "--eval-batches", "1",
                    "--device", "cpu",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue(output_file.is_file())
            self.assertIn("Starting from random initialization", result.stdout)
            self.assertIn("validation_loss=", result.stdout)
            self.assertIn("Saved experimental checkpoint", result.stdout)
            self.assertTrue((root / "token_cache" / "train.bin").is_file())
            self.assertTrue((root / "token_cache" / "validation.bin").is_file())
            first_payload = load_checkpoint(output_file)
            self.assertEqual(first_payload["step"], 2)

            continued_output = root / "continued.pt"
            continuation = subprocess.run(
                [
                    sys.executable, "-m", "training.train",
                    "--input", str(train_file),
                    "--validation", str(validation_file),
                    "--tokenizer", str(tokenizer_file),
                    "--output", str(continued_output),
                    "--init-checkpoint", str(output_file),
                    "--steps", "1",
                    "--batch-size", "2",
                    "--seq-len", "8",
                    "--d-model", "32",
                    "--n-layers", "1",
                    "--n-heads", "4",
                    "--eval-interval", "1",
                    "--eval-batches", "1",
                    "--device", "cpu",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                continuation.returncode, 0,
                msg=continuation.stdout + continuation.stderr,
            )
            self.assertIn("Continuing project checkpoint step 2", continuation.stdout)
            continued_payload = load_checkpoint(continued_output)
            self.assertEqual(continued_payload["step"], 3)
            self.assertIn("optimizer_state", continued_payload)
            run_metadata = json.loads((root / "continued.pt.run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_metadata["optimizer_state"], "restored_adamw_state")


if __name__ == "__main__":
    unittest.main()
