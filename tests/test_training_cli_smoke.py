import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from joseph.tokenizer import ByteBPETokenizer
from training.checkpoint import load_checkpoint


class TrainingCliSmokeTests(unittest.TestCase):
    def test_cli_runs_masked_dialogue_training_checkpoint_resume_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_samples = [
                "<|user|> Hi. <|assistant|> Hello! <|eot|>",
                "Plain text teaches language patterns from examples.",
                "<|user|> Why is the sky blue? <|assistant|> Air scatters blue light more than red light. <|eot|>",
                "Plants use sunlight to make food from water and carbon dioxide.",
                "<|user|> Can you admit uncertainty? <|assistant|> Yes. I should say when I do not know instead of guessing. <|eot|>",
                "Careful practice helps people understand difficult ideas.",
            ] * 8
            validation_samples = [
                "<|user|> What is gravity? <|assistant|> Gravity attracts objects that have mass. <|eot|>",
                "Clear writing makes complex ideas easier to understand.",
            ]
            train_file = root / "train.txt"
            validation_file = root / "validation.txt"
            train_file.write_text("\n".join(train_samples) + "\n", encoding="utf-8")
            validation_file.write_text("\n".join(validation_samples) + "\n", encoding="utf-8")

            # Fit only on training text. Validation remains held out.
            tokenizer = ByteBPETokenizer.train(train_samples, vocab_size=320)
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
                    "--steps", "3",
                    "--batch-size", "2",
                    "--seq-len", "16",
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

            train_cache = root / "token_cache" / "train.bin"
            validation_cache = root / "token_cache" / "validation.bin"
            self.assertTrue(train_cache.is_file())
            self.assertTrue(validation_cache.is_file())
            train_manifest = json.loads(
                (root / "token_cache" / "train.bin.manifest.json").read_text(encoding="utf-8")
            )
            validation_manifest = json.loads(
                (root / "token_cache" / "validation.bin.manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreater(train_manifest["dialogue_sample_count"], 0)
            self.assertGreater(validation_manifest["dialogue_sample_count"], 0)
            self.assertGreater(train_manifest["selected_target_count"], 0)
            self.assertEqual(
                (root / "token_cache" / "train.bin.loss-mask.bin").stat().st_size,
                train_manifest["token_count"],
            )
            self.assertIn("assistant", train_manifest["loss_policy"])

            first_payload = load_checkpoint(output_file)
            self.assertEqual(first_payload["step"], 3)

            run_metadata = json.loads(
                (root / "tiny.pt.run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                run_metadata["train_dialogue_sample_count"],
                train_manifest["dialogue_sample_count"],
            )
            self.assertGreater(run_metadata["train_loss_target_count"], 0)

            # Exercise the real generation CLI against the newly saved checkpoint.
            generated_file = root / "generated.samples.md"
            generation = subprocess.run(
                [
                    sys.executable, "-m", "training.generate_samples",
                    "--checkpoint", str(output_file),
                    "--tokenizer", str(tokenizer_file),
                    "--output", str(generated_file),
                    "--max-new-tokens", "4",
                    "--temperature", "0",
                    "--seed", "2026",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                generation.returncode, 0,
                msg=generation.stdout + generation.stderr,
            )
            self.assertTrue(generated_file.is_file())
            generated_text = generated_file.read_text(encoding="utf-8")
            self.assertIn("## Prompt 1", generated_text)
            self.assertIn("## Prompt 15", generated_text)

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
                    "--seq-len", "16",
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
            self.assertIn("Continuing project checkpoint step 3", continuation.stdout)
            continued_payload = load_checkpoint(continued_output)
            self.assertEqual(continued_payload["step"], 4)
            self.assertIn("optimizer_state", continued_payload)
            continued_metadata = json.loads(
                (root / "continued.pt.run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                continued_metadata["optimizer_state"],
                "restored_adamw_state",
            )


if __name__ == "__main__":
    unittest.main()
