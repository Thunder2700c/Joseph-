import tempfile
import unittest
from pathlib import Path

from joseph.loss_mask import build_loss_mask
from joseph.tokenizer import ByteBPETokenizer
from training.prepare_dialogue_corpus import (
    DOLLY_LICENSE,
    prepare_dolly_samples,
    dolly_row_to_sample,
)
from training.prepare_oasst1_corpus import mix_corpora


class DialogueSerializationTests(unittest.TestCase):
    def test_dolly_row_uses_distinct_user_and_assistant_roles(self):
        sample = dolly_row_to_sample({
            "instruction": "Explain gravity simply.",
            "context": "A planet has mass.",
            "response": "Gravity is the attraction between masses.",
        })
        self.assertIsNotNone(sample)
        self.assertIn("<|user|>", sample)
        self.assertIn("Context: A planet has mass.", sample)
        self.assertIn("<|assistant|>", sample)
        self.assertIn("<|eot|>", sample)
        self.assertNotIn("Assistant:", sample)

    def test_dolly_split_is_deterministic_and_reports_sharealike_license(self):
        rows = [
            {
                "instruction": f"Give me three ideas about topic {index}.",
                "context": "",
                "response": f"Here are three useful ideas for topic {index}.",
            }
            for index in range(100)
        ]
        train_a, validation_a, report_a = prepare_dolly_samples(rows, validation_percent=10)
        train_b, validation_b, report_b = prepare_dolly_samples(rows, validation_percent=10)
        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertEqual(report_a, report_b)
        self.assertEqual(len(train_a) + len(validation_a), 100)
        self.assertEqual(report_a["license"], DOLLY_LICENSE)
        self.assertEqual(report_a["duplicates_removed"], 0)

    def test_dolly_duplicates_and_malformed_rows_are_rejected(self):
        row = {
            "instruction": "Summarize this short passage.",
            "context": "A useful context passage.",
            "response": "This is a concise summary.",
        }
        rows = [row, dict(row), {"instruction": "Missing answer", "response": None}]
        # A small deterministic split can be empty; use a larger set and retain
        # one duplicate and one malformed row.
        rows = [
            {
                "instruction": f"Summarize passage number {index}.",
                "context": "A useful context passage.",
                "response": f"This is summary number {index}.",
            }
            for index in range(50)
        ] + [row, dict(row), {"instruction": "Missing answer", "response": None}]
        train, validation, report = prepare_dolly_samples(rows, validation_percent=10)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertEqual(len(train) + len(validation), 51)

    def test_mixture_caps_dialogue_share_when_one_full_pass_is_too_large(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_train = [
                f"General learning sample {index} explains a useful topic with clear details."
                for index in range(80)
            ]
            base_validation = [
                f"Held out base validation sample {index} tests general language."
                for index in range(12)
            ]
            dialogue_train = [
                f"<|user|> Question {index}: explain this subject clearly. "
                f"<|assistant|> Answer {index}: here is a careful explanation with examples. <|eot|>"
                for index in range(240)
            ]
            dialogue_validation = [
                f"<|user|> Held out question {index}? "
                f"<|assistant|> Held out answer {index}. <|eot|>"
                for index in range(12)
            ]

            tokenizer = ByteBPETokenizer.train(
                base_train + base_validation + dialogue_train + dialogue_validation,
                vocab_size=512,
            )
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(tokenizer_path)
            base_train_path = root / "base-train.txt"
            base_validation_path = root / "base-validation.txt"
            base_train_path.write_text("\\n".join(base_train) + "\\n", encoding="utf-8")
            base_validation_path.write_text(
                "\\n".join(base_validation) + "\\n", encoding="utf-8"
            )

            first = mix_corpora(
                str(base_train_path),
                str(base_validation_path),
                dialogue_train,
                dialogue_validation,
                str(tokenizer_path),
                str(root / "mixed-a"),
                dialogue_fraction=0.35,
                seed=2026,
            )
            second = mix_corpora(
                str(base_train_path),
                str(base_validation_path),
                dialogue_train,
                dialogue_validation,
                str(tokenizer_path),
                str(root / "mixed-b"),
                dialogue_fraction=0.35,
                seed=2026,
            )

            self.assertGreater(first["dialogue_source_token_count"], first["base_train_token_count"])
            self.assertLess(first["dialogue_train_samples_used"], len(dialogue_train))
            self.assertGreaterEqual(first["dialogue_token_fraction_achieved_estimate"], 0.34)
            self.assertLessEqual(first["dialogue_token_fraction_achieved_estimate"], 0.36)
            self.assertEqual(first["dialogue_train_samples_used"], second["dialogue_train_samples_used"])
            self.assertEqual(first["train_sha256"], second["train_sha256"])
            self.assertEqual(first["dialogue_token_fraction_achieved_estimate"],
                             second["dialogue_token_fraction_achieved_estimate"])

    def test_response_mask_excludes_user_text_and_masks_only_assistant_turns(self):
        user, assistant, eot = 5, 6, 10
        sample_ids = [
            user, 250, assistant, 251, eot,
            user, 252, assistant, 253, eot,
        ]
        mask = build_loss_mask(sample_ids, assistant_id=assistant, eot_id=eot)
        self.assertEqual(mask, [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        self.assertEqual(len(mask), len(sample_ids) + 2)

    def test_plain_text_keeps_full_next_token_loss_except_bos(self):
        mask = build_loss_mask([250, 251, 252], assistant_id=6, eot_id=10)
        self.assertEqual(mask, [0, 1, 1, 1, 1])

    def test_malformed_dialogue_without_eot_is_rejected(self):
        with self.assertRaises(ValueError):
            build_loss_mask([5, 250, 6, 251], assistant_id=6, eot_id=10)


if __name__ == "__main__":
    unittest.main()
