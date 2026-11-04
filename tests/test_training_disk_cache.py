import tempfile
import unittest
from pathlib import Path

import torch

from joseph.tokenizer import ByteBPETokenizer
from training.train import get_batch, open_token_stream
from training.tokenize_corpus import tokenize_text_file


class DiskBackedTrainingTests(unittest.TestCase):
    def test_token_cache_round_trip_and_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                "Hu Tao lives in Liyue Harbor.",
                "A merchant arrives with news from another nation.",
                "The Wangsheng Funeral Parlor serves the people of Liyue.",
            ]
            source = root / "samples.txt"
            source.write_text("\n".join(samples) + "\n", encoding="utf-8")
            tokenizer = ByteBPETokenizer.train(samples * 3, vocab_size=320)
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(tokenizer_path)

            cache = root / "tokens.bin"
            manifest = tokenize_text_file(source, tokenizer_path, cache)
            tokens = open_token_stream(cache, manifest["token_count"])

            self.assertEqual(tokens.dtype, torch.int32)
            self.assertEqual(manifest["sample_count"], len(samples))
            self.assertEqual(int(tokens[0]), tokenizer.special_token_id("<bos>"))
            self.assertEqual(
                int((tokens == tokenizer.special_token_id("<eos>")).sum()),
                len(samples),
            )
            x, y, mask = get_batch(tokens, batch_size=2, seq_len=8, device="cpu")
            self.assertIsNone(mask)
            self.assertEqual(tuple(x.shape), (2, 8))
            self.assertEqual(tuple(y.shape), (2, 8))
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_token_cache_rejects_incorrect_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            path.write_bytes(b"too short")
            with self.assertRaises(ValueError):
                open_token_stream(path, token_count=100)


if __name__ == "__main__":
    unittest.main()
