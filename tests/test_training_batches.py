import unittest

try:
    import torch
    from training.train import get_batch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class TrainingBatchTests(unittest.TestCase):
    def test_vectorized_batch_has_correct_next_token_and_loss_mask_alignment(self):
        data = torch.arange(64, dtype=torch.int32)
        mask = (torch.arange(64) % 3 == 0).to(torch.uint8)
        generator = torch.Generator(device="cpu").manual_seed(2026)
        x, y, selected = get_batch(
            data, batch_size=4, seq_len=8, device="cpu",
            generator=generator, loss_masks=mask,
        )
        self.assertTrue(torch.equal(y, x + 1))
        self.assertTrue(torch.equal(selected, mask[y].bool()))

    def test_batch_rejects_unaligned_loss_mask(self):
        data = torch.arange(64, dtype=torch.int32)
        with self.assertRaises(ValueError):
            get_batch(data, 2, 8, "cpu", loss_masks=torch.ones(12, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
