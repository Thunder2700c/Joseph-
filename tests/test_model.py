import unittest

try:
    import torch
    from joseph.config import ModelConfig
    from joseph.model import JosephLanguageModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class ModelTests(unittest.TestCase):
    def test_default_vocabulary_matches_project_tokenizer(self):
        config = ModelConfig()
        self.assertEqual(config.vocab_size, 8192)
        self.assertLess(config.estimated_weight_bytes(), 1_000_000_000)

    def test_parameter_estimate_matches_tied_weight_model(self):
        config = ModelConfig(
            vocab_size=320, d_model=32, n_layers=2, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        actual = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(actual, config.estimated_parameter_count())

    def test_logits_loss_and_generation_shapes(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        inputs = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(inputs, targets)
        self.assertEqual(tuple(logits.shape), (2, 8, config.vocab_size))
        self.assertTrue(torch.isfinite(loss).item())
        generated = model.generate(inputs[:, :3], max_new_tokens=2, temperature=0)
        self.assertEqual(tuple(generated.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
