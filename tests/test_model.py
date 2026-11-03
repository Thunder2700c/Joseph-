import unittest
from unittest.mock import patch
import math

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

    def test_embeddings_use_small_gpt_style_initialization(self):
        torch.manual_seed(7)
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=2, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        self.assertLess(abs(float(model.token_embedding.weight.std()) - 0.02), 0.004)
        self.assertLess(abs(float(model.position_embedding.weight.std()) - 0.02), 0.004)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

        inputs = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        model.eval()
        _, loss = model(inputs, targets)
        # A fresh model should begin near the uniform-vocabulary baseline,
        # not at the 100+ cross-entropy observed in the previous model.
        self.assertLess(float(loss), math.log(config.vocab_size) + 0.5)
        self.assertGreater(float(loss), math.log(config.vocab_size) - 0.5)

    def test_generation_stops_at_eos(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        inputs = torch.randint(0, config.vocab_size, (1, 3))

        def emit_eos(token_ids, targets=None):
            logits = torch.zeros(
                token_ids.size(0), token_ids.size(1), config.vocab_size
            )
            logits[:, :, 2] = 10.0
            return logits, None

        with patch.object(model, "forward", side_effect=emit_eos):
            generated = model.generate(
                inputs, max_new_tokens=10, temperature=0,
                eos_token_id=2, pad_token_id=0,
            )
        self.assertEqual(tuple(generated.shape), (1, 4))
        self.assertEqual(int(generated[0, -1]), 2)

    def test_masked_loss_equals_cross_entropy_on_only_selected_targets(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        model.eval()
        inputs = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        mask = torch.zeros((2, 8), dtype=torch.bool)
        mask[:, 4:7] = True

        logits, masked_loss = model(inputs, targets, mask)
        token_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        expected = token_losses[mask].mean()
        self.assertTrue(torch.allclose(masked_loss, expected, atol=1e-6))

    def test_masked_loss_rejects_empty_target_set(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        inputs = torch.randint(0, config.vocab_size, (1, 8))
        targets = torch.randint(0, config.vocab_size, (1, 8))
        with self.assertRaises(ValueError):
            model(inputs, targets, torch.zeros((1, 8), dtype=torch.bool))

    def test_generation_stops_at_end_of_turn(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        inputs = torch.randint(0, config.vocab_size, (1, 3))

        def emit_eot(token_ids, targets=None, loss_mask=None):
            logits = torch.zeros(
                token_ids.size(0), token_ids.size(1), config.vocab_size
            )
            logits[:, :, 10] = 10.0
            return logits, None

        with patch.object(model, "forward", side_effect=emit_eot):
            generated = model.generate(
                inputs, max_new_tokens=10, temperature=0,
                stop_token_ids=[10], pad_token_id=0,
            )
        self.assertEqual(tuple(generated.shape), (1, 4))
        self.assertEqual(int(generated[0, -1]), 10)

    def test_generation_forbids_structural_tokens(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        inputs = torch.randint(0, config.vocab_size, (1, 3))

        def emit_forbidden(token_ids, targets=None, loss_mask=None):
            logits = torch.zeros(
                token_ids.size(0), token_ids.size(1), config.vocab_size
            )
            logits[:, :, 6] = 10.0
            logits[:, :, 7] = 5.0
            return logits, None

        with patch.object(model, "forward", side_effect=emit_forbidden):
            generated = model.generate(
                inputs, max_new_tokens=1, temperature=0,
                forbidden_token_ids=[6], pad_token_id=0,
            )
        self.assertEqual(int(generated[0, -1]), 7)

    def test_training_can_skip_logit_return(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        model = JosephLanguageModel(config)
        inputs = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(inputs, targets, return_logits=False)
        self.assertIsNone(logits)
        self.assertTrue(torch.isfinite(loss).item())

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
