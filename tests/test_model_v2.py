import unittest
from unittest.mock import patch

import torch

from joseph.config_v2 import (
    INFERENCE_BUNDLE_MAX_BYTES,
    HuTaoLlamaConfig,
)
from joseph.model_v2 import HuTaoLlamaModel, RMSNorm, RotaryEmbedding


class HuTaoLlamaConfigTests(unittest.TestCase):
    def test_candidate_is_approximately_1_8b_and_fits_estimated_q4_bundle(self):
        config = HuTaoLlamaConfig()
        config.validate()
        self.assertEqual(config.estimated_parameter_count(), 1_782_827_520)
        estimate = config.validate_inference_estimate()
        self.assertGreater(estimate, 900_000_000)
        self.assertLess(estimate, 1_100_000_000)
        self.assertLessEqual(estimate, INFERENCE_BUNDLE_MAX_BYTES)

    def test_candidate_fp16_kv_cache_is_approximately_192_mib_at_4k(self):
        config = HuTaoLlamaConfig()
        self.assertEqual(config.estimated_kv_cache_bytes(), 201_326_592)

    def test_invalid_gqa_and_rope_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible by num_key_value_heads"):
            HuTaoLlamaConfig(num_attention_heads=20, num_key_value_heads=3).validate()
        with self.assertRaisesRegex(ValueError, "even for rotary"):
            HuTaoLlamaConfig(hidden_size=100, num_attention_heads=20).validate()


class HuTaoLlamaModelTests(unittest.TestCase):
    def small_config(self, **kwargs):
        values = {
            "vocab_size": 320,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32,
            "dropout": 0.0,
        }
        values.update(kwargs)
        return HuTaoLlamaConfig(**values)

    def test_forward_and_masked_loss_shapes(self):
        config = self.small_config()
        model = HuTaoLlamaModel(config)
        tokens = torch.randint(0, config.vocab_size, (2, 12))
        targets = torch.randint(0, config.vocab_size, (2, 12))
        mask = torch.zeros((2, 12))
        mask[:, 5:] = 1

        logits, loss = model(tokens, targets, mask)
        self.assertEqual(tuple(logits.shape), (2, 12, config.vocab_size))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_masked_loss_backpropagates_finite_gradients(self):
        config = self.small_config()
        model = HuTaoLlamaModel(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        mask = torch.zeros((2, 16))
        mask[:, 8:] = 1

        _, loss = model(tokens, targets, mask, return_logits=False)
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_tied_embeddings_share_the_same_parameter(self):
        model = HuTaoLlamaModel(self.small_config(tie_word_embeddings=True))
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_untied_output_head_has_expected_parameter_count(self):
        tied = self.small_config(tie_word_embeddings=True)
        untied = self.small_config(tie_word_embeddings=False)
        difference = (
            HuTaoLlamaConfig(**untied.to_dict()).estimated_parameter_count()
            - tied.estimated_parameter_count()
        )
        self.assertEqual(difference, tied.vocab_size * tied.hidden_size)

    def test_generation_stops_on_end_of_turn(self):
        config = self.small_config()
        model = HuTaoLlamaModel(config)
        stop_token_id = 7
        prompt = torch.tensor([[1, 2, 3]])
        forced_logits = torch.zeros(1, prompt.shape[1], config.vocab_size)
        forced_logits[:, :, stop_token_id] = 100.0
        # Patch the model forward pass so this test exercises stopping logic,
        # not the random model's untrained token preferences.
        with patch.object(model, "forward", return_value=(forced_logits, None)):
            generated = model.generate(
                prompt,
                max_new_tokens=8,
                temperature=0.0,
                stop_token_ids=[stop_token_id],
            )
        self.assertEqual(generated.shape[1], prompt.shape[1] + 1)
        self.assertEqual(int(generated[0, -1]), stop_token_id)

    def test_rotary_embedding_preserves_tensor_shape_and_device(self):
        rotary = RotaryEmbedding(head_dim=16, theta=10_000.0)
        query = torch.randn(2, 4, 10, 16)
        key = torch.randn(2, 4, 10, 16)
        rotated_query, rotated_key = rotary(query, key)
        self.assertEqual(rotated_query.shape, query.shape)
        self.assertEqual(rotated_key.shape, key.shape)
        self.assertTrue(torch.isfinite(rotated_query).all())
        self.assertTrue(torch.isfinite(rotated_key).all())

    def test_rms_norm_preserves_shape_and_is_finite(self):
        norm = RMSNorm(hidden_size=64, eps=1e-5)
        values = torch.randn(2, 10, 64)
        output = norm(values)
        self.assertEqual(output.shape, values.shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
