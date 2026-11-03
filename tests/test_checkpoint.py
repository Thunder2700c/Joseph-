import tempfile
import unittest
from pathlib import Path

import torch

from joseph.config import ModelConfig
from joseph.model import JosephLanguageModel
from training.checkpoint import load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_data_parallel_checkpoint_stores_unwrapped_state_dict(self):
        config = ModelConfig(
            vocab_size=270, d_model=32, n_layers=1, n_heads=4,
            block_size=16, dropout=0.0,
        )
        base_model = JosephLanguageModel(config)
        wrapped_model = torch.nn.DataParallel(base_model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parallel.pt"
            save_checkpoint(path, wrapped_model, config, 3)
            payload = load_checkpoint(path)
            self.assertEqual(payload["step"], 3)
            self.assertFalse(any(key.startswith("module.") for key in payload["model_state"]))
            restored = JosephLanguageModel(config)
            restored.load_state_dict(payload["model_state"], strict=True)
            for expected, actual in zip(base_model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
