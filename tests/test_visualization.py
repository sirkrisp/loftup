import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from vis.callbacks import FeatureVisualization, feature_panel


class Upsampler(torch.nn.Module):
    def forward(self, low, image):
        return torch.nn.functional.interpolate(low, image.shape[-2:], mode="bilinear", align_corners=False)


class VisualizationTests(unittest.TestCase):
    def test_constant_features_are_finite_and_shared_colors_match(self):
        image = torch.ones(1, 3, 8, 8)
        low = torch.ones(1, 4, 2, 2)
        high = torch.ones(1, 4, 8, 8)
        panel = np.array(feature_panel(image, low, high))
        self.assertEqual(panel.shape, (32, 32, 3))
        np.testing.assert_array_equal(panel[24:, 8:16], panel[24:, 24:32])

    def test_callback_restores_modes_and_logs_once_per_step(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = TensorBoardLogger(directory)
            trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False,
                                      global_step=2, log_dir=logger.log_dir, loggers=[logger])
            module = SimpleNamespace(model=torch.nn.AdaptiveAvgPool2d((4, 4)),
                                     upsampler=Upsampler(), device=torch.device("cpu"))
            module.model.eval()
            module.upsampler.train()
            callback = FeatureVisualization(image_dir=directory, max_images=1,
                                            input_size=16, output_size=16, every_n_steps=2)
            batch = {"img": torch.randn(1, 3, 16, 16), "img_path": ["sample.jpg"]}
            callback.on_validation_batch_end(trainer, module, None, batch, 0)
            before = torch.random.get_rng_state().clone()
            callback.on_train_batch_end(trainer, module, None, batch, 0)
            torch.testing.assert_close(before, torch.random.get_rng_state())
            self.assertFalse(module.model.training)
            self.assertTrue(module.upsampler.training)
            files = list(Path(logger.log_dir).rglob("*.png"))
            self.assertEqual(len(files), 1)
            modified = files[0].stat().st_mtime_ns
            callback.on_validation_epoch_end(trainer, module)
            self.assertEqual(files[0].stat().st_mtime_ns, modified)
            logger.finalize("success")

    def test_fixed_images_and_rank_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (32, 48)).save(Path(directory) / "example.png")
            callback = FeatureVisualization(image_dir=directory, output_size=16)
            callback.setup(SimpleNamespace(is_global_zero=False), None, "fit")
            self.assertEqual(callback.examples, [])
            callback.setup(SimpleNamespace(is_global_zero=True), None, "fit")
            self.assertEqual(callback.examples[0][1].shape, (1, 3, 16, 16))
            callback._log(SimpleNamespace(is_global_zero=False), None)
            callback._log(SimpleNamespace(is_global_zero=True, sanity_checking=True), None)


if __name__ == "__main__":
    unittest.main()
