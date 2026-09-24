import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from vis.callbacks import FeatureVisualization, feature_panel


class Upsampler(torch.nn.Module):
    def forward(self, low, image):
        return torch.nn.functional.interpolate(low, image.shape[-2:], mode="bilinear", align_corners=False)


class PreviewModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.model = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.upsampler = Upsampler()

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class VisualizationTests(unittest.TestCase):
    def test_empty_cache_is_seeded_without_changing_rng_and_uploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            tensorboard = TensorBoardLogger(directory)
            wandb = Mock(spec=WandbLogger)
            loader = DataLoader(TensorDataset(torch.randn(8, 3, 16, 16)), batch_size=1)
            trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False,
                                      global_step=500, val_dataloaders=[loader],
                                      loggers=[tensorboard, wandb])
            callback = FeatureVisualization(image_dir=directory, max_images=2,
                                            input_size=16, output_size=16)
            before = torch.random.get_rng_state().clone()
            callback.on_train_start(trainer, PreviewModel())
            torch.testing.assert_close(before, torch.random.get_rng_state())
            self.assertEqual(len(callback.examples), 2)
            callback.on_train_batch_end(trainer, PreviewModel(), None, None, 0)
            wandb.log_image.assert_called_once()
            self.assertEqual(wandb.log_image.call_args.kwargs['step'], 500)
            self.assertEqual(len(wandb.log_image.call_args.kwargs['images']), 2)
            # Existing caches and nonzero ranks must not open another stream.
            trainer.val_dataloaders = Mock(side_effect=AssertionError('unexpected read'))
            callback.on_train_start(trainer, None)
            trainer.is_global_zero = False
            callback.examples = []
            callback.on_train_start(trainer, None)
            tensorboard.finalize('success')

    def test_mid_epoch_resume_logs_before_next_validation_without_callback_state(self):
        loader = DataLoader(TensorDataset(torch.ones(8, 3, 16, 16)), batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / 'resume.ckpt')

            class SaveMidEpoch(Callback):
                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    if batch_idx == 1:
                        trainer.save_checkpoint(checkpoint)

            options = dict(accelerator='cpu', devices=1, max_epochs=1,
                           enable_checkpointing=False, enable_model_summary=False,
                           enable_progress_bar=False)
            first = Trainer(max_steps=2, logger=False, callbacks=[SaveMidEpoch()], **options)
            first.fit(PreviewModel(), loader, loader)
            callback = FeatureVisualization(image_dir=directory, max_images=1,
                                            input_size=16, output_size=16, every_n_steps=3)
            logger = TensorBoardLogger(directory)
            resumed = Trainer(max_steps=4, logger=logger, callbacks=[callback], **options)
            resumed.fit(PreviewModel(), loader, loader, ckpt_path=checkpoint)
            files = list(Path(logger.log_dir).rglob('step_00000003_0.png'))
            self.assertEqual(len(files), 1)
            self.assertEqual(callback._last_step, 3)

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
