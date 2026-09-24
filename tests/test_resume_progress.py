import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Callback, LightningModule, Trainer

from vis.progress import ResumeProgressBar


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def training_step(self, batch, batch_idx):
        return (self.weight * batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class ResumeProgressTests(unittest.TestCase):
    def test_mid_epoch_resume_has_total_and_excludes_restored_batches_from_rate(self):
        loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "resume.ckpt")

            class SaveMidEpoch(Callback):
                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    if batch_idx == 1:
                        trainer.save_checkpoint(checkpoint)

            options = dict(accelerator="cpu", devices=1, logger=False,
                           enable_checkpointing=False, enable_model_summary=False,
                           max_epochs=1, reload_dataloaders_every_n_epochs=1)
            fresh_bar = ResumeProgressBar()
            trainer = Trainer(max_steps=2, callbacks=[fresh_bar, SaveMidEpoch()], **options)
            trainer.fit(TinyModel(), loader)
            self.assertEqual(fresh_bar.train_progress_bar.total, 4)
            self.assertEqual(fresh_bar.train_progress_bar.initial, 0)

            resumed_bar = ResumeProgressBar()
            trainer = Trainer(max_steps=4, callbacks=[resumed_bar], **options)
            trainer.fit(TinyModel(), loader, ckpt_path=checkpoint)
            bar = resumed_bar.train_progress_bar
            self.assertEqual(bar.total, 4)
            self.assertEqual(bar.initial, 2)
            self.assertEqual(bar.n, 4)
            self.assertEqual(bar.desc, "Epoch 0: ")
            self.assertEqual(trainer.global_step, 4)
