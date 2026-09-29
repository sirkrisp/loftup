from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Callback, LightningModule, Trainer

from train_loftup_stage1 import my_app


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.training_calls = 0

    def training_step(self, batch, batch_idx):
        self.training_calls += 1
        return (self.weight * batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        self.log('val/loss', (self.weight * batch[0]).square().mean())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.01)


class Stage1ModeTests(unittest.TestCase):
    def test_validate_only_routes_without_checkpoint_upload_or_fit(self):
        config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
        with tempfile.TemporaryDirectory() as directory, \
                initialize_config_dir(config_dir=config_dir, version_base='1.1'):
            cfg = compose(config_name='train_loftup_stage1', overrides=[
                'gpu=2x5090', 'validate_only=true', 'resume_from=auto', f'output_root={directory}'])
            with patch('train_loftup_stage1.resolve_resume_checkpoint', return_value='selected.ckpt'), \
                    patch('train_loftup_stage1.LoftUpStage1'), \
                    patch('train_loftup_stage1.create_training_loaders', return_value=('train', 'val')), \
                    patch('train_loftup_stage1.create_logging', return_value=([], [])), \
                    patch('train_loftup_stage1.configure_checkpoint_upload') as upload, \
                    patch('train_loftup_stage1.Trainer') as trainer:
                my_app.__wrapped__(cfg)
            trainer.return_value.validate.assert_called_once()
            self.assertEqual(trainer.return_value.validate.call_args.kwargs,
                             {'dataloaders': 'val', 'ckpt_path': 'selected.ckpt'})
            trainer.return_value.fit.assert_not_called()
            trainer.return_value.save_checkpoint.assert_not_called()
            upload.assert_not_called()
            self.assertFalse(trainer.call_args.kwargs['enable_checkpointing'])

    def test_validation_restores_weights_and_resume_finishes_original_epoch(self):
        loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)
        options = dict(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / 'resume.ckpt')

            class SaveMidEpoch(Callback):
                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    if batch_idx == 1:
                        trainer.save_checkpoint(checkpoint)

            first = Trainer(max_epochs=1, max_steps=2, callbacks=[SaveMidEpoch()], **options)
            first.fit(TinyModel(), loader)
            saved = torch.load(checkpoint, weights_only=False)
            modified = Path(checkpoint).stat().st_mtime_ns
            evaluator = TinyModel()
            Trainer(**options).validate(evaluator, loader, ckpt_path=checkpoint)
            self.assertEqual(evaluator.training_calls, 0)
            torch.testing.assert_close(evaluator.weight, saved['state_dict']['weight'])
            self.assertEqual(Path(checkpoint).stat().st_mtime_ns, modified)
            resumed = TinyModel()
            trainer = Trainer(max_epochs=1, **options)
            trainer.fit(resumed, loader, ckpt_path=checkpoint)
            self.assertEqual(trainer.global_step, 4)
            self.assertEqual(resumed.training_calls, 2)


if __name__ == '__main__':
    unittest.main()
