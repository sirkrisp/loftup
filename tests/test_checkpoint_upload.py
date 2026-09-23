import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from checkpoint_upload import HubCheckpointIO, HubUploadPreflight


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.trainer.save_checkpoint(str(Path(self.trainer.default_root_dir) / 'periodic.ckpt'))
        return self.weight * batch.mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.01)


class CheckpointUploadTests(unittest.TestCase):
    @patch('checkpoint_upload.HfApi')
    def test_all_lightning_save_paths_upload_after_local_save(self, factory):
        api = factory.return_value
        api.repo_info.return_value.private = True
        uploaded = []
        def upload(**kwargs):
            self.assertTrue(Path(kwargs['path_or_fileobj']).is_file())
            self.assertIn('state_dict', torch.load(kwargs['path_or_fileobj'], weights_only=False))
            self.assertEqual(kwargs['repo_id'], 'Krispin/loftup')
            uploaded.append(kwargs['path_in_repo'])
        api.upload_file.side_effect = upload
        with tempfile.TemporaryDirectory() as root:
            io = HubCheckpointIO('Krispin/loftup', 'stage1/test/run')
            trainer = Trainer(default_root_dir=root, accelerator='cpu', devices=1,
                max_epochs=1, logger=False, enable_progress_bar=False, enable_model_summary=False,
                plugins=[io], callbacks=[HubUploadPreflight(io), ModelCheckpoint(dirpath=root)])
            trainer.fit(TinyModel(), DataLoader(torch.ones(2, 1), batch_size=1))
            trainer.save_checkpoint(str(Path(root) / 'final.ckpt'))
        self.assertEqual(len(uploaded), 3)
        self.assertTrue(any(x.endswith('/periodic.ckpt') for x in uploaded))
        self.assertTrue(any(x.endswith('/final.ckpt') for x in uploaded))
        api.create_repo.assert_called_once_with(repo_id='Krispin/loftup', repo_type='model', private=True, exist_ok=True)

    @patch('checkpoint_upload.time.sleep')
    @patch('checkpoint_upload.HfApi')
    def test_failure_retries_and_keeps_local_checkpoint(self, factory, sleep):
        factory.return_value.repo_info.return_value.private = True
        factory.return_value.upload_file.side_effect = OSError('offline')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'model.ckpt'
            with self.assertRaisesRegex(RuntimeError, 'saved locally'):
                HubCheckpointIO('Krispin/loftup', 'stage2/run').save_checkpoint({'x': 1}, path)
            self.assertEqual(torch.load(path, weights_only=True), {'x': 1})
        self.assertEqual(factory.return_value.upload_file.call_count, 3)

    @patch('checkpoint_upload.HfApi')
    def test_public_repo_rejected(self, factory):
        factory.return_value.repo_info.return_value.private = False
        with self.assertRaisesRegex(RuntimeError, 'public repository'):
            HubCheckpointIO('Krispin/loftup', 'stage1/run').prepare()
        factory.return_value.upload_file.assert_not_called()

    def test_nonzero_rank_does_not_prepare(self):
        with patch.object(HubCheckpointIO, 'prepare') as prepare:
            trainer = SimpleNamespace(is_global_zero=False,
                strategy=SimpleNamespace(broadcast=lambda x: x))
            HubUploadPreflight(HubCheckpointIO('Krispin/loftup', 'stage1/run')).on_fit_start(trainer, None)
            prepare.assert_not_called()
