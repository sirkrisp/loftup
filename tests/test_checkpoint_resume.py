import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir

from checkpoint_resume import resolve_resume_checkpoint


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.final = self.root / 'experiment[1].ckpt'

    def checkpoint(self, relative, timestamp):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'checkpoint')
        os.utime(path, ns=(timestamp, timestamp))
        return str(path.resolve())

    def resolve(self, option='auto'):
        with contextlib.redirect_stdout(io.StringIO()):
            return resolve_resume_checkpoint(option, self.final)

    def test_latest_across_periodic_epoch_and_final_files(self):
        periodic = self.checkpoint('experiment[1]_10000.ckpt', 20)
        self.checkpoint('experiment[1]_5000.ckpt', 10)
        self.assertEqual(self.resolve(), periodic)
        epoch = self.checkpoint('experiment[1]/epoch=0-step=12000.ckpt', 30)
        self.assertEqual(self.resolve(), epoch)
        final = self.checkpoint('experiment[1].ckpt', 40)
        self.assertEqual(self.resolve('latest'), final)
        # Save time wins even when a previous run had a higher step count.
        periodic = self.checkpoint('experiment[1]_5000.ckpt', 50)
        self.assertEqual(self.resolve(), periodic)

    def test_other_experiments_partial_files_and_directories_are_ignored(self):
        expected = self.checkpoint('experiment[1]_5000.ckpt', 10)
        for name in ('experiment[1]_other_10000.ckpt', 'experiment1_10000.ckpt',
                     'experiment[1]_10000.ckpt.part', 'other/last.ckpt'):
            self.checkpoint(name, 100)
        (self.root / 'experiment[1]_15000.ckpt').mkdir()
        self.assertEqual(self.resolve(), expected)

    def test_no_checkpoints_starts_fresh_and_overrides_pass_through(self):
        self.assertIsNone(self.resolve())
        self.checkpoint('experiment[1]_5000.ckpt', 10)
        self.assertIsNone(self.resolve(None))
        self.assertEqual(self.resolve('/explicit/model.ckpt'), '/explicit/model.ckpt')

    def test_both_stage_configs_enable_auto_and_allow_overrides(self):
        config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
        with initialize_config_dir(config_dir=config_dir, version_base='1.1'):
            for stage in ('train_loftup_stage1', 'train_loftup_stage2'):
                self.assertEqual(compose(config_name=stage).resume_from, 'auto')
                self.assertIsNone(compose(config_name=stage, overrides=['resume_from=null']).resume_from)
                self.assertEqual(compose(config_name=stage,
                    overrides=['resume_from=/tmp/model.ckpt']).resume_from, '/tmp/model.ckpt')


if __name__ == '__main__':
    unittest.main()
