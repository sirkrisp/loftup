from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer

from train_loftup_stage2 import LoftUpStage2
from ema import EMA
from training_utils import load_stage1_training_weights


class TinyUpsampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.5))

    def forward(self, features, image):
        return self.scale * F.interpolate(features, image.shape[-2:], mode='bilinear', align_corners=False)


class Stage2InitializationTests(unittest.TestCase):
    def test_gpu_preset_and_explicit_overrides(self):
        config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
        with initialize_config_dir(config_dir=config_dir, version_base='1.1'):
            cfg = compose(config_name='train_loftup_stage2')
            self.assertEqual((cfg.num_gpus, cfg.batch_size, cfg.accumulation_steps), (4, 2, 1))
            cfg = compose(config_name='train_loftup_stage2', overrides=['gpu=2x5090'])
            self.assertEqual((cfg.num_gpus, cfg.batch_size, cfg.accumulation_steps), (2, 2, 2))
            self.assertEqual(cfg.validation_every_n_steps * cfg.accumulation_steps, 2000)
            cfg = compose(config_name='train_loftup_stage2', overrides=['gpu=2x5090', 'batch_size=1'])
            self.assertEqual((cfg.num_gpus, cfg.batch_size, cfg.accumulation_steps), (2, 1, 2))

    def test_validation_interval_counts_accumulated_optimizer_steps(self):
        class AccumulatingModel(LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.automatic_optimization = False
                self.validation_steps = []

            def training_step(self, batch, batch_idx):
                opt = self.optimizers()
                if batch_idx % 2 == 0:
                    opt.zero_grad()
                self.manual_backward(self.weight.square() / 2)
                if batch_idx % 2 == 1:
                    opt.step()

            def validation_step(self, batch, batch_idx):
                self.validation_steps.append(self.global_step)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=.01)

        train = DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)
        val = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
        model = AccumulatingModel()
        trainer = Trainer(accelerator='cpu', devices=1, max_epochs=1,
                          val_check_interval=min(len(train), 2 * 2), num_sanity_val_steps=0,
                          logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model, train, val)
        self.assertEqual(model.validation_steps, [2, 4])

    def test_pretrained_student_trains_and_ema_teacher_produces_hr_loss(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            source_model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.AvgPool2d(4))
            source_upsampler = TinyUpsampler()
            with torch.no_grad():
                source_upsampler.scale.fill_(2.0)
            state = {f'model.{key}': value for key, value in source_model.state_dict().items()}
            state.update({f'upsampler.{key}': value for key, value in source_upsampler.state_dict().items()})
            checkpoint = str(Path(directory) / 'stage1.ckpt')
            torch.save({'state_dict': state, 'global_step': 120000}, checkpoint)
            featurizer = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.AvgPool2d(4))
            with patch('train_loftup_stage2.get_featurizer', return_value=(featurizer, 4, 3)), \
                    patch('train_loftup_stage2.get_upsampler', return_value=TinyUpsampler()):
                model = LoftUpStage2(
                    model_type='test', activation_type='token', n_jitters=1, max_pad=0,
                    max_zoom=0, max_rotate=0, kernel_size=4, final_size=4, lr=.01,
                    random_projection=None, predicted_uncertainty=False, filter_ent_weight=0,
                    tv_weight=0, upsampler='loftup', downsampler='attention',
                    chkpt_dir=str(Path(directory) / 'stage2.ckpt'), hr_res=32, hr_weight=20,
                    consistency_method='bilinear', pretrained_upsampler=checkpoint,
                    affinity_loss=True, rec_weight=1, l1_affinity=False, use_prototypes=False,
                    n_freqs=20, sam_mask_reg=0, sam_mask_hr_alpha=0, sam_mask_hr_reg=0,
                    use_crop_upsampler=True)
            for key, value in model.model.state_dict().items():
                torch.testing.assert_close(value, source_model.state_dict()[key])
            self.assertTrue(model.upsampler.scale.requires_grad)
            self.assertIsInstance(model.crop_upsampler, EMA)
            teacher = model.crop_upsampler.ema_model
            self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
            torch.testing.assert_close(model.upsampler.scale, source_upsampler.scale)
            torch.testing.assert_close(teacher.scale, source_upsampler.scale)
            self.assertEqual(model.ema_update_after, 0)
            teacher_before = teacher.scale.detach().clone()
            optimizer = model.configure_optimizers()
            model.accumulation_steps = 2
            model._trainer = SimpleNamespace(num_training_batches=21, global_step=0)
            losses = []

            def backward(loss):
                losses.append(float(loss.detach()))
                loss.backward()

            with patch.object(model, 'optimizers', return_value=optimizer), \
                    patch.object(model, 'manual_backward', side_effect=backward), \
                    patch.object(optimizer, 'step', wraps=optimizer.step) as step, \
                    patch.object(optimizer, 'zero_grad', wraps=optimizer.zero_grad) as zero, \
                    patch.object(model, 'clip_gradients'), patch.object(model, 'log') as log, \
                    patch('train_loftup_stage2.random.choice', return_value=16):
                for batch_idx in range(21):
                    previous_teacher = teacher.scale.detach().clone()
                    model.training_step({'img': torch.randn(1, 3, 32, 32), 'label': None}, batch_idx)
                    updates = (batch_idx + 1) // 2 + int(batch_idx == 20)
                    self.assertEqual(step.call_count, updates)
                    self.assertEqual(model.crop_upsampler.step.item(), updates)
                    self.assertIsNone(teacher.scale.grad)
                    self.assertTrue(model.upsampler.training)
                    self.assertFalse(teacher.training)
                    if batch_idx == 1:
                        # First optimizer step initializes EMA from the updated student.
                        torch.testing.assert_close(teacher.scale, model.upsampler.scale)
                    elif batch_idx == 20:
                        # The next scheduled EMA update averages the current student.
                        decay = model.crop_upsampler.get_current_decay()
                        expected = previous_teacher * decay + model.upsampler.scale.detach() * (1 - decay)
                        torch.testing.assert_close(teacher.scale, expected)
                        self.assertFalse(torch.equal(teacher.scale, previous_teacher))
                    else:
                        torch.testing.assert_close(teacher.scale, previous_teacher)
                self.assertEqual(zero.call_count, 11)
            totals = [float(call.args[1]) for call in log.call_args_list if call.args[0] == 'loss/total']
            self.assertEqual(len(totals), 21)
            for backward_loss, total, divisor in zip(losses, totals, [2] * 20 + [1]):
                self.assertAlmostEqual(backward_loss, total / divisor, places=6)
            logged = {call.args[0]: call.args[1] for call in log.call_args_list}
            self.assertGreater(float(logged['loss/hr']), 0)
            self.assertFalse(torch.equal(teacher.scale, teacher_before))
            self.assertFalse(torch.equal(model.upsampler.scale.detach(), teacher_before))

    def test_incompatible_stage1_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'bad.ckpt'
            torch.save({'state_dict': {}}, checkpoint)
            with self.assertRaises(RuntimeError):
                load_stage1_training_weights(torch.nn.Linear(1, 1), TinyUpsampler(), checkpoint)


if __name__ == '__main__':
    unittest.main()
