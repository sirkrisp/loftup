from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import torch
import torch.nn.functional as F

from train_loftup_stage2 import LoftUpStage2
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
            cfg = compose(config_name='train_loftup_stage2', overrides=['gpu=2x5090', 'batch_size=1'])
            self.assertEqual((cfg.num_gpus, cfg.batch_size, cfg.accumulation_steps), (2, 1, 2))

    def test_pretrained_student_trains_and_fixed_teacher_produces_hr_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            source_model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.AvgPool2d(4))
            source_upsampler = TinyUpsampler()
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
            self.assertFalse(any(p.requires_grad for p in model.crop_upsampler.parameters()))
            self.assertEqual(model.ema_update_after, 0)
            teacher_before = model.crop_upsampler.scale.detach().clone()
            optimizer = model.configure_optimizers()
            model.accumulation_steps = 2
            model._trainer = SimpleNamespace(num_training_batches=3, global_step=0)
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
                model.training_step({'img': torch.randn(1, 3, 32, 32), 'label': None}, 0)
                step.assert_not_called()
                torch.testing.assert_close(model.upsampler.scale, teacher_before)
                model.training_step({'img': torch.randn(1, 3, 32, 32), 'label': None}, 1)
                self.assertEqual(step.call_count, 1)
                model.training_step({'img': torch.randn(1, 3, 32, 32), 'label': None}, 2)
                self.assertEqual(step.call_count, 2)
                self.assertEqual(zero.call_count, 2)
            totals = [call.args[1] for call in log.call_args_list if call.args[0] == 'loss/total']
            for backward_loss, total, divisor in zip(losses, totals, (2, 2, 1)):
                self.assertAlmostEqual(backward_loss, total / divisor, places=6)
            logged = {call.args[0]: call.args[1] for call in log.call_args_list}
            self.assertGreater(float(logged['loss/hr']), 0)
            torch.testing.assert_close(model.crop_upsampler.scale, teacher_before)
            self.assertFalse(torch.equal(model.upsampler.scale.detach(), teacher_before))

    def test_incompatible_stage1_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'bad.ckpt'
            torch.save({'state_dict': {}}, checkpoint)
            with self.assertRaises(RuntimeError):
                load_stage1_training_weights(torch.nn.Linear(1, 1), TinyUpsampler(), checkpoint)


if __name__ == '__main__':
    unittest.main()
