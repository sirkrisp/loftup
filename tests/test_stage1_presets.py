import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from featurizers import get_featurizer
from train_loftup_stage1 import LoftUpStage1


class Stage1PresetTests(unittest.TestCase):
    def test_all_gpu_and_backbone_combinations(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            for gpu in ("1x3090", "1xh100", "2x5090", "4x3090", "4xh100", "4x4090", "8xv100", "8x5090"):
                for model in ("dinov3splus", "dinov3base"):
                    with self.subTest(gpu=gpu, model=model):
                        cfg = compose(config_name="train_loftup_stage1", overrides=[f"gpu={gpu}", f"model_type={model}"])
                        self.assertEqual(cfg.batch_size * cfg.num_gpus * cfg.accumulation_steps, 8)
                        self.assertEqual(cfg.num_gpus, int(gpu[0]))
                        self.assertEqual(cfg.lr, 1e-4)
                        self.assertEqual(cfg.weight_decay, 0.0)
                        self.assertEqual(cfg.epochs, 1)
                        self.assertEqual(cfg.dataset, "sa1b_webdataset")
                        self.assertEqual(cfg.sa1b_sample_size, 1000000)

    def test_released_optimizer_and_frozen_parameter_exclusion(self):
        model = LoftUpStage1.__new__(LoftUpStage1)
        torch.nn.Module.__init__(model)
        model.trainable = torch.nn.Parameter(torch.ones(1))
        model.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        model.lr = 1e-4
        model.weight_decay = 0.0
        optimizer = model.configure_optimizers()
        reference = torch.optim.NAdam([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
        self.assertIsInstance(optimizer, torch.optim.NAdam)
        self.assertEqual(optimizer.defaults, reference.defaults)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        self.assertIs(optimizer.param_groups[0]["params"][0], model.trainable)

    def test_explicit_batch_override(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name="train_loftup_stage1", overrides=["gpu=1xh100", "batch_size=1", "accumulation_steps=8"])
            self.assertEqual((cfg.batch_size, cfg.accumulation_steps), (1, 8))

    def test_base_selects_correct_checkpoint_and_dimensions(self):
        with patch("featurizers.DINOv3.DINOv3Featurizer") as factory:
            model, patch_size, dim = get_featurizer("dinov3base")
            factory.assert_called_once_with("dinov3_vitb16", 16, "token")
            self.assertIs(model, factory.return_value)
            self.assertEqual((patch_size, dim), (16, 768))


if __name__ == "__main__":
    unittest.main()
