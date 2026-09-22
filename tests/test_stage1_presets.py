import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir

from featurizers import get_featurizer


class Stage1PresetTests(unittest.TestCase):
    def test_all_gpu_and_backbone_combinations(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            for gpu in ("1x3090", "1xh100", "4x3090", "4xh100", "4x4090"):
                for model in ("dinov3splus", "dinov3base"):
                    with self.subTest(gpu=gpu, model=model):
                        cfg = compose(config_name="train_loftup_stage1", overrides=[f"gpu={gpu}", f"model_type={model}"])
                        self.assertEqual(cfg.batch_size * cfg.num_gpus * cfg.accumulation_steps, 8)
                        self.assertEqual(cfg.num_gpus, int(gpu[0]))
                        self.assertEqual(cfg.lr, 1e-4)
                        self.assertEqual(cfg.epochs, 1)
                        self.assertEqual(cfg.dataset, "sa1b")
                        self.assertEqual(cfg.sa1b_sample_size, 1000000)

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
