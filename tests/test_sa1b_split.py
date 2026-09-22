import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from datasets.sa1b import SA1B
from training_utils import validation_reconstruction_loss


class SA1BSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        folder = self.root / "sa1b"
        folder.mkdir()
        for index in range(40):
            (folder / f"sa_{index}.jpg").touch()
            (folder / f"sa_{index}.json").touch()
        (folder / "incomplete.jpg").touch()

    def dataset(self, split, **kwargs):
        return SA1B(self.root, split, None, None, **kwargs)

    def test_disjoint_complete_and_reproducible(self):
        train = self.dataset("train", sample_size=None)
        val = self.dataset("val", sample_size=None)
        self.assertEqual((len(train), len(val)), (38, 2))
        self.assertFalse(set(train.image_files) & set(val.image_files))
        self.assertEqual(len(set(train.image_files + val.image_files)), 40)
        self.assertEqual(val.image_files, self.dataset("val").image_files)
        self.assertTrue(all(Path(p).is_file() for p in train.label_files))

    def test_enumeration_order_does_not_change_split(self):
        expected = self.dataset("val").image_files
        paths = list((self.root / "sa1b").glob("*.jpg"))
        with patch("datasets.sa1b.glob.glob", return_value=[str(p) for p in reversed(paths)]):
            self.assertEqual(expected, self.dataset("val").image_files)

    def test_cap_fraction_and_seed(self):
        train = self.dataset("train", sample_size=20, val_fraction=0.2)
        val = self.dataset("val", sample_size=20, val_fraction=0.2)
        self.assertEqual((len(train), len(val)), (16, 4))
        self.assertFalse(set(train.image_files) & set(val.image_files))
        self.assertNotEqual(val.image_files, self.dataset("val", split_seed=7).image_files)
        self.assertEqual(len(self.dataset("val", sample_size=2)), 1)
        self.assertEqual(len(self.dataset("train", sample_size=2)), 1)

    def test_invalid_inputs(self):
        for options in ({"val_fraction": 0}, {"val_fraction": 1}, {"sample_size": 1}):
            with self.assertRaises(ValueError):
                self.dataset("train", **options)
        with self.assertRaises(ValueError):
            self.dataset("test")
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, "at least two"):
                SA1B(empty, "train", None, None)

    def test_validation_metric_detects_reconstruction_error(self):
        img = torch.ones(2, 3, 32, 32)
        backbone = torch.nn.AdaptiveAvgPool2d((14, 14))
        def upsample(features, guidance):
            return torch.ones_like(guidance) * 3
        loss = validation_reconstruction_loss(backbone, upsample, img, 32)
        self.assertAlmostEqual(loss.item(), 4.0)


if __name__ == "__main__":
    unittest.main()
