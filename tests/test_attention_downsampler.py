import unittest

import torch
import torch.nn.functional as F

from training_utils import AttentionDownsampler, apply_jitter


class AttentionDownsamplerTests(unittest.TestCase):
    def test_rectangular_grids_and_gradients(self):
        for kernel in (14, 16):
            for height, width in ((224, 208), (208, 224), (224, 240), (239, 227), (16, 31)):
                with self.subTest(kernel=kernel, height=height, width=width):
                    model = AttentionDownsampler(3, kernel, 224 // kernel)
                    x = torch.randn(1, 3, height, width, requires_grad=True)
                    output = model(x, x)
                    self.assertEqual(output.shape, (1, 3, height // kernel, width // kernel))
                    output.square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertTrue(torch.isfinite(model.w.grad).all())

    def test_width_only_jitter_regression(self):
        image = torch.randn(1, 3, 224, 224)
        jittered = apply_jitter(image, 0, (0, 0, 1.0, 0.95, 0))
        self.assertEqual(jittered.shape[-2:], (224, 212))
        model = AttentionDownsampler(3, 16, 14)
        self.assertEqual(model(jittered, jittered).shape[-2:], (14, 13))

    def test_uniform_attention_matches_patch_averages(self):
        model = AttentionDownsampler(2, 16, 14, blur_attn=False)
        with torch.no_grad():
            model.w.zero_()
            model.b.zero_()
        x = torch.randn(1, 2, 224, 208)
        torch.testing.assert_close(model(x, x), F.avg_pool2d(x, 16, stride=16))

    def test_too_small_input(self):
        model = AttentionDownsampler(3, 16, 14)
        x = torch.randn(1, 3, 224, 15)
        with self.assertRaisesRegex(ValueError, "at least"):
            model(x, x)


if __name__ == "__main__":
    unittest.main()
