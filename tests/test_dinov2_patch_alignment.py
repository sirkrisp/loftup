import unittest
from unittest.mock import patch

import torch

from featurizers.DINOv2 import DINOv2Featurizer, DinoVisionTransformer
from training_utils import AttentionDownsampler, apply_jitter


class DINOv2PatchAlignmentTests(unittest.TestCase):
    def setUp(self):
        # Exercise the real patch embedding and transformer without downloading weights.
        backbone = DinoVisionTransformer(
            img_size=224, patch_size=14, embed_dim=24, depth=1, num_heads=3,
        ).eval()
        with patch('torch.hub.load', return_value=backbone):
            self.model = DINOv2Featurizer('dinov2_vits14', 14, 'token').eval()
        self.model.dim = 24

    @torch.no_grad()
    def test_jittered_width_matches_reconstruction_grid(self):
        image = torch.randn(1, 3, 224, 518)
        jittered = apply_jitter(image, 0, (0, 0, 1.0, 1.1, 0))
        self.assertEqual(jittered.shape[-1], 569)
        features = self.model(jittered)
        reference = self.model(jittered[..., :560])
        torch.testing.assert_close(features, reference)
        downsampled = AttentionDownsampler(3, 14, 16)(jittered, jittered)
        self.assertEqual(features.shape[-2:], downsampled.shape[-2:])

    @torch.no_grad()
    def test_aligned_inputs_unchanged_and_both_axes_trimmed(self):
        for height, width in ((224, 224), (239, 227)):
            with self.subTest(height=height, width=width):
                image = torch.randn(1, 3, height, width)
                aligned = image[..., :height // 14 * 14, :width // 14 * 14]
                expected = self.model.model.forward_features(aligned)['x_norm_patchtokens']
                actual = self.model(image).flatten(2).transpose(1, 2)
                torch.testing.assert_close(actual, expected)

    def test_too_small_image_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'at least 14'):
            self.model(torch.randn(1, 3, 224, 13))


if __name__ == '__main__':
    unittest.main()
