import unittest

import torch
import torch.nn.functional as F

from training_utils import affinity_mse_loss, compute_affinity_matrix_batch


class AffinityLossTests(unittest.TestCase):
    def test_matches_dense_values_and_gradients(self):
        for alpha in (0, .1, .25):
            with self.subTest(alpha=alpha):
                x = torch.randn(2, 5, 8, 10, dtype=torch.double, requires_grad=True)
                y = torch.randn_like(x, requires_grad=True)
                expected = F.mse_loss(compute_affinity_matrix_batch(x, alpha),
                                      compute_affinity_matrix_batch(y, alpha))
                expected_grads = torch.autograd.grad(expected, (x, y))
                actual = affinity_mse_loss(x, y, alpha)
                actual_grads = torch.autograd.grad(actual, (x, y))
                torch.testing.assert_close(actual, expected)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_identical_and_zero_features_are_finite(self):
        for features in (torch.zeros(1, 4, 8, 8), torch.randn(1, 4, 8, 8)):
            features.requires_grad_()
            loss = affinity_mse_loss(features, features.detach())
            torch.testing.assert_close(loss, torch.tensor(0.), atol=1e-6, rtol=0)
            loss.backward()
            self.assertTrue(torch.isfinite(features.grad).all())

    def test_large_spatial_grid_has_bounded_saved_tensors(self):
        x = torch.randn(1, 8, 128, 128, requires_grad=True)
        y = torch.randn_like(x)
        saved_sizes = []

        def pack(tensor):
            saved_sizes.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = affinity_mse_loss(x, y)
            loss.backward()
        self.assertLessEqual(max(saved_sizes), x.numel())


if __name__ == '__main__':
    unittest.main()
