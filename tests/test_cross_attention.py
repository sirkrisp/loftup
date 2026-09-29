from copy import deepcopy
import unittest

import torch

from upsamplers.layers import CrossAttentionLayer


def dense_attention(layer, query, key, value):
    q = layer.norm_q(query).transpose(0, 1)
    k = layer.norm_kv(key).transpose(0, 1)
    v = layer.norm_kv(value).transpose(0, 1)
    output, _ = layer.attention(q, k, v, need_weights=True)
    return output.transpose(0, 1)


class CrossAttentionTests(unittest.TestCase):
    def test_outputs_and_gradients_match_dense_attention(self):
        torch.manual_seed(1)
        efficient = CrossAttentionLayer(20, heads=4).double()
        dense = deepcopy(efficient)
        inputs = [torch.randn(2, length, 20, dtype=torch.double, requires_grad=True)
                  for length in (13, 7, 7)]
        reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        actual = efficient(*inputs)
        expected = dense_attention(dense, *reference_inputs)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for actual_input, expected_input in zip(inputs, reference_inputs):
            torch.testing.assert_close(actual_input.grad, expected_input.grad)
        for actual_param, expected_param in zip(efficient.parameters(), dense.parameters()):
            torch.testing.assert_close(actual_param.grad, expected_param.grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_stage2_head_dimension_uses_less_gpu_memory(self):
        # DINOv3 S+ has 384 feature channels plus 20 position channels: 101/head.
        layer = CrossAttentionLayer(404, heads=4).cuda()
        inputs = [torch.randn(2, length, 404, device='cuda', requires_grad=True)
                  for length in (4096, 256, 256)]

        def peak(forward):
            layer.zero_grad(set_to_none=True)
            for value in inputs:
                value.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            with torch.autocast('cuda', dtype=torch.float16):
                output = forward(layer, *inputs)
                loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - before

        dense_peak = peak(dense_attention)
        efficient_peak = peak(lambda layer, *inputs: layer(*inputs))
        self.assertLess(efficient_peak, dense_peak)
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in inputs))


if __name__ == '__main__':
    unittest.main()
