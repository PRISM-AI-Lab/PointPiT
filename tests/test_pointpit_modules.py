"""CPU smoke tests for the two PointPiT components."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pointpit = load_module("pointpit_module", "pointcept/models/peft/pointpit.py")
gso = load_module("gso_module", "pointcept/utils/gso.py")


class SceneAwareStructuralAdapterTest(unittest.TestCase):
    def test_shape_identity_initialization_and_backward(self):
        adapter = pointpit.SceneAwareStructuralAdapter(8, hidden_channels=4)
        point = SimpleNamespace(
            feat=torch.randn(7, 8, requires_grad=True),
            batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        )
        output = adapter(point)

        self.assertEqual(output.shape, point.feat.shape)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))
        output.sum().backward()
        self.assertIsNotNone(adapter.up.weight.grad)

    def test_scene_pooling_does_not_mix_batches(self):
        features = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        pooled = pointpit._scene_mean(features, batch)
        expected = torch.tensor([[2.0, 4.0], [10.0, 20.0]])
        self.assertTrue(torch.equal(pooled, expected))

    def test_half_precision_pooling_uses_float_accumulation(self):
        features = torch.ones(70_000, 1, dtype=torch.float16)
        batch = torch.zeros(70_000, dtype=torch.long)
        pooled = pointpit._scene_mean(features, batch)
        self.assertTrue(torch.equal(pooled, torch.ones_like(pooled)))


class GradientSubspaceOptimizerTest(unittest.TestCase):
    def test_partition_variant_direction_is_suppressed(self):
        projector = gso.GradientSubspaceOptimizer(
            rank=1,
            partition_weight=2.0,
            history_size=4,
            warmup_steps=1,
            update_interval=1,
        )
        projected = projector.update(
            [torch.tensor([1.0, 1.0]), torch.tensor([1.0, -1.0])]
        )
        self.assertGreater(projected[0].abs().item(), 0.9)
        self.assertLess(projected[1].abs().item(), 1e-5)

    def test_flatten_and_assign_round_trip(self):
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(1))
        first.grad = torch.tensor([2.0, 4.0])
        second.grad = torch.tensor([6.0])
        flat = gso.GradientSubspaceOptimizer.flatten_gradients(
            [first, second], scale=2.0
        )
        self.assertTrue(torch.equal(flat, torch.tensor([1.0, 2.0, 3.0])))
        gso.GradientSubspaceOptimizer.assign_gradients(
            [first, second], torch.tensor([7.0, 8.0, 9.0])
        )
        self.assertTrue(torch.equal(first.grad, torch.tensor([7.0, 8.0])))
        self.assertTrue(torch.equal(second.grad, torch.tensor([9.0])))


if __name__ == "__main__":
    unittest.main()
