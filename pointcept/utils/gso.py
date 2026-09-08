"""Memory-conscious Gradient Subspace Optimization for PointPiT.

The full Fisher and partition covariance matrices are D x D and are not
materialized.  Instead, their action is represented in the span of a short
history of task gradients and partition residuals.  The only eigendecomposition
therefore operates on a small (at most 2 * history_size) matrix.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import torch


class GradientSubspaceOptimizer:
    """Estimate and apply the partition-stable gradient subspace online."""

    def __init__(
        self,
        rank: int = 32,
        partition_weight: float = 0.5,
        history_size: int = 16,
        warmup_steps: int = 8,
        update_interval: int = 8,
        eps: float = 1e-8,
    ) -> None:
        if rank <= 0 or history_size <= 0:
            raise ValueError("rank and history_size must be positive")
        if partition_weight < 0:
            raise ValueError("partition_weight must be non-negative")
        if warmup_steps < 0 or update_interval <= 0:
            raise ValueError("warmup_steps must be >= 0 and update_interval > 0")

        self.rank = rank
        self.partition_weight = partition_weight
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.eps = eps
        self.step = 0
        self.task_history = deque(maxlen=history_size)
        self.partition_history = deque(maxlen=history_size)
        self.subspace = None

    @staticmethod
    def flatten_gradients(
        parameters: Iterable[torch.nn.Parameter], scale: float = 1.0
    ) -> torch.Tensor:
        """Flatten gradients, inserting zeros for parameters unused in a pass."""
        parts = []
        device = None
        for parameter in parameters:
            device = parameter.device
            if parameter.grad is None:
                parts.append(
                    torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
                )
            else:
                parts.append(parameter.grad.detach().float().reshape(-1) / scale)
        if not parts:
            if device is None:
                device = torch.device("cpu")
            return torch.empty(0, dtype=torch.float32, device=device)
        return torch.cat(parts)

    @staticmethod
    def assign_gradients(
        parameters: Iterable[torch.nn.Parameter], gradient: torch.Tensor
    ) -> None:
        """Write a flat vector back into each parameter's gradient buffer."""
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            value = gradient[offset : offset + size].view_as(parameter)
            parameter.grad = value.to(dtype=parameter.dtype)
            offset += size
        if offset != gradient.numel():
            raise ValueError("gradient size does not match the parameter list")

    def update(self, partition_gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        """Update the low-rank estimate and return the projected mean gradient."""
        if len(partition_gradients) < 2:
            raise ValueError("GSO requires gradients from at least two partitions")
        if any(g.shape != partition_gradients[0].shape for g in partition_gradients):
            raise ValueError("all partition gradients must have the same shape")

        stacked = torch.stack(
            [gradient.detach().float() for gradient in partition_gradients]
        )
        mean_gradient = stacked.mean(dim=0)
        # Let GradScaler skip an overflowing update without poisoning the
        # streaming covariance estimate used by future finite steps.
        if not torch.isfinite(stacked).all():
            return mean_gradient
        self.task_history.append(mean_gradient.clone())

        # K - 1 residuals contain all independent variation directions.
        for residual in stacked[:-1] - mean_gradient:
            self.partition_history.append(residual.clone())

        self.step += 1
        should_refresh = self.step >= self.warmup_steps and (
            self.subspace is None or self.step % self.update_interval == 0
        )
        if should_refresh:
            self.subspace = self._estimate_subspace()

        if self.subspace is None:
            return mean_gradient
        coefficients = self.subspace.transpose(0, 1) @ mean_gradient
        return self.subspace @ coefficients

    def _estimate_subspace(self):
        task = list(self.task_history)
        partition = list(self.partition_history)
        candidates = task + partition
        if not candidates:
            return None

        # Twice-reorthogonalized modified Gram-Schmidt is stable for the short
        # history used here and avoids a D x D allocation.
        basis = []
        for vector in candidates:
            candidate = vector.clone()
            for _ in range(2):
                for direction in basis:
                    candidate.add_(direction, alpha=-torch.dot(direction, candidate))
            norm = torch.linalg.vector_norm(candidate)
            if torch.isfinite(norm) and norm > self.eps:
                basis.append(candidate / norm)

        if not basis:
            return None
        q_basis = torch.stack(basis, dim=1)
        task_coordinates = torch.stack([q_basis.T @ g for g in task])
        fisher = task_coordinates.T @ task_coordinates / max(len(task), 1)

        if partition:
            partition_coordinates = torch.stack([q_basis.T @ g for g in partition])
            covariance = (
                partition_coordinates.T @ partition_coordinates / len(partition)
            )
            objective = fisher - self.partition_weight * covariance
        else:
            objective = fisher

        eigenvalues, eigenvectors = torch.linalg.eigh(objective)
        positive = torch.nonzero(eigenvalues > self.eps, as_tuple=False).flatten()
        if positive.numel() == 0:
            return None
        selected = positive[-min(self.rank, positive.numel()) :]
        return (q_basis @ eigenvectors[:, selected]).contiguous()


__all__ = ["GradientSubspaceOptimizer"]
