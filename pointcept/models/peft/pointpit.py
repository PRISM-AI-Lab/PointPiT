"""PointPiT parameter-efficient tuning modules.

This file implements the Scene-aware Structural Adapter (SSA) described in
the PointPiT paper.  The module is deliberately independent of a particular
point-cloud backbone: it only expects a Pointcept ``Point`` carrying ``feat``
and ``batch`` tensors.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _scene_mean(features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Average variable-length point features independently for each scene."""
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, C], got {features.shape}")
    if batch.ndim != 1 or batch.numel() != features.shape[0]:
        raise ValueError("batch must have shape [N] and align with features")
    if batch.numel() == 0:
        return features.new_empty((0, features.shape[1]))

    batch = batch.long()
    num_scenes = int(batch.max().item()) + 1
    accumulation_dtype = (
        torch.float32
        if features.dtype in (torch.float16, torch.bfloat16)
        else features.dtype
    )
    pooled = torch.zeros(
        (num_scenes, features.shape[1]),
        dtype=accumulation_dtype,
        device=features.device,
    )
    pooled.index_add_(0, batch, features.to(accumulation_dtype))
    count = torch.bincount(batch, minlength=num_scenes).to(accumulation_dtype)
    return (pooled / count.clamp_min_(1).unsqueeze(1)).to(features.dtype)


def _scene_softmax(scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over points belonging to the same scene."""
    if scores.numel() == 0:
        return scores
    output = torch.empty_like(scores)
    # Scene batches are small in segmentation workloads.  The explicit loop
    # avoids imposing another CUDA-extension dependency on this PEFT module.
    for scene_idx in range(int(batch.max().item()) + 1):
        mask = batch == scene_idx
        output[mask] = torch.softmax(scores[mask].float(), dim=0).to(scores.dtype)
    return output


class SceneAwareStructuralAdapter(nn.Module):
    """Fuse local token adaptation with a global scene descriptor.

    The down/up bottleneck provides the local branch.  The same compact latent
    space is used to form a scene query and attention-pooled context, keeping
    the parameter overhead below one percent for the Sonata/PTv3 backbone.
    A zero-initialized up projection makes insertion into a pretrained network
    an identity operation at initialization.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("channels and hidden_channels must be positive")

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Linear(channels, hidden_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_channels, channels)

        # Query/value projections operate in the compact bottleneck space,
        # avoiding quadratic growth in the backbone channel dimension.  The
        # gate is point-wise and shared across channels.
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.value_scale = nn.Parameter(torch.ones(hidden_channels))
        self.gate = nn.Linear(hidden_channels, 1)
        self.local_scale = nn.Parameter(torch.ones(channels))
        self.context_scale = nn.Parameter(torch.ones(channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.eye_(self.query.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.ones_(self.value_scale)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.ones_(self.local_scale)
        nn.init.ones_(self.context_scale)

    def forward(self, point):
        features = self.norm(point.feat)
        batch = point.batch.long()
        hidden = self.dropout(self.activation(self.down(features)))

        local = self.up(hidden)
        scene_query = self.query(_scene_mean(hidden, batch))
        scores = (hidden * scene_query[batch]).sum(dim=-1)
        scores = scores / math.sqrt(self.hidden_channels)
        attention = _scene_softmax(scores, batch)

        weighted_value = hidden * self.value_scale * attention.unsqueeze(1)
        num_scenes = scene_query.shape[0]
        scene_context = hidden.new_zeros((num_scenes, self.hidden_channels))
        scene_context.index_add_(0, batch, weighted_value)
        context = self.up(scene_context)[batch]

        gate = torch.sigmoid(self.gate(hidden))
        return (
            gate * local * self.local_scale
            + (1.0 - gate) * context * self.context_scale
        )


__all__ = ["SceneAwareStructuralAdapter"]
