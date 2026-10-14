"""Physically sparse connection layers for the local-learning experiments."""

from __future__ import annotations

import torch
from torch import Tensor


class SparseConnectionLayer:
    """A flat edge list with a ReLU gate and bias on every connection."""

    def __init__(
        self,
        source_count: int,
        target_count: int,
        *,
        generator: torch.Generator,
        weight_low: float = 0.08,
        weight_high: float = 0.20,
        bias_low: float,
        bias_high: float,
    ) -> None:
        self.source_count = source_count
        self.target_count = target_count
        self.source_index = torch.arange(source_count).repeat(target_count)
        self.target_index = torch.arange(target_count).repeat_interleave(
            source_count
        )
        edge_count = source_count * target_count
        self.weight = weight_low + (weight_high - weight_low) * torch.rand(
            edge_count,
            generator=generator,
        )
        self.bias = bias_low + (bias_high - bias_low) * torch.rand(
            edge_count,
            generator=generator,
        )
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    @property
    def edge_count(self) -> int:
        return self.source_index.numel()

    @property
    def possible_edge_count(self) -> int:
        return self.source_count * self.target_count

    def edge_contribution(self, source: Tensor) -> Tensor:
        """Return each edge's recorded nonnegative forward contribution."""
        pre_activation = (
            source[self.source_index] * self.weight - self.bias
        )
        return torch.relu(pre_activation)

    def forward(self, source: Tensor) -> tuple[Tensor, Tensor]:
        contribution = self.edge_contribution(source)
        active = contribution > 0
        target = torch.zeros(
            self.target_count,
            dtype=source.dtype,
            device=source.device,
        )
        target.scatter_add_(0, self.target_index, contribution)
        return target, active

    def filter_edges(self, keep: Tensor) -> None:
        self.source_index = self.source_index[keep]
        self.target_index = self.target_index[keep]
        self.weight = self.weight[keep]
        self.bias = self.bias[keep]

    def append_edges(
        self,
        source_index: Tensor,
        target_index: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> None:
        self.source_index = torch.cat([self.source_index, source_index])
        self.target_index = torch.cat([self.target_index, target_index])
        self.weight = torch.cat([self.weight, weight])
        self.bias = torch.cat([self.bias, bias])

    def existing_pair_mask(self) -> Tensor:
        mask = torch.zeros(
            self.target_count,
            self.source_count,
            dtype=torch.bool,
        )
        mask[self.target_index, self.source_index] = True
        return mask
