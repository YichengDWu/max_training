"""Small PyTorch-style neural network facade for ``max_training``."""

from __future__ import annotations

from math import prod
from typing import Any

from max.driver import CPU
from max.experimental import functional as F
from max.experimental.nn import (
    Linear as _Linear,
    Module,
    ModuleList,
    Sequential,
    module_dataclass,
)
from max.experimental.tensor import Tensor, default_device

from . import autograd


def Parameter(data: Any, *, requires_grad: bool = True) -> Tensor:
    """Creates a trainable tensor parameter.

    This mirrors the part of ``torch.nn.Parameter`` needed by the PyTorch
    tutorial examples: assigning the returned Tensor to a ``Module`` attribute
    makes it visible through MAX's existing parameter traversal.
    """
    tensor = data if isinstance(data, Tensor) else Tensor(data)
    return autograd.requires_grad_(tensor, requires_grad)


class Linear(_Linear):
    """Linear layer whose parameters are trainable by default."""

    def __init__(
        self,
        in_dim: Any,
        out_dim: Any,
        *,
        bias: bool = True,
        device: Any | None = None,
    ) -> None:
        # Match PyTorch's CPU default even when MAX can see a GPU.
        with default_device(device if device is not None else CPU()):
            super().__init__(in_dim, out_dim, bias=bias)
        for parameter in getattr(self, "parameters")():
            autograd.requires_grad_(parameter)


class Flatten(Module[[Tensor], Tensor]):
    """Flattens a contiguous range of dimensions using ``reshape``."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = [int(dim) for dim in x.shape]
        rank = len(shape)
        if rank == 0:
            return x

        start = self.start_dim if self.start_dim >= 0 else self.start_dim + rank
        end = self.end_dim if self.end_dim >= 0 else self.end_dim + rank
        if start < 0 or end < start or end >= rank:
            raise ValueError(
                f"invalid flatten dims start_dim={self.start_dim}, "
                f"end_dim={self.end_dim} for rank {rank}"
            )

        flat_dim = prod(shape[start : end + 1])
        return F.reshape(x, [*shape[:start], flat_dim, *shape[end + 1 :]])


def mse_loss(
    input: Tensor,
    target: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Mean squared error loss with PyTorch-style reductions."""
    squared = (input - target) ** 2
    if reduction == "mean":
        return squared.mean(axis=None)
    if reduction == "sum":
        return squared.sum(axis=None)
    if reduction == "none":
        return squared
    raise ValueError(
        "mse_loss reduction must be 'mean', 'sum', or 'none', "
        f"got {reduction!r}"
    )


class MSELoss(Module[[Tensor, Tensor], Tensor]):
    """Module wrapper for :func:`mse_loss`."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss(input, target, reduction=self.reduction)


__all__ = [
    "Flatten",
    "Linear",
    "MSELoss",
    "Module",
    "ModuleList",
    "Parameter",
    "Sequential",
    "module_dataclass",
    "mse_loss",
]
