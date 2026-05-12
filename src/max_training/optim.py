"""Prototype optimizers for experimental tensor training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from max.experimental import functional as F
from max.experimental.tensor import Tensor

from . import autograd


def _normalize_parameters(parameters: Iterable[Any]) -> tuple[Tensor, ...]:
    normalized: list[Tensor] = []
    for parameter in parameters:
        if isinstance(parameter, Tensor):
            normalized.append(parameter)
            continue
        if (
            isinstance(parameter, tuple)
            and len(parameter) == 2
            and isinstance(parameter[1], Tensor)
        ):
            normalized.append(parameter[1])
            continue
        raise TypeError(
            "optimizer parameters must be Tensors or (name, Tensor) pairs"
        )
    return tuple(normalized)


class SGD:
    """A minimal stochastic gradient descent optimizer."""

    def __init__(
        self,
        parameters: Iterable[Any],
        *,
        lr: float,
        momentum: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if momentum < 0:
            raise ValueError(f"momentum must be non-negative, got {momentum}")
        self.parameters = _normalize_parameters(parameters)
        self.lr = lr
        self.momentum = momentum
        self._momentum_buffers: dict[int, Tensor] = {}

    def step(self) -> None:
        """Applies one SGD update to all parameters with gradients."""
        with autograd.no_grad():
            for parameter in self.parameters:
                grad = getattr(parameter, "grad", None)
                if grad is None:
                    continue
                parameter._sync_realize()
                update = grad
                if self.momentum:
                    momentum_buffer = self._momentum_buffers.get(id(parameter))
                    if momentum_buffer is None:
                        momentum_buffer = F.zeros_like(parameter)
                        momentum_buffer._sync_realize()
                        self._momentum_buffers[id(parameter)] = momentum_buffer
                    update = self.momentum * momentum_buffer + grad
                    F.buffer_store(momentum_buffer, update)
                F.buffer_store(parameter, parameter - self.lr * update)
                autograd.mark_dirty(parameter)

    def zero_grad(self) -> None:
        """Clears gradients for all optimizer parameters."""
        for parameter in self.parameters:
            autograd.zero_grad(parameter)


class RMSprop:
    """A small RMSprop optimizer compatible with the PyTorch tutorial example."""

    def __init__(
        self,
        parameters: Iterable[Any],
        *,
        lr: float,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not 0 <= alpha < 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        self.parameters = _normalize_parameters(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self._square_avgs: dict[int, Tensor] = {}

    def step(self) -> None:
        """Applies one RMSprop update to all parameters with gradients."""
        with autograd.no_grad():
            for parameter in self.parameters:
                grad = getattr(parameter, "grad", None)
                if grad is None:
                    continue
                parameter._sync_realize()
                square_avg = self._square_avgs.get(id(parameter))
                if square_avg is None:
                    square_avg = F.zeros_like(parameter)
                    square_avg._sync_realize()
                    self._square_avgs[id(parameter)] = square_avg

                new_square_avg = (
                    self.alpha * square_avg
                    + (1 - self.alpha) * grad * grad
                )
                F.buffer_store(square_avg, new_square_avg)
                update = grad / (F.sqrt(new_square_avg) + self.eps)
                F.buffer_store(parameter, parameter - self.lr * update)
                autograd.mark_dirty(parameter)

    def zero_grad(self) -> None:
        """Clears gradients for all optimizer parameters."""
        for parameter in self.parameters:
            autograd.zero_grad(parameter)
