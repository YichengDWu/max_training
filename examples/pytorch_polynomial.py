"""PyTorch tutorial-style polynomial fitting examples on ``max_training``."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
from max.driver import CPU
from max.experimental.tensor import Tensor
from max_training import autograd, nn, optim


def _tensor(data: object) -> Tensor:
    return Tensor.from_dlpack(np.asarray(data, dtype=np.float32))


def _parameters(module: Any) -> list[Tensor]:
    return list(getattr(module, "parameters")())


def _zero_grad(module: Any) -> None:
    getattr(module, "zero_grad")()


def _backward(tensor: Tensor) -> None:
    getattr(tensor, "backward")()


def _sine_data(points: int) -> tuple[Tensor, Tensor, Tensor]:
    x_np = np.linspace(-math.pi, math.pi, points, dtype=np.float32)
    x = _tensor(x_np)
    y = _tensor(np.sin(x_np))
    xx = _tensor(np.stack([x_np, x_np**2, x_np**3], axis=1))
    return x, y, xx


def run_autograd(points: int, steps: int) -> float:
    x, y, _ = _sine_data(points)
    a = nn.Parameter(Tensor(0.0, device=CPU()))
    b = nn.Parameter(Tensor(0.0, device=CPU()))
    c = nn.Parameter(Tensor(0.0, device=CPU()))
    d = nn.Parameter(Tensor(0.0, device=CPU()))
    learning_rate = 1e-6

    loss = Tensor(0.0, device=CPU())
    for _ in range(steps):
        y_pred = a + b * x + c * x**2 + d * x**3
        loss = ((y_pred - y) ** 2).sum(axis=None)
        _backward(loss)
        with autograd.no_grad():
            a_grad = getattr(a, "grad")
            b_grad = getattr(b, "grad")
            c_grad = getattr(c, "grad")
            d_grad = getattr(d, "grad")
            assert a_grad is not None
            assert b_grad is not None
            assert c_grad is not None
            assert d_grad is not None
            a -= learning_rate * a_grad
            b -= learning_rate * b_grad
            c -= learning_rate * c_grad
            d -= learning_rate * d_grad
            for parameter in (a, b, c, d):
                setattr(parameter, "grad", None)

    print(
        "autograd result: "
        f"loss={loss.item():.6f}, "
        f"y={a.item():.6f}+{b.item():.6f}x+"
        f"{c.item():.6f}x^2+{d.item():.6f}x^3"
    )
    return loss.item()


def run_nn(points: int, steps: int) -> float:
    _, y, xx = _sine_data(points)
    linear = nn.Linear(3, 1)
    linear.weight = nn.Parameter(_tensor([[0.0, 0.0, 0.0]]))
    linear.bias = nn.Parameter(_tensor([0.0]))
    model = nn.Sequential(linear, nn.Flatten(0, 1))
    loss_fn = nn.MSELoss(reduction="sum")
    learning_rate = 1e-6

    loss = Tensor(0.0, device=CPU())
    for _ in range(steps):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)
        _zero_grad(model)
        _backward(loss)
        with autograd.no_grad():
            for parameter in _parameters(model):
                grad = getattr(parameter, "grad")
                assert grad is not None
                parameter -= learning_rate * grad

    print(
        "nn result: "
        f"loss={loss.item():.6f}, "
        f"y={linear.bias.item():.6f}+{linear.weight[0, 0].item():.6f}x+"
        f"{linear.weight[0, 1].item():.6f}x^2+"
        f"{linear.weight[0, 2].item():.6f}x^3"
    )
    return loss.item()


class Polynomial3(nn.Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(Tensor(0.0, device=CPU()))
        self.b = nn.Parameter(Tensor(0.0, device=CPU()))
        self.c = nn.Parameter(Tensor(0.0, device=CPU()))
        self.d = nn.Parameter(Tensor(0.0, device=CPU()))

    def forward(self, x: Tensor) -> Tensor:
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

    def string(self) -> str:
        return (
            f"y={self.a.item():.6f}+{self.b.item():.6f}x+"
            f"{self.c.item():.6f}x^2+{self.d.item():.6f}x^3"
        )


def run_optim(points: int, steps: int) -> float:
    x, y, _ = _sine_data(points)
    model = Polynomial3()
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.RMSprop(_parameters(model), lr=1e-3)

    loss = Tensor(0.0, device=CPU())
    for _ in range(steps):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        _backward(loss)
        optimizer.step()

    print(f"optim result: loss={loss.item():.6f}, {model.string()}")
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("autograd", "nn", "optim", "all"),
        default="all",
    )
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    if args.mode in ("autograd", "all"):
        run_autograd(args.points, args.steps)
    if args.mode in ("nn", "all"):
        run_nn(args.points, args.steps)
    if args.mode in ("optim", "all"):
        run_optim(args.points, args.steps)


if __name__ == "__main__":
    main()
