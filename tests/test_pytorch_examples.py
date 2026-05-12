from __future__ import annotations

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


def test_model_parameters_are_callable_and_iterable() -> None:
    model = nn.Sequential(nn.Linear(3, 1), nn.Flatten(0, 1))

    named_parameters = list(model.parameters)
    positional_parameters = _parameters(model)

    assert [name for name, _ in named_parameters] == ["0.weight", "0.bias"]
    assert positional_parameters == [parameter for _, parameter in named_parameters]


def test_mse_loss_reductions() -> None:
    prediction = _tensor([1.0, 2.0, 3.0])
    target = _tensor([0.0, 2.0, 5.0])

    assert nn.MSELoss(reduction="sum")(prediction, target).item() == 5.0
    assert abs(nn.MSELoss(reduction="mean")(prediction, target).item() - 5 / 3) < 1e-6


def test_pytorch_nn_polynomial_example_loss_decreases() -> None:
    _, y, xx = _sine_data(points=32)
    linear = nn.Linear(3, 1)
    linear.weight = nn.Parameter(_tensor([[0.0, 0.0, 0.0]]))
    linear.bias = nn.Parameter(_tensor([0.0]))
    model = nn.Sequential(linear, nn.Flatten(0, 1))
    loss_fn = nn.MSELoss(reduction="sum")

    initial_loss = loss_fn(model(xx), y).item()
    learning_rate = 1e-6
    for _ in range(40):
        loss = loss_fn(model(xx), y)
        _zero_grad(model)
        _backward(loss)
        with autograd.no_grad():
            for parameter in _parameters(model):
                grad = getattr(parameter, "grad")
                assert grad is not None
                parameter -= learning_rate * grad

    assert loss_fn(model(xx), y).item() < initial_loss


class Polynomial3(nn.Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(Tensor(0.0, device=CPU()))
        self.b = nn.Parameter(Tensor(0.0, device=CPU()))
        self.c = nn.Parameter(Tensor(0.0, device=CPU()))
        self.d = nn.Parameter(Tensor(0.0, device=CPU()))

    def forward(self, x: Tensor) -> Tensor:
        return self.a + self.b * x + self.c * x**2 + self.d * x**3


def test_pytorch_custom_module_rmsprop_example_loss_decreases() -> None:
    x, y, _ = _sine_data(points=32)
    model = Polynomial3()
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = optim.RMSprop(_parameters(model), lr=1e-3)

    initial_loss = loss_fn(model(x), y).item()
    for _ in range(20):
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        _backward(loss)
        optimizer.step()

    assert loss_fn(model(x), y).item() < initial_loss
