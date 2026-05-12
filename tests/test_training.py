from __future__ import annotations

import max_training
import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max_training import autograd, optim


def _tensor(data: object) -> Tensor:
    return Tensor.from_dlpack(np.asarray(data, dtype=np.float32))


class Dense(Module[[Tensor], Tensor]):
    def __init__(self, weight: object, bias: object) -> None:
        self.weight = _tensor(weight)
        self.bias = _tensor(bias)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


def test_linear_regression_sgd_cpu_converges() -> None:
    model = Dense([[0.0]], [0.0])

    for _, parameter in model.parameters:
        autograd.requires_grad_(parameter)
    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=0.2)

    x = _tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = _tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0]])

    loss = (((model(x) - y) ** 2).mean(axis=None)).item()
    for _ in range(20):
        prediction = model(x)
        loss_tensor = ((prediction - y) ** 2).mean(axis=None)
        loss = loss_tensor.item()
        autograd.backward(loss_tensor)
        optimizer.step()
        optimizer.zero_grad()

    assert loss < 1e-4
    assert abs(model.weight.item() - 2.0) < 1e-3
    assert abs(model.bias.item() - 1.0) < 1e-3


def test_compiled_linear_regression_sgd_cpu_converges() -> None:
    model = Dense([[0.0]], [0.0])

    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=0.2)
    x = _tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = _tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0]])

    def mse_loss(model: Dense, x: Tensor, y: Tensor) -> Tensor:
        return ((model(x) - y) ** 2).mean(axis=None)

    train_step = max_training.compile_train_step(
        model,
        TensorType(DType.float32, [5, 1], device=CPU()),
        TensorType(DType.float32, [5, 1], device=CPU()),
        loss_fn=mse_loss,
        optimizer=optimizer,
    )

    loss = Tensor(0.0, device=CPU())
    for _ in range(20):
        loss = train_step(x, y)

    assert loss.item() < 1e-4
    assert abs(model.weight.item() - 2.0) < 1e-3
    assert abs(model.bias.item() - 1.0) < 1e-3


class TinyMLP(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.fc1 = Dense([[1.0], [-1.0]], [0.0, 0.0])
        self.fc2 = Dense([[0.5, -0.5]], [0.0])

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def test_mlp_relu_sgd_cpu_loss_decreases() -> None:
    model = TinyMLP()

    for _, parameter in model.parameters:
        autograd.requires_grad_(parameter)
    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=0.01)

    x = _tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = _tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0]])

    initial_loss = (((model(x) - y) ** 2).mean(axis=None)).item()
    loss_tensor = ((model(x) - y) ** 2).mean(axis=None)
    autograd.backward(loss_tensor)

    assert getattr(model.fc1.weight, "grad", None) is not None
    assert getattr(model.fc1.bias, "grad", None) is not None
    assert getattr(model.fc2.weight, "grad", None) is not None
    assert getattr(model.fc2.bias, "grad", None) is not None

    optimizer.step()
    optimizer.zero_grad()
    updated_loss = (((model(x) - y) ** 2).mean(axis=None)).item()

    assert updated_loss < initial_loss
