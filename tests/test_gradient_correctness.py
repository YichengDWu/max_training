from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from max.driver import CPU
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max_training import autograd


def _tensor(data: object) -> Tensor:
    return Tensor.from_dlpack(np.asarray(data, dtype=np.float32))


def _to_numpy(tensor: Tensor) -> np.ndarray:
    return np.from_dlpack(tensor.driver_tensor.to(CPU()))


def _finite_difference(
    loss_fn: Callable[..., float],
    values: list[np.ndarray],
    parameter_index: int,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    values = [value.astype(np.float64).copy() for value in values]
    grad = np.zeros_like(values[parameter_index], dtype=np.float64)

    for index in np.ndindex(values[parameter_index].shape):
        values[parameter_index][index] += eps
        plus = loss_fn(*values)
        values[parameter_index][index] -= 2 * eps
        minus = loss_fn(*values)
        values[parameter_index][index] += eps
        grad[index] = (plus - minus) / (2 * eps)

    return grad.astype(np.float32)


def test_composite_gradient_matches_finite_difference_cpu() -> None:
    x_np = np.array([[0.2, -0.4, 0.7], [1.1, -0.3, 0.5]], dtype=np.float32)
    w_np = np.array([[0.3, -0.2], [0.5, 0.7], [-0.6, 0.4]], dtype=np.float32)
    bias_np = np.array([0.25, -0.15], dtype=np.float32)
    target_np = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)

    x = autograd.requires_grad_(_tensor(x_np))
    w = autograd.requires_grad_(_tensor(w_np))
    bias = autograd.requires_grad_(_tensor(bias_np))
    target = _tensor(target_np)

    prediction = F.transpose(F.reshape(x, [3, 2]), 0, 1) @ w + bias
    loss = ((prediction - target) ** 2).mean(axis=None)
    autograd.backward(loss)

    def numpy_loss(x_value: np.ndarray, w_value: np.ndarray, bias_value: np.ndarray) -> float:
        prediction = x_value.reshape(3, 2).T @ w_value + bias_value
        return float(np.mean((prediction - target_np) ** 2))

    values = [x_np, w_np, bias_np]
    expected_x = _finite_difference(numpy_loss, values, 0)
    expected_w = _finite_difference(numpy_loss, values, 1)
    expected_bias = _finite_difference(numpy_loss, values, 2)

    assert isinstance(x.grad, Tensor)
    assert isinstance(w.grad, Tensor)
    assert isinstance(bias.grad, Tensor)
    np.testing.assert_allclose(_to_numpy(x.grad), expected_x, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(_to_numpy(w.grad), expected_w, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        _to_numpy(bias.grad), expected_bias, rtol=2e-3, atol=2e-3
    )


def test_linear_relu_mse_gradients_match_pytorch_cpu() -> None:
    torch = pytest.importorskip("torch")

    x_np = np.array(
        [[0.5, -1.0], [1.5, 0.25], [-0.75, 0.8]],
        dtype=np.float32,
    )
    y_np = np.array([[0.2], [-0.4], [0.7]], dtype=np.float32)
    w1_np = np.array([[0.7, -0.2], [-0.4, 0.9]], dtype=np.float32)
    b1_np = np.array([0.1, -0.3], dtype=np.float32)
    w2_np = np.array([[0.6, -0.5]], dtype=np.float32)
    b2_np = np.array([0.05], dtype=np.float32)

    x = _tensor(x_np)
    y = _tensor(y_np)
    w1 = autograd.requires_grad_(_tensor(w1_np))
    b1 = autograd.requires_grad_(_tensor(b1_np))
    w2 = autograd.requires_grad_(_tensor(w2_np))
    b2 = autograd.requires_grad_(_tensor(b2_np))

    hidden = F.relu(x @ w1.T + b1)
    prediction = hidden @ w2.T + b2
    loss = ((prediction - y) ** 2).mean(axis=None)
    autograd.backward(loss)

    tx = torch.tensor(x_np)
    ty = torch.tensor(y_np)
    tw1 = torch.tensor(w1_np, requires_grad=True)
    tb1 = torch.tensor(b1_np, requires_grad=True)
    tw2 = torch.tensor(w2_np, requires_grad=True)
    tb2 = torch.tensor(b2_np, requires_grad=True)

    thidden = torch.relu(tx @ tw1.T + tb1)
    tloss = ((thidden @ tw2.T + tb2 - ty) ** 2).mean()
    tloss.backward()

    assert isinstance(w1.grad, Tensor)
    assert isinstance(b1.grad, Tensor)
    assert isinstance(w2.grad, Tensor)
    assert isinstance(b2.grad, Tensor)
    np.testing.assert_allclose(_to_numpy(w1.grad), tw1.grad.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(_to_numpy(b1.grad), tb1.grad.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(_to_numpy(w2.grad), tw2.grad.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(_to_numpy(b2.grad), tb2.grad.numpy(), rtol=1e-5, atol=1e-6)
