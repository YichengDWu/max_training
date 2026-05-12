from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU
from max.experimental.tensor import Tensor
from max_training import autograd, optim


def _tensor(data: object) -> Tensor:
    return Tensor.from_dlpack(np.asarray(data, dtype=np.float32))


def _to_numpy(t: Tensor) -> np.ndarray:
    return np.from_dlpack(t.driver_tensor.to(CPU()))


def test_square_sum_backward_cpu() -> None:
    x = autograd.requires_grad_(_tensor([2.0, -3.0]))

    loss = (x * x).sum(axis=None)
    autograd.backward(loss)

    grad = getattr(x, "grad", None)
    assert getattr(loss, "requires_grad")
    assert isinstance(grad, Tensor)
    np.testing.assert_allclose(_to_numpy(grad), [4.0, -6.0])


def test_broadcast_bias_backward_cpu() -> None:
    x = _tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = autograd.requires_grad_(_tensor([0.5, 1.5, 2.5]))

    loss = (x + bias).sum(axis=None)
    autograd.backward(loss)

    grad = getattr(bias, "grad", None)
    assert isinstance(grad, Tensor)
    np.testing.assert_allclose(_to_numpy(grad), [2.0, 2.0, 2.0])


def test_matmul_backward_cpu() -> None:
    x = autograd.requires_grad_(
        _tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    w = autograd.requires_grad_(
        _tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    )

    loss = (x @ w).sum(axis=None)
    autograd.backward(loss)

    x_grad = getattr(x, "grad", None)
    w_grad = getattr(w, "grad", None)
    assert isinstance(x_grad, Tensor)
    assert isinstance(w_grad, Tensor)
    np.testing.assert_allclose(
        _to_numpy(x_grad), [[18.0, 27.0], [18.0, 27.0]]
    )
    np.testing.assert_allclose(
        _to_numpy(w_grad), [[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]]
    )


def test_inplace_mutation_before_backward_raises_cpu() -> None:
    x = autograd.requires_grad_(_tensor([2.0, 3.0]))
    loss = (x * x).sum(axis=None)

    x += 1.0

    with pytest.raises(RuntimeError, match="modified in place"):
        autograd.backward(loss)


def test_optimizer_step_marks_existing_tape_stale_cpu() -> None:
    x = autograd.requires_grad_(_tensor([2.0, 3.0]))
    optimizer = optim.SGD([x], lr=0.1)
    loss = (x * x).sum(axis=None)

    autograd.backward(loss)
    optimizer.step()

    with pytest.raises(RuntimeError, match="modified in place"):
        autograd.backward(loss)
