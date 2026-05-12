"""Prototype reverse-mode automatic differentiation for experimental tensors."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_GRAD_ENABLED: ContextVar[bool] = ContextVar("_GRAD_ENABLED", default=True)
_SUPPORTED_OPS = frozenset(
    {
        "add",
        "broadcast_to",
        "div",
        "exp",
        "log",
        "matmul",
        "mean",
        "mo_relu",
        "mul",
        "negate",
        "pow",
        "relu",
        "reshape",
        "sigmoid",
        "sqrt",
        "sub",
        "sum",
        "tanh",
        "transpose",
    }
)


@dataclass(frozen=True)
class GradFn:
    """Backward edge recorded on a tensor produced by a differentiable op."""

    parents: tuple[Any, ...]
    backward: Callable[[Any], tuple[Any | None, ...]]


def _is_tensor(value: Any) -> bool:
    from max.experimental.tensor import Tensor

    return isinstance(value, Tensor)


def _requires_grad(tensor: Any) -> bool:
    return bool(getattr(tensor, "_requires_grad", False))


def _set_requires_grad(tensor: Any, value: bool) -> None:
    setattr(tensor, "_requires_grad", value)


def _grad_fn(tensor: Any) -> GradFn | None:
    return getattr(tensor, "_grad_fn", None)


def _set_grad_fn(tensor: Any, grad_fn: GradFn | None) -> None:
    setattr(tensor, "_grad_fn", grad_fn)


def _shape_tuple(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


@contextlib.contextmanager
def no_grad() -> Generator[None]:
    """Temporarily disables autograd recording."""

    token = _GRAD_ENABLED.set(False)
    try:
        yield
    finally:
        _GRAD_ENABLED.reset(token)


def requires_grad_(tensor: Any, value: bool = True) -> Any:
    """Marks ``tensor`` as a leaf that should accumulate gradients."""

    _set_requires_grad(tensor, value)
    if not value:
        _set_grad_fn(tensor, None)
    if not hasattr(tensor, "_grad"):
        setattr(tensor, "_grad", None)
    return tensor


def zero_grad(tensor: Any) -> None:
    """Clears the accumulated gradient on ``tensor``."""

    setattr(tensor, "_grad", None)


def detach(tensor: Any) -> Any:
    """Returns a tensor view detached from the autograd tape."""

    from max.experimental.tensor import Tensor

    if tensor.is_distributed:
        raise NotImplementedError(
            "autograd.detach does not support sharded tensors yet"
        )
    tensor._sync_realize()
    detached = Tensor(storage=tensor.driver_tensor)
    _set_requires_grad(detached, False)
    _set_grad_fn(detached, None)
    setattr(detached, "_grad", None)
    return detached


def _transpose_last_two(tensor: Any) -> Any:
    from max.experimental import functional as F

    return F.transpose(tensor, -1, -2)


def _unbroadcast(grad: Any, target: Any) -> Any:
    """Sums broadcasted dimensions in ``grad`` back to ``target``'s shape."""

    from max.experimental import functional as F

    target_shape = _shape_tuple(target)
    while len(_shape_tuple(grad)) > len(target_shape):
        grad = F.squeeze(F.sum(grad, axis=0), axis=0)

    for axis, target_dim in enumerate(target_shape):
        grad_shape = _shape_tuple(grad)
        if axis >= len(grad_shape):
            break
        if target_dim == 1 and grad_shape[axis] != 1:
            grad = F.sum(grad, axis=axis)

    if _shape_tuple(grad) != target_shape:
        grad = F.reshape(grad, target_shape)
    return grad


def _sum_to_input(grad: Any, x: Any, axis: int | None) -> Any:
    from max.experimental import functional as F

    return F.broadcast_to(grad, x.shape)


def _mean_to_input(grad: Any, x: Any, axis: int | None) -> Any:
    if axis is None:
        denom = x.num_elements()
    else:
        denom = int(x.shape[axis])
    return _sum_to_input(grad, x, axis) / denom


def _op_name(graph_op: Callable[..., object]) -> str:
    return getattr(graph_op, "__name__", type(graph_op).__name__)


def _grad_add(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    lhs, rhs = args[:2]
    return (
        _unbroadcast(grad, lhs) if _is_tensor(lhs) else None,
        _unbroadcast(grad, rhs) if _is_tensor(rhs) else None,
    )


def _grad_sub(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    lhs, rhs = args[:2]
    return (
        _unbroadcast(grad, lhs) if _is_tensor(lhs) else None,
        _unbroadcast(-grad, rhs) if _is_tensor(rhs) else None,
    )


def _grad_mul(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    lhs, rhs = args[:2]
    lhs_grad = _unbroadcast(grad * rhs, lhs) if _is_tensor(lhs) else None
    rhs_grad = _unbroadcast(grad * lhs, rhs) if _is_tensor(rhs) else None
    return lhs_grad, rhs_grad


def _grad_div(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    lhs, rhs = args[:2]
    lhs_grad = _unbroadcast(grad / rhs, lhs) if _is_tensor(lhs) else None
    rhs_grad = (
        _unbroadcast(-(grad * lhs) / (rhs * rhs), rhs)
        if _is_tensor(rhs)
        else None
    )
    return lhs_grad, rhs_grad


def _grad_negate(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    return (-grad,)


def _grad_pow(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    base, exponent = args[:2]
    if _is_tensor(exponent):
        raise NotImplementedError("autograd pow only supports scalar exponents")
    base_grad = grad * exponent * (base ** (exponent - 1))
    return (_unbroadcast(base_grad, base), None)


def _grad_matmul(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    lhs, rhs = args[:2]
    if len(_shape_tuple(lhs)) != 2 or len(_shape_tuple(rhs)) != 2:
        raise NotImplementedError(
            "autograd matmul currently supports 2D inputs only"
        )
    lhs_grad = grad @ _transpose_last_two(rhs) if _is_tensor(lhs) else None
    rhs_grad = _transpose_last_two(lhs) @ grad if _is_tensor(rhs) else None
    return lhs_grad, rhs_grad


def _grad_transpose(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    from max.experimental import functional as F

    _, dim1, dim2 = args[:3]
    return (F.transpose(grad, dim1, dim2), None, None)


def _grad_reshape(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    from max.experimental import functional as F

    x = args[0]
    return (F.reshape(grad, x.shape), None)


def _grad_broadcast_to(
    args: tuple[Any, ...], grad: Any
) -> tuple[Any | None, ...]:
    x = args[0]
    return (_unbroadcast(grad, x), None)


def _grad_sum(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    x = args[0]
    axis = args[1] if len(args) > 1 else -1
    return (_sum_to_input(grad, x, axis), None)


def _grad_mean(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    x = args[0]
    axis = args[1] if len(args) > 1 else -1
    return (_mean_to_input(grad, x, axis), None)


def _grad_relu(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    from max.experimental import functional as F

    x = args[0]
    return (F.where(x > 0, grad, F.zeros_like(grad)),)


def _grad_exp(
    args: tuple[Any, ...], grad: Any, result: Any
) -> tuple[Any | None, ...]:
    return (grad * result,)


def _grad_log(args: tuple[Any, ...], grad: Any) -> tuple[Any | None, ...]:
    x = args[0]
    return (grad / x,)


def _grad_sqrt(
    args: tuple[Any, ...], grad: Any, result: Any
) -> tuple[Any | None, ...]:
    return (grad / (2 * result),)


def _grad_sigmoid(
    args: tuple[Any, ...], grad: Any, result: Any
) -> tuple[Any | None, ...]:
    return (grad * result * (1 - result),)


def _grad_tanh(
    args: tuple[Any, ...], grad: Any, result: Any
) -> tuple[Any | None, ...]:
    return (grad * (1 - result * result),)


def _gradient_for(
    graph_op: Callable[..., object],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    grad: Any,
) -> tuple[Any | None, ...]:
    name = _op_name(graph_op)
    if kwargs:
        raise NotImplementedError(
            f"autograd for {name} does not support keyword arguments yet"
        )

    if name == "add":
        return _grad_add(args, grad)
    if name == "sub":
        return _grad_sub(args, grad)
    if name == "mul":
        return _grad_mul(args, grad)
    if name == "div":
        return _grad_div(args, grad)
    if name == "negate":
        return _grad_negate(args, grad)
    if name == "pow":
        return _grad_pow(args, grad)
    if name == "matmul":
        return _grad_matmul(args, grad)
    if name == "transpose":
        return _grad_transpose(args, grad)
    if name == "reshape":
        return _grad_reshape(args, grad)
    if name == "broadcast_to":
        return _grad_broadcast_to(args, grad)
    if name == "sum":
        return _grad_sum(args, grad)
    if name == "mean":
        return _grad_mean(args, grad)
    if name in ("relu", "mo_relu"):
        return _grad_relu(args, grad)
    if name == "exp":
        return _grad_exp(args, grad, result)
    if name == "log":
        return _grad_log(args, grad)
    if name == "sqrt":
        return _grad_sqrt(args, grad, result)
    if name == "sigmoid":
        return _grad_sigmoid(args, grad, result)
    if name == "tanh":
        return _grad_tanh(args, grad, result)

    raise NotImplementedError(f"autograd for operation {name!r} is not implemented")


def record(
    graph_op: Callable[..., object],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> Any:
    """Records a differentiable op result when any input requires gradients."""

    if not _GRAD_ENABLED.get():
        return result
    if any(_is_tensor(arg) and _requires_grad(arg) for arg in kwargs.values()):
        raise NotImplementedError(
            "autograd does not support Tensor keyword arguments yet"
        )
    if not _is_tensor(result):
        if any(_is_tensor(arg) and _requires_grad(arg) for arg in args):
            raise NotImplementedError("autograd only supports single Tensor outputs")
        return result

    parent_indices = tuple(
        index
        for index, arg in enumerate(args)
        if _is_tensor(arg) and _requires_grad(arg)
    )
    if not parent_indices:
        return result

    op_name = _op_name(graph_op)
    if op_name not in _SUPPORTED_OPS and _grad_fn(result) is not None:
        return result

    def backward_fn(grad: Any) -> tuple[Any | None, ...]:
        all_grads = _gradient_for(graph_op, args, kwargs, result, grad)
        return tuple(all_grads[index] for index in parent_indices)

    _set_requires_grad(result, True)
    _set_grad_fn(
        result,
        GradFn(
            parents=tuple(args[index] for index in parent_indices),
            backward=backward_fn,
        ),
    )
    setattr(result, "_grad", None)
    setattr(result, "_op_name", op_name)
    return result


def _topological_sort(root: Any) -> list[Any]:
    visited: set[int] = set()
    ordered: list[Any] = []

    def visit(tensor: Any) -> None:
        if id(tensor) in visited:
            return
        visited.add(id(tensor))
        grad_fn = _grad_fn(tensor)
        if grad_fn is not None:
            for parent in grad_fn.parents:
                visit(parent)
        ordered.append(tensor)

    visit(root)
    return ordered


def backward(tensor: Any, grad: Any | None = None) -> None:
    """Runs reverse-mode automatic differentiation from ``tensor``."""

    from max.experimental import functional as F

    if grad is None:
        if tensor.num_elements() != 1:
            raise RuntimeError(
                "grad must be provided when calling backward on a non-scalar tensor"
            )
        grad = F.ones_like(tensor)

    with no_grad():
        topo = _topological_sort(tensor)
        for current in topo:
            if _grad_fn(current) is not None:
                setattr(current, "_grad", None)
        setattr(tensor, "_grad", grad)
        for current in reversed(topo):
            current_grad = getattr(current, "_grad", None)
            grad_fn = _grad_fn(current)
            if current_grad is None or grad_fn is None:
                continue
            parent_grads = grad_fn.backward(current_grad)
            for parent, parent_grad in zip(
                grad_fn.parents, parent_grads, strict=True
            ):
                if parent_grad is None or not _requires_grad(parent):
                    continue
                existing = getattr(parent, "_grad", None)
                setattr(
                    parent,
                    "_grad",
                    parent_grad if existing is None else existing + parent_grad,
                )
