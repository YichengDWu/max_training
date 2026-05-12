"""Opt-in monkey patches that attach training support to MAX experimental APIs."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from max.experimental import functional as F
from max.experimental.functional import spmd_ops
from max.experimental.tensor import Tensor

from . import autograd
from .compile import install_module_methods

_PATCHED = False

_FUNCTIONAL_OP_NAMES = (
    "add",
    "broadcast_to",
    "div",
    "exp",
    "log",
    "matmul",
    "mean",
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
)


class _ParametersView:
    """Iterable MAX-style pairs that is also callable PyTorch-style."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __iter__(self) -> Any:
        yield from self._module.local_parameters
        for prefix, descendant in self._module.descendants:
            for name, parameter in descendant.local_parameters:
                yield f"{prefix}.{name}", parameter

    def __call__(self) -> Any:
        return (parameter for _, parameter in self)

    def items(self) -> Any:
        return iter(self)

    def values(self) -> Any:
        return self()


def _named_graph_op(name: str) -> Callable[..., Any]:
    def graph_op(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("marker function should not be called")

    graph_op.__name__ = name
    return graph_op


def _record_args(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if name in ("mean", "sum") and args:
        axis = kwargs.get("axis", args[1] if len(args) > 1 else -1)
        return (args[0], axis), {}
    return args, kwargs


def _wrap_recording(
    name: str,
    graph_op: Callable[..., Any],
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    if getattr(fn, "_max_training_recording", False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        record_args, record_kwargs = _record_args(name, args, kwargs)
        return autograd.record(graph_op, record_args, record_kwargs, result)

    setattr(wrapped, "_max_training_recording", True)
    return wrapped


def _patch_tensor() -> None:
    if getattr(Tensor, "_max_training_tensor_patched", False):
        return

    def requires_grad_get(self: Tensor) -> bool:
        return bool(getattr(self, "_requires_grad", False))

    def requires_grad_set(self: Tensor, value: bool) -> None:
        autograd.requires_grad_(self, value)

    def grad_get(self: Tensor) -> Tensor | None:
        return getattr(self, "_grad", None)

    def grad_set(self: Tensor, value: Tensor | None) -> None:
        setattr(self, "_grad", value)

    def requires_grad_(self: Tensor, value: bool = True) -> Tensor:
        return autograd.requires_grad_(self, value)

    def backward(self: Tensor, grad: Tensor | None = None) -> None:
        autograd.backward(self, grad)

    def zero_grad(self: Tensor) -> None:
        autograd.zero_grad(self)

    def detach(self: Tensor) -> Tensor:
        return autograd.detach(self)

    def pow(self: Tensor, exponent: Any) -> Tensor:
        return self**exponent

    def iadd(self: Tensor, value: Any) -> Tensor:
        self._sync_realize()
        F.buffer_store(self, self + value)
        return self

    def isub(self: Tensor, value: Any) -> Tensor:
        self._sync_realize()
        F.buffer_store(self, self - value)
        return self

    def imul(self: Tensor, value: Any) -> Tensor:
        self._sync_realize()
        F.buffer_store(self, self * value)
        return self

    def itruediv(self: Tensor, value: Any) -> Tensor:
        self._sync_realize()
        F.buffer_store(self, self / value)
        return self

    Tensor.requires_grad = property(requires_grad_get, requires_grad_set)  # type: ignore[attr-defined]
    Tensor.grad = property(grad_get, grad_set)  # type: ignore[attr-defined]
    Tensor.requires_grad_ = requires_grad_  # type: ignore[attr-defined]
    Tensor.backward = backward  # type: ignore[attr-defined]
    Tensor.zero_grad = zero_grad  # type: ignore[attr-defined]
    Tensor.detach = detach  # type: ignore[attr-defined]
    Tensor.pow = pow  # type: ignore[attr-defined]
    setattr(Tensor, "__iadd__", iadd)
    setattr(Tensor, "__isub__", isub)
    setattr(Tensor, "__imul__", imul)
    setattr(Tensor, "__itruediv__", itruediv)
    setattr(Tensor, "_max_training_tensor_patched", True)


def _patch_functional() -> None:
    original_functional = spmd_ops.functional

    def training_functional(
        graph_op: Callable[..., Any] | None = None,
        rule: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        if graph_op is None:
            return functools.partial(training_functional, rule=rule)

        wrapped = original_functional(graph_op, rule=rule)
        return _wrap_recording(
            getattr(graph_op, "__name__", type(graph_op).__name__),
            graph_op,
            wrapped,
        )

    setattr(spmd_ops, "functional", training_functional)
    setattr(F, "functional", training_functional)

    for name in _FUNCTIONAL_OP_NAMES:
        fn = getattr(spmd_ops, name, None)
        if fn is None:
            fn = getattr(F, name, None)
        if fn is None:
            continue

        patched = _wrap_recording(name, _named_graph_op(name), fn)
        setattr(spmd_ops, name, patched)
        setattr(F, name, patched)


def _patch_module_parameters() -> None:
    from max.experimental.nn import Module

    if getattr(Module, "_max_training_parameters_patched", False):
        return

    def parameters(self: Module[Any, Any]) -> _ParametersView:
        return _ParametersView(self)

    setattr(Module, "parameters", property(parameters))
    setattr(Module, "_max_training_parameters_patched", True)


def enable() -> None:
    """Installs all training patches. Safe to call more than once."""
    global _PATCHED
    if _PATCHED:
        return

    _patch_tensor()
    _patch_functional()
    _patch_module_parameters()
    install_module_methods()
    _PATCHED = True
