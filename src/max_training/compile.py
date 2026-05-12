"""Compiled training-step adapter for experimental modules."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from max.driver import Buffer
from max.engine import Model
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn._compile_utils import (
    InputType,
    _detect_signals,
    _flatten_input_types,
    _flatten_outputs,
    _InputSlot,
    _OutputSlot,
    _reconstruct_outputs,
    _wrap_graph_inputs,
    flatten_input_buffers,
)
from max.experimental.realization_context import (
    GraphRealizationContext,
    _session,
)
from max.experimental.tensor import Tensor, realization_context
from max.graph import Graph
from max.nn.comm.allreduce import Signals

from . import autograd
from .optim import SGD


class CompiledTrainStep:
    """Compiled single-step training function returned by ``compile_train_step``.

    The compiled graph takes the module's parameter buffers followed by the
    user inputs. Each call mutates those parameter buffers in place and returns
    the traced loss tensor.
    """

    def __init__(
        self,
        engine_model: Model,
        parameters: list[tuple[str, Tensor]],
        input_slots: list[_InputSlot],
        output_slots: list[_OutputSlot],
        signal_buffers: list[Buffer],
        unary: bool,
    ) -> None:
        self._engine_model = engine_model
        self._parameters = parameters
        self._input_slots = input_slots
        self._output_slots = output_slots
        self._signal_buffers = signal_buffers
        self._unary = unary

    @property
    def engine_model(self) -> Model:
        """The underlying :class:`~max.engine.Model` for capture/replay."""
        return self._engine_model

    def __call__(self, *args: Any) -> Any:
        """Runs one compiled training step and returns the loss."""
        flat_args: list[Any] = []
        for name, parameter in self._parameters:
            if parameter.is_distributed:
                raise NotImplementedError(
                    "CompiledTrainStep does not support sharded "
                    f"parameters yet: {name}"
                )
            parameter._sync_realize()
            flat_args.extend(parameter.buffers)

        flat_args.extend(flatten_input_buffers(args, self._input_slots))
        flat_args.extend(self._signal_buffers)

        try:
            raw_results = list(self._engine_model(*flat_args))
        except (TypeError, ValueError) as e:
            raise TypeError(
                "Compiled train step call failed to bind arguments "
                f"({e}). Expected {len(self._parameters)} parameter "
                f"buffer(s), {len(self._input_slots)} user argument(s), "
                f"and {len(self._signal_buffers)} signal buffer(s)."
            ) from e

        return _reconstruct_outputs(
            raw_results, self._output_slots, self._unary
        )


@dataclasses.dataclass(frozen=True)
class _CompiledTrainStepTrace:
    graph: Graph
    parameters: list[tuple[str, Tensor]]
    input_slots: list[_InputSlot]
    output_slots: list[_OutputSlot]
    signals: Signals | None
    unary: bool


def _module_requires_grad_(module: Module[Any, Any], value: bool = True) -> Any:
    for _, parameter in module.parameters:
        autograd.requires_grad_(parameter, value)
    return module


def _module_zero_grad(module: Module[Any, Any]) -> None:
    for _, parameter in module.parameters:
        autograd.zero_grad(parameter)


def compile_train_step(
    module: Module[Any, Any],
    *input_types: InputType,
    loss_fn: Callable[..., Tensor],
    optimizer: Any,
    custom_extensions: Iterable[Path] = (),
) -> CompiledTrainStep:
    """Compiles a single SGD training step for an experimental module."""
    trace = _trace_train_step(
        module,
        input_types,
        loss_fn=loss_fn,
        optimizer=optimizer,
        custom_extensions=custom_extensions,
    )

    session_model = _session().load(trace.graph)
    cached_sig_bufs = trace.signals.buffers() if trace.signals is not None else []

    return CompiledTrainStep(
        engine_model=session_model,
        parameters=trace.parameters,
        input_slots=trace.input_slots,
        output_slots=trace.output_slots,
        signal_buffers=cached_sig_bufs,
        unary=trace.unary,
    )


def trace_train_step(
    module: Module[Any, Any],
    *input_types: InputType,
    loss_fn: Callable[..., Tensor],
    optimizer: Any,
    custom_extensions: Iterable[Path] = (),
) -> Graph:
    """Traces a single SGD training step and returns the raw graph."""
    return _trace_train_step(
        module,
        input_types,
        loss_fn=loss_fn,
        optimizer=optimizer,
        custom_extensions=custom_extensions,
    ).graph


def _trace_train_step(
    module: Module[Any, Any],
    input_types: Sequence[InputType],
    *,
    loss_fn: Callable[..., Tensor],
    optimizer: Any,
    custom_extensions: Iterable[Path] = (),
) -> _CompiledTrainStepTrace:
    if not isinstance(optimizer, SGD):
        raise NotImplementedError(
            "compile_train_step currently supports max_training.optim.SGD only"
        )

    parameters = list(module.parameters)
    if not parameters:
        raise ValueError("compile_train_step requires module parameters")

    parameter_names_by_id: dict[int, str] = {}
    for name, parameter in parameters:
        if parameter.is_distributed:
            raise NotImplementedError(
                "compile_train_step does not support sharded "
                f"parameters yet: {name}"
            )
        parameter._sync_realize()
        parameter_names_by_id[id(parameter)] = name

    trainable_ids: set[int] = set()
    for parameter in optimizer.parameters:
        try:
            parameter_names_by_id[id(parameter)]
        except KeyError as e:
            raise ValueError(
                "optimizer contains a tensor that is not a parameter "
                "of this module"
            ) from e
        trainable_ids.add(id(parameter))

    trainable_names = [
        name for name, parameter in parameters if id(parameter) in trainable_ids
    ]
    if not trainable_names:
        raise ValueError("optimizer does not contain trainable parameters")

    parameter_input_types = [
        parameter.type.as_buffer() for _, parameter in parameters
    ]
    user_graph_types, input_slots = _flatten_input_types(input_types)
    signals = _detect_signals(input_types, parameters=parameters)
    graph_types = [*parameter_input_types, *user_graph_types]
    if signals is not None:
        graph_types.extend(signals.input_types())

    graph = Graph(
        f"{type(module).__qualname__}.train_step",
        input_types=graph_types,
        custom_extensions=custom_extensions,
    )

    sig_buf_values = None
    if signals is not None:
        n_sig = len(signals.devices)
        sig_buf_values = [
            graph.inputs[len(graph_types) - n_sig + i].buffer
            for i in range(n_sig)
        ]

    ctx = GraphRealizationContext(graph, signal_buffers=sig_buf_values)
    with realization_context(ctx), ctx:
        parameter_values = graph.inputs[: len(parameters)]
        graph_parameters = {
            name: Tensor.from_graph_value(value)
            for (name, _), value in zip(
                parameters, parameter_values, strict=True
            )
        }

        n_user_inputs = len(user_graph_types)
        user_values = graph.inputs[
            len(parameters) : len(parameters) + n_user_inputs
        ]
        inputs = _wrap_graph_inputs(list(user_values), input_slots)

        def as_graph_parameter(name: str, tensor: Tensor) -> Tensor:
            parameter = graph_parameters[name]
            if name in trainable_names:
                autograd.requires_grad_(parameter)
            return parameter

        with module._mapped_parameters(as_graph_parameter):
            loss = loss_fn(module, *inputs)
            if not isinstance(loss, Tensor):
                raise TypeError("compile_train_step loss_fn must return a Tensor")

            autograd.backward(loss)

            with autograd.no_grad():
                for name in trainable_names:
                    parameter = graph_parameters[name]
                    grad = getattr(parameter, "grad", None)
                    if grad is None:
                        raise RuntimeError(
                            f"loss did not produce a gradient for {name!r}"
                        )
                    F.buffer_store(
                        parameter,
                        parameter - optimizer.lr * grad,
                    )

        flat_values, output_slots, unary = _flatten_outputs(loss)
        graph.output(*flat_values)

    return _CompiledTrainStepTrace(
        graph=graph,
        parameters=parameters,
        input_slots=input_slots,
        output_slots=output_slots,
        signals=signals,
        unary=unary,
    )


def install_module_methods() -> None:
    """Installs training helpers onto ``max.experimental.nn.Module``."""
    if getattr(Module, "_max_training_patched", False):
        return

    setattr(Module, "requires_grad_", _module_requires_grad_)
    setattr(Module, "zero_grad", _module_zero_grad)
    setattr(Module, "compile_train_step", compile_train_step)
    setattr(Module, "trace_train_step", trace_train_step)
    setattr(Module, "_max_training_patched", True)
