"""Opt-in prototype training support for ``max.experimental`` tensors.

Importing this package installs lightweight monkey patches on top of the
existing experimental Tensor and Module APIs. The MAX source package remains
usable without training support unless this package is imported.
"""

from __future__ import annotations

from . import autograd, optim
from .compile import CompiledTrainStep, compile_train_step, trace_train_step
from .patches import enable

enable()

from . import nn  # noqa: E402
from .mlir import graph_module_asm  # noqa: E402
from .nn import mse_loss  # noqa: E402

__all__ = [
    "CompiledTrainStep",
    "autograd",
    "compile_train_step",
    "enable",
    "graph_module_asm",
    "mse_loss",
    "nn",
    "optim",
    "trace_train_step",
]
