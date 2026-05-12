"""MLIR helpers for current MAX graph modules."""

from __future__ import annotations

from typing import Any


def graph_module_asm(graph: Any, **kwargs: Any) -> str:
    """Returns MLIR text from the current MAX ``builtin.ModuleOp`` wrapper."""
    return graph._module.asm(**kwargs)
