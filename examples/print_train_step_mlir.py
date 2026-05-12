"""Prints the MLIR for a prototype compiled training step."""

from __future__ import annotations

import argparse
from pathlib import Path

import max_training
from max.driver import CPU
from max.dtype import DType
from max.experimental.nn import Module
from max.experimental.tensor import Tensor, default_device
from max.graph import TensorType
from max_training import optim


class Dense(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.weight = Tensor([[0.0]], device=CPU())
        self.bias = Tensor([0.0], device=CPU())

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


def mse_loss(model: Dense, x: Tensor, y: Tensor) -> Tensor:
    return ((model(x) - y) ** 2).mean(axis=None)


def build_mlir(*, lr: float) -> str:
    with default_device(CPU()):
        model = Dense()

    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=lr)
    graph = max_training.trace_train_step(
        model,
        TensorType(DType.float32, [5, 1], device=CPU()),
        TensorType(DType.float32, [5, 1], device=CPU()),
        loss_fn=mse_loss,
        optimizer=optimizer,
    )
    return max_training.graph_module_asm(
        graph,
        assume_verified=True,
        enable_debug_info=False,
        pretty_debug_info=False,
        use_local_scope=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mlir = build_mlir(lr=args.lr)
    if args.output is None:
        print(mlir)
    else:
        args.output.write_text(mlir)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
