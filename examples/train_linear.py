"""Train a tiny linear model with the opt-in max_training package."""

from __future__ import annotations

import argparse

from max.driver import CPU
from max.experimental.nn import Module
from max.experimental.tensor import Tensor, default_device
from max_training import autograd, optim


class Dense(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.weight = Tensor([[0.0]], device=CPU())
        self.bias = Tensor([0.0], device=CPU())

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


def loss_for(model: Dense, x: Tensor, y: Tensor) -> Tensor:
    prediction = model(x)
    return ((prediction - y) ** 2).mean(axis=None)


def train(steps: int, lr: float, log_every: int) -> None:
    with default_device(CPU()):
        model = Dense()
    for _, parameter in model.parameters:
        autograd.requires_grad_(parameter)

    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=lr)

    x = Tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], device=CPU())
    y = 2 * x + 1

    for step in range(steps + 1):
        loss = loss_for(model, x, y)
        if step == 0 or step == steps or step % log_every == 0:
            print(
                "step={step:03d} loss={loss:.6f} weight={weight:.6f} "
                "bias={bias:.6f}".format(
                    step=step,
                    loss=loss.item(),
                    weight=model.weight.item(),
                    bias=model.bias.item(),
                )
            )

        if step == steps:
            break

        autograd.backward(loss)
        optimizer.step()
        optimizer.zero_grad()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.lr < 0:
        raise ValueError("--lr must be non-negative")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")

    train(steps=args.steps, lr=args.lr, log_every=args.log_every)


if __name__ == "__main__":
    main()
