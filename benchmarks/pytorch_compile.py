"""Benchmark PyTorch eager/compile against max_training eager/compile.

The workload is intentionally small and boring: a one-hidden-layer MLP trained
with mean squared error and SGD.  The benchmark keeps data, initialization,
shape, dtype, learning rate, warmup steps, and measured steps identical across
engines so the output is useful for capability and compile-overhead comparison.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import re
import statistics
import time
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import torch
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType

import max_training
from max_training import autograd, optim


DeviceName = Literal["cpu", "cuda"]
TorchFloat32MatmulPrecision = Literal["highest", "high", "medium"]


@dataclasses.dataclass(frozen=True)
class Workload:
    x: np.ndarray
    y: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclasses.dataclass
class BenchResult:
    engine: str
    device: str
    status: str
    compile_time_s: float | None = None
    median_step_ms: float | None = None
    mean_step_ms: float | None = None
    steps_per_s: float | None = None
    samples_per_s: float | None = None
    initial_loss: float | None = None
    final_loss: float | None = None
    train_steps: int | None = None
    error: str | None = None


class Dense(Module[[Tensor], Tensor]):
    def __init__(self, weight: Tensor, bias: Tensor) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


class MaxMLP(Module[[Tensor], Tensor]):
    def __init__(self, workload: Workload, device: Device) -> None:
        self.fc1 = Dense(
            _max_tensor(workload.w1, device),
            _max_tensor(workload.b1, device),
        )
        self.fc2 = Dense(
            _max_tensor(workload.w2, device),
            _max_tensor(workload.b2, device),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class TorchMLP(torch.nn.Module):
    def __init__(self, workload: Workload, device: torch.device) -> None:
        super().__init__()
        hidden_dim, input_dim = workload.w1.shape
        output_dim, _ = workload.w2.shape
        self.fc1 = torch.nn.Linear(
            input_dim, hidden_dim, device=device, dtype=torch.float32
        )
        self.fc2 = torch.nn.Linear(
            hidden_dim, output_dim, device=device, dtype=torch.float32
        )
        self.reset_parameters_from(workload, device)

    def reset_parameters_from(
        self, workload: Workload, device: torch.device
    ) -> None:
        with torch.no_grad():
            self.fc1.weight.copy_(_torch_tensor(workload.w1, device))
            self.fc1.bias.copy_(_torch_tensor(workload.b1, device))
            self.fc2.weight.copy_(_torch_tensor(workload.w2, device))
            self.fc2.bias.copy_(_torch_tensor(workload.b2, device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_workload(
    *,
    seed: int,
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> Workload:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch_size, input_dim), dtype=np.float32)
    target_w1 = _scaled_normal(rng, (hidden_dim, input_dim), input_dim)
    target_b1 = np.zeros((hidden_dim,), dtype=np.float32)
    target_w2 = _scaled_normal(rng, (output_dim, hidden_dim), hidden_dim)
    target_b2 = np.zeros((output_dim,), dtype=np.float32)

    hidden = np.maximum(x @ target_w1.T + target_b1, 0)
    y = hidden @ target_w2.T + target_b2

    return Workload(
        x=x,
        y=y.astype(np.float32),
        w1=_scaled_normal(rng, (hidden_dim, input_dim), input_dim),
        b1=np.zeros((hidden_dim,), dtype=np.float32),
        w2=_scaled_normal(rng, (output_dim, hidden_dim), hidden_dim),
        b2=np.zeros((output_dim,), dtype=np.float32),
    )


def _scaled_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    fan_in: int,
) -> np.ndarray:
    return (rng.standard_normal(shape, dtype=np.float32) / np.sqrt(fan_in)).astype(
        np.float32
    )


def _torch_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


def _max_tensor(array: np.ndarray, device: Device) -> Tensor:
    tensor = Tensor.from_dlpack(np.array(array, dtype=np.float32, copy=True))
    if device != CPU():
        tensor = tensor.to(device)
    return tensor


def _max_loss(model: MaxMLP, x: Tensor, y: Tensor) -> Tensor:
    return ((model(x) - y) ** 2).mean(axis=None)


def _torch_loss(
    model: TorchMLP,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return ((model(x) - y) ** 2).mean()


def _torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_sync(device: Device) -> None:
    sync = getattr(device, "synchronize", None)
    if callable(sync):
        sync()


def _time_steps(
    *,
    step: Callable[[], Any],
    loss_item: Callable[[Any], float],
    evaluate_loss: Callable[[], float],
    sync: Callable[[], None],
    warmup_steps: int,
    steps: int,
    repeats: int,
) -> tuple[list[float], float, int]:
    train_steps = 0
    for _ in range(warmup_steps):
        step()
        train_steps += 1
    sync()

    per_step_s: list[float] = []
    final_loss = 0.0
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        loss = None
        for _ in range(steps):
            loss = step()
            train_steps += 1
        sync()
        ended = time.perf_counter()
        final_loss = loss_item(loss)
        per_step_s.append((ended - started) / steps)

    sync()
    final_loss = evaluate_loss()
    return per_step_s, final_loss, train_steps


def _result_from_timings(
    *,
    engine: str,
    device: str,
    batch_size: int,
    compile_time_s: float | None,
    initial_loss: float,
    final_loss: float,
    train_steps: int,
    per_step_s: list[float],
) -> BenchResult:
    median_s = statistics.median(per_step_s)
    mean_s = statistics.mean(per_step_s)
    return BenchResult(
        engine=engine,
        device=device,
        status="ok" if final_loss < initial_loss else "loss_not_decreased",
        compile_time_s=compile_time_s,
        median_step_ms=median_s * 1000,
        mean_step_ms=mean_s * 1000,
        steps_per_s=1 / median_s,
        samples_per_s=batch_size / median_s,
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_steps=train_steps,
    )


def benchmark_torch_eager(
    workload: Workload,
    *,
    device_name: DeviceName,
    lr: float,
    warmup_steps: int,
    steps: int,
    repeats: int,
) -> BenchResult:
    device = torch.device(device_name)
    model = TorchMLP(workload, device)
    x = _torch_tensor(workload.x, device)
    y = _torch_tensor(workload.y, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = _torch_loss(model, x, y)
        loss.backward()
        optimizer.step()
        return loss

    initial_loss = float(_torch_loss(model, x, y).detach().cpu())
    timings, final_loss, train_steps = _time_steps(
        step=step,
        loss_item=lambda loss: float(loss.detach().cpu()),
        evaluate_loss=lambda: float(_torch_loss(model, x, y).detach().cpu()),
        sync=lambda: _torch_sync(device),
        warmup_steps=warmup_steps,
        steps=steps,
        repeats=repeats,
    )
    return _result_from_timings(
        engine="torch_eager",
        device=device_name,
        batch_size=workload.x.shape[0],
        compile_time_s=0.0,
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_steps=train_steps,
        per_step_s=timings,
    )


def benchmark_torch_compile(
    workload: Workload,
    *,
    device_name: DeviceName,
    lr: float,
    warmup_steps: int,
    steps: int,
    repeats: int,
    compile_mode: str,
    fullgraph: bool,
) -> BenchResult:
    device = torch.device(device_name)
    model = TorchMLP(workload, device)
    x = _torch_tensor(workload.x, device)
    y = _torch_tensor(workload.y, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def eager_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = _torch_loss(model, x, y)
        loss.backward()
        optimizer.step()
        return loss

    torch_compile_kwargs: dict[str, Any] = {"fullgraph": fullgraph}
    if compile_mode != "default":
        torch_compile_kwargs["mode"] = compile_mode
    compiled_step = torch.compile(eager_step, **torch_compile_kwargs)
    initial_loss = float(_torch_loss(model, x, y).detach().cpu())

    _torch_sync(device)
    started = time.perf_counter()
    compiled_step()
    _torch_sync(device)
    compile_time_s = time.perf_counter() - started
    model.reset_parameters_from(workload, device)
    optimizer.zero_grad(set_to_none=True)

    timings, final_loss, train_steps = _time_steps(
        step=compiled_step,
        loss_item=lambda loss: float(loss.detach().cpu()),
        evaluate_loss=lambda: float(_torch_loss(model, x, y).detach().cpu()),
        sync=lambda: _torch_sync(device),
        warmup_steps=warmup_steps,
        steps=steps,
        repeats=repeats,
    )
    return _result_from_timings(
        engine=_torch_compile_engine_name(compile_mode, fullgraph),
        device=device_name,
        batch_size=workload.x.shape[0],
        compile_time_s=compile_time_s,
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_steps=train_steps,
        per_step_s=timings,
    )


def _torch_compile_engine_name(compile_mode: str, fullgraph: bool) -> str:
    name = "torch_compile"
    if compile_mode != "default":
        name += f"[{compile_mode}]"
    if fullgraph:
        name += "[fullgraph]"
    return name


def benchmark_max_eager(
    workload: Workload,
    *,
    device_name: DeviceName,
    lr: float,
    warmup_steps: int,
    steps: int,
    repeats: int,
) -> BenchResult:
    device = _max_device(device_name)
    model = MaxMLP(workload, device)
    x = _max_tensor(workload.x, device)
    y = _max_tensor(workload.y, device)
    for _, parameter in model.parameters:
        autograd.requires_grad_(parameter)
    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=lr)

    def step() -> Tensor:
        loss = _max_loss(model, x, y)
        autograd.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        return loss

    initial_loss = _max_loss(model, x, y).item()
    timings, final_loss, train_steps = _time_steps(
        step=step,
        loss_item=lambda loss: loss.item(),
        evaluate_loss=lambda: _max_loss(model, x, y).item(),
        sync=lambda: _max_sync(device),
        warmup_steps=warmup_steps,
        steps=steps,
        repeats=repeats,
    )
    return _result_from_timings(
        engine="max_eager",
        device=device_name,
        batch_size=workload.x.shape[0],
        compile_time_s=0.0,
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_steps=train_steps,
        per_step_s=timings,
    )


def benchmark_max_compile(
    workload: Workload,
    *,
    device_name: DeviceName,
    lr: float,
    warmup_steps: int,
    steps: int,
    repeats: int,
) -> BenchResult:
    device = _max_device(device_name)
    model = MaxMLP(workload, device)
    x = _max_tensor(workload.x, device)
    y = _max_tensor(workload.y, device)
    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=lr)
    initial_loss = _max_loss(model, x, y).item()

    started = time.perf_counter()
    train_step = max_training.compile_train_step(
        model,
        TensorType(DType.float32, list(workload.x.shape), device=device),
        TensorType(DType.float32, list(workload.y.shape), device=device),
        loss_fn=_max_loss,
        optimizer=optimizer,
    )
    compile_time_s = time.perf_counter() - started

    timings, final_loss, train_steps = _time_steps(
        step=lambda: train_step(x, y),
        loss_item=lambda loss: loss.item(),
        evaluate_loss=lambda: _max_loss(model, x, y).item(),
        sync=lambda: _max_sync(device),
        warmup_steps=warmup_steps,
        steps=steps,
        repeats=repeats,
    )
    return _result_from_timings(
        engine="max_compile",
        device=device_name,
        batch_size=workload.x.shape[0],
        compile_time_s=compile_time_s,
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_steps=train_steps,
        per_step_s=timings,
    )


def _trace_max_compile_graph(
    workload: Workload,
    *,
    device_name: DeviceName,
    lr: float,
) -> str:
    device = _max_device(device_name)
    model = MaxMLP(workload, device)
    optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=lr)
    graph = max_training.trace_train_step(
        model,
        TensorType(DType.float32, list(workload.x.shape), device=device),
        TensorType(DType.float32, list(workload.y.shape), device=device),
        loss_fn=_max_loss,
        optimizer=optimizer,
    )
    return max_training.graph_module_asm(
        graph,
        assume_verified=True,
        enable_debug_info=False,
        pretty_debug_info=False,
        use_local_scope=True,
    )


def _mlir_op_counts(mlir: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for line in mlir.splitlines():
        match = re.search(r"=\s+([A-Za-z0-9_.]+)(?:\(|\s|\{)", line)
        if match is not None:
            counts[match.group(1)] += 1
    return counts


def _print_max_graph_stats(
    workload: Workload,
    *,
    device_name: DeviceName,
    lr: float,
    dump_mlir: str | None,
) -> None:
    mlir = _trace_max_compile_graph(workload, device_name=device_name, lr=lr)
    if dump_mlir is not None:
        with open(dump_mlir, "w") as f:
            f.write(mlir)

    counts = _mlir_op_counts(mlir)
    total_ops = sum(counts.values())
    structural_ops = {
        "mo.chain.create",
        "mo.constant",
        "rmo.broadcast_to",
        "rmo.reshape",
        "rmo.mo.transpose",
    }
    high_level_ops = sum(
        count for op, count in counts.items() if op not in structural_ops
    )

    print("max_compile graph stats")
    print(f"device: {device_name}")
    print(f"shape: x={workload.x.shape}, y={workload.y.shape}")
    print(f"total_mlir_ops: {total_ops}")
    print(f"high_level_compute_or_memory_ops: {high_level_ops}")
    print(
        "note: this is pre-lowering MLIR, so it counts graph operations, "
        "not final CUDA kernel launches."
    )
    if dump_mlir is not None:
        print(f"mlir: {dump_mlir}")
    print("\nop counts:")
    for op, count in counts.most_common():
        print(f"{op:28s} {count}")


def _max_device(device_name: DeviceName) -> Device:
    if device_name == "cpu":
        return CPU()
    if accelerator_count() < 1:
        raise RuntimeError("MAX cannot see an accelerator")
    return Accelerator()


def _validate_device(device_name: DeviceName) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    if device_name == "cuda" and accelerator_count() < 1:
        raise RuntimeError("MAX cannot see an accelerator")


def _configure_torch_float32_matmul(
    device_name: DeviceName,
    precision: TorchFloat32MatmulPrecision,
) -> None:
    """Configures PyTorch float32 matmul precision for CUDA benchmarks."""
    if device_name != "cuda":
        return

    torch.set_float32_matmul_precision(precision)
    allow_tf32 = precision != "highest"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def _torch_precision_summary(device_name: DeviceName) -> list[str]:
    if device_name != "cuda":
        return []

    return [
        f"torch_float32_matmul_precision: {torch.get_float32_matmul_precision()}",
        "torch.backends.cuda.matmul.allow_tf32: "
        f"{torch.backends.cuda.matmul.allow_tf32}",
        f"torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}",
    ]


def _run_or_error(
    engine: str,
    fn: Callable[[], BenchResult],
    device_name: DeviceName,
) -> BenchResult:
    try:
        return fn()
    except Exception as exc:
        return BenchResult(
            engine=engine,
            device=device_name,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _print_table(results: list[BenchResult]) -> None:
    headers = [
        "engine",
        "status",
        "compile_s",
        "median_ms",
        "steps/s",
        "samples/s",
        "initial_loss",
        "final_loss",
    ]
    rows: list[list[str]] = []
    for result in results:
        rows.append(
            [
                result.engine,
                result.status,
                _fmt(result.compile_time_s, "{:.6f}"),
                _fmt(result.median_step_ms, "{:.3f}"),
                _fmt(result.steps_per_s, "{:.2f}"),
                _fmt(result.samples_per_s, "{:.1f}"),
                _fmt(result.initial_loss, "{:.6f}"),
                _fmt(result.final_loss, "{:.6f}"),
            ]
        )

    widths = [
        max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))
    ]
    print("  ".join(cell.ljust(width) for cell, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))

    errors = [result for result in results if result.error]
    for result in errors:
        print(f"\n{result.engine} error: {result.error}")


def _fmt(value: float | None, pattern: str) -> str:
    return "n/a" if value is None else pattern.format(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--torch-compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument("--torch-fullgraph", action="store_true")
    parser.add_argument(
        "--torch-float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help=(
            "PyTorch float32 matmul precision. The default, high, enables "
            "TF32 on CUDA for a fairer comparison with MAX GPU matmul "
            "lowering. Use highest for strict FP32 PyTorch matmul."
        ),
    )
    parser.add_argument(
        "--max-graph-stats",
        action="store_true",
        help="Print pre-lowering MLIR operation counts for the MAX train step.",
    )
    parser.add_argument(
        "--dump-max-mlir",
        help="Write the pre-lowering MAX train-step MLIR to this path.",
    )
    args = parser.parse_args()

    device_name = cast(DeviceName, args.device)
    torch_precision = cast(
        TorchFloat32MatmulPrecision,
        args.torch_float32_matmul_precision,
    )
    _configure_torch_float32_matmul(device_name, torch_precision)
    _validate_device(device_name)

    workload = _make_workload(
        seed=args.seed,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    )

    if args.max_graph_stats:
        _print_max_graph_stats(
            workload,
            device_name=device_name,
            lr=args.lr,
            dump_mlir=args.dump_max_mlir,
        )
        return

    common = dict(
        workload=workload,
        device_name=device_name,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        repeats=args.repeats,
    )
    results = [
        _run_or_error(
            "torch_eager",
            lambda: benchmark_torch_eager(**common),
            device_name,
        ),
        _run_or_error(
            _torch_compile_engine_name(
                args.torch_compile_mode,
                args.torch_fullgraph,
            ),
            lambda: benchmark_torch_compile(
                **common,
                compile_mode=args.torch_compile_mode,
                fullgraph=args.torch_fullgraph,
            ),
            device_name,
        ),
        _run_or_error(
            "max_eager",
            lambda: benchmark_max_eager(**common),
            device_name,
        ),
        _run_or_error(
            "max_compile",
            lambda: benchmark_max_compile(**common),
            device_name,
        ),
    ]

    if args.json:
        print(json.dumps([dataclasses.asdict(result) for result in results], indent=2))
    else:
        for line in _torch_precision_summary(device_name):
            print(line)
        if device_name == "cuda":
            print()
        _print_table(results)


if __name__ == "__main__":
    main()
