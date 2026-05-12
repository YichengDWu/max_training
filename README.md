# max_training

`max_training` is an opt-in prototype training package layered on top of MAX
`max.experimental`. It lives outside the Modular source tree and does not
require edits to core MAX/Mojo files.

The package explores PyTorch-equivalent training behavior on MAX without trying
to clone PyTorch's public API exactly. Current support includes reverse-mode
autograd, trainable parameters, a small `nn` facade, SGD/RMSprop-style
optimizers, eager training loops, compiled SGD train steps, MLIR inspection, and
PyTorch comparison benchmarks.

## How It Works

`max_training` does not patch files inside the installed `max` wheel. The hack
is done at Python import time. `import max_training` calls `enable()`, which
mutates the live `max.experimental` Python objects in the current process.

The eager-mode hooks are:

- `Tensor` gets extra Python-side attributes and methods: `requires_grad`,
  `.grad`, `requires_grad_()`, `backward()`, `zero_grad()`, `detach()`, and
  basic in-place arithmetic helpers. Grad metadata is stored directly on Tensor
  objects as `_requires_grad`, `_grad`, and `_grad_fn`.
- `max.experimental.functional` and
  `max.experimental.functional.spmd_ops` functions such as `add`, `matmul`,
  `mean`, `reshape`, `relu`, and `transpose` are wrapped. The wrapper calls the
  original MAX op, then records a `GradFn` edge when any Tensor input requires
  gradients.
- `Module.parameters` is replaced with a view that is both MAX-style iterable
  `(name, parameter)` and PyTorch-style callable, so examples can use either
  `for name, p in module.parameters` or `module.parameters()`.

The autograd tape is intentionally small. Each supported op has a handwritten
backward rule in `autograd.py`. `backward(loss)` topologically sorts the graph
from the loss Tensor, walks it in reverse, and accumulates Tensor gradients into
leaf `.grad` fields. Optimizers then update parameter buffers with MAX ops such
as `F.buffer_store(parameter, parameter - lr * grad)`.

The compiled training path uses MAX's Python graph machinery directly:

1. `compile_train_step()` reads the module's current parameters and creates a
   `Graph` whose inputs are parameter buffers followed by user inputs.
2. It enters `GraphRealizationContext`, converts graph inputs back into
   Tensor objects with `Tensor.from_graph_value()`, and temporarily remaps the
   module's parameters through `module._mapped_parameters(...)`.
3. It runs the user `loss_fn(module, *inputs)` while the autograd wrappers are
   active, so the forward graph and backward graph are both traced as MAX graph
   ops instead of being executed eagerly.
4. It calls `autograd.backward(loss)` during tracing, reads each graph
   parameter's `.grad`, and appends the SGD update into the same graph with
   `F.buffer_store(parameter, parameter - optimizer.lr * grad)`.
5. The resulting graph is loaded with MAX's internal `_session().load(...)`.
   At runtime, `CompiledTrainStep.__call__` passes the model's live parameter
   buffers plus user input buffers to the loaded `Model`, so the compiled graph
   mutates the model parameters in place and returns the loss.

MLIR inspection is also intentionally direct: `graph_module_asm()` calls the
current graph object's private `graph._module.asm(...)` hook. This is useful
for experimentation, but it is tied to MAX internals.

This package targets training semantics and convergence behavior, not full
PyTorch API compatibility. It relies on private/experimental MAX Python APIs,
so it should be treated as a prototype rather than a stable extension surface.

## Supported Surface

- Tensor gradients: `requires_grad`, `.grad`, `backward()`, `zero_grad()`,
  `detach()`, and `autograd.no_grad()`.
- Autograd ops: elementwise arithmetic, `matmul`, `sum`, `mean`, `reshape`,
  `broadcast_to`, `transpose`, `relu`, `sigmoid`, `tanh`, `exp`, `log`, and
  `sqrt`.
- Layers and losses: `nn.Parameter`, `nn.Linear`, `nn.Flatten`,
  `nn.Sequential`, and `nn.MSELoss`.
- Optimizers: eager `SGD`, SGD momentum, and prototype `RMSprop`.
- Compiled training: fixed-shape single-step SGD training graphs.

## Requirements

Dependencies are managed with `uv`. The lockfile resolves `modular` and `max`
from the Modular nightly package index. Python 3.12 is required by the current
nightly wheels.

Install Python 3.12 if `uv` has not already provisioned it:

```bash
uv python install 3.12
```

Install the development environment:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Run examples:

```bash
uv run python examples/train_linear.py
```

On a CUDA machine, run the GPU smoke check:

```bash
scripts/gpu_smoke.sh
```

Useful knobs:

```bash
RUN_TESTS=0 BATCH_SIZE=1024 INPUT_DIM=256 HIDDEN_DIM=512 OUTPUT_DIM=64 \
scripts/gpu_smoke.sh
```

## Minimal Eager Example

```python
from max.driver import CPU
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max_training import autograd, optim


class Dense(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.weight = Tensor([[0.0]], device=CPU())
        self.bias = Tensor([0.0], device=CPU())

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


model = Dense()
for _, parameter in model.parameters:
    autograd.requires_grad_(parameter)

optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=0.2)
x = Tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], device=CPU())
y = Tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0]], device=CPU())

for _ in range(20):
    loss = ((model(x) - y) ** 2).mean(axis=None)
    autograd.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

print(loss.item(), model.weight.item(), model.bias.item())
```

Run the packaged example:

```bash
uv run python examples/train_linear.py
```

## Minimal Compiled Train Step

Using the `Dense`, `x`, and `y` definitions from the eager example:

```python
import max_training
from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max_training import optim


def mse_loss(model: Dense, x: Tensor, y: Tensor) -> Tensor:
    return ((model(x) - y) ** 2).mean(axis=None)


model = Dense()
optimizer = optim.SGD((parameter for _, parameter in model.parameters), lr=0.2)

train_step = max_training.compile_train_step(
    model,
    TensorType(DType.float32, [5, 1], device=CPU()),
    TensorType(DType.float32, [5, 1], device=CPU()),
    loss_fn=mse_loss,
    optimizer=optimizer,
)

for _ in range(20):
    loss = train_step(x, y)
```

The compiled train step mutates the model's parameter buffers in place and
returns the traced loss tensor for that step.

To inspect generated MLIR:

```bash
uv run python examples/print_train_step_mlir.py
```

## PyTorch Tutorial-Style Example

The package can run the polynomial fitting examples from "Learning PyTorch with
Examples" with small import/name changes.

```bash
uv run python examples/pytorch_polynomial.py --mode all
```

## PyTorch Compile Benchmark

`benchmarks/pytorch_compile.py` compares the same one-hidden-layer MLP training
step across four engines:

1. PyTorch eager
2. PyTorch `torch.compile`
3. `max_training` eager
4. `max_training` compiled train step

It uses the same generated data, initialization, dtype, shapes, learning rate,
warmup steps, and measured steps for each engine. Compile time is reported
separately from steady-state step latency.

CPU:

```bash
uv run python benchmarks/pytorch_compile.py \
    --device cpu --batch-size 128 --warmup-steps 5 --steps 20 --repeats 5
```

NVIDIA GPU:

```bash
MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
uv run python benchmarks/pytorch_compile.py \
    --device cuda --batch-size 128 --warmup-steps 5 --steps 20 --repeats 5
```

Optional PyTorch compile mode:

```bash
uv run python benchmarks/pytorch_compile.py \
    --device cuda --batch-size 1024 --input-dim 256 --hidden-dim 512 \
    --output-dim 64 --torch-compile-mode max-autotune
```

To inspect the pre-lowering MAX train-step graph for the same workload:

```bash
MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
uv run python benchmarks/pytorch_compile.py \
    --device cuda --batch-size 1024 --input-dim 256 --hidden-dim 512 \
    --output-dim 64 --max-graph-stats --dump-max-mlir /tmp/max_training.mlir
```

The same check can be run through the smoke script:

```bash
DUMP_MAX_MLIR=/tmp/max_training.mlir scripts/gpu_smoke.sh
```

## Recorded Artifacts

Recorded benchmark and graph artifacts are kept outside the main README:

- [RTX 5090 benchmark results](docs/benchmarks/rtx5090.md)
- [Train-step MLIR](docs/mlir/train_step.md)

## Tests

```bash
uv run pytest
```

## Current Limitations

- Single-device tensors only.
- Compiled training currently applies the SGD learning rate only; eager training
  supports SGD, SGD momentum, and prototype RMSprop.
- Fixed input shapes for compiled train steps.
- Prototype backward rules for a small set of tensor ops.
- No Dataset/DataLoader, torchvision, convolution, pooling, or CrossEntropyLoss.

## License

MIT
