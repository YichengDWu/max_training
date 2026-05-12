# H100 NVL Benchmark Results

These results were measured on a Vast.ai 1x H100 NVL instance:

- GPU: NVIDIA H100 NVL, compute capability 9.0
- Driver: 580.126.09
- CUDA: 13.0
- Python: 3.12.3
- MAX/Modular: 26.4.0.dev2026051206
- PyTorch: 2.11.0+cu130
- `ptxas`: `/usr/local/cuda/bin/ptxas`

The GPU smoke run passed:

```text
15 passed, 2 warnings in 430.65s (0:07:10)
```

The long first-run time was from MAX building its interpreter op cache for the
fresh virtual environment. Later runs reused that cache.

`examples/train_linear.py` converged:

```text
step=000 loss=9.000000 weight=0.000000 bias=0.000000
step=005 loss=0.006047 weight=1.999360 bias=0.922240
step=010 loss=0.000037 weight=2.000000 bias=0.993953
step=015 loss=0.000000 weight=2.000000 bias=0.999530
step=020 loss=0.000000 weight=2.000000 bias=0.999963
```

## Small CUDA Workload

This smoke benchmark was captured before
`benchmarks/pytorch_compile.py` defaulted PyTorch CUDA float32 matmul precision
to `high`. PyTorch printed a warning that TF32 tensor cores were available but
not enabled.

```bash
RUN_SYNC=1 RUN_TESTS=1 RUN_EXAMPLE=1 RUN_BENCHMARK=1 \
BATCH_SIZE=128 INPUT_DIM=64 HIDDEN_DIM=128 OUTPUT_DIM=16 \
WARMUP_STEPS=2 STEPS=5 REPEATS=2 TORCH_COMPILE_MODE=max-autotune \
scripts/gpu_smoke.sh
```

```text
engine                       status  compile_s  median_ms  steps/s  samples/s  initial_loss  final_loss
---------------------------  ------  ---------  ---------  -------  ---------  ------------  ----------
torch_eager                  ok      0.000000   0.502      1991.57  254920.8   0.878079      0.722721
torch_compile[max-autotune]  ok      29.346093  0.504      1984.23  253981.1   0.878079      0.722721
max_eager                    ok      0.000000   65.061     15.37    1967.4     0.878090      0.722726
max_compile                  ok      45.576143  0.187      5357.44  685752.5   0.878090      0.722731
```

## Larger CUDA Workload

This run also used PyTorch's strict FP32 CUDA matmul behavior.

```bash
MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
uv run --no-sync python benchmarks/pytorch_compile.py \
    --device cuda --batch-size 1024 --input-dim 256 --hidden-dim 512 \
    --output-dim 64 --warmup-steps 5 --steps 20 --repeats 3 \
    --torch-compile-mode max-autotune
```

```text
engine                       status  compile_s  median_ms  steps/s  samples/s  initial_loss  final_loss
---------------------------  ------  ---------  ---------  -------  ---------  ------------  ----------
torch_eager                  ok      0.000000   0.477      2098.33  2148692.5  0.981289      0.665561
torch_compile[max-autotune]  ok      56.935637  0.383      2608.76  2671368.6  0.981289      0.665561
max_eager                    ok      0.000000   65.164     15.35    15714.1    0.981284      0.665561
max_compile                  ok      46.234905  0.173      5771.29  5909797.5  0.981284      0.665604
```

PyTorch `max-autotune` printed several Triton candidate failures because some
candidate kernels exceeded the H100 NVL resource limit:

```text
OutOfMemoryError: out of resource: triton_mm Required: 262144 Hardware limit:232448
```

The benchmark still completed with `ok` status and selected valid kernels.

## Larger CUDA Workload With PyTorch TF32

This run explicitly enabled PyTorch TF32:

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Reported PyTorch settings:

```text
torch_float32_matmul_precision high
torch.backends.cuda.matmul.allow_tf32 True
```

Results:

```text
engine                       status  compile_s  median_ms  steps/s  samples/s  initial_loss  final_loss
---------------------------  ------  ---------  ---------  -------  ---------  ------------  ----------
torch_eager                  ok      0.000000   0.472      2119.32  2170183.3  0.981283      0.665560
torch_compile[max-autotune]  ok      22.253812  0.389      2571.01  2632714.8  0.981283      0.665603
max_eager                    ok      0.000000   65.599     15.24    15610.0    0.981284      0.665561
max_compile                  ok      0.023838   0.174      5758.85  5897066.7  0.981284      0.665604
```

The `max_compile` compile time in this table is a warm-cache number because the
same workload had already been compiled in the previous run.

## MAX GPU ASM Dump

MAX GPU asm dumping was tested with the internal session API:

```python
from max.experimental.realization_context import _session

_session()._dump_gpu_asm("/workspace/max_training_h100nvl/asm_dump_h100/train_step_%.ptx")
```

Dumping only happened after the compiled train step was executed once. The run
produced 18 PTX files under:

```text
/workspace/max_training_h100nvl/asm_dump_h100/
```

The dumped PTX confirms that MAX's compiled train-step matmul kernels used TF32
tensor core instructions:

```text
train_step_e102ca62cf155e2173b83610c342489e.ptx:375:
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32

train_step_0b7407d4aabb6fbfddf79f7404cb75d5.ptx:370:
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
```

This means the MAX vs PyTorch numbers should be interpreted as TF32-enabled
MAX compiled training against the selected PyTorch precision mode, not as a
strict full-FP32 comparison.
