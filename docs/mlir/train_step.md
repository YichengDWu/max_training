# Train-Step MLIR

This is the output from:

```bash
uv run python examples/print_train_step_mlir.py
```

```mlir
module {
  mo.graph @Dense.train_step(%arg0: !mo.buffer<[1, 1], f32>, %arg1: !mo.buffer<[1], f32>, %arg2: !mo.tensor<[5, 1], f32>, %arg3: !mo.tensor<[5, 1], f32>) -> !mo.tensor<[1], f32> attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2", "input3"], result_names = ["output0"]} {
    %0 = mo.chain.create()
    %outTensor, %outChain = rmo.mo.mutable.load(%arg0, %0) : (!mo.buffer<[1, 1], f32>, !mo.chain) -> (!mo.tensor<[1, 1], f32>, !mo.chain)
    %1 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %2 = rmo.mo.transpose(%outTensor, %1) : (!mo.tensor<[1, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 1], f32>
    %3 = rmo.matmul(%arg2, %2) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[1, 1], f32>) -> !mo.tensor<[5, 1], f32>
    %outTensor_0, %outChain_1 = rmo.mo.mutable.load(%arg1, %outChain) : (!mo.buffer<[1], f32>, !mo.chain) -> (!mo.tensor<[1], f32>, !mo.chain)
    %4 = rmo.add(%3, %outTensor_0) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[5, 1], f32>
    %5 = rmo.sub(%4, %arg3) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>) -> !mo.tensor<[5, 1], f32>
    %6 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %7 = rmo.pow(%5, %6) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5, 1], f32>
    %8 = rmo.reshape(%7) {newShape = #mosh<ape[5]> : !mosh.ape} : (!mo.tensor<[5, 1], f32>) -> !mo.tensor<[5], f32>
    %9 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
    %10 = rmo.mo.reduce.mean(%8, %9) : (!mo.tensor<[5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
    %11 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %12 = rmo.broadcast_to(%11) {newShape = #mosh<ape[1]> : !mosh.ape} : (!mo.tensor<[], f32>) -> !mo.tensor<[1], f32>
    %13 = rmo.broadcast_to(%12) {newShape = #mosh<ape[5, 1]> : !mosh.ape} : (!mo.tensor<[1], f32>) -> !mo.tensor<[5, 1], f32>
    %14 = mo.constant {value = #M.dense_array<5.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %15 = rmo.div(%13, %14) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5, 1], f32>
    %16 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %17 = rmo.mul(%15, %16) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5, 1], f32>
    %18 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %19 = rmo.pow(%5, %18) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[5, 1], f32>
    %20 = rmo.mul(%17, %19) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>) -> !mo.tensor<[5, 1], f32>
    %21 = rmo.mo.negative(%20) : (!mo.tensor<[5, 1], f32>) -> !mo.tensor<[5, 1], f32>
    %22 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
    %23 = rmo.mo.reduce.add(%20, %22) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1, 1], f32>
    %24 = rmo.reshape(%23) {newShape = #mosh<ape[1]> : !mosh.ape} : (!mo.tensor<[1, 1], f32>) -> !mo.tensor<[1], f32>
    %25 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %26 = rmo.mo.transpose(%2, %25) : (!mo.tensor<[1, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 1], f32>
    %27 = rmo.matmul(%20, %26) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[1, 1], f32>) -> !mo.tensor<[5, 1], f32>
    %28 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %29 = rmo.mo.transpose(%arg2, %28) : (!mo.tensor<[5, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 5], f32>
    %30 = rmo.matmul(%29, %20) : (!mo.tensor<[1, 5], f32>, !mo.tensor<[5, 1], f32>) -> !mo.tensor<[1, 1], f32>
    %31 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %32 = rmo.mo.transpose(%30, %31) : (!mo.tensor<[1, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 1], f32>
    %33 = mo.constant {value = #M.dense_array<2.000000e-01> : tensor<f32>} : !mo.tensor<[], f32>
    %34 = rmo.mul(%33, %32) : (!mo.tensor<[], f32>, !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32>
    %outTensor_2, %outChain_3 = rmo.mo.mutable.load(%arg0, %outChain_1) : (!mo.buffer<[1, 1], f32>, !mo.chain) -> (!mo.tensor<[1, 1], f32>, !mo.chain)
    %35 = rmo.sub(%outTensor_2, %34) : (!mo.tensor<[1, 1], f32>, !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32>
    %36 = rmo.mo.mutable.store(%arg0, %35, %outChain_3) : (!mo.buffer<[1, 1], f32>, !mo.tensor<[1, 1], f32>, !mo.chain) -> !mo.chain
    %37 = mo.constant {value = #M.dense_array<2.000000e-01> : tensor<f32>} : !mo.tensor<[], f32>
    %38 = rmo.mul(%37, %24) : (!mo.tensor<[], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
    %outTensor_4, %outChain_5 = rmo.mo.mutable.load(%arg1, %36) : (!mo.buffer<[1], f32>, !mo.chain) -> (!mo.tensor<[1], f32>, !mo.chain)
    %39 = rmo.sub(%outTensor_4, %38) : (!mo.tensor<[1], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
    %40 = rmo.mo.mutable.store(%arg1, %39, %outChain_5) : (!mo.buffer<[1], f32>, !mo.tensor<[1], f32>, !mo.chain) -> !mo.chain
    mo.output %10 : !mo.tensor<[1], f32>
  }
}
```
