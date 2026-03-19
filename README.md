## GEMM Benchmark supporting multiple GPU BLAS implementaitons 


Goals:
- Use the familiar interface of rocblas-bench/hipblaslt-bench
- Implement native support for cuBLAS, cuBLASLT, and other GPU BLAS libraries
- Expose implementation specific tuneables to the user

### Cloning the Repository
To get started, clone the repository from GitHub using:

```
git clone git@github.com:ROCm/mblas-bench.git
cd mblas-bench
```

### Building for ROCm only
```
cmake -S src -B build -DWITH_ROCM=true -DWITH_CUDA=false  
cmake --build build -j 
```

#### Conditional ROCm backends
ROCm builds include both hipBLASLt and rocBLAS backends in a single `mblaship` library. Disable backends individually if needed:
```
cmake -S src -B build -DWITH_CUDA=false -DWITH_HIPBLASLT=true -DWITH_ROCBLAS=false  # hipBLASLt only
cmake -S src -B build -DWITH_CUDA=false -DWITH_HIPBLASLT=false -DWITH_ROCBLAS=true  # rocBLAS only
```

### Building for CUDA only
```
cmake -S src -B build -DWITH_ROCM=false -DWITH_CUDA=true 
cmake --build build -j
```

### Building for both CUDA and ROCm
```
cmake -S src -B build -DWITH_ROCM=true -DWITH_CUDA=true
cmake --build build -j
```

### Running on ROCm or CUDA
First, select a driver based on which backend you'd like to use
#### ROCm backends
- hipBLASLt: `export DRIVER="hipblaslt"`
- rocBLAS: `export DRIVER="rocblas"`

#### CUDA backends
- cuBLAS: `export DRIVER="cublas"`
- cuBLASLt: `export DRIVER="cublaslt"`

#### Run 4k gemms on the command line
Use the below commands to run "4k" gemms on ROCm or CUDA. You'll need to make sure you've set your driver as shown above
| Precision | Base Command                                                                                                                                                                                                                                                                            |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FP64      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 10 --cold_iters 2 --function matmul --precision d --rotating 512 --driver ${DRIVER}                                                                    |
| FP32      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 10 --cold_iters 2 --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --function matmul --rotating 512 --driver ${DRIVER}      |
| TF32      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA T --transposeB N --initialization trig_float --iters 10 --cold_iters 2 --compute_type CUBLAS_COMPUTE_32F_FAST_TF32 --function gemm_ex --precision s --rotating 512 --driver ${DRIVER}                         |
| FP16      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 10 --cold_iters 2 --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r --function matmul --rotating 512 --driver ${DRIVER}    |
| BF16      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 10 --cold_iters 2 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --function matmul --rotating 512 --driver ${DRIVER}|
| FP8       | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA T --transposeB N --initialization trig_float --iters 10 --cold_iters 2 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --function matmul --rotating 512 --driver ${DRIVER}      |
| INT8      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA T --transposeB N --initialization rand_int --iters 10 --cold_iters 2 --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --function matmul --rotating 512 --driver ${DRIVER}         |

#### Running gemms from a YAML file
Instead of specifying all parameters on the command line, you can use the `--yaml` argument to provide a configuration file that defines one or more GEMM operations. This is especially useful for running benchmarks across multiple matrix sizes, data types, or configurations.

The YAML file should contain a list of GEMM configurations. Each configuration can specify parameters such as matrix dimensions (`m`, `n`, `k`), transpose operations (`transA`, `transB`), data types (`a_type`, `b_type`, `c_type`, `d_type`), compute type, and other options.

For example, to run all the 4k gemms from the table above using the provided example YAML file:
```
build/mblas-bench --yaml examples/4k_gemms.yaml --driver ${DRIVER}
```