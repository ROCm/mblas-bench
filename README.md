### Basic BLAS benchmark for GEMMs


Goals:
- Use the familiar interface of rocblas-bench
- Implement native support for cuBLAS, cuBLASLT, and Intel's GPU BLAS
- Expose implementation specific tuneables to the user

## Building for both CUDA and ROCm
cmake -S src -B build
cmake --build build

## Building for ROCm only
cmake -S src -B build -DWITH\_CUDA=false
cmake --build build

## Building for CUDA only
cmake -S src -B build -DWITH\_ROCM=false
cmake --build build
