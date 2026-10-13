---
title: "cuBLASLt solution information worth recording"
date: "2026-10-13"
research_question: "Should the B200 cotenant experiment record solution information, and what solution information does cuBLASLt expose?"
depth: "quick"
---

# cuBLASLt solution information worth recording

## Executive summary

Yes. The experiment must record the selected cuBLASLt solution to distinguish a
change in heuristic selection from a performance change experienced by the same
algorithm configuration. cuBLASLt exposes a heuristic-result record plus
queryable configuration and capability attributes on its opaque algorithm
descriptor. The implementation now records the heuristic result fields,
complete available configuration fingerprint, and runtime cuBLASLt version.
The public structured API does not provide a stable human-readable kernel name
or a predicted execution time.

## Key takeaways

- `solution_index` is only the rank in one heuristic query; it is not a durable
  algorithm identity.
- A useful selected-solution fingerprint is algorithm ID plus tile, stages,
  split-K, reduction, CTA swizzle, custom option, inner shape, and cluster shape.
- `wavesCount` is an estimated device-utilization metric, not a timing estimate.
- `workspaceSize` belongs to each heuristic result and can distinguish otherwise
  similar candidates or explain why a candidate is admissible.
- The opaque algorithm descriptor may be serialized, but NVIDIA only promises
  restoration with the same cuBLAS library version; record
  `cublasLtGetVersion()` alongside any serialized descriptor.

## Findings

### Heuristic-result information

`cublasLtMatmulAlgoGetHeuristic()` returns
`cublasLtMatmulHeuristicResult_t`. Its documented fields are:

| Field | Meaning | Record? |
|---|---|---|
| `algo` | Configured opaque algorithm descriptor | Yes; query its configuration attributes |
| `workspaceSize` | Actual required workspace bytes | Yes |
| `state` | Status; other fields are valid only on success | Yes, especially when retaining rejected/failed candidates |
| `wavesCount` | Utilization estimate; `1.0` suggests full-GPU occupancy | Yes |
| `reserved[4]` | Reserved | No |

The result has no documented predicted latency field. Actual benchmark time must
continue to come from CUDA-event measurement.

### Selected algorithm configuration

The configured `cublasLtMatmulAlgo_t` can be queried with
`cublasLtMatmulAlgoConfigGetAttribute()`. The current documentation exposes:

| Attribute | Meaning | Current CSV |
|---|---|---:|
| `CUBLASLT_ALGO_CONFIG_ID` | Algorithm index | Yes |
| `CUBLASLT_ALGO_CONFIG_TILE_ID` | Selected tile | Yes |
| `CUBLASLT_ALGO_CONFIG_STAGES_ID` | Selected pipeline stages configuration | Yes |
| `CUBLASLT_ALGO_CONFIG_SPLITK_NUM` | Number of parallel K splits | Yes |
| `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME` | Split-K reduction scheme | Yes |
| `CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING` | CTA coordinate swizzle | Yes |
| `CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION` | Algorithm-specific option | Yes |
| `CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID` | Internal kernel shape | Yes, or `-1` when unavailable in the build headers |
| `CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID` | Thread-block cluster shape | Yes, or `-1` when unavailable in the build headers |

Stages and the two shape fields are particularly relevant to a Blackwell B200
experiment. Algorithm ID alone is insufficient because one algorithm can have
multiple valid configurations.

### Algorithm capability information

`cublasLtMatmulAlgoCapGetAttribute()` reports what an algorithm *can* support,
not necessarily what the selected configuration uses. Available capabilities
include split-K support, reduction masks, CTA-swizzle support, supported tile
and stages IDs, custom-option range, batch modes, epilogue and pointer-mode
masks, numerical implementation flags, minimum A/B/C/D alignment, custom memory
order, and floating-point-emulation support.

For this experiment, capability fields are secondary. The most useful optional
ones are `CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS` and minimum alignment fields.
Large arrays of every supported tile or stage do not need to be repeated in
each CSV row; they can be stored once in a sidecar keyed by library version and
algorithm ID.

### Versioning and serialization

NVIDIA documents `cublasLtMatmulAlgo_t` as trivially serializable and restorable
for use with the same cuBLAS library version. `cublasLtGetVersion()` returns the
runtime cuBLASLt version. Therefore, a raw descriptor blob can be a useful
debug artifact, but it is not a portable long-term identifier and should not
replace explicit configuration columns.

## Mechanisms / how it works

1. Build the operation, layout, and preference descriptors.
2. Apply the SM-count target to the operation descriptor.
3. Query one or more heuristic results.
4. For each successful result, record its rank, state, required workspace, and
   wave count.
5. Query the result's configured algorithm attributes to construct its
   fingerprint.
6. Execute that same configured `algo` and associate measured time/throughput
   with the fingerprint.

## Specifications / numbers

| Item | Value or range | Unit | Notes |
|---|---:|---|---|
| `wavesCount` interpretation | `1.0` | waves | Suggests full-GPU occupancy; not predicted duration |
| CTA swizzling | `0` or `1` | enum value | Other values reserved in current documentation |
| Serialized algorithm compatibility | Same cuBLAS version | version scope | Not promised across versions |

## Confidence and gaps

| Topic | Confidence | Gap / follow-up |
|---|---|---|
| Heuristic fields | high | Official current NVIDIA documentation |
| Configuration attributes | high | Availability of newer inner/cluster attributes should be compile-time guarded for older CUDA toolkits |
| Stable kernel name | medium-high | No human-readable kernel-name field appears in the documented structured algorithm APIs; profiler/logging output is separate and may be unstable |
| Cross-version identity | high | NVIDIA explicitly limits descriptor restoration to the same library version |

## Inference and synthesis (clearly labeled)

For selection-effect analysis, treat two solutions as the same only when their
complete available configuration fingerprints match under the same runtime
cuBLASLt version. This is stricter and more reliable than comparing heuristic
rank or algorithm ID alone. Capability data should be normalized into a sidecar
rather than duplicated across 2,100 invocation rows.

## Sources

| # | Source | Type | URL |
|---|---|---|---|
| 1 | NVIDIA cuBLAS documentation — `cublasLtMatmulHeuristicResult_t` | official documentation | https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulheuristicresult-t |
| 2 | NVIDIA cuBLAS documentation — algorithm configuration attributes | official documentation | https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgoconfigattributes-t |
| 3 | NVIDIA cuBLAS documentation — algorithm capability attributes | official documentation | https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocapattributes-t |
| 4 | NVIDIA cuBLAS documentation — `cublasLtMatmulAlgo_t` | official documentation | https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgo-t |
| 5 | NVIDIA cuBLAS documentation — `cublasLtGetVersion()` | official documentation | https://docs.nvidia.com/cuda/cublas/index.html#cublasltgetversion |

**Accessed:** 2026-10-13

## Further reading

- [cuBLASLt heuristic query](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgogetheuristic)
- [cuBLASLt algorithm validation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck)
