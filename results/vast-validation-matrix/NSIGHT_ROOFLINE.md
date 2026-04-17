# Nsight Compute / roofline (optional vast.ai row)

**Goal:** Decide whether the next bottleneck at fixed **N, D** is **DRAM**, **SM**, or **kernel mix** (SGEMV vs CUB vs BFS).

## Suggested commands (A100 / Ada)

After `make` on the GPU instance:

```bash
ncu --set full -o mars_query -k regex:cosine -f ./validate 10000 768
# or target SGEMV name from cuBLAS if using cublas path:
# ncu --kernel-name-base demangled ...
```

Report (paste into `validation_matrix.template.json` or a sibling JSON):

- **DRAM throughput** (% of peak) during similarity vs during CUB radix sort.
- **Achieved occupancy** and **warp issue efficiency** for the dominant kernel.
- **Time %** per stage if using Nsight Systems: `nsys profile -o mars_q ./validate 10000 768`

This satisfies the **vast-roofline** todo as a **repeatable harness description**; numbers are host-specific and belong under `results/` per run.
