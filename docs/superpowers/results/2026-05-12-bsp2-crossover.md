# BSP2 megamap: GPU vs CPU-opt crossover

**Date:** 2026-05-12
**Map:** `examples/maps/fetched/megamap-bsp2/e3m6.bsp`
**File size:** `1089124` bytes (~1.0 MB)
**BSP2 node count (approximate):** ~4700 (smaller than design-spec 50k target — Quake 1 maps are inherently small; documented in `2026-05-12-bsp2-megamap-summary.txt`)
**Hardware:** NVIDIA GeForce RTX 3080 Laptop GPU (sm_86) + AMD Ryzen 9 5900HS with Radeon Graphics

## Sweep results

| frames  | cpu_opt (s) | gpu_opt (s) | baseline (s) | gpu / cpu_opt |
|---------|-------------|-------------|--------------|---------------|
| 1000    | 0.000435    | 0.153094    | 0.000698     | 351.69        |
| 3000    | 0.000784    | 0.134672    | 0.002147     | 171.83        |
| 10000   | 0.001959    | 0.141202    | 0.007072     | 72.06         |
| 30000   | 0.005103    | 0.164873    | 0.021701     | 32.31         |
| 100000  | 0.015532    | 0.211905    | 0.071391     | 13.64         |
| 300000  | 0.046319    | 0.357513    | 0.209918     | 7.72          |
| 1000000 | 0.167506    | 0.861419    | 0.744584     | 5.14          |

## Crossover

GPU does not beat CPU-opt at any frame count in this sweep. At the largest tested point (1M frames) GPU is still 5.14x slower than CPU-opt. A simple linear fit (`time = a + b * frames`) over all 7 points gives:

- CPU-opt: `time ~= -5.0e-4 + 1.67e-7 * frames`
- GPU-opt: `time ~= 0.140    + 7.21e-7 * frames`

Both intercept and per-frame slope are larger for the GPU on this map, so the lines never cross: the kernel's per-camera work is small enough that PCIe/launch overhead dominates and the AVX2+OpenMP CPU stays ahead at every scale.

Compared to e1m1's ~5M crossover documented in `GPU_AMORTIZATION.md`, no shift toward lower frame counts is observed at this map size — if anything the crossover has moved further out (or disappeared) because the BSP2 megamap is still in the small-node-count regime.

## Interpretation

- The current BSP2 (e3m6 compiled with qbsp -bsp2) is ~4700 nodes; the design-spec target was 50k+. With fewer nodes per camera, the GPU kernel's per-camera work stays small, keeping the crossover at high frame counts (or out of reach entirely on this 3080 Laptop).
- The earlier hypothesis (GPU wins at ~50k-frames on a megamap) cannot be tested with this BSP2. To get a true megamap-scale BSP2, a larger .map source (e.g., a func_msgboard jam map with >50k brushes) must be supplied via `MAP_SOURCE` to `build_megamap.sh`. The build_megamap.sh + ericw-tools path is in place; only the .map asset is missing.
- The recorded numbers still document the small-BSP2 baseline that future runs can compare against. They also confirm that on consumer Ampere mobile silicon, the GPU per-frame slope (~0.72 us/frame) is over 4x the CPU's (~0.17 us/frame at 4 threads) for this map — the GPU needs significantly more per-camera work to amortize that overhead.

Raw CSV: `2026-05-12-bsp2-crossover.csv`.
