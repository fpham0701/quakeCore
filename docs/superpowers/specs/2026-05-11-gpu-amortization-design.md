# GPU Amortization — Design Spec

**Status:** Approved (brainstorming complete, awaiting implementation plan)
**Date:** 2026-05-11
**Source motivation:** `GPU_AMORTIZATION.md` in repo root

## 1. Scope

Implement approaches **1, 2, and 4** from `GPU_AMORTIZATION.md`:

1. Persistent CUDA context + persistent device buffers + one-time static map upload.
2. CUDA Graphs to collapse per-frame submission overhead.
4. First-class multi-view-per-frame input so the benchmark can demonstrate the regime where GPU amortization wins.

Out of scope (explicitly deferred):

- Approach 3 (GPU-resident visibility, graphics-API interop). Requires committing to OpenGL/Vulkan/DX12 and a downstream draw consumer; this repo is a benchmarking harness, not a renderer.
- Approach 5 (persistent megakernel). The doc itself recommends skipping unless profiling proves graph launch is the bottleneck.

## 2. Goals and non-goals

**Goals**

- Make steady-state per-frame GPU traversal cost visible in the existing benchmark output, separately from cold-start cost.
- Keep the existing `RunGpuOptimizedTraversal` one-shot function intact so existing parity numbers do not move.
- Preserve the parity contract: `visited_nodes` and `accepted_leafs` must match baseline exactly.
- No changes to the traversal kernel itself. The kernel is already at 81% SM throughput; the spec is about the host harness around it.

**Non-goals**

- Performance tuning of the kernel.
- New graphics-API interop.
- A test framework (the repo has none; testing remains manual via `quakecore_bench`).
- A new standalone app for the amortized engine. The bench is where the comparison matters.

## 3. Architecture

Three units, each with one responsibility:

1. **`quakecore_gpu_engine` library** (existing target, modified). Gains a new translation unit `src/engines/gpu_context.cu` defining the opaque context struct and four C-style entry points. The existing `src/engines/gpu_opt_engine.cu` is preserved so the existing one-shot `RunGpuOptimizedTraversal` continues to report cold-start cost. The traversal kernel is extracted into a shared internal header `src/engines/gpu_kernel.cuh` and included by both translation units — one definition, two callers.

2. **`include/quakecore/gpu_context.hpp`** (new public header). Declares the opaque struct + four free functions. No CUDA types leak into this header. Consumers that include it do not need `<cuda_runtime.h>`.

3. **`src/apps/bench_main.cpp`** (modified). Gains a `--views-per-frame K` CLI flag (default 1) and a new `gpu_amortized` row in the report.

**Unit contract:** the context owns exactly the GPU-side state that `GPU_AMORTIZATION.md` identifies as one-time — CUDA context init, device buffers, packed BSP upload, CUDA Graph. Host-side per-frame frustum building stays on the caller side, as the doc prescribes. Nothing else moves.

## 4. Public API

`include/quakecore/gpu_context.hpp`:

```cpp
#pragma once

#include <vector>
#include "quakecore/types.hpp"

namespace quakecore {

// Opaque — full definition lives in src/engines/gpu_context.cu.
struct GpuTraversalContext;

// One-time setup. Allocates and uploads packed nodes/leafs once.
// Captures a CUDA Graph sized for max_views_per_frame on the first
// GpuContextRun call (lazy, so the cost is visible to the caller).
GpuTraversalContext* GpuContextCreate(const BspData& bsp,
                                      int max_views_per_frame,
                                      int block_size);

// One frame of work. views_for_frame.size() must equal the
// max_views_per_frame passed to Create — see API decision below.
// Returns stats summed over just this frame's views.
TraversalStats GpuContextRun(GpuTraversalContext* ctx,
                             const std::vector<Camera>& views_for_frame);

// Releases device memory, CUDA Graph, frustum staging buffer.
// Does not destroy the CUDA primary context (process-wide, reused).
void GpuContextDestroy(GpuTraversalContext* ctx);

}  // namespace quakecore
```

Four explicit API decisions:

- **`max_views_per_frame` is fixed at Create time.** This is what makes graph capture sound — launch geometry and all kernel parameters are computed once from this number, the graph is captured once, and every subsequent run reuses it without `cudaGraphExecKernelNodeSetParams`. The benchmark fixes K via `--views-per-frame`, so this is not a usability constraint in practice.
- **`Run` requires exactly `max_views_per_frame` views, not "up to".** This keeps all kernel parameters constant across launches — the only thing that changes between frames is the *contents* of the pinned host frustum buffer, which the captured `cudaMemcpyAsync` copies fresh on every launch. Supporting a variable count would force either re-capturing the graph or using `cudaGraphExecKernelNodeSetParams`, neither of which serves the benchmark. Mismatched sizes throw.
- **Graph capture is lazy, on the first `Run`.** Capture needs at least one real launch path to record, and placing it in `Run` means first-frame timing naturally includes the capture cost — information we want to *measure*, not hide.
- **`Run` returns per-frame stats, not aggregated.** The caller sums them. This keeps the API symmetric with `RunBaselineTraversal` and makes parity-checking trivial.

## 5. Data flow

### At `Create` (one-time costs, paid once)

1. Warm the CUDA primary context with a no-op `cudaFree(0)` — absorbs the ~150 ms first-call hit here so the first `Run` does not pay it.
2. Compute total device buffer size: `[nodes][leafs][frustum_max][stats]` with 256-byte alignment between regions. `frustum_max = max_views_per_frame × 6 planes`.
3. Single `cudaMalloc` for the whole region. Store base pointer and offsets in the context struct.
4. Pack `BspData::nodes` and `BspData::leafs` into `NodePacked` / `LeafPacked` (same code as the existing one-shot, moved). Single `cudaMemcpy` H2D for the nodes+leafs prefix.
5. Allocate pinned host frustum staging buffer of `frustum_max × sizeof(PlaneGpu)` via `cudaMallocHost`. Pinned registration is amortized because every frame uses this buffer.
6. Pre-compute launch geometry (threads/blocks) sized for `max_views_per_frame`.
7. Create a non-default CUDA stream for graph capture. The default stream cannot be captured.

### At first `Run` (graph capture, paid once)

1. Build all `max_views_per_frame × 6` frustum planes from `views_for_frame` into the pinned host buffer.
2. Record into the graph: (a) async H2D copy of the frustum buffer, (b) `cudaMemsetAsync` of the device stats region to zero, (c) kernel launch with all parameters (including `num_cameras = max_views_per_frame`) as constants, (d) async D2H of `DeviceStats` (32 B) into a pinned host result slot. Wrap with `cudaStreamBeginCapture` / `cudaStreamEndCapture` on a dedicated non-default stream.
3. `cudaGraphInstantiate` → `cudaGraphExec_t` stored in the context. Discard the `cudaGraph_t`.
4. `cudaGraphLaunch` once on the capture stream to actually run frame 1.
5. `cudaStreamSynchronize`. Read stats from pinned host slot. Return.

### At steady-state `Run` (every subsequent frame)

1. Build all `max_views_per_frame × 6` frustum planes into the pinned host buffer. The captured graph's H2D node points at this same buffer, so just-rewritten contents propagate to the device on launch.
2. `cudaGraphLaunch(graph_exec, stream)`.
3. `cudaStreamSynchronize`.
4. Return stats from pinned host slot.

Per-frame work after warmup is: one host frustum build + one graph launch + one sync. No malloc, no kernel parameter updates, no per-call kernel launch overhead.

## 6. Benchmark integration

### CLI changes (`src/apps/app_common.hpp`)

- New flag `--views-per-frame K` (default 1).
- New field in `BenchmarkConfig`: `int views_per_frame{1};`.
- One new branch in the parse loop.

### Camera-path interpretation

`GenerateCameraPath(bsp, frames * views_per_frame, seed)` produces a flat list. The bench groups consecutive K cameras into one frame. With K=1, total camera count is identical to today, so existing parity numbers do not shift.

### Report rows

Four engines reported by `quakecore_bench`:

| Row | What it measures |
|---|---|
| `baseline` | Existing — one-shot over all `frames × K` cameras |
| `cpu_opt` | Existing — one-shot over all `frames × K` cameras |
| `gpu_opt` | Existing — one-shot over all `frames × K` cameras (includes cold-start) |
| `gpu_amortized` | New — Create + N×Run + Destroy, with per-frame timing collected |

The `gpu_amortized` row reports four numbers in addition to the existing totals:

- `time_s` — wall-clock for the whole Create+loop+Destroy, comparable to other rows.
- `first_frame_ms` — Run #1 wall-clock (includes graph capture).
- `steady_median_us` — median of Run #2..N wall-clock, per frame.
- `steady_p99_us` — 99th percentile of Run #2..N. Captures graph-launch jitter, which is what approach 2 is supposed to drive down.

Per-frame timing is wall-clock (`std::chrono::high_resolution_clock`) around each `GpuContextRun`, not CUDA events. Reason: `Run` includes `cudaStreamSynchronize`, and we want to measure what the caller actually pays. CUDA events would understate by excluding the sync.

### CSV schema

Existing columns unchanged. New columns `first_frame_ms`, `steady_median_us`, `steady_p99_us` are added for all rows (empty for the three that do not measure per-frame). Keeps the CSV rectangular for downstream tooling.

### Parity check

`gpu_amortized` sums per-frame stats and compares to baseline on `visited_nodes` and `accepted_leafs`, same predicate as today. New `correctness_gpu_amortized=PASS/FAIL` line. Non-zero exit if any of the three checks fail.

### Standalone apps

`quakecore_gpu_opt` is unchanged — it continues to demonstrate cold-start cost. No new standalone app is added for the amortized engine; the bench is where the comparison matters.

## 7. Parity, testing, edge cases

### Parity contract

The existing `CheckParity` predicate (`visited_nodes` and `accepted_leafs` match baseline) is preserved verbatim. The `gpu_amortized` row's stats come from summing per-frame `TraversalStats` returned from `GpuContextRun`. Summation is exact `uint64_t` addition; totals fit in `uint64_t` for any realistic workload.

### The kernel is byte-identical to today

This is the load-bearing claim of the spec. The traversal kernel — including the reject-corner `PlanePass` predicate, the per-thread stack, the warp/block reductions — is not modified. `gpu_kernel.cuh` is moved from inside `gpu_opt_engine.cu` to a shared header without edits. If `gpu_opt` passes parity today, `gpu_amortized` passes parity tomorrow, modulo the per-frame split.

### The per-frame split is the only new correctness risk

Running K cameras in one launch over N frames must produce the same total counts as one launch of N×K cameras. The kernel already supports this — it is a grid-stride loop over `num_cameras`, and per-camera work is independent. The single thing that could break parity is wrong frustum-buffer indexing.

Mitigations:

- Frustum pack order: `views_for_frame[i].plane[j]` → host buffer index `i*6+j`, identical to the existing one-shot.
- Per-frame the buffer is fully rewritten; tail beyond `num_cameras × 6` is zero-padded; the kernel never reads it because the grid-stride loop bounds on `num_cameras`.

### Edge cases

| Case | Behavior |
|---|---|
| `views_for_frame.size() != max_views_per_frame` | Throw `std::runtime_error`. Caller bug. The bench never hits this since it groups exactly K cameras per frame. |
| `max_views_per_frame <= 0` in Create | Throw `std::runtime_error`. |
| `frames == 0` | Bench skips the `gpu_amortized` row. No Create/Destroy. |
| `frames == 1` | `first_frame_ms` populated; `steady_median_us` and `steady_p99_us` reported as empty in CSV / `nan` in stdout. |
| Map with zero nodes/leafs | Existing code path in Create. Kernel returns immediately. |
| `block_size < 32` | Round up to 32 (existing behavior preserved). |
| `cudaGraphInstantiate` failure | Bubble up as `std::runtime_error` from the first `Run`. No silent fallback. |

### Test plan (manual, since repo has no test framework)

1. Build: `cmake -S . -B build -DQUAKECORE_BUILD_CPU_OPT=ON -DQUAKECORE_BUILD_GPU_OPT=ON -DQUAKECORE_BUILD_BENCH=ON && cmake --build build -j`.
2. `./build/quakecore_bench --map <map.bsp> --frames 1024 --seed 7`. Expect `correctness_gpu_amortized=PASS`. `visited_nodes` and `accepted_leafs` must match the existing `gpu_opt` row exactly (K=1 default).
3. `./build/quakecore_bench --map <map.bsp> --views-per-frame 8 --frames 1024 --seed 7`. Total cameras = 8192. Compare `visited_nodes` / `accepted_leafs` to a one-shot run with `--frames 8192`. Must be identical — proves per-frame split is exact.
4. `./build/quakecore_bench --map <map.bsp> --views-per-frame 1 --frames 10000`. Confirm `steady_median_us` lands in single-digit µs on an A100 — the headline number the doc predicts.
5. `compute-sanitizer ./build/quakecore_bench --map <map.bsp> --frames 100 --views-per-frame 8` — must report no errors.

## 8. Build system

Additive only.

- `src/engines/gpu_context.cu` added to the `quakecore_gpu_engine` source list in `CMakeLists.txt`.
- Kernel moved into `src/engines/gpu_kernel.cuh` (new header, included by both `gpu_opt_engine.cu` and `gpu_context.cu`).
- Public header `include/quakecore/gpu_context.hpp` is picked up automatically by the existing `include_directories`.
- No new options, dependencies, or toolchain bumps.
- CUDA target architecture stays `80` (A100) per `CMakeLists.txt`.

## 9. Files touched

**New:**

- `include/quakecore/gpu_context.hpp`
- `src/engines/gpu_context.cu`
- `src/engines/gpu_kernel.cuh`

**Modified:**

- `CMakeLists.txt` — one line to add `gpu_context.cu` to the gpu engine library.
- `src/engines/gpu_opt_engine.cu` — kernel body removed (now in `gpu_kernel.cuh`); kept as the one-shot wrapper.
- `src/apps/app_common.hpp` — add `views_per_frame` field and parse branch.
- `src/apps/bench_main.cpp` — add `gpu_amortized` row, per-frame loop, percentile reporting, parity check, CSV column additions.

**Untouched:**

- All traversal kernel logic.
- Baseline and CPU engines.
- `src/apps/gpu_opt_main.cpp`, `cpu_opt_main.cpp`, `baseline_main.cpp`.
- BSP parser, camera path, frustum builder.
