# quakeCore ↔ QuakeSpasm Integration — Design

**Date:** 2026-05-11
**Status:** Approved, ready for plan
**Scope:** B-narrow, CPU-only (option (i))

---

## 1. Goal

Replace QuakeSpasm's per-leaf frustum cull with quakeCore's SoA + AVX2 cull, while leaving QuakeSpasm's PVS pipeline intact. This is a CPU-only integration; the GPU engine stays isolated in the standalone `quakecore_bench` and is not pulled into the integrated game binary.

The point is to demonstrate quakeCore's CPU-opt path running inside a real renderer on real maps, with bit-for-bit parity against vanilla QuakeSpasm and reproducible per-frame timing.

## 2. Decisions log (resolved during brainstorming)

| Question | Decision | Why |
|---|---|---|
| Which Quake source? | **QuakeSpasm** (modern source port) | Original id-software/quake on Windows 11 is a multi-day yak-shave (DirectDraw, VC6, palettized SW raster). QuakeSpasm builds cleanly on Windows, SDL2-based, clean hook points in `R_MarkSurfaces` / `R_CullBox`. |
| Replace PVS, frustum cull, or both? | **B-narrow**: keep PVS, replace only frustum cull | B-wide (no PVS) would do strictly more work than vanilla on classic maps and lose by orders of magnitude. Keeping PVS preserves Quake's main visibility win. |
| Include GPU engine in integrated binary? | **No (option (i))** | B-narrow's per-frame work is ~10 µs on CPU; CUDA kernel launch + H2D + D2H is 20–50 µs minimum on Windows. The GPU engine is designed for many-camera parallelism, which B-narrow doesn't have. GPU stays in `quakecore_bench` unchanged. |
| Repo structure | **Submodule (option 2)** | QuakeSpasm pinned under `third_party/quakespasm/` inside this repo. Single clone, reproducible "which version did we test against", no vendoring 50k LOC of someone else's C. |
| Where does the new code live? | **Option (α)**: new engine pair `RunFlatCullBaseline` + `RunFlatCullCpuOptimized` | Preserves `quakecore_bench`'s "bench is canonical test" invariant: the bench parity-checks the *exact* code path the integrated binary runs. Failures are caught on the standalone bench before they become missing geometry on screen. |
| Per-frame data flow | **Cull all leaves, intersect with PVS after** | One linear AVX2 pass over a contiguous SoA — exactly the workload AVX2 is fastest at. Per-frame cost is constant w.r.t. camera position, making the in-game perf comparison reproducible. The "wasted" work on PVS-invisible leaves is ~10 µs/frame on a 30k-leaf map — well in the noise. |

## 3. Architecture

Two pieces of code in one repo:

### 3.1 quakeCore additions

A new engine pair in `include/quakecore/engines.hpp`:

```cpp
struct FlatCullInput {
  const float* mins_x;   // [count], 32-byte aligned
  const float* mins_y;
  const float* mins_z;
  const float* maxs_x;
  const float* maxs_y;
  const float* maxs_z;
  int count;
};

void RunFlatCullBaseline   (const FlatCullInput&, const Frustum&, uint32_t* out_mask);
void RunFlatCullCpuOptimized(const FlatCullInput&, const Frustum&, uint32_t* out_mask);
```

- `out_mask` length: `ceil(count / 32)` `uint32_t`s. Bit `i` set ⇔ leaf `i` accepted by frustum.
- **Baseline**: scalar Akenine-Möller reject-corner predicate, matching `src/core/frustum.cpp::AabbIntersectsFrustum` exactly.
- **CPU-opt**: AVX2, 8 AABBs per iteration, lifted from `IntersectsPlaneAvx` in `cpu_opt_engine.cpp`. No OpenMP (single-camera workload too small to amortize thread spawn).
- The existing three engines (`RunBaselineTraversal`, `RunCpuOptimizedTraversal`, `RunGpuOptimizedTraversal`) stay **completely untouched**.

A new C ABI header `include/quakecore/flat_cull_c.h` wraps these as `extern "C"` for QuakeSpasm to link without dragging C++ into a C codebase.

A new CMake target `quakecore_flat` — static library containing only `RunFlatCull{Baseline,CpuOptimized}` + the C shim. **Zero CUDA dependencies.** This is the only thing the QuakeSpasm fork links.

### 3.2 QuakeSpasm fork

Pinned via submodule at `third_party/quakespasm/`. Adds two new files and edits two existing ones; ~200–300 LOC added, ~20 LOC modified.

- **`Quake/quakecore_glue.c`** (new): owns the per-map SoA AABB buffer + `accept_mask` scratch buffer. `QC_OnMapLoaded(model_t*)` builds the SoA from `model->leafs[1..numleafs]`, skipping the solid outside leaf 0. `QC_OnMapUnload()` frees. Called from `R_NewMap`.
- **`Quake/r_quakecore.c`** (new): exposes `QC_MarkSurfacesViaQuakeCore(int mode)`. Builds a 6-plane frustum from QuakeSpasm's `frustum[]` (4 side planes) plus near/far derived from `r_refdef`; calls the selected flat-cull engine via the C shim; iterates `accept_mask AND pvs_bitmask`, calling QuakeSpasm's existing `R_MarkLeaf` for each surviving leaf.
- **`Quake/gl_rmain.c`** (edit): at the top of `R_MarkSurfaces`, branch on the new `r_cullmode` cvar:
  - `0` → vanilla code path (unchanged)
  - `1` → `QC_MarkSurfacesViaQuakeCore(BASELINE)`
  - `2` → `QC_MarkSurfacesViaQuakeCore(CPU_OPT)`
- **`Quake/r_main.c`** (edit): register `r_cullmode` (default `0`), `r_cullmode_check` (default `0`), `r_cull_stats` (default `0`).

## 4. Data flow

### 4.1 Map load (once per map)

When QuakeSpasm finishes `Mod_LoadBrushModel` for the world, the integration glue walks `cl.worldmodel->leafs[1..numleafs]` (leaf 0 is the solid outside leaf and never visible) and copies each leaf's `minmaxs[6]` into a SoA AABB buffer:

```
mins_x[N], mins_y[N], mins_z[N], maxs_x[N], maxs_y[N], maxs_z[N]
```

Six float arrays, each 32-byte-aligned (AVX2 requirement), where `N = numleafs - 1`. Allocated once per map, freed on map change. The renderer's own structures stay untouched. Cost on a huge map (~30k leaves): ~720 KB.

### 4.2 Per-frame

In `R_MarkSurfaces`, **replacing** the vanilla per-leaf `R_CullBox` loop (when `r_cullmode != 0`):

1. Build a 6-plane `quakecore::Frustum` from QuakeSpasm's `frustum[]` + near/far from `r_refdef`. This happens after PVS for the current viewleaf is computed (PVS computation is unchanged).
2. Call `RunFlatCullCpuOptimized(soa_buffer, N, frustum, accept_mask)`. All N leaves get frustum-tested in one shot.
3. Iterate `for i in 0..N`: if `accept_mask` bit `i` set AND PVS bit `i` set → mark leaf `i+1`'s surfaces via the existing surface-marking code.

No gather/scatter, no allocation in the hot path. Output is a single bitmask.

**Why cull-everything-then-intersect-with-PVS instead of compact-then-cull:**

Compact-then-cull does less frustum work but pays for it with a gather step and an index-mapping table. Cull-everything does ~10 µs/frame of "wasted" work on PVS-invisible leaves but is one linear AVX2 pass over a contiguous SoA — exactly the workload AVX2 is fastest at, and exactly the workload that's easy to compare across engines in the bench. Per-frame cull time is also constant w.r.t. camera position, which makes the in-game perf comparison clean.

If profiling later shows the cull-everything path matters, the switch to compact-then-cull is mechanical and reversible.

## 5. Windows build & toolchain

- **Visual Studio 2022 Community** — free, includes MSVC + Windows SDK. CUDA Toolkit is **not** required for the integrated binary.
- **CMake 3.20+** drives the quakeCore side. Generates a VS2022 solution. `/arch:AVX2` for the `quakecore_flat` target.
- **QuakeSpasm fork** uses its existing `Quakespasm-2022/Quake/Quakespasm.sln`. We add `quakecore_flat.lib` + include path to the project's linker/include settings. SDL2 stays as-is (QuakeSpasm vendors a Windows SDL2 build).
- **Build orchestration**: `scripts/build_windows.bat` runs CMake configure + MSBuild for `quakecore_flat.lib`, then invokes MSBuild on the QuakeSpasm `.sln`. One command from a fresh clone.
- The standalone `quakecore_bench` (with CUDA) stays Linux-only. Bench parity check for the new flat-cull engines runs on Linux as part of the existing pipeline.

## 6. Verification & parity

Three layers:

1. **Standalone bench parity** (extends `bench_main.cpp::CheckParity`): for each `(map, camera)` in the existing bench config, additionally feed `BspData::leafs` as a flat AABB array to both `RunFlatCullBaseline` and `RunFlatCullCpuOptimized`. Asserts:
   - `accept_mask` bit-for-bit identical between baseline-flat and cpuopt-flat.
   - `popcount(accept_mask) == TraversalStats::accepted_leafs` from the existing tree-traversal engines (guaranteed by construction: internal node AABBs are unions of their children's AABBs, so any leaf that survives flat-cull would also be visited by tree-traversal, and any subtree the tree rejects contains only leaves that flat-cull would also reject).
   - Existing tree-traversal parity check stays. Bench exits non-zero on any divergence (existing convention, return 2).
2. **In-game parity self-check** (debug cvar `r_cullmode_check 1`): runs vanilla AND quakeCore cull each frame, compares the final leaf-marked sets, logs divergence count to the console. Off by default (too expensive for normal play).
3. **Visual smoke**: play `start`, `e1m1`, and one community map on each `r_cullmode` value. No missing geometry, no flicker.

**Done bar for B-narrow:** layers 1 and 2 green on ≥3 maps; layer 3 manually confirmed.

## 7. Measurement & output

QuakeSpasm fork emits one CSV row per frame when `r_cull_stats 1`:

```
frame, cullmode, n_leaves_total, n_pvs_visible, n_frustum_accepted, cull_us
```

`cull_us` is measured via `Sys_DoubleTime()` around **just the frustum-cull step** — for `r_cullmode 0` that's the vanilla per-leaf `R_CullBox` loop; for `r_cullmode 1/2` that's the `RunFlatCull*` call only. PVS lookup and surface-marking are *not* included (those are identical across modes and would dilute the signal). `r_cull_stats_dump <path>` console command flushes to file. Replaying an identical demo against `r_cullmode 0/1/2` produces three CSVs — clean A/B/C comparison material.

## 8. Out of scope (explicit non-goals)

- GPU engine in the integrated binary
- BSP version extension in quakeCore's parser (QuakeSpasm handles loading for the integrated binary; integrated binary inherits BSP2/2PSB support for free; standalone bench stays BSP29-only)
- Replacing QuakeSpasm's PVS or surface batching
- Linux/Mac builds of the integrated binary (Windows-first; can be added later)
- Further perf optimization of the cull path beyond the existing AVX2 inner loop
- Multi-camera / shadow caster / lookahead workloads (option (ii) from brainstorming — not picked)
- Demonstrating GPU win on Quake gameplay (option (iii) from brainstorming — not picked)

## 9. Open implementation details (for the plan, not the spec)

These are deliberately deferred to the planning step — they're tactical, not architectural:

- Exact mapping from QuakeSpasm's 4-plane `frustum[]` to quakeCore's 6-plane `Frustum` (near/far synthesis from `r_refdef`)
- C ABI shim layout for `Frustum` (replace `std::array` with C-compatible struct)
- `accept_mask` storage ownership and lifetime (probably in `quakecore_glue.c`, allocated at map load, sized to `ceil(numleafs / 32)`)
- QuakeSpasm submodule pin (a recent stable tag — to be selected at plan time)
- How `scripts/build_windows.bat` discovers VS2022 install path (vswhere.exe vs. user-supplied)
