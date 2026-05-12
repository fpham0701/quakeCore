# BSP2 parser + megamap support — design

**Status:** draft for implementation planning
**Date:** 2026-05-12
**Branch:** `andy/bsp2-megamap`

## Goal

Make the GPU engine "make sense" by loading BSPs ~10–100× larger than Quake 1 v29 can express. The current GPU kernel hits ~81% SM throughput but is overhead-bound below ~5M cameras (see `GPU_AMORTIZATION.md`). Growing per-thread work via a larger BSP shifts the crossover so the GPU wins at tens of thousands of cameras rather than millions.

The route there is supporting the BSP2 on-disk format (32-bit indices, float bounds), used by modern open Quake megamaps, while preserving cross-engine parity on every existing v29 map.

**Concrete deliverables:**

- BSP2 read path in `src/io/bsp_parser.cpp`.
- Widened canonical in-memory types (`BspNodeDisk`, `BspLeafDisk`) that store both v29 and BSP2 data losslessly.
- The three engines (baseline, CPU-opt, GPU-opt) updated to consume the widened types — no algorithmic changes.
- At least one real BSP2 megamap available under `examples/maps/fetched/<dataset>/`, with a reproducible acquisition script.
- `quakecore_bench` runs cross-engine parity on the megamap and exits zero.

## Scope and non-goals

**In scope:**

- Read BSP2 (`'BSP2'` magic) alongside v29 (`0x1D`).
- Widen canonical in-memory types.
- Update engines' "build internal repr" steps to consume the widened types.
- Acquisition path for one real megamap (primary: compile from open `.map` source via `ericw-tools`; fallback: redistributable precompiled `.bsp`).
- Add manifest entry / fetch step so `quakecore_bench` can run on it reproducibly.

**Out of scope:**

- BSP29a / "2PSB" intermediate variant. The Quake community has consolidated on BSP2. Skip unless we encounter a map that needs it.
- Quake 2 / Quake 3 / Source BSPs (different formats entirely).
- Any change to the traversal algorithm, parity contract, frustum predicate, or camera path.
- Procedural BSP generation. Useful follow-up but not part of this work.
- Any new optimization of the GPU kernel itself. The kernel is already at 81% SM throughput; this design only feeds it bigger inputs.

## Background

quakeCore's parser today (`src/io/bsp_parser.cpp`) reads Quake 1 v29 only. v29 uses 16-bit indices and quantized int16 AABB bounds in nodes and leaves, which caps maps at roughly 32k nodes / 8k leaves / 64k faces. Modern Quake megamaps (Arcane Dimensions, Tronyn megamaps, Sock's epics) routinely exceed those limits and ship as BSP2.

BSP2 is a minimal widening of v29: same header layout (15 lumps, same indices), same general lump structure, only the per-element struct sizes change. Float-precision AABB bounds replace quantized int16; 32-bit indices replace 16-bit. The community spec lives in `ericw-tools`' `bspfile.h` (open-source).

The GPU engine processes one camera per thread with `__launch_bounds__(256, 4)` on sm_80. Filling an A100 (108 SMs × 1024 threads/SM = ~110k concurrent threads) requires a healthy multiple of that in cameras. A 50k–100k-node BSP increases per-thread runtime ~50–100× relative to e1m1, which proportionally shrinks the crossover camera count where the GPU wins.

## Format delta (v29 → BSP2)

Header is identical except magic. The fields we currently parse change as follows:

| Struct        | v29 field                                  | BSP2 field                            |
|---------------|-------------------------------------------|--------------------------------------|
| `dnode_t`     | `int16_t children[2]`                      | `int32_t children[2]`                 |
| `dnode_t`     | `int16_t mins[3]`, `int16_t maxs[3]`       | `float mins[3]`, `float maxs[3]`      |
| `dnode_t`     | `uint16_t firstface`, `uint16_t numfaces`  | `uint32_t firstface`, `uint32_t numfaces` |
| `dleaf_t`     | `int16_t mins[3]`, `int16_t maxs[3]`       | `float mins[3]`, `float maxs[3]`      |
| `dleaf_t`     | `uint16_t firstmarksurface`, `uint16_t nummarksurfaces` | `uint32_t firstmarksurface`, `uint32_t nummarksurfaces` |
| `dclipnode_t` | `int16_t` everything                       | `int32_t` everything                  |
| `dedge_t`     | `uint16_t v[2]`                            | `uint32_t v[2]`                       |
| marksurfaces  | `uint16_t`                                 | `uint32_t`                            |

Today's parser reads only vertices, planes, nodes, leafs, and models. The plane and model structs are unchanged between v29 and BSP2. Only nodes and leafs need a parallel struct family. The plan will pin exact field offsets against `ericw-tools/include/common/bspfile.hh` before coding.

Leaf-encoding convention `-(leaf+1)` (negative child = leaf, non-negative = node) is identical in both formats; the index width just grows from int16 to int32.

## Component changes

### `include/quakecore/types.hpp` — widen canonical types

`BspNodeDisk` and `BspLeafDisk` are the canonical in-memory representations that all three engines consume. They widen to store both formats losslessly:

```cpp
struct BspNodeDisk {
  int32_t planenum{0};
  int32_t children[2]{0, 0};  // <0 encodes leaf index as -(leaf+1)
  float   mins[3]{0.0F, 0.0F, 0.0F};
  float   maxs[3]{0.0F, 0.0F, 0.0F};
  int32_t firstface{0};
  int32_t numfaces{0};
};

struct BspLeafDisk {
  int32_t contents{0};
  int32_t visofs{0};
  float   mins[3]{0.0F, 0.0F, 0.0F};
  float   maxs[3]{0.0F, 0.0F, 0.0F};
  int32_t firstmarksurface{0};
  int32_t nummarksurfaces{0};
};
```

Rationale: every engine already operates in float for plane tests. Storing canonical bounds as float removes a per-traversal int16-to-float cast and keeps the engine code uniform regardless of source-format precision. Children at int32 is required for BSP2 anyway and is harmless on v29 input (just sign-extension).

Public API impact: `BspData` is the value type returned by `ParseBspFile`. Widening members of structs it contains is a binary-compatibility change but a source-compatibility change only for code that reads `int16_t` directly (currently: just the three engines, all in this repo).

### `src/io/bsp_parser.cpp` — dispatch on magic

- Add magic detection on the first 4 bytes of the header:
  - `0x0000001D` → v29 (existing path).
  - ASCII `"BSP2"` (= `0x32505342` little-endian) → BSP2 (new path).
  - Anything else → `runtime_error("Unsupported BSP magic: 0x<hex>. Supported: v29 (0x1D), BSP2.")`.
- Define a `DNodeV2` and `DLeafV2` parallel to today's locally-scoped `DNode`/`DLeaf` with the BSP2 field layout.
- Branch the `ReadLumpAs<T>` calls for the nodes and leafs lumps on detected version. All other lumps (vertices, planes, models) use the existing single-format readers.
- Both branches populate the same widened `BspData`. v29 widens int16 → float on load; BSP2 reads float directly.
- Existing lump-bounds validation in `ReadLumpAs<T>` applies unchanged.

### `include/quakecore/bsp_parser.hpp` — docstring only

Update the comment to say "Quake 1 BSP (version 29 or BSP2)". Function signature unchanged.

### `src/engines/baseline_engine.cpp` — drop int16→float cast

Baseline reads `BspNodeDisk` directly during recursion. Today's code casts `n.mins[i]` (int16) to float when building the AABB for the frustum test. With widened types, the read is already float — delete the cast. Traversal logic, recursion structure, parity-relevant counters: unchanged.

### `src/engines/cpu_opt_engine.cpp` — SoA builder

The `NodeSoA` build step copies bounds and children out of `BspNodeDisk`. Source widens from int16 to float; destination is already float. Net change: remove the explicit conversion. SoA layout, AVX2 intrinsics in `IntersectsPlaneAvx`, OpenMP scheduling, parity-relevant counters: unchanged.

### `src/engines/gpu_opt_engine.cu` — packer

The host-side packer that produces `NodePacked` / `LeafPacked` reads from `BspNodeDisk` / `BspLeafDisk` and writes the 16-byte-aligned GPU structs. Source widens from int16 to float; the packed GPU struct was already float for bounds and int32 for children. Net change: drop the conversion. The kernel itself, `__launch_bounds__(256, 4)`, `kStackMax=128`, and `PlanePass` are unchanged.

### Parity contract

`bench_main.cpp::CheckParity` compares `visited_nodes` and `accepted_leafs` across engines.

- **On v29 maps:** int16 → float widening is exact (every int16 representable as float). All three engines see identical bounds. Parity is preserved.
- **On BSP2 maps:** all three engines read the same float bounds from `BspData`. Same predicate (`AabbIntersectsFrustum` / `IntersectsPlaneAvx` / `PlanePass` — Akenine-Möller reject-corner). Parity must hold.

If parity diverges on the BSP2 path, the most likely root cause is a wrong field offset in `DNodeV2`/`DLeafV2`; the test plan below catches this.

## Acquisition path for the megamap

Two-tier approach so implementation is not blocked by toolchain friction.

### Primary: compile from open `.map` source

`ericw-tools` (`qbsp -bsp2`) is BSD-licensed, builds with CMake on Linux. The plan will:

1. Pick a specific `.map` source (license-checked, ≥50k expected nodes after compile). Candidate seed list (final pick during implementation, after license verification):
   - Maps from `quake_map_source` GitHub repo (same upstream we already pull from in `fetch_maps.sh`) that target BSP2.
   - Sock's open-source map releases (`func_msgboard` releases, license to be verified per-map).
   - Tronyn megamap releases (per-map license check).
2. Add `scripts/maps/build_megamap.sh` that:
   - Clones `ericw-tools` at a pinned commit into `examples/maps/.cache/ericw-tools/`.
   - Builds it (CMake, system gcc / clang).
   - Compiles the chosen `.map` with `qbsp -bsp2 -nopercent`.
   - Drops the resulting `.bsp` into `examples/maps/fetched/<dataset>/`.
3. Extend `scripts/fetch_maps.sh` with a new dataset entry (or invoke `build_megamap.sh` as a step) so `bash scripts/fetch_maps.sh` produces the megamap in one command.

### Fallback: redistributable precompiled BSP2

If `ericw-tools` integration is friction during implementation, the plan permits checking in (or `curl`-ing) one freely-licensed precompiled BSP2 map.

**Trigger to switch to fallback:** `ericw-tools` build fails or compile produces an unusable output on the dev box after one reasonable debugging pass (~30 min).

**Validity criteria for the chosen fallback BSP** (must hold before this fallback is acceptable):

- The chosen map's license unambiguously permits redistribution of compiled BSPs.
- The precompiled BSP is hosted at a stable URL we control (GitHub release on a quakeCore fork, or similar) — no third-party links that may rot.
- The compiled map is BSP2 format (not BSP29a / 2PSB) and exceeds ~50k nodes.

The plan will name a concrete fallback candidate that satisfies these criteria *before* implementation begins, so the trigger doesn't require fresh license research mid-flight.

## Data flow (unchanged at the engine boundary)

```
ParseBspFile(path) → BspData (widened types) → engine.BuildInternalRepr(BspData) → engine.Run(cameras) → TraversalStats → CheckParity
```

The only structural change is the type widening inside `BspData`. The contract between parser and engines (value-passed `BspData`) is unchanged. The contract between engines and `bench_main` (`TraversalStats`) is unchanged. The frustum predicate is unchanged across all three engines.

## Error handling

- **Unknown magic:** `runtime_error("Unsupported BSP magic: 0x<hex>. Supported: v29 (0x1D), BSP2.")`. Caller sees a clear, actionable message.
- **Truncated BSP2 lump:** reuses existing `ReadLumpAs` bounds check; error message names the BSP2 lump variant (e.g., "Lump exceeds file size: nodes (BSP2)").
- **Mismatched lump element size:** existing `(length % sizeof(T)) != 0` check fires with `T` = `DNodeV2` etc.; error message identifies the variant.
- **Existing v29 maps:** must continue to load and produce byte-identical `TraversalStats`. Regression guard described below.

## Testing

`quakecore_bench` is the canonical test. Implementation must run it on at least two configurations and exit zero:

1. **Regression: small v29 map.** `quakecore_bench --map examples/maps/fetched/<v29-map>.bsp --frames 2000 --seed 7` — must exit zero with `correctness_baseline_vs_cpu_opt=PASS` and `correctness_baseline_vs_gpu_opt=PASS`. Stats must equal pre-widening baseline (this proves type widening didn't introduce drift on v29).
2. **New path: BSP2 megamap.** `quakecore_bench --map examples/maps/fetched/<megamap>.bsp --frames 2000 --seed 7` — must exit zero with both parity checks PASS.

Additional smoke check: parse `e1m1` before and after widening, compare node/leaf counts and a few sampled bound values; identical means the v29 reader correctly populated the widened struct.

Performance reporting (not a pass/fail gate, but a recorded result): on the BSP2 megamap, sweep `--frames` from 1k to 1M in log steps and identify the crossover where GPU wall-clock ≤ CPU-opt wall-clock. The hypothesis is that this crossover lands at tens of thousands of frames on a ~50k–100k-node BSP2 map, versus the ~5M frames observed on e1m1 — i.e., the crossover should move ~100× lower. A 10× shift would still be a positive result; a sub-10× shift means the kernel is not actually scaling with per-thread work and would require investigation.

## Honest scope estimate

- BSP2 parser dispatch + type widening + three-engine adapter updates: ~1 day with `ericw-tools/bspfile.hh` treated as authoritative.
- Megamap acquisition + script integration: ~0.5 day if `ericw-tools` compiles cleanly on the dev box; ~few hours if the fallback path activates.
- Benchmark run + GPU crossover analysis: ~0.5 day.

**Total: ~2 days of focused work.**

## Risks and mitigations

- **Risk:** `ericw-tools` fails to build on the target Linux box. **Mitigation:** the fallback redistributable-BSP path is named in the plan before implementation starts. Implementation cannot stall waiting for a license check that wasn't done upfront.
- **Risk:** a parser field offset is wrong in `DNodeV2` or `DLeafV2` and parity passes accidentally on a small map. **Mitigation:** parity check on a small map *and* the megamap. A wrong offset will diverge across engines on the larger BSP because misaligned reads accumulate.
- **Risk:** GPU stack depth `kStackMax=128` is too shallow for a megamap's BSP tree. **Mitigation:** during implementation, measure observed max recursion depth on the megamap (instrument baseline traversal). If >100, raise `kStackMax` (tunable, not algorithm change). If the megamap exceeds reasonable stack bounds entirely, swap to a smaller megamap (we have a candidate list).
- **Risk:** v29 widening introduces float-precision drift in some edge case. **Mitigation:** int16 → float is exact for all values in the int16 range; the regression test on e1m1 catches this if a sign-extension or signedness mistake slips in.
- **Risk:** binary compatibility break in `BspData` affects external consumers. **Mitigation:** none needed — `BspData` has no external consumers; the only callers are the three engines in this repo, all updated in the same change.

## Open questions deferred to the implementation plan

- Concrete megamap pick (name, source URL, license).
- `ericw-tools` pin (commit hash).
- Whether the new fetch step is a flag on `fetch_maps.sh` or a separate script invoked from it.
- Whether the crossover sweep gets a dedicated bench mode or is run ad-hoc via the existing CLI.
