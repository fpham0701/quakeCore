# Live Quake Co-Traversal — Design Spec

**Date:** 2026-05-12
**Status:** Draft, pending user review
**Author:** brainstormed with Claude
**Related:** `docs/superpowers/specs/2026-05-12-bsp2-megamap-support-design.md`

## Goal

Run the quakeCore frustum-culling benchmark in tandem with a live, modern
Quake source port, comparing per-frame frustum-cull work against the
game's own native cull pass on the *same* camera and view planes. Support
two deployment modes:

- **Local:** game and benchmark sidecar on the same workstation.
- **Remote:** game on the workstation, benchmark sidecar on Perlmutter
  (A100), packets streamed over TCP via an SSH-forwarded port.

The benchmark must not perturb the game's frame loop — the game never
blocks on the sidecar.

## Non-Goals

- Replacing the game's renderer or visibility pipeline.
- Modifying quakeCore's three engines to consume PVS. This spec compares
  the frustum-cull *stage* only.
- Live HUD / dashboard. CSV per run is enough for v1.
- Automating Perlmutter setup. v1 documents a manual Slurm + tunnel flow.
- Supporting Quake ports other than Ironwail in v1.

## Target Port

**Ironwail**, pinned to a specific upstream commit (recorded in
`thirdparty/ironwail.commit`). Rationale: plain C, GPL, well-isolated
`R_MarkLeaves` / `R_MarkSurfaces` / `R_CullBox` path, fast renderer
(meaningful baseline), runs the maps fetched by `scripts/fetch_maps.sh`.

## Architecture

Three processes, one versioned wire protocol:

```
┌────────────────────┐   FramePacket    ┌────────────────────┐
│ Ironwail (patched) │ ───────────────► │ quakecore_live     │
│   frame_probe.c    │   SHM or TCP     │   sidecar          │
│ runs cull + emits  │ ◄─── (none) ──── │ baseline+CPU+GPU   │
└────────────────────┘                  └─────────┬──────────┘
                                                  │ CSV
                                                  ▼
                                         results/<run_id>.csv
```

Coupling is **fire-and-forget async**. Game never waits.

## Components

### 1. `libquakecore_frame_protocol` (new)

Small C library (C99, no C++) shared by Ironwail patch and sidecar.
Exposes:

- Packet structs (`FramePacket`, `HandshakePacket`) — fixed layout,
  little-endian, no padding (`#pragma pack(1)`).
- `qcfp_hash_file(path, out32)` — sha256 of a BSP file.
- `FrameTransport` vtable with two implementations:
  - `qcfp_shm_open_producer/consumer(name)` — POSIX SHM SPSC ring,
    power-of-two slot count, lock-free using atomic head/tail.
    Capacity default 1024 packets.
  - `qcfp_tcp_connect/listen(host, port)` — length-prefixed framed TCP.
- `qcfp_send(transport, packet) -> {OK, DROPPED, ERROR}` — never blocks
  beyond a single atomic CAS on the SHM path.
- `qcfp_recv(transport, &packet) -> {OK, EMPTY, EOF, ERROR}`.

Header lives at `include/quakecore_live/frame_protocol.h`. Implementation
under `src/live/`.

### 2. Ironwail `frame_probe` patch

A new C file `frame_probe.c` added to Ironwail's source, plus three
small edits:

- `host.c`: parse two CLI flags — `--qcfp-transport <spec>` and
  `--qcfp-handshake-once` — and call `FrameProbe_Init()` after world
  load, `FrameProbe_Shutdown()` on quit.
- `gl_rmain.c` (or equivalent): call `FrameProbe_BeginCull()` immediately
  after `R_MarkLeaves`, `FrameProbe_EndCull()` immediately after the
  cull pass, and `FrameProbe_Emit()` right before `R_DrawWorld`.
- `CMakeLists.txt` (Ironwail's): link `libquakecore_frame_protocol`.

`FrameProbe_Emit` constructs a `FramePacket` from:
- `cl.time` → `frame_id` (frame counter, monotonically increasing).
- `r_refdef.vieworg`, `vpn`, `vright`, `vup` → camera.
- `frustum[0..3]` extended to 6 planes by appending a near plane at
  ~1e-3 in front of the eye and a far plane at ~1e9 behind, both with
  inward normals — chosen so the 6-plane reject-corner test reduces
  exactly to the 4-plane test for any finite AABB inside the world.
- `c_brush_polys`-style counters captured around the cull pass for
  `game_visited_nodes` / `game_accepted_leafs`. (Exact counter sites
  determined during implementation; v1 may stub these as 0 if the
  Ironwail counters don't decompose cleanly — the timing comparison is
  the headline.)

Handshake: the first call to `FrameProbe_Emit` (or `FrameProbe_Init` if
`--qcfp-handshake-once` is set) sends a `HandshakePacket` carrying the
BSP path and sha256.

### 3. `quakecore_live` sidecar binary

New binary built when `QUAKECORE_BUILD_LIVE=ON`. CLI:

```
quakecore_live --transport <spec> --csv out.csv [--replay file.bin]
               [--record file.bin] [--threads N] [--block-size N]
```

Behavior:

1. Listen for `HandshakePacket`. Validate sha256 against the BSP at the
   path provided; refuse mismatches.
2. Parse the BSP once via existing `BspParser`.
3. Per `FramePacket`:
   - Build a `std::vector<Camera>{1}` from the packet.
   - Construct a 6-plane `Frustum` from the packet's planes.
   - Run `RunBaselineTraversal`, `RunCpuOptimizedTraversal`,
     `RunGpuOptimizedTraversal` (per existing `engines.hpp` contract),
     each timed with `std::chrono::steady_clock`.
   - Cross-check parity (baseline vs CPU-opt vs GPU-opt) — fail-fast on
     internal divergence, same as `quakecore_bench`.
   - Append CSV row: `frame_id, game_ns, baseline_ns, cpu_ns, gpu_ns,
     game_visited, game_accepted, qc_visited, qc_accepted, drops_so_far`.
4. On `--record`, also dump raw packets to a binary file for offline
   replay. `--replay` mode reads such a file instead of a live transport,
   for reproducible benchmarking.

GPU initialization (CUDA context, packed-struct upload) happens once
during handshake so per-frame work excludes warm-up.

## Wire Protocol (v1)

Little-endian, packed:

```c
struct PacketHeader {
    uint32_t magic;       // 'Q','C','F','P' (0x50464351 LE)
    uint16_t version;     // 1
    uint16_t type;        // 1=handshake, 2=frame
    uint32_t length;      // payload bytes following this header
};

struct HandshakePacket {
    PacketHeader hdr;
    uint8_t  bsp_sha256[32];
    uint16_t bsp_path_len;
    char     bsp_path[];      // bsp_path_len bytes, not null-terminated
};

struct FramePacket {
    PacketHeader hdr;
    uint64_t frame_id;
    uint64_t t_game_cull_start_ns;
    uint64_t t_game_cull_end_ns;
    uint32_t game_visited_nodes;
    uint32_t game_accepted_leafs;
    uint8_t  bsp_sha256[32];
    float    cam_origin[3];
    float    cam_forward[3];
    float    cam_right[3];
    float    cam_up[3];
    uint8_t  n_planes;        // always 6 in v1
    uint8_t  _pad[3];
    struct { float n[3]; float d; } planes[6];
};
```

Total `FramePacket` size: 200 bytes. Trivially fits in any reasonable
SHM slot or TCP MSS.

## Transports

### SHM ring (local)

- POSIX `shm_open` + `mmap` of a fixed-size segment.
- Single-producer single-consumer.
- 1024-slot ring of `FramePacket`-sized cells (~200 KB).
- Head/tail are `_Atomic uint64_t`, monotonic, masked on access.
- Producer-side: if `head - tail == capacity`, return `DROPPED` without
  writing. Producer never spins.
- Consumer-side: blocking with a `sem_t` posted on each enqueue, plus a
  non-blocking poll mode.

### TCP (remote)

- Single connection, length-prefixed framing (the `length` field in
  `PacketHeader`).
- Sidecar listens; Ironwail connects. On disconnect, Ironwail logs and
  continues without emitting — playing the game is never interrupted.
- TCP_NODELAY on, no Nagle batching.

## Map Identity & Index Stability

- `bsp_sha256` is the contract that both sides see the same world.
- Node indices used in `visited_nodes` counters are positions in the BSP
  `nodes` lump — Ironwail's `cl.worldmodel->nodes[i]` and quakecore's
  `BspData.nodes[i]` correspond by index because both loaders preserve
  on-disk order.

## Parity Semantics

- **quakecore-internal parity** (baseline vs CPU-opt vs GPU-opt):
  hard-checked, sidecar exits with code 2 on divergence — same contract
  as `quakecore_bench` today.
- **game-vs-quakecore parity**: reported in CSV, *not* enforced. Their
  cull predicates may differ in edge cases (on-plane handling,
  short-circuit order). Divergence is interesting, not fatal.

## Perlmutter Feasibility & Plan

Verdict: feasible for streaming, not for tight loops.

- **Reachability**: Perlmutter compute nodes are not internet-reachable.
  Use an interactive Slurm allocation; from the workstation, open an SSH
  tunnel to a NERSC login node forwarding a port to the compute node's
  listener. `ssh -L 5000:nidXXXXXX:5000 perlmutter.nersc.gov`.
- **Latency budget**: ~30–100 ms WAN RTT. Because coupling is async
  fire-and-forget, this only affects CSV freshness, never the game.
- **Bandwidth**: 200 B/packet × 200 fps = 40 KB/s. Negligible.
- **Failure modes**: tunnel drops, Slurm allocation expires, sidecar
  OOM. All three manifest as transport disconnect, which the game
  handles by ceasing to emit and logging. No state corruption.
- **v1 deliverable**: a documented manual flow — `scripts/perlmutter_live.md`
  with the Slurm script and tunnel command. Automation deferred.

## Build & CMake

- New CMake option `QUAKECORE_BUILD_LIVE=ON` (default OFF) gates:
  - `libquakecore_frame_protocol` static lib (C99).
  - `quakecore_live` executable.
- Ironwail patch lives at `thirdparty/ironwail/patches/0001-frame-probe.patch`
  plus the new files under `thirdparty/ironwail/frame_probe/`.
- A helper script `scripts/build_ironwail_with_probe.sh` clones Ironwail
  at the pinned commit, applies the patch, builds it linking against the
  installed `libquakecore_frame_protocol`.

## Testing

- Unit: SHM ring SPSC under producer-overrun (`DROPPED` accounting),
  packet roundtrip, sha256 of a known BSP.
- Integration: a "fake game" test binary that opens an SHM transport and
  emits a deterministic camera path identical to the existing
  `GenerateCameraPath`. Run `quakecore_live` against it and confirm that
  the CSV output matches `quakecore_bench --csv` on the same map+seed.
  This is the canonical regression test — it proves the live path
  doesn't perturb traversal results.
- Manual: run patched Ironwail on `start.bsp` locally, confirm
  `quakecore_live` produces a CSV row per frame and that timings are
  in the expected ballpark.
- Perlmutter dry run: same as manual, but sidecar on an A100. Verify
  end-to-end packet flow over the tunnel.

## Risks & Open Questions

- **Ironwail cull counter decomposition.** Ironwail's cull is folded
  into surface marking; cleanly extracting `visited_nodes` /
  `accepted_leafs` counters may require more invasive instrumentation
  than v1 budgets for. Fallback: ship game-side timing only, leave
  parity counters at 0. Decided at implementation time.
- **6-plane synthesis correctness.** The synthetic near/far planes must
  be far enough that no in-world AABB ever has a reject corner on the
  cull side. Validated via the integration test (game-side vs
  quakecore-side counts must match exactly when Ironwail's 4-plane test
  agrees with quakecore's 6-plane test on the same AABBs).
- **SHM cleanup on crash.** If Ironwail crashes mid-run the SHM segment
  leaks until reboot or manual `shm_unlink`. v1 unlinks on sidecar
  startup matching `name` prefix; documented behavior.
- **NERSC policy on long-running listeners.** Interactive Slurm jobs
  have wall-time limits; for multi-hour play sessions the user will need
  to renew the allocation. Out of scope to automate; documented.

## Deliverables Checklist (v1)

- [ ] `libquakecore_frame_protocol` with SHM + TCP transports, unit
      tests, sha256 helper.
- [ ] Ironwail patch series at pinned commit + build helper script.
- [ ] `quakecore_live` sidecar binary with CSV output and replay mode.
- [ ] Integration test using fake-game emitter that reproduces
      `quakecore_bench` results bit-for-bit.
- [ ] Local end-to-end demo on a fetched map.
- [ ] `scripts/perlmutter_live.md` documenting the manual Slurm + tunnel
      flow.
