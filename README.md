# quakeCore

Standalone Quake 1 BSP parsing and culling benchmark project for HPC experimentation on CPU and GPU architectures (including NERSC Perlmutter).

## What This Repository Contains

- A minimal BSP v29 parser for Quake-style `.bsp` files.
- Three engine implementations:
  - `baseline`: sequential traversal/culling reference.
  - `cpu_opt`: CPU-optimized traversal path (OpenMP + AVX where enabled).
  - `gpu_opt`: CUDA traversal path.
- A unified benchmark runner (`quakecore_bench`) for correctness and performance comparison.
- On-demand map fetch tooling for open/freely redistributable benchmark maps.

## Project Layout

- `include/quakecore/` - public headers and shared types.
- `src/io/` - BSP parsing and validation.
- `src/core/` - frustum math and shared logic.
- `src/engines/` - baseline/CPU/GPU engines.
- `src/apps/` - executable entrypoints.
- `scripts/` - fetch and maintenance scripts.
- `examples/maps/` - fetched example maps and manifest.
- `legacy_src/` - extracted Quake legacy reference source.

## Build Requirements

- CMake >= 3.23
- C++17 compiler
- Optional:
  - OpenMP (for CPU optimized target)
  - CUDA toolkit + compiler (for GPU target)

## Build Options

The build is baseline-first by default.

- `QUAKECORE_BUILD_BASELINE` (default: `ON`)
- `QUAKECORE_BUILD_CPU_OPT` (default: `OFF`)
- `QUAKECORE_BUILD_GPU_OPT` (default: `OFF`)
- `QUAKECORE_BUILD_BENCH` (default: `OFF`)
- `QUAKECORE_ENABLE_OPENMP` (default: `ON`)
- `QUAKECORE_ENABLE_AVX2` (default: `ON`)

### Default build (baseline only)

```bash
cmake -S . -B build
cmake --build build -j
```

### Baseline + CPU optimized

```bash
cmake -S . -B build -DQUAKECORE_BUILD_CPU_OPT=ON
cmake --build build -j
```

### Baseline + CPU + GPU

```bash
cmake -S . -B build -DQUAKECORE_BUILD_CPU_OPT=ON -DQUAKECORE_BUILD_GPU_OPT=ON
cmake --build build -j
```

### Full build (including unified benchmark)

```bash
cmake -S . -B build \
  -DQUAKECORE_BUILD_CPU_OPT=ON \
  -DQUAKECORE_BUILD_GPU_OPT=ON \
  -DQUAKECORE_BUILD_BENCH=ON
cmake --build build -j
```

## Example Map Fetching

Fetch open benchmark maps on demand:

```bash
bash scripts/fetch_maps.sh
```

Useful variants:

```bash
bash scripts/fetch_maps.sh --dataset uquake1-master
bash scripts/fetch_maps.sh --dataset quake-local-source --quake-source-dir ../Quake
bash scripts/fetch_maps.sh --clean
```

Fetched maps are written to:

- `examples/maps/fetched/<dataset>/`

Manifest:

- `examples/maps/MANIFEST.csv`

## Running Targets

Use any `.bsp` path (for example one under `examples/maps/fetched/...`).

### Baseline

```bash
./build/quakecore_baseline --map /path/to/map.bsp --frames 2000 --seed 7
```

### CPU optimized

```bash
./build/quakecore_cpu_opt --map /path/to/map.bsp --frames 2000 --threads 64 --seed 7
```

### GPU optimized

```bash
./build/quakecore_gpu_opt --map /path/to/map.bsp --frames 2000 --block-size 256 --seed 7
```

### Unified benchmark

```bash
./build/quakecore_bench --map /path/to/map.bsp --frames 2000 --threads 64 --block-size 256 --seed 7 --csv bench.csv
```

## Notes

- GPU targets require CUDA at configure/build time.
- The benchmark path is intended for cross-engine comparison and regression checks.
- This project focuses on traversal/culling benchmarking, not full Quake rendering/gameplay.
