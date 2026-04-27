# Example BSP Maps (On-Demand)

This project intentionally does not vendor large binary map files in git.  
Use the fetch script to download a broad, freely redistributable benchmark set.

## Fetch

```bash
cd quakeCore
bash scripts/fetch_maps.sh
```

Optional:

```bash
# Fetch only one dataset
bash scripts/fetch_maps.sh --dataset uquake1-master

# Force refresh existing downloads
bash scripts/fetch_maps.sh --force

# Clean generated map cache/output/manifest
bash scripts/fetch_maps.sh --clean

# Extract from adjacent Quake source tree explicitly
bash scripts/fetch_maps.sh --dataset quake-local-source --quake-source-dir ../Quake
```

This downloads open Quake-derived datasets and extracts all discovered `*.bsp` files into:

- `examples/maps/fetched/<dataset>/`

It also generates:

- `examples/maps/MANIFEST.csv`

The manifest includes dataset, filename, size, and SHA-256 checksum for reproducibility.

Current dataset set:
- `quake_map_source-master`
- `uquake1-master`
- `quake-local-source` (local extraction from your `@Quake` checkout when available)

## Use With Engines

Pick a map from `examples/maps/fetched/...` and run:

```bash
./build/quakecore_baseline --map examples/maps/fetched/<dataset>/<map>.bsp --frames 2000
./build/quakecore_cpu_opt --map examples/maps/fetched/<dataset>/<map>.bsp --frames 2000 --threads 64
./build/quakecore_gpu_opt --map examples/maps/fetched/<dataset>/<map>.bsp --frames 2000 --block-size 256
./build/quakecore_bench --map examples/maps/fetched/<dataset>/<map>.bsp --frames 2000 --threads 64 --block-size 256 --csv bench.csv
```
