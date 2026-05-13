#!/usr/bin/env bash
set -euo pipefail

cd "$SCRATCH/quakeCore"

cmake -S . -B build \
	-DQUAKECORE_BUILD_CPU_OPT=ON \
	-DQUAKECORE_BUILD_GPU_OPT=ON \
	-DQUAKECORE_BUILD_BENCH=ON
cmake --build build -j

mkdir -p examples/maps/.cache/community_packs
for p in honey unforgiven mapjam6 func_mapjam9; do
	curl -L --fail \
		-o examples/maps/.cache/community_packs/${p}.zip \
		https://www.quaddicted.com/filebase/${p}.zip
	mkdir -p examples/maps/fetched/community-${p}
	unzip -j -o examples/maps/.cache/community_packs/${p}.zip '*maps/*.bsp' \
		-d examples/maps/fetched/community-${p}/
done


