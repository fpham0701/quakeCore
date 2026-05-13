#!/bin/bash
#SBATCH --account=m4341_g
#SBATCH --job-name=quakecore_bench
#SBATCH --time=00:20:00
#SBATCH --constraint="gpu&hbm40g"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --qos=regular
#SBATCH --output=benchmark.log
#SBATCH --error=benchmark.err

set -euo pipefail

cd "$SCRATCH/quakeCore"
OUT=validation_results/perlmutter_$SLURM_JOB_ID
mkdir -p $OUT

# Supported on-disk formats: v29 (0x0000001D LE) and BSP2 (ASCII "BSP2").
# BSP29a/2PSB and Half-Life v30 are rejected by the parser, so skip them
# here to keep the batch job from aborting under `set -e`.
is_supported_bsp() {
	local magic
	magic=$(head -c 4 -- "$1" 2>/dev/null | od -An -tx1 | tr -d ' \n') || return 1
	[[ "$magic" == "1d000000" || "$magic" == "42535032" ]]
}

for f in examples/maps/fetched/community-*/*.bsp; do
	is_supported_bsp "$f" || continue
	name=$(basename "$f" .bsp)
	pack=$(basename "$(dirname "$f")")
	./build/quakecore_bench --map "$f" --frames 2000 --threads 16 \
		--block-size 256 --seed 7 \
		--csv $OUT/${pack}_${name}.csv 2>&1 | tee -a $OUT/bench.log
done

# heavier stress test
for f in examples/maps/fetched/community-unforgiven/{unf1,unf2,unf3}.bsp \
	examples/maps/fetched/community-honey/start.bsp; do
	is_supported_bsp "$f" || continue
	name=$(basename "$f" .bsp)
	./build/quakecore_bench --map "$f" --frames 10000 --threads 16 \
		--block-size 256 --seed 7 \
		--csv $OUT/heavy_${name}_f10000.csv 2>&1 | tee -a $OUT/heavy.log
done

# parity bug check for unf3
for seed in 7 23; do
	./build/quakecore_bench --map examples/maps/fetched/community-unforgiven/unf3.bsp \
		--frames 10000 --threads 16 --block-size 256 --seed $seed \
		--csv $OUT/parity_unf_seed${seed}.csv 2>&1 | tee -a $OUT/parity.log
done
