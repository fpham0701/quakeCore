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

cd $SCRATCH/quakeCore
OUT=validation_results/perlmutter_$SLURM_JOB_ID
mkdir -p $OUT

for f in examples/maps/fetched/community-*/*.bsp; do
	name=$(basename "$f" .bsp)
	pack=$(basename "$(dirname "$f")")
	./build/quakecore_bench --map "$f" --frames 2000 --threads 32 \
		--block-size 256 --seed 7 \
		--csv $OUT/${pack}_${name}.csv 2>&1 | tee -a $OUT/bench.log
done

# heavier stress test
for f in examples/maps/fetched/community-unforgiven/{unf1,unf2,unf3}.bsp \
	examples/maps/fetched/community-honey/start.bsp; do
	name=$(basename "$f" .bsp)
	./build/quakecore_bench --map "$f" --frames 10000 --threads 32 \
		--block-size 256 --seed 7 \
		--csv $OUT/heavy_${name}_f10000.csv 2>&1 | tee -a $OUT/heavy.log
done

# parity bug check for unf3
for seed in 7 23; do
	./build/quakecore_bench --map examples/maps/fetched/community-unforgiven/unf3.bsp \
		--frames 10000 --threads 32 --block-size 256 --seed $seed \
		--csv $OUT/parity_unf_seed${seed}.csv 2>&1 | tee -a $OUT/parity.log
done
