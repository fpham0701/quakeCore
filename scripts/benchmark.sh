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

MAP="${1:?usage: sbatch $0 <map.bsp>}"
BIN="${BIN:-./build/quakecore_bench}"
MASTER="benchmark.out"

[[ -x "$BIN" ]] || { echo "missing binary: $BIN (build with -DQUAKECORE_BUILD_BENCH=ON)" >&2; exit 1; }
[[ -f "$MAP" ]] || { echo "missing map: $MAP" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

RUN_CSV="$(mktemp -t quakecore_bench.XXXXXX.csv)"
trap 'rm -f "$RUN_CSV"' EXIT
MAP_NAME="$(basename "$MAP")"

echo "sweep,threads,block_size,frames,seed,map,engine,frames_run,time_s,fps,visited_nodes,visited_leafs,culled_nodes,accepted_leafs" > "$MASTER"

run_one() {
	local sweep="$1" threads="$2" block="$3" frames="$4" seed="$5"
	printf '[%-10s] threads=%-3s block=%-4s frames=%-5s seed=%s\n' "$sweep" "$threads" "$block" "$frames" "$seed"
	srun --ntasks=1 --cpus-per-task="$threads" --gpus-per-task=1 \
		"$BIN" --map "$MAP" --frames "$frames" --threads "$threads" \
		       --block-size "$block" --seed "$seed" --csv "$RUN_CSV" >/dev/null
	tail -n +2 "$RUN_CSV" \
		| awk -v s="$sweep" -v t="$threads" -v b="$block" -v f="$frames" -v sd="$seed" -v m="$MAP_NAME" \
		'BEGIN{OFS=","} { print s,t,b,f,sd,m,$0 }' \
		>> "$MASTER"
}

SEEDS=(7 13 42)
THREADS_DEFAULT="${SLURM_CPUS_PER_TASK:-16}"
FRAMES_DEFAULT=2000

# 1) Block-size sweep
for b in 32 64 128 256 512 1024; do
	for s in "${SEEDS[@]}"; do
		run_one "block_size" "$THREADS_DEFAULT" "$b" "$FRAMES_DEFAULT" "$s"
	done
done

# 2) Workload scaling (frames)
for f in 500 1000 2000 5000 10000; do
	for s in "${SEEDS[@]}"; do
		run_one "frames" "$THREADS_DEFAULT" 512 "$f" "$s"
	done
done

echo
echo "Sweep complete: $MASTER ($(($(wc -l < "$MASTER") - 1)) rows)"
