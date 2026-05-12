#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="${ROOT_DIR}/build/quakecore_bench"
MAPS_DIR="${ROOT_DIR}/examples/maps/fetched/quake_map_source-master"
FRAMES="${FRAMES:-500}"

if [[ ! -x "${BENCH}" ]]; then
  echo "error: build first: cmake --build ${ROOT_DIR}/build -j" >&2
  exit 1
fi
if [[ ! -d "${MAPS_DIR}" ]]; then
  echo "error: fetch maps first: bash ${ROOT_DIR}/scripts/fetch_maps.sh --dataset quake_map_source-master" >&2
  exit 1
fi

fail=0
total=0
for bsp in "${MAPS_DIR}"/*.bsp; do
  total=$((total + 1))
  name="$(basename "${bsp}")"
  if ! out="$("${BENCH}" --map "${bsp}" --frames "${FRAMES}" --threads 4 --block-size 256 --seed 7 2>&1)"; then
    echo "FAIL ${name}: bench exited non-zero"
    echo "${out}" | tail -5 | sed 's/^/    /'
    fail=$((fail + 1))
    continue
  fi
  if ! grep -q 'correctness_cpu_opt=PASS' <<<"${out}"; then
    echo "FAIL ${name}: cpu_opt parity"
    fail=$((fail + 1))
    continue
  fi
  if ! grep -q 'correctness_gpu_opt=PASS' <<<"${out}"; then
    echo "FAIL ${name}: gpu_opt parity"
    fail=$((fail + 1))
    continue
  fi
  echo "PASS ${name}"
done

echo
echo "${total} maps, ${fail} failures"
exit "${fail}"
