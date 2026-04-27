#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAP_ROOT="${ROOT_DIR}/examples/maps"
CACHE_DIR="${MAP_ROOT}/.cache"
OUT_DIR="${MAP_ROOT}/fetched"
MANIFEST="${MAP_ROOT}/MANIFEST.csv"
TARGET_DATASET="${TARGET_DATASET:-all}"
FORCE_DOWNLOAD=0
CLEAN_ONLY=0
QUAKE_SOURCE_DIR_DEFAULT="${ROOT_DIR}/../Quake"
QUAKE_SOURCE_DIR="${QUAKE_SOURCE_DIR:-${QUAKE_SOURCE_DIR_DEFAULT}}"

EXTRACT_SCRIPT="${ROOT_DIR}/scripts/maps/extract_bsp_assets.py"
MANIFEST_SCRIPT="${ROOT_DIR}/scripts/maps/write_manifest.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      TARGET_DATASET="${2:-}"
      shift 2
      ;;
    --force)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --clean)
      CLEAN_ONLY=1
      shift
      ;;
    --quake-source-dir)
      QUAKE_SOURCE_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: fetch_maps.sh [--dataset <name>|all] [--force] [--clean] [--quake-source-dir <path>]

  --dataset <name>  Fetch only one dataset (default: all)
  --force           Re-download archives and re-extract dataset(s)
  --clean           Remove generated fetched maps/cache/manifest and exit
  --quake-source-dir <path>
                    Use a specific Quake source checkout for local extraction
                    (default: ../Quake relative to quakeCore)

Datasets:
  quake_map_source-master
  uquake1-master
  quake-local-source
EOF
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${CACHE_DIR}" "${OUT_DIR}"

if [[ ${CLEAN_ONLY} -eq 1 ]]; then
  echo "Cleaning generated map artifacts under ${MAP_ROOT}"
  rm -rf "${CACHE_DIR}" "${OUT_DIR}" "${MANIFEST}" "${MAP_ROOT}/bench_smoke.csv"
  mkdir -p "${CACHE_DIR}" "${OUT_DIR}"
  echo "Clean complete."
  exit 0
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd tar
need_cmd python3

if [[ ! -f "${EXTRACT_SCRIPT}" ]]; then
  echo "error: extractor script not found: ${EXTRACT_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST_SCRIPT}" ]]; then
  echo "error: manifest script not found: ${MANIFEST_SCRIPT}" >&2
  exit 1
fi

declare -a DATASETS=(
  "quake_map_source-master|https://codeload.github.com/fzwoch/quake_map_source/tar.gz/refs/heads/master|Quake map source package with free texture replacements"
  "uquake1-master|https://codeload.github.com/mikezila/uQuake1/tar.gz/refs/heads/master|Public-domain Quake BSP loading project with bundled sample BSPs"
  "quake-local-source|local://quake-source|Extract BSPs from local @Quake source tree (BSP/PAK scan)"
)

echo "Fetching open Quake-derived map datasets into ${OUT_DIR}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r name url note <<< "${entry}"
  if [[ "${TARGET_DATASET}" != "all" && "${TARGET_DATASET}" != "${name}" ]]; then
    continue
  fi

  archive="${CACHE_DIR}/${name}.tar.gz"
  extract_dir="${CACHE_DIR}/${name}"
  dataset_out="${OUT_DIR}/${name}"

  if [[ "${name}" == "quake-local-source" ]]; then
    if [[ ! -d "${QUAKE_SOURCE_DIR}" ]]; then
      echo "-> skipping quake-local-source (directory not found: ${QUAKE_SOURCE_DIR})"
      continue
    fi
    rm -rf "${extract_dir}"
    mkdir -p "${extract_dir}"
    cp -R "${QUAKE_SOURCE_DIR}/." "${extract_dir}/"
  else
    if [[ ${FORCE_DOWNLOAD} -eq 0 && -s "${archive}" && -d "${dataset_out}" ]]; then
      echo "-> skipping ${name} (already fetched; use --force to refresh)"
      continue
    fi

    echo "-> downloading ${name}"
    curl -L --retry 3 --fail -o "${archive}" "${url}"

    rm -rf "${extract_dir}"
    mkdir -p "${extract_dir}"
    tar -xzf "${archive}" -C "${extract_dir}" --strip-components=1
  fi

  rm -rf "${dataset_out}"
  mkdir -p "${dataset_out}"

  echo "-> extracting BSP files from ${name}"
  copied_count="$(python3 "${EXTRACT_SCRIPT}" --source-dir "${extract_dir}" --output-dir "${dataset_out}")"
  echo "-> ${name}: ${copied_count} bsp files copied"
  echo "   note: ${note}"
done

echo "Writing manifest ${MANIFEST}"
python3 "${MANIFEST_SCRIPT}" --maps-root "${OUT_DIR}" --manifest-path "${MANIFEST}"

echo "Done. Example maps available under: ${OUT_DIR}"
