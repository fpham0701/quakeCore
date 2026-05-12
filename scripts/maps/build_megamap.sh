#!/usr/bin/env bash
set -euo pipefail

# Builds a BSP2 megamap from an open .map source by cloning ericw-tools at a
# pinned tag, building qbsp, and compiling the .map.
#
# Output: examples/maps/fetched/megamap-bsp2/<name>.bsp
#
# Override via env:
#   ERICW_TAG     tag/commit to check out (default v0.18.1)
#   FORCE=1       rebuild ericw-tools and recompile even if outputs exist
#   MAP_SOURCE    path or URL of the .map to compile (default: see below)
#   MAP_NAME      output filename stem (default: derived from MAP_SOURCE)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAP_ROOT="${ROOT_DIR}/examples/maps"
CACHE_DIR="${MAP_ROOT}/.cache"
ERICW_DIR="${CACHE_DIR}/ericw-tools"
ERICW_BUILD="${ERICW_DIR}/build"
OUT_DIR="${MAP_ROOT}/fetched/megamap-bsp2"
ERICW_TAG="${ERICW_TAG:-v0.18.1}"
ERICW_REPO="${ERICW_REPO:-https://github.com/ericwa/ericw-tools.git}"

# Default source: jam2_sock.map from func_mapjam2 release. Engineer must
# confirm the URL resolves and the license still permits redistribution
# before relying on it. The README at the candidate's source should be
# checked once and the license recorded in the commit message below.
MAP_SOURCE_DEFAULT_URL="https://www.quaddicted.com/reviews/files/func_mapjam2.zip"
MAP_SOURCE_DEFAULT_INSIDE_ZIP="func_mapjam2/maps/jam2_sock.map"

MAP_SOURCE="${MAP_SOURCE:-}"
MAP_NAME="${MAP_NAME:-}"

mkdir -p "${CACHE_DIR}" "${OUT_DIR}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: required command not found: $1" >&2; exit 1; }
}
need_cmd git
need_cmd cmake
need_cmd make
need_cmd curl
need_cmd unzip

# 1. Clone or update ericw-tools at the pinned tag.
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -rf "${ERICW_DIR}"
fi

if [[ ! -d "${ERICW_DIR}/.git" ]]; then
  echo "-> cloning ${ERICW_REPO} @ ${ERICW_TAG}"
  git clone --depth 1 --branch "${ERICW_TAG}" "${ERICW_REPO}" "${ERICW_DIR}"
else
  echo "-> ericw-tools clone present at ${ERICW_DIR}"
fi

# 2. Configure + build qbsp.
QBSP_BIN="${ERICW_BUILD}/qbsp/qbsp"
if [[ ! -x "${QBSP_BIN}" || "${FORCE:-0}" == "1" ]]; then
  echo "-> building ericw-tools/qbsp"
  cmake -S "${ERICW_DIR}" -B "${ERICW_BUILD}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${ERICW_BUILD}" --target qbsp -j
fi

if [[ ! -x "${QBSP_BIN}" ]]; then
  echo "error: qbsp build did not produce ${QBSP_BIN}" >&2
  exit 1
fi

echo "-> qbsp ready: ${QBSP_BIN}"

# 3. Acquire the .map source.
if [[ -z "${MAP_SOURCE}" ]]; then
  ZIP_PATH="${CACHE_DIR}/$(basename "${MAP_SOURCE_DEFAULT_URL}")"
  UNPACK_DIR="${CACHE_DIR}/$(basename "${MAP_SOURCE_DEFAULT_URL}" .zip)"
  if [[ ! -f "${ZIP_PATH}" || "${FORCE:-0}" == "1" ]]; then
    echo "-> downloading ${MAP_SOURCE_DEFAULT_URL}"
    curl -L --retry 3 --fail -o "${ZIP_PATH}" "${MAP_SOURCE_DEFAULT_URL}"
  fi
  if [[ ! -d "${UNPACK_DIR}" || "${FORCE:-0}" == "1" ]]; then
    rm -rf "${UNPACK_DIR}"
    mkdir -p "${UNPACK_DIR}"
    unzip -q -o "${ZIP_PATH}" -d "${UNPACK_DIR}"
  fi
  MAP_SOURCE="${UNPACK_DIR}/${MAP_SOURCE_DEFAULT_INSIDE_ZIP}"
fi

if [[ ! -f "${MAP_SOURCE}" ]]; then
  echo "error: .map source not found: ${MAP_SOURCE}" >&2
  echo "  set MAP_SOURCE to an absolute path of a valid .map" >&2
  exit 1
fi

if [[ -z "${MAP_NAME}" ]]; then
  MAP_NAME="$(basename "${MAP_SOURCE}" .map)"
fi

# 4. Compile to BSP2.
OUT_BSP="${OUT_DIR}/${MAP_NAME}.bsp"
if [[ ! -f "${OUT_BSP}" || "${FORCE:-0}" == "1" ]]; then
  WORK_DIR="${CACHE_DIR}/qbsp-work-${MAP_NAME}"
  rm -rf "${WORK_DIR}"
  mkdir -p "${WORK_DIR}"
  cp "${MAP_SOURCE}" "${WORK_DIR}/${MAP_NAME}.map"
  echo "-> compiling ${MAP_NAME}.map -> ${MAP_NAME}.bsp (BSP2)"
  "${QBSP_BIN}" -bsp2 -nopercent "${WORK_DIR}/${MAP_NAME}.map" "${WORK_DIR}/${MAP_NAME}.bsp"
  mv "${WORK_DIR}/${MAP_NAME}.bsp" "${OUT_BSP}"
fi

if [[ ! -f "${OUT_BSP}" ]]; then
  echo "error: qbsp did not produce ${OUT_BSP}" >&2
  exit 1
fi

# 5. Print quick stats: file size and BSP2 magic confirmation.
size_bytes=$(stat -c %s "${OUT_BSP}")
magic_hex=$(head -c 4 "${OUT_BSP}" | od -An -tx1 | tr -d ' \n')
echo
echo "BSP2 megamap ready:"
echo "  path:  ${OUT_BSP}"
echo "  bytes: ${size_bytes}"
echo "  magic: 0x${magic_hex}  (expect 42535032 for BSP2)"
