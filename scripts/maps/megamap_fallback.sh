#!/usr/bin/env bash
set -euo pipefail

# Fallback when build_megamap.sh cannot build ericw-tools in the local env.
# Downloads a precompiled BSP2 megamap from a Release asset on a fork
# under our control. The asset URL and SHA-256 are pinned here.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/examples/maps/fetched/megamap-bsp2"
mkdir -p "${OUT_DIR}"

# !!! FILL IN BEFORE USE !!!
ASSET_URL="${ASSET_URL:-https://github.com/<your-handle>/quakeCore-megamaps/releases/download/v1/<name>.bsp}"
ASSET_SHA="${ASSET_SHA:-<paste sha256 here>}"
OUT_BSP="${OUT_DIR}/$(basename "${ASSET_URL}")"

if [[ "${ASSET_URL}" == *"<your-handle>"* || "${ASSET_SHA}" == *"<paste"* ]]; then
  echo "error: ASSET_URL and ASSET_SHA are unfilled placeholders" >&2
  echo "  edit ${BASH_SOURCE[0]} or set ASSET_URL / ASSET_SHA in the env" >&2
  exit 1
fi

if [[ ! -f "${OUT_BSP}" ]]; then
  echo "-> downloading ${ASSET_URL}"
  curl -L --retry 3 --fail -o "${OUT_BSP}" "${ASSET_URL}"
fi

actual_sha="$(sha256sum "${OUT_BSP}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${ASSET_SHA}" ]]; then
  echo "error: SHA-256 mismatch on ${OUT_BSP}" >&2
  echo "  expected ${ASSET_SHA}" >&2
  echo "  actual   ${actual_sha}" >&2
  exit 1
fi

magic_hex=$(head -c 4 "${OUT_BSP}" | od -An -tx1 | tr -d ' \n')
echo "BSP2 megamap (fallback) ready:"
echo "  path:  ${OUT_BSP}"
echo "  bytes: $(stat -c %s "${OUT_BSP}")"
echo "  magic: 0x${magic_hex}  (expect 42535032)"
