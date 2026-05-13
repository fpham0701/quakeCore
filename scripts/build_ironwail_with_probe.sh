#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="$(cat "$HERE/thirdparty/ironwail.commit")"
WORK="$HERE/build/ironwail_src"
DEST="$HERE/build/ironwail"
mkdir -p "$WORK" "$DEST"

if [ ! -d "$WORK/.git" ]; then
  git clone https://github.com/andrei-drexler/ironwail "$WORK"
fi
git -C "$WORK" fetch --all --quiet
git -C "$WORK" checkout "$COMMIT"

# Copy probe sources in place.
cp -r "$HERE/thirdparty/ironwail/frame_probe" "$WORK/Quake/"

# Apply the patch series.
for p in "$HERE/thirdparty/ironwail/patches/"*.patch; do
  echo "Applying $p"
  git -C "$WORK" apply --check "$p"
  git -C "$WORK" apply "$p"
done

# Build. Pass include + lib paths via env so the patched Makefile picks them up.
QUAKECORE_DIR="$HERE" make -C "$WORK/Quake" -j"$(nproc)"

cp "$WORK/Quake/ironwail" "$DEST/ironwail"
echo "Built: $DEST/ironwail"
