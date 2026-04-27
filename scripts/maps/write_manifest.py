#!/usr/bin/env python3
"""Write CSV manifest (including SHA256) for fetched BSP maps."""

import argparse
import csv
import hashlib
import pathlib


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BSP dataset manifest CSV.")
    parser.add_argument("--maps-root", required=True, help="Root directory containing dataset subdirectories.")
    parser.add_argument("--manifest-path", required=True, help="Output CSV manifest path.")
    args = parser.parse_args()

    maps_root = pathlib.Path(args.maps_root).resolve()
    manifest_path = pathlib.Path(args.manifest_path).resolve()

    rows = []
    for dataset_dir in sorted(path for path in maps_root.iterdir() if path.is_dir()):
        for bsp in sorted(dataset_dir.glob("*.bsp")):
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "map_file": bsp.name,
                    "size_bytes": bsp.stat().st_size,
                    "sha256": file_sha256(bsp),
                    "path": str(bsp),
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["dataset", "map_file", "size_bytes", "sha256", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"maps_total={len(rows)}")


if __name__ == "__main__":
    main()
