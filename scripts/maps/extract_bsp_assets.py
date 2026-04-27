#!/usr/bin/env python3
"""Extract BSP files from a source tree into a flat output directory.

This script scans for:
1) direct *.bsp files
2) BSP entries inside Quake *.pak files
"""

import argparse
import pathlib
import shutil
import struct


def unique_out_path(out_dir: pathlib.Path, candidate_name: str) -> pathlib.Path:
    out_file = out_dir / candidate_name
    stem = out_file.stem
    suffix = 0
    while out_file.exists():
        suffix += 1
        out_file = out_dir / f"{stem}_{suffix}.bsp"
    return out_file


def copy_direct_bsp_files(src_root: pathlib.Path, out_dir: pathlib.Path) -> int:
    count = 0
    for bsp_path in src_root.rglob("*.bsp"):
        rel_parts = [part for part in bsp_path.parts if part != ".git"]
        rel_name = "__".join(rel_parts[-3:]) if len(rel_parts) >= 3 else "__".join(rel_parts)
        out_file = unique_out_path(out_dir, rel_name)
        shutil.copy2(bsp_path, out_file)
        count += 1
    return count


def extract_bsp_from_pak_files(src_root: pathlib.Path, out_dir: pathlib.Path) -> int:
    count = 0
    for pak_path in src_root.rglob("*.pak"):
        with pak_path.open("rb") as fp:
            header = fp.read(12)
            if len(header) != 12:
                continue
            magic, dir_offset, dir_size = struct.unpack("<4sii", header)
            if magic != b"PACK" or dir_offset < 0 or dir_size < 0 or (dir_size % 64) != 0:
                continue

            fp.seek(dir_offset)
            directory = fp.read(dir_size)
            if len(directory) != dir_size:
                continue

            entries = dir_size // 64
            for i in range(entries):
                base = i * 64
                name_raw = directory[base : base + 56]
                file_pos, file_len = struct.unpack("<ii", directory[base + 56 : base + 64])
                name = name_raw.split(b"\x00", 1)[0].decode("latin-1", errors="ignore")
                if not name.lower().endswith(".bsp"):
                    continue
                if file_pos < 0 or file_len <= 0:
                    continue

                fp.seek(file_pos)
                data = fp.read(file_len)
                if len(data) != file_len:
                    continue

                safe_name = name.replace("/", "__")
                out_file = unique_out_path(out_dir, f"{pak_path.stem}__{safe_name}")
                out_file.write_bytes(data)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BSP maps from directory and PAK files.")
    parser.add_argument("--source-dir", required=True, help="Source directory to scan.")
    parser.add_argument("--output-dir", required=True, help="Directory to write extracted BSP files.")
    args = parser.parse_args()

    src_root = pathlib.Path(args.source_dir).resolve()
    out_dir = pathlib.Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_direct_bsp_files(src_root, out_dir)
    copied += extract_bsp_from_pak_files(src_root, out_dir)
    print(copied)


if __name__ == "__main__":
    main()
