#!/usr/bin/env python3
"""
convert.py
-----------
Utility script to convert PNG images to JPEG format at multiple compression levels
and copy the resulting directory structure to a processed dataset folder.
"""

import os
import shutil
from PIL import Image

# --- CONFIGURATION ---
SRC_DIR = "./data/raw/"
DST_DIR = "./data/processed/"
JPG_QUALITIES = [25, 65, 85]  # List of JPEG qualities


def copy_dir_with_png_to_jpg(src_dir: str, dst_dir: str, jpg_quality: int) -> None:
    """
    Recursively copies a directory, converting PNG files to JPEG with a given quality.

    Args:
        src_dir (str): Source directory containing the original files.
        dst_dir (str): Destination base directory for processed files.
        jpg_quality (int): JPEG quality level (0–100).
    """
    dst_root = os.path.join(dst_dir, str(jpg_quality))
    os.makedirs(dst_root, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_root, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for d in dirs:
            os.makedirs(os.path.join(target_root, d), exist_ok=True)

        for fname in files:
            if fname.lower().endswith(".zip"):
                print(f"Skipped {fname}")
                continue

            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target_root, fname)

            if fname.lower().endswith(".png"):
                try:
                    with Image.open(src_path) as img:
                        img.convert("RGB").save(
                            os.path.splitext(dst_path)[0] + ".jpg",
                            "JPEG",
                            quality=jpg_quality,
                        )
                    print(f"[Q{jpg_quality}] Converted {src_path} -> {dst_path}")
                except (OSError, ValueError) as e:
                    print(f"[Q{jpg_quality}] Failed to convert {src_path}: {e}")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"[Q{jpg_quality}] Copied {src_path} -> {dst_path}")


if __name__ == "__main__":
    for quality in JPG_QUALITIES:
        print(f"\n=== Processing quality {quality} ===")
        copy_dir_with_png_to_jpg(SRC_DIR, DST_DIR, jpg_quality=quality)
