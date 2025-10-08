#!/usr/bin/env python3

import os
import shutil
from PIL import Image

# --- CONFIGURATION ---
SRC_DIR = "./data/raw/"
DST_DIR = "./data/processed/"
JPG_QUALITIES = [25, 65, 85]  # List of JPEG qualities

def copy_dir_with_png_to_jpg(src_dir, dst_dir, jpg_quality):
    # Create destination folder with compression level as subfolder
    dst_dir_with_quality = os.path.join(dst_dir, str(jpg_quality))
    if not os.path.exists(dst_dir_with_quality):
        os.makedirs(dst_dir_with_quality)
    
    for root, dirs, files in os.walk(src_dir):
        # Relative path from source
        rel_path = os.path.relpath(root, src_dir)
        dest_root = os.path.join(dst_dir_with_quality, rel_path)
        
        # Make sure destination subdirectories exist
        for d in dirs:
            os.makedirs(os.path.join(dest_root, d), exist_ok=True)
        
        for file in files:
            if file.lower().endswith(".zip"):
                print(f"Skipped {file}")
                continue  # Skip zip files
            
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            
            if file.lower().endswith(".png"):
                # Convert PNG to JPG
                dest_file = os.path.splitext(dest_file)[0] + ".jpg"
                try:
                    with Image.open(src_file) as img:
                        rgb_img = img.convert("RGB")  # Remove alpha
                        rgb_img.save(dest_file, "JPEG", quality=jpg_quality)
                    print(f"[Q{jpg_quality}] Converted {src_file} -> {dest_file}")
                except Exception as e:
                    print(f"[Q{jpg_quality}] Failed to convert {src_file}: {e}")
            else:
                # Copy other files as-is
                shutil.copy2(src_file, dest_file)
                print(f"[Q{jpg_quality}] Copied {src_file} -> {dest_file}")

if __name__ == "__main__":
    for quality in JPG_QUALITIES:
        print(f"\n=== Processing quality {quality} ===")
        copy_dir_with_png_to_jpg(SRC_DIR, DST_DIR, jpg_quality=quality)
