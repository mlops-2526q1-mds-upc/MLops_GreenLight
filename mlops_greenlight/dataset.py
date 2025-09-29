import requests
from tqdm import tqdm
import zipfile
import os
import hashlib

# Output directory for extracted data
OUTPUT_DIR = os.path.join("data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "https://zenodo.org/records/12706046/files/dataset_train_rgb.zip?download=1": {
        "filename": "dataset_train_rgb.zip",
        "md5": "cf0ac345bcd9793c77ec1b6ce3b6fa32",
    },
    "https://zenodo.org/records/12706046/files/dataset_test_rgb.zip?download=1": {
        "filename": "dataset_test_rgb.zip",
        "md5": "ecea5fae3532c32608def305e66fdf8f",
    },
}

def md5sum(filepath, chunk_size=8192):
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url, output_path, chunk_size=1024*1024):
    """Download a file from a URL"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc=f"Extracting {os.path.basename(zip_path)}"):
            zip_ref.extract(member, extract_to)

if __name__ == "__main__":
    for url, meta in FILES.items():
        filename = meta["filename"]
        expected_md5 = meta["md5"]
        file_path = os.path.join(OUTPUT_DIR, filename)

        # Check if file exists and matches checksum
        if os.path.exists(file_path):
            print(f"Found {filename}, verifying checksum...")
            if md5sum(file_path) == expected_md5:
                print(f"{filename} already downloaded and verified, skipping.")
            else:
                print(f"WARNING: {filename} exists but checksum mismatch, re-downloading...")
                download_file(url, file_path)
        else:
            print(f"\nDownloading {filename}...")
            download_file(url, file_path)

        print(f"\nExtracting {filename}...")
        extract_zip(file_path, OUTPUT_DIR)

    print("\nAll files downloaded, verified, and extracted to ./data/raw/")

