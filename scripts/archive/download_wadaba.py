"""Download WaDaBa (Plastic Waste Database of Images) from the official website.

The dataset is hosted at http://wadaba.pcz.pl and contains 4000 images
of plastic waste items across multiple categories (PET, HDPE, PP, PS, etc).
It is split into 20 zip archives (Set 1 through Set 20).
"""

import os
import sys
import zipfile
import time
import requests
import urllib3

# Suppress SSL warnings since wadaba.pcz.pl has cert issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external_datasets", "wadaba")
ZIP_DIR = os.path.join(OUT_DIR, "_zips")

# All 20 sets from http://wadaba.pcz.pl/#download
SETS = [
    "WaDaBa_1-5",
    "WaDaBa_6-10",
    "WaDaBa_11-15",
    "WaDaBa_16-20",
    "WaDaBa_21-25",
    "WaDaBa_26-30",
    "WaDaBa_31-35",
    "WaDaBa_36-40",
    "WaDaBa_41-45",
    "WaDaBa_46-50",
    "WaDaBa_51-55",
    "WaDaBa_56-60",
    "WaDaBa_61-65",
    "WaDaBa_66-70",
    "WaDaBa_71-75",
    "WaDaBa_76-80",
    "WaDaBa_81-85",
    "WaDaBa_86-90",
    "WaDaBa_91-95",
    "WaDaBa_96-100",
]

BASE_URL = "http://wadaba.pcz.pl/data/{}.zip"


def download_file(url, dest, retries=3):
    """Download a file with retry logic and progress reporting."""
    for attempt in range(retries):
        try:
            print(f"  Downloading: {url}")
            r = requests.get(url, stream=True, timeout=120, verify=False)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 / total
                        print(f"\r  Progress: {downloaded/(1024*1024):.1f} MB / {total/(1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
            print()
            return True
        except Exception as e:
            print(f"\n  Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)

    total_images = 0
    failed = []

    for i, set_name in enumerate(SETS, 1):
        zip_name = f"{set_name}.zip"
        zip_path = os.path.join(ZIP_DIR, zip_name)
        url = BASE_URL.format(set_name)

        print(f"\n[{i}/{len(SETS)}] {set_name}")

        # Skip if already downloaded and extracted
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000:
            print(f"  ZIP already exists ({os.path.getsize(zip_path)/(1024*1024):.1f} MB), skipping download...")
        else:
            if not download_file(url, zip_path):
                print(f"  FAILED to download {set_name}")
                failed.append(set_name)
                continue

        # Extract
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                img_count = len([m for m in members if m.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"  Extracting {len(members)} files ({img_count} images)...")
                zf.extractall(OUT_DIR)
                total_images += img_count
        except zipfile.BadZipFile:
            print(f"  ERROR: Bad zip file for {set_name}. Deleting and retrying...")
            os.remove(zip_path)
            if download_file(url, zip_path):
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        members = zf.namelist()
                        img_count = len([m for m in members if m.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                        zf.extractall(OUT_DIR)
                        total_images += img_count
                except Exception as e2:
                    print(f"  FAILED again: {e2}")
                    failed.append(set_name)
            else:
                failed.append(set_name)

    # Summary
    print("\n" + "=" * 60)
    print(f"WaDaBa download complete!")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Total images extracted: {total_images}")
    if failed:
        print(f"  FAILED sets: {', '.join(failed)}")
    else:
        print(f"  All 20 sets downloaded successfully!")

    # Count actual files on disk
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    actual_count = 0
    for root, dirs, files in os.walk(OUT_DIR):
        if "_zips" in root:
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in img_exts:
                actual_count += 1
    print(f"  Total image files on disk: {actual_count}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
