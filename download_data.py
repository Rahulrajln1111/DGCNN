#!/usr/bin/env python3
"""
Download ModelNet10 dataset.

Downloads and extracts ModelNet10 .off files to ./data/ModelNet10/raw/
Run this on A100 before training.
"""

import os
import zipfile
import urllib.request

URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
DATA_DIR = "./data/ModelNet10"
RAW_DIR = os.path.join(DATA_DIR, "raw")
ZIP_PATH = os.path.join(DATA_DIR, "ModelNet10.zip")


def download():
    if os.path.exists(RAW_DIR) and len(os.listdir(RAW_DIR)) >= 10:
        print(f"[Data] ModelNet10 already exists at {RAW_DIR}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"[Data] Downloading ModelNet10 from {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print(f"[Data] Downloaded to {ZIP_PATH}")

    print("[Data] Extracting ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    # Move extracted ModelNet10/ contents into raw/
    extracted = os.path.join(DATA_DIR, "ModelNet10")
    if os.path.exists(extracted) and not os.path.exists(RAW_DIR):
        os.rename(extracted, RAW_DIR)
    elif os.path.exists(extracted) and os.path.exists(RAW_DIR):
        # Merge
        import shutil
        for item in os.listdir(extracted):
            src = os.path.join(extracted, item)
            dst = os.path.join(RAW_DIR, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(extracted)

    # Clean up zip
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"[Data] ModelNet10 ready at {RAW_DIR}")
    classes = sorted([d for d in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, d))])
    print(f"[Data] Classes: {classes}")


if __name__ == "__main__":
    download()
