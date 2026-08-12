"""
clean_data.py
--------------
Part A cleaning step: scans your processed data (train/val/test) and:
  1. Removes corrupt / unreadable images (can't be opened)
  2. Removes exact duplicate images (same content, found using a hash)
  3. Flags images that are unusually small or broken (all-black/all-white)
  4. Prints a cleaning report you can paste directly into your report.pdf

Run AFTER preprocess.py, from your project root:
    python clean_data.py

It modifies data/processed/ in place (deletes bad files) and prints
a summary you should copy into your report under "cleaning steps".
"""

import os
import hashlib
import numpy as np
from PIL import Image

PROCESSED_DIR = os.path.join("data", "processed")
SPLITS = ["train", "val", "test"]

MIN_STD_DEV = 5.0   # if pixel std-dev is below this, image is near-blank (broken)


def file_hash(path):
    """Compute a hash of the raw file bytes, to detect exact duplicates."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def is_corrupt_or_broken(path):
    """Try to open the image and check it isn't blank/broken.
    Returns (is_bad, reason)."""
    try:
        img = Image.open(path)
        img.verify()  # checks file integrity without fully decoding
    except Exception as e:
        return True, f"unreadable ({e})"

    try:
        # re-open because verify() closes the file handle
        img = Image.open(path).convert("L")  # grayscale for a quick check
        arr = np.array(img)
        if arr.std() < MIN_STD_DEV:
            return True, "near-blank / broken (very low pixel variation)"
    except Exception as e:
        return True, f"could not analyze ({e})"

    return False, None


def main():
    if not os.path.isdir(PROCESSED_DIR):
        print(f"'{PROCESSED_DIR}' not found. Run preprocess.py first.")
        return

    seen_hashes = {}   # hash -> path of first occurrence (kept)
    report = {"corrupt_removed": [], "duplicates_removed": [], "kept": 0}

    for split in SPLITS:
        split_dir = os.path.join(PROCESSED_DIR, split)
        if not os.path.isdir(split_dir):
            continue

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)

                # 1. Check corrupt / broken
                bad, reason = is_corrupt_or_broken(fpath)
                if bad:
                    report["corrupt_removed"].append((fpath, reason))
                    os.remove(fpath)
                    continue

                # 2. Check duplicate (exact file content match)
                h = file_hash(fpath)
                if h in seen_hashes:
                    report["duplicates_removed"].append(
                        (fpath, "duplicate of " + seen_hashes[h])
                    )
                    os.remove(fpath)
                    continue

                seen_hashes[h] = fpath
                report["kept"] += 1

    # ---- Print report ----
    print("=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)

    print(f"\nCorrupt/broken images removed: {len(report['corrupt_removed'])}")
    for path, reason in report["corrupt_removed"][:20]:
        print(f"  - {path}  [{reason}]")
    if len(report["corrupt_removed"]) > 20:
        print(f"  ... and {len(report['corrupt_removed']) - 20} more")

    print(f"\nDuplicate images removed: {len(report['duplicates_removed'])}")
    for path, reason in report["duplicates_removed"][:20]:
        print(f"  - {path}  [{reason}]")
    if len(report["duplicates_removed"]) > 20:
        print(f"  ... and {len(report['duplicates_removed']) - 20} more")

    print(f"\nTotal images kept (clean): {report['kept']}")

    print("\nPer-class counts after cleaning:")
    for split in SPLITS:
        split_dir = os.path.join(PROCESSED_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"\n  [{split}]")
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                print(f"    {class_name:12s}: {count}")

    print("\nDone. Copy the counts above into your report.pdf under 'Data Cleaning'.")


if __name__ == "__main__":
    main()
