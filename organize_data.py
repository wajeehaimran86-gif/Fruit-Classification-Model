"""
organize_data.py
------------------
Copies images from the two cloned Fruits-360 sparse-checkout repos
into a clean data/raw/<class_name>/ structure, ready for preprocess.py.

Run this from your project root:
    C:\\Users\\HP\\Desktop\\Fruit_SVM_project>

Expected folders already present (from your git sparse-checkout steps):
    fruits-360-original-size/Training/Apple Braeburn 1/
    fruits-360-original-size/Training/Banana 3/
    fruits-360-original-size/Training/Banana 4/
    fruits-360-original-size/Training/Orange 2/
    fruits-360-original-size/Training/Orange 3/
    fruits-360-original-size/Training/Orange 4/
    fruits-360-original-size/Training/Grape 1/
    fruits-360-original-size/Training/Strawberry 2/
    fruits-360-original-size/Training/Strawberry 3/
    fruits-360-mango/Training/Mango 1/
    fruits-360-mango/Training/Mango Red 1/

Output:
    data/raw/apple/
    data/raw/banana/
    data/raw/orange/
    data/raw/grapes/
    data/raw/strawberry/
    data/raw/mango/

Usage:
    python organize_data.py
"""

import os
import shutil

# Map: final class name -> list of (repo_folder, variety_folder) to merge
SOURCE_MAP = {
    "apple": [
        ("fruits-360-original-size", "Apple Braeburn 1"),
    ],
    "banana": [
        ("fruits-360-original-size", "Banana 3"),
        ("fruits-360-original-size", "Banana 4"),
    ],
    "orange": [
        ("fruits-360-original-size", "Orange 2"),
        ("fruits-360-original-size", "Orange 3"),
        ("fruits-360-original-size", "Orange 4"),
    ],
    "grapes": [
        ("fruits-360-original-size", "Grape 1"),
    ],
    "strawberry": [
        ("fruits-360-original-size", "Strawberry 2"),
        ("fruits-360-original-size", "Strawberry 3"),
    ],
    "mango": [
        ("fruits-360-mango", "Mango 1"),
        ("fruits-360-mango", "Mango Red 1"),
    ],
}

OUT_DIR = os.path.join("data", "raw")


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def main():
    total_copied = 0

    for class_name, sources in SOURCE_MAP.items():
        dest_dir = os.path.join(OUT_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        class_count = 0

        for repo_folder, variety_folder in sources:
            src_dir = os.path.join(repo_folder, "Training", variety_folder)

            if not os.path.isdir(src_dir):
                print(f"  [WARNING] Not found, skipping: {src_dir}")
                continue

            files = [f for f in os.listdir(src_dir) if is_image_file(f)]

            for fname in files:
                src_path = os.path.join(src_dir, fname)
                # Prefix with a short tag from the source variety so
                # filenames never collide when merging multiple varieties
                tag = variety_folder.replace(" ", "_")
                dst_name = f"{tag}_{fname}"
                dst_path = os.path.join(dest_dir, dst_name)

                shutil.copyfile(src_path, dst_path)
                class_count += 1

        print(f"{class_name:12s} -> {class_count} images copied")
        total_copied += class_count

    print(f"\nDone. Total images copied: {total_copied}")
    print(f"Check the '{OUT_DIR}' folder.")
    print("\nNext step: add your own phone photos into these same")
    print("data/raw/<class_name>/ folders, then run preprocess.py")


if __name__ == "__main__":
    main()
