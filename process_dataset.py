import os
import cv2
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


def merge_dataset(source_root, target_root, splits=("train", "test")):
    os.makedirs(target_root, exist_ok=True)

    sample_split = os.path.join(source_root, splits[0])

    classes = [
        d for d in os.listdir(sample_split)
        if os.path.isdir(os.path.join(sample_split, d))
    ]

    for cls in classes:
        os.makedirs(os.path.join(target_root, cls), exist_ok=True)

    for split in splits:
        split_dir = os.path.join(source_root, split)

        for cls in classes:
            src_dir = os.path.join(split_dir, cls)

            if not os.path.exists(src_dir):
                continue

            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(target_root, cls, file)

                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file)
                    dst_file = os.path.join(target_root, cls, f"{base}_{split}{ext}")

                shutil.copy2(src_file, dst_file)

    print(f"Dataset merged into: {target_root}")


def filter_dataset(input_root, output_root, min_size=700):
    os.makedirs(output_root, exist_ok=True)

    for cls in os.listdir(input_root):
        class_input = os.path.join(input_root, cls)

        if not os.path.isdir(class_input):
            continue

        class_output = os.path.join(output_root, cls)
        os.makedirs(class_output, exist_ok=True)

        for filename in os.listdir(class_input):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            file_path = os.path.join(class_input, filename)

            try:
                with Image.open(file_path) as img:
                    width, _ = img.size
                    if width > min_size:
                        shutil.copy2(file_path, os.path.join(class_output, filename))

            except Exception as e:
                print(f"Skipping {filename}: {e}")

    print("Filtering completed.")


def balance_class(input_dir, output_dir, target_count=3494):
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    random.shuffle(images)
    selected = images[:target_count]

    for img in selected:
        shutil.copy2(os.path.join(input_dir, img), os.path.join(output_dir, img))

    print(f"Copied {len(selected)} images")


def remove_large_white_spots(img, thresh, min_area):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    cleaned = gray.copy()

    for label in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label]
        if area < min_area:
            continue

        pts_y, pts_x = np.where(labels == label)
        touches_border = (
            np.any(pts_y == 0) or
            np.any(pts_y == h - 1) or
            np.any(pts_x == 0) or
            np.any(pts_x == w - 1)
        )

        if touches_border:
            cleaned[labels == label] = 0

    return cleaned


def remove_large_white_spots_full_dataset(input_root, output_root, threshold=220, min_area=1000):
    for category in os.listdir(input_root):
        category_input = os.path.join(input_root, category)
        category_output = os.path.join(output_root, category)

        if not os.path.isdir(category_input):
            continue

        os.makedirs(category_output, exist_ok=True)

        images = sorted(os.listdir(category_input))
        print(f"\nProcessing {category} ({len(images)} images)...")

        for filename in tqdm(images, desc=category):
            path = os.path.join(category_input, filename)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read {filename}")
                continue

            cleaned = remove_large_white_spots(img, thresh=threshold, min_area=min_area)

            out_path = os.path.join(category_output, filename)
            cv2.imwrite(out_path, cleaned)

    print("\nDataset saved at:", output_root)


def split_dataset(base_dir, output_base):
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_base, split), exist_ok=True)

    for cls in tqdm(os.listdir(base_dir), desc="Splitting classes"):
        class_path = os.path.join(base_dir, cls)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff"))]

        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(output_base, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for f in split_files:
                src = os.path.join(class_path, f)
                dst = os.path.join(split_dir, f)
                shutil.copy2(src, dst)

        print(f"{cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    print("\nDataset divided into train, val, and test sets!")
    print(f"Saved to: {output_base}")