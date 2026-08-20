import os
import cv2
from tqdm import tqdm


def preprocess_image(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read: {input_path}")
        return

    denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    # enhanced = clahe.apply(denoised)

    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, normalized)


def preprocess_folder(input_root, output_root, subset):
    print(f"\nProcessing subset: {subset}")

    input_dir = os.path.join(input_root, subset)

    image_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"  → Found {len(image_paths)} images")

    for input_path in tqdm(image_paths, desc=subset, ncols=100):
        rel_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, rel_path)

        preprocess_image(input_path, output_path)

    print(f"Done: {subset}")


def start_preprocessing(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for subset in ["train", "val", "test"]:
        preprocess_folder(input_root, output_root, subset)

    print("\nPreprocessed dataset saved in:", output_root)

