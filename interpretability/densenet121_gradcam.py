import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

DATASET_ROOT = "dataset_name"
MODEL_PATH   = "path_to_model"
OUTPUT_DIR   = "name_of_output_dir"
TARGET_CLASS = "class_name"

SELECTED_FILES = [""]

os.makedirs(OUTPUT_DIR, exist_ok=True)

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dir = os.path.join(DATASET_ROOT, "test")
test_data = datasets.ImageFolder(test_dir, transform=val_test_transforms)
class_names = test_data.classes
target_idx = class_names.index(TARGET_CLASS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.classifier.in_features, len(class_names)))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

target_layers = [model.features.denseblock4.denselayer16.conv2]
cam = GradCAM(model=model, target_layers=target_layers)

def overlay(pil_img, grayscale_cam):
    img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = cv2.resize(grayscale_cam, (224, 224))
    return show_cam_on_image(img, cam_resized, use_rgb=True)


def save_pair(img_path, true_label, idx, fname):
    orig = Image.open(img_path).convert("RGB")
    input_tensor = val_test_transforms(orig).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    pred_label = output.argmax(dim=1).item()
    confidence = probs[0, pred_label].item()

    targets = [ClassifierOutputTarget(pred_label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    vis = overlay(orig, grayscale_cam)

    correct = '✓' if pred_label == true_label else '✗'
    pred_name = class_names[pred_label]
    color = 'green' if pred_label == true_label else 'red'

    fig, ax = plt.subplots(figsize=(3, 3.2))
    ax.imshow(orig.resize((224, 224)))
    ax.axis('off')
    ax.set_title(f'{TARGET_CLASS} {idx}', fontsize=10, fontweight='bold', pad=6)
    plt.tight_layout(pad=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, f'{idx:02d}_{fname}_orig.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    fig, ax = plt.subplots(figsize=(3, 3.2))
    ax.imshow(vis)
    ax.axis('off')
    ax.set_title(f'DenseNet-121  Pred: {pred_name} ({confidence * 100:.1f}%) {correct}', fontsize=10, fontweight='bold', color=color, pad=6)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{idx:02d}_{fname}_overlay.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ {fname} — Pred: {pred_name} ({confidence*100:.1f}%)")

file_map = {}
for img_path, label in test_data.samples:
    fname = os.path.splitext(os.path.basename(img_path))[0]
    file_map[fname] = (img_path, label)

for idx, fname in enumerate(SELECTED_FILES, start=1):
    if fname in file_map:
        img_path, label = file_map[fname]
        save_pair(img_path, label, idx, fname)
    else:
        print(f"✗ File {fname} not found!")

print(f"\nAll images saved to: {OUTPUT_DIR}/")