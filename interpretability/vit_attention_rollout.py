import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import timm
from torchvision import datasets, transforms
from PIL import Image
import cv2

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

model = timm.create_model('vit_base_patch16_224.dino', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

class AttentionRollout:
    def __init__(self, model, discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attention_maps = []

    def get_attention_map(self, input_tensor):
        self.attention_maps = []
        hooks = []
        for block in self.model.blocks:
            def make_hook():
                def hook_fn(module, input, output):
                    B, N, C = input[0].shape
                    qkv = module.qkv(input[0])
                    qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv.unbind(0)
                    scale = (C // module.num_heads) ** -0.5
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)
                    self.attention_maps.append(attn.detach().cpu())
                return hook_fn
            h = block.attn.register_forward_hook(make_hook())
            hooks.append(h)

        with torch.no_grad():
            output = self.model(input_tensor)
        for h in hooks:
            h.remove()

        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

        num_tokens = self.attention_maps[0].shape[-1]
        rollout = torch.eye(num_tokens)
        for attn in self.attention_maps:
            attn_avg = attn[0].mean(dim=0)
            flat = attn_avg.flatten()
            threshold = torch.quantile(flat, self.discard_ratio)
            attn_avg[attn_avg < threshold] = 0
            attn_avg = attn_avg + torch.eye(num_tokens)
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            rollout = attn_avg @ rollout

        mask = rollout[0, 1:]
        grid_size = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(grid_size, grid_size).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask, pred_class, confidence


def overlay(pil_image, mask, alpha=0.5):
    img_np = np.array(pil_image.resize((224, 224))).astype(np.uint8)
    mask_resized = cv2.resize(mask, (224, 224))
    heatmap = (cm_module.jet(mask_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return blended


def save_pair(img_path, true_label, idx, fname):
    orig = Image.open(img_path).convert("RGB")
    input_tensor = val_test_transforms(orig).unsqueeze(0).to(device)
    mask, pred_label, confidence = rollout.get_attention_map(input_tensor)
    blended = overlay(orig, mask)

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
    ax.imshow(blended)
    ax.axis('off')
    ax.set_title(f'ViT Dino  Pred: {pred_name} ({confidence * 100:.1f}%) {correct}', fontsize=10, fontweight='bold', color=color, pad=6)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{idx:02d}_{fname}_overlay.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ {fname} — Pred: {pred_name} ({confidence*100:.1f}%)")


rollout = AttentionRollout(model, discard_ratio=0.9)

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