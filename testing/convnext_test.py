import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DATASET_ROOT = "dataset_name"
MODEL_PATH = "path_to_saved_model"

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"), transform=val_test_transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Test size: {len(test_data)}")
print(f"Classes: {test_data.classes}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.convnext_base(weights=None)
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, 4)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

all_preds, all_labels, all_probs = [], [], []
test_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item() * inputs.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_loss /= len(test_loader.dataset)
test_acc = 100 * correct / total
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_data.classes, digits=4))

cm = confusion_matrix(all_labels, all_preds)
cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')

displayed_vals = []
for i in range(cm_norm.shape[0]):
    row = [round(cm_norm[i, j], 3) for j in range(cm_norm.shape[1])]
    diff = round(1.0 - sum(row), 3)
    row[i] = round(row[i] + diff, 3)
    displayed_vals.append(row)

fig, ax = plt.subplots(figsize=(7, 6.5))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

for k in range(1, len(test_data.classes)):
    ax.axhline(k - 0.5, color='white', linewidth=2)
    ax.axvline(k - 0.5, color='white', linewidth=2)

ax.set_xticks(np.arange(len(test_data.classes)))
ax.set_yticks(np.arange(len(test_data.classes)))
ax.set_xticklabels(test_data.classes, fontsize=10, fontweight='bold', rotation=0)
ax.set_yticklabels(test_data.classes, fontsize=10, fontweight='bold', rotation=90, va='center')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel("Predikovaná trieda", fontsize=11, labelpad=12, color='#555')
ax.set_ylabel("Skutočná trieda",    fontsize=11, labelpad=12, color='#555')

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = displayed_vals[i][j]
        text_color = "white"          if val >= 0.5 else "#1a3250"
        sub_color  = (1, 1, 1, 0.55) if val >= 0.5 else "#7a9bb5"

        ax.text(j, i - 0.12, f"{cm[i, j]}", ha="center", va="center", color=text_color, fontsize=11, fontweight='bold', fontfamily='monospace')
        ax.text(j, i + 0.2, f"({val:.3f})", ha="center", va="center", color=sub_color, fontsize=8.5, fontfamily='monospace')

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.ax.tick_params(labelsize=9, length=0)
cbar.outline.set_visible(False)

fig.text(0.5, 1.01, f"ConvNeXt  ·  Testovacia úspešnosť: {test_acc:.2f}%", ha='center', fontsize=11, fontweight='bold', color='#1a1a1a', fontfamily='monospace')

plt.tight_layout()
plt.savefig("convnext_confusion_matrix.png", dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.show()

n_classes = len(test_data.classes)
all_labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ['#1f77b4', '#e05c2a', '#2ca02c', '#9467bd']

fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
fig_roc.patch.set_facecolor('#ffffff')
ax_roc.set_facecolor('#ffffff')

for i, (cls_name, color) in enumerate(zip(test_data.classes, colors)):
    ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f"{cls_name}  (AUC = {roc_auc[i]:.3f})")

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.02])
ax_roc.set_xlabel("Falošne pozitívna miera", fontsize=11, labelpad=10, color='#555')
ax_roc.set_ylabel("Pravdivo pozitívna miera", fontsize=11, labelpad=10, color='#555')

for spine in ax_roc.spines.values():
    spine.set_visible(False)
ax_roc.tick_params(length=0, labelsize=9)
ax_roc.grid(True, linestyle='--', alpha=0.3)
ax_roc.legend(loc='lower right', fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig("convnext_roc_curve.png", dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.show()

print("\nPer-Class Accuracy:")
for i, class_name in enumerate(test_data.classes):
    class_correct = cm[i, i]
    class_total   = cm[i].sum()
    class_acc     = 100 * class_correct / class_total if class_total > 0 else 0
    print(f"  {class_name:20s}: {class_acc:.2f}% ({class_correct}/{class_total})")

print("\nTesting finished!")