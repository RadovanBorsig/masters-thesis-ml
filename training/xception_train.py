import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import ticker
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import timm

DATASET_ROOT = "dataset_name"
EXP_NAME = "experiment_name"

train_dir = os.path.join(DATASET_ROOT, "train")
val_dir   = os.path.join(DATASET_ROOT, "val")
test_dir  = os.path.join(DATASET_ROOT, "test")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(299, scale=(0.80, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10,translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=val_test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

print(f"Dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
print(f"Classes: {train_data.classes}\n")

model = timm.create_model('xception', pretrained=True, num_classes=4)
model.fc = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.fc.in_features, 4))

for param in model.parameters():
    param.requires_grad = False

modules_to_unfreeze = [model.block12, model.conv3, model.bn3, model.conv4, model.bn4, model.fc]

for module in modules_to_unfreeze:
    for param in module.parameters():
        param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Training on: {device}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Percentage trainable: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.2f}%\n")

criterion = nn.CrossEntropyLoss()

pretrained_params = []
new_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if "fc" in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)

optimizer = optim.AdamW([
    {'params': pretrained_params, 'lr': 0.0001},
    {'params': new_params, 'lr': 0.001}
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

early_stopping_patience = 8
best_val_loss = float("inf")
early_stop_counter = 0

num_epochs = 30
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"Starting training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    start = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, dynamic_ncols=True)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_loss = running_loss / ((batch_idx + 1) * train_loader.batch_size)
        batch_acc = 100 * correct / total

        pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "Acc": f"{batch_acc:.2f}%"})

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        print("Validation loss improved.")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stop counter: {early_stop_counter}/{early_stopping_patience}")

    if early_stop_counter >= early_stopping_patience:
        print("\nEarly stopping triggered!")
        break

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"Time: {elapsed:.1f}s")

torch.save(model.state_dict(), f"training/models/xception_{EXP_NAME}.pth")
print(f"Final model saved as xception_{EXP_NAME}.pth")

print("\nFinal Training Metrics:")
print(f"Train Loss: {train_losses[-1]:.4f}")
print(f"Train Accuracy: {train_accs[-1]:.2f}%")
print(f"Validation Loss: {val_losses[-1]:.4f}")
print(f"Validation Accuracy: {val_accs[-1]:.2f}%")

epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Trénovacia strata", linewidth=2)
plt.plot(epochs, val_losses, label="Validačná strata", linewidth=2)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.title("Trénovacia a validačná strata", fontsize=14, fontweight='bold')
plt.xlabel("Epocha")
plt.ylabel("Strata")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Trénovacia úspešnosť", linewidth=2)
plt.plot(epochs, val_accs, label="Validačná úspešnosť", linewidth=2)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.title("Trénovacia a validačná úspešnosť", fontsize=14, fontweight='bold')
plt.xlabel("Epocha")
plt.ylabel("Úspešnosť (%)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"xception_progress_{EXP_NAME}.png", dpi=300, bbox_inches='tight')
plt.show()

print("Evaluating on test set...\n")
model.eval()
all_preds, all_labels = [], []
test_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_loss /= len(test_loader.dataset)
test_acc = 100 * correct / total

print(f"\nTest set results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
