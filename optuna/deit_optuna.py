import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

DATASET_ROOT = "dataset_name"

train_dir = os.path.join(DATASET_ROOT, "train")
val_dir   = os.path.join(DATASET_ROOT, "val")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):

    lr_backbone = trial.suggest_float("lr_backbone", 5e-5, 5e-4, log=True)
    lr_classifier = trial.suggest_float("lr_classifier", 3e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = timm.create_model('deit3_base_patch16_224', pretrained=True, num_classes=4,
                              drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)

    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks[8:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.norm.parameters():
        param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True

    model.to(device)

    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "head" in name:
                new_params.append(param)
            else:
                pretrained_params.append(param)

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': lr_backbone},
        {'params': new_params, 'lr': lr_classifier}
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    num_epochs = 15
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct, total = 0, 0

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

        scheduler.step(val_loss)
        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\nBest trial:")
    print("  Value:", study.best_trial.value)
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
