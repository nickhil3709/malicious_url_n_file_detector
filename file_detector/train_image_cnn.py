import os
import json
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ------------------------------------------------------------------ #
# Args
# ------------------------------------------------------------------ #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to dataset root containing train/val/test subfolders.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0,  # safer on Windows
                   help="Dataloader workers; 0 recommended on Windows.")
    p.add_argument("--model-out", type=str, default="../models/malware_cnn.pth")
    p.add_argument("--labelmap-out", type=str, default="../models/malware_labelmap.json")
    p.add_argument("--no-pretrained", action="store_true",
                   help="If set, do NOT load ImageNet weights.")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Path resolution helper
# ------------------------------------------------------------------ #
def resolve_data_dir(arg_path: str | None) -> Path:
    """
    If user passes a path, use it. Otherwise try common fallbacks relative to script.
    """
    if arg_path:
        d = Path(arg_path).expanduser().resolve()
        return d

    here = Path(__file__).parent.resolve()
    candidates = [
        here / "splittedDataset",
        here / "splitted_dataset",
        here.parent / "splittedDataset",
        here.parent / "splitted_dataset",
    ]
    for c in candidates:
        if c.is_dir():
            return c

    raise FileNotFoundError(
        "Could not locate dataset. Pass --data-dir pointing to your splittedDataset root."
    )


# ------------------------------------------------------------------ #
# Validate required structure
# ------------------------------------------------------------------ #
def check_dataset_structure(root: Path) -> None:
    expected = [
        root / "train" / "benign",
        root / "train" / "malware",
        root / "val" / "benign",
        root / "val" / "malware",
        root / "test" / "benign",
        root / "test" / "malware",
    ]
    missing = [str(p) for p in expected if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Dataset structure error. Missing:\n  " + "\n  ".join(missing)
        )


# ------------------------------------------------------------------ #
# Data transforms
# Many malware image datasets are grayscale; convert to 3ch for ResNet.
# ------------------------------------------------------------------ #
def make_transforms():
    base_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # force 3ch
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    base_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return {"train": base_train, "val": base_eval, "test": base_eval}


# ------------------------------------------------------------------ #
# Build datasets & dataloaders
# ------------------------------------------------------------------ #
def build_dataloaders(root: Path, batch_size: int, num_workers: int):
    tforms = make_transforms()
    image_datasets = {
        split: datasets.ImageFolder(root / split, transform=tforms[split])
        for split in ["train", "val", "test"]
    }
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ["train", "val", "test"]
    }
    sizes = {k: len(ds) for k, ds in image_datasets.items()}
    classes = image_datasets["train"].classes  # sorted order from subfolder names
    class_to_idx = image_datasets["train"].class_to_idx
    return image_datasets, dataloaders, sizes, classes, class_to_idx


# ------------------------------------------------------------------ #
# Build model
# ------------------------------------------------------------------ #
def build_model(num_classes: int, no_pretrained: bool = False):
    m = models.resnet18(pretrained=(not no_pretrained))
    in_ftrs = m.fc.in_features
    m.fc = nn.Linear(in_ftrs, num_classes)
    return m


# ------------------------------------------------------------------ #
# Train
# ------------------------------------------------------------------ #
def train(model, dataloaders, dataset_sizes, criterion, optimizer, device, epochs, model_out):
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]
            print(f"{phase:>5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), model_out)
                print(f"  [*] Saved new best model (val acc={best_acc:.4f}).")

    # Load best before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest val accuracy: {best_acc:.4f}")
    return model


# ------------------------------------------------------------------ #
# Test
# ------------------------------------------------------------------ #
def test(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\n[TEST] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("[TEST] Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    args = get_args()

    data_root = resolve_data_dir(args.data_dir)
    check_dataset_structure(data_root)

    print(f"[INFO] Using dataset root: {data_root}")
    image_datasets, dataloaders, dataset_sizes, classes, class_to_idx = build_dataloaders(
        data_root, args.batch_size, args.num_workers
    )

    print(f"[INFO] Classes: {classes}")
    print(f"[INFO] Class->Idx: {class_to_idx}")
    print(f"Samples | train={dataset_sizes['train']}  val={dataset_sizes['val']}  test={dataset_sizes['test']}")

    # Save label map for inference later (e.g., in FastAPI)
    os.makedirs(Path(args.labelmap_out).parent, exist_ok=True)
    with open(args.labelmap_out, "w") as f:
        json.dump(class_to_idx, f)
    print(f"[INFO] Saved label map -> {args.labelmap_out}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_model(num_classes=len(classes), no_pretrained=args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    os.makedirs(Path(args.model_out).parent, exist_ok=True)
    model = train(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_out=args.model_out,
    )

    # Test (reload best just in case)
    if Path(args.model_out).is_file():
        model.load_state_dict(torch.load(args.model_out, map_location=device))
    test(model, dataloaders["test"], device, classes)

    print(f"\n[SAVED] Best model -> {args.model_out}")


if __name__ == "__main__":
    main()