from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.classify_data import CLASSIFY_LABELS, create_tiny_placeholder, imagefolder_has_images
from contest_agent.training.common import env_bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight PyTorch scene classifier.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small"], default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="models/classify/classifier.pt")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print("classify training plan")
    print(f"data={args.data} model={args.model} epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} device={args.device}")
    if args.dry_run:
        print("dry-run: no training executed")
        return 0
    if args.epochs > 3 and not (args.allow_long_run or env_bool("RUN_FULL_TRAIN", False)):
        print("refusing long classify training because RUN_FULL_TRAIN=false")
        return 2

    data_root = Path(args.data)
    if not imagefolder_has_images(data_root):
        print("data is missing or empty; creating tiny synthetic placeholder for smoke test only")
        create_tiny_placeholder(data_root)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except Exception as exc:
        print(f"torch/torchvision import failed: {exc}")
        print("install: pip install -r requirements-train.txt")
        return 2

    transform = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(data_root / "train", transform=transform)
    val_ds = datasets.ImageFolder(data_root / "val", transform=transform) if (data_root / "val").exists() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    if args.model == "resnet18":
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(train_ds.classes))
    else:
        net = models.mobilenet_v3_small(weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(train_ds.classes))
    device = torch.device(args.device)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    net.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(1, len(train_loader)):.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "state_dict": net.state_dict(), "classes": train_ds.classes}, output)
    (output.parent / "class_names.json").write_text(json.dumps(train_ds.classes or CLASSIFY_LABELS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}")
    print(f"validation_images={len(val_ds) if val_ds else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

