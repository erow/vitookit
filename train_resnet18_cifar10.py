#!/usr/bin/env python3
"""Train ResNet18 on CIFAR10 using vitookit APIs (CPU-compatible)."""

import argparse
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import gin
from vitookit.models.build_model import build_model


def get_args():
    parser = argparse.ArgumentParser("Train ResNet18 on CIFAR10")
    parser.add_argument("--data_location", default="/tmp/cifar10_data", type=str)
    parser.add_argument("--output_dir", default="outputs/resnet18-cifar10", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--input_size", default=32, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  acc={100.*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    gin.parse_config("")

    # Build model via vitookit
    print("Building ResNet18 via vitookit build_model()...")
    model = build_model("resnet18", num_classes=10)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Build CIFAR10 datasets with standard transforms for 32x32
    print("Building CIFAR10 datasets...")
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = datasets.CIFAR10(args.data_location, train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(args.data_location, train=False, transform=val_transform, download=True)
    nb_classes = 10
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples, Classes: {nb_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nStarting training for {args.epochs} epochs on {device}...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}\n")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"time={elapsed:.1f}s")

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"  => val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(args.output_dir, "checkpoint_best.pth")
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                }, ckpt_path)
                print(f"  => Best model saved ({val_acc:.1f}%)")

    # Save final checkpoint
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs,
        "val_acc": best_val_acc,
    }, os.path.join(args.output_dir, "checkpoint_final.pth"))

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.1f}%")
    print(f"Checkpoints saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
