"""Entry point: python train.py --config configs/base.yaml

Wires together everything under ``src/``: builds the loaders and the model
from the YAML config, runs the training loop with a step learning-rate
schedule, keeps the best checkpoint on the validation split, and reports
the test split exactly once at the very end.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.data import CIFAR10_CLASSES, build_dataloaders
from src.engine import build_confusion_matrix, evaluate, train_one_epoch
from src.model import build_model
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--output", default=None, help="override output.dir")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_run_dir(base: str) -> Path:
    """Create a timestamped run directory so two runs never overwrite."""
    run_dir = Path(base) / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, best_acc: float) -> None:
    """Persist everything needed to resume training, not just the weights."""
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        },
        path,
    )


def plot_confusion_matrix(cm, classes, path: Path) -> None:
    """Save a labelled confusion matrix as a PNG for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("CIFAR-10 confusion matrix (test split)")

    threshold = cm.max() / 2.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, str(cm[row, col]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.output is not None:
        cfg["output"]["dir"] = args.output

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))

    run_dir = make_run_dir(cfg["output"]["dir"])
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    print(f"[info] device   : {device}")
    print(f"[info] run dir  : {run_dir}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model    : {cfg['model']['name']} ({num_params / 1e6:.2f}M params)")
    print(
        f"[info] batches  : {len(train_loader)} train / "
        f"{len(val_loader)} val / {len(test_loader)} test"
    )

    train_cfg = cfg["train"]
    epochs = train_cfg["epochs"]
    patience = train_cfg.get("early_stop_patience", epochs)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg["lr_milestones"],
        gamma=train_cfg["lr_gamma"],
    )

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    ckpt_dir = run_dir / "checkpoints"

    best_acc = 0.0
    best_epoch = -1
    stale = 0
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        print(
            f"[{epoch + 1:03d}/{epochs:03d}] lr {current_lr:.4f} | "
            f"train {train_loss:.4f} acc {train_acc:.4f} | "
            f"val {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            stale = 0
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scheduler, epoch, best_acc)
        else:
            stale += 1

        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scheduler, epoch, best_acc)

        if stale >= patience:
            print(f"[info] early stopping: no val improvement for {patience} epochs")
            break

    writer.close()
    print(f"[info] best val acc {best_acc:.4f} at epoch {best_epoch}")

    # The test split is touched exactly once, with the best weights.
    # weights_only=False is acceptable here: this file was written by this
    # same script minutes ago. Never load third-party weights this way.
    state = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    test_loss, test_acc, preds, targets = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    cm = build_confusion_matrix(preds, targets, cfg["model"]["num_classes"])
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print(f"[test] loss {test_loss:.4f} acc {test_acc:.4f}")
    for name, acc in zip(CIFAR10_CLASSES, per_class_acc):
        print(f"       {name:<11s} {acc:.4f}")

    plot_confusion_matrix(cm, CIFAR10_CLASSES, run_dir / "confusion_matrix.png")

    summary = {
        "model": cfg["model"]["name"],
        "params": num_params,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_class_acc": {n: float(a) for n, a in zip(CIFAR10_CLASSES, per_class_acc)},
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[info] artifacts: {run_dir}")


if __name__ == "__main__":
    main()
