"""Dataset and dataloader construction for CIFAR-10."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Per-channel mean and std of the CIFAR-10 training set, computed over all
# 50k images. Using fixed constants instead of recomputing them each run
# keeps experiments comparable.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transforms(augment: bool) -> transforms.Compose:
    """Return the transform pipeline for training or evaluation.

    Augmentation is applied to training images only. Validating on
    augmented images would make the metric depend on random crops and
    hide real progress.
    """
    ops = []
    if augment:
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(ops)


def build_dataloaders(cfg: dict, val_size: int = 5000):
    """Build the train / val / test loaders described by ``cfg``.

    The official 50k training images are split into 45k train and 5k val.
    Two dataset objects are created so that augmentation reaches only the
    training half; both use the same permutation, so no image lands in
    both splits. The 10k test set is left untouched and should be used
    once, at the end.
    """
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 0)
    seed = cfg.get("seed", 42)

    train_full = datasets.CIFAR10(
        root, train=True, download=True, transform=build_transforms(augment=True)
    )
    val_full = datasets.CIFAR10(
        root, train=True, download=True, transform=build_transforms(augment=False)
    )
    test_set = datasets.CIFAR10(
        root, train=False, download=True, transform=build_transforms(augment=False)
    )

    lengths = [len(train_full) - val_size, val_size]

    generator = torch.Generator().manual_seed(seed)
    train_set, _ = random_split(train_full, lengths, generator=generator)

    generator = torch.Generator().manual_seed(seed)
    _, val_set = random_split(val_full, lengths, generator=generator)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    return train_loader, val_loader, test_loader
