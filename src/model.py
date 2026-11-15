"""Model definitions and the factory used by the training script."""

import torch.nn as nn
import torchvision


class SimpleCNN(nn.Module):
    """A small VGG-style baseline for 32x32 inputs.

    Six convolutions in three stages, each stage followed by 2x2 max
    pooling, so the spatial size goes 32 -> 16 -> 8 -> 4. Batch
    normalisation after every convolution keeps the deeper stacks
    trainable at a reasonable learning rate.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 images.

    The ImageNet version starts with a 7x7 stride-2 convolution and a 3x3
    stride-2 max pool, which turns a 32x32 input into 8x8 before the first
    residual stage. Replacing the stem with a 3x3 stride-1 convolution and
    dropping the max pool keeps the resolution, which is what CIFAR models
    are expected to do.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model named in ``cfg['model']['name']``."""
    name = cfg["model"]["name"].lower()
    num_classes = cfg["model"]["num_classes"]

    if name == "cnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes)
    raise ValueError(f"unknown model name: {name}")
