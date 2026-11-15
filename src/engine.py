"""Training and evaluation loops."""

import torch
from sklearn.metrics import confusion_matrix


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full pass over ``loader`` and update the weights.

    Returns:
        (mean_loss, accuracy) averaged over every sample seen, so a short
        final batch does not get the same weight as a full one.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    """Measure loss and accuracy without touching the weights.

    With ``return_predictions=True`` it also returns every prediction and
    label, which is what the confusion matrix needs.
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * targets.size(0)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if return_predictions:
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    mean_loss = running_loss / total
    accuracy = correct / total

    if return_predictions:
        return (
            mean_loss,
            accuracy,
            torch.cat(all_preds).numpy(),
            torch.cat(all_targets).numpy(),
        )
    return mean_loss, accuracy


def build_confusion_matrix(preds, targets, num_classes: int = 10):
    """Return a num_classes x num_classes matrix.

    Row = true class, column = predicted class. A good model has large
    values on the diagonal; off-diagonal cells show which classes are
    being confused with which.
    """
    return confusion_matrix(targets, preds, labels=list(range(num_classes)))
