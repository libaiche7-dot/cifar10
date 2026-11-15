"""Shared helpers: seeding, checkpointing, logging, metrics."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix every source of randomness so that a run can be reproduced.

    Weight initialisation, data shuffling and dropout all draw random
    numbers. Seeding them keeps two runs comparable, which is the only
    way to tell whether a change actually helped.

    Args:
        seed: value every generator is re-initialised with.
        deterministic: force cuDNN to pick deterministic kernels. Costs
            some speed, but without it two identical runs can still differ
            in the last digits.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the requested device, falling back to CPU with a warning.

    Keeping this in one place means the same code runs on a laptop, a
    rented GPU box and a CPU-only machine without edits.
    """
    if prefer.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(prefer)
