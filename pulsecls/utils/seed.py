from __future__ import annotations

import random

import numpy as np
import torch

from pulsecls.utils.device import get_device


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Respect the trainer's selection; auto would switch npu:1 back to npu:0.
    if device is None:
        device = get_device()
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

