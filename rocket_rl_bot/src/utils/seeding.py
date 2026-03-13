from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic_torch: bool = False) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def derive_seed(base_seed: int, offset: int) -> int:
    return int(base_seed + offset * 9973)


def seed_worker(worker_id: int, base_seed: int) -> int:
    return set_global_seeds(derive_seed(base_seed, worker_id))
