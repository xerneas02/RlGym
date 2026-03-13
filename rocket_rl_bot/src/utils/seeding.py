from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

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


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def preserved_random_state(seed: Optional[int] = None, deterministic_torch: bool = False) -> Iterator[None]:
    state = capture_rng_state()
    try:
        if seed is not None:
            set_global_seeds(seed, deterministic_torch=deterministic_torch)
        yield
    finally:
        restore_rng_state(state)


def derive_seed(base_seed: int, offset: int) -> int:
    return int(base_seed + offset * 9973)


def seed_worker(worker_id: int, base_seed: int) -> int:
    return set_global_seeds(derive_seed(base_seed, worker_id))
