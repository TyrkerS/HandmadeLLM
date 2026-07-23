"""Token-bin data loading: uint16 memmaps + random-offset batch sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_bin(path: str | Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def write_bin(path: str | Path, ids: np.ndarray) -> None:
    assert ids.dtype == np.uint16
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(path)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    if torch.device(device).type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
