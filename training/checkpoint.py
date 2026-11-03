"""Safe-ish checkpoint writing with a hard serialized-file size ceiling."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from joseph.config import ModelConfig


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: ModelConfig,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    payload = {
        "format": "joseph-state-v1",
        "config": config.to_dict(),
        "step": int(step),
        "model_state": source_model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        size = temporary.stat().st_size
        if size > config.max_weight_bytes:
            raise ValueError(
                f"checkpoint is {size} bytes, above the "
                f"{config.max_weight_bytes}-byte limit"
            )
        temporary.replace(target)
        return size
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if payload.get("format") != "joseph-state-v1":
        raise ValueError("unsupported Joseph checkpoint format")
    ModelConfig.from_dict(payload["config"])
    return payload
