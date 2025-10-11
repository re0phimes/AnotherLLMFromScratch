"""Checkpoint 持久化工具。

作用：保存/恢复训练状态（模型、优化器、调度器、AMP scaler、随机种子）。
使用位置：Trainer 在训练过程中定期调用 `save_checkpoint`，恢复训练时调用
`load_checkpoint` + `restore_training_state`。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class CheckpointState:
    step: int
    epoch: int
    global_batch_size: int
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[Dict[str, Any]] = None
    rng_state: Optional[Dict[str, Any]] = None


def _atomic_write(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(".tmp")
    shutil.move(str(src), tmp_path)
    tmp_path.replace(dst)


def save_checkpoint(
    output_dir: Path,
    *,
    state: CheckpointState,
    is_best: bool = False,
    filename: str = "checkpoint.pt",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / filename

    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp_file:
        torch.save(
            {
                "step": state.step,
                "epoch": state.epoch,
                "global_batch_size": state.global_batch_size,
                "model_state": state.model_state,
                "optimizer_state": state.optimizer_state,
                "scheduler_state": state.scheduler_state,
                "scaler_state": state.scaler_state,
                "rng_state": state.rng_state,
            },
            tmp_file.name,
        )
        tmp_temp_path = Path(tmp_file.name)
    _atomic_write(tmp_temp_path, ckpt_path)

    if is_best:
        best_path = output_dir / "best.pt"
        shutil.copy2(ckpt_path, best_path)
        return best_path

    return ckpt_path


def load_checkpoint(path: Path, map_location: Optional[str | torch.device] = None) -> CheckpointState:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location)
    return CheckpointState(
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        global_batch_size=int(payload.get("global_batch_size", 0)),
        model_state=dict(payload.get("model_state", {})),
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        scaler_state=payload.get("scaler_state"),
        rng_state=payload.get("rng_state"),
    )


def capture_training_state(
    *,
    step: int,
    epoch: int,
    global_batch_size: int,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> CheckpointState:
    rng_state = {
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    return CheckpointState(
        step=step,
        epoch=epoch,
        global_batch_size=global_batch_size,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict() if optimizer is not None else None,
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        scaler_state=scaler.state_dict() if scaler is not None else None,
        rng_state=rng_state,
    )


def restore_training_state(
    state: CheckpointState,
    *,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True,
) -> None:
    model.load_state_dict(state.model_state, strict=strict)
    if optimizer is not None and state.optimizer_state is not None:
        optimizer.load_state_dict(state.optimizer_state)
    if scheduler is not None and state.scheduler_state is not None:
        scheduler.load_state_dict(state.scheduler_state)
    if scaler is not None and state.scaler_state is not None:
        scaler.load_state_dict(state.scaler_state)
    if state.rng_state is not None:
        torch.set_rng_state(state.rng_state.get("torch"))
        if torch.cuda.is_available() and "cuda" in state.rng_state:
            torch.cuda.set_rng_state_all(state.rng_state["cuda"])
