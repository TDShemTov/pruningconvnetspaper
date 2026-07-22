"""Shared train/eval loop.

Used for both baseline training (Step 2 of CLAUDE.md) and retraining a model
after pruning (Step 8) — same loop, just called again on a smaller model.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.metrics import compute_metrics


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    # lr/weight_decay defaults below are SGD-tuned (standard ResNet/CIFAR values).
    # Adam/AdamW typically want a much smaller lr (~1e-3-4e-3) and, for AdamW
    # specifically, a larger weight_decay (~0.01-0.05, since its decay isn't
    # entangled with Adam's per-parameter adaptive scaling the way Adam's is) --
    # switching `optimizer` without re-tuning these two is a common way to get
    # a silently-diverging or stalled run, since nothing here auto-adjusts them.
    lr: float = 0.1
    momentum: float = 0.9  # only used by "sgd"
    weight_decay: float = 5e-4
    optimizer: str = "sgd"  # "sgd" | "adam" | "adamw"
    scheduler: str = "cosine"  # "cosine" | "step" | "none"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    seed: int = 42
    # Prints one line per epoch (train/val loss + val accuracy/balanced_accuracy/f1/auc).
    # A single train_model call can run for many minutes with no other output otherwise.
    verbose: bool = True


def _build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'")


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(config.epochs // 3, 1), gamma=0.1)
    if config.scheduler == "none":
        return None
    raise ValueError(f"Unsupported scheduler '{config.scheduler}'")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        proba = torch.softmax(out, dim=1)
        all_true.append(y.cpu().numpy())
        all_pred.append(proba.argmax(dim=1).cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_proba = np.concatenate(all_proba)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["loss"] = total_loss / n
    return metrics


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: TrainConfig = None,
) -> dict:
    """Train `model` on `train_dataset`, evaluating on `test_dataset` every epoch.

    Returns {"history": [per-epoch metrics dicts], "final_metrics": last epoch's metrics}.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, device)
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, **test_metrics})
        if config.verbose:
            print(
                f"  epoch {epoch + 1}/{config.epochs}: train_loss={train_loss:.4f} "
                f"val_loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} "
                f"balanced_acc={test_metrics['balanced_accuracy']:.4f} f1={test_metrics['f1']:.4f} "
                f"auc={test_metrics['auc']:.4f}",
                flush=True,
            )

    return {"history": history, "final_metrics": history[-1] if history else {}}
