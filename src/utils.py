from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.metrics import batch_topk_hits

ARXIV_CATEGORIES = {
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math": "Mathematics",
    "math-ph": "Mathematical Physics",
    "nlin": "Nonlinear Sciences",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "quant-ph": "Quantum Physics",
    "stat": "Statistics",
}

INDEX_TO_LABEL = list(ARXIV_CATEGORIES)
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(INDEX_TO_LABEL)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def make_portable_path(path: str | Path, base_dir: str | Path | None = None) -> str:
    target_path = Path(path)
    if not target_path.is_absolute():
        return target_path.as_posix()

    reference_dir = Path.cwd() if base_dir is None else Path(base_dir)
    try:
        return target_path.relative_to(reference_dir).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(target_path, start=reference_dir)).as_posix()
        except ValueError:
            return target_path.as_posix()


def resolve_portable_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    reference_dir = Path.cwd() if base_dir is None else Path(base_dir)
    return reference_dir / path_obj


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def combine_text(title: str, abstract: str) -> str:
    return normalize_whitespace(f"{title} {abstract}")


def normalize_categories(raw_categories: str) -> list[str]:
    categories = {
        token.split(".", maxsplit=1)[0] for token in raw_categories.split() if token
    }
    return sorted(category for category in categories if category in LABEL_TO_INDEX)


def encode_soft_labels(categories: list[str]) -> torch.Tensor:
    if not categories:
        raise ValueError("At least one category is required to build a soft label")
    labels = torch.zeros(len(LABEL_TO_INDEX), dtype=torch.float32)
    weight = 1.0 / len(categories)
    for category in categories:
        labels[LABEL_TO_INDEX[category]] = weight
    return labels


def decode_soft_labels(labels: torch.Tensor) -> list[str]:
    if labels.ndim != 1:
        raise ValueError("decode_soft_labels expects a 1D label tensor")
    if labels.numel() != len(INDEX_TO_LABEL):
        raise ValueError(
            "Expected label tensor of size "
            f"{len(INDEX_TO_LABEL)}, got {labels.numel()}",
        )
    active_indices = (labels > 0).nonzero(as_tuple=False).flatten().tolist()
    return [INDEX_TO_LABEL[index] for index in active_indices]


class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = 1.0 if self.class_weights is None else self.class_weights.unsqueeze(0)
        weighted_targets = targets * weights
        loss = -(weighted_targets * log_probs).sum(dim=-1)
        return loss.mean()


def compute_class_weights(train_labels: torch.Tensor) -> torch.Tensor:
    if train_labels.ndim != 2:
        raise ValueError("train_labels must be a 2D tensor")
    support_counts = (train_labels > 0).sum(dim=0).to(torch.float32)
    support_counts = support_counts.clamp_min(1.0)
    inverse_frequency = 1.0 / support_counts
    normalized = inverse_frequency / inverse_frequency.mean()
    return normalized


def create_loss(
    loss_cfg: DictConfig,
    train_labels: torch.Tensor | None = None,
) -> tuple[nn.Module, torch.Tensor | None]:
    class_weights = None
    if bool(loss_cfg.weighted):
        if train_labels is None:
            raise ValueError("train_labels are required when loss.weighted=true")
        class_weights = compute_class_weights(train_labels)
    return SoftTargetCrossEntropyLoss(class_weights=class_weights), class_weights


def stable_hash_int(value: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def create_optimizer(model: nn.Module, optimizer_cfg: DictConfig) -> Optimizer:
    name = optimizer_cfg.name.lower()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")
    return AdamW(
        parameters,
        lr=float(optimizer_cfg.lr),
        weight_decay=float(optimizer_cfg.weight_decay),
    )


def create_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: DictConfig,
    total_steps: int,
) -> LambdaLR | None:
    name = scheduler_cfg.name.lower()
    if name == "none" or total_steps <= 0:
        return None

    warmup_ratio = float(getattr(scheduler_cfg, "warmup_ratio", 0.0))
    warmup_steps = int(total_steps * warmup_ratio)
    num_cycles = float(getattr(scheduler_cfg, "num_cycles", 0.5))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)

        progress_steps = max(1, total_steps - warmup_steps)
        progress = float(current_step - warmup_steps) / progress_steps
        progress = min(max(progress, 0.0), 1.0)

        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            cosine = 0.5 * (1.0 + np.cos(np.pi * 2.0 * num_cycles * progress))
            return max(0.0, float(cosine))
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg.name}")

    return LambdaLR(optimizer, lr_lambda)


def checkpoint_payload(
    model: nn.Module,
    cfg: DictConfig,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    classifier_config = getattr(model, "classifier_config")
    return {
        "epoch": epoch,
        "metrics": metrics,
        "model_name": getattr(model, "model_name"),
        "encoder_type": getattr(model, "encoder_type", "hf"),
        "num_labels": getattr(model, "num_labels"),
        "hidden_size": getattr(model, "hidden_size"),
        "freeze_encoder": getattr(model, "freeze_encoder"),
        "classifier_config": classifier_config,
        "classifier_state_dict": getattr(model, "classifier").state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: DictConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    checkpoint_path = Path(path)
    ensure_directory(checkpoint_path.parent)
    torch.save(checkpoint_payload(model, cfg, epoch, metrics), checkpoint_path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    desc: str = "epoch",
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    if (
        getattr(model, "freeze_encoder", False)
        and getattr(model, "encoder", None) is not None
    ):
        model.encoder.eval()

    total_loss = 0.0
    total_examples = 0
    total_top1 = 0
    total_top2 = 0

    iterator = tqdm(dataloader, desc=desc, leave=False)
    for batch in iterator:
        embeddings = batch["embedding"].to(device)
        labels = batch["labels"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings=embeddings)
            loss = criterion(torch.log_softmax(logits, dim=-1), labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.inference_mode():
                logits = model(embeddings=embeddings)
                loss = criterion(torch.log_softmax(logits, dim=-1), labels)

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_top1 += batch_topk_hits(logits.detach(), labels, k=1)
        total_top2 += batch_topk_hits(logits.detach(), labels, k=2)

    return {
        "loss": total_loss / max(1, total_examples),
        "top1": total_top1 / max(1, total_examples),
        "top2": total_top2 / max(1, total_examples),
    }
