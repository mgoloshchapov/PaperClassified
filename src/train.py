# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import build_dataloader
from src.model import PaperClassifier
from src.utils import (
    INDEX_TO_LABEL,
    create_loss,
    create_optimizer,
    create_scheduler,
    ensure_directory,
    resolve_device,
    run_epoch,
    save_checkpoint,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


def resolve_model_input_dim(
    cfg: DictConfig, dataloader: torch.utils.data.DataLoader
) -> int | None:
    if str(cfg.model.encoder_type) != "precomputed":
        return None
    if cfg.model.input_dim is not None:
        return int(cfg.model.input_dim)
    return int(dataloader.dataset.embeddings.shape[-1])


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    device = resolve_device(cfg.device)
    data_dir = to_absolute_path(str(cfg.data.data_dir))
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    hydra_log_path = hydra_output_dir / f"{HydraConfig.get().job.name}.log"

    train_loader = build_dataloader(
        data_dir=data_dir,
        split="train",
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        shuffle=True,
        embedding_template=str(cfg.data.embedding_template),
        labels_template=str(cfg.data.labels_template),
        ids_template=str(cfg.data.ids_template),
    )
    val_loader = build_dataloader(
        data_dir=data_dir,
        split="val",
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        shuffle=False,
        embedding_template=str(cfg.data.embedding_template),
        labels_template=str(cfg.data.labels_template),
        ids_template=str(cfg.data.ids_template),
    )
    input_dim = resolve_model_input_dim(cfg, train_loader)

    model = PaperClassifier(
        model_name=cfg.model.name,
        num_labels=len(INDEX_TO_LABEL),
        num_layers=int(cfg.model.num_layers),
        hidden_dim=int(cfg.model.hidden_dim),
        dropout=float(cfg.model.dropout),
        encoder_type=str(cfg.model.encoder_type),
        input_dim=input_dim,
        freeze_encoder=bool(cfg.model.freeze_encoder),
    ).to(device)

    optimizer = create_optimizer(model, cfg.optimizer)
    total_steps = len(train_loader) * int(cfg.epochs)
    scheduler = create_scheduler(optimizer, cfg.scheduler, total_steps)
    criterion, class_weights = create_loss(cfg.loss, train_loader.dataset.labels)
    criterion = criterion.to(device)

    checkpoint_dir = ensure_directory(to_absolute_path(str(cfg.checkpoint_dir)))
    best_checkpoint_path = Path(checkpoint_dir) / cfg.best_checkpoint_name
    last_checkpoint_path = Path(checkpoint_dir) / cfg.last_checkpoint_name

    LOGGER.info("Training log: %s", hydra_log_path)
    LOGGER.info("Data dir: %s", data_dir)
    LOGGER.info("Checkpoint dir: %s", checkpoint_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Weighted loss enabled: %s", bool(cfg.loss.weighted))
    if class_weights is not None:
        LOGGER.info(
            "Class weights: %s",
            [round(float(weight), 4) for weight in class_weights.tolist()],
        )

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    best_top2 = float("-inf")
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, int(cfg.epochs) + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            desc=f"train-{epoch}",
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scheduler=None,
            desc=f"val-{epoch}",
        )

        epoch_metrics = {
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top2": train_metrics["top2"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top2": val_metrics["top2"],
        }

        LOGGER.info(
            (
                "epoch=%s train_loss=%.4f train_top1=%.4f train_top2=%.4f "
                "val_loss=%.4f val_top1=%.4f val_top2=%.4f"
            ),
            epoch,
            train_metrics["loss"],
            train_metrics["top1"],
            train_metrics["top2"],
            val_metrics["loss"],
            val_metrics["top1"],
            val_metrics["top2"],
        )

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **epoch_metrics})

        save_checkpoint(last_checkpoint_path, model, cfg, epoch, val_metrics)
        if val_metrics["top2"] > best_top2:
            best_top2 = val_metrics["top2"]
            best_metrics = val_metrics
            save_checkpoint(best_checkpoint_path, model, cfg, epoch, val_metrics)

    if wandb_run is not None:
        wandb_run.summary["best_val_top2"] = best_top2
        if best_metrics is not None:
            wandb_run.summary["best_val_top1"] = best_metrics["top1"]
            wandb_run.summary["best_val_loss"] = best_metrics["loss"]
        wandb_run.finish()

    LOGGER.info("Saved checkpoints to %s", checkpoint_dir)


if __name__ == "__main__":
    main()
