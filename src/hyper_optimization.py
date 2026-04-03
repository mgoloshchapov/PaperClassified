# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import optuna
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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
    resolve_device,
    run_epoch,
    seed_everything,
)


def objective_factory(cfg: DictConfig) -> callable:
    device = resolve_device(cfg.device)
    data_dir = to_absolute_path(str(cfg.data.data_dir))
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
    criterion, _ = create_loss(cfg.loss, train_loader.dataset.labels)
    criterion = criterion.to(device)

    def objective(trial: optuna.Trial) -> float:
        num_layers = trial.suggest_int(
            "num_layers",
            int(cfg.optuna.search.num_layers.min),
            int(cfg.optuna.search.num_layers.max),
        )
        hidden_dim = trial.suggest_categorical(
            "hidden_dim",
            list(cfg.optuna.search.hidden_dim.choices),
        )
        dropout = trial.suggest_float(
            "dropout",
            float(cfg.optuna.search.dropout.min),
            float(cfg.optuna.search.dropout.max),
        )
        learning_rate = trial.suggest_float(
            "lr",
            float(cfg.optuna.search.lr.min),
            float(cfg.optuna.search.lr.max),
            log=True,
        )
        scheduler_name = trial.suggest_categorical(
            "scheduler",
            list(cfg.optuna.search.scheduler.choices),
        )

        model = PaperClassifier(
            model_name=cfg.model.name,
            num_labels=len(INDEX_TO_LABEL),
            num_layers=num_layers,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            encoder_type=str(cfg.model.encoder_type),
            input_dim=(
                None
                if str(cfg.model.encoder_type) != "precomputed"
                else int(
                    (
                        cfg.model.input_dim
                        if cfg.model.input_dim is not None
                        else train_loader.dataset.embeddings.shape[-1]
                    ),
                )
            ),
            freeze_encoder=bool(cfg.model.freeze_encoder),
        ).to(device)

        optimizer_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.optimizer, resolve=True)
        )
        optimizer_cfg.lr = float(learning_rate)
        optimizer = create_optimizer(model, optimizer_cfg)

        scheduler_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.scheduler, resolve=True)
        )
        scheduler_cfg.name = scheduler_name
        total_steps = len(train_loader) * int(cfg.epochs)
        scheduler = create_scheduler(optimizer, scheduler_cfg, total_steps)

        best_val_top1 = float("-inf")
        final_metrics: dict[str, float] = {}

        for epoch in range(1, int(cfg.epochs) + 1):
            train_metrics = run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                desc=f"optuna-train-{trial.number}-{epoch}",
            )
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scheduler=None,
                desc=f"optuna-val-{trial.number}-{epoch}",
            )

            final_metrics = {
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top2": train_metrics["top2"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top2": val_metrics["top2"],
            }
            best_val_top1 = max(best_val_top1, val_metrics["top1"])

            trial.report(val_metrics["top1"], step=epoch)
            if trial.should_prune():
                trial.set_user_attr("pruned_epoch", epoch)
                raise optuna.TrialPruned()

        for key, value in final_metrics.items():
            trial.set_user_attr(key, value)
        return best_val_top1

    return objective


@hydra.main(config_path="../conf", config_name="hyper_optimization", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    sampler = TPESampler(seed=int(cfg.optuna.sampler_seed))
    pruner = MedianPruner(
        n_startup_trials=int(cfg.optuna.pruner.n_startup_trials),
        n_warmup_steps=int(cfg.optuna.pruner.n_warmup_steps),
    )
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=cfg.optuna.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective_factory(cfg),
        n_trials=int(cfg.optuna.n_trials),
        timeout=None if cfg.optuna.timeout is None else int(cfg.optuna.timeout),
    )

    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
