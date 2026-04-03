# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import build_dataloader
from src.metrics import decode_active_labels, decode_top_k
from src.model import PaperClassifier
from src.utils import (
    INDEX_TO_LABEL,
    create_loss,
    load_checkpoint,
    resolve_device,
    run_epoch,
)


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> PaperClassifier:
    checkpoint = load_checkpoint(checkpoint_path)
    classifier_config = checkpoint["classifier_config"]
    model = PaperClassifier(
        model_name=checkpoint["model_name"],
        num_labels=int(checkpoint["num_labels"]),
        num_layers=int(classifier_config["num_layers"]),
        hidden_dim=int(classifier_config["hidden_dim"]),
        dropout=float(classifier_config["dropout"]),
        encoder_type=str(checkpoint.get("encoder_type", "hf")),
        input_dim=int(checkpoint["hidden_size"]),
        freeze_encoder=bool(checkpoint["freeze_encoder"]),
    )
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    return model.to(device)


@hydra.main(config_path="../conf", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.device)
    data_dir = to_absolute_path(str(cfg.data.data_dir))
    checkpoint_path = to_absolute_path(str(cfg.checkpoint_path))
    model = load_model_from_checkpoint(checkpoint_path, device)
    train_loader = build_dataloader(
        data_dir=data_dir,
        split="train",
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

    dataloader = build_dataloader(
        data_dir=data_dir,
        split=cfg.split,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        shuffle=False,
        embedding_template=str(cfg.data.embedding_template),
        labels_template=str(cfg.data.labels_template),
        ids_template=str(cfg.data.ids_template),
    )
    metrics = run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scheduler=None,
        desc=f"evaluate-{cfg.split}",
    )

    print(
        (
            f"split={cfg.split} "
            f"loss={metrics['loss']:.4f} "
            f"top1={metrics['top1']:.4f} top2={metrics['top2']:.4f}"
        ),
    )

    if cfg.output_predictions:
        printed = 0
        with torch.inference_mode():
            for batch in dataloader:
                embeddings = batch["embedding"].to(device)
                labels = batch["labels"]
                ids = batch.get("id", [])
                logits = model(embeddings=embeddings)
                predicted = decode_top_k(logits.cpu(), INDEX_TO_LABEL, k=int(cfg.top_k))
                targets = decode_active_labels(labels, INDEX_TO_LABEL)

                for paper_id, prediction, target in zip(
                    ids, predicted, targets, strict=False
                ):
                    print(
                        {
                            "id": paper_id,
                            "predicted_top_k": prediction,
                            "target_labels": target,
                        },
                    )
                    printed += 1
                    if printed >= int(cfg.max_prediction_rows):
                        return


if __name__ == "__main__":
    main()
