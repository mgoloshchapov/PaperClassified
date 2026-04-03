# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import build_dataloader, split_tensor_paths
from src.metrics import batch_topk_hits, decode_active_labels
from src.utils import INDEX_TO_LABEL


def build_baseline_log_probs(
    train_labels: torch.Tensor,
    top_k: int,
    epsilon: float,
) -> tuple[torch.Tensor, list[str]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")

    support_counts = (train_labels > 0).sum(dim=0).to(torch.float32)
    num_labels = int(support_counts.numel())
    k = min(top_k, num_labels)
    top_values, top_indices = torch.topk(support_counts, k=k)

    probabilities = torch.full((num_labels,), float(epsilon), dtype=torch.float32)
    remaining_mass = 1.0 - epsilon * num_labels
    if remaining_mass <= 0:
        raise ValueError("epsilon is too large for the number of classes")

    if float(top_values.sum().item()) > 0:
        probabilities[top_indices] += remaining_mass * (top_values / top_values.sum())
    else:
        probabilities[top_indices] += remaining_mass / k

    probabilities /= probabilities.sum()
    baseline_labels = [INDEX_TO_LABEL[index] for index in top_indices.tolist()]
    return probabilities.log(), baseline_labels


@hydra.main(config_path="../conf", config_name="evaluate_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    data_dir = to_absolute_path(str(cfg.data.data_dir))

    reference_paths = split_tensor_paths(
        data_dir=data_dir,
        split=cfg.reference_split,
        embedding_template=str(cfg.data.embedding_template),
        labels_template=str(cfg.data.labels_template),
        ids_template=str(cfg.data.ids_template),
        texts_template=str(cfg.data.texts_template),
    )
    reference_labels = torch.load(reference_paths["labels"], map_location="cpu")
    baseline_log_probs, baseline_labels = build_baseline_log_probs(
        train_labels=reference_labels,
        top_k=int(cfg.top_k),
        epsilon=float(cfg.epsilon),
    )

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

    total_loss = 0.0
    total_examples = 0
    total_top1 = 0
    total_top2 = 0

    for batch in dataloader:
        labels = batch["labels"]
        batch_size = labels.size(0)
        batch_log_probs = baseline_log_probs.unsqueeze(0).expand(batch_size, -1)
        loss = F.kl_div(batch_log_probs, labels, reduction="batchmean")

        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_top1 += batch_topk_hits(batch_log_probs, labels, k=1)
        total_top2 += batch_topk_hits(batch_log_probs, labels, k=2)

    print(
        f"split={cfg.split} "
        f"loss={total_loss / max(1, total_examples):.4f} "
        f"top1={total_top1 / max(1, total_examples):.4f} "
        f"top2={total_top2 / max(1, total_examples):.4f}",
    )
    print(
        {
            "reference_split": cfg.reference_split,
            "predicted_top_k": baseline_labels,
        },
    )

    if cfg.output_predictions:
        printed = 0
        for batch in dataloader:
            labels = batch["labels"]
            ids = batch.get("id", [])
            targets = decode_active_labels(labels, INDEX_TO_LABEL)

            for paper_id, target in zip(ids, targets, strict=False):
                print(
                    {
                        "id": paper_id,
                        "predicted_top_k": baseline_labels,
                        "target_labels": target,
                    },
                )
                printed += 1
                if printed >= int(cfg.max_prediction_rows):
                    return


if __name__ == "__main__":
    main()
