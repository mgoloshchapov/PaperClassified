from __future__ import annotations

import torch


def batch_topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices
    label_support = labels > 0
    hits = label_support.gather(dim=1, index=topk)
    return int(hits.any(dim=1).sum().item())


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    return batch_topk_hits(logits, labels, k) / max(1, labels.size(0))


def decode_top_k(
    logits: torch.Tensor,
    index_to_label: list[str],
    k: int = 3,
) -> list[list[str]]:
    if k <= 0:
        raise ValueError("k must be positive")

    topk = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices
    return [[index_to_label[index] for index in row.tolist()] for row in topk]


def decode_active_labels(
    labels: torch.Tensor,
    index_to_label: list[str],
) -> list[list[str]]:
    decoded: list[list[str]] = []
    for row in labels:
        active = (row > 0).nonzero(as_tuple=False).flatten().tolist()
        decoded.append([index_to_label[index] for index in active])
    return decoded
