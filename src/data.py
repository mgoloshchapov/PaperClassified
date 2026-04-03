from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from src.utils import (
    combine_text,
    encode_soft_labels,
    normalize_categories,
    stable_hash_int,
)


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    text: str
    labels: torch.Tensor


class JSONLRowDataset(IterableDataset):
    def __init__(self, raw_path: str | Path) -> None:
        self.raw_path = Path(raw_path)

    def __iter__(self) -> Iterator[dict[str, object]]:
        for record in iter_raw_papers(self.raw_path):
            yield {
                "id": record.paper_id,
                "text": record.text,
                "labels": record.labels,
            }


def iter_raw_papers(
    raw_path: str | Path,
    progress_desc: str | None = None,
) -> Iterator[PaperRecord]:
    input_path = Path(raw_path)
    with input_path.open("r", encoding="utf-8") as handle:
        iterator = handle
        if progress_desc is not None:
            iterator = tqdm(handle, desc=progress_desc, unit="rows")

        for line in iterator:
            row = json.loads(line)
            paper_id = str(row.get("id", "")).strip()
            if not paper_id:
                continue

            text = combine_text(
                str(row.get("title", "")).strip(),
                str(row.get("abstract", "")).strip(),
            )
            if not text:
                continue

            categories = normalize_categories(str(row.get("categories", "")))
            if not categories:
                continue

            yield PaperRecord(
                paper_id=paper_id,
                text=text,
                labels=encode_soft_labels(categories),
            )


def select_deterministic_sample(
    raw_path: str | Path,
    sample_size: int,
    seed: int,
) -> list[PaperRecord]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    heap: list[tuple[int, int, PaperRecord]] = []
    counter = 0

    for record in iter_raw_papers(raw_path, progress_desc="Sampling papers"):
        score = stable_hash_int(record.paper_id, seed)
        entry = (-score, counter, record)
        if len(heap) < sample_size:
            heapq.heappush(heap, entry)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, entry)
        counter += 1

    if len(heap) < sample_size:
        raise ValueError(
            "Requested sample_size="
            f"{sample_size}, but found only {len(heap)} valid records",
        )

    sampled = [(-neg_score, record.paper_id, record) for neg_score, _, record in heap]
    sampled.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in sampled]


def split_records(
    records: list[PaperRecord],
    split_sizes: dict[str, int],
) -> dict[str, list[PaperRecord]]:
    required_keys = {"train", "val", "test"}
    if set(split_sizes) != required_keys:
        raise ValueError(f"split_sizes must contain {required_keys}")

    total_requested = sum(split_sizes.values())
    if len(records) < total_requested:
        raise ValueError(
            f"Need {total_requested} records for splitting, received {len(records)}",
        )

    train_end = split_sizes["train"]
    val_end = train_end + split_sizes["val"]
    test_end = val_end + split_sizes["test"]
    selected = records[:test_end]

    return {
        "train": selected[:train_end],
        "val": selected[train_end:val_end],
        "test": selected[val_end:test_end],
    }


class TensorSplitDataset(Dataset):
    def __init__(
        self,
        embeddings_path: str | Path,
        labels_path: str | Path,
        ids_path: str | Path | None = None,
    ) -> None:
        self.embeddings = torch.load(Path(embeddings_path), map_location="cpu")
        self.labels = torch.load(Path(labels_path), map_location="cpu")
        self.ids = None
        if ids_path is not None and Path(ids_path).exists():
            self.ids = torch.load(Path(ids_path), map_location="cpu")

        if len(self.embeddings) != len(self.labels):
            raise ValueError("Embeddings and labels must have the same length")
        if self.ids is not None and len(self.ids) != len(self.labels):
            raise ValueError("Ids and labels must have the same length")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {
            "embedding": self.embeddings[index],
            "labels": self.labels[index],
        }
        if self.ids is not None:
            item["id"] = self.ids[index]
        return item


def split_tensor_paths(
    data_dir: str | Path,
    split: str,
    embedding_template: str = "{split}_emb.pt",
    labels_template: str = "{split}_labels.pt",
    ids_template: str = "{split}_ids.pt",
    texts_template: str = "{split}_texts.pt",
) -> dict[str, Path]:
    base_dir = Path(data_dir)
    return {
        "embedding": base_dir / embedding_template.format(split=split),
        "labels": base_dir / labels_template.format(split=split),
        "ids": base_dir / ids_template.format(split=split),
        "texts": base_dir / texts_template.format(split=split),
    }


def load_split_dataset(
    data_dir: str | Path,
    split: str,
    embedding_template: str = "{split}_emb.pt",
    labels_template: str = "{split}_labels.pt",
    ids_template: str = "{split}_ids.pt",
) -> TensorSplitDataset:
    paths = split_tensor_paths(
        data_dir=data_dir,
        split=split,
        embedding_template=embedding_template,
        labels_template=labels_template,
        ids_template=ids_template,
    )
    return TensorSplitDataset(
        embeddings_path=paths["embedding"],
        labels_path=paths["labels"],
        ids_path=paths["ids"],
    )


def load_split_texts(
    data_dir: str | Path,
    split: str,
    texts_template: str = "{split}_texts.pt",
) -> list[str]:
    paths = split_tensor_paths(
        data_dir=data_dir,
        split=split,
        texts_template=texts_template,
    )
    return torch.load(paths["texts"], map_location="cpu")


def build_dataloader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool | None = None,
    embedding_template: str = "{split}_emb.pt",
    labels_template: str = "{split}_labels.pt",
    ids_template: str = "{split}_ids.pt",
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"

    dataset = load_split_dataset(
        data_dir=data_dir,
        split=split,
        embedding_template=embedding_template,
        labels_template=labels_template,
        ids_template=ids_template,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
