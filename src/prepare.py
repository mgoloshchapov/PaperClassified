# ruff: noqa: E402

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import hydra
import kagglehub
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import PaperRecord, select_deterministic_sample, split_records
from src.model import PaperClassifier
from src.utils import (
    INDEX_TO_LABEL,
    ensure_directory,
    make_portable_path,
    resolve_device,
    save_json,
    seed_everything,
)


def materialize_snapshot(source: Path, destination: Path) -> Path:
    ensure_directory(destination.parent)
    if destination.exists():
        return destination

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in tqdm(members, desc="Extracting archive", unit="files"):
            archive.extract(member, target_dir)


def resolve_snapshot_path(cfg: DictConfig) -> Path:
    data_dir = Path(to_absolute_path(str(cfg.data.data_dir)))
    raw_path = Path(to_absolute_path(str(cfg.data.raw_path)))
    if raw_path.exists():
        return raw_path

    if not cfg.download.enabled:
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    os.environ["KAGGLEHUB_CACHE"] = str(data_dir)
    download_target = Path(kagglehub.dataset_download(cfg.download.dataset))

    if raw_path.exists():
        return raw_path

    if download_target.is_file() and download_target.suffix in {".archive", ".zip"}:
        extract_archive(download_target, data_dir)
    elif download_target.is_dir():
        archive_candidates = list(download_target.rglob("*.archive")) + list(
            download_target.rglob("*.zip")
        )
        if archive_candidates and not raw_path.exists():
            extract_archive(archive_candidates[0], data_dir)

    if raw_path.exists():
        return raw_path

    candidates = list(data_dir.rglob(Path(cfg.data.raw_filename).name))
    if not candidates:
        raise FileNotFoundError(
            "Could not locate "
            f"{cfg.data.raw_filename} after download from {cfg.download.dataset}",
        )

    return materialize_snapshot(candidates[0], raw_path)


def validate_sample_size(cfg: DictConfig) -> int:
    split_total = sum(int(size) for size in cfg.data.split_sizes.values())
    if int(cfg.sample_size) != split_total:
        raise ValueError(
            f"sample_size={cfg.sample_size} must match total split size={split_total}",
        )
    return split_total


def encode_split(
    split_name: str,
    records: list[PaperRecord],
    tokenizer: AutoTokenizer,
    model: PaperClassifier,
    batch_size: int,
    device: torch.device,
    data_dir: Path,
    max_length: int,
) -> None:
    embeddings: list[torch.Tensor] = []
    texts = [record.text for record in records]
    labels = torch.stack([record.labels for record in records]).to(torch.float32)
    ids = [record.paper_id for record in records]

    model.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(records), batch_size),
            desc=f"Encoding {split_name}",
            leave=False,
        ):
            batch_records = records[start : start + batch_size]
            batch_texts = [record.text for record in batch_records]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batch_embeddings = model.encode(**encoded).cpu()
            embeddings.append(batch_embeddings)

    torch.save(torch.cat(embeddings, dim=0), data_dir / f"{split_name}_emb.pt")
    torch.save(texts, data_dir / f"{split_name}_texts.pt")
    torch.save(labels.cpu(), data_dir / f"{split_name}_labels.pt")
    torch.save(ids, data_dir / f"{split_name}_ids.pt")


@hydra.main(config_path="../conf", config_name="prepare", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    data_dir = ensure_directory(to_absolute_path(str(cfg.data.data_dir)))
    raw_path = resolve_snapshot_path(cfg)
    sample_size = validate_sample_size(cfg)

    records = select_deterministic_sample(
        raw_path=raw_path,
        sample_size=sample_size,
        seed=int(cfg.seed),
    )
    split_records_map = split_records(records, dict(cfg.data.split_sizes))

    if str(cfg.model.encoder_type) != "hf":
        raise ValueError(
            "src/prepare.py requires an hf encoder_type to build SciBERT embeddings"
        )

    device = resolve_device(cfg.device)
    model = PaperClassifier(
        model_name=cfg.model.name,
        num_labels=len(INDEX_TO_LABEL),
        num_layers=int(cfg.model.num_layers),
        hidden_dim=int(cfg.model.hidden_dim),
        dropout=float(cfg.model.dropout),
        encoder_type=str(cfg.model.encoder_type),
        freeze_encoder=bool(cfg.model.freeze_encoder),
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    for split_name, split_data in tqdm(
        split_records_map.items(),
        desc="Saving split caches",
        unit="split",
    ):
        encode_split(
            split_name=split_name,
            records=split_data,
            tokenizer=tokenizer,
            model=model,
            batch_size=int(cfg.preprocess_batch_size),
            device=device,
            data_dir=data_dir,
            max_length=int(cfg.model.max_length),
        )

    metadata_path = Path(to_absolute_path(str(cfg.data.metadata_path)))
    save_json(
        metadata_path,
        {
            "label_to_index": {label: idx for idx, label in enumerate(INDEX_TO_LABEL)},
            "index_to_label": INDEX_TO_LABEL,
            "model_name": cfg.model.name,
            "encoder_type": str(cfg.model.encoder_type),
            "freeze_encoder": bool(cfg.model.freeze_encoder),
            "max_length": int(cfg.model.max_length),
            "split_sizes": {
                key: int(value) for key, value in cfg.data.split_sizes.items()
            },
            "sample_size": sample_size,
            "seed": int(cfg.seed),
            "raw_path": make_portable_path(raw_path, metadata_path.parent),
        },
    )

    print(f"Prepared tensor caches in {data_dir}")


if __name__ == "__main__":
    main()
