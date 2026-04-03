# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baseline.model_tfidf import (
    build_tfidf_vectorizer,
    save_vectorizer,
    transform_texts_to_tensor,
)
from src.data import load_split_texts
from src.utils import ensure_directory, load_json, make_portable_path, save_json


@hydra.main(config_path="../conf", config_name="prepare_tfidf", version_base=None)
def main(cfg: DictConfig) -> None:
    data_dir = ensure_directory(to_absolute_path(str(cfg.data.data_dir)))
    metadata_path = Path(to_absolute_path(str(cfg.data.metadata_path)))
    metadata = load_json(metadata_path)

    train_texts = load_split_texts(
        data_dir=data_dir,
        split="train",
        texts_template=str(cfg.data.texts_template),
    )
    vectorizer = build_tfidf_vectorizer(
        max_features=(
            None if cfg.tfidf.max_features is None else int(cfg.tfidf.max_features)
        ),
        min_df=int(cfg.tfidf.min_df),
        max_df=float(cfg.tfidf.max_df),
        ngram_range=(int(cfg.tfidf.ngram_range[0]), int(cfg.tfidf.ngram_range[1])),
    )
    vectorizer.fit(train_texts)

    for split in tqdm(
        ("train", "val", "test"), desc="Encoding TF-IDF splits", unit="split"
    ):
        texts = load_split_texts(
            data_dir=data_dir,
            split=split,
            texts_template=str(cfg.data.texts_template),
        )
        embeddings = transform_texts_to_tensor(vectorizer, texts)
        torch.save(
            embeddings, data_dir / str(cfg.data.embedding_template).format(split=split)
        )

    vectorizer_path = Path(to_absolute_path(str(cfg.tfidf.vectorizer_path)))
    save_vectorizer(vectorizer_path, vectorizer)

    metadata["tfidf"] = {
        "embedding_template": str(cfg.data.embedding_template),
        "vectorizer_path": make_portable_path(vectorizer_path, metadata_path.parent),
        "vocab_size": len(vectorizer.vocabulary_),
        "max_features": cfg.tfidf.max_features,
        "min_df": int(cfg.tfidf.min_df),
        "max_df": float(cfg.tfidf.max_df),
        "ngram_range": [int(cfg.tfidf.ngram_range[0]), int(cfg.tfidf.ngram_range[1])],
    }
    save_json(metadata_path, metadata)

    print(f"Saved TF-IDF embeddings to {data_dir}")


if __name__ == "__main__":
    main()
