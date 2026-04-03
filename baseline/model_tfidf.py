from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int | None,
    min_df: int,
    max_df: float,
    ngram_range: tuple[int, int],
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def transform_texts_to_tensor(
    vectorizer: TfidfVectorizer,
    texts: list[str],
) -> torch.Tensor:
    sparse_matrix = vectorizer.transform(texts)
    dense_array = sparse_matrix.toarray()
    return torch.tensor(dense_array, dtype=torch.float32)


def save_vectorizer(path: str | Path, vectorizer: TfidfVectorizer) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(vectorizer, handle)
