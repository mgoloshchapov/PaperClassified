from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.utils import ARXIV_CATEGORIES, resolve_portable_path

DEFAULT_TEXT = (
    "Gamma-ray lines constraints in the NMSSM We present the computation of the "
    "loop-induced self-annihilation of dark matter particles into two photons in "
    "the framework of the NMSSM. This process is a theoretically clean observable "
    'with a "smoking-gun" signature but is experimentally very challenging to detect. '
    "The rates were computed with the help of the SloopS program, an automatic code "
    "initially designed for the evaluation of processes at the one-loop level in the "
    "MSSM. We focused on a light neutralino scenario and discuss how the signal can "
    "be enhanced in the NMSSM with respect to the MSSM and then compared with the "
    "present limits given by the dedicated search of the FERMI-LAT satellite on the "
    "monochromatic gamma lines."
)

TOP_K = 3
METADATA_PATH = Path("models/best.metadata.json")


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or DEFAULT_TEXT
    text = " ".join(text.split())

    metadata = json.loads(METADATA_PATH.read_text())
    metadata_dir = METADATA_PATH.parent
    tokenizer_source = metadata.get("tokenizer_dir")
    if tokenizer_source:
        tokenizer_source = resolve_portable_path(tokenizer_source, metadata_dir)
    else:
        tokenizer_source = metadata["model_name"]
    onnx_path = resolve_portable_path(metadata["onnx_path"], metadata_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=int(metadata["max_length"]),
        return_tensors="np",
    )
    if "token_type_ids" not in inputs:
        inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs["token_type_ids"].astype(np.int64),
    }
    logits = session.run(["logits"], ort_inputs)[0][0]
    top_indices = np.argsort(logits)[::-1][:TOP_K]

    for index in top_indices:
        label = metadata["labels"][int(index)]
        print(f"{label}: {ARXIV_CATEGORIES[label]}")


if __name__ == "__main__":
    main()
