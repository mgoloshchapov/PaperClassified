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


def parse_args(argv: list[str]) -> tuple[str, int]:
    top_k = TOP_K
    text_parts: list[str] = []
    index = 0

    while index < len(argv):
        argument = argv[index]
        if argument.startswith("--top_k=") or argument.startswith("--top-k="):
            top_k = int(argument.split("=", maxsplit=1)[1])
        elif argument in {"--top_k", "--top-k"}:
            index += 1
            if index >= len(argv):
                raise ValueError(f"Missing value for {argument}")
            top_k = int(argv[index])
        else:
            text_parts.append(argument)
        index += 1

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    text = " ".join(text_parts).strip() or DEFAULT_TEXT
    return text, top_k


def main() -> None:
    text, top_k = parse_args(sys.argv[1:])
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
    top_indices = np.argsort(logits)[::-1][:top_k]

    print(text)
    print("------------------------------")
    for index in top_indices:
        label = metadata["labels"][int(index)]
        print(f"{label}: {ARXIV_CATEGORIES[label]}")


if __name__ == "__main__":
    main()
