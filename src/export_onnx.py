# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import onnxruntime as ort
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import PaperClassifier
from src.utils import (
    INDEX_TO_LABEL,
    ensure_directory,
    load_checkpoint,
    make_portable_path,
    resolve_device,
    save_json,
)

LOGGER = logging.getLogger(__name__)


class OnnxExportWrapper(nn.Module):
    def __init__(self, model: PaperClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder = self.model.encoder
        if encoder is None:
            raise ValueError("ONNX export wrapper requires an HF encoder")

        input_shape = tuple(input_ids.shape)
        embedding_output = encoder.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        extended_attention_mask = encoder.get_extended_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_shape,
            dtype=embedding_output.dtype,
        )
        encoder_outputs = encoder.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
        )
        cls_embedding = encoder_outputs.last_hidden_state[:, 0]
        return self.model.classifier(cls_embedding)


def resolve_export_path(path_value: str) -> Path:
    path = Path(to_absolute_path(path_value))
    ensure_directory(path.parent)
    return path


def resolve_optional_path(path_value: str | None, default_path: Path) -> Path:
    if path_value is None:
        return default_path
    path = Path(to_absolute_path(path_value))
    ensure_directory(path.parent)
    return path


def resolve_max_length(cfg: DictConfig, checkpoint: dict[str, Any]) -> int:
    if cfg.max_length is not None:
        return int(cfg.max_length)

    checkpoint_cfg = checkpoint.get("config", {})
    if isinstance(checkpoint_cfg, dict):
        model_cfg = checkpoint_cfg.get("model", {})
        if isinstance(model_cfg, dict) and model_cfg.get("max_length") is not None:
            return int(model_cfg["max_length"])

    return 384


def load_full_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PaperClassifier, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    encoder_type = str(checkpoint.get("encoder_type", "hf"))
    if encoder_type != "hf":
        raise ValueError(
            "ONNX export supports only full Hugging Face encoder checkpoints. "
            f"Received encoder_type={encoder_type}.",
        )

    classifier_config = checkpoint["classifier_config"]
    model = PaperClassifier(
        model_name=checkpoint["model_name"],
        num_labels=int(checkpoint["num_labels"]),
        num_layers=int(classifier_config["num_layers"]),
        hidden_dim=int(classifier_config["hidden_dim"]),
        dropout=float(classifier_config["dropout"]),
        encoder_type="hf",
        input_dim=None,
        freeze_encoder=bool(checkpoint["freeze_encoder"]),
    ).to(device)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    if model.encoder is not None and hasattr(
        model.encoder.config, "_attn_implementation"
    ):
        model.encoder.config._attn_implementation = "eager"
    model.eval()
    if model.encoder is not None:
        model.encoder.eval()
    return model, checkpoint


def build_dummy_inputs(
    tokenizer: AutoTokenizer,
    text: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    encoded = tokenizer(
        [text] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

    ort_inputs = {
        "input_ids": encoded["input_ids"].cpu().numpy(),
        "attention_mask": encoded["attention_mask"].cpu().numpy(),
        "token_type_ids": encoded["token_type_ids"].cpu().numpy(),
    }
    torch_inputs = {key: value.to(device) for key, value in encoded.items()}
    return torch_inputs, ort_inputs


def export_to_onnx(
    model: PaperClassifier,
    output_path: Path,
    inputs: dict[str, torch.Tensor],
    opset_version: int,
) -> None:
    wrapper = OnnxExportWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        ),
        output_path.as_posix(),
        export_params=True,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,
    )


def verify_export(
    model: PaperClassifier,
    output_path: Path,
    torch_inputs: dict[str, torch.Tensor],
    ort_inputs: dict[str, np.ndarray],
    verify_atol: float,
    verify_rtol: float,
) -> float:
    with torch.inference_mode():
        torch_logits = model(**torch_inputs).cpu().numpy()

    session = ort.InferenceSession(
        output_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_logits = session.run(["logits"], ort_inputs)[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    np.testing.assert_allclose(
        torch_logits,
        ort_logits,
        atol=verify_atol,
        rtol=verify_rtol,
    )
    return max_abs_diff


def export_metadata(
    path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    tokenizer_dir: Path | None,
    checkpoint: dict[str, Any],
    max_length: int,
    opset_version: int,
) -> None:
    base_dir = path.parent
    metadata = {
        "checkpoint_path": make_portable_path(checkpoint_path, base_dir),
        "onnx_path": make_portable_path(onnx_path, base_dir),
        "tokenizer_dir": (
            None
            if tokenizer_dir is None
            else make_portable_path(tokenizer_dir, base_dir)
        ),
        "model_name": checkpoint["model_name"],
        "encoder_type": checkpoint.get("encoder_type", "hf"),
        "num_labels": int(checkpoint["num_labels"]),
        "labels": INDEX_TO_LABEL,
        "label_to_index": {label: index for index, label in enumerate(INDEX_TO_LABEL)},
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_metrics": checkpoint["metrics"],
        "max_length": int(max_length),
        "opset_version": int(opset_version),
        "inputs": ["input_ids", "attention_mask", "token_type_ids"],
        "outputs": ["logits"],
    }
    save_json(path, metadata)


@hydra.main(config_path="../conf", config_name="export_onnx", version_base=None)
def main(cfg: DictConfig) -> None:
    device = resolve_device(str(cfg.device))
    checkpoint_path = Path(to_absolute_path(str(cfg.checkpoint_path)))
    output_path = resolve_export_path(str(cfg.output_path))
    metadata_path = resolve_optional_path(
        cfg.metadata_path,
        output_path.with_suffix(".metadata.json"),
    )
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    hydra_log_path = hydra_output_dir / f"{HydraConfig.get().job.name}.log"

    model, checkpoint = load_full_model_from_checkpoint(checkpoint_path, device)
    max_length = resolve_max_length(cfg, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
    tokenizer_dir = None
    if bool(cfg.save_tokenizer):
        tokenizer_dir = resolve_optional_path(
            cfg.tokenizer_dir,
            output_path.parent / f"{output_path.stem}_tokenizer",
        )
        tokenizer.save_pretrained(tokenizer_dir)

    torch_inputs, ort_inputs = build_dummy_inputs(
        tokenizer=tokenizer,
        text=str(cfg.example_text),
        batch_size=int(cfg.dummy_batch_size),
        max_length=max_length,
        device=device,
    )

    LOGGER.info("Export log: %s", hydra_log_path)
    LOGGER.info("Checkpoint path: %s", checkpoint_path)
    LOGGER.info("Output path: %s", output_path)
    LOGGER.info("Metadata path: %s", metadata_path)
    LOGGER.info("Tokenizer dir: %s", tokenizer_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Max length: %s", max_length)
    LOGGER.info("Opset version: %s", cfg.opset_version)

    export_to_onnx(
        model=model,
        output_path=output_path,
        inputs=torch_inputs,
        opset_version=int(cfg.opset_version),
    )

    if bool(cfg.verify):
        max_abs_diff = verify_export(
            model=model,
            output_path=output_path,
            torch_inputs=torch_inputs,
            ort_inputs=ort_inputs,
            verify_atol=float(cfg.verify_atol),
            verify_rtol=float(cfg.verify_rtol),
        )
        LOGGER.info("ONNX verification passed with max_abs_diff=%.6e", max_abs_diff)

    export_metadata(
        path=metadata_path,
        checkpoint_path=checkpoint_path,
        onnx_path=output_path,
        tokenizer_dir=tokenizer_dir,
        checkpoint=checkpoint,
        max_length=max_length,
        opset_version=int(cfg.opset_version),
    )
    LOGGER.info("Exported ONNX model to %s", output_path)


if __name__ == "__main__":
    main()
