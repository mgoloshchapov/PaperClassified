from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class PaperClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
        encoder_type: str = "hf",
        input_dim: int | None = None,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.encoder_type = encoder_type
        self.freeze_encoder = freeze_encoder
        if encoder_type == "hf":
            self.encoder = AutoModel.from_pretrained(model_name)
            self.hidden_size = int(self.encoder.config.hidden_size)
        elif encoder_type == "precomputed":
            if input_dim is None:
                raise ValueError("input_dim is required for precomputed embeddings")
            self.encoder = None
            self.hidden_size = int(input_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.classifier_config = {
            "num_layers": int(num_layers),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
        }
        self.classifier = self._build_classifier(
            num_labels=num_labels,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        if freeze_encoder and self.encoder is not None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def _build_classifier(
        self,
        num_labels: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers: list[nn.Module] = []
        input_dim = self.hidden_size

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ],
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_labels))
        return nn.Sequential(*layers)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder is None:
            raise ValueError("encode is only available for hf encoder_type")
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        embeddings: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings is None:
            if input_ids is None:
                raise ValueError("Either embeddings or input_ids must be provided")
            embeddings = self.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return self.classifier(embeddings)
