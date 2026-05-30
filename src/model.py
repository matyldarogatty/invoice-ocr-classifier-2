from typing import Optional

import torch
import torch.nn as nn
from config import NUM_CLASSES


def _conv_feature_block() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class InvoiceCNN(nn.Module):
    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()
        n_out = num_classes if num_classes is not None else NUM_CLASSES

        self.features = _conv_feature_block()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class InvoiceCNNWithLayout(nn.Module):
    """Late fusion: image CNN embedding + layout MLP -> classifier."""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        layout_dim: int = 10,
        layout_hidden: int = 32,
        image_embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_out = num_classes if num_classes is not None else NUM_CLASSES
        self.layout_dim = layout_dim
        self.image_embed_dim = image_embed_dim
        self.layout_hidden = layout_hidden

        self.features = _conv_feature_block()
        self.image_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, image_embed_dim),
            nn.ReLU(),
        )

        layout_layers: list[nn.Module] = [
            nn.Linear(layout_dim, layout_hidden),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layout_layers.append(nn.Dropout(dropout))
        self.layout_mlp = nn.Sequential(*layout_layers)

        self.classifier = nn.Linear(image_embed_dim + layout_hidden, n_out)

    def forward(self, image: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_head(self.features(image))
        lay_emb = self.layout_mlp(layout)
        fused = torch.cat([img_emb, lay_emb], dim=1)
        return self.classifier(fused)
