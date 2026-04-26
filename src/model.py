from typing import Optional

import torch.nn as nn
from config import NUM_CLASSES


class InvoiceCNN(nn.Module):
    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()
        n_out = num_classes if num_classes is not None else NUM_CLASSES

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

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
