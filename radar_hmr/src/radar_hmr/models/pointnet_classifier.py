import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetClassifier(nn.Module):

    def __init__(
        self,
        num_classes=3,
    ):

        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        # x:
        # (B, N, 3)

        x = self.mlp1(x)

        # Global max pooling
        x = torch.max(
            x,
            dim=1
        )[0]

        x = self.classifier(x)

        return x