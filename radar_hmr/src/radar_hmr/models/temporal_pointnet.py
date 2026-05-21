import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),
        )

    def forward(self, x):

        # x:
        # (B,F,P,3)

        B, F, P, C = x.shape

        # Flatten frames
        x = x.reshape(
            B * F,
            P,
            C
        )

        # Point-wise MLP
        x = self.mlp(x)

        # Global max pool
        x = torch.max(
            x,
            dim=1
        )[0]

        # Restore frame dimension
        x = x.reshape(
            B,
            F,
            256
        )

        return x


class TemporalPointNet(nn.Module):

    def __init__(
        self,
        num_classes=3,
    ):

        super().__init__()

        self.encoder = (
            PointNetEncoder()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        # -----------------------------------
        # PointNet Frame Encoder
        # -----------------------------------

        x = self.encoder(x)

        # x:
        # (B,F,256)

        # -----------------------------------
        # Temporal Modeling
        # -----------------------------------

        x, _ = self.lstm(x)

        # Last timestep
        x = x[:, -1, :]

        # -----------------------------------
        # Classification
        # -----------------------------------

        x = self.classifier(x)

        return x