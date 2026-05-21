import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
)

from radar_hmr.models.pointnet_classifier import (
    PointNetClassifier
)

# -----------------------------------
# CONFIG
# -----------------------------------

DATASET_DIR = (
    r"D:\Study_Resources\DRDO\Gesture_Recognition\processed\pointnet"
)

BATCH_SIZE = 8

EPOCHS = 30

LR = 1e-3

# -----------------------------------
# Load Dataset
# -----------------------------------

X = np.load(
    os.path.join(
        DATASET_DIR,
        "X_pointnet.npy"
    )
)

y = np.load(
    os.path.join(
        DATASET_DIR,
        "y_pointnet.npy"
    )
)

print(
    f"[INFO] X shape: {X.shape}"
)

print(
    f"[INFO] y shape: {y.shape}"
)

# -----------------------------------
# Convert to Torch
# -----------------------------------

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.long
)

dataset = TensorDataset(
    X,
    y
)

# -----------------------------------
# Train/Test Split
# -----------------------------------

train_size = int(
    0.8 * len(dataset)
)

test_size = (
    len(dataset)
    - train_size
)

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

# -----------------------------------
# Model
# -----------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"[INFO] Using device: {device}"
)

model = PointNetClassifier(
    num_classes=3
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# -----------------------------------
# Training
# -----------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            preds == batch_y
        ).sum().item()

        total += batch_y.size(0)

    train_acc = (
        correct / total
    ) * 100

    # -----------------------------------
    # Evaluation
    # -----------------------------------

    model.eval()

    test_correct = 0

    test_total = 0

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(device)

            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            test_correct += (
                preds == batch_y
            ).sum().item()

            test_total += batch_y.size(0)

    test_acc = (
        test_correct / test_total
    ) * 100

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Test Acc: {test_acc:.2f}%"
    )