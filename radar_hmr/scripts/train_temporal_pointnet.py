import os
import copy
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    Subset,
)

from sklearn.model_selection import (
    KFold
)

from radar_hmr.models.temporal_pointnet import (
    TemporalPointNet
)

# -----------------------------------
# CONFIG
# -----------------------------------

DATASET_DIR = (
    r"D:\Study_Resources\DRDO\Gesture_Recognition\processed\pointnet"
)

BATCH_SIZE = 4

EPOCHS = 40

LR = 1e-3

NUM_FOLDS = 5

PATIENCE = 8

# -----------------------------------
# Load Dataset
# -----------------------------------

X = np.load(
    os.path.join(
        DATASET_DIR,
        "X.npy"
    )
)

y = np.load(
    os.path.join(
        DATASET_DIR,
        "y.npy"
    )
)

print(
    f"[INFO] X shape: {X.shape}"
)

print(
    f"[INFO] y shape: {y.shape}"
)

# -----------------------------------
# Torch Dataset
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
# Device
# -----------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"[INFO] Using device: {device}"
)

# -----------------------------------
# K-Fold
# -----------------------------------

kf = KFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=42
)

fold_accuracies = []

# -----------------------------------
# Cross Validation Loop
# -----------------------------------

for fold, (train_idx, test_idx) in enumerate(
    kf.split(dataset)
):

    print(
        f"\n========== FOLD {fold+1}/{NUM_FOLDS} =========="
    )

    train_dataset = Subset(
        dataset,
        train_idx
    )

    test_dataset = Subset(
        dataset,
        test_idx
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

    model = TemporalPointNet(
        num_classes=3
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # -----------------------------------
    # Early Stopping
    # -----------------------------------

    best_test_acc = 0

    best_model_state = None

    patience_counter = 0

    # -----------------------------------
    # Training
    # -----------------------------------

    for epoch in range(EPOCHS):

        # -----------------------------------
        # Train
        # -----------------------------------

        model.train()

        train_loss = 0

        train_correct = 0

        train_total = 0

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

            train_loss += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            train_correct += (
                preds == batch_y
            ).sum().item()

            train_total += batch_y.size(0)

        train_acc = (
            train_correct
            / train_total
        ) * 100

        # -----------------------------------
        # Evaluate
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
            test_correct
            / test_total
        ) * 100

        print(
            f"Fold {fold+1} | "
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.2f}% | "
            f"Test: {test_acc:.2f}%"
        )

        # -----------------------------------
        # Early Stopping Logic
        # -----------------------------------

        if test_acc > best_test_acc:

            best_test_acc = test_acc

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "\n[INFO] Early stopping triggered"
            )

            break

    # -----------------------------------
    # Save Best Fold Result
    # -----------------------------------

    fold_accuracies.append(
        best_test_acc
    )

    print(
        f"\n[INFO] Best Fold Accuracy: "
        f"{best_test_acc:.2f}%"
    )

# -----------------------------------
# Final Results
# -----------------------------------

fold_accuracies = np.array(
    fold_accuracies
)

print(
    "\n========== FINAL RESULTS =========="
)

print(
    f"Fold Accuracies: "
    f"{fold_accuracies}"
)

print(
    f"Mean Accuracy: "
    f"{fold_accuracies.mean():.2f}%"
)

print(
    f"Std Accuracy: "
    f"{fold_accuracies.std():.2f}%"
)