import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from radar_hmr.pipelines.sequence_processor import (
    RadarSequenceProcessor
)

# -----------------------------------
# Dataset Root
# -----------------------------------

DATASET_ROOT = (
    r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1"
)

GESTURES = [
    "LS",
    "RS",
    "ST",
]

# -----------------------------------
# Radar Processor
# -----------------------------------

processor = RadarSequenceProcessor(
    cfar_threshold=4
)

# -----------------------------------
# Global Accumulation
# -----------------------------------

all_points_global = []

# -----------------------------------
# Iterate Gestures
# -----------------------------------

for gesture in GESTURES:

    gesture_path = os.path.join(
        DATASET_ROOT,
        gesture
    )

    bin_files = sorted(
        glob.glob(
            os.path.join(
                gesture_path,
                "*.bin"
            )
        )
    )

    print(
        f"\n========== {gesture} =========="
    )

    print(
        f"[INFO] Found {len(bin_files)} bin files"
    )

    # -----------------------------------
    # Process Each Bin File
    # -----------------------------------

    for idx, bin_file in enumerate(bin_files):

        print(
            f"\n[INFO] Processing file {idx+1}/{len(bin_files)}"
        )

        print(bin_file)

        try:

            sequence = (
                processor.process_sequence(
                    bin_file
                )
            )

            # -----------------------------------
            # Accumulate Points
            # -----------------------------------

            for frame_points in sequence:

                if len(frame_points) > 0:

                    all_points_global.append(
                        frame_points
                    )

        except Exception as e:

            print(
                f"[ERROR] Failed on file:"
            )

            print(bin_file)

            print(e)

# -----------------------------------
# Gesture-wise Visualization
# -----------------------------------

gesture_points = {}

for gesture in GESTURES:

    gesture_points[gesture] = []

# -----------------------------------
# Reprocess per gesture
# -----------------------------------

for gesture in GESTURES:

    gesture_path = os.path.join(
        DATASET_ROOT,
        gesture
    )

    bin_files = sorted(
        glob.glob(
            os.path.join(
                gesture_path,
                "*.bin"
            )
        )
    )

    print(
        f"\n[INFO] Accumulating gesture: {gesture}"
    )

    for bin_file in bin_files:

        try:

            sequence = (
                processor.process_sequence(
                    bin_file
                )
            )

            for frame_points in sequence:

                if len(frame_points) > 0:

                    gesture_points[
                        gesture
                    ].append(frame_points)

        except Exception as e:

            print(e)

# -----------------------------------
# Plot
# -----------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

for idx, gesture in enumerate(GESTURES):

    ax = axes[idx]

    if len(gesture_points[gesture]) == 0:

        continue

    points = np.concatenate(
        gesture_points[gesture],
        axis=0
    )

    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        cmap='jet',
        s=3
    )

    ax.set_title(
        f"{gesture} Gesture"
    )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    ax.grid(True)

    ax.axis('equal')

fig.colorbar(
    scatter,
    ax=axes.ravel().tolist(),
    label="Power"
)

plt.suptitle(
    "Gesture-wise Radar Point Clouds"
)

plt.tight_layout()

plt.show()