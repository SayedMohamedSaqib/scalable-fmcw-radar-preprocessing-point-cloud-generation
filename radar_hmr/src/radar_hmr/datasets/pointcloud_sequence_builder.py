import os
import numpy as np

from radar_hmr.datasets.base_builder import (
    BaseDatasetBuilder
)

from radar_hmr.datasets.labels import (
    LABELS
)

from radar_hmr.pipelines.sequence_processor import (
    RadarSequenceProcessor
)


class PointCloudSequenceBuilder(
    BaseDatasetBuilder
):

    def __init__(
        self,
        dataset_root,
        output_dir,
        top_k_points=64,
        num_frames=32,
    ):

        super().__init__(
            dataset_root
        )

        self.output_dir = output_dir

        self.top_k_points = (
            top_k_points
        )

        self.num_frames = (
            num_frames
        )

        self.processor = (
            RadarSequenceProcessor(
                cfar_threshold=4
            )
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    def normalize_points(
        self,
        points,
    ):

        # -----------------------------------
        # Empty Frame
        # -----------------------------------

        if len(points) == 0:

            return np.zeros(
                (
                    self.top_k_points,
                    3
                ),
                dtype=np.float32
            )

        # -----------------------------------
        # Sort By Power
        # -----------------------------------

        idx = np.argsort(
            points[:, 2]
        )[::-1]

        points = points[idx]

        # -----------------------------------
        # Keep Top-K
        # -----------------------------------

        points = points[
            :self.top_k_points
        ]

        # -----------------------------------
        # Pad If Needed
        # -----------------------------------

        if len(points) < self.top_k_points:

            pad = np.zeros(
                (
                    self.top_k_points
                    - len(points),
                    3
                ),
                dtype=np.float32
            )

            points = np.vstack(
                [points, pad]
            )

        # -----------------------------------
        # Feature Normalization
        # -----------------------------------

        # Normalize x/y coordinates
        # Keeps geometry near [-1,1]

        points[:, 0] = (
            points[:, 0] / 2.0
        )

        points[:, 1] = (
            points[:, 1] / 2.0
        )

        # Log compression on power
        # Radar amplitudes are highly skewed

        points[:, 2] = np.log1p(
            points[:, 2]
        )

        # Normalize power to [0,1]

        max_power = np.max(
            points[:, 2]
        )

        if max_power > 0:

            points[:, 2] = (
                points[:, 2]
                / max_power
            )

        # -----------------------------------
        # Final Float32
        # -----------------------------------

        return points.astype(
            np.float32
        )

    def build(self):

        X = []

        y = []

        gesture_files = (
            self.get_gesture_files()
        )

        for gesture, files in (
            gesture_files.items()
        ):

            print(
                f"\n========== {gesture} =========="
            )

            for idx, bin_file in enumerate(files):

                print(
                    f"\n[INFO] Processing "
                    f"{idx+1}/{len(files)}"
                )

                print(bin_file)

                try:

                    sequence = (
                        self.processor.process_sequence(
                            bin_file
                        )
                    )

                    normalized_sequence = []

                    for frame_points in sequence:

                        normalized_points = (
                            self.normalize_points(
                                frame_points
                            )
                        )

                        normalized_sequence.append(
                            normalized_points
                        )

                    normalized_sequence = np.array(
                        normalized_sequence
                    )

                    # -----------------------------------
                    # Fixed Number Of Frames
                    # -----------------------------------

                    normalized_sequence = (
                        normalized_sequence[
                            :self.num_frames
                        ]
                    )

                    if len(
                        normalized_sequence
                    ) < self.num_frames:

                        missing = (
                            self.num_frames
                            - len(
                                normalized_sequence
                            )
                        )

                        padding = np.zeros(
                            (
                                missing,
                                self.top_k_points,
                                3
                            ),
                            dtype=np.float32
                        )

                        normalized_sequence = (
                            np.concatenate(
                                [
                                    normalized_sequence,
                                    padding
                                ],
                                axis=0
                            )
                        )

                    X.append(
                        normalized_sequence
                    )

                    y.append(
                        LABELS[gesture]
                    )

                except Exception as e:

                    print(
                        "[ERROR]"
                    )

                    print(e)

        # -----------------------------------
        # Final Arrays
        # -----------------------------------

        X = np.array(X)

        y = np.array(y)

        print(
            f"\n[INFO] Final X shape: {X.shape}"
        )

        print(
            f"[INFO] Final y shape: {y.shape}"
        )

        # -----------------------------------
        # Save
        # -----------------------------------

        np.save(
            os.path.join(
                self.output_dir,
                "X.npy"
            ),
            X
        )

        np.save(
            os.path.join(
                self.output_dir,
                "y.npy"
            ),
            y
        )

        print(
            "\n[INFO] Dataset saved"
        )