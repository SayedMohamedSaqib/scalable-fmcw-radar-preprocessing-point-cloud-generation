import os
import numpy as np

from radar_hmr.datasets.pointcloud_sequence_builder import (
    PointCloudSequenceBuilder
)


class PointNetDatasetBuilder:

    def __init__(
        self,
        dataset_root,
        output_dir,
    ):

        self.sequence_builder = (
            PointCloudSequenceBuilder(
                dataset_root=dataset_root,
                output_dir=output_dir,
            )
        )

        self.output_dir = output_dir

    def build(self):

        self.sequence_builder.build()

        X = np.load(
            os.path.join(
                self.output_dir,
                "X.npy"
            )
        )

        y = np.load(
            os.path.join(
                self.output_dir,
                "y.npy"
            )
        )

        num_samples = X.shape[0]

        X = X.reshape(
            num_samples,
            -1,
            3
        )

        print(
            f"\n[INFO] PointNet X shape: "
            f"{X.shape}"
        )

        np.save(
            os.path.join(
                self.output_dir,
                "X_pointnet.npy"
            ),
            X
        )

        np.save(
            os.path.join(
                self.output_dir,
                "y_pointnet.npy"
            ),
            y
        )

        print(
            "\n[INFO] PointNet dataset saved"
        )