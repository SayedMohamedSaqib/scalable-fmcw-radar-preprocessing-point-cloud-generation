import os
import glob

from radar_hmr.datasets.labels import (
    LABELS
)


class BaseDatasetBuilder:

    def __init__(
        self,
        dataset_root,
    ):

        self.dataset_root = dataset_root

    def get_gesture_files(self):

        gesture_files = {}

        for gesture in LABELS.keys():

            gesture_path = os.path.join(
                self.dataset_root,
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

            gesture_files[
                gesture
            ] = bin_files

        return gesture_files