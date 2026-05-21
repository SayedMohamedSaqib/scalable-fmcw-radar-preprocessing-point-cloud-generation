import numpy as np


class CFARDetector:
    """
    Simple 2D CA-CFAR detector.
    """

    def __init__(
        self,
        guard_cells=2,
        training_cells=6,
        threshold_scale=3.0,
    ):

        self.guard_cells = guard_cells

        self.training_cells = training_cells

        self.threshold_scale = threshold_scale

    def detect(self, ra_map):

        detections = []

        rows, cols = ra_map.shape

        g = self.guard_cells

        t = self.training_cells

        for r in range(
            t + g,
            rows - (t + g)
        ):

            for c in range(
                t + g,
                cols - (t + g)
            ):

                row_start = r - (t + g)
                row_end = r + t + g + 1

                col_start = c - (t + g)
                col_end = c + t + g + 1

                window = ra_map[
                    row_start:row_end,
                    col_start:col_end
                ]

                guard_start_r = t
                guard_end_r = t + 2 * g + 1

                guard_start_c = t
                guard_end_c = t + 2 * g + 1

                training_window = window.copy()

                training_window[
                    guard_start_r:guard_end_r,
                    guard_start_c:guard_end_c
                ] = 0

                noise_level = np.mean(
                    training_window
                )

                threshold = (
                    noise_level
                    * self.threshold_scale
                )

                cell_value = ra_map[r, c]

                if cell_value > threshold:

                    detections.append(
                        (r, c, cell_value)
                    )

        print(
            f"[INFO] CFAR detections: {len(detections)}"
        )

        return detections