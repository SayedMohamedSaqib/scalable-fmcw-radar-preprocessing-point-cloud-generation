import numpy as np

from radar_hmr.legacy.radar.configuration import (
    RANGE_RESOLUTION,
)


class PointCloudGenerator:
    """
    Convert CFAR detections into
    radar point cloud coordinates.
    """

    def __init__(
        self,
        num_angle_bins=64,
    ):
        self.num_angle_bins = num_angle_bins

    def angle_bin_to_theta(self, angle_bin):
        normalized = (
            (angle_bin / self.num_angle_bins) * 2
        ) - 1

        normalized = np.clip(
            normalized,
            -1,
            1,
        )

        theta = np.arcsin(normalized)

        return theta

    def generate(
        self,
        detections,
    ):
        points = []

        for det in detections:
            angle_bin, range_bin, power = det

            range_m = (
                range_bin
                * RANGE_RESOLUTION
            )

            theta = self.angle_bin_to_theta(
                angle_bin
            )

            x = range_m * np.sin(theta)

            y = range_m * np.cos(theta)

            points.append(
                [
                    x,
                    y,
                    power,
                ]
            )

        points = np.array(points)

        print(
            f"[INFO] Generated {len(points)} point cloud points"
        )

        return points