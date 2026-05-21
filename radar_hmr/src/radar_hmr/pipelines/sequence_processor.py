import numpy as np

from radar_hmr.pipelines.radar_pipeline import (
    RadarPipeline
)


class RadarSequenceProcessor:
    """
    Process full radar recordings
    into temporal point-cloud sequences.
    """

    def __init__(
        self,
        cfar_threshold=4,
    ):

        self.pipeline = RadarPipeline(
            cfar_threshold=cfar_threshold
        )

    def process_sequence(
        self,
        bin_file,
        doppler_idx=30,
    ):

        print(
            "\n====== SEQUENCE PROCESSING START ======\n"
        )

        results = self.pipeline.process(
            bin_file=bin_file,
            frame_idx=0,
            doppler_idx=doppler_idx,
        )

        angle_tensor = results[
            "angle_tensor"
        ]

        num_frames = angle_tensor.shape[0]

        all_points = []

        for frame_idx in range(num_frames):

            print(
                f"[INFO] Processing frame {frame_idx+1}/{num_frames}"
            )

            ra_map = angle_tensor[
                frame_idx,
                doppler_idx,
                :,
                :
            ]

            detections = (
                self.pipeline.cfar.detect(
                    ra_map
                )
            )

            points = (
                self.pipeline.pc_generator.generate(
                    detections
                )
            )

            all_points.append(points)

        print(
            "\n====== SEQUENCE COMPLETE ======\n"
        )

        return all_points