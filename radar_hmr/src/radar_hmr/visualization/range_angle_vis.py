import numpy as np
import matplotlib.pyplot as plt


class RangeAngleVisualizer:
    """
    Visualize Range-Angle maps.
    """

    def __init__(self):
        pass

    def plot(
        self,
        angle_tensor,
        frame_idx=0,
        doppler_idx=30,
        save_path=None,
    ):
        """
        Plot Range-Angle heatmap.

        Shape:
        (frames, doppler, angle, range)
        """

        ra_map = angle_tensor[
            frame_idx,
            doppler_idx,
            :,
            :
        ]

        ra_map_db = 20 * np.log10(
            ra_map + 1e-6
        )

        plt.figure(figsize=(10, 6))

        plt.imshow(
            ra_map_db.T,
            aspect='auto',
            origin='lower',
            cmap='jet'
        )

        plt.title(
            f"Range-Angle Map | Frame {frame_idx} | Doppler {doppler_idx}"
        )

        plt.xlabel("Angle Bins")
        plt.ylabel("Range Bins")

        plt.colorbar(label="Power (dB)")

        if save_path:

            plt.savefig(save_path)

            print(f"[INFO] Saved plot to {save_path}")

        plt.show()