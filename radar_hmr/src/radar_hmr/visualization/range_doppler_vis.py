import numpy as np
import matplotlib.pyplot as plt


class RangeDopplerVisualizer:
    """
    Visualize Range-Doppler heatmaps.
    """

    def __init__(self):
        pass

    def plot(
        self,
        doppler_tensor,
        frame_idx=0,
        rx_idx=0,
        save_path=None,
    ):
        """
        Plot Range-Doppler map.

        Shape:
        (frames, doppler_bins, rx, range_bins)
        """

        rd_map = doppler_tensor[
            frame_idx,
            :,
            rx_idx,
            :
        ]

        rd_map_db = 20 * np.log10(
            rd_map + 1e-6
        )

        plt.figure(figsize=(10, 6))

        plt.imshow(
            rd_map_db.T,
            aspect='auto',
            origin='lower',
            cmap='jet'
        )

        plt.title(
            f"Range-Doppler Map | Frame {frame_idx} | RX {rx_idx}"
        )

        plt.xlabel("Doppler Bins")
        plt.ylabel("Range Bins")

        plt.colorbar(label="Power (dB)")

        if save_path:
            plt.savefig(save_path)
            print(f"[INFO] Saved plot to {save_path}")

        plt.show()