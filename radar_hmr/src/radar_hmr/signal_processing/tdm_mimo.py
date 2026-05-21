import numpy as np


class TDMMIMOProcessor:
    """
    Organize TDM-MIMO chirps by transmitter.
    """

    def __init__(self, num_tx=3):
        self.num_tx = num_tx

    def separate_tx(self, radar_tensor):
        """
        Convert:

        (frames, total_chirps, rx, range_bins)

        into:

        (frames, tx, chirps_per_tx, rx, range_bins)
        """

        num_frames = radar_tensor.shape[0]

        total_chirps = radar_tensor.shape[1]

        num_rx = radar_tensor.shape[2]

        num_range_bins = radar_tensor.shape[3]

        chirps_per_tx = total_chirps // self.num_tx

        tx_tensor = np.zeros(
            (
                num_frames,
                self.num_tx,
                chirps_per_tx,
                num_rx,
                num_range_bins,
            ),
            dtype=radar_tensor.dtype,
        )

        for tx in range(self.num_tx):

            tx_tensor[:, tx, :, :, :] = radar_tensor[
                :,
                tx::self.num_tx,
                :,
                :
            ]

        print("[INFO] TDM-MIMO TX separation complete")

        print(f"[INFO] Output shape: {tx_tensor.shape}")

        return tx_tensor