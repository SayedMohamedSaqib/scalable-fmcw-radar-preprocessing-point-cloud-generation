import numpy as np


class VirtualArrayFormer:
    """
    Form azimuth virtual array for AWR1843.
    """

    def __init__(self):
        pass

    def form_virtual_array(self, tx_tensor):
        """
        Use only TX0 and TX1
        for azimuth FFT.

        Input:
        (frames, tx, doppler, rx, range)

        Output:
        (frames, doppler, virtual_ant, range)
        """

        num_frames = tx_tensor.shape[0]

        num_doppler = tx_tensor.shape[2]

        num_range = tx_tensor.shape[4]

        # Use only TX0 + TX1
        azimuth_tx = [0, 1]

        num_virtual_ant = len(azimuth_tx) * 4

        virtual_array = np.zeros(
            (
                num_frames,
                num_doppler,
                num_virtual_ant,
                num_range,
            ),
            dtype=tx_tensor.dtype,
        )

        ant_idx = 0

        for tx in azimuth_tx:

            for rx in range(4):

                virtual_array[
                    :,
                    :,
                    ant_idx,
                    :
                ] = tx_tensor[
                    :,
                    tx,
                    :,
                    rx,
                    :
                ]

                ant_idx += 1

        print(
            "[INFO] AWR1843 azimuth virtual array formed"
        )

        print(
            f"[INFO] Output shape: {virtual_array.shape}"
        )

        return virtual_array