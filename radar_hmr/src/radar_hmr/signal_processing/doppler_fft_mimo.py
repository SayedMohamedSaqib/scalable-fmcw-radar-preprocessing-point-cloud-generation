import numpy as np


class MIMODopplerProcessor:
    """
    Doppler FFT for separated TDM-MIMO TX tensors.
    """

    def __init__(self, windowing=True):
        self.windowing = windowing

    def apply_window(self, tx_tensor):
        """
        Apply Doppler window across chirps.
        """

        window = np.hanning(tx_tensor.shape[2])

        window = window.reshape(
            1, 1, -1, 1, 1
        )

        return tx_tensor * window

    def compute_fft(self, tx_tensor):
        """
        Doppler FFT across chirps-per-TX dimension.
        """

        doppler_fft = np.fft.fft(
            tx_tensor,
            axis=2
        )

        doppler_fft = np.fft.fftshift(
            doppler_fft,
            axes=2
        )

        return doppler_fft

    def compute_magnitude(self, doppler_fft):

        return np.abs(doppler_fft)

    def process(self, tx_tensor):

        print("[INFO] Starting MIMO Doppler FFT")

        if self.windowing:

            tx_tensor = self.apply_window(
                tx_tensor
            )

            print("[INFO] Applied Doppler window")

        doppler_fft = self.compute_fft(
            tx_tensor
        )

        magnitude = self.compute_magnitude(
            doppler_fft
        )

        print("[INFO] MIMO Doppler FFT complete")
        print(f"[INFO] Output shape: {magnitude.shape}")

        return magnitude