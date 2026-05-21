import numpy as np


class DopplerFFTProcessor:
    """
    Performs Doppler FFT on range FFT radar tensor.
    """

    def __init__(self, windowing=True):
        self.windowing = windowing

    def apply_window(self, range_fft_tensor):
        """
        Apply Hanning window across chirp dimension.
        """

        window = np.hanning(range_fft_tensor.shape[1])

        window = window.reshape(1, -1, 1, 1)

        return range_fft_tensor * window

    def compute_fft(self, range_fft_tensor):
        """
        Doppler FFT across chirps.
        """

        doppler_fft = np.fft.fft(range_fft_tensor, axis=1)

        doppler_fft = np.fft.fftshift(
            doppler_fft,
            axes=1
        )

        return doppler_fft

    def compute_magnitude(self, doppler_fft):
        """
        Magnitude spectrum.
        """

        return np.abs(doppler_fft)

    def process(self, range_fft_tensor):
        """
        Full Doppler FFT pipeline.
        """

        print("[INFO] Starting Doppler FFT")

        if self.windowing:
            range_fft_tensor = self.apply_window(
                range_fft_tensor
            )

            print("[INFO] Applied Doppler window")

        doppler_fft = self.compute_fft(
            range_fft_tensor
        )

        magnitude = self.compute_magnitude(
            doppler_fft
        )

        print("[INFO] Doppler FFT complete")
        print(f"[INFO] Output shape: {magnitude.shape}")

        return magnitude