import numpy as np


class AngleFFTProcessor:
    """
    Perform angle FFT across
    virtual antenna dimension.
    """

    def __init__(
        self,
        num_angle_bins=64,
        windowing=True,
    ):
        self.num_angle_bins = num_angle_bins

        self.windowing = windowing

    def apply_window(self, virtual_array):

        window = np.hanning(
            virtual_array.shape[2]
        )

        window = window.reshape(
            1, 1, -1, 1
        )

        return virtual_array * window

    def compute_fft(self, virtual_array):

        angle_fft = np.fft.fft(
            virtual_array,
            n=self.num_angle_bins,
            axis=2
        )

        angle_fft = np.fft.fftshift(
            angle_fft,
            axes=2
        )

        return angle_fft

    def compute_magnitude(self, angle_fft):

        return np.abs(angle_fft)

    def process(self, virtual_array):

        print("[INFO] Starting Angle FFT")

        if self.windowing:

            virtual_array = self.apply_window(
                virtual_array
            )

            print("[INFO] Applied angle window")

        angle_fft = self.compute_fft(
            virtual_array
        )

        magnitude = self.compute_magnitude(
            angle_fft
        )

        print("[INFO] Angle FFT complete")

        print(
            f"[INFO] Output shape: {magnitude.shape}"
        )

        return magnitude