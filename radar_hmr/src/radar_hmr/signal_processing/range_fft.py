import numpy as np


class RangeFFTProcessor:
    """
    Performs range FFT on ADC radar tensor.
    """

    def __init__(self, windowing=True):
        self.windowing = windowing

    def apply_window(self, adc_tensor):
        """
        Apply Hanning window across ADC samples.
        """

        window = np.hanning(adc_tensor.shape[-1])

        return adc_tensor * window

    def compute_fft(self, adc_tensor):
        """
        Perform FFT along ADC sample dimension.
        """

        range_fft = np.fft.fft(adc_tensor, axis=-1)

        return range_fft

    def compute_magnitude(self, range_fft):
        """
        Compute magnitude spectrum.
        """

        magnitude = np.abs(range_fft)

        return magnitude

    def process(self, adc_tensor):
        """
        Full range FFT pipeline.
        """

        print("[INFO] Starting Range FFT")

        if self.windowing:
            adc_tensor = self.apply_window(adc_tensor)
            print("[INFO] Applied Hanning window")

        range_fft = self.compute_fft(adc_tensor)

        magnitude = self.compute_magnitude(range_fft)

        print("[INFO] Range FFT complete")
        print(f"[INFO] Output shape: {magnitude.shape}")

        return magnitude