import numpy as np


class DCA1000ADCDecoder:
    """
    Decoder for raw DCA1000 ADC captures.

    Supports:
    - xWR1642
    - TDM MIMO
    - complex ADC samples
    """

    def __init__(
        self,
        num_rx=4,
        num_tx=3,
        adc_samples=256,
        loops_per_frame=60,
    ):
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.adc_samples = adc_samples
        self.loops_per_frame = loops_per_frame

        # total chirps in one frame
        self.chirps_per_frame = self.num_tx * self.loops_per_frame

    def read_raw(self, file_path):
        """
        Read raw int16 ADC stream from DCA1000 binary.
        """

        raw = np.fromfile(file_path, dtype=np.int16)

        print(f"[INFO] Loaded raw ADC data")
        print(f"[INFO] Total int16 samples: {len(raw)}")

        return raw

    def iq_to_complex(self, raw_adc):
        """
        Convert interleaved IQ samples to complex numbers.

        Input format:
        I0 Q0 I1 Q1 I2 Q2 ...

        Output:
        complex64 array
        """

        i_data = raw_adc[0::2]
        q_data = raw_adc[1::2]

        complex_adc = i_data.astype(np.float32) + 1j * q_data.astype(np.float32)

        print(f"[INFO] Converted IQ -> complex")
        print(f"[INFO] Complex samples: {len(complex_adc)}")

        return complex_adc

    def organize_adc(self, complex_adc):
        """
        Organize ADC stream into radar tensor.

        Output shape:
        (num_frames, chirps_per_frame, num_rx, adc_samples)
        """

        samples_per_chirp = self.num_rx * self.adc_samples

        total_chirps = len(complex_adc) // samples_per_chirp

        num_frames = total_chirps // self.chirps_per_frame

        usable_chirps = num_frames * self.chirps_per_frame

        complex_adc = complex_adc[
            : usable_chirps * samples_per_chirp
        ]

        adc_tensor = complex_adc.reshape(
            num_frames,
            self.chirps_per_frame,
            self.num_rx,
            self.adc_samples,
        )

        print(f"[INFO] ADC tensor shape:")
        print(adc_tensor.shape)

        return adc_tensor

    def decode(self, file_path):
        """
        Full decoding pipeline.
        """

        raw_adc = self.read_raw(file_path)

        complex_adc = self.iq_to_complex(raw_adc)

        adc_tensor = self.organize_adc(complex_adc)

        return adc_tensor