import numpy as np
import matplotlib.pyplot as plt

from radar_hmr.signal_processing.adc_decoder import (
    DCA1000ADCDecoder
)

from radar_hmr.signal_processing.range_fft import (
    RangeFFTProcessor
)

from radar_hmr.signal_processing.tdm_mimo import (
    TDMMIMOProcessor
)

from radar_hmr.signal_processing.doppler_fft_mimo import (
    MIMODopplerProcessor
)

from radar_hmr.signal_processing.virtual_array import (
    VirtualArrayFormer
)

from radar_hmr.signal_processing.angle_fft import (
    AngleFFTProcessor
)

from radar_hmr.signal_processing.cfar import (
    CFARDetector
)

BIN_FILE = r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1\LS\adc_data_ls2.bin"

# -----------------------------------
# ADC Decode
# -----------------------------------

decoder = DCA1000ADCDecoder()

adc_tensor = decoder.decode(BIN_FILE)

# -----------------------------------
# Range FFT
# -----------------------------------

range_processor = RangeFFTProcessor()

range_spectrum = range_processor.process(
    adc_tensor
)

# -----------------------------------
# TDM Separation
# -----------------------------------

tdm_processor = TDMMIMOProcessor()

tx_tensor = tdm_processor.separate_tx(
    range_spectrum
)

# -----------------------------------
# Doppler FFT
# -----------------------------------

doppler_processor = MIMODopplerProcessor()

doppler_tensor = doppler_processor.process(
    tx_tensor
)

# -----------------------------------
# Virtual Array
# -----------------------------------

virtual_array_former = VirtualArrayFormer()

virtual_array = virtual_array_former.form_virtual_array(
    doppler_tensor
)

# -----------------------------------
# Angle FFT
# -----------------------------------

angle_processor = AngleFFTProcessor()

angle_tensor = angle_processor.process(
    virtual_array
)

# -----------------------------------
# Select Range-Angle Slice
# -----------------------------------

ra_map = angle_tensor[
    0,
    30,
    :,
    :
]

# -----------------------------------
# CFAR Detection
# -----------------------------------

cfar = CFARDetector()

detections = cfar.detect(
    ra_map
)

# -----------------------------------
# Visualization
# -----------------------------------

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

for det in detections:

    angle_bin, range_bin, _ = det

    plt.scatter(
        angle_bin,
        range_bin,
        c='white',
        s=10
    )

plt.title(
    "CFAR Detections on Range-Angle Map"
)

plt.xlabel("Angle Bins")

plt.ylabel("Range Bins")

plt.colorbar(label="Power (dB)")

plt.show()