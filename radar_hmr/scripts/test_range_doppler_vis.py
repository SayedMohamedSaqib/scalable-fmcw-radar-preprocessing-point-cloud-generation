from radar_hmr.signal_processing.adc_decoder import DCA1000ADCDecoder
from radar_hmr.signal_processing.range_fft import RangeFFTProcessor
from radar_hmr.signal_processing.doppler_fft import DopplerFFTProcessor
from radar_hmr.visualization.range_doppler_vis import (
    RangeDopplerVisualizer
)

BIN_FILE = r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1\LS\adc_data_ls2.bin"

# -----------------------------------
# Decode ADC
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
# Doppler FFT
# -----------------------------------

doppler_processor = DopplerFFTProcessor()

doppler_spectrum = doppler_processor.process(
    range_spectrum
)

# -----------------------------------
# Visualization
# -----------------------------------

visualizer = RangeDopplerVisualizer()

visualizer.plot(
    doppler_spectrum,
    frame_idx=0,
    rx_idx=0,
)