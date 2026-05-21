from radar_hmr.signal_processing.adc_decoder import DCA1000ADCDecoder
from radar_hmr.signal_processing.range_fft import RangeFFTProcessor
from radar_hmr.signal_processing.tdm_mimo import TDMMIMOProcessor
from radar_hmr.signal_processing.doppler_fft_mimo import (
    MIMODopplerProcessor
)

from radar_hmr.visualization.range_doppler_vis import (
    RangeDopplerVisualizer
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
# MIMO Doppler FFT
# -----------------------------------

doppler_processor = MIMODopplerProcessor()

doppler_tensor = doppler_processor.process(
    tx_tensor
)

# -----------------------------------
# Visualization
# -----------------------------------

visualizer = RangeDopplerVisualizer()

visualizer.plot(
    doppler_tensor[:, 0, :, :, :],
    frame_idx=0,
    rx_idx=0,
)