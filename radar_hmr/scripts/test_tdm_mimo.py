from radar_hmr.signal_processing.adc_decoder import DCA1000ADCDecoder
from radar_hmr.signal_processing.range_fft import RangeFFTProcessor
from radar_hmr.signal_processing.tdm_mimo import TDMMIMOProcessor

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
# TDM MIMO Separation
# -----------------------------------

tdm_processor = TDMMIMOProcessor()

tx_tensor = tdm_processor.separate_tx(
    range_spectrum
)

print("\nFinal TX Tensor Shape:")
print(tx_tensor.shape)

print("\nTX0 Chirp Shape:")
print(tx_tensor[0, 0].shape)

print("\nTX1 Chirp Shape:")
print(tx_tensor[0, 1].shape)

print("\nTX2 Chirp Shape:")
print(tx_tensor[0, 2].shape)