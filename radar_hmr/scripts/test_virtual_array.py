from radar_hmr.signal_processing.adc_decoder import DCA1000ADCDecoder

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
# Virtual Array Formation
# -----------------------------------

virtual_array_former = VirtualArrayFormer()

virtual_array = virtual_array_former.form_virtual_array(
    doppler_tensor
)

print("\nFinal Virtual Array Shape:")
print(virtual_array.shape)

print("\nSingle Antenna Slice Shape:")
print(virtual_array[0, :, 0, :].shape)