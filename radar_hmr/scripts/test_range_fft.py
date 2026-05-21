from radar_hmr.signal_processing.adc_decoder import DCA1000ADCDecoder
from radar_hmr.signal_processing.range_fft import RangeFFTProcessor

BIN_FILE = r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1\LS\adc_data_ls2.bin"

decoder = DCA1000ADCDecoder()

adc_tensor = decoder.decode(BIN_FILE)

fft_processor = RangeFFTProcessor()

range_spectrum = fft_processor.process(adc_tensor)

print("\nFinal Range FFT Shape:")
print(range_spectrum.shape)

print("\nSample Magnitudes:")
print(range_spectrum[0, 0, 0, :10])