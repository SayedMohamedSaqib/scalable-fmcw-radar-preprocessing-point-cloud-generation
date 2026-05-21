from radar_hmr.signal_processing.adc_decoder import DCA1000ADCDecoder
from radar_hmr.signal_processing.range_fft import RangeFFTProcessor
from radar_hmr.signal_processing.doppler_fft import DopplerFFTProcessor

BIN_FILE = r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1\LS\adc_data_ls2.bin"

decoder = DCA1000ADCDecoder()

adc_tensor = decoder.decode(BIN_FILE)

range_processor = RangeFFTProcessor()

range_spectrum = range_processor.process(
    adc_tensor
)

doppler_processor = DopplerFFTProcessor()

doppler_spectrum = doppler_processor.process(
    range_spectrum
)

print("\nFinal Doppler FFT Shape:")
print(doppler_spectrum.shape)

print("\nSample Doppler Magnitudes:")
print(doppler_spectrum[0, :, 0, 10][:10])