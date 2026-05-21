from radar_hmr.signal_processing.adc_decoder import DCA1000ADCDecoder

BIN_FILE = r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1\LS\adc_data_ls2.bin"

decoder = DCA1000ADCDecoder()

adc_tensor = decoder.decode(BIN_FILE)

print("\nFinal ADC Tensor Shape:")
print(adc_tensor.shape)