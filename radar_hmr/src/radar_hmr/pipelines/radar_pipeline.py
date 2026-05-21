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

from radar_hmr.pipelines.point_cloud_generator import (
    PointCloudGenerator
)


class RadarPipeline:
    """
    End-to-end FMCW radar pipeline.
    """

    def __init__(
        self,
        cfar_threshold=4,
    ):

        self.decoder = DCA1000ADCDecoder()

        self.range_processor = RangeFFTProcessor()

        self.tdm_processor = TDMMIMOProcessor()

        self.doppler_processor = (
            MIMODopplerProcessor()
        )

        self.virtual_array_former = (
            VirtualArrayFormer()
        )

        self.angle_processor = (
            AngleFFTProcessor()
        )

        self.cfar = CFARDetector(
            threshold_scale=cfar_threshold
        )

        self.pc_generator = (
            PointCloudGenerator()
        )

    def process(
        self,
        bin_file,
        frame_idx=0,
        doppler_idx=30,
    ):

        print(
            "\n========== RADAR PIPELINE START =========="
        )

        # ---------------------------------
        # ADC Decode
        # ---------------------------------

        adc_tensor = self.decoder.decode(
            bin_file
        )

        # ---------------------------------
        # Range FFT
        # ---------------------------------

        range_spectrum = (
            self.range_processor.process(
                adc_tensor
            )
        )

        # ---------------------------------
        # TDM MIMO
        # ---------------------------------

        tx_tensor = (
            self.tdm_processor.separate_tx(
                range_spectrum
            )
        )

        # ---------------------------------
        # Doppler FFT
        # ---------------------------------

        doppler_tensor = (
            self.doppler_processor.process(
                tx_tensor
            )
        )

        # ---------------------------------
        # Virtual Array
        # ---------------------------------

        virtual_array = (
            self.virtual_array_former.form_virtual_array(
                doppler_tensor
            )
        )

        # ---------------------------------
        # Angle FFT
        # ---------------------------------

        angle_tensor = (
            self.angle_processor.process(
                virtual_array
            )
        )

        # ---------------------------------
        # Select RA Slice
        # ---------------------------------

        ra_map = angle_tensor[
            frame_idx,
            doppler_idx,
            :,
            :
        ]

        # ---------------------------------
        # CFAR
        # ---------------------------------

        detections = self.cfar.detect(
            ra_map
        )

        # ---------------------------------
        # Point Cloud
        # ---------------------------------

        points = self.pc_generator.generate(
            detections
        )

        print(
            "=========== PIPELINE COMPLETE ===========\n"
        )

        return {
            "adc_tensor": adc_tensor,
            "range_spectrum": range_spectrum,
            "doppler_tensor": doppler_tensor,
            "angle_tensor": angle_tensor,
            "detections": detections,
            "points": points,
        }