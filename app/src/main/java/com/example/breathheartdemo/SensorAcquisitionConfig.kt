package com.example.breathheartdemo

/**
 * One authoritative description of the acquisition/protocol boundary.
 *
 * [hardwareSampleRateHz] describes the sensor setting when it is known.
 * [decodedValueRateHz] is the expected number of complete decoded int16 values
 * arriving from BLE each second. [processingSampleRateHz] is the rate used by
 * every downstream filter, VMD window and sleep feature. Keeping these fields
 * separate prevents BLE callback frequency from being mistaken for a physical
 * sampling rate.
 */
data class SensorAcquisitionConfig(
    val hardwareSampleRateHz: Int?,
    val decodedValueRateHz: Int,
    val processingSampleRateHz: Int,
    val decodedValuesPerProcessorSample: Int,
    val evidence: String
) {
    init {
        require(hardwareSampleRateHz == null || hardwareSampleRateHz > 0)
        require(decodedValueRateHz > 0)
        require(processingSampleRateHz > 0)
        require(decodedValuesPerProcessorSample > 0)
        require(decodedValueRateHz == processingSampleRateHz * decodedValuesPerProcessorSample) {
            "decodedValueRateHz must equal processingSampleRateHz * decodedValuesPerProcessorSample"
        }
        require(evidence.isNotBlank())
    }

    val inputResamplingMode: String
        get() = if (decodedValuesPerProcessorSample == 1) {
            "direct_decoded_value_to_processor"
        } else {
            "adjacent_${decodedValuesPerProcessorSample}_value_mean_to_processor"
        }

    companion object {
        /**
         * Current profile inferred from the supplied input_timing logs and the
         * user-declared hardware setting. It is not presented as firmware-level
         * verification; a future hardware protocol can replace this one object.
         */
        val OFFLINE_INFERRED_PROFILE = SensorAcquisitionConfig(
            hardwareSampleRateHz = 50,
            decodedValueRateHz = 100,
            processingSampleRateHz = 50,
            decodedValuesPerProcessorSample = 2,
            evidence = "user_declared_hardware_rate_plus_offline_input_timing_inference"
        )
    }
}
