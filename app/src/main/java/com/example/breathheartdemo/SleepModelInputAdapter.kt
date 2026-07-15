package com.example.breathheartdemo

import kotlin.math.abs

data class SleepModelWindow(
    val heart: FloatArray,
    val respiration: FloatArray,
    val timestampMs: Long
)

data class SleepModelInputError(
    val code: Int,
    val message: String,
    val timestampMs: Long?
)

sealed class SleepModelInputResult {
    data class Ready(val window: SleepModelWindow) : SleepModelInputResult()
    data class Invalid(val error: SleepModelInputError) : SleepModelInputResult()
}

/**
 * Read-only bridge from the existing signal-processing buffers to the sleep
 * model.  It never changes Processor output and never uses the template-
 * enhanced display waveform as a model input.
 */
class SleepModelInputAdapter(
    private val fsHz: Int,
    private val windowSeconds: Int = 330
) {
    private val requiredSamples = fsHz * windowSeconds
    private val expectedStepMs = (1_000L / fsHz).coerceAtLeast(1L)

    fun prepare(heartBuffer: RingBuffer, respirationBuffer: RingBuffer): SleepModelInputResult {
        val (heartTimes, heartValues) = heartBuffer.snapshot()
        val (respTimes, respValues) = respirationBuffer.snapshot()
        val timestampMs = minOf(
            heartTimes.lastOrNull() ?: Long.MAX_VALUE,
            respTimes.lastOrNull() ?: Long.MAX_VALUE
        ).takeUnless { it == Long.MAX_VALUE }

        val available = minOf(heartValues.size, respValues.size)
        if (available < requiredSamples) {
            val collectedSeconds = available / fsHz.toFloat()
            return SleepModelInputResult.Invalid(
                SleepModelInputError(
                    code = 1001,
                    message = "data not enough: %.1fs/%ds collected".format(
                        collectedSeconds,
                        windowSeconds
                    ),
                    timestampMs = timestampMs
                )
            )
        }

        val heartStart = heartValues.size - requiredSamples
        val respStart = respValues.size - requiredSamples
        val selectedHeartTimes = heartTimes.copyOfRange(heartStart, heartTimes.size)
        val selectedRespTimes = respTimes.copyOfRange(respStart, respTimes.size)
        val selectedHeart = heartValues.copyOfRange(heartStart, heartValues.size)
        val selectedResp = respValues.copyOfRange(respStart, respValues.size)

        for (i in 0 until requiredSamples) {
            if (abs(selectedHeartTimes[i] - selectedRespTimes[i]) > expectedStepMs) {
                return invalidQuality("heart/respiration timestamps are not aligned", timestampMs)
            }
            if (i > 0) {
                val heartStep = selectedHeartTimes[i] - selectedHeartTimes[i - 1]
                val respStep = selectedRespTimes[i] - selectedRespTimes[i - 1]
                if (heartStep <= 0L || respStep <= 0L ||
                    heartStep > expectedStepMs * 2L || respStep > expectedStepMs * 2L
                ) {
                    return invalidQuality("signal contains a sampling or motion gap", timestampMs)
                }
            }
        }

        if (!selectedHeart.all { it.isFinite() } || !selectedResp.all { it.isFinite() }) {
            return invalidQuality("signal contains NaN or infinite values", timestampMs)
        }
        if (peakToPeak(selectedHeart) < 1e-3f) {
            return invalidQuality("heart signal is flat", timestampMs)
        }
        if (peakToPeak(selectedResp) < 1e-3f) {
            return invalidQuality("respiration signal is flat", timestampMs)
        }

        return SleepModelInputResult.Ready(
            SleepModelWindow(
                heart = selectedHeart,
                respiration = selectedResp,
                timestampMs = minOf(selectedHeartTimes.last(), selectedRespTimes.last())
            )
        )
    }

    private fun invalidQuality(message: String, timestampMs: Long?): SleepModelInputResult.Invalid {
        return SleepModelInputResult.Invalid(
            SleepModelInputError(code = 1002, message = message, timestampMs = timestampMs)
        )
    }

    private fun peakToPeak(values: FloatArray): Float {
        var minimum = Float.POSITIVE_INFINITY
        var maximum = Float.NEGATIVE_INFINITY
        for (value in values) {
            if (value < minimum) minimum = value
            if (value > maximum) maximum = value
        }
        return if (minimum.isFinite() && maximum.isFinite()) maximum - minimum else 0f
    }
}
