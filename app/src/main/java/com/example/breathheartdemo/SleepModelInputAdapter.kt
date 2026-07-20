package com.example.breathheartdemo

import kotlin.math.abs

data class SleepModelWindow(
    val heart: FloatArray,
    val respiration: FloatArray,
    val timestampMs: Long,
    val imputedFraction: Float,
    val maxGapMs: Long
)

data class SleepModelInputError(
    val code: Int,
    val message: String,
    val timestampMs: Long?
)

data class SleepModelInputProgress(
    val collectedSamples: Int,
    val requiredSamples: Int,
    val fsHz: Int,
    val timestampMs: Long?,
    val continuous: Boolean,
    val imputedSamples: Int = 0,
    val maxGapMs: Long = 0L,
    val qualityAcceptable: Boolean = true
) {
    val collectedSeconds: Float
        get() = collectedSamples / fsHz.toFloat()
    val requiredSeconds: Int
        get() = requiredSamples / fsHz
    val remainingSeconds: Float
        get() = ((requiredSamples - collectedSamples).coerceAtLeast(0)) / fsHz.toFloat()
    val fraction: Float
        get() = (collectedSamples.toFloat() / requiredSamples.coerceAtLeast(1)).coerceIn(0f, 1f)
    val ready: Boolean
        get() = collectedSamples >= requiredSamples && qualityAcceptable
}

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
    private val requiredSpanMs = (requiredSamples - 1L) * expectedStepMs
    private val maxInterpolatedGapMs = 3_500L
    private val maxImputedFraction = 0.05f

    fun progress(heartBuffer: RingBuffer, respirationBuffer: RingBuffer): SleepModelInputProgress {
        val heartTimes = heartBuffer.timestampsSnapshot()
        val respTimes = respirationBuffer.timestampsSnapshot()
        return inspectProgress(heartTimes, respTimes)
    }

    fun prepare(heartBuffer: RingBuffer, respirationBuffer: RingBuffer): SleepModelInputResult {
        val (heartTimes, heartValues) = heartBuffer.snapshot()
        val (respTimes, respValues) = respirationBuffer.snapshot()
        val progress = inspectProgress(heartTimes, respTimes)
        val timestampMs = progress.timestampMs

        if (!progress.ready) {
            if (progress.collectedSamples >= requiredSamples && !progress.qualityAcceptable) {
                return invalidQuality(
                    "motion gap exceeds tolerance: max ${progress.maxGapMs}ms, " +
                        "imputed ${"%.1f".format(progress.imputedSamples * 100f / requiredSamples)}%",
                    timestampMs
                )
            }
            return SleepModelInputResult.Invalid(
                SleepModelInputError(
                    code = 1001,
                    message = "continuous data not enough: %.1fs/%ds collected".format(
                        progress.collectedSeconds,
                        progress.requiredSeconds
                    ),
                    timestampMs = timestampMs
                )
            )
        }

        val endMs = timestampMs ?: return invalidQuality("signal has no timestamp", null)
        val startMs = endMs - requiredSpanMs
        val selectedHeart = reconstruct(heartTimes, heartValues, startMs, endMs)
            ?: return invalidQuality("heart signal does not cover the model window", timestampMs)
        val selectedResp = reconstruct(respTimes, respValues, startMs, endMs)
            ?: return invalidQuality("respiration signal does not cover the model window", timestampMs)
        val imputedSamples = maxOf(selectedHeart.imputedSamples, selectedResp.imputedSamples)
        val maxGapMs = maxOf(selectedHeart.maxGapMs, selectedResp.maxGapMs)
        val imputedFraction = imputedSamples / requiredSamples.toFloat()
        if (maxGapMs > maxInterpolatedGapMs || imputedFraction > maxImputedFraction) {
            return invalidQuality(
                "motion gap exceeds tolerance: max ${maxGapMs}ms, " +
                    "imputed ${"%.1f".format(imputedFraction * 100f)}%",
                timestampMs
            )
        }

        if (!selectedHeart.values.all { it.isFinite() } || !selectedResp.values.all { it.isFinite() }) {
            return invalidQuality("signal contains NaN or infinite values", timestampMs)
        }
        if (peakToPeak(selectedHeart.values) < 1e-3f) {
            return invalidQuality("heart signal is flat", timestampMs)
        }
        if (peakToPeak(selectedResp.values) < 1e-3f) {
            return invalidQuality("respiration signal is flat", timestampMs)
        }

        return SleepModelInputResult.Ready(
            SleepModelWindow(
                heart = selectedHeart.values,
                respiration = selectedResp.values,
                timestampMs = endMs,
                imputedFraction = imputedFraction,
                maxGapMs = maxGapMs
            )
        )
    }

    /**
     * Measure elapsed coverage of the latest model window. Brief missing spans are
     * tolerated and later reconstructed, while long or frequent motion remains invalid.
     */
    private fun inspectProgress(
        heartTimes: LongArray,
        respTimes: LongArray
    ): SleepModelInputProgress {
        if (heartTimes.isEmpty() || respTimes.isEmpty()) {
            return SleepModelInputProgress(
                collectedSamples = 0,
                requiredSamples = requiredSamples,
                fsHz = fsHz,
                timestampMs = null,
                continuous = true,
                qualityAcceptable = true
            )
        }

        val timestampMs = minOf(
            heartTimes.last(),
            respTimes.last()
        ).takeUnless { it == Long.MAX_VALUE }
        val firstCommonMs = maxOf(heartTimes.first(), respTimes.first())
        val coverageMs = ((timestampMs ?: firstCommonMs) - firstCommonMs).coerceAtLeast(0L)
        val coverageSamples = ((coverageMs / expectedStepMs) + 1L)
            .coerceAtMost(requiredSamples.toLong())
            .toInt()
        val metrics = if (coverageSamples >= requiredSamples && timestampMs != null) {
            val startMs = timestampMs - requiredSpanMs
            combineMetrics(
                timelineMetrics(heartTimes, startMs, timestampMs),
                timelineMetrics(respTimes, startMs, timestampMs)
            )
        } else {
            combineMetrics(
                timelineMetrics(heartTimes, firstCommonMs, timestampMs ?: firstCommonMs),
                timelineMetrics(respTimes, firstCommonMs, timestampMs ?: firstCommonMs)
            )
        }
        val imputedFraction = metrics.imputedSamples / requiredSamples.toFloat()
        val qualityAcceptable = metrics.maxGapMs <= maxInterpolatedGapMs &&
            imputedFraction <= maxImputedFraction
        return SleepModelInputProgress(
            collectedSamples = coverageSamples,
            requiredSamples = requiredSamples,
            fsHz = fsHz,
            timestampMs = timestampMs,
            continuous = metrics.imputedSamples == 0,
            imputedSamples = metrics.imputedSamples,
            maxGapMs = metrics.maxGapMs,
            qualityAcceptable = qualityAcceptable
        )
    }

    private fun reconstruct(
        times: LongArray,
        values: FloatArray,
        startMs: Long,
        endMs: Long
    ): Reconstruction? {
        if (times.isEmpty() || values.size != times.size || times.first() > startMs || times.last() < endMs) {
            return null
        }
        val output = FloatArray(requiredSamples)
        var right = times.binarySearch(startMs).let { if (it >= 0) it else (-it - 1) }
        var imputed = 0
        var maxGapMs = 0L
        for (index in 0 until requiredSamples) {
            val targetMs = startMs + index * expectedStepMs
            while (right < times.size && times[right] < targetMs) right++
            if (right >= times.size) return null
            if (times[right] == targetMs) {
                output[index] = values[right]
                continue
            }
            val left = right - 1
            if (left < 0) return null
            val gapMs = times[right] - times[left]
            if (gapMs <= 0L) return null
            maxGapMs = maxOf(maxGapMs, gapMs)
            imputed++
            val fraction = (targetMs - times[left]).toFloat() / gapMs.toFloat()
            output[index] = values[left] + (values[right] - values[left]) * fraction
        }
        return Reconstruction(output, imputed, maxGapMs)
    }

    private fun timelineMetrics(times: LongArray, startMs: Long, endMs: Long): GapMetrics {
        if (times.size < 2 || endMs <= startMs) return GapMetrics(0, 0L)
        var index = times.binarySearch(startMs).let {
            (if (it >= 0) it else -it - 1).coerceAtLeast(1)
        }
        var missing = 0
        var maxGapMs = 0L
        while (index < times.size && times[index] <= endMs) {
            val previous = times[index - 1]
            val current = times[index]
            val clippedStart = maxOf(previous, startMs)
            val clippedEnd = minOf(current, endMs)
            val gapMs = clippedEnd - clippedStart
            if (gapMs > expectedStepMs * 2L) {
                maxGapMs = maxOf(maxGapMs, current - previous)
                missing += ((gapMs / expectedStepMs) - 1L).coerceAtLeast(0L).toInt()
            }
            index++
        }
        return GapMetrics(missing, maxGapMs)
    }

    private fun combineMetrics(first: GapMetrics, second: GapMetrics): GapMetrics = GapMetrics(
        imputedSamples = maxOf(first.imputedSamples, second.imputedSamples),
        maxGapMs = maxOf(first.maxGapMs, second.maxGapMs)
    )

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

    private data class Reconstruction(
        val values: FloatArray,
        val imputedSamples: Int,
        val maxGapMs: Long
    )

    private data class GapMetrics(
        val imputedSamples: Int,
        val maxGapMs: Long
    )
}
