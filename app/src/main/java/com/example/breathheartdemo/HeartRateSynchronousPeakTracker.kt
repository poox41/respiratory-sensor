package com.example.breathheartdemo

import kotlin.math.abs

/**
 * Causal beat-boundary tracker driven by an accepted heart-rate trajectory.
 *
 * A local maximum is selected only inside a narrow window around the next
 * physically predicted beat. This prevents respiration peaks and secondary
 * mechanical peaks from creating an extra cycle. The selected timestamps are
 * used only to segment the display-only morphology waveform.
 */
class HeartRateSynchronousPeakTracker(
    private val fsHz: Int,
    private val minimumBpm: Float = 40f,
    private val maximumBpm: Float = 110f,
    private val searchHalfWidthFraction: Float = 0.18f,
    private val maximumPhaseCorrectionFraction: Float = 0.12f,
    private val envelopeAlpha: Float = 0.04f
) {
    private var previousPreviousValue = 0f
    private var previousValue = 0f
    private var previousTimeMs = 0L
    private var sampleCount = 0
    private var envelope = 0f
    private var lastPeakTimeMs = 0L
    private var smoothedIntervalMs = 0f
    private var candidateTimeMs = 0L
    private var candidateValue = Float.NEGATIVE_INFINITY
    private var lastSampleTimeMs = 0L

    fun next(value: Float, timeMs: Long, acceptedBpm: Float?): Long? {
        val expectedStepMs = (1_000L / fsHz).coerceAtLeast(1L)
        if (lastSampleTimeMs > 0L && timeMs - lastSampleTimeMs > expectedStepMs * 3L) {
            clear()
        }
        lastSampleTimeMs = timeMs

        envelope += (abs(value) - envelope) * envelopeAlpha
        val bpm = acceptedBpm?.takeIf { it.isFinite() && it in minimumBpm..maximumBpm }
        if (bpm == null) {
            clearPhase()
            rememberSample(value, timeMs)
            return null
        }
        val requestedInterval = 60_000f / bpm
        smoothedIntervalMs = if (smoothedIntervalMs <= 0f) {
            requestedInterval
        } else {
            val maximumChange = smoothedIntervalMs * 0.10f
            smoothedIntervalMs +
                (requestedInterval - smoothedIntervalMs).coerceIn(-maximumChange, maximumChange) * 0.25f
        }

        val isLocalExtremum = sampleCount >= 2 &&
            abs(previousValue) > abs(previousPreviousValue) && abs(previousValue) >= abs(value)
        var emittedPeak: Long? = null
        if (lastPeakTimeMs == 0L) {
            if (isLocalExtremum && abs(previousValue) > envelope * 0.65f) {
                lastPeakTimeMs = previousTimeMs
                emittedPeak = previousTimeMs
            }
        } else {
            val predictedTimeMs = lastPeakTimeMs + smoothedIntervalMs.toLong()
            val halfWidthMs = (smoothedIntervalMs * searchHalfWidthFraction).toLong()
            val windowStartMs = predictedTimeMs - halfWidthMs
            val windowEndMs = predictedTimeMs + halfWidthMs
            if (isLocalExtremum && previousTimeMs in windowStartMs..windowEndMs &&
                abs(previousValue) > candidateValue
            ) {
                candidateValue = abs(previousValue)
                candidateTimeMs = previousTimeMs
            }
            if (timeMs >= windowEndMs) {
                val selectedTimeMs = candidateTimeMs.takeIf { it > lastPeakTimeMs }
                    ?: predictedTimeMs
                val maximumCorrectionMs =
                    (smoothedIntervalMs * maximumPhaseCorrectionFraction).toLong()
                val correctionMs = (selectedTimeMs - predictedTimeMs)
                    .coerceIn(-maximumCorrectionMs, maximumCorrectionMs)
                val minimumIntervalMs = (60_000f / maximumBpm).toLong()
                val maximumIntervalMs = (60_000f / minimumBpm).toLong()
                lastPeakTimeMs = (predictedTimeMs + correctionMs).coerceIn(
                    lastPeakTimeMs + minimumIntervalMs,
                    lastPeakTimeMs + maximumIntervalMs
                )
                emittedPeak = lastPeakTimeMs
                candidateTimeMs = 0L
                candidateValue = Float.NEGATIVE_INFINITY
            }
        }
        rememberSample(value, timeMs)
        return emittedPeak
    }

    private fun rememberSample(value: Float, timeMs: Long) {
        previousPreviousValue = previousValue
        previousValue = value
        previousTimeMs = timeMs
        sampleCount++
    }

    private fun clearPhase() {
        lastPeakTimeMs = 0L
        smoothedIntervalMs = 0f
        candidateTimeMs = 0L
        candidateValue = Float.NEGATIVE_INFINITY
    }

    fun clear() {
        previousPreviousValue = 0f
        previousValue = 0f
        previousTimeMs = 0L
        sampleCount = 0
        envelope = 0f
        clearPhase()
        lastSampleTimeMs = 0L
    }
}
