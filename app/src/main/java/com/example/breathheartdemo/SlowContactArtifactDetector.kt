package com.example.breathheartdemo

import kotlin.math.sqrt

/**
 * Detect a slow contact-pressure change that is too smooth for the ordinary
 * point-to-point motion detector but still produces a large heart-band
 * transient after causal filtering.
 *
 * The decision deliberately requires evidence from two independent domains:
 * a large two-second range in raw ADC pressure and an unusual rise in the
 * two-second heart-band RMS relative to the preceding twenty seconds. Smooth
 * abdominal respiration alone therefore does not satisfy the complete gate.
 */
class SlowContactArtifactDetector(
    private val fsHz: Int,
    rawWindowSeconds: Int = 2,
    baselineSeconds: Int = 20,
    private val minimumRawRange: Float = 900f,
    private val minimumRmsRatio: Float = 2.5f,
    private val strongRawRange: Float = 600f,
    private val strongRmsRatio: Float = 3.2f,
    private val minimumHeartRms: Float = 15f
) {
    data class Observation(
        val ready: Boolean = false,
        val triggered: Boolean = false,
        val rawRange: Float = 0f,
        val heartRms: Float = 0f,
        val baselineRms: Float? = null,
        val rmsRatio: Float? = null
    )

    private val windowSize = (fsHz * rawWindowSeconds).coerceAtLeast(2)
    private val historySize = baselineSeconds.coerceAtLeast(3)
    private val rawWindow = FloatArray(windowSize)
    private val heartWindow = FloatArray(windowSize)
    private val rmsHistory = FloatArray(historySize)
    private var windowIndex = 0
    private var windowCount = 0
    private var historyIndex = 0
    private var historyCount = 0
    private var samplesSinceEvaluation = 0
    private var latest = Observation()

    fun next(raw: Float, heart: Float): Observation {
        rawWindow[windowIndex] = raw
        heartWindow[windowIndex] = heart
        windowIndex = (windowIndex + 1) % windowSize
        if (windowCount < windowSize) windowCount++
        samplesSinceEvaluation++

        if (windowCount < windowSize || samplesSinceEvaluation < fsHz) {
            return latest.copy(triggered = false)
        }
        samplesSinceEvaluation = 0

        var rawMinimum = Float.POSITIVE_INFINITY
        var rawMaximum = Float.NEGATIVE_INFINITY
        var heartSum = 0.0
        for (index in 0 until windowCount) {
            val rawValue = rawWindow[index]
            rawMinimum = minOf(rawMinimum, rawValue)
            rawMaximum = maxOf(rawMaximum, rawValue)
            heartSum += heartWindow[index]
        }
        val heartMean = heartSum / windowCount
        var heartEnergy = 0.0
        for (index in 0 until windowCount) {
            val centered = heartWindow[index] - heartMean
            heartEnergy += centered * centered
        }
        val heartRms = sqrt(heartEnergy / windowCount).toFloat()
        val rawRange = (rawMaximum - rawMinimum).coerceAtLeast(0f)

        if (historyCount < historySize) {
            addHistory(heartRms)
            latest = Observation(rawRange = rawRange, heartRms = heartRms)
            return latest
        }

        val sorted = rmsHistory.copyOf()
        sorted.sort()
        val baseline = sorted[sorted.size / 2].coerceAtLeast(5f)
        val ratio = heartRms / baseline
        val ordinaryJointBurst = rawRange >= minimumRawRange && ratio >= minimumRmsRatio
        val strongEnergyBurst = rawRange >= strongRawRange && ratio >= strongRmsRatio
        val triggered = heartRms >= minimumHeartRms &&
            (ordinaryJointBurst || strongEnergyBurst)

        // Never teach the baseline with a confirmed artifact. Median history is
        // updated only by windows that remain eligible as ordinary physiology.
        if (!triggered) addHistory(heartRms)
        latest = Observation(
            ready = true,
            triggered = triggered,
            rawRange = rawRange,
            heartRms = heartRms,
            baselineRms = baseline,
            rmsRatio = ratio
        )
        return latest
    }

    fun clear() {
        rawWindow.fill(0f)
        heartWindow.fill(0f)
        rmsHistory.fill(0f)
        windowIndex = 0
        windowCount = 0
        historyIndex = 0
        historyCount = 0
        samplesSinceEvaluation = 0
        latest = Observation()
    }

    private fun addHistory(value: Float) {
        rmsHistory[historyIndex] = value
        historyIndex = (historyIndex + 1) % historySize
        if (historyCount < historySize) historyCount++
    }
}
