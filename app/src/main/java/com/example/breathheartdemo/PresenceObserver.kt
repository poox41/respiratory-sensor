package com.example.breathheartdemo

import kotlin.math.sqrt

/**
 * Provisional person/contact estimator.
 *
 * Processor uses non-PRESENT states only as a conservative safety block for
 * formal rates, the enhanced template and model output. Raw/experimental
 * signals remain available because PRESENT is not yet a validated proof of a
 * person or of physiological signal quality.
 */
class PresenceObserver(
    private val fsHz: Int,
    private val windowSeconds: Int = 10,
    private val evaluationIntervalMs: Long = 1_000L,
    private val evidenceEvaluations: Int = 3,
    private val absentEnvelopeThreshold: Float = 13f,
    private val presentEnvelopeThreshold: Float = 16f,
    private val absentHeartStdThreshold: Float = 18f,
    private val presentHeartStdThreshold: Float = 21f
) {
    enum class State { PRESENT, ABSENT, UNCERTAIN }

    data class Observation(
        val state: State = State.UNCERTAIN,
        val ready: Boolean = false,
        val score: Float? = null,
        val heartEnvelope10s: Float? = null,
        val heartStd10s: Float? = null
    )

    private val capacity = fsHz * windowSeconds
    private val heartValues = FloatArray(capacity)
    private val envelopeValues = FloatArray(capacity)
    private var index = 0
    private var count = 0
    private var lastTimeMs = Long.MIN_VALUE
    private var lastEvaluationMs = Long.MIN_VALUE
    private var candidateState = State.UNCERTAIN
    private var candidateCount = 0
    private var current = Observation()

    fun next(cleanHeart: Float, cleanHeartEnvelope: Float, timeMs: Long): Observation {
        // Do not combine samples from opposite sides of a BLE or motion gap.
        if (lastTimeMs != Long.MIN_VALUE &&
            (timeMs <= lastTimeMs || timeMs - lastTimeMs > 100L)
        ) {
            resetWindow()
        }
        lastTimeMs = timeMs

        heartValues[index] = cleanHeart
        envelopeValues[index] = cleanHeartEnvelope
        index = (index + 1) % capacity
        if (count < capacity) count++

        if (count < capacity) return current
        if (lastEvaluationMs != Long.MIN_VALUE &&
            timeMs - lastEvaluationMs < evaluationIntervalMs
        ) return current
        lastEvaluationMs = timeMs

        val envelopeMedian = median(envelopeValues, count)
        val heartStd = standardDeviation(heartValues, count)
        val score = (
            normalized(envelopeMedian, absentEnvelopeThreshold, presentEnvelopeThreshold) +
                normalized(heartStd, absentHeartStdThreshold, presentHeartStdThreshold)
            ) / 2f

        val observedState = when {
            envelopeMedian >= presentEnvelopeThreshold &&
                heartStd >= presentHeartStdThreshold -> State.PRESENT
            envelopeMedian <= absentEnvelopeThreshold &&
                heartStd <= absentHeartStdThreshold -> State.ABSENT
            else -> State.UNCERTAIN
        }

        if (observedState == candidateState) {
            candidateCount++
        } else {
            candidateState = observedState
            candidateCount = 1
        }
        val stableState = if (candidateCount >= evidenceEvaluations) observedState else current.state
        current = Observation(
            state = stableState,
            ready = true,
            score = score,
            heartEnvelope10s = envelopeMedian,
            heartStd10s = heartStd
        )
        return current
    }

    fun reset() {
        lastTimeMs = Long.MIN_VALUE
        resetWindow()
    }

    private fun resetWindow() {
        index = 0
        count = 0
        lastEvaluationMs = Long.MIN_VALUE
        candidateState = State.UNCERTAIN
        candidateCount = 0
        current = Observation()
    }

    private fun median(values: FloatArray, size: Int): Float {
        val sorted = values.copyOf(size)
        sorted.sort()
        val middle = size / 2
        return if (size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2f else sorted[middle]
    }

    private fun standardDeviation(values: FloatArray, size: Int): Float {
        var sum = 0.0
        for (i in 0 until size) sum += values[i]
        val mean = sum / size
        var squaredError = 0.0
        for (i in 0 until size) {
            val error = values[i] - mean
            squaredError += error * error
        }
        return sqrt(squaredError / size).toFloat()
    }

    private fun normalized(value: Float, low: Float, high: Float): Float =
        ((value - low) / (high - low)).coerceIn(0f, 1f)
}
