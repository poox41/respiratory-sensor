package com.example.breathheartdemo

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Converts overlapping block-VMD results into a sample-paced display stream.
 *
 * Consecutive windows overlap by windowLength - stepLength. The overlap is
 * used to resolve VMD's arbitrary IMF sign. Only the new tail is queued, so an
 * accepted 16 s result never replaces the visible chart. Normalization uses a
 * slowly changing robust scale and therefore retains the IMF's natural
 * beat-to-beat amplitude modulation.
 */
class VmdHeartbeatStreamPlayer(
    private val fsHz: Int,
    private val stepSeconds: Int,
    private val startupReserveMs: Long = 500L,
    private val scaleUpdateWeight: Float = 0.22f,
    private val emitFirstWindowTail: Boolean = false
) {
    private val stepSamples = fsHz * stepSeconds
    private val queue = ArrayDeque<Float>()
    private var previousCentered: FloatArray? = null
    private var displayScale: Float? = null
    private var reserveSamplesRemaining = 0
    private var hasQueuedStream = false

    /** Offer one complete VMD window. The first window establishes alignment. */
    @Synchronized
    fun update(values: FloatArray) {
        if (values.size <= stepSamples || values.any { !it.isFinite() }) return
        val centered = center(values)
        val previous = previousCentered
        if (previous != null) {
            val overlapCount = minOf(previous.size - stepSamples, centered.size - stepSamples)
            if (overlapCount > fsHz) {
                val correlation = correlation(
                    previous = previous,
                    previousStart = stepSamples,
                    current = centered,
                    currentStart = 0,
                    count = overlapCount
                )
                if (correlation < 0f) {
                    for (index in centered.indices) centered[index] = -centered[index]
                }
            }
        }

        // Use a high robust percentile for block-to-block scale alignment, but
        // never hard-clip the waveform. Hard clipping at +/-1.25 created long
        // flat tops that no later display-gain adjustment could recover.
        val currentScale = percentileAbsolute(centered, 0.99f).coerceAtLeast(1e-6f)
        displayScale = displayScale?.let { old ->
            old * (1f - scaleUpdateWeight) + currentScale * scaleUpdateWeight
        } ?: currentScale

        // A display stream may emit the clean tail of its first complete window
        // immediately. Later windows still contribute only their new tail.
        if (previous != null || emitFirstWindowTail) {
            val scale = displayScale ?: currentScale
            val start = (centered.size - stepSamples).coerceAtLeast(0)
            val tail = FloatArray(centered.size - start) { index ->
                centered[start + index] / scale
            }
            if (previous != null) softenSeam(tail)
            for (value in tail) queue.addLast(value)
            if (!hasQueuedStream) {
                hasQueuedStream = true
                reserveSamplesRemaining =
                    (startupReserveMs * fsHz / 1_000L).toInt().coerceAtLeast(0)
            }
        }
        previousCentered = centered
    }

    /** Return at most one VMD sample for each incoming sensor sample. */
    @Synchronized
    fun next(): Float? {
        if (!hasQueuedStream) return null
        if (reserveSamplesRemaining > 0) {
            reserveSamplesRemaining--
            return null
        }
        return if (queue.isEmpty()) null else queue.removeFirst()
    }

    @Synchronized
    fun clear() {
        queue.clear()
        previousCentered = null
        displayScale = null
        reserveSamplesRemaining = 0
        hasQueuedStream = false
    }

    private fun center(values: FloatArray): FloatArray {
        val mean = values.average().toFloat()
        return FloatArray(values.size) { index -> values[index] - mean }
    }

    private fun softenSeam(tail: FloatArray) {
        if (tail.isEmpty() || queue.isEmpty()) return
        val previousLast = queue.last()
        val correction = previousLast - tail.first()
        val blendSamples = minOf(fsHz / 5, tail.size).coerceAtLeast(1)
        for (index in 0 until blendSamples) {
            val remaining = 1f - index.toFloat() / blendSamples.toFloat()
            tail[index] += correction * remaining
        }
    }

    private fun percentileAbsolute(values: FloatArray, percentile: Float): Float {
        val magnitudes = FloatArray(values.size) { index -> abs(values[index]) }
        magnitudes.sort()
        val index = ((magnitudes.lastIndex) * percentile)
            .toInt()
            .coerceIn(0, magnitudes.lastIndex)
        return magnitudes[index]
    }

    private fun correlation(
        previous: FloatArray,
        previousStart: Int,
        current: FloatArray,
        currentStart: Int,
        count: Int
    ): Float {
        var product = 0.0
        var previousEnergy = 0.0
        var currentEnergy = 0.0
        for (offset in 0 until count) {
            val first = previous[previousStart + offset].toDouble()
            val second = current[currentStart + offset].toDouble()
            product += first * second
            previousEnergy += first * first
            currentEnergy += second * second
        }
        val denominator = sqrt(previousEnergy * currentEnergy)
        return if (denominator > 1e-12) (product / denominator).toFloat() else 0f
    }
}
