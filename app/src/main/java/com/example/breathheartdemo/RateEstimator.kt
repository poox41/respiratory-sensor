package com.example.breathheartdemo

import kotlin.math.max
import kotlin.math.min

class PeakRateEstimator(
    private val fsHz: Int,
    private val refractoryMs: Long,
    private val windowSec: Int,
    private val minBpm: Float = 40f,
    private val maxBpm: Float = 180f
) {
    private val values = RingBuffer(fsHz * windowSec)

    fun add(tMs: Long, v: Float) {
        values.add(tMs, v)
    }

   fun estimate(): Float? {
        val (_, vs) = values.snapshot()
        val n = vs.size
        if (n < fsHz) return null

        var mean = 0f
        for (x in vs) mean += x
        mean /= n

        // Precompute Hann window
        val hann = FloatArray(n) { i -> 0.5f * (1f - kotlin.math.cos(2.0 * Math.PI * i / (n - 1)).toFloat()) }
        var winSumSq = 0f
        for (w in hann) winSumSq += w * w
        val winNorm = n / winSumSq  // normalization factor to compensate for window energy loss

        val minHz = max(0.1f, minBpm / 60f)
        val maxHz = max(minHz, maxBpm / 60f)

        val kMin = max(1, (minHz * n / fsHz).toInt())
        val kMax = min(n / 2, (maxHz * n / fsHz).toInt())
        if (kMax <= kMin) return null

        var bestK = -1
        var bestMag = 0.0
        val twoPi = 2.0 * Math.PI

        for (k in kMin..kMax) {
            val w = twoPi * k / n
            var re = 0.0
            var im = 0.0
            for (i in 0 until n) {
                val x = (vs[i] - mean).toDouble()
                val ang = w * i
                re += x * hann[i] * kotlin.math.cos(ang)
                im -= x * hann[i] * kotlin.math.sin(ang)
            }
            val mag = (re * re + im * im) * winNorm
            if (mag > bestMag) {
                bestMag = mag
                bestK = k
            }
        }

        if (bestK < 0) return null
        val freq = bestK.toFloat() * fsHz / n.toFloat()
        return freq * 60f
    }

    fun clear() {
        values.clear()
    }
}

class RateSmoother(
    private val historySize: Int,
    private val alpha: Float,
    private val maxStepPerUpdate: Float
) {
    private val history = ArrayDeque<Float>()
    private var current: Float? = null

    fun update(raw: Float?): Float? {
        if (raw == null) return current

        history.addLast(raw)
        while (history.size > historySize) {
            history.removeFirst()
        }

        val target = median(history)
        val previous = current
        if (previous == null) {
            current = target
            return current
        }

        val blended = previous + (target - previous) * alpha
        val delta = (blended - previous).coerceIn(-maxStepPerUpdate, maxStepPerUpdate)
        current = previous + delta
        return current
    }

    fun clear() {
        history.clear()
        current = null
    }

    private fun median(values: Collection<Float>): Float {
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[mid - 1] + sorted[mid]) / 2f
        } else {
            sorted[mid]
        }
    }
}

class RespRateGate(
    private val minAmplitude: Float,
    private val maxJumpRpm: Float,
    private val maxHoldMs: Long
) {
    private var lastAcceptedRaw: Float? = null
    private var lastAcceptedAtMs: Long = 0L

    fun filter(
        nowMs: Long,
        rawRate: Float?,
        signalAmplitude: Float,
        hasEnoughWaveform: Boolean,
        currentDisplayed: Float?
    ): Float? {
        if (!hasEnoughWaveform || signalAmplitude < minAmplitude || rawRate == null) {
            return if (currentDisplayed != null && nowMs - lastAcceptedAtMs <= maxHoldMs) {
                currentDisplayed
            } else {
                null
            }
        }

        val previousRaw = lastAcceptedRaw
        if (previousRaw != null && kotlin.math.abs(rawRate - previousRaw) > maxJumpRpm) {
            return if (currentDisplayed != null && nowMs - lastAcceptedAtMs <= maxHoldMs) {
                currentDisplayed
            } else {
                null
            }
        }

        lastAcceptedRaw = rawRate
        lastAcceptedAtMs = nowMs
        return rawRate
    }

    fun clear() {
        lastAcceptedRaw = null
        lastAcceptedAtMs = 0L
    }
}
