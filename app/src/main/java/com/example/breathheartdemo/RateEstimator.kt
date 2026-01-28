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
        if (n < fsHz * 2) return null

        var mean = 0f
        for (x in vs) mean += x
        mean /= n

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
                re += x * kotlin.math.cos(ang)
                im -= x * kotlin.math.sin(ang)
            }
            val mag = re * re + im * im
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
