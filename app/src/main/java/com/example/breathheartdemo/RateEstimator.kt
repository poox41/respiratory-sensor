package com.example.breathheartdemo

import kotlin.math.abs
import kotlin.math.max

class PeakRateEstimator(
    private val fsHz: Int,
    private val refractoryMs: Long,
    private val windowSec: Int
) {
    private val values = RingBuffer(fsHz * windowSec)

    fun add(tMs: Long, v: Float) {
        values.add(tMs, v)
    }

    fun estimate(): Float? {
        val (ts, vs) = values.snapshot()
        if (vs.size < fsHz * 2) return null

        var mean = 0f
        for (x in vs) mean += x
        mean /= vs.size

        var meanAbs = 0f
        for (x in vs) meanAbs += abs(x - mean)
        meanAbs /= vs.size

        val thr = mean + 0.8f * max(0.001f, meanAbs)

        val peaks = ArrayList<Long>()
        var lastAccepted = 0L

        for (i in 1 until vs.size - 1) {
            val a = vs[i - 1]
            val b = vs[i]
            val c = vs[i + 1]
            if (b > a && b > c && b > thr) {
                val t = ts[i]
                if (peaks.isEmpty() || (t - lastAccepted) >= refractoryMs) {
                    peaks.add(t)
                    lastAccepted = t
                }
            }
        }

        if (peaks.size < 2) return null

        var sumDt = 0f
        var cnt = 0
        for (i in 1 until peaks.size) {
            val dt = (peaks[i] - peaks[i - 1]).toFloat() / 1000f
            if (dt > 0.1f) {
                sumDt += dt
                cnt++
            }
        }
        if (cnt == 0) return null
        val avgPeriod = sumDt / cnt
        return 60f / avgPeriod
    }

    fun clear() {
        values.clear()
    }
}
