package com.example.breathheartdemo

/**
 * Integer-factor averaging decimator used at the audited BLE protocol boundary.
 * Every decoded wire reading remains in the raw export; only the Processor
 * input is reduced from two readings per interval to one 50 Hz sample.
 */
internal class SensorSampleDownsampler(
    private val factor: Int
) {
    private var sum = 0f
    private var count = 0

    init {
        require(factor >= 1) { "factor must be at least one" }
    }

    fun next(value: Float, timeMs: Long): Sample? {
        sum += value
        count++
        if (count < factor) return null

        val output = Sample(tMs = timeMs, x = sum / count.toFloat())
        sum = 0f
        count = 0
        return output
    }

    fun clear() {
        sum = 0f
        count = 0
    }
}
