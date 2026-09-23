package com.example.breathheartdemo

import kotlin.math.abs
import kotlin.math.max

/**
 * Detects low ADC headroom and flat-topped waveforms before hard rail clipping.
 *
 * Hard clipping (<= 10 or >= 4085) remains the responsibility of [Processor].
 * This detector deliberately requires repeated evidence so isolated ADC spikes
 * do not invalidate a complete vital-sign window.
 */
class NearSaturationDetector(
    fsHz: Int,
    private val warningLow: Float = 95f,
    private val warningHigh: Float = 4000f,
    private val nearRailLow: Float = 32f,
    private val nearRailHigh: Float = 4063f,
    private val warningRunLength: Int = 4,
    private val nearRailHitsRequired: Int = 3,
    private val plateauRunLength: Int = 5
) {
    private val capacity = max(1, fsHz)
    private val rawHistory = FloatArray(capacity)
    private val nearRailHistory = BooleanArray(capacity)
    private var index = 0
    private var count = 0
    private var nearRailHits = 0
    private var warningRun = 0
    private var plateauRun = 0
    private var previous: Float? = null
    private var differenceEma = 0f

    fun next(value: Float): Boolean {
        val oldNearRail = if (count == capacity) nearRailHistory[index] else false
        if (oldNearRail) nearRailHits--

        val isNearRail = value <= nearRailLow || value >= nearRailHigh
        rawHistory[index] = value
        nearRailHistory[index] = isNearRail
        if (isNearRail) nearRailHits++
        index = (index + 1) % capacity
        if (count < capacity) count++

        val inWarningZone = value <= warningLow || value >= warningHigh
        warningRun = if (inWarningZone) warningRun + 1 else 0

        val last = previous
        val difference = if (last == null) 0f else abs(value - last)
        previous = value

        var minimum = Float.POSITIVE_INFINITY
        var maximum = Float.NEGATIVE_INFINITY
        for (i in 0 until count) {
            val sample = rawHistory[i]
            if (sample < minimum) minimum = sample
            if (sample > maximum) maximum = sample
        }
        val range = if (minimum <= maximum) maximum - minimum else 0f
        val extremeMargin = max(2f, range * 0.02f)
        val atLocalExtreme = range >= 100f &&
            (value <= minimum + extremeMargin || value >= maximum - extremeMargin)
        val historicalDifference = differenceEma
        // A smooth respiratory turning point also has a small derivative for
        // several samples, but it is not a clipped plateau. At the paired ADC
        // resolution (0.5 count), require effectively identical consecutive
        // values; the previous 0.5-count floor falsely classified ordinary
        // sinusoidal maxima and repeatedly held the processor inactive.
        val flatDifferenceLimit = minOf(0.1f, historicalDifference * 0.10f)
        val flatAtExtreme = last != null && atLocalExtreme && difference <= flatDifferenceLimit
        plateauRun = if (flatAtExtreme) plateauRun + 1 else 0

        // Do not let a motion spike permanently inflate the flatness reference.
        if (last != null && difference <= 240f) {
            differenceEma = if (differenceEma == 0f) difference
            else differenceEma * 0.98f + difference * 0.02f
        }

        return warningRun >= warningRunLength ||
            nearRailHits >= nearRailHitsRequired ||
            plateauRun >= plateauRunLength
    }

    fun clear() {
        rawHistory.fill(0f)
        nearRailHistory.fill(false)
        index = 0
        count = 0
        nearRailHits = 0
        warningRun = 0
        plateauRun = 0
        previous = null
        differenceEma = 0f
    }
}
