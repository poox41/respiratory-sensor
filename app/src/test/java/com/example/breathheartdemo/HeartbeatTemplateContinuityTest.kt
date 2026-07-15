package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import org.junit.Assert.assertTrue
import org.junit.Test

class HeartbeatTemplateContinuityTest {
    @Test
    fun enhancedDisplayLimitsPhaseResetJumps() {
        val fsHz = 50
        val enhancer = HeartbeatTemplateEnhancer(fsHz = fsHz)
        val peakTimes = setOf(
            800L, 1_600L, 2_400L, 3_200L, 3_900L,
            4_700L, 5_500L, 6_200L, 7_000L, 7_800L
        )
        var previous: Float? = null
        var maximumReadyStep = 0f
        var readySamples = 0

        for (timeMs in 0L..8_200L step 20L) {
            val phase = (timeMs % 800L).toDouble() / 800.0
            val input = sin(2.0 * PI * phase).toFloat()
            val result = enhancer.next(
                value = input,
                timeMs = timeMs,
                detectedPeakTimeMs = timeMs.takeIf { it in peakTimes }
            )
            if (result.ready) {
                val prior = previous
                if (prior != null) maximumReadyStep = maxOf(maximumReadyStep, abs(result.value - prior))
                previous = result.value
                readySamples++
            }
        }

        assertTrue("template should become ready", readySamples > fsHz)
        assertTrue("phase reset jump was $maximumReadyStep", maximumReadyStep <= 0.281f)
    }
}
