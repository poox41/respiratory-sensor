package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import org.junit.Assert.assertTrue
import org.junit.Test

class HeartbeatTemplateBlendTest {
    @Test
    fun representativeCycleBlendPreservesSharperRealMorphology() {
        val robust = HeartbeatTemplateEnhancer(
            fsHz = 50,
            representativeCycleBlend = 0f
        )
        val selected = HeartbeatTemplateEnhancer(
            fsHz = 50,
            representativeCycleBlend = 0.75f
        )
        val representative = HeartbeatTemplateEnhancer(
            fsHz = 50,
            representativeCycleBlend = 1f
        )
        val peakTimes = (1L..12L).map { it * 800L }.toSet()
        var robustTemplate = emptyList<Float>()
        var selectedTemplate = emptyList<Float>()
        var representativeTemplate = emptyList<Float>()
        var selectedQuality = 0f

        for (timeMs in 0L..9_800L step 20L) {
            val cycleIndex = (timeMs / 800L).toInt()
            val phase = (timeMs % 800L).toDouble() / 800.0
            val jitter = ((cycleIndex % 5) - 2) * 0.004
            val impact = exp(-((phase - (0.30 + jitter)) / 0.032).let { it * it })
            val recoil = exp(-((phase - (0.48 - jitter)) / 0.060).let { it * it })
            val input = (
                0.12 * sin(2.0 * PI * phase) +
                    1.20 * impact -
                    0.58 * recoil
                ).toFloat()
            val detectedPeak = timeMs.takeIf { it in peakTimes }
            val robustResult = robust.next(input, timeMs, detectedPeak)
            val selectedResult = selected.next(input, timeMs, detectedPeak)
            val representativeResult = representative.next(input, timeMs, detectedPeak)
            if (robustResult.template.isNotEmpty()) robustTemplate = robustResult.template
            if (selectedResult.template.isNotEmpty()) selectedTemplate = selectedResult.template
            if (representativeResult.template.isNotEmpty()) {
                representativeTemplate = representativeResult.template
            }
            selectedQuality = selectedResult.quality ?: selectedQuality
        }

        assertTrue("robust template was not built", robustTemplate.isNotEmpty())
        assertTrue("selected template was not built", selectedTemplate.isNotEmpty())
        assertTrue("representative template was not built", representativeTemplate.isNotEmpty())
        val robustDistance = meanAbsoluteDistance(robustTemplate, representativeTemplate)
        val selectedDistance = meanAbsoluteDistance(selectedTemplate, representativeTemplate)
        assertTrue(
            "75% blend should stay closer to its representative real cycle: " +
                "$selectedDistance >= $robustDistance",
            selectedDistance < robustDistance
        )
        assertTrue("template consistency dropped too far: $selectedQuality", selectedQuality >= 0.80f)
    }

    private fun meanAbsoluteDistance(first: List<Float>, second: List<Float>): Float {
        var sum = 0f
        for (index in first.indices) {
            sum += kotlin.math.abs(first[index] - second[index])
        }
        return sum / first.size.toFloat()
    }
}
