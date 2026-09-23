package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Continuous display reconstruction around the accepted heart-rate track.
 *
 * A narrow fundamental branch suppresses respiratory leakage while parallel
 * branches retain signal energy that is actually present near 2f and 3f.
 * Keeping those harmonics prevents the display from degenerating into an
 * artificial pure sine wave. All branch centers slew gradually, so a new
 * four-second VMD result cannot replace or jump the displayed waveform. This
 * output is for waveform presentation only; it never feeds the BPM estimator
 * or sleep model.
 */
class AdaptiveTrackedHeartFilter(
    private val fsHz: Int,
    initialCenterHz: Float = 1.2f,
    private val q: Float = 3f,
    private val secondHarmonicQ: Float = 4f,
    private val thirdHarmonicQ: Float = 5f,
    private val secondHarmonicWeight: Float = 0.70f,
    private val thirdHarmonicWeight: Float = 0.45f,
    private val minimumCenterHz: Float = 40f / 60f,
    private val maximumCenterHz: Float = 110f / 60f,
    private val maximumSlewHzPerSecond: Float = 0.15f
) {
    data class Observation(
        val value: Float,
        val centerHz: Float,
        val hasAcceptedTarget: Boolean
    )

    private class ResonatorState {
        var x1 = 0f
        var x2 = 0f
        var y1 = 0f
        var y2 = 0f

        fun next(input: Float, coefficients: Coefficients): Float {
            val output = coefficients.b0 * input +
                coefficients.b1 * x1 +
                coefficients.b2 * x2 -
                coefficients.a1 * y1 -
                coefficients.a2 * y2
            x2 = x1
            x1 = input
            y2 = y1
            y1 = output
            return output
        }

        fun clear() {
            x1 = 0f
            x2 = 0f
            y1 = 0f
            y2 = 0f
        }
    }

    private data class Coefficients(
        val b0: Float,
        val b1: Float,
        val b2: Float,
        val a1: Float,
        val a2: Float
    )

    private val resetCenterHz = initialCenterHz.coerceIn(minimumCenterHz, maximumCenterHz)
    private val first = ResonatorState()
    private val second = ResonatorState()
    private val secondHarmonic = ResonatorState()
    private val thirdHarmonic = ResonatorState()
    private var centerHz = resetCenterHz
    private var lastAcceptedTargetHz: Float? = null

    fun next(input: Float, acceptedBpm: Float?): Observation {
        val acceptedTarget = acceptedBpm
            ?.takeIf { it.isFinite() && it in 40f..110f }
            ?.div(60f)
        if (acceptedTarget != null) lastAcceptedTargetHz = acceptedTarget
        val targetHz = lastAcceptedTargetHz ?: centerHz
        val maximumStep = maximumSlewHzPerSecond / fsHz.toFloat()
        centerHz += (targetHz - centerHz).coerceIn(-maximumStep, maximumStep)
        centerHz = centerHz.coerceIn(minimumCenterHz, maximumCenterHz)

        val fundamentalCoefficients = coefficients(centerHz, q)
        val firstOutput = first.next(input, fundamentalCoefficients)
        val fundamentalOutput = second.next(firstOutput, fundamentalCoefficients)
        val secondOutput = secondHarmonic.next(
            input,
            coefficients((centerHz * 2f).coerceAtMost(fsHz * 0.45f), secondHarmonicQ)
        )
        val thirdOutput = thirdHarmonic.next(
            input,
            coefficients((centerHz * 3f).coerceAtMost(fsHz * 0.45f), thirdHarmonicQ)
        )
        val output = fundamentalOutput +
            secondHarmonicWeight * secondOutput +
            thirdHarmonicWeight * thirdOutput
        return Observation(
            value = output,
            centerHz = centerHz,
            hasAcceptedTarget = lastAcceptedTargetHz != null
        )
    }

    fun clear() {
        first.clear()
        second.clear()
        secondHarmonic.clear()
        thirdHarmonic.clear()
        lastAcceptedTargetHz = null
        centerHz = resetCenterHz
    }

    private fun coefficients(frequencyHz: Float, branchQ: Float): Coefficients {
        val omega = 2.0 * PI * frequencyHz / fsHz.toDouble()
        val alpha = (sin(omega) / (2.0 * branchQ)).toFloat()
        val a0 = 1f + alpha
        return Coefficients(
            b0 = alpha / a0,
            b1 = 0f,
            b2 = -alpha / a0,
            a1 = (-2f * cos(omega).toFloat()) / a0,
            a2 = (1f - alpha) / a0
        )
    }
}
