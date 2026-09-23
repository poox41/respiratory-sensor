package com.example.breathheartdemo

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Output of beat-synchronous template enhancement.
 *
 * [value] is a normalized, reconstructed display signal and is intentionally
 * kept separate from cleanHeart. The replayed waveform never drives BPM;
 * only [quality] may be used as supporting evidence for already detected beats.
 */
data class HeartTemplateEnhancement(
    val value: Float = 0f,
    val quality: Float? = null,
    val cycles: Int = 0,
    val ready: Boolean = false,
    val peakProcessed: Boolean = false,
    /** One normalized representative cycle for transparent paper display. */
    val template: List<Float> = emptyList()
)

/**
 * Builds a representative mechanical-heartbeat template from detected beats.
 * Consecutive peak-to-peak cycles are resampled, baseline removed, normalized,
 * polarity aligned and averaged.  The averaged template is then replayed using
 * the detected beat timing solely to make periodic morphology easier to view.
 */
class HeartbeatTemplateEnhancer(
    private val fsHz: Int,
    private val templatePoints: Int = 64,
    private val maxCycles: Int = 8,
    private val minCycleMs: Long = 500L,
    private val maxCycleMs: Long = 1_500L,
    private val minimumCorrelation: Float = 0.35f,
    private val maximumAlignmentShiftPoints: Int = 6,
    /**
     * Display-only morphology blend.  0 uses the robust median template;
     * 1 uses the retained real cycle most representative of its peers.
     */
    private val representativeCycleBlend: Float = 0f,
    /** Do not draw a publication-style trace from mutually inconsistent beats. */
    private val minimumReadyQuality: Float = 0.75f,
    /** Display-only slew limit; at 50 Hz this permits at most 0.28 per sample. */
    private val maximumDisplaySlopePerSecond: Float = 14f
) {
    private val historyTimes = ArrayDeque<Long>()
    private val historyValues = ArrayDeque<Float>()
    private val historyCapacity = fsHz * 5
    private val cycles = ArrayDeque<FloatArray>()
    private val acceptedIntervalsMs = ArrayDeque<Long>()

    private var template: FloatArray? = null
    private var templateQuality: Float? = null
    private var previousPeakTimeMs = 0L
    private var latestPeakTimeMs = 0L
    private var lastSampleTimeMs = 0L
    private var lastEnhancedValue = 0f
    private var hasEnhancedValue = false

    fun next(value: Float, timeMs: Long, detectedPeakTimeMs: Long?): HeartTemplateEnhancement {
        val expectedStepMs = (1_000L / fsHz).coerceAtLeast(1L)
        if (lastSampleTimeMs > 0L && timeMs - lastSampleTimeMs > expectedStepMs * 3L) {
            clear()
        }
        lastSampleTimeMs = timeMs

        historyTimes.addLast(timeMs)
        historyValues.addLast(value)
        while (historyTimes.size > historyCapacity) {
            historyTimes.removeFirst()
            historyValues.removeFirst()
        }

        var peakProcessed = false
        if (detectedPeakTimeMs != null && detectedPeakTimeMs > latestPeakTimeMs) {
            peakProcessed = true
            if (previousPeakTimeMs > 0L) {
                val intervalMs = detectedPeakTimeMs - previousPeakTimeMs
                if (intervalMs in minCycleMs..maxCycleMs && intervalIsConsistent(intervalMs)) {
                    val cycle = extractCycle(previousPeakTimeMs, detectedPeakTimeMs)
                    if (cycle != null) acceptCycle(cycle, intervalMs)
                }
            }
            previousPeakTimeMs = detectedPeakTimeMs
            latestPeakTimeMs = detectedPeakTimeMs
        }

        val ready = cycles.size >= 5 && template != null &&
            (templateQuality ?: 0f) >= minimumReadyQuality
        val enhancedValue = if (ready) {
            continuousDisplayValue(currentTemplateValue(timeMs))
        } else {
            hasEnhancedValue = false
            lastEnhancedValue = 0f
            0f
        }
        return HeartTemplateEnhancement(
            value = enhancedValue,
            quality = templateQuality,
            cycles = cycles.size,
            ready = ready,
            peakProcessed = peakProcessed,
            template = if (ready) displayTemplate().toList() else emptyList()
        )
    }

    private fun intervalIsConsistent(intervalMs: Long): Boolean {
        if (acceptedIntervalsMs.size < 3) return true
        val sorted = acceptedIntervalsMs.sorted()
        val median = sorted[sorted.size / 2]
        return intervalMs >= (median * 0.80f).toLong() &&
            intervalMs <= (median * 1.20f).toLong()
    }

    private fun extractCycle(startMs: Long, endMs: Long): FloatArray? {
        if (historyTimes.isEmpty() || startMs >= endMs) return null
        val times = historyTimes.toLongArray()
        val values = historyValues.toFloatArray()
        if (times.first() > startMs || times.last() < endMs) return null

        val output = FloatArray(templatePoints)
        var sourceIndex = 0
        val durationMs = endMs - startMs
        for (i in 0 until templatePoints) {
            val targetMs = startMs + durationMs * i / templatePoints
            while (sourceIndex + 1 < times.size && times[sourceIndex + 1] < targetMs) {
                sourceIndex++
            }
            if (sourceIndex + 1 >= times.size) return null
            val t0 = times[sourceIndex]
            val t1 = times[sourceIndex + 1]
            val fraction = if (t1 > t0) {
                (targetMs - t0).toFloat() / (t1 - t0).toFloat()
            } else 0f
            output[i] = values[sourceIndex] + (values[sourceIndex + 1] - values[sourceIndex]) * fraction
        }

        return if (normalizeCycle(output)) output else null
    }

    private fun normalizeCycle(cycle: FloatArray): Boolean {
        val mean = cycle.average().toFloat()
        var energy = 0.0
        for (i in cycle.indices) {
            cycle[i] -= mean
            energy += cycle[i] * cycle[i]
        }
        val rms = sqrt(energy / cycle.size).toFloat()
        if (!rms.isFinite() || rms < 1e-3f) return false
        for (i in cycle.indices) cycle[i] /= rms
        return true
    }

    private fun acceptCycle(cycle: FloatArray, intervalMs: Long) {
        val existingTemplate = template
        var acceptedCycle = cycle
        if (existingTemplate == null) {
            var strongestIndex = 0
            for (i in 1 until cycle.size) {
                if (abs(cycle[i]) > abs(cycle[strongestIndex])) strongestIndex = i
            }
            if (cycle[strongestIndex] < 0f) invert(cycle)
        } else {
            val alignment = bestAlignment(cycle, existingTemplate)
            acceptedCycle = alignment.first
            val correlation = alignment.second
            if (cycles.size >= 3 && correlation < minimumCorrelation) return
        }

        cycles.addLast(acceptedCycle)
        acceptedIntervalsMs.addLast(intervalMs)
        while (cycles.size > maxCycles) cycles.removeFirst()
        while (acceptedIntervalsMs.size > maxCycles) acceptedIntervalsMs.removeFirst()
        rebuildTemplate()
    }

    private fun rebuildTemplate() {
        if (cycles.isEmpty()) {
            template = null
            templateQuality = null
            return
        }
        var alignedCycles = cycles.map { it.copyOf() }
        // Re-align the retained cycles to the evolving robust template. This
        // corrects small detector jitter without imposing a synthetic BPM.
        repeat(2) {
            val reference = pointwiseMedian(alignedCycles)
            alignedCycles = alignedCycles.map { bestAlignment(it, reference).first }
        }
        cycles.clear()
        for (cycle in alignedCycles) cycles.addLast(cycle)

        val medianTemplate = pointwiseMedian(alignedCycles)
        val robustTemplate = FloatArray(templatePoints)
        for (i in robustTemplate.indices) {
            val previous = medianTemplate[(i - 1 + templatePoints) % templatePoints]
            val next = medianTemplate[(i + 1) % templatePoints]
            robustTemplate[i] = (previous + 2f * medianTemplate[i] + next) / 4f
        }

        // Preserve the sharper peak/trough morphology of a real retained
        // cycle without choosing an arbitrary or extreme beat.  The medoid is
        // the cycle with the highest mean similarity to all other retained
        // cycles.  This branch remains display-only and never feeds BPM.
        val representativeCycle = representativeMedoid(alignedCycles, robustTemplate)
        val blend = representativeCycleBlend.coerceIn(0f, 1f)
        val displayShape = FloatArray(templatePoints) { index ->
            (1f - blend) * robustTemplate[index] + blend * representativeCycle[index]
        }

        val mean = displayShape.average().toFloat()
        var maxAbs = 0f
        for (i in displayShape.indices) {
            displayShape[i] -= mean
            maxAbs = maxOf(maxAbs, abs(displayShape[i]))
        }
        if (maxAbs > 1e-6f) {
            for (i in displayShape.indices) displayShape[i] /= maxAbs
        }
        closePeriodicSeam(displayShape)
        template = displayShape

        var qualitySum = 0f
        for (cycle in cycles) qualitySum += abs(correlation(cycle, displayShape))
        templateQuality = (qualitySum / cycles.size.toFloat()).coerceIn(0f, 1f)
    }

    private fun representativeMedoid(
        sourceCycles: List<FloatArray>,
        reference: FloatArray
    ): FloatArray {
        if (sourceCycles.size == 1) return sourceCycles.first().copyOf()
        var bestIndex = 0
        var bestMeanSimilarity = Float.NEGATIVE_INFINITY
        for (first in sourceCycles.indices) {
            var similaritySum = 0f
            for (second in sourceCycles.indices) {
                if (first == second) continue
                similaritySum += abs(correlation(sourceCycles[first], sourceCycles[second]))
            }
            val meanSimilarity = similaritySum / (sourceCycles.size - 1).toFloat()
            if (meanSimilarity > bestMeanSimilarity) {
                bestMeanSimilarity = meanSimilarity
                bestIndex = first
            }
        }
        val result = sourceCycles[bestIndex].copyOf()
        if (correlation(result, reference) < 0f) invert(result)
        return result
    }

    private fun pointwiseMedian(sourceCycles: List<FloatArray>): FloatArray {
        val result = FloatArray(templatePoints)
        for (point in 0 until templatePoints) {
            val values = sourceCycles.map { it[point] }.sorted()
            val middle = values.size / 2
            result[point] = if (values.size % 2 == 0) {
                (values[middle - 1] + values[middle]) / 2f
            } else values[middle]
        }
        return result
    }

    private fun bestAlignment(cycle: FloatArray, reference: FloatArray): Pair<FloatArray, Float> {
        var bestCycle = cycle.copyOf()
        var bestCorrelation = Float.NEGATIVE_INFINITY
        for (shift in -maximumAlignmentShiftPoints..maximumAlignmentShiftPoints) {
            val shifted = circularShift(cycle, shift)
            var value = correlation(shifted, reference)
            if (value < 0f) {
                invert(shifted)
                value = -value
            }
            if (value > bestCorrelation) {
                bestCorrelation = value
                bestCycle = shifted
            }
        }
        return bestCycle to bestCorrelation
    }

    private fun circularShift(values: FloatArray, shift: Int): FloatArray {
        val output = FloatArray(values.size)
        for (i in values.indices) {
            var source = (i - shift) % values.size
            if (source < 0) source += values.size
            output[i] = values[source]
        }
        return output
    }

    private fun displayTemplate(): FloatArray {
        val current = template?.copyOf() ?: return FloatArray(0)
        var strongest = 0
        for (i in 1 until current.size) {
            if (abs(current[i]) > abs(current[strongest])) strongest = i
        }
        if (current[strongest] < 0f) invert(current)
        val target = (current.size * 0.30f).toInt()
        return circularShift(current, target - strongest)
    }

    private fun currentTemplateValue(timeMs: Long): Float {
        val currentTemplate = template ?: return 0f
        if (latestPeakTimeMs <= 0L || acceptedIntervalsMs.isEmpty()) return 0f
        val sortedIntervals = acceptedIntervalsMs.sorted()
        val medianIntervalMs = sortedIntervals[sortedIntervals.size / 2].coerceAtLeast(1L)
        val elapsedMs = timeMs - latestPeakTimeMs
        if (elapsedMs < 0L || elapsedMs > medianIntervalMs * 2L) return 0f

        val phase = (elapsedMs % medianIntervalMs).toFloat() / medianIntervalMs.toFloat()
        val point = phase * templatePoints
        val index0 = floor(point).toInt().coerceIn(0, templatePoints - 1)
        val index1 = (index0 + 1) % templatePoints
        val fraction = point - floor(point)
        return currentTemplate[index0] + (currentTemplate[index1] - currentTemplate[index0]) * fraction
    }

    /**
     * Make the first and last samples meet at the same value. The template is
     * replayed cyclically, so an open seam would otherwise create a vertical
     * display jump once per estimated heartbeat period.
     */
    private fun closePeriodicSeam(values: FloatArray) {
        if (values.size < 2) return
        val edgeValue = (values.first() + values.last()) / 2f
        values[0] = edgeValue
        values[values.lastIndex] = edgeValue
    }

    /**
     * Prevent a hard phase reset from drawing a non-physiological vertical
     * line. This is a display-only slew limiter and never feeds peak detection
     * or BPM calculation.
     */
    private fun continuousDisplayValue(target: Float): Float {
        if (!hasEnhancedValue) {
            hasEnhancedValue = true
            lastEnhancedValue = target
            return target
        }
        val maximumStep = (maximumDisplaySlopePerSecond / fsHz.toFloat()).coerceAtLeast(0.01f)
        val step = (target - lastEnhancedValue).coerceIn(-maximumStep, maximumStep)
        lastEnhancedValue += step
        return lastEnhancedValue
    }

    private fun correlation(a: FloatArray, b: FloatArray): Float {
        var cross = 0.0
        var energyA = 0.0
        var energyB = 0.0
        for (i in a.indices) {
            cross += a[i] * b[i]
            energyA += a[i] * a[i]
            energyB += b[i] * b[i]
        }
        val denominator = sqrt(energyA * energyB)
        return if (denominator > 1e-9) (cross / denominator).toFloat() else 0f
    }

    private fun invert(values: FloatArray) {
        for (i in values.indices) values[i] = -values[i]
    }

    fun clear() {
        historyTimes.clear()
        historyValues.clear()
        cycles.clear()
        acceptedIntervalsMs.clear()
        template = null
        templateQuality = null
        previousPeakTimeMs = 0L
        latestPeakTimeMs = 0L
        lastSampleTimeMs = 0L
        lastEnhancedValue = 0f
        hasEnhancedValue = false
    }
}
