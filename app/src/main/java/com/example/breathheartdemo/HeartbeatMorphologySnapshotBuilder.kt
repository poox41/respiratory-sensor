package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * Immutable display-only waveform reconstructed from completed real cycles.
 *
 * The timestamps are a compact presentation timeline ending at the latest
 * accepted peak. [boundaryTimesMs] contains the start and end of all cycles.
 * This snapshot must never be used as the input of BPM or the sleep model.
 */
data class MorphologyWaveformSnapshot(
    val timesMs: LongArray = LongArray(0),
    val values: FloatArray = FloatArray(0),
    val boundaryTimesMs: List<Long> = emptyList(),
    val cycles: Int = 0,
    val quality: Float? = null,
    val durationMs: Long = 0L,
    val medianRrMs: Long? = null,
    val cycleVariationRms: Float = 0f,
    val secondaryPeakRatio: Float = 0f,
    val ready: Boolean = false
)

/**
 * Builds an A/B-test waveform from the latest eight completed real cycles.
 *
 * Compared with [HeartbeatTemplateEnhancer], this class does not replay one
 * fixed template into the future. It waits for a cycle to finish, then mixes
 * a morphology constraint with that cycle's own low-frequency residual. The
 * resulting chart therefore has one-beat latency but preserves actual RR and
 * restrained beat-to-beat differences.
 */
class HeartbeatMorphologySnapshotBuilder(
    private val fsHz: Int,
    private val phasePoints: Int = 96,
    private val maxCycles: Int = 8,
    private val minCycleMs: Long = 500L,
    private val maxCycleMs: Long = 1_500L,
    private val baseWeight: Float = 0.82f,
    private val minimumCorrelation: Float = 0.25f,
    private val maximumAlignmentShiftPoints: Int = 8
) {
    private data class CycleRecord(
        val shape: FloatArray,
        val intervalMs: Long,
        val rms: Float
    )

    private data class ExtractedCycle(val shape: FloatArray, val rms: Float)

    private val historyTimes = ArrayDeque<Long>()
    private val historyValues = ArrayDeque<Float>()
    private val historyCapacity = fsHz * 5
    private val cycles = ArrayDeque<CycleRecord>()
    private var previousPeakTimeMs = 0L
    private var lastSampleTimeMs = 0L
    private var acceptanceReference: FloatArray? = null
    private var latestSnapshot = MorphologyWaveformSnapshot()

    /**
     * Adds every detail-channel sample. Returns a status/snapshot only when a
     * new peak is processed, so callers do no reconstruction work per sample.
     */
    fun next(value: Float, timeMs: Long, detectedPeakTimeMs: Long?): MorphologyWaveformSnapshot? {
        val expectedStepMs = (1_000L / fsHz).coerceAtLeast(1L)
        var resetForGap = false
        if (lastSampleTimeMs > 0L && timeMs - lastSampleTimeMs > expectedStepMs * 3L) {
            clear()
            resetForGap = true
        }
        lastSampleTimeMs = timeMs

        historyTimes.addLast(timeMs)
        historyValues.addLast(value)
        while (historyTimes.size > historyCapacity) {
            historyTimes.removeFirst()
            historyValues.removeFirst()
        }

        val peakTimeMs = detectedPeakTimeMs ?: return if (resetForGap) statusOnly() else null
        if (peakTimeMs <= previousPeakTimeMs) return null
        val startPeakMs = previousPeakTimeMs
        previousPeakTimeMs = peakTimeMs
        if (startPeakMs <= 0L) return statusOnly()

        val intervalMs = peakTimeMs - startPeakMs
        if (intervalMs !in minCycleMs..maxCycleMs || !intervalIsConsistent(intervalMs)) {
            return latestSnapshot.takeIf { it.ready } ?: statusOnly()
        }
        val extracted = extractCycle(startPeakMs, peakTimeMs)
            ?: return latestSnapshot.takeIf { it.ready } ?: statusOnly()
        if (!acceptCycle(extracted, intervalMs)) {
            return latestSnapshot.takeIf { it.ready } ?: statusOnly()
        }

        latestSnapshot = if (cycles.size >= maxCycles) {
            rebuildSnapshot(peakTimeMs)
        } else {
            statusOnly()
        }
        return latestSnapshot
    }

    private fun statusOnly(): MorphologyWaveformSnapshot = MorphologyWaveformSnapshot(
        cycles = cycles.size,
        quality = latestSnapshot.quality,
        medianRrMs = medianIntervalMs(),
        ready = false
    )

    private fun intervalIsConsistent(intervalMs: Long): Boolean {
        if (cycles.size < 3) return true
        val median = medianIntervalMs() ?: return true
        return intervalMs >= (median * 0.65f).toLong() &&
            intervalMs <= (median * 1.35f).toLong()
    }

    private fun medianIntervalMs(): Long? {
        if (cycles.isEmpty()) return null
        val sorted = cycles.map { it.intervalMs }.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2L
        } else {
            sorted[middle]
        }
    }

    private fun extractCycle(startMs: Long, endMs: Long): ExtractedCycle? {
        if (historyTimes.isEmpty() || startMs >= endMs) return null
        val times = historyTimes.toLongArray()
        val values = historyValues.toFloatArray()
        if (times.first() > startMs || times.last() < endMs) return null

        val output = FloatArray(phasePoints)
        var sourceIndex = 0
        val durationMs = endMs - startMs
        for (point in 0 until phasePoints) {
            val targetMs = startMs + durationMs * point / phasePoints
            while (sourceIndex + 1 < times.size && times[sourceIndex + 1] < targetMs) {
                sourceIndex++
            }
            if (sourceIndex + 1 >= times.size) return null
            val firstTime = times[sourceIndex]
            val secondTime = times[sourceIndex + 1]
            val fraction = if (secondTime > firstTime) {
                (targetMs - firstTime).toFloat() / (secondTime - firstTime).toFloat()
            } else {
                0f
            }
            output[point] = values[sourceIndex] +
                (values[sourceIndex + 1] - values[sourceIndex]) * fraction
        }

        val mean = output.average().toFloat()
        var energy = 0.0
        for (index in output.indices) {
            output[index] -= mean
            energy += output[index] * output[index]
        }
        val rms = sqrt(energy / output.size).toFloat()
        if (!rms.isFinite() || rms < 1e-3f) return null
        for (index in output.indices) output[index] /= rms
        return ExtractedCycle(output, rms)
    }

    private fun acceptCycle(extracted: ExtractedCycle, intervalMs: Long): Boolean {
        val reference = acceptanceReference
        if (reference != null && cycles.size >= 3) {
            val (_, correlation) = bestAlignment(extracted.shape, reference)
            if (correlation < minimumCorrelation) return false
        }
        cycles.addLast(CycleRecord(extracted.shape, intervalMs, extracted.rms))
        while (cycles.size > maxCycles) cycles.removeFirst()
        acceptanceReference = rebuildReference(cycles.map { it.shape })
        return true
    }

    private fun rebuildReference(sourceCycles: List<FloatArray>): FloatArray {
        var aligned = sourceCycles.map { it.copyOf() }
        repeat(2) {
            val reference = circularSmooth(pointwiseMedian(aligned), passes = 1)
            aligned = aligned.map { bestAlignment(it, reference).first }
        }
        return circularSmooth(pointwiseMedian(aligned), passes = 2)
    }

    private fun rebuildSnapshot(endTimeMs: Long): MorphologyWaveformSnapshot {
        val records = cycles.toList()
        var aligned = records.map { it.shape.copyOf() }
        repeat(3) {
            val reference = circularSmooth(pointwiseMedian(aligned), passes = 1)
            aligned = aligned.map { bestAlignment(it, reference).first }
        }

        val medianTemplate = pointwiseMedian(aligned)
        var robustTemplate = circularSmooth(medianTemplate, passes = 2)
        var medoid = circularSmooth(representativeMedoid(aligned), passes = 2)
        if (centeredCorrelation(medoid, robustTemplate) < 0f) invert(medoid)

        var base = FloatArray(phasePoints) { index ->
            0.72f * medoid[index] + 0.28f * robustTemplate[index]
        }
        base = circularSmooth(normalizeByAbsolutePeak(base), passes = 1)
        val oriented = orientAndRotate(base, targetFraction = 0.29f)
        base = oriented.first
        val phaseShift = oriented.second
        robustTemplate = circularShift(robustTemplate, phaseShift)
        medoid = circularShift(medoid, phaseShift)
        aligned = aligned.map { cycle ->
            val shifted = circularShift(cycle, phaseShift)
            if (centeredCorrelation(shifted, base) < 0f) {
                invert(shifted)
            }
            shifted
        }
        closePeriodicSeam(base)
        base = displayScale(base)
        closePeriodicSeam(base)

        val quality = aligned.map { abs(centeredCorrelation(it, base)) }.average().toFloat()
        val medianRms = medianFloat(records.map { it.rms })
        val mainIndex = indexOfMaximum(base)
        val commonEdge = (base.first() + base.last()) / 2f
        val displayShapes = ArrayList<FloatArray>(maxCycles)

        for (index in records.indices) {
            var realShape = normalizeByAbsolutePeak(circularSmooth(aligned[index], passes = 1))
            val residual = FloatArray(phasePoints) { point -> realShape[point] - base[point] }
            val smoothResidual = harmonicLowPass(residual, harmonics = 7)
            var shape = FloatArray(phasePoints) { point ->
                baseWeight * base[point] + (1f - baseWeight) * (base[point] + smoothResidual[point])
            }
            shape = constrainSecondaryPeaks(shape, mainIndex)
            closePeriodicSeam(shape, commonEdge)
            shape = displayScale(shape)
            closePeriodicSeam(shape, commonEdge)
            val amplitude = (1f + 0.18f * (records[index].rms / medianRms - 1f))
                .coerceIn(0.92f, 1.08f)
            for (point in shape.indices) shape[point] *= amplitude
            displayShapes.add(shape)
        }

        val intervals = records.map { it.intervalMs }
        val durationMs = intervals.sum()
        val startTimeMs = endTimeMs - durationMs
        val times = ArrayList<Long>()
        val values = ArrayList<Float>()
        val boundaries = ArrayList<Long>(maxCycles + 1)
        var elapsedMs = 0L
        for (index in records.indices) {
            val intervalMs = records[index].intervalMs
            val sampleCount = max(20, (intervalMs * fsHz / 1_000f).roundToInt())
            val shape = displayShapes[index]
            boundaries.add(startTimeMs + elapsedMs)
            for (sample in 0 until sampleCount) {
                val phasePoint = sample.toFloat() / sampleCount.toFloat() * phasePoints.toFloat()
                times.add(startTimeMs + elapsedMs + intervalMs * sample / sampleCount)
                values.add(interpolatePeriodic(shape, phasePoint))
            }
            elapsedMs += intervalMs
        }
        boundaries.add(startTimeMs + elapsedMs)
        times.add(startTimeMs + elapsedMs)
        values.add(displayShapes.last().first())

        val cycleVariation = pointwiseVariation(displayShapes)
        val secondaryPeakRatio = displayShapes
            .map { secondPositivePeakRatio(it) }
            .sorted()
            .let { sorted -> sorted[sorted.size / 2] }
        return MorphologyWaveformSnapshot(
            timesMs = times.toLongArray(),
            values = values.toFloatArray(),
            boundaryTimesMs = boundaries,
            cycles = records.size,
            quality = quality.coerceIn(0f, 1f),
            durationMs = durationMs,
            medianRrMs = medianIntervalMs(),
            cycleVariationRms = cycleVariation,
            secondaryPeakRatio = secondaryPeakRatio,
            ready = true
        )
    }

    private fun representativeMedoid(sourceCycles: List<FloatArray>): FloatArray {
        var bestIndex = 0
        var bestMeanSimilarity = Float.NEGATIVE_INFINITY
        for (first in sourceCycles.indices) {
            var similarity = 0f
            for (second in sourceCycles.indices) {
                if (first == second) continue
                similarity += abs(centeredCorrelation(sourceCycles[first], sourceCycles[second]))
            }
            val meanSimilarity = similarity / (sourceCycles.size - 1).coerceAtLeast(1).toFloat()
            if (meanSimilarity > bestMeanSimilarity) {
                bestMeanSimilarity = meanSimilarity
                bestIndex = first
            }
        }
        return sourceCycles[bestIndex].copyOf()
    }

    private fun pointwiseMedian(sourceCycles: List<FloatArray>): FloatArray {
        val output = FloatArray(phasePoints)
        for (point in 0 until phasePoints) {
            val values = sourceCycles.map { it[point] }.sorted()
            val middle = values.size / 2
            output[point] = if (values.size % 2 == 0) {
                (values[middle - 1] + values[middle]) / 2f
            } else {
                values[middle]
            }
        }
        return output
    }

    private fun circularSmooth(values: FloatArray, passes: Int): FloatArray {
        var output = values.copyOf()
        repeat(passes) {
            val source = output
            output = FloatArray(source.size) { index ->
                val minusTwo = source[wrappedIndex(index - 2, source.size)]
                val minusOne = source[wrappedIndex(index - 1, source.size)]
                val plusOne = source[wrappedIndex(index + 1, source.size)]
                val plusTwo = source[wrappedIndex(index + 2, source.size)]
                (minusTwo + 2f * minusOne + 3f * source[index] + 2f * plusOne + plusTwo) / 9f
            }
        }
        return output
    }

    private fun harmonicLowPass(values: FloatArray, harmonics: Int): FloatArray {
        val count = values.size
        val output = FloatArray(count)
        val dc = values.average().toFloat()
        for (sample in 0 until count) {
            var reconstructed = dc.toDouble()
            for (harmonic in 1..harmonics.coerceAtMost(count / 2 - 1)) {
                var real = 0.0
                var imaginary = 0.0
                for (source in 0 until count) {
                    val angle = 2.0 * PI * harmonic * source / count.toDouble()
                    real += values[source] * cos(angle)
                    imaginary -= values[source] * sin(angle)
                }
                val outputAngle = 2.0 * PI * harmonic * sample / count.toDouble()
                reconstructed += 2.0 / count.toDouble() *
                    (real * cos(outputAngle) - imaginary * sin(outputAngle))
            }
            output[sample] = reconstructed.toFloat()
        }
        return output
    }

    private fun constrainSecondaryPeaks(values: FloatArray, mainIndex: Int): FloatArray {
        val output = values.copyOf()
        val main = output[mainIndex]
        val exclusion = max(6, output.size / 9)
        val cap = 0.42f * main
        for (index in 1 until output.lastIndex) {
            val isPeak = output[index] > output[index - 1] && output[index] >= output[index + 1]
            if (!isPeak || abs(index - mainIndex) <= exclusion || output[index] <= cap) continue
            val excess = output[index] - cap
            for (point in output.indices) {
                val direct = abs(point - index)
                val distance = minOf(direct, output.size - direct).toDouble()
                output[point] -= (excess * 0.85 * exp(-0.5 * (distance / 4.0) * (distance / 4.0))).toFloat()
            }
        }
        return output
    }

    private fun secondPositivePeakRatio(values: FloatArray): Float {
        val peaks = ArrayList<Pair<Float, Int>>()
        for (index in 1 until values.lastIndex) {
            if (values[index] > values[index - 1] && values[index] >= values[index + 1] && values[index] > 0f) {
                peaks.add(values[index] to index)
            }
        }
        if (peaks.isEmpty()) return 1f
        peaks.sortByDescending { it.first }
        val main = peaks.first()
        val exclusion = max(4, values.size / 10)
        val second = peaks.drop(1)
            .filter { abs(it.second - main.second) > exclusion }
            .maxOfOrNull { it.first } ?: 0f
        return second / main.first.coerceAtLeast(1e-6f)
    }

    private fun pointwiseVariation(shapes: List<FloatArray>): Float {
        var total = 0.0
        for (point in 0 until phasePoints) {
            val mean = shapes.map { it[point] }.average()
            val variance = shapes.map { value ->
                val difference = value[point] - mean
                difference * difference
            }.average()
            total += sqrt(variance)
        }
        return (total / phasePoints).toFloat()
    }

    private fun displayScale(values: FloatArray, negativeLimit: Float = 0.68f): FloatArray {
        val output = values.copyOf()
        val positivePeak = output.maxOrNull()?.coerceAtLeast(1e-6f) ?: 1f
        for (index in output.indices) {
            output[index] /= positivePeak
            if (output[index] < 0f) {
                output[index] = (-negativeLimit * tanh((-output[index] / negativeLimit).toDouble())).toFloat()
            }
        }
        return output
    }

    private fun normalizeByAbsolutePeak(values: FloatArray): FloatArray {
        val output = values.copyOf()
        val mean = output.average().toFloat()
        var maximum = 0f
        for (index in output.indices) {
            output[index] -= mean
            maximum = max(maximum, abs(output[index]))
        }
        if (maximum > 1e-6f) {
            for (index in output.indices) output[index] /= maximum
        }
        return output
    }

    private fun orientAndRotate(values: FloatArray, targetFraction: Float): Pair<FloatArray, Int> {
        val output = values.copyOf()
        var strongest = 0
        for (index in 1 until output.size) {
            if (abs(output[index]) > abs(output[strongest])) strongest = index
        }
        if (output[strongest] < 0f) invert(output)
        val mainIndex = indexOfMaximum(output)
        val shift = (output.size * targetFraction).roundToInt() - mainIndex
        return circularShift(output, shift) to shift
    }

    private fun bestAlignment(cycle: FloatArray, reference: FloatArray): Pair<FloatArray, Float> {
        var bestCycle = cycle.copyOf()
        var bestCorrelation = Float.NEGATIVE_INFINITY
        for (shift in -maximumAlignmentShiftPoints..maximumAlignmentShiftPoints) {
            val shifted = circularShift(cycle, shift)
            var correlation = centeredCorrelation(shifted, reference)
            if (correlation < 0f) {
                invert(shifted)
                correlation = -correlation
            }
            if (correlation > bestCorrelation) {
                bestCorrelation = correlation
                bestCycle = shifted
            }
        }
        return bestCycle to bestCorrelation
    }

    private fun centeredCorrelation(first: FloatArray, second: FloatArray): Float {
        val firstMean = first.average().toFloat()
        val secondMean = second.average().toFloat()
        var cross = 0.0
        var firstEnergy = 0.0
        var secondEnergy = 0.0
        for (index in first.indices) {
            val left = first[index] - firstMean
            val right = second[index] - secondMean
            cross += left * right
            firstEnergy += left * left
            secondEnergy += right * right
        }
        val denominator = sqrt(firstEnergy * secondEnergy)
        return if (denominator > 1e-9) (cross / denominator).toFloat() else 0f
    }

    private fun circularShift(values: FloatArray, shift: Int): FloatArray {
        val output = FloatArray(values.size)
        for (index in values.indices) {
            output[index] = values[wrappedIndex(index - shift, values.size)]
        }
        return output
    }

    private fun wrappedIndex(index: Int, size: Int): Int {
        val remainder = index % size
        return if (remainder < 0) remainder + size else remainder
    }

    private fun interpolatePeriodic(values: FloatArray, point: Float): Float {
        val first = point.toInt().coerceIn(0, values.lastIndex)
        val second = (first + 1) % values.size
        val fraction = point - point.toInt().toFloat()
        return values[first] + (values[second] - values[first]) * fraction
    }

    private fun closePeriodicSeam(values: FloatArray, targetEdge: Float? = null) {
        if (values.size < 2) return
        val edge = targetEdge ?: (values.first() + values.last()) / 2f
        val count = 6.coerceAtMost(values.size / 2)
        for (point in 0 until count) {
            val blend = point.toFloat() / count.toFloat()
            values[point] = edge * (1f - blend) + values[point] * blend
            values[values.lastIndex - point] =
                edge * (1f - blend) + values[values.lastIndex - point] * blend
        }
        values[0] = edge
        values[values.lastIndex] = edge
    }

    private fun indexOfMaximum(values: FloatArray): Int {
        var result = 0
        for (index in 1 until values.size) {
            if (values[index] > values[result]) result = index
        }
        return result
    }

    private fun medianFloat(values: List<Float>): Float {
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2f
        } else {
            sorted[middle]
        }
    }

    private fun invert(values: FloatArray) {
        for (index in values.indices) values[index] = -values[index]
    }

    fun clear() {
        historyTimes.clear()
        historyValues.clear()
        cycles.clear()
        previousPeakTimeMs = 0L
        lastSampleTimeMs = 0L
        acceptanceReference = null
        latestSnapshot = MorphologyWaveformSnapshot()
    }
}
