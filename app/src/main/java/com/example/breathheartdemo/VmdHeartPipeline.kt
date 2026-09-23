package com.example.breathheartdemo

import java.util.concurrent.Executors
import java.util.concurrent.ThreadFactory
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

data class VmdHeartConfig(
    val windowSeconds: Int = 16,
    val stepSeconds: Int = 4,
    /** Published Nature Electronics structure: K=4 followed by K=5. */
    val firstLayerModeCount: Int = 4,
    val secondLayerModeCount: Int = 5,
    val cardiacLowHz: Float = 0.9f,
    val cardiacHighHz: Float = 2.0f,
    val prefilterLowHz: Float = 0.1f,
    val prefilterHighHz: Float = 5.0f,
    /** Retained for source compatibility with earlier single-layer callers. */
    val modeCount: Int = 8,
    val alpha: Float = 1_400f,
    val maxInitialCenterHz: Float = 6f,
    val maxIterations: Int = 300,
    val tolerance: Float = 1e-6f
)

data class VmdHeartAnalysis(
    val candidates: List<HeartModeCandidate>,
    val modeCentersHz: FloatArray,
    val firstLayerCentersHz: FloatArray,
    /** Narrow final IMF used only for heart-rate estimation and tracking. */
    val heartbeatWaveform: FloatArray,
    /** VMD-anchored f0 + harmonic reconstruction used only for morphology display. */
    val morphologyWaveform: FloatArray,
    val morphologyHarmonicOrders: IntArray,
    val respirationWaveform: FloatArray,
    val firstSelectedMode: Int,
    val secondSelectedMode: Int,
    val computeTimeMs: Long
)

data class VmdHeartStatus(
    val tracking: HeartTrackingResult = HeartTrackingResult(),
    val modeCentersHz: List<Float> = emptyList(),
    val candidateCount: Int = 0,
    val morphologyHarmonicOrders: List<Int> = emptyList(),
    val completedWindowEndMs: Long = 0L,
    val computeTimeMs: Long = 0L,
    val error: String? = null
)

data class VmdHeartWorkRequest(
    val windowEndMs: Long,
    val raw: FloatArray,
    val respiration: FloatArray,
    val heartbeatTemplate: FloatArray,
    val fixedBpm: Float? = null,
    val fixedPeriodicity: Float = 0f,
    val fixedPeakRatio: Float = 0f,
    val respirationRpm: Float? = null
)

data class VmdHeartWorkResult(
    val windowEndMs: Long,
    val analysis: VmdHeartAnalysis? = null,
    val error: String? = null,
    val fixedBpm: Float? = null,
    val fixedPeriodicity: Float = 0f,
    val fixedPeakRatio: Float = 0f,
    val respirationRpm: Float? = null
)

/**
 * Allocation-conscious VMD and feature extraction for one 16-second block.
 * Work is expected to run off the sampling/UI thread through [VmdHeartWorker].
 */
class VmdHeartAnalyzer(
    private val fsHz: Int,
    private val config: VmdHeartConfig = VmdHeartConfig()
) {
    private data class Decomposition(
        val modes: Array<FloatArray>,
        val centersHz: FloatArray,
        val peakRatios: FloatArray
    )

    private data class MorphologyReconstruction(
        val waveform: FloatArray,
        val harmonicOrders: IntArray
    )

    fun analyze(request: VmdHeartWorkRequest): VmdHeartAnalysis {
        require(request.raw.size == request.respiration.size) {
            "raw and respiration windows must have identical lengths"
        }
        require(request.raw.size >= fsHz * config.windowSeconds * 9 / 10) {
            "insufficient samples for VMD window"
        }
        val startedNs = System.nanoTime()
        // Reproduce the paper's signal path on a complete background block:
        // 0.1-5 Hz zero-phase preprocessing -> K=4 -> cardiac IMF -> K=5.
        val prefiltered = paperBandPass(request.raw)
        val firstLayer = decompose(
            source = prefiltered,
            modeCount = config.firstLayerModeCount,
            maxInitialCenterHz = config.prefilterHighHz
        )
        val respirationIndex = firstLayer.centersHz.indices.minByOrNull {
            firstLayer.centersHz[it]
        } ?: 0
        val firstSelection = selectCardiacMode(firstLayer)
        val secondLayer = decompose(
            source = firstLayer.modes[firstSelection.index],
            modeCount = config.secondLayerModeCount,
            maxInitialCenterHz = config.prefilterHighHz
        )
        val secondSelection = selectCardiacMode(secondLayer)
        val morphology = reconstructMorphology(
            prefiltered = prefiltered,
            fundamental = secondLayer.modes[secondSelection.index],
            fundamentalHz = secondSelection.spectrum.peakHz
        )
        val candidates = listOf(
            extractPaperCandidate(
                decomposition = secondLayer,
                selectedMode = secondSelection.index,
                peakHz = secondSelection.spectrum.peakHz,
                respiration = firstLayer.modes[respirationIndex],
                heartbeatTemplate = request.heartbeatTemplate,
                spectrumPeakRatio = secondSelection.spectrum.peakRatio
            )
        )
        return VmdHeartAnalysis(
            candidates = candidates,
            modeCentersHz = secondLayer.centersHz,
            firstLayerCentersHz = firstLayer.centersHz,
            heartbeatWaveform = secondLayer.modes[secondSelection.index],
            morphologyWaveform = morphology.waveform,
            morphologyHarmonicOrders = morphology.harmonicOrders,
            respirationWaveform = firstLayer.modes[respirationIndex],
            firstSelectedMode = firstSelection.index + 1,
            secondSelectedMode = secondSelection.index + 1,
            computeTimeMs = (System.nanoTime() - startedNs) / 1_000_000L
        )
    }

    private fun decompose(
        source: FloatArray,
        modeCount: Int,
        maxInitialCenterHz: Float
    ): Decomposition {
        val signal = detrend(source)
        val sampleCount = signal.size - signal.size % 2
        val halfSamples = sampleCount / 2
        val mirroredCount = sampleCount * 2
        val fftSize = nextPowerOfTwo(mirroredCount)
        val halfFft = fftSize / 2

        val spectrumRe = FloatArray(fftSize)
        val spectrumIm = FloatArray(fftSize)
        for (i in 0 until halfSamples) spectrumRe[i] = signal[halfSamples - 1 - i]
        for (i in 0 until sampleCount) spectrumRe[halfSamples + i] = signal[i]
        for (i in 0 until halfSamples) {
            spectrumRe[halfSamples + sampleCount + i] = signal[sampleCount - 1 - i]
        }
        Radix2Fft.transform(spectrumRe, spectrumIm, inverse = false)

        // VMD uses an fft-shifted, one-sided spectrum. Arrays are flattened
        // mode-major to avoid per-bin objects and complex-number allocations.
        val positiveRe = FloatArray(fftSize)
        val positiveIm = FloatArray(fftSize)
        for (shifted in halfFft until fftSize) {
            val sourceIndex = (shifted + halfFft) % fftSize
            positiveRe[shifted] = spectrumRe[sourceIndex]
            positiveIm[shifted] = spectrumIm[sourceIndex]
        }

        val kCount = modeCount
        val modeRe = FloatArray(kCount * fftSize)
        val modeIm = FloatArray(kCount * fftSize)
        val runningRe = FloatArray(fftSize)
        val runningIm = FloatArray(fftSize)
        val centers = FloatArray(kCount) { mode ->
            if (kCount == 1) 0f
            else (mode.toFloat() / (kCount - 1).toFloat()) *
                maxInitialCenterHz / fsHz.toFloat()
        }

        for (iteration in 0 until config.maxIterations) {
            java.util.Arrays.fill(runningRe, 0f)
            java.util.Arrays.fill(runningIm, 0f)
            for (mode in 0 until kCount) {
                val base = mode * fftSize
                for (bin in halfFft until fftSize) {
                    runningRe[bin] += modeRe[base + bin]
                    runningIm[bin] += modeIm[base + bin]
                }
            }

            var change = 0.0
            for (mode in 0 until kCount) {
                val base = mode * fftSize
                var weightedFrequency = 0.0
                var energySum = 0.0
                for (bin in halfFft until fftSize) {
                    val index = base + bin
                    val oldRe = modeRe[index]
                    val oldIm = modeIm[index]
                    runningRe[bin] -= oldRe
                    runningIm[bin] -= oldIm
                    val frequency = (bin - halfFft).toFloat() / fftSize.toFloat()
                    val delta = frequency - centers[mode]
                    val denominator = 1f + config.alpha * delta * delta
                    val newRe = (positiveRe[bin] - runningRe[bin]) / denominator
                    val newIm = (positiveIm[bin] - runningIm[bin]) / denominator
                    modeRe[index] = newRe
                    modeIm[index] = newIm
                    runningRe[bin] += newRe
                    runningIm[bin] += newIm
                    val differenceRe = newRe - oldRe
                    val differenceIm = newIm - oldIm
                    change += differenceRe * differenceRe + differenceIm * differenceIm
                    val energy = newRe.toDouble() * newRe + newIm.toDouble() * newIm
                    weightedFrequency += frequency * energy
                    energySum += energy
                }
                if (energySum > 1e-18) centers[mode] = (weightedFrequency / energySum).toFloat()
            }
            if (change / fftSize.toDouble() < config.tolerance) break
        }

        val modes = Array(kCount) { FloatArray(sampleCount) }
        val peakRatios = FloatArray(kCount)
        val fullRe = FloatArray(fftSize)
        val fullIm = FloatArray(fftSize)
        val unshiftedRe = FloatArray(fftSize)
        val unshiftedIm = FloatArray(fftSize)
        val positiveBandPowers = FloatArray(fftSize / 2)
        for (mode in 0 until kCount) {
            java.util.Arrays.fill(fullRe, 0f)
            java.util.Arrays.fill(fullIm, 0f)
            val base = mode * fftSize
            for (bin in halfFft until fftSize) {
                fullRe[bin] = modeRe[base + bin]
                fullIm[bin] = modeIm[base + bin]
            }
            // Hermitian reflection in shifted coordinates.
            for (bin in 1 until halfFft) {
                val mirror = fftSize - bin
                fullRe[bin] = fullRe[mirror]
                fullIm[bin] = -fullIm[mirror]
            }
            fullRe[0] = fullRe[fftSize - 1]
            fullIm[0] = -fullIm[fftSize - 1]
            fullIm[halfFft] = 0f

            var powerCount = 0
            var peakPower = 0f
            for (bin in halfFft until fftSize) {
                val frequencyHz = (bin - halfFft).toFloat() * fsHz / fftSize.toFloat()
                if (frequencyHz in 0.5f..minOf(8f, fsHz * 0.49f)) {
                    val power = fullRe[bin] * fullRe[bin] + fullIm[bin] * fullIm[bin]
                    positiveBandPowers[powerCount++] = power
                    if (power > peakPower) peakPower = power
                }
            }
            java.util.Arrays.sort(positiveBandPowers, 0, powerCount)
            val medianPower = if (powerCount == 0) 0f else positiveBandPowers[powerCount / 2]
            peakRatios[mode] = peakPower / max(medianPower, 1e-12f)

            // ifftshift into the conventional FFT order before reconstruction.
            java.util.Arrays.fill(unshiftedRe, 0f)
            java.util.Arrays.fill(unshiftedIm, 0f)
            for (shifted in 0 until fftSize) {
                val destination = (shifted + halfFft) % fftSize
                unshiftedRe[destination] = fullRe[shifted]
                unshiftedIm[destination] = fullIm[shifted]
            }
            Radix2Fft.transform(unshiftedRe, unshiftedIm, inverse = true)
            for (i in 0 until sampleCount) modes[mode][i] = unshiftedRe[halfSamples + i]
        }
        return Decomposition(modes, FloatArray(kCount) { centers[it] * fsHz }, peakRatios)
    }

    private data class SpectrumMetrics(
        val peakHz: Float,
        val peakRatio: Float,
        val inBandPower: Double,
        val totalPower: Double
    )

    private data class CardiacSelection(
        val index: Int,
        val spectrum: SpectrumMetrics
    )

    /** FFT selection rule reported by the paper for both VMD layers. */
    private fun selectCardiacMode(decomposition: Decomposition): CardiacSelection {
        var bestIndex = 0
        var bestSpectrum = spectrumMetrics(decomposition.modes[0])
        var bestScore = Double.NEGATIVE_INFINITY
        for (index in decomposition.modes.indices) {
            val spectrum = if (index == 0) bestSpectrum else spectrumMetrics(decomposition.modes[index])
            val concentration = spectrum.inBandPower / max(spectrum.totalPower, 1e-18)
            val score = concentration * (1.0 + minOf(spectrum.peakRatio / 20f, 2f))
            if (score > bestScore) {
                bestScore = score
                bestIndex = index
                bestSpectrum = spectrum
            }
        }
        return CardiacSelection(bestIndex, bestSpectrum)
    }

    private fun spectrumMetrics(values: FloatArray): SpectrumMetrics {
        val centered = detrend(values)
        val fftSize = nextPowerOfTwo(centered.size * 4)
        val real = FloatArray(fftSize)
        val imaginary = FloatArray(fftSize)
        for (index in centered.indices) {
            val window = if (centered.size <= 1) 1.0 else {
                0.5 - 0.5 * cos(2.0 * PI * index / (centered.size - 1).toDouble())
            }
            real[index] = (centered[index] * window).toFloat()
        }
        Radix2Fft.transform(real, imaginary, inverse = false)
        val bandPowers = FloatArray(fftSize / 2)
        var bandCount = 0
        var peakPower = 0f
        var peakHz = config.cardiacLowHz
        var inBandPower = 0.0
        var totalPower = 0.0
        for (bin in 1 until fftSize / 2) {
            val frequencyHz = bin.toFloat() * fsHz / fftSize.toFloat()
            if (frequencyHz > config.prefilterHighHz) break
            val power = real[bin] * real[bin] + imaginary[bin] * imaginary[bin]
            if (frequencyHz >= config.prefilterLowHz) totalPower += power
            if (frequencyHz in config.cardiacLowHz..config.cardiacHighHz) {
                inBandPower += power
                bandPowers[bandCount++] = power
                if (power > peakPower) {
                    peakPower = power
                    peakHz = frequencyHz
                }
            }
        }
        java.util.Arrays.sort(bandPowers, 0, bandCount)
        val medianPower = if (bandCount == 0) 0f else bandPowers[bandCount / 2]
        return SpectrumMetrics(
            peakHz = peakHz,
            peakRatio = peakPower / max(medianPower, 1e-12f),
            inBandPower = inBandPower,
            totalPower = totalPower
        )
    }

    private fun extractPaperCandidate(
        decomposition: Decomposition,
        selectedMode: Int,
        peakHz: Float,
        respiration: FloatArray,
        heartbeatTemplate: FloatArray,
        spectrumPeakRatio: Float
    ): HeartModeCandidate {
        val mode = decomposition.modes[selectedMode]
        val lag = (fsHz / peakHz).roundToInt().coerceAtLeast(2)
        return HeartModeCandidate(
            centerHz = peakHz,
            periodicity = normalizedAutocorrelation(mode, lag).coerceIn(0f, 1f),
            templateCorrelation = templateConsistency(mode, lag, heartbeatTemplate).coerceIn(0f, 1f),
            distinctHarmonicOrders = harmonicSupport(decomposition, selectedMode, peakHz),
            respirationCoherence = respirationPhaseCoherence(mode, respiration),
            spectrumPeakRatio = spectrumPeakRatio,
            multilayerConfirmed = true
        )
    }

    /**
     * Restore repeatable mechanical-heart morphology without changing the rate
     * estimator.  The paper-aligned final IMF supplies f0; display detail is
     * reconstructed from zero-phase bands around f0, 2f0, 3f0 and 4f0.  A
     * harmonic is admitted only when it repeats at the fundamental period and
     * does not materially reduce whole-cycle periodicity.  Per-order amplitude
     * caps prevent a high-frequency vibration from dominating the waveform.
     */
    private fun reconstructMorphology(
        prefiltered: FloatArray,
        fundamental: FloatArray,
        fundamentalHz: Float
    ): MorphologyReconstruction {
        if (!fundamentalHz.isFinite() || fundamentalHz !in config.cardiacLowHz..config.cardiacHighHz) {
            return MorphologyReconstruction(fundamental.copyOf(), intArrayOf(1))
        }
        val halfWidth = 0.28f
        val baseLow = max(0.65f, (1f - halfWidth) * fundamentalHz)
        val baseHigh = minOf(2.4f, (1f + halfWidth) * fundamentalHz)
        if (baseHigh - baseLow < 0.12f) {
            return MorphologyReconstruction(fundamental.copyOf(), intArrayOf(1))
        }
        val base = adaptiveZeroPhaseBandPass(prefiltered, baseLow, baseHigh)
        val baseRms = rms(base)
        val lag = (fsHz / fundamentalHz).roundToInt().coerceAtLeast(2)
        if (baseRms < 1e-8f || normalizedAutocorrelation(base, lag) < 0.30f) {
            return MorphologyReconstruction(fundamental.copyOf(), intArrayOf(1))
        }
        val waveform = FloatArray(base.size) { index -> base[index] / baseRms }
        val includedOrders = ArrayList<Int>(4)
        includedOrders.add(1)
        var previousPeriodicity = normalizedAutocorrelation(waveform, lag)
        for (order in 2..4) {
            val low = max(0.8f, (order - halfWidth) * fundamentalHz)
            val high = minOf(config.prefilterHighHz, (order + halfWidth) * fundamentalHz)
            if (low >= config.prefilterHighHz || high - low < 0.12f) break
            val component = adaptiveZeroPhaseBandPass(prefiltered, low, high)
            val componentRms = rms(component)
            val rawRatio = componentRms / baseRms
            val componentPeriodicity = normalizedAutocorrelation(component, lag)
            if (rawRatio < 0.025f || componentPeriodicity < 0.30f) continue
            val cap = when (order) {
                2 -> 0.55f
                3 -> 0.34f
                else -> 0.22f
            }
            val ratio = minOf(rawRatio, cap)
            val candidate = FloatArray(waveform.size) { index ->
                waveform[index] + component[index] / max(componentRms, 1e-8f) * ratio
            }
            val candidatePeriodicity = normalizedAutocorrelation(candidate, lag)
            if (candidatePeriodicity + 0.05f < previousPeriodicity) continue
            for (index in waveform.indices) waveform[index] = candidate[index]
            previousPeriodicity = candidatePeriodicity
            includedOrders.add(order)
        }
        return MorphologyReconstruction(waveform, includedOrders.toIntArray())
    }

    private fun adaptiveZeroPhaseBandPass(
        source: FloatArray,
        lowHz: Float,
        highHz: Float
    ): FloatArray {
        val forward = applyAdaptiveBandPass(detrend(source), lowHz, highHz)
        forward.reverse()
        val backward = applyAdaptiveBandPass(forward, lowHz, highHz)
        backward.reverse()
        return backward
    }

    private fun applyAdaptiveBandPass(
        source: FloatArray,
        lowHz: Float,
        highHz: Float
    ): FloatArray {
        val highPass = BiquadFilter.highPass(lowHz, fsHz, 0.7071068f)
        val lowPass = BiquadFilter.lowPass(highHz, fsHz, 0.7071068f)
        return FloatArray(source.size) { index -> lowPass.next(highPass.next(source[index])) }
    }

    private fun rms(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        var mean = 0.0
        for (value in values) mean += value
        mean /= values.size
        var energy = 0.0
        for (value in values) {
            val centered = value - mean
            energy += centered * centered
        }
        return sqrt(energy / values.size).toFloat()
    }

    private fun harmonicSupport(
        decomposition: Decomposition,
        selectedMode: Int,
        fundamentalHz: Float
    ): Int {
        var support = 1
        for (order in 2..3) {
            val target = fundamentalHz * order
            if (decomposition.centersHz.indices.any { index ->
                    index != selectedMode && abs(decomposition.centersHz[index] - target) <= 0.32f * fundamentalHz
                }
            ) support++
        }
        return support
    }

    private fun paperBandPass(source: FloatArray): FloatArray {
        val centered = detrend(source)
        val forward = applyPaperBandPass(centered)
        forward.reverse()
        val backward = applyPaperBandPass(forward)
        backward.reverse()
        return backward
    }

    private fun applyPaperBandPass(source: FloatArray): FloatArray {
        val hp1 = BiquadFilter.highPass(config.prefilterLowHz, fsHz, 0.5411961f)
        val hp2 = BiquadFilter.highPass(config.prefilterLowHz, fsHz, 1.306563f)
        val lp1 = BiquadFilter.lowPass(config.prefilterHighHz, fsHz, 0.5411961f)
        val lp2 = BiquadFilter.lowPass(config.prefilterHighHz, fsHz, 1.306563f)
        return FloatArray(source.size) { index ->
            lp2.next(lp1.next(hp2.next(hp1.next(source[index]))))
        }
    }

    private fun detrend(source: FloatArray): FloatArray {
        val n = source.size
        val output = FloatArray(n)
        if (n == 0) return output
        val meanX = (n - 1) / 2.0
        var meanY = 0.0
        for (value in source) meanY += value
        meanY /= n
        var covariance = 0.0
        var varianceX = 0.0
        for (i in source.indices) {
            val dx = i - meanX
            covariance += dx * (source[i] - meanY)
            varianceX += dx * dx
        }
        val slope = if (varianceX > 0.0) covariance / varianceX else 0.0
        for (i in source.indices) output[i] = (source[i] - meanY - slope * (i - meanX)).toFloat()
        return output
    }

    private fun normalizedAutocorrelation(values: FloatArray, lag: Int): Float {
        if (lag <= 0 || lag >= values.size) return 0f
        var mean = 0.0
        for (value in values) mean += value
        mean /= values.size
        var cross = 0.0
        var leftEnergy = 0.0
        var rightEnergy = 0.0
        for (i in lag until values.size) {
            val left = values[i - lag] - mean
            val right = values[i] - mean
            cross += left * right
            leftEnergy += left * left
            rightEnergy += right * right
        }
        return if (leftEnergy > 1e-18 && rightEnergy > 1e-18) {
            (cross / sqrt(leftEnergy * rightEnergy)).toFloat().coerceAtLeast(0f)
        } else 0f
    }

    private fun templateConsistency(
        values: FloatArray,
        lag: Int,
        externalTemplate: FloatArray
    ): Float {
        val points = 64
        val cycleCount = values.size / lag
        if (cycleCount < 4) return 0f
        val cycles = Array(cycleCount) { FloatArray(points) }
        val meanCycle = FloatArray(points)
        var accepted = 0
        for (cycleIndex in 0 until cycleCount) {
            val start = cycleIndex * lag
            if (start + lag > values.size) break
            val cycle = cycles[accepted]
            for (point in 0 until points) {
                val position = point.toFloat() * (lag - 1) / (points - 1).toFloat()
                val low = position.toInt()
                val high = minOf(low + 1, lag - 1)
                val fraction = position - low
                cycle[point] = values[start + low] * (1f - fraction) + values[start + high] * fraction
            }
            if (!normalize(cycle)) continue
            if (accepted > 0 && correlation(cycle, cycles[0]) < 0f) {
                for (point in cycle.indices) cycle[point] = -cycle[point]
            }
            for (point in cycle.indices) meanCycle[point] += cycle[point]
            accepted++
        }
        if (accepted < 4 || !normalize(meanCycle)) return 0f
        val correlations = FloatArray(accepted)
        for (index in 0 until accepted) correlations[index] = abs(correlation(cycles[index], meanCycle))
        correlations.sort()
        val selfConsistency = correlations[accepted / 2]
        val external = if (externalTemplate.size >= 8) {
            abs(correlationResampled(meanCycle, externalTemplate))
        } else 0f
        return max(selfConsistency, external)
    }

    private fun respirationPhaseCoherence(values: FloatArray, respiration: FloatArray): Float {
        if (values.size != respiration.size || values.size < 8) return 0f
        var meanResp = 0.0
        for (value in respiration) meanResp += value
        meanResp /= respiration.size
        var respirationEnergy = 0.0
        for (value in respiration) {
            val centered = value - meanResp
            respirationEnergy += centered * centered
        }
        if (respirationEnergy < 1e-12) return 0f

        var best = 0.0
        for (harmonic in 1..6) {
            var dotSin = 0.0
            var dotCos = 0.0
            var sinEnergy = 0.0
            var cosEnergy = 0.0
            var valueEnergy = 0.0
            for (i in values.indices) {
                val previous = respiration[max(0, i - 1)] - meanResp
                val next = respiration[minOf(respiration.lastIndex, i + 1)] - meanResp
                val phase = atan2(-(next - previous).toDouble(), 2.0 * (respiration[i] - meanResp))
                val phaseSin = sin(harmonic * phase)
                val phaseCos = cos(harmonic * phase)
                val value = values[i].toDouble()
                dotSin += value * phaseSin
                dotCos += value * phaseCos
                sinEnergy += phaseSin * phaseSin
                cosEnergy += phaseCos * phaseCos
                valueEnergy += value * value
            }
            if (valueEnergy > 1e-18) {
                val sinCorrelation = dotSin / sqrt(max(1e-18, valueEnergy * sinEnergy))
                val cosCorrelation = dotCos / sqrt(max(1e-18, valueEnergy * cosEnergy))
                best = max(best, sqrt(sinCorrelation * sinCorrelation + cosCorrelation * cosCorrelation))
            }
        }
        return best.toFloat().coerceIn(0f, 1f)
    }

    private fun normalize(values: FloatArray): Boolean {
        var mean = 0.0
        for (value in values) mean += value
        mean /= values.size
        var energy = 0.0
        for (i in values.indices) {
            values[i] = (values[i] - mean).toFloat()
            energy += values[i] * values[i]
        }
        val rms = sqrt(energy / values.size).toFloat()
        if (!rms.isFinite() || rms < 1e-8f) return false
        for (i in values.indices) values[i] /= rms
        return true
    }

    private fun correlation(first: FloatArray, second: FloatArray): Float {
        if (first.size != second.size || first.isEmpty()) return 0f
        var dot = 0.0
        var firstEnergy = 0.0
        var secondEnergy = 0.0
        for (i in first.indices) {
            dot += first[i] * second[i]
            firstEnergy += first[i] * first[i]
            secondEnergy += second[i] * second[i]
        }
        return if (firstEnergy > 1e-18 && secondEnergy > 1e-18) {
            (dot / sqrt(firstEnergy * secondEnergy)).toFloat()
        } else 0f
    }

    private fun correlationResampled(first: FloatArray, second: FloatArray): Float {
        val resampled = FloatArray(first.size)
        for (i in resampled.indices) {
            val position = i.toFloat() * (second.size - 1) / (resampled.size - 1).toFloat()
            val low = position.toInt()
            val high = minOf(low + 1, second.lastIndex)
            val fraction = position - low
            resampled[i] = second[low] * (1f - fraction) + second[high] * fraction
        }
        if (!normalize(resampled)) return 0f
        return correlation(first, resampled)
    }

    private fun nextPowerOfTwo(value: Int): Int {
        var result = 1
        while (result < value) result = result shl 1
        return result
    }
}

/** Single-flight background runner: a slow VMD frame is skipped, never queued. */
class VmdHeartWorker(
    fsHz: Int,
    config: VmdHeartConfig = VmdHeartConfig(),
    private val onResult: (VmdHeartWorkResult) -> Unit
) : AutoCloseable {
    private val analyzer = VmdHeartAnalyzer(fsHz, config)
    private val busy = AtomicBoolean(false)
    private val generation = AtomicLong(0L)
    private val deliveryLock = Any()
    @Volatile private var closed = false
    private val executor = Executors.newSingleThreadExecutor(ThreadFactory { task ->
        Thread(task, "heart-vmd-worker").apply {
            isDaemon = true
            priority = Thread.NORM_PRIORITY - 1
        }
    })

    fun offer(request: VmdHeartWorkRequest): Boolean {
        if (closed || !busy.compareAndSet(false, true)) return false
        val acceptedGeneration = generation.get()
        executor.execute {
            val result = try {
                VmdHeartWorkResult(
                    windowEndMs = request.windowEndMs,
                    analysis = analyzer.analyze(request),
                    fixedBpm = request.fixedBpm,
                    fixedPeriodicity = request.fixedPeriodicity,
                    fixedPeakRatio = request.fixedPeakRatio,
                    respirationRpm = request.respirationRpm
                )
            } catch (failure: Throwable) {
                VmdHeartWorkResult(
                    windowEndMs = request.windowEndMs,
                    error = failure.message ?: failure.javaClass.simpleName,
                    fixedBpm = request.fixedBpm,
                    fixedPeriodicity = request.fixedPeriodicity,
                    fixedPeakRatio = request.fixedPeakRatio,
                    respirationRpm = request.respirationRpm
                )
            }
            try {
                synchronized(deliveryLock) {
                    if (!closed && generation.get() == acceptedGeneration) onResult(result)
                }
            } finally {
                busy.set(false)
            }
        }
        return true
    }

    fun invalidate() {
        synchronized(deliveryLock) {
            generation.incrementAndGet()
        }
    }

    override fun close() {
        synchronized(deliveryLock) {
            closed = true
            generation.incrementAndGet()
        }
        executor.shutdownNow()
    }
}

/** Iterative radix-2 FFT over primitive arrays. */
private object Radix2Fft {
    fun transform(real: FloatArray, imaginary: FloatArray, inverse: Boolean) {
        require(real.size == imaginary.size && real.size > 0 && real.size and (real.size - 1) == 0)
        val size = real.size
        var j = 0
        for (i in 1 until size) {
            var bit = size shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val realTemp = real[i]
                real[i] = real[j]
                real[j] = realTemp
                val imaginaryTemp = imaginary[i]
                imaginary[i] = imaginary[j]
                imaginary[j] = imaginaryTemp
            }
        }

        var length = 2
        while (length <= size) {
            val angle = (if (inverse) 2.0 else -2.0) * PI / length
            val rootRe = cos(angle).toFloat()
            val rootIm = sin(angle).toFloat()
            var start = 0
            while (start < size) {
                var twiddleRe = 1f
                var twiddleIm = 0f
                for (offset in 0 until length / 2) {
                    val even = start + offset
                    val odd = even + length / 2
                    val oddRe = real[odd] * twiddleRe - imaginary[odd] * twiddleIm
                    val oddIm = real[odd] * twiddleIm + imaginary[odd] * twiddleRe
                    real[odd] = real[even] - oddRe
                    imaginary[odd] = imaginary[even] - oddIm
                    real[even] += oddRe
                    imaginary[even] += oddIm
                    val nextRe = twiddleRe * rootRe - twiddleIm * rootIm
                    twiddleIm = twiddleRe * rootIm + twiddleIm * rootRe
                    twiddleRe = nextRe
                }
                start += length
            }
            length = length shl 1
        }
        if (inverse) {
            for (i in 0 until size) {
                real[i] /= size.toFloat()
                imaginary[i] /= size.toFloat()
            }
        }
    }
}
