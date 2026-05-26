package com.example.breathheartdemo

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sqrt

class SleepFeatureExtractor {
    fun extractEpoch(heart50Hz: FloatArray, respiration50Hz: FloatArray): FloatArray {
        val ecg = resampleLinear(heart50Hz, sourceHz = 50, targetHz = 100)
        val resp = resampleLinear(respiration50Hz, sourceHz = 50, targetHz = 25)
        val out = FloatArray(FEATURE_DIM) { 0f }

        out[0] = mean(ecg)
        out[1] = std(ecg)
        out[2] = robustRange(ecg)
        out[3] = flatRatio(ecg)
        out[4] = zeroCrossingRate(ecg)
        out[5] = mean(resp)
        out[6] = std(resp)
        out[7] = robustRange(resp)
        out[8] = flatRatio(resp)
        out[9] = zeroCrossingRate(resp)

        val ecgPeaks = detectPeaks(
            values = center(ecg),
            fsHz = 100f,
            minDistanceSec = 0.45f,
            prominenceScale = 0.35f,
            stdScale = 0.25f,
            square = true
        )
        fillHrFeatures(out, ecgPeaks, fsHz = 100f)

        val respCentered = center(resp)
        val respPeaks = detectPeaks(
            values = respCentered,
            fsHz = 25f,
            minDistanceSec = 1.0f,
            prominenceScale = 0.20f,
            stdScale = 0.15f,
            square = false
        )
        fillRespFeatures(out, respCentered, respPeaks, fsHz = 25f)

        for (i in out.indices) {
            if (!out[i].isFinite()) out[i] = 0f
        }
        return out
    }

    private fun fillHrFeatures(out: FloatArray, peaks: IntArray, fsHz: Float) {
        out[12] = peaks.size.toFloat()
        val rrAll = intervals(peaks, fsHz)
        val rr = rrAll.filter { it in 0.4f..1.5f }.toFloatArray()
        if (rr.size < 2) {
            out[24] = rr.size.toFloat()
            out[25] = rrAll.size.toFloat()
            return
        }

        val hr = rr.map { 60f / max(it, 1e-6f) }.toFloatArray()
        out[10] = mean(hr)
        out[11] = std(hr)
        out[13] = mean(rr)
        out[14] = std(rr)
        out[15] = median(rr)
        out[16] = percentile(rr, 75f) - percentile(rr, 25f)
        out[17] = std(rr)

        val diff = diffs(rr)
        out[18] = if (diff.isNotEmpty()) sqrt(mean(diff.map { it * it }.toFloatArray())) else 0f
        out[19] = if (diff.isNotEmpty()) diff.count { abs(it) > 0.02f }.toFloat() / diff.size else 0f
        out[20] = if (diff.isNotEmpty()) diff.count { abs(it) > 0.05f }.toFloat() / diff.size else 0f
        out[21] = minValue(hr)
        out[22] = maxValue(hr)
        out[23] = out[22] - out[21]
        out[24] = rr.size.toFloat()
        out[25] = rrAll.size.toFloat()
        out[26] = rr.size.toFloat() / max(1, rrAll.size)
        out[27] = 1f - out[26]
    }

    private fun fillRespFeatures(out: FloatArray, resp: FloatArray, peaks: IntArray, fsHz: Float) {
        out[30] = peaks.size.toFloat()
        val cycleAll = intervals(peaks, fsHz)
        val cycle = cycleAll.filter { it in 1.0f..10.0f }.toFloatArray()
        if (cycle.size >= 2) {
            val rate = cycle.map { 60f / max(it, 1e-6f) }.toFloatArray()
            out[28] = mean(rate)
            out[29] = std(rate)
            out[31] = std(cycle)
            out[32] = percentile(cycle, 75f) - percentile(cycle, 25f)
            out[36] = median(rate)
            out[37] = percentile(rate, 75f) - percentile(rate, 25f)
            out[38] = minValue(rate)
            out[39] = maxValue(rate)
            out[40] = out[39] - out[38]
            out[41] = std(rate) / (mean(rate) + 1e-6f)
            out[42] = mean(cycle)
            out[43] = std(cycle)
            out[44] = std(cycle) / (mean(cycle) + 1e-6f)
            val cycleDiff = diffs(cycle)
            out[45] = if (cycleDiff.isNotEmpty()) sqrt(mean(cycleDiff.map { it * it }.toFloatArray())) else 0f
            out[46] = if (cycleDiff.isNotEmpty()) cycleDiff.count { abs(it) > 0.20f }.toFloat() / cycleDiff.size else 0f
            if (out[36].isFinite() && out[36] > 0f && out[10].isFinite()) {
                out[56] = out[10] / (out[36] + 1e-6f)
            }
        }

        out[33] = percentile(resp, 75f) - percentile(resp, 25f)
        val d = diffs(resp)
        out[34] = std(d)
        out[35] = robustRange(d)

        val amps = chunkAmplitudes(resp, chunks = 10)
        if (amps.size >= 2) {
            out[47] = mean(amps)
            out[48] = std(amps)
            out[49] = out[48] / (out[47] + 1e-6f)
            out[50] = percentile(amps, 75f) - percentile(amps, 25f)
            out[51] = maxValue(amps) - minValue(amps)
            val lowThr = max(percentile(amps, 20f), 1e-12f)
            out[52] = amps.count { it <= lowThr }.toFloat() / amps.size
        }

        val absDiff = d.map { abs(it) }.toFloatArray()
        if (absDiff.isNotEmpty()) {
            val strictThr = max(std(resp) * 0.005f, 1e-8f)
            out[53] = absDiff.count { it < strictThr }.toFloat() / absDiff.size
            val win = max(1, (fsHz * 2f).toInt())
            val step = max(1, win / 2)
            val globalAmp = robustRange(resp)
            val pauseAmpThr = max(globalAmp * 0.10f, 1e-12f)
            var pauseCount = 0
            var total = 0
            var start = 0
            while (start + win <= resp.size) {
                val local = resp.copyOfRange(start, start + win)
                total++
                if (robustRange(local) <= pauseAmpThr) pauseCount++
                start += step
            }
            out[54] = pauseCount.toFloat()
            out[55] = if (total > 0) pauseCount.toFloat() / total else 0f
        }
    }

    private fun detectPeaks(
        values: FloatArray,
        fsHz: Float,
        minDistanceSec: Float,
        prominenceScale: Float,
        stdScale: Float,
        square: Boolean
    ): IntArray {
        if (values.size < (fsHz * if (square) 3f else 5f).toInt()) return IntArray(0)
        val y = if (square) values.map { it * it }.toFloatArray() else values
        if (std(y) < 1e-12f) return IntArray(0)
        val minDistance = max(1, (fsHz * minDistanceSec).toInt())
        val prominence = max(
            max(
                (percentile(y, 90f) - percentile(y, 50f)) * prominenceScale,
                std(y) * stdScale
            ),
            1e-12f
        )

        val peaks = ArrayList<Int>()
        var lastPeak = -minDistance
        for (i in 1 until y.lastIndex) {
            if (i - lastPeak < minDistance) continue
            if (y[i] > y[i - 1] && y[i] >= y[i + 1]) {
                val left = max(0, i - minDistance)
                val right = min(y.lastIndex, i + minDistance)
                val base = max(minValue(y.copyOfRange(left, i + 1)), minValue(y.copyOfRange(i, right + 1)))
                if (y[i] - base >= prominence) {
                    peaks.add(i)
                    lastPeak = i
                }
            }
        }
        return peaks.toIntArray()
    }

    private fun resampleLinear(values: FloatArray, sourceHz: Int, targetHz: Int): FloatArray {
        if (values.isEmpty() || sourceHz == targetHz) return values.copyOf()
        val outSize = max(1, values.size * targetHz / sourceHz)
        val out = FloatArray(outSize)
        val scale = (values.size - 1).toFloat() / max(1, outSize - 1)
        for (i in out.indices) {
            val pos = i * scale
            val left = pos.toInt().coerceIn(0, values.lastIndex)
            val right = min(values.lastIndex, left + 1)
            val frac = pos - left
            out[i] = values[left] * (1f - frac) + values[right] * frac
        }
        return out
    }

    private fun center(values: FloatArray): FloatArray {
        val med = median(values)
        return values.map { if (it.isFinite()) it - med else 0f }.toFloatArray()
    }

    private fun intervals(peaks: IntArray, fsHz: Float): FloatArray {
        if (peaks.size < 2) return FloatArray(0)
        return FloatArray(peaks.size - 1) { i -> (peaks[i + 1] - peaks[i]) / fsHz }
    }

    private fun diffs(values: FloatArray): FloatArray {
        if (values.size < 2) return FloatArray(0)
        return FloatArray(values.size - 1) { i -> values[i + 1] - values[i] }
    }

    private fun chunkAmplitudes(values: FloatArray, chunks: Int): FloatArray {
        val chunkLen = values.size / chunks
        if (chunkLen < 5) return FloatArray(0)
        return FloatArray(chunks) { i ->
            val seg = values.copyOfRange(i * chunkLen, (i + 1) * chunkLen)
            percentile(seg, 95f) - percentile(seg, 5f)
        }
    }

    private fun mean(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        var sum = 0f
        var count = 0
        for (v in values) {
            if (v.isFinite()) {
                sum += v
                count++
            }
        }
        return if (count == 0) 0f else sum / count
    }

    private fun std(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        val m = mean(values)
        var sum = 0f
        var count = 0
        for (v in values) {
            if (v.isFinite()) {
                val d = v - m
                sum += d * d
                count++
            }
        }
        return if (count == 0) 0f else sqrt(sum / count)
    }

    private fun robustRange(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        return percentile(values, 95f) - percentile(values, 5f)
    }

    private fun zeroCrossingRate(values: FloatArray): Float {
        if (values.size < 2) return 0f
        var count = 0
        for (i in 1 until values.size) {
            if (sign(values[i]) != sign(values[i - 1])) count++
        }
        return count.toFloat() / (values.size - 1)
    }

    private fun flatRatio(values: FloatArray): Float {
        if (values.size < 2) return 1f
        val eps = max(std(values) * 1e-3f, 1e-12f)
        var count = 0
        for (i in 1 until values.size) {
            if (abs(values[i] - values[i - 1]) < eps) count++
        }
        return count.toFloat() / (values.size - 1)
    }

    private fun percentile(values: FloatArray, percentile: Float): Float {
        val sorted = values.filter { it.isFinite() }.sorted()
        if (sorted.isEmpty()) return 0f
        val pos = (percentile / 100f) * (sorted.size - 1)
        val left = pos.toInt()
        val right = min(sorted.lastIndex, left + 1)
        val frac = pos - left
        return sorted[left] * (1f - frac) + sorted[right] * frac
    }

    private fun median(values: FloatArray): Float = percentile(values, 50f)

    private fun minValue(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        var out = Float.POSITIVE_INFINITY
        for (v in values) if (v.isFinite() && v < out) out = v
        return if (out.isFinite()) out else 0f
    }

    private fun maxValue(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        var out = Float.NEGATIVE_INFINITY
        for (v in values) if (v.isFinite() && v > out) out = v
        return if (out.isFinite()) out else 0f
    }

    companion object {
        const val FEATURE_DIM = 57
    }
}
