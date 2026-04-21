package com.example.breathheartdemo

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
// Split respiration and heart components, then estimate bpm and rpm.
class Processor(private val fsHz: Int) {
    private val cap = fsHz * 30
    val rawBuf = RingBuffer(cap)
    val respBuf = RingBuffer(cap)
    val hrBuf = RingBuffer(cap)

    private val respLP = MovingAverage(windowSize = fsHz * 2)
    private val hrSmooth = ShortSmoother(windowSize = maxOf(1, fsHz / 10))

    private val hrEstimator = PeakRateEstimator(
        fsHz = fsHz,
        refractoryMs = 250,
        windowSec = 10,
        minBpm = 40f,
        maxBpm = 180f
    )
    private val respEstimator = PeakRateEstimator(
        fsHz = fsHz,
        refractoryMs = 1500,
        windowSec = 30,
        minBpm = 6f,
        maxBpm = 30f
    )
    private val hrRateSmoother = RateSmoother(
        historySize = 3,
        alpha = 0.45f,
        maxStepPerUpdate = 3f
    )
    private val respRateSmoother = RateSmoother(
        historySize = 5,
        alpha = 0.25f,
        maxStepPerUpdate = 0.8f
    )
    private val respRateGate = RespRateGate(
        minAmplitude = 0.12f,
        maxJumpRpm = 4f,
        maxHoldMs = 12_000L
    )

    private val _rates = MutableStateFlow(Rates())
    val rates: StateFlow<Rates> = _rates

    private var lastHrEstimateMs = 0L
    private var lastRespEstimateMs = 0L

    fun onSample(sample: Sample) {
        val t = sample.tMs
        val x = sample.x

        val resp = respLP.next(x)
        val hrRaw = x - resp
        val hr = hrSmooth.next(hrRaw)

        rawBuf.add(t, x)
        respBuf.add(t, resp)
        hrBuf.add(t, hr)

        hrEstimator.add(t, hr)
        respEstimator.add(t, resp)

        var nextBpm = _rates.value.bpm
        var nextRpm = _rates.value.rpm

        if (t - lastHrEstimateMs >= 1000L) {
            lastHrEstimateMs = t
            nextBpm = hrRateSmoother.update(hrEstimator.estimate())
        }

        if (t - lastRespEstimateMs >= 2000L) {
            lastRespEstimateMs = t
            val rawRespRate = respEstimator.estimate()
            val respAmplitude = measureRecentAmplitude(windowSec = 12)
            val gatedRespRate = respRateGate.filter(
                nowMs = t,
                rawRate = rawRespRate,
                signalAmplitude = respAmplitude,
                hasEnoughWaveform = respBufHasEnoughData(windowSec = 10),
                currentDisplayed = _rates.value.rpm
            )
            nextRpm = respRateSmoother.update(gatedRespRate)
        }

        if (nextBpm != _rates.value.bpm || nextRpm != _rates.value.rpm) {
            _rates.value = Rates(
                bpm = nextBpm,
                rpm = nextRpm
            )
        }
    }

    fun reset() {
        rawBuf.clear()
        respBuf.clear()
        hrBuf.clear()
        hrEstimator.clear()
        respEstimator.clear()
        hrRateSmoother.clear()
        respRateSmoother.clear()
        respRateGate.clear()
        lastHrEstimateMs = 0L
        lastRespEstimateMs = 0L
        _rates.value = Rates()
    }

    private fun respBufHasEnoughData(windowSec: Int): Boolean {
        val (ts, vs) = respBuf.snapshot()
        if (vs.size < fsHz * windowSec / 2) return false
        val latestTs = ts.lastOrNull() ?: return false
        val minTs = latestTs - windowSec * 1000L
        var count = 0
        for (i in vs.indices) {
            if (ts[i] >= minTs) count++
        }
        return count >= fsHz * windowSec * 8 / 10
    }

    private fun measureRecentAmplitude(windowSec: Int): Float {
        val (ts, vs) = respBuf.snapshot()
        if (vs.isEmpty()) return 0f
        val latestTs = ts.lastOrNull() ?: return 0f
        val minTs = latestTs - windowSec * 1000L
        var minV = Float.POSITIVE_INFINITY
        var maxV = Float.NEGATIVE_INFINITY
        for (i in vs.indices) {
            if (ts[i] < minTs) continue
            val v = vs[i]
            if (v < minV) minV = v
            if (v > maxV) maxV = v
        }
        return if (minV == Float.POSITIVE_INFINITY || maxV == Float.NEGATIVE_INFINITY) {
            0f
        } else {
            maxV - minV
        }
    }
}
