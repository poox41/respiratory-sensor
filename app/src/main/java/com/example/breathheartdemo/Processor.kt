package com.example.breathheartdemo

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
// Split respiration and heart components, then estimate bpm and rpm.
class Processor(private val fsHz: Int) {
    private val cap = fsHz * 330
    val rawBuf = RingBuffer(cap)
    val respBuf = RingBuffer(cap)
    val hrBuf = RingBuffer(cap)

    // Pre-filter (0.05~8 Hz): DC remove + 8 Hz LP
    private val preLP = MovingAverage(windowSize = 8)  // 8 samples (~160ms, ~6 Hz LP)

    private val respLP = MovingAverage(windowSize = fsHz * 2)  // 100 samples (2s window)

    private val minHrAmplitude = 0.01f  // minimum HR signal amplitude
    private var isHrValid = true  // set false when amplitude too low, blocks hrBuf writes

    // Heart rate bandpass 0.8~3 Hz
    private val hrCalc = MovingAverage(windowSize = maxOf(1, fsHz / 5))  // HR calculation: 10 samples
    private val hrDisplaySmooth = DualSmoother(windowSize = 12)  // HR display: 12+12 samples

    // HR display normalization: 5s sliding window (250 samples at 50Hz)
    private val NORM_WINDOW = fsHz * 5
    private val normHistory = FloatArray(NORM_WINDOW)
    private var normIdx = 0
    private var normCount = 0

    private fun normalizeForDisplay(value: Float): Float {
        normHistory[normIdx] = value
        normIdx = (normIdx + 1) % NORM_WINDOW
        if (normCount < NORM_WINDOW) normCount++

        var minV = Float.MAX_VALUE
        var maxV = Float.MIN_VALUE
        for (i in 0 until normCount) {
            val v = normHistory[i]
            if (v < minV) minV = v
            if (v > maxV) maxV = v
        }
       val range = maxV - minV
        return (if (range > 1e-6f) ((value - minV) / range) * 2f - 1f else 0f).coerceIn(-1f, 1f)
    }
//    private val hrEstimator = PeakRateEstimator(
//        fsHz = fsHz,
//        refractoryMs = 250,
//        windowSec = 10,
//        minBpm = 40f,
//        maxBpm = 180f
//    )
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

    // ======= Signal Quality Detection =======
    enum class SignalState { MEASURING, MOTION }
    private var signalState = SignalState.MEASURING
    private var prevX = 0f
    private var lastMotionMs = 0L
    private val motionWindow = 120  // ~2.4s at 50 Hz
    private val motionThreshold = 80f  // ADC change threshold
    private val motionHist = BooleanArray(motionWindow)
    private var motionHistIdx = 0
    private var motionHistCount = 0
    private var motionExceed = 0
    private val _signalQuality = MutableStateFlow(SignalState.MEASURING)
    private var validHrCount = 0
    private var validRespCount = 0
    val signalQuality: StateFlow<SignalState> = _signalQuality
    private val _peakTimes = MutableStateFlow<List<Long>>(emptyList())
    val peakTimes: StateFlow<List<Long>> = _peakTimes
    private val peakTimesList = mutableListOf<Long>()  // mutable list for peak tracking
    private val hrEstimator = PeakRateEstimator(
        fsHz = fsHz,
        refractoryMs = 250,
        windowSec = 10,
        minBpm = 40f,
        maxBpm = 180f
    )
    private var lastValidBpm: Float? = null
    private var peakRising = false
    private var lastPeakT = 0L

    // DC removal: slow exponential-moving-average
    private var dcOffset = 0f
    private val dcAlpha = 0.995f // ~4 s time constant at 50 Hz
   var sensorLogger: SensorDataLogger? = null
   // Rolling buffers for live preview (raw vs centered)
    private val recentRaw = IntArray(16)
    private var rawIdx = 0
    private var rawCount = 0
    private val recentCentered = IntArray(16)
    private var centeredIdx = 0
    private var centeredCount = 0

    private val _rawPreview = MutableStateFlow("")
    val rawPreview: StateFlow<String> = _rawPreview
    private val _centeredPreview = MutableStateFlow("")
    val centeredPreview: StateFlow<String> = _centeredPreview

    private var lastHrEstimateMs = 0L
   private var lastRespEstimateMs = 0L
    private var lastBpmEstimate: Float? = null
    private var lastRpmEstimate: Float? = null

    fun onSample(sample: Sample) {
        val t = sample.tMs
        val rawX = sample.x


        // ---------- DC removal ----------

        dcOffset = dcOffset * dcAlpha + rawX * (1f - dcAlpha)
        val x = rawX - dcOffset // centered around 0
        
        val xF = preLP.next(x)  // 8 Hz low-pass pre-filter
        sensorLogger?.log(t, rawX, x, xF)
        val diff = kotlin.math.abs(x - prevX)
        prevX = x
        if (motionHist[motionHistIdx]) motionExceed--
        motionHist[motionHistIdx] = diff > motionThreshold
        if (motionHist[motionHistIdx]) motionExceed++
        motionHistIdx = (motionHistIdx + 1) % motionWindow
        if (motionHistCount < motionWindow) motionHistCount++
        val ratio = if (motionHistCount > 0) motionExceed.toFloat() / motionHistCount else 0f
        if (ratio > 0.3f) {
            lastMotionMs = t
            signalState = SignalState.MOTION
            _signalQuality.value = SignalState.MOTION
        } else if (t - lastMotionMs > 3000L && signalState != SignalState.MEASURING) {
            signalState = SignalState.MEASURING
            _signalQuality.value = SignalState.MEASURING
        }

        // Track raw and centered values for the live preview
        recentRaw[rawIdx] = rawX.toInt()
        rawIdx = (rawIdx + 1) % 16
        if (rawCount < 16) rawCount++

        recentCentered[centeredIdx] = xF.toInt()
        centeredIdx = (centeredIdx + 1) % 16
        if (centeredCount < 16) centeredCount++

        _rawPreview.value = (0 until rawCount).joinToString(" ") { i ->
            val v = recentRaw[(rawIdx - rawCount + i + 16) % 16]
            if (v >= 0) " $v" else "$v"
        }
        _centeredPreview.value = (0 until centeredCount).joinToString(" ") { i ->
            val v = recentCentered[(centeredIdx - centeredCount + i + 16) % 16]
            if (v >= 0) " $v" else "$v"
        }

        // --- Respiration: extract via long MA (retain 0.1~0.5 Hz) ---
        val resp = respLP.next(xF)

        // --- Heart rate: subtract respiration, split into calc + display ---
        val hrRaw = xF - resp
        val hr = hrCalc.next(hrRaw)              // calculation channel (light MA, 10 samples)
        val hrDisplay = hrDisplaySmooth.next(hrRaw)  // display channel (DualSmoother, 12+12 samples)

        rawBuf.add(t, xF)       // pre-filtered signal
        respBuf.add(t, resp)    // respiration waveform
        val normHr = normalizeForDisplay(hrDisplay)
        hrBuf.add(t, normHr)    // heart rate waveform (-1~1)

        if (signalState == SignalState.MEASURING) {
            hrEstimator.add(t, hr)
            respEstimator.add(t, resp)

            // Threshold-based peak detection on normalized HR (-1 to 1)
            if (normHr > 0.3f && !peakRising && (t - lastPeakT) > 300L) {
                peakRising = true
                lastPeakT = t
                peakTimesList.add(t)
                if (peakTimesList.size > 50) peakTimesList.removeAt(0)
                _peakTimes.value = peakTimesList.toList()
            }
            if (normHr < -0.1f) peakRising = false

            // Calculate BPM from last two peak intervals
            if (peakTimesList.size >= 2) {
                val interval = t - peakTimesList[peakTimesList.size - 2]
                if (interval > 0L) {
                    val bpm = 60000f / interval
                    if (bpm in 40f..180f) lastValidBpm = bpm
                }
            }
        }



        var nextBpm = lastBpmEstimate
        var nextRpm = lastRpmEstimate


        // ---- Vital signs: slow-decay BPM/RPM validation ----
        // BPM from DFT estimator (PeakRateEstimator), every 1s
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

        val bpmValid = nextBpm != null && nextBpm >= 40f && nextBpm <= 180f
        val rpmValid = nextRpm != null && nextRpm >= 8f && nextRpm <= 30f
        validHrCount = if (bpmValid) minOf(validHrCount + 1, 10) else maxOf(validHrCount - 1, 0)
        validRespCount = if (rpmValid) minOf(validRespCount + 1, 10) else maxOf(validRespCount - 1, 0)
        val hrReliable = validHrCount >= 3
        val respReliable = validRespCount >= 3
        if (hrReliable || respReliable) {
        if (nextBpm != _rates.value.bpm || nextRpm != _rates.value.rpm) {
                _rates.value = Rates(bpm = nextBpm, rpm = nextRpm)
        }
        }


        lastBpmEstimate = nextBpm
        lastRpmEstimate = nextRpm
    }
    fun reset() {
        rawBuf.clear()
        respBuf.clear()
        hrBuf.clear()
        dcOffset = 0f
        isHrValid = false
        normIdx = 0; normCount = 0
        rawIdx = 0; rawCount = 0
        centeredIdx = 0; centeredCount = 0
        _rawPreview.value = ""
        _centeredPreview.value = ""
        sensorLogger = null
        hrEstimator.clear()
        respEstimator.clear()
        hrRateSmoother.clear()
        respRateSmoother.clear()
        respRateGate.clear()
        lastHrEstimateMs = 0L
       lastRespEstimateMs = 0L
        _rates.value = Rates()
        lastBpmEstimate = null
        lastRpmEstimate = null
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

    private fun measureHrAmplitude(windowSec: Int): Float {
        val (ts, vs) = hrBuf.snapshot()
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
        return if (minV == Float.POSITIVE_INFINITY || maxV == Float.NEGATIVE_INFINITY) 0f else maxV - minV
    }
}






























