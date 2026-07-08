package com.example.breathheartdemo

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
// Split respiration and heart components, then estimate bpm and rpm.
class Processor(private val fsHz: Int) {
    private val cap = fsHz * 330
   val rawBuf = RingBuffer(cap)
   val preBuf = RingBuffer(cap)
  val respBuf = RingBuffer(cap)
    val respDisplayBuf = RingBuffer(cap)
   val hrBuf = RingBuffer(cap)

    // Pre-filter (0.05~8 Hz): DC remove + 8 Hz LP
    private val preLP = MovingAverage(windowSize = 8)  // 8 samples (~160ms, ~6 Hz LP)
    private val notch50Hz = NotchFilter(freqHz = 50f, sampleRate = fsHz, q = 30f)  // 50Hz工频陷波
    private val antiAliasLPF = LowPassFilter(cutoffHz = 5f, sampleRate = fsHz)  // 5Hz低通抗混叠

    private val respFIR = FIRFilter(order = 633, cutoffHz = 0.5f, sampleRate = fsHz)  // 633阶FIR低通
    private val hrHPF = HighPassFilter(alpha = 0.964f)     // ~0.3 Hz cutoff, removes breathing residue
    private val hrHPF2 = HighPassFilter(alpha = 0.964f)    // 二级级联

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
    private var smoothHrCenter = 0f
    private var smoothHrP2P = 0f
    // Respiration display normalization: 10s window, EMA p2p smoothing (α=0.08)
    private val RESP_NORM_WINDOW = fsHz * 10
    private val respNormHistory = FloatArray(RESP_NORM_WINDOW)
    private var respNormIdx = 0
    private var respNormCount = 0
    private var smoothRespP2P = 0f
    private var smoothRespCenter = 0f

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
        val center = (minV + maxV) / 2f
        val rawP2P = maxV - minV
        // EMA smooth center (α=0.05)
        if (normCount == 1) smoothHrCenter = center
        else smoothHrCenter += (center - smoothHrCenter) * 0.05f
        // P2P: rise immediately, fall slowly (α=0.08)
        if (smoothHrP2P == 0f || rawP2P > smoothHrP2P) {
            smoothHrP2P = rawP2P
        } else {
            smoothHrP2P += (rawP2P - smoothHrP2P) * 0.08f
        }
        val displayP2P = smoothHrP2P.coerceIn(30f, 200f)
        val halfP2P = displayP2P / 2f
        return if (halfP2P > 1e-6f) {
            ((value - smoothHrCenter) / halfP2P).coerceIn(-1f, 1f)
        } else 0f
    }
    private fun normalizeRespForDisplay(value: Float): Float {
        respNormHistory[respNormIdx] = value
        respNormIdx = (respNormIdx + 1) % RESP_NORM_WINDOW
        if (respNormCount < RESP_NORM_WINDOW) respNormCount++

        var minV = Float.MAX_VALUE
        var maxV = Float.MIN_VALUE
        for (i in 0 until respNormCount) {
            val v = respNormHistory[i]
            if (v < minV) minV = v
            if (v > maxV) maxV = v
        }
        val rawP2P = maxV - minV
        val center = (minV + maxV) / 2f
        if (respNormCount == 1) smoothRespCenter = center
        else smoothRespCenter += (center - smoothRespCenter) * 0.08f

        if (smoothRespP2P == 0f || rawP2P > smoothRespP2P) {
            smoothRespP2P = rawP2P   // 上升立即跟随，避免削顶
        } else {
            smoothRespP2P += (rawP2P - smoothRespP2P) * 0.08f  // 下降缓慢衰减
        }
        val displayP2P = smoothRespP2P.coerceIn(80f, 3000f)
        val halfP2P = displayP2P / 2f

        return if (halfP2P > 1e-6f) {
            ((value - smoothRespCenter) / halfP2P).coerceIn(-1f, 1f)
        } else 0f
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
        windowSec = 20,
        minBpm = 40f,
        maxBpm = 180f
    )
    private var lastValidBpm: Float? = null
    private var peakRising = false
    private var lastPeakT = 0L
    private val motionAbsBuf = RingBuffer(fsHz * 2)
    private var motionFlag = false
    private var motionEndTime = 0L
    private val breathDetector = BreathingRateDetector()
    private val heartSep = DifferentialHeartSeparator()
    private val hrPeakDetector = DualPeakDetector(threshold = 41f)
    private var lastPeakBpm: Float? = null

    // DC removal: slow exponential-moving-average
    private var dcOffset = 0f
    private val dcAlpha = 0.999f // ~20 s time constant at 50 Hz
    private var sampleCount = 0L
    private var noSignalCount = 0
    private var signalRecoveryCount = 0
    private val initialBaseline = MovingAverage(windowSize = fsHz * 2)  // 100 samples, 2s window
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
    private var lastValidAdc = 0f

    fun onSample(sample: Sample) {
        val t = sample.tMs
        val rawX = sample.x
        // val validatedRawX = if (rawX == 0f || rawX >= 4094.5f) lastValidAdc else rawX.also { lastValidAdc = it }
        val validatedRawX = rawX


        // 固定去直流：2048 = 0V（12位ADC中点）
        val x = validatedRawX - 2600f
        motionAbsBuf.add(t, kotlin.math.abs(x))
        if (!motionFlag) {
            val (_, absVs) = motionAbsBuf.snapshot()
            val minT = t - 1000L; var peakAbs = 0f
            for (i in absVs.indices) { if (absVs[i] >= minT && absVs[i] > peakAbs) peakAbs = absVs[i] }
            if (peakAbs > 1638f) { motionFlag = true; motionEndTime = t + 15000L }
        } else if (t >= motionEndTime) {
            val (_, absVs) = motionAbsBuf.snapshot()
            val minT = t - 2000L; var stablePeak = 0f
            for (i in absVs.indices) { if (absVs[i] >= minT && absVs[i] > stablePeak) stablePeak = absVs[i] }
            if (stablePeak < 819f) motionFlag = false
        }
        if (x in -100f..100f) {
            noSignalCount++
            signalRecoveryCount = 0
        } else {
            signalRecoveryCount++
            if (signalRecoveryCount >= 20) {
                noSignalCount = 0
            }
        }
        
        val xNotch = notch50Hz.next(x)          // 50Hz工频陷波
        val xClean = antiAliasLPF.next(xNotch)  // 5Hz低通抗混叠
        val xF = xClean  // 跳过 preLP，直接使用5Hz低通输出

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
        val resp = respFIR.next(xF)

        // --- Heart rate: subtract respiration, split into calc + display ---
        val hrRaw = heartSep.next(xF)
        val hr = hrCalc.next(hrRaw)              // calculation channel (light MA, 10 samples)
        val hrDisplay = hrDisplaySmooth.next(hrRaw)  // display channel (DualSmoother, 12+12 samples)

        val normHr = normalizeForDisplay(hrDisplay)  // 用于峰值检测，不写入缓冲区
        sensorLogger?.log(t, rawX, x, xF, resp, hr, normHr)  // normHr列
        val signalActive = noSignalCount < fsHz * 10
        rawBuf.add(t, if (signalActive) x else 0f)  // DC-removed ADC waveform
        preBuf.add(t, if (signalActive) xF else 0f)       // pre-filtered signal
        respBuf.add(t, if (signalActive) resp else 0f)    // respiration waveform
        respDisplayBuf.add(t, if (signalActive) resp else 0f)  // 原始呼吸信号，无归一化
        hrBuf.add(t, if (signalActive) hrDisplay else 0f)    // 原始心率信号，无归一化

        if (signalState == SignalState.MEASURING && signalActive && !motionFlag) {
            // Dual peak detection on heartSep output (firstPeak + envelope + secondPeak → AO → BPM)
            val peakBpm = hrPeakDetector.next(hrRaw, t)
            if (peakBpm != null) lastPeakBpm = peakBpm

            // Threshold-based peak detection on normalized HR (-1 to 1)
            if (normHr > 0.3f && !peakRising && (t - lastPeakT) > 350L) {
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
        // BPM from peak detector (replaces DFT)
        if (t - lastHrEstimateMs >= 1000L) {
            lastHrEstimateMs = t
            nextBpm = hrRateSmoother.update(lastPeakBpm)
        }

        val newRpm = breathDetector.next(resp, t)
        if (newRpm != null) {
            val respAmplitude = measureRecentAmplitude(windowSec = 12)
            val gatedRpm = respRateGate.filter(
                nowMs = t, rawRate = newRpm,
                signalAmplitude = respAmplitude,
                hasEnoughWaveform = respBufHasEnoughData(windowSec = 10),
                currentDisplayed = _rates.value.rpm
            )
            nextRpm = respRateSmoother.update(gatedRpm)
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
        preBuf.clear()
       respBuf.clear()
        respDisplayBuf.clear()
        hrBuf.clear()
        dcOffset = 0f
        sampleCount = 0
        noSignalCount = 0
        signalRecoveryCount = 0
        initialBaseline.clear()
        isHrValid = false
        normIdx = 0; normCount = 0
        smoothHrCenter = 0f; smoothHrP2P = 0f
        respNormIdx = 0; respNormCount = 0; smoothRespP2P = 0f; smoothRespCenter = 0f
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
        hrHPF.clear()
        notch50Hz.clear()
        antiAliasLPF.clear()
        heartSep.clear()
        hrPeakDetector.clear()
        motionFlag = false; motionEndTime = 0L; motionAbsBuf.clear()
        breathDetector.clear()
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






























