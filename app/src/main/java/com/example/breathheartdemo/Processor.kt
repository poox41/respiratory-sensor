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
   // UI-only waveform from the validated steeper heart-band experiment.
   // hrBuf remains unchanged because it is also consumed by the sleep model.
   val cleanHeartBuf = RingBuffer(cap)
   // Beat-synchronous averaged/template-reconstructed signal for display only.
   val enhancedHeartBuf = RingBuffer(cap)

    // Pre-filter (0.05~8 Hz): DC remove + 8 Hz LP
    private val preLP = MovingAverage(windowSize = 8)  // 8 samples (~160ms, ~6 Hz LP)
    private val notch50Hz = NotchFilter(freqHz = 50f, sampleRate = fsHz, q = 30f)  // 50Hz工频陷波
    private val antiAliasLPF = LowPassFilter(cutoffHz = 8f, sampleRate = fsHz)  // 8Hz低通抗混叠
    private val dcBlocker = DcBlocker()

    private val respFIR = FIRFilter(RESP_FIR_COEFFS)  // 633阶FIR低通
    private val hrHPF = HighPassFilter(alpha = 0.964f)     // ~0.3 Hz cutoff, removes breathing residue
    private val hrHPF2 = HighPassFilter(alpha = 0.939f)   // ~0.5 Hz cutoff, suppress residual breathing

    private val minHrAmplitude = 0.01f  // minimum HR signal amplitude
    private var isHrValid = true  // set false when amplitude too low, blocks hrBuf writes

    // Heart rate bandpass 0.8~3 Hz
    private val hrCalc = MovingAverage(windowSize = maxOf(1, fsHz / 5))  // HR calculation: 10 samples
    private val hrDisplaySmooth = DualSmoother(windowSize = 12)
    private val hrPostLPF = LowPassFilter(cutoffHz = 3f, sampleRate = fsHz)  // 3Hz LPF, suppress >2.5Hz noise

    // Parallel diagnostic channel: direct 0.8~4 Hz band-pass, without the differential separator.
    // It is logged for comparison only and does not affect the current UI or BPM output.
    private val candidateHeartHPF = BiquadFilter.highPass(cutoffHz = 0.8f, sampleRate = fsHz)
    private val candidateHeartLPF = BiquadFilter.lowPass(cutoffHz = 4f, sampleRate = fsHz)
    private val candidateHeartSmooth = MovingAverage(windowSize = 3)
    private val candidateEnvelopeSmooth = MovingAverage(windowSize = maxOf(1, fsHz / 5))

    // Official heart waveform path: fourth-order 0.8~4 Hz band-pass.
    // This is the source for the UI waveform, grouped beat detector and formal BPM.
    private val cleanHeartHPF1 = BiquadFilter.highPass(cutoffHz = 0.8f, sampleRate = fsHz)
    private val cleanHeartHPF2 = BiquadFilter.highPass(cutoffHz = 0.8f, sampleRate = fsHz)
    private val cleanHeartLPF1 = BiquadFilter.lowPass(cutoffHz = 4f, sampleRate = fsHz)
    private val cleanHeartLPF2 = BiquadFilter.lowPass(cutoffHz = 4f, sampleRate = fsHz)
    private val cleanHeartSmooth = MovingAverage(windowSize = 3)
    private val cleanHeartEnvelopeSmooth = MovingAverage(windowSize = maxOf(1, fsHz))
    private val heartbeatTemplateEnhancer = HeartbeatTemplateEnhancer(fsHz = fsHz)
    // Observation only: logged for calibration, never gates signals or rates.
    private val presenceObserver = PresenceObserver(fsHz = fsHz)
    private val _presenceObservation = MutableStateFlow(PresenceObserver.Observation())
    val presenceObservation: StateFlow<PresenceObserver.Observation> = _presenceObservation

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
            smoothHrP2P += (rawP2P - smoothHrP2P) * 0.12f
        }
        val displayP2P = smoothHrP2P.coerceIn(30f, 1200f)
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
            smoothRespP2P += (rawP2P - smoothRespP2P) * 0.12f  // 下降缓慢衰减
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
        alpha = 0.4f,
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
    private val _cleanPeakTimes = MutableStateFlow<List<Long>>(emptyList())
    val cleanPeakTimes: StateFlow<List<Long>> = _cleanPeakTimes
    private val cleanPeakTimesList = mutableListOf<Long>()
    private val _heartTemplateStatus = MutableStateFlow(HeartTemplateEnhancement())
    val heartTemplateStatus: StateFlow<HeartTemplateEnhancement> = _heartTemplateStatus
    private val hrEstimator = PeakRateEstimator(
        fsHz = fsHz,
        refractoryMs = 250,
        windowSec = 20,
        minBpm = 40f,
        maxBpm = 180f
    )
    private var lastValidBpm: Float? = null
    private var lastValidBpmWasNull = true
    private var smoothBpm: Float? = null
    private var zeroCrossBpm: Float? = null
    private var prevHrRaw = 0f
    private var lastHrZeroT = 0L
    private var peakRising = false
    private var lastPeakT = 0L
    private val motionAbsBuf = RingBuffer(fsHz * 2)
    private val motionRawBuf = RingBuffer(fsHz * 2)
    private var motionFlag = false
    private var motionEndTime = 0L
    private var lastBypassMotionDetection = false
    private val breathDetector = BreathingRateDetector()
    private val zeroCrossRpm = ZeroCrossingRpm()
    private val respCycleRateDetector = RespCycleRateDetector()
    // Legacy A/B diagnostics below are still logged, but they do not drive the UI or formal BPM.
    private val heartSep = DifferentialHeartSeparator()
    private val hrPeakDetector = DualPeakDetector()
    private val windowedHeartPeakDetector = WindowedHeartPeakDetector()
    private val heartPeriodicityEstimator = HeartPeriodicityEstimator(fsHz = fsHz)
    private val cleanHeartPeriodicityEstimator = HeartPeriodicityEstimator(fsHz = fsHz)
    private val cleanHeartBeatDetector = WindowedHeartPeakDetector(
        threshold = 0.8f,
        groupWindowMs = 250L,
        refractoryMs = 550L
    )
    private var lastCleanPeakBpm: Float? = null
    private var lastCleanPeakBpmTimeMs = 0L
    private var prevRawX = 2600f
    private var lastPeakBpm: Float? = null

    // DC removal: slow exponential-moving-average
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

    fun onSample(sample: Sample, bypassMotionDetection: Boolean = false) {
        val t = sample.tMs
        val rawX = sample.x
        val validatedRawX = rawX

        // Mock data intentionally has a much larger respiration amplitude than
        // the real-sensor motion threshold. Reset the motion state when the
        // source mode changes, then bypass motion rejection for mock samples.
        if (bypassMotionDetection != lastBypassMotionDetection) {
            motionFlag = false
            motionEndTime = 0L
            motionAbsBuf.clear()
            motionRawBuf.clear()
            prevRawX = rawX
            _signalQuality.value = SignalState.MEASURING
            lastBypassMotionDetection = bypassMotionDetection
        }

        val x = validatedRawX
        // 体动检测：基于相邻采样点的差值（rawX变化率），不依赖基线
        if (!bypassMotionDetection) {
            val diff = kotlin.math.abs(rawX - prevRawX)
            prevRawX = rawX
            motionAbsBuf.add(t, diff)
            motionRawBuf.add(t, rawX)
            if (!motionFlag) {
                val (motionTs, motionValues) = motionAbsBuf.snapshot()
                val minT = t - 2000L
                var peakDiff = 0f
                var rapidChangeCount = 0
                for (i in motionValues.indices) {
                    if (motionTs[i] >= minT) {
                        if (motionValues[i] > peakDiff) peakDiff = motionValues[i]
                        if (motionValues[i] > 160f) rapidChangeCount++
                    }
                }
                val (rawMotionTs, rawMotionValues) = motionRawBuf.snapshot()
                val rawMinT = t - 2000L
                var rawMin = Float.POSITIVE_INFINITY
                var rawMax = Float.NEGATIVE_INFINITY
                var rawMotionCount = 0
                for (i in rawMotionValues.indices) {
                    if (rawMotionTs[i] >= rawMinT) {
                        val value = rawMotionValues[i]
                        if (value < rawMin) rawMin = value
                        if (value > rawMax) rawMax = value
                        rawMotionCount++
                    }
                }
                val rawRange = if (rawMotionCount >= fsHz * 2 && rawMin <= rawMax) rawMax - rawMin else 0f
                // Large but smooth abdominal respiration is not motion. Enter
                // motion state only for a sharp jump, or when a large pressure
                // excursion also contains several rapid changes.
                val suddenMotion = peakDiff > 240f
                val irregularLargeMotion = rawRange > 1200f && rapidChangeCount >= 3
                if (suddenMotion || irregularLargeMotion) {
                    motionFlag = true
                    motionEndTime = t + 3000L
                    _signalQuality.value = SignalState.MOTION
                }
            } else if (t >= motionEndTime) {
                val (motionTs, motionValues) = motionAbsBuf.snapshot()
                val minT = t - 2000L
                var stableDiff = 0f
                var rapidChangeCount = 0
                for (i in motionValues.indices) {
                    if (motionTs[i] >= minT) {
                        if (motionValues[i] > stableDiff) stableDiff = motionValues[i]
                        if (motionValues[i] > 160f) rapidChangeCount++
                    }
                }
                // Recovery depends on the absence of rapid/irregular changes,
                // not on total pressure range, so smooth breathing can resume.
                if (stableDiff < 200f && rapidChangeCount <= 1) {
                    motionFlag = false
                    _signalQuality.value = SignalState.MEASURING
                }
            }
            // 体动期间跳过所有信号处理（滤波器、缓冲区写入、特征提取全部暂停）
            if (motionFlag) return
        }
        // ADC饱和保护：跳过近满量程或零的采样点，避免平顶信号污染滤波器
        if (rawX >= 4085f || rawX <= 10f) return

        val xDc = dcBlocker.next(x)             // 去直流
        val xNotch = notch50Hz.next(xDc)          // 50Hz工频陷波
        val xClean = antiAliasLPF.next(xNotch)  // 8Hz低通抗混叠
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
        val hrF1 = hrHPF.next(hrRaw)   // 0.3 Hz HPF
        val hrF = hrHPF2.next(hrF1)
        val hrFilt = hrPostLPF.next(hrF)   // 3Hz LPF, suppress >2.5Hz noise
        val hr = hrCalc.next(hrFilt)              // calculation channel (light MA, 10 samples)
        val hrDisplay = hrDisplaySmooth.next(hrFilt)  // display channel (DualSmoother, 12+12 samples)
        val heartCandidateRaw = candidateHeartLPF.next(candidateHeartHPF.next(xF))
        val heartCandidate = candidateHeartSmooth.next(heartCandidateRaw)
        val heartEnvelope = candidateEnvelopeSmooth.next(kotlin.math.abs(heartCandidateRaw))
        val cleanHeartRaw = cleanHeartLPF2.next(
            cleanHeartLPF1.next(cleanHeartHPF2.next(cleanHeartHPF1.next(xF)))
        )
        val cleanHeart = cleanHeartSmooth.next(cleanHeartRaw)
        val cleanHeartEnvelope = cleanHeartEnvelopeSmooth.next(kotlin.math.abs(cleanHeartRaw))
        val presenceObservation = presenceObserver.next(cleanHeart, cleanHeartEnvelope, t)
        _presenceObservation.value = presenceObservation
        // 过零检测BPM：独立于normHr和DualSmoother，基于hrRaw的上升沿过零
        if (prevHrRaw < 0f && hrF >= 0f && (t - lastHrZeroT) > 250L) {
            if (lastHrZeroT > 0L) {
                val interval = t - lastHrZeroT
                if (interval > 200L) {
                    val bpm = 60000f / interval
                    if (bpm in 40f..180f) zeroCrossBpm = bpm
                }
            }
            lastHrZeroT = t
        }
        prevHrRaw = hrF

        val normHr = normalizeForDisplay(hrDisplay)
        val windowedHeartDetection = windowedHeartPeakDetector.next(normHr, t)
        val heartPeriodicity = heartPeriodicityEstimator.next(hrFilt, t)
        val cleanHeartPeriodicity = cleanHeartPeriodicityEstimator.next(cleanHeart, t)
        val cleanPeakInput = kotlin.math.abs(cleanHeart) / maxOf(cleanHeartEnvelope, 5f)
        val cleanHeartDetection = cleanHeartBeatDetector.next(cleanPeakInput, t)
        if (cleanHeartDetection.detected) {
            cleanPeakTimesList.add(cleanHeartDetection.peakTimeMs ?: t)
            if (cleanPeakTimesList.size > 50) cleanPeakTimesList.removeAt(0)
            _cleanPeakTimes.value = cleanPeakTimesList.toList()
            val detectedBpm = cleanHeartDetection.bpm
            val periodicBpm = cleanHeartPeriodicity.bpm
            val agreesWithPeriodicity = periodicBpm == null ||
                (detectedBpm != null && kotlin.math.abs(detectedBpm - periodicBpm) <= 15f)
            if (detectedBpm != null && agreesWithPeriodicity) {
                lastCleanPeakBpm = detectedBpm
                lastCleanPeakBpmTimeMs = t
            }
        }
        val templateEnhancement = heartbeatTemplateEnhancer.next(
            value = cleanHeart,
            timeMs = t,
            detectedPeakTimeMs = if (cleanHeartDetection.detected) cleanHeartDetection.peakTimeMs else null
        )
        if (templateEnhancement.peakProcessed) {
            _heartTemplateStatus.value = templateEnhancement
        }
        val rpmZc = zeroCrossRpm.next(resp, t)
        val respCycleDetection = respCycleRateDetector.next(resp, t)
        val signalActive = true
        rawBuf.add(t, if (signalActive) xDc else 0f)  // DC-removed ADC waveform
        preBuf.add(t, if (signalActive) xF else 0f)       // pre-filtered signal
        respBuf.add(t, if (signalActive) resp else 0f)    // respiration waveform
        respDisplayBuf.add(t, if (signalActive) resp else 0f)  // 原始呼吸信号，无归一化
        hrBuf.add(t, if (signalActive) hrDisplay else 0f)    // 原始心率信号，无归一化
        cleanHeartBuf.add(t, if (signalActive) cleanHeart else 0f)
        enhancedHeartBuf.add(t, if (signalActive && templateEnhancement.ready) templateEnhancement.value else 0f)

        var thresholdPeakDetected = false
        var dualPeakBpmThisSample: Float? = null
        if (signalState == SignalState.MEASURING && signalActive && !motionFlag) {
            // Dual peak detection on heartSep output (firstPeak + envelope + secondPeak → AO → BPM)
            hrEstimator.add(t, hr)
            val peakBpm = hrPeakDetector.next(hrFilt, t)
            dualPeakBpmThisSample = peakBpm
            if (peakBpm != null) lastPeakBpm = peakBpm

            // Threshold-based peak detection on normalized HR (-1 to 1)
            if (normHr > 0.3f && !peakRising && (t - lastPeakT) > 350L) {
                thresholdPeakDetected = true
                peakRising = true
                lastPeakT = t
                peakTimesList.add(t)
                if (peakTimesList.size > 50) peakTimesList.removeAt(0)
                _peakTimes.value = peakTimesList.toList()
                // 在新峰值到达时计算BPM（避免在两次峰值之间持续更新）
                if (peakTimesList.size >= 2) {
                    val interval = peakTimesList[peakTimesList.size - 1] - peakTimesList[peakTimesList.size - 2]
                    if (interval > 0L) {
                        val bpm = 60000f / interval
                        if (bpm in 40f..180f) lastValidBpm = bpm
                    }
                }
            }
            if (normHr < -0.1f) peakRising = false
        }

        sensorLogger?.log(
            tMs = t,
            rawX = rawX,
            xDc = xDc,
            dc = dcBlocker.dc,
            xF = xF,
            resp = resp,
            hrRaw = hrRaw,
            hrF1 = hrF1,
            hrF = hrF,
            hrFilt = hrFilt,
            hrCalc = hr,
            hrDisplay = hrDisplay,
            heartCandidate = heartCandidate,
            heartEnvelope = heartEnvelope,
            cleanHeart = cleanHeart,
            cleanHeartEnvelope = cleanHeartEnvelope,
            heartTemplateEnhanced = templateEnhancement.value,
            heartTemplateQuality = templateEnhancement.quality,
            heartTemplateCycles = templateEnhancement.cycles,
            heartTemplateReady = templateEnhancement.ready,
            normHr = normHr,
            windowedPeak = windowedHeartDetection.detected,
            windowedBpm = windowedHeartDetection.bpm,
            periodicBpm = heartPeriodicity.bpm,
            periodicQuality = heartPeriodicity.quality,
            cleanPeriodicBpm = cleanHeartPeriodicity.bpm,
            cleanPeriodicQuality = cleanHeartPeriodicity.quality,
            cleanPeak = cleanHeartDetection.detected,
            cleanPeakBpm = cleanHeartDetection.bpm,
            thresholdPeak = thresholdPeakDetected,
            dualPeakBpm = dualPeakBpmThisSample,
            bpm = _rates.value.bpm,
            rpm = _rates.value.rpm,
            rpmZc = rpmZc,
            respCycle = respCycleDetection.crossing,
            cycleRpm = respCycleDetection.rpm,
            presenceState = presenceObservation.state.name,
            presenceScore = presenceObservation.score,
            heartEnvelope10s = presenceObservation.heartEnvelope10s,
            heartStd10s = presenceObservation.heartStd10s
        )



        var nextBpm = lastBpmEstimate
        var nextRpm = lastRpmEstimate

        // Formal BPM: clean-heart grouped beats, cross-checked by waveform periodicity.
        if (t - lastHrEstimateMs >= 1000L) {
            lastHrEstimateMs = t
            val peakFresh = if (lastCleanPeakBpm != null && t - lastCleanPeakBpmTimeMs <= 6_000L) {
                lastCleanPeakBpm
            } else null
            val periodic = cleanHeartPeriodicity.bpm
            val periodicQuality = cleanHeartPeriodicity.quality ?: 0f
            val bpmToUse = when {
                peakFresh != null && periodic != null && kotlin.math.abs(peakFresh - periodic) <= 15f ->
                    peakFresh * 0.75f + periodic * 0.25f
                peakFresh != null && periodic == null -> peakFresh
                peakFresh == null && periodic != null && periodicQuality >= 0.22f -> periodic
                else -> null
            }
            if (bpmToUse != null) {
                nextBpm = hrRateSmoother.update(bpmToUse)
            } else if (t - lastCleanPeakBpmTimeMs > 6_000L) {
                nextBpm = null
                hrRateSmoother.clear()
            }
        }


        // ---- Vital signs: slow-decay BPM/RPM validation ----
        // Legacy threshold/DFT path retained for A/B logging only.
        if (false && t - lastHrEstimateMs >= 1000L) {
            lastHrEstimateMs = t
            val peakBpmVal = lastValidBpm
            val dftBpm = hrEstimator.estimate()
            // 峰值检测为主，DFT为备选
            val bpmToUse = peakBpmVal ?: dftBpm
            // 峰值检测首次生效时重置平滑器，清除DFT遗留的错误历史
            if (peakBpmVal != null && lastValidBpmWasNull) {
                hrRateSmoother.clear()
            }
            lastValidBpmWasNull = peakBpmVal == null
            nextBpm = hrRateSmoother.update(bpmToUse)
            // EMA on final BPM output: suppress single-beat jumps
            if (nextBpm != null) {
                smoothBpm = if (smoothBpm == null) nextBpm
                    else smoothBpm!! * 0.85f + nextBpm!! * 0.15f
                nextBpm = smoothBpm
            }
        }

        if (!motionFlag) {
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
        cleanHeartBuf.clear()
        enhancedHeartBuf.clear()
        zeroCrossBpm = null
        prevHrRaw = 0f
        lastHrZeroT = 0L
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
        hrHPF2.clear()
        hrPostLPF.clear()
        dcBlocker.clear()
        notch50Hz.clear()
        antiAliasLPF.clear()
        candidateHeartHPF.clear()
        candidateHeartLPF.clear()
        candidateHeartSmooth.clear()
        candidateEnvelopeSmooth.clear()
        cleanHeartHPF1.clear()
        cleanHeartHPF2.clear()
        cleanHeartLPF1.clear()
        cleanHeartLPF2.clear()
        cleanHeartSmooth.clear()
        cleanHeartEnvelopeSmooth.clear()
        heartbeatTemplateEnhancer.clear()
        _heartTemplateStatus.value = HeartTemplateEnhancement()
        heartSep.clear()
        hrPeakDetector.clear()
        windowedHeartPeakDetector.clear()
        heartPeriodicityEstimator.clear()
        cleanHeartPeriodicityEstimator.clear()
        cleanHeartBeatDetector.clear()
        cleanPeakTimesList.clear()
        _cleanPeakTimes.value = emptyList()
        lastCleanPeakBpm = null
        lastCleanPeakBpmTimeMs = 0L
        motionFlag = false; motionEndTime = 0L; motionAbsBuf.clear(); motionRawBuf.clear()
        presenceObserver.reset(); _presenceObservation.value = PresenceObserver.Observation()
        lastBypassMotionDetection = false
        _signalQuality.value = SignalState.MEASURING
        breathDetector.clear()
        zeroCrossRpm.clear()
        respCycleRateDetector.clear()
        lastHrEstimateMs = 0L
       lastRespEstimateMs = 0L
        _rates.value = Rates()
        smoothBpm = null
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






























