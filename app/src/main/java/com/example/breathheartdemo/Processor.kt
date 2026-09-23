package com.example.breathheartdemo

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
// Split respiration and heart components, then estimate bpm and rpm.
class Processor(
    private val fsHz: Int,
    private val backgroundVmdEnabled: Boolean = true
) {
    private val cap = fsHz * 330
   val rawBuf = RingBuffer(cap)
   val preBuf = RingBuffer(cap)
  val respBuf = RingBuffer(cap)
   val respDisplayBuf = RingBuffer(cap)
   // Low-delay respiration band used only for rate acquisition. The long FIR
   // remains the display/sleep waveform because its linear phase is useful,
   // but its 6.3 s group delay should not postpone the first RPM value.
   private val respRateBuf = RingBuffer(cap)
   val hrBuf = RingBuffer(cap)
   // Low-latency 0.65-4 Hz evidence waveform retained for fixed-reference BPM
   // estimation. The primary UI card uses heartDetailBuf consistently.
   val cleanHeartBuf = RingBuffer(cap)
   /** Continuous harmonic-preserving reconstruction centered on the accepted heart track. */
   val trackedHeartBuf = RingBuffer(cap)
   /** Paper-aligned K=4 -> K=5 VMD cardiac IMF used by the primary heartbeat card. */
   val vmdHeartBuf = RingBuffer(cap)
   /** VMD-anchored harmonic reconstruction used by the primary enhancement card. */
   val vmdMorphologyBuf = RingBuffer(cap)
   /**
    * Dedicated sleep-model histories. UI waveform histories freeze during an
    * invalid interval and resume as a new path after the timestamp gap. These
    * histories retain a short motion gap so SleepModelInputAdapter can audit
    * and, only within its explicit limits, reconstruct it. Saturation and
    * source changes clear them.
    */
   val sleepHeartBuf = RingBuffer(cap)
   val sleepRespBuf = RingBuffer(cap)
   /** Paper section 3.3.3: x[n-5] - MA10{x[n]} after the 8 Hz acquisition LPF. */
   val paperHeartBuf = RingBuffer(cap)
   // Real continuous mechanical-heart detail (not a repeated template).
   // This causal 1-8 Hz morphology-candidate channel is display-only and never feeds
   // BPM or the sleep model.
   val heartDetailBuf = RingBuffer(cap)
   // Beat-synchronous averaged/template-reconstructed signal for display only.
   val enhancedHeartBuf = RingBuffer(cap)
   // A/B channel: latest eight completed real cycles with constrained morphology.
   val morphologyHeartBuf = RingBuffer(cap)

    // Pre-filter: slow DC removal followed by post-sampling 8 Hz noise limiting.
    private val acquisitionLPF = LowPassFilter(cutoffHz = 8f, sampleRate = fsHz)
    private val dcBlocker = DcBlocker()

    // UI-only causal path.  It is deliberately independent of the formal
    // estimators, so a motion reset can invalidate BPM/VMD state without
    // stopping or zeroing the two live waveform cards.
    private val displayAcquisitionLPF = LowPassFilter(cutoffHz = 8f, sampleRate = fsHz)
    private val displayDcBlocker = DcBlocker()
    private val displayRespFIR = FIRFilter(RESP_FIR_COEFFS)
    private val displayRespHighPass1 = BiquadFilter.highPass(
        cutoffHz = 0.08f, sampleRate = fsHz, q = 0.5411961f
    )
    private val displayRespHighPass2 = BiquadFilter.highPass(
        cutoffHz = 0.08f, sampleRate = fsHz, q = 1.306563f
    )
    private val displayHeartHighPass1 = BiquadFilter.highPass(
        cutoffHz = 0.65f, sampleRate = fsHz, q = 0.5411961f
    )
    private val displayHeartHighPass2 = BiquadFilter.highPass(
        cutoffHz = 0.65f, sampleRate = fsHz, q = 1.306563f
    )
    private val displayHeartLowPass1 = BiquadFilter.lowPass(
        cutoffHz = 4f, sampleRate = fsHz, q = 0.5411961f
    )
    private val displayHeartLowPass2 = BiquadFilter.lowPass(
        cutoffHz = 4f, sampleRate = fsHz, q = 1.306563f
    )
    private val displayHeartSmooth = MovingAverage(windowSize = 3)

    private val respFIR = FIRFilter(RESP_FIR_COEFFS)  // linear-phase low-pass
    // The FIR alone passes DC and contact drift. Raw-only four-position
    // evaluation selected a conservative causal 0.08 Hz fourth-order high-pass:
    // it raises 0.08-0.50 Hz concentration without deleting slow 6 rpm breaths.
    // Keep respBuf on the legacy FIR output for sleep-model compatibility;
    // the cleaned path drives respiratory rate and the live waveform.
    private val respHighPass1 = BiquadFilter.highPass(
        cutoffHz = 0.08f, sampleRate = fsHz, q = 0.5411961f
    )
    private val respHighPass2 = BiquadFilter.highPass(
        cutoffHz = 0.08f, sampleRate = fsHz, q = 1.306563f
    )
    private val respRateHighPass1 = BiquadFilter.highPass(
        cutoffHz = 0.08f, sampleRate = fsHz, q = 0.5411961f
    )
    private val respRateHighPass2 = BiquadFilter.highPass(
        cutoffHz = 0.08f, sampleRate = fsHz, q = 1.306563f
    )
    private val respRateLowPass1 = BiquadFilter.lowPass(
        cutoffHz = 0.50f, sampleRate = fsHz, q = 0.5411961f
    )
    private val respRateLowPass2 = BiquadFilter.lowPass(
        cutoffHz = 0.50f, sampleRate = fsHz, q = 1.306563f
    )
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

    // Official heart waveform path: standard fourth-order Butterworth HP and
    // LP sections. Different Q values are required; duplicating two Q=0.707
    // biquads is not a fourth-order Butterworth response. The 0.65 Hz lower
    // edge keeps the fundamental of the supported 40 bpm lower limit.
    // This remains the grouped-beat and fixed-reference BPM evidence path.
    private val cleanHeartHPF1 = BiquadFilter.highPass(
        cutoffHz = 0.65f, sampleRate = fsHz, q = 0.5411961f
    )
    private val cleanHeartHPF2 = BiquadFilter.highPass(
        cutoffHz = 0.65f, sampleRate = fsHz, q = 1.306563f
    )
    private val cleanHeartLPF1 = BiquadFilter.lowPass(
        cutoffHz = 4f, sampleRate = fsHz, q = 0.5411961f
    )
    private val cleanHeartLPF2 = BiquadFilter.lowPass(
        cutoffHz = 4f, sampleRate = fsHz, q = 1.306563f
    )
    private val cleanHeartSmooth = MovingAverage(windowSize = 3)
    private val cleanHeartEnvelopeSmooth = MovingAverage(windowSize = maxOf(1, fsHz))
    private val adaptiveTrackedHeartFilter = AdaptiveTrackedHeartFilter(fsHz = fsHz)
    // Display-only detail path selected across the four supplied positions.
    // A standard fourth-order 1-8 Hz Butterworth retained essentially the
    // same cross-cycle morphology as 1-10 Hz (0.918 vs 0.920) while excluding
    // more non-cardiac high-frequency vibration. It is intentionally separate
    // from cleanHeart, peak detection, BPM and the sleep model.
    private val templateDetailHPF1 = BiquadFilter.highPass(
        cutoffHz = 1f, sampleRate = fsHz, q = 0.5411961f
    )
    private val templateDetailHPF2 = BiquadFilter.highPass(
        cutoffHz = 1f, sampleRate = fsHz, q = 1.306563f
    )
    private val templateDetailLPF1 = BiquadFilter.lowPass(
        cutoffHz = 8f, sampleRate = fsHz, q = 0.5411961f
    )
    private val templateDetailLPF2 = BiquadFilter.lowPass(
        cutoffHz = 8f, sampleRate = fsHz, q = 1.306563f
    )
    private val heartbeatTemplateEnhancer = HeartbeatTemplateEnhancer(fsHz = fsHz)
    private val heartRateSynchronousPeakTracker = HeartRateSynchronousPeakTracker(fsHz = fsHz)
    private val morphologyRateReferenceTracker = MorphologyRateReferenceTracker()
    private val heartbeatMorphologyBuilder = HeartbeatMorphologySnapshotBuilder(fsHz = fsHz)
    private val heartbeatMorphologyPlayer = HeartbeatMorphologyDelayedPlayer(fsHz = fsHz)
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

    private val _rates = MutableStateFlow(Rates())
    val rates: StateFlow<Rates> = _rates
    private val _heartRespHarmonicRisk = MutableStateFlow(false)
    /** True when the accepted heart candidate is close to 2x..5x respiration. */
    val heartRespHarmonicRisk: StateFlow<Boolean> = _heartRespHarmonicRisk
    private val vmdConfig = VmdHeartConfig(
        windowSeconds = 16,
        stepSeconds = 4,
        firstLayerModeCount = 4,
        secondLayerModeCount = 5,
        cardiacLowHz = 0.9f,
        cardiacHighHz = 2.0f,
        alpha = 1_400f
    )
    private val vmdRawBuf = RingBuffer(fsHz * (vmdConfig.windowSeconds + 1))
    private val vmdRespBuf = RingBuffer(fsHz * (vmdConfig.windowSeconds + 1))
    private val heartModeTracker = HeartModeTracker()
    private val heartRateArbitrator = HeartRateArbitrator()
    private val _vmdHeartStatus = MutableStateFlow(VmdHeartStatus())
    val vmdHeartStatus: StateFlow<VmdHeartStatus> = _vmdHeartStatus
    private val vmdHeartbeatStreamPlayer = VmdHeartbeatStreamPlayer(
        fsHz = fsHz,
        stepSeconds = vmdConfig.stepSeconds
    )
    private val vmdMorphologyStreamPlayer = VmdHeartbeatStreamPlayer(
        fsHz = fsHz,
        stepSeconds = vmdConfig.stepSeconds,
        emitFirstWindowTail = true
    )
    private val _vmdDisplayReady = MutableStateFlow(false)
    /** True after the sample-paced VMD display has accumulated a visible history. */
    val vmdDisplayReady: StateFlow<Boolean> = _vmdDisplayReady
    private val _heartRateDecision = MutableStateFlow(HeartRateDecision())
    /** Carries VMD/fixed-reference provenance and the stale flag for the UI/logger. */
    val heartRateDecision: StateFlow<HeartRateDecision> = _heartRateDecision
    private val vmdHeartWorker = VmdHeartWorker(
        fsHz = fsHz,
        config = vmdConfig,
        onResult = ::handleVmdHeartResult
    )
    private var lastVmdScheduleMs = 0L
    private var vmdCleanSampleCount = 0
    @Volatile private var lastVmdCompletedWindowMs = 0L
    @Volatile private var latestFixedBpm: Float? = null
    @Volatile private var latestFixedPeriodicity = 0f
    @Volatile private var latestFixedPeakRatio = 0f

    // ======= Signal Quality Detection =======
    enum class SignalState { MEASURING, RECOVERING, MOTION, NEAR_SATURATION, SATURATED }
    private var signalState = SignalState.MEASURING
    private val recoveryWarmupMs = 16_000L
    private var recoveryWarmupUntilMs = 0L
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
    private val _heartMorphologySnapshot = MutableStateFlow(MorphologyWaveformSnapshot())
    val heartMorphologySnapshot: StateFlow<MorphologyWaveformSnapshot> = _heartMorphologySnapshot
    private val _morphologyPlaybackBoundaryTimes = MutableStateFlow<List<Long>>(emptyList())
    val morphologyPlaybackBoundaryTimes: StateFlow<List<Long>> = _morphologyPlaybackBoundaryTimes
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
    private var consecutiveClippedSamples = 0
    private var saturationActive = false
    private var saturationHoldUntilMs = 0L
    private val nearSaturationDetector = NearSaturationDetector(fsHz = fsHz)
    private var nearSaturationActive = false
    private var nearSaturationHoldUntilMs = 0L
    private val slowContactArtifactDetector = SlowContactArtifactDetector(fsHz = fsHz)
    private val _slowContactArtifact =
        MutableStateFlow(SlowContactArtifactDetector.Observation())
    /** Diagnostic evidence for slow contact-pressure changes. */
    val slowContactArtifact: StateFlow<SlowContactArtifactDetector.Observation> =
        _slowContactArtifact
    private var lastBypassMotionDetection = false
    private val zeroCrossRpm = ZeroCrossingRpm()
    private val respirationStartupRateDetector = RespirationStartupRateDetector()
    private val respCycleRateDetector = RespCycleRateDetector()
    // Legacy A/B diagnostics below are still logged, but they do not drive the UI or formal BPM.
    private val heartSep = DifferentialHeartSeparator()
    private val hrPeakDetector = DualPeakDetector()
    private val windowedHeartPeakDetector = WindowedHeartPeakDetector()
    private val heartPeriodicityEstimator = HeartPeriodicityEstimator(fsHz = fsHz)
    private val cleanHeartPeriodicityEstimator = HeartPeriodicityEstimator(
        fsHz = fsHz,
        windowSeconds = 10,
        minBpm = 40f,
        maxBpm = 120f,
        minimumQuality = 0.25f
    )
    private val cleanHeartSpectrumEstimator = HeartSpectrumEstimator(
        fsHz = fsHz,
        windowSeconds = 10,
        minBpm = 40f,
        maxBpm = 120f,
        minimumPeakRatio = 3f
    )
    private val heartRateAmbiguityResolver = HeartRateAmbiguityResolver(
        minBpm = 40f,
        maxBpm = 120f,
        lowRateReviewBpm = 55f,
        confirmationFrames = 3
    )
    private val cleanHeartBeatDetector = WindowedHeartPeakDetector(
        threshold = 0.8f,
        groupWindowMs = 250L,
        refractoryMs = 500L,
        minIntervalMs = 500L,
        maxIntervalMs = 1_500L
    )
    private var latestIntervalBpm: Float? = null
    private var latestIntervalQuality = 0f
    private var latestIntervalTimeMs = 0L
    private var lastAcceptedFormalBpmTimeMs = 0L
    private var lastAcceptedRespRateTimeMs = 0L
    private var prevRawX = 2600f
    private var lastPeakBpm: Float? = null

    // DC removal: slow exponential-moving-average
    @Volatile
   var sensorLogger: SensorDataLogger? = null
    @Volatile
    private var latestSleepPrediction: SleepStateResult? = null

    fun updateSleepPrediction(result: SleepStateResult?) {
        latestSleepPrediction = result
    }
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
    private var lastPreviewEmitMs = Long.MIN_VALUE

    private var lastHrEstimateMs = 0L
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
            // A mock/real source change is also a contact/template change.
            // Do not allow a VMD path or heartbeat template to cross it.
            resetVitalPathAfterInvalidSignal(clearSleepModelHistory = true)
            heartModeTracker.clear()
            heartRateArbitrator.clear()
            _vmdHeartStatus.value = VmdHeartStatus()
            _heartRateDecision.value = HeartRateDecision()
            prevRawX = rawX
            lastBypassMotionDetection = bypassMotionDetection
        }

        // A flat top/bottom is physical ADC saturation, not filterable noise.
        // Require three consecutive rail samples so one isolated ADC outlier
        // does not discard a complete analysis window. Once confirmed, stop
        // all vital processing and wait three clean seconds before rebuilding
        // the causal filters and the 10-second estimators from scratch.
        val clipped = rawX >= 4085f || rawX <= 10f
        if (clipped) {
            consecutiveClippedSamples++
            if (consecutiveClippedSamples >= 3) {
                saturationHoldUntilMs = t + 3_000L
                if (!saturationActive) {
                    saturationActive = true
                    nearSaturationDetector.clear()
                    nearSaturationActive = false
                    nearSaturationHoldUntilMs = 0L
                    resetVitalPathAfterInvalidSignal(clearSleepModelHistory = true)
                }
                setSignalState(SignalState.SATURATED)
            }
            return
        }
        consecutiveClippedSamples = 0
        if (saturationActive) {
            if (t < saturationHoldUntilMs) return
            saturationActive = false
            saturationHoldUntilMs = 0L
            resetVitalPathAfterInvalidSignal(clearSleepModelHistory = true)
            beginRecoveryWarmup(t)
        }

        // Low headroom and flat-topped extrema can distort heartbeat
        // morphology before the ADC reaches the hard 0/4095 rails. Repeated
        // evidence is required: four samples beyond 4000/95, three samples in
        // one second beyond 4063/32, or a five-sample flat local extremum.
        // This is a warning, not confirmed physical clipping: freeze formal
        // output briefly and invalidate only the VMD block that touched it.
        // Keeping causal/display/sleep histories prevents a short pressure
        // maximum from resetting collection progress to zero.
        val nearSaturation = nearSaturationDetector.next(rawX)
        if (nearSaturation) {
            nearSaturationHoldUntilMs = t + 3_000L
            if (!nearSaturationActive) {
                nearSaturationActive = true
                invalidateVmdWindowAfterWarning()
                _rates.value = Rates()
            }
            setSignalState(SignalState.NEAR_SATURATION)
        }
        if (nearSaturationActive) {
            if (t >= nearSaturationHoldUntilMs) {
                nearSaturationActive = false
                nearSaturationHoldUntilMs = 0L
                nearSaturationDetector.clear()
                setSignalState(SignalState.MEASURING)
            }
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
                // A single 240-ADC step occurs in otherwise static recordings
                // (for example BLE/contact impulses) and used to blank both
                // heartbeat cards for 19 s. Require several rapid changes;
                // sustained displacement remains covered by the range/energy
                // detector below.
                val suddenMotion = peakDiff > 300f && rapidChangeCount >= 3
                val irregularLargeMotion = rawRange > 1200f && rapidChangeCount >= 3
                if (suddenMotion || irregularLargeMotion) {
                    motionFlag = true
                    motionEndTime = t + 3000L
                    resetVitalPathAfterInvalidSignal(
                        clearSleepModelHistory = false,
                        preserveMorphologyDisplay = true
                    )
                    setSignalState(SignalState.MOTION)
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
                    resetVitalPathAfterInvalidSignal(
                        clearSleepModelHistory = false,
                        preserveMorphologyDisplay = true
                    )
                    beginRecoveryWarmup(t)
                }
            }
            // Formal rates remain invalid during motion, but processing keeps
            // running so the independent display path and diagnostic CSV do
            // not freeze for the complete three-second hold.
        }
        finishRecoveryWarmupIfReady(t)

        // Always advance the display-only filters for non-saturated samples.
        // These states are not cleared by an ordinary motion event, preventing
        // the visible waveform from restarting at zero after every trigger.
        val displayXDc = displayDcBlocker.next(x)
        val displayXF = displayAcquisitionLPF.next(displayXDc)
        val displayRespLowPass = displayRespFIR.next(displayXF)
        val displayResp = displayRespHighPass2.next(
            displayRespHighPass1.next(displayRespLowPass)
        )
        val displayHeart = displayHeartSmooth.next(
            displayHeartLowPass2.next(
                displayHeartLowPass1.next(
                    displayHeartHighPass2.next(displayHeartHighPass1.next(displayXF))
                )
            )
        )

        val xDc = dcBlocker.next(x)             // slow baseline removal
        // At fs=50 Hz, a digital 50 Hz notch is outside the 25 Hz Nyquist
        // band.  Limit in-band noise here; true anti-aliasing belongs in hardware.
        val xClean = acquisitionLPF.next(xDc)
        val xF = xClean

        // Track raw and centered values for the live preview
        recentRaw[rawIdx] = rawX.toInt()
        rawIdx = (rawIdx + 1) % 16
        if (rawCount < 16) rawCount++

        recentCentered[centeredIdx] = xF.toInt()
        centeredIdx = (centeredIdx + 1) % 16
        if (centeredCount < 16) centeredCount++

        if (lastPreviewEmitMs == Long.MIN_VALUE || t - lastPreviewEmitMs >= 250L) {
            _rawPreview.value = (0 until rawCount).joinToString(" ") { i ->
                val v = recentRaw[(rawIdx - rawCount + i + 16) % 16]
                if (v >= 0) " $v" else "$v"
            }
            _centeredPreview.value = (0 until centeredCount).joinToString(" ") { i ->
                val v = recentCentered[(centeredIdx - centeredCount + i + 16) % 16]
                if (v >= 0) " $v" else "$v"
            }
            lastPreviewEmitMs = t
        }

        // --- Respiration: extract via long MA (retain 0.1~0.5 Hz) ---
        val respLowPass = respFIR.next(xF)
        val resp = respHighPass2.next(respHighPass1.next(respLowPass))
        val respRateSignal = respRateLowPass2.next(
            respRateLowPass1.next(
                respRateHighPass2.next(respRateHighPass1.next(xF))
            )
        )

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
        val slowContactObservation = slowContactArtifactDetector.next(rawX, cleanHeart)
        _slowContactArtifact.value = slowContactObservation
        if (!bypassMotionDetection &&
            signalState == SignalState.MEASURING &&
            slowContactObservation.triggered
        ) {
            // This event is smoother than the point-to-point motion rule, but
            // its joint raw-pressure and heart-band energy rise makes the
            // filtered waveform invalid. Reuse the established motion hold and
            // full 16 s clean-window recovery instead of drawing the transient.
            motionFlag = true
            motionEndTime = t + 3_000L
            resetVitalPathAfterInvalidSignal(
                clearSleepModelHistory = false,
                preserveMorphologyDisplay = false
            )
            _slowContactArtifact.value = slowContactObservation
            setSignalState(SignalState.MOTION)
        }
        val acceptedDisplayBpm = _vmdHeartStatus.value.tracking.acceptedBpm
            ?: _heartRateDecision.value.bpm?.takeIf { !_heartRateDecision.value.stale }
        val trackedHeart = adaptiveTrackedHeartFilter.next(xF, acceptedDisplayBpm)
        val heartTemplateInput = templateDetailLPF2.next(
            templateDetailLPF1.next(templateDetailHPF2.next(templateDetailHPF1.next(xF)))
        )
        // The formal BPM rules remain strict.  Morphology segmentation alone
        // may retain the last trustworthy period for 30 s while VMD reacquires;
        // this prevents the enhancement card from repeatedly freezing during
        // the 5-26 s clean-signal gaps observed in sensor_20260909_152249.csv.
        val morphologyVmdFresh = lastVmdCompletedWindowMs > 0L &&
            t - lastVmdCompletedWindowMs <= vmdConfig.stepSeconds * 2_000L
        val morphologyRateReference = if (signalState == SignalState.MOTION || motionFlag) {
            // Do not consume or clear the last display-only reference while
            // motion is active.  It may be reused during RECOVERING, but never
            // becomes a formal BPM result.
            MorphologyRateReference()
        } else {
            morphologyRateReferenceTracker.update(
                timeMs = t,
                signalValid = true,
                decision = _heartRateDecision.value,
                tracking = _vmdHeartStatus.value.tracking,
                vmdFresh = morphologyVmdFresh,
                fixedBpm = latestFixedBpm,
                fixedPeriodicity = latestFixedPeriodicity,
                fixedPeakRatio = latestFixedPeakRatio
            )
        }
        val synchronizedMorphologyPeakTimeMs = heartRateSynchronousPeakTracker.next(
            value = heartTemplateInput,
            timeMs = t,
            acceptedBpm = morphologyRateReference.bpm
        )
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
        val cleanHeartSpectrum = cleanHeartSpectrumEstimator.next(cleanHeart, t)
        val cleanPeakInput = kotlin.math.abs(cleanHeart) / maxOf(cleanHeartEnvelope, 5f)
        val cleanHeartDetection = cleanHeartBeatDetector.next(cleanPeakInput, t)
        if (cleanHeartDetection.detected) {
            cleanPeakTimesList.add(cleanHeartDetection.peakTimeMs ?: t)
            if (cleanPeakTimesList.size > 50) cleanPeakTimesList.removeAt(0)
            _cleanPeakTimes.value = cleanPeakTimesList.toList()
            if (cleanHeartDetection.bpm != null) {
                latestIntervalBpm = cleanHeartDetection.bpm
                latestIntervalQuality = cleanHeartDetection.quality
                latestIntervalTimeMs = cleanHeartDetection.peakTimeMs ?: t
            }
        }
        // Build the display-only template from the higher-detail 1-8 Hz path.
        // Beat boundaries are constrained by the accepted VMD/final heart-rate
        // track instead of the broad 0.65-4 Hz peak detector, which may lock to
        // respiration or a secondary mechanical peak. This waveform never
        // feeds BPM or the sleep model.
        val templatePeakTimeMs = synchronizedMorphologyPeakTimeMs
        val templateEnhancement = heartbeatTemplateEnhancer.next(
            value = heartTemplateInput,
            timeMs = t,
            detectedPeakTimeMs = templatePeakTimeMs
        )
        if (templateEnhancement.peakProcessed) {
            _heartTemplateStatus.value = templateEnhancement
        }
        val morphologySnapshot = heartbeatMorphologyBuilder.next(
            value = heartTemplateInput,
            timeMs = t,
            detectedPeakTimeMs = templatePeakTimeMs
        )
        if (morphologySnapshot != null) {
            _heartMorphologySnapshot.value = morphologySnapshot
            if (morphologySnapshot.ready) {
                heartbeatMorphologyPlayer.update(morphologySnapshot)
                _morphologyPlaybackBoundaryTimes.value =
                    heartbeatMorphologyPlayer.delayedBoundaryTimesMs.filter { it <= t }
            } else if (morphologySnapshot.cycles == 0) {
                // Keep the last valid player while a replacement template is
                // rebuilt. Explicit source/contact/saturation resets clear it.
                _morphologyPlaybackBoundaryTimes.value = emptyList()
            }
        }
        val morphologyPlaybackValue = heartbeatMorphologyPlayer.next(
            timeMs = t,
            allowHeldRepeat = signalState == SignalState.MOTION ||
                signalState == SignalState.RECOVERING
        )
        var morphologyBoundaryThisSample = false
        morphologyPlaybackValue?.let { value ->
            val latestDisplayTimeMs = morphologyHeartBuf
                .valueRange(windowMs = 12_000L)
                ?.latestTimeMs
            val needsDisplaySeed = latestDisplayTimeMs == null ||
                t - latestDisplayTimeMs > 250L
            val displaySeeded = needsDisplaySeed &&
                heartbeatMorphologyPlayer.seedDisplayHistory(
                    buffer = morphologyHeartBuf,
                    timeMs = t,
                    currentValue = value
                )
            if (!displaySeeded) morphologyHeartBuf.add(t, value)
            val visibleBoundaries =
                heartbeatMorphologyPlayer.delayedBoundaryTimesMs.filter { it <= t }
            if (visibleBoundaries != _morphologyPlaybackBoundaryTimes.value) {
                _morphologyPlaybackBoundaryTimes.value = visibleBoundaries
            }
            morphologyBoundaryThisSample =
                heartbeatMorphologyPlayer.delayedBoundaryTimesMs.any { it == t }
        }
        val rpmZc = zeroCrossRpm.next(respRateSignal, t)
        val startupRpm = respirationStartupRateDetector.next(respRateSignal, t)
        val respCycleDetection = respCycleRateDetector.next(respRateSignal, t)
        val signalActive = signalState == SignalState.MEASURING
        // Causal diagnostic paths keep running independently. The displayed VMD
        // morphology is handled below as a delayed stream and freezes whenever
        // the current block is invalid.
        rawBuf.add(t, displayXDc)
        preBuf.add(t, displayXF)
        respDisplayBuf.add(t, displayResp)
        cleanHeartBuf.add(t, displayHeart)
        // The detail chart is continuous during valid acquisition and freezes
        // on its last complete history during motion/recovery. This avoids both
        // drawing the motion impulse and clearing/redrawing the whole card.
        if (signalActive) heartDetailBuf.add(t, heartTemplateInput)
        var vmdHeartbeatDisplayValue: Float? = null
        var vmdMorphologyDisplayValue: Float? = null
        vmdHeartbeatStreamPlayer.next()?.let { value ->
            vmdHeartbeatDisplayValue = value
            vmdHeartBuf.add(t, value)
        }
        vmdMorphologyStreamPlayer.next()?.let { value ->
            vmdMorphologyDisplayValue = value
            vmdMorphologyBuf.add(t, value)
            if (!_vmdDisplayReady.value) {
                val visible = vmdMorphologyBuf.valueRange(windowMs = 2_000L)
                if (visible != null && visible.sampleCount >= fsHz * 2) {
                    _vmdDisplayReady.value = true
                }
            }
        }
        if (signalActive) {
            // Preserve the sleep model's historical input distribution while the
            // rate/display path rejects very slow contact drift.
            respBuf.add(t, respLowPass)
            respRateBuf.add(t, respRateSignal)
            hrBuf.add(t, hrDisplay)    // 原始心率信号，无归一化
            trackedHeartBuf.add(t, trackedHeart.value)
            sleepHeartBuf.add(t, cleanHeart)
            sleepRespBuf.add(t, respLowPass)
            paperHeartBuf.add(t, hrRaw)
            if (templateEnhancement.ready) {
                enhancedHeartBuf.add(t, templateEnhancement.value)
            }
        }
        // Dedicated clean-window inputs for background VMD. Raw ADC is used;
        // detrending is performed inside the block analyzer.
        if ((signalState == SignalState.MEASURING ||
                signalState == SignalState.RECOVERING) && !motionFlag) {
            vmdRawBuf.add(t, rawX)
            vmdRespBuf.add(t, resp)
            vmdCleanSampleCount++
        }

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
            trackedHeart = trackedHeart.value,
            trackedHeartCenterHz = trackedHeart.centerHz,
            vmdHeartbeat = vmdHeartbeatDisplayValue,
            vmdMorphology = vmdMorphologyDisplayValue,
            heartTemplateInput = heartTemplateInput,
            heartTemplateEnhanced = templateEnhancement.value,
            heartTemplateQuality = templateEnhancement.quality,
            heartTemplateCycles = templateEnhancement.cycles,
            heartTemplateReady = templateEnhancement.ready,
            heartMorphologyEnhanced = morphologyPlaybackValue,
            heartMorphologyReady = morphologyPlaybackValue != null,
            heartMorphologyCycles = _heartMorphologySnapshot.value.cycles,
            heartMorphologyQuality = _heartMorphologySnapshot.value.quality,
            heartMorphologyDelayMs = heartbeatMorphologyPlayer.delayMs,
            heartMorphologyBoundary = morphologyBoundaryThisSample,
            heartMorphologyReferenceBpm = morphologyRateReference.bpm,
            heartMorphologyReferenceSource = morphologyRateReference.source.name,
            heartMorphologyReferenceAgeMs = morphologyRateReference.ageMs,
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
            heartRateSource = _heartRateDecision.value.source.name,
            heartRateStale = _heartRateDecision.value.stale,
            heartRatePrimaryCandidate = _heartRateDecision.value.primaryCandidateBpm,
            heartRateAlternateCandidate = _heartRateDecision.value.alternateCandidateBpm,
            heartRateAmbiguityReason = _heartRateDecision.value.ambiguityReason.name,
            heartRateConfidence = _heartRateDecision.value.confidence,
            vmdTrackingState = _vmdHeartStatus.value.tracking.state.name,
            vmdComputeTimeMs = _vmdHeartStatus.value.computeTimeMs,
            signalQuality = signalState.name,
            slowContactRawRange = slowContactObservation.rawRange,
            slowContactRmsRatio = slowContactObservation.rmsRatio,
            rpm = _rates.value.rpm,
            rpmZc = rpmZc,
            respCycle = respCycleDetection.crossing,
            cycleRpm = respCycleDetection.rpm,
            presenceState = presenceObservation.state.name,
            presenceScore = presenceObservation.score,
            heartEnvelope10s = presenceObservation.heartEnvelope10s,
            heartStd10s = presenceObservation.heartStd10s,
            sleepPredictedState = latestSleepPrediction?.state,
            sleepProbability = latestSleepPrediction?.takeIf { it.code == 0 }?.confidence,
            sleepResultCode = latestSleepPrediction?.code,
            sleepModelVersion = latestSleepPrediction?.modelVersion,
            sleepPredictionTimeMs = latestSleepPrediction?.timestampMs,
            sleepInputOutlierRatio = latestSleepPrediction?.inputOutlierRatio,
            sleepInputImputedFraction = latestSleepPrediction?.inputImputedFraction,
            sleepInputMaxGapMs = latestSleepPrediction?.inputMaxGapMs
        )



        var nextBpm = lastBpmEstimate
        var nextRpm = lastRpmEstimate

        // Formal BPM: retain multiple autocorrelation/spectrum peaks, then use
        // grouped beat intervals as an independent tie-breaker. No single
        // peak detector is allowed to publish BPM on its own.
        if (t - lastHrEstimateMs >= 1000L) {
            lastHrEstimateMs = t
            val periodicQuality = cleanHeartPeriodicity.quality ?: 0f
            val spectrumPeakRatio = cleanHeartSpectrum.peakRatio ?: 0f

            val intervalFresh = latestIntervalTimeMs > 0L &&
                t - latestIntervalTimeMs <= 3_000L
            val rateResolution = heartRateAmbiguityResolver.update(
                periodicity = cleanHeartPeriodicity,
                spectrum = cleanHeartSpectrum,
                intervalBpm = latestIntervalBpm.takeIf { intervalFresh },
                intervalQuality = latestIntervalQuality.takeIf { intervalFresh } ?: 0f,
                respirationRpm = lastRpmEstimate,
                invalidSignal = signalState != SignalState.MEASURING || motionFlag,
                templateCorrelation = templateEnhancement.quality
                    ?.takeIf { templateEnhancement.ready } ?: 0f
            )
            val bpmToUse = rateResolution.bpm
            val resolvedPeriodicQuality = bpmToUse?.let { resolvedBpm ->
                cleanHeartPeriodicity.candidates
                    .filter { kotlin.math.abs(it.bpm - resolvedBpm) <= 6f }
                    .maxOfOrNull { it.strength }
            } ?: periodicQuality
            val resolvedSpectrumPeakRatio = bpmToUse?.let { resolvedBpm ->
                cleanHeartSpectrum.candidates
                    .filter { kotlin.math.abs(it.bpm - resolvedBpm) <= 6f }
                    .maxOfOrNull { it.strength }
            } ?: spectrumPeakRatio
            latestFixedBpm = bpmToUse
            latestFixedPeriodicity = resolvedPeriodicQuality
            latestFixedPeakRatio = resolvedSpectrumPeakRatio

            // This is an ambiguity warning, not a notch command: removing the
            // bin would also remove a real heart rate at the same frequency.
            _heartRespHarmonicRisk.value = if (bpmToUse != null && lastRpmEstimate != null) {
                (2..5).any { harmonic ->
                    kotlin.math.abs(bpmToUse - lastRpmEstimate!! * harmonic) <= 3f
                }
            } else false
            val vmdFresh = lastVmdCompletedWindowMs > 0L &&
                t - lastVmdCompletedWindowMs <= vmdConfig.stepSeconds * 2_000L
            var trackingForArbitration = _vmdHeartStatus.value.tracking
            if (backgroundVmdEnabled && !vmdFresh &&
                (trackingForArbitration.state == HeartTrackingState.LOCKED ||
                    trackingForArbitration.state == HeartTrackingState.FUSED)
            ) {
                trackingForArbitration = heartModeTracker.update(
                    candidates = emptyList(),
                    fixedBpm = null,
                    fixedPeriodicity = 0f,
                    fixedSpectrumPeakRatio = 0f,
                    respirationRpm = lastRpmEstimate,
                    invalidSignal = true
                )
                _vmdHeartStatus.value = _vmdHeartStatus.value.copy(
                    tracking = trackingForArbitration
                )
            }
            val arbitration = if (backgroundVmdEnabled) {
                heartRateArbitrator.update(
                    tracking = trackingForArbitration,
                    vmdFresh = vmdFresh,
                    fixedBpm = bpmToUse,
                    fixedPeriodicity = resolvedPeriodicQuality,
                    fixedPeakRatio = resolvedSpectrumPeakRatio,
                    invalidSignal = signalState != SignalState.MEASURING || motionFlag
                )
            } else {
                HeartRateDecision(
                    bpm = rateResolution.displayBpm,
                    source = if (bpmToUse == null) {
                        HeartRateSource.NONE
                    } else {
                        HeartRateSource.FIXED_REFERENCE
                    },
                    stale = rateResolution.stale
                )
            }
            _heartRateDecision.value = arbitration.copy(
                ambiguityReason = rateResolution.reason,
                primaryCandidateBpm = rateResolution.primaryCandidateBpm,
                alternateCandidateBpm = rateResolution.alternateCandidateBpm,
                confidence = rateResolution.confidence
            )
            if (arbitration.bpm != null && !arbitration.stale) {
                nextBpm = hrRateSmoother.update(arbitration.bpm)
                lastAcceptedFormalBpmTimeMs = t
            } else if (arbitration.bpm != null) {
                // Preserve the last visible value, but provenance marks it stale.
                nextBpm = arbitration.bpm
            } else if (lastAcceptedFormalBpmTimeMs == 0L || t - lastAcceptedFormalBpmTimeMs > 6_000L) {
                nextBpm = null
                hrRateSmoother.clear()
            }
        }

        if (signalState != SignalState.MOTION && !motionFlag) {
            maybeScheduleVmdHeartAnalysis(t)
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

        if (!motionFlag && signalState == SignalState.MEASURING) {
            val newRpm = respCycleDetection.rpm ?: startupRpm
            if (newRpm != null &&
                respRateBufHasEnoughData(windowSec = 6) &&
                measureRecentRespRateAmplitude(windowSec = 8) >= 10f
            ) {
                nextRpm = respRateSmoother.update(newRpm)
                lastAcceptedRespRateTimeMs = t
            } else if (lastAcceptedRespRateTimeMs == 0L || t - lastAcceptedRespRateTimeMs > 15_000L) {
                nextRpm = null
                respRateSmoother.clear()
            }
        }

        if (signalState == SignalState.NEAR_SATURATION) {
            // The waveform histories keep advancing, but a near-rail sample is
            // not valid evidence for a newly published vital-sign value.
            nextBpm = null
            nextRpm = null
            validHrCount = 0
            validRespCount = 0
        }

        val bpmValid = nextBpm != null && nextBpm >= 40f && nextBpm <= 110f &&
            !_heartRateDecision.value.stale
        val rpmValid = nextRpm != null && nextRpm >= 8f && nextRpm <= 30f
        validHrCount = if (bpmValid) minOf(validHrCount + 1, 10) else maxOf(validHrCount - 1, 0)
        validRespCount = if (rpmValid) minOf(validRespCount + 1, 10) else maxOf(validRespCount - 1, 0)
        val hrReliable = validHrCount >= 3
        val respReliable = validRespCount >= 3
        val publishedBpm = when {
            nextBpm == null -> null
            hrReliable -> nextBpm
            else -> _rates.value.bpm
        }
        val publishedRpm = when {
            nextRpm == null -> null
            respReliable -> nextRpm
            else -> _rates.value.rpm
        }
        if (publishedBpm != _rates.value.bpm || publishedRpm != _rates.value.rpm) {
            _rates.value = Rates(bpm = publishedBpm, rpm = publishedRpm)
        }


        lastBpmEstimate = nextBpm
        lastRpmEstimate = nextRpm
    }

    private fun maybeScheduleVmdHeartAnalysis(windowEndMs: Long) {
        if (!backgroundVmdEnabled) return
        val requiredSamples = fsHz * vmdConfig.windowSeconds
        if (vmdCleanSampleCount < requiredSamples) return
        if (lastVmdScheduleMs > 0L &&
            windowEndMs - lastVmdScheduleMs < vmdConfig.stepSeconds * 1_000L
        ) return
        lastVmdScheduleMs = windowEndMs
        val (_, raw) = vmdRawBuf.snapshotLatest(requiredSamples)
        val (_, respiration) = vmdRespBuf.snapshotLatest(requiredSamples)
        if (raw.size != requiredSamples || respiration.size != requiredSamples) return
        vmdHeartWorker.offer(
            VmdHeartWorkRequest(
                windowEndMs = windowEndMs,
                raw = raw,
                respiration = respiration,
                heartbeatTemplate = _heartTemplateStatus.value.template.toFloatArray(),
                fixedBpm = latestFixedBpm,
                fixedPeriodicity = latestFixedPeriodicity,
                fixedPeakRatio = latestFixedPeakRatio,
                respirationRpm = lastRpmEstimate
            )
        )
        // If the previous frame is still running, this frame is deliberately
        // skipped: single-flight backpressure avoids an ever-growing queue.
    }

    private fun handleVmdHeartResult(work: VmdHeartWorkResult) {
        val analysis = work.analysis
        if (analysis == null) {
            _vmdHeartStatus.value = _vmdHeartStatus.value.copy(
                completedWindowEndMs = work.windowEndMs,
                error = work.error ?: "VMD analysis failed"
            )
            return
        }
        val tracking = heartModeTracker.update(
            candidates = analysis.candidates,
            fixedBpm = work.fixedBpm,
            fixedPeriodicity = work.fixedPeriodicity,
            fixedSpectrumPeakRatio = work.fixedPeakRatio,
            respirationRpm = work.respirationRpm,
            invalidSignal = false
        )
        // Stream the paper-aligned final IMF regardless of the formal tracker
        // state. Quality gates still control BPM, while the display keeps the
        // measured amplitude/phase evolution and never swaps whole windows.
        vmdHeartbeatStreamPlayer.update(analysis.heartbeatWaveform)
        // Add only real f0-synchronous detail recovered from the measured
        // prefiltered signal. The stream player appends the new tail at sample
        // cadence, so this block result never replaces the visible chart.
        vmdMorphologyStreamPlayer.update(analysis.morphologyWaveform)
        lastVmdCompletedWindowMs = work.windowEndMs
        _vmdHeartStatus.value = VmdHeartStatus(
            tracking = tracking,
            modeCentersHz = analysis.modeCentersHz.toList(),
            candidateCount = analysis.candidates.size,
            morphologyHarmonicOrders = analysis.morphologyHarmonicOrders.toList(),
            completedWindowEndMs = work.windowEndMs,
            computeTimeMs = analysis.computeTimeMs,
            error = null
        )
    }

    /** Release the background VMD thread when the owning UI/service is disposed. */
    fun close() {
        vmdHeartWorker.close()
    }

    /**
     * Drop only block-analysis state that contains a near-rail warning.
     * Previously this path shared the hard-saturation reset and erased sleep
     * progress plus the already displayed VMD waveform for at least 19 s.
     */
    private fun invalidateVmdWindowAfterWarning() {
        vmdRawBuf.clear()
        vmdRespBuf.clear()
        vmdCleanSampleCount = 0
        lastVmdScheduleMs = 0L
        lastVmdCompletedWindowMs = 0L
        vmdHeartbeatStreamPlayer.clear()
        vmdMorphologyStreamPlayer.clear()
        vmdHeartWorker.invalidate()
        heartModeTracker.clear()
        heartRateArbitrator.clear()
        _vmdHeartStatus.value = VmdHeartStatus()
        _heartRateDecision.value = HeartRateDecision()
    }

    /** Rebuild all causal vital estimators after motion or ADC saturation. */
    private fun resetVitalPathAfterInvalidSignal(
        clearSleepModelHistory: Boolean,
        preserveMorphologyDisplay: Boolean = false
    ) {
        // cleanHeartBuf and morphologyHeartBuf are presentation histories.
        // Retain their last valid samples while motion/saturation is rejected,
        // so the cards freeze instead of disappearing. When clean samples
        // resume, Waveform segments the path at the timestamp gap and cannot
        // draw a false connecting line. A manual/source reset clears both.
        trackedHeartBuf.clear()
        vmdHeartBuf.clear()
        vmdHeartbeatStreamPlayer.clear()
        vmdMorphologyStreamPlayer.clear()
        // Motion invalidates the VMD analysis window but not the already drawn
        // history. Keep it frozen through the 16 s clean-window recovery and
        // resume by appending a new timestamp-separated segment.
        if (clearSleepModelHistory) {
            vmdMorphologyBuf.clear()
            _vmdDisplayReady.value = false
        }
        paperHeartBuf.clear()
        // Ordinary motion/contact recovery must not erase the visible history.
        // New detail samples are withheld below until MEASURING resumes, so the
        // chart freezes cleanly and then starts a segmented continuation.
        if (clearSleepModelHistory) heartDetailBuf.clear()
        enhancedHeartBuf.clear()
        slowContactArtifactDetector.clear()
        dcBlocker.clear()
        acquisitionLPF.clear()
        respFIR.clear()
        respHighPass1.clear()
        respHighPass2.clear()
        respRateHighPass1.clear()
        respRateHighPass2.clear()
        respRateLowPass1.clear()
        respRateLowPass2.clear()
        respRateBuf.clear()
        heartSep.clear()
        hrHPF.clear()
        hrHPF2.clear()
        hrPostLPF.clear()
        hrCalc.clear()
        hrDisplaySmooth.clear()
        hrPeakDetector.clear()
        windowedHeartPeakDetector.clear()
        heartPeriodicityEstimator.clear()
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
        adaptiveTrackedHeartFilter.clear()
        if (clearSleepModelHistory) {
            templateDetailHPF1.clear()
            templateDetailHPF2.clear()
            templateDetailLPF1.clear()
            templateDetailLPF2.clear()
        }
        heartbeatTemplateEnhancer.clear()
        heartRateSynchronousPeakTracker.clear()
        _heartTemplateStatus.value = HeartTemplateEnhancement()
        heartbeatMorphologyBuilder.clear()
        if (!preserveMorphologyDisplay) {
            morphologyRateReferenceTracker.clear()
            heartbeatMorphologyPlayer.clear()
            _heartMorphologySnapshot.value = MorphologyWaveformSnapshot()
            _morphologyPlaybackBoundaryTimes.value = emptyList()
        }
        cleanHeartPeriodicityEstimator.clear()
        cleanHeartSpectrumEstimator.clear()
        heartRateAmbiguityResolver.clear()
        cleanHeartBeatDetector.clear()
        latestIntervalBpm = null
        latestIntervalQuality = 0f
        latestIntervalTimeMs = 0L
        respCycleRateDetector.clear()
        respirationStartupRateDetector.clear()
        hrRateSmoother.clear()
        respRateSmoother.clear()
        cleanPeakTimesList.clear()
        _cleanPeakTimes.value = emptyList()
        lastAcceptedFormalBpmTimeMs = 0L
        lastAcceptedRespRateTimeMs = 0L
        lastBpmEstimate = null
        lastRpmEstimate = null
        validHrCount = 0
        validRespCount = 0
        _rates.value = Rates()
        _heartRespHarmonicRisk.value = false
        // A block method must never bridge a motion/saturation interval.  Drop
        // every pre-event sample and invalidate any in-flight worker result.
        vmdRawBuf.clear()
        vmdRespBuf.clear()
        vmdCleanSampleCount = 0
        lastVmdScheduleMs = 0L
        lastVmdCompletedWindowMs = 0L
        latestFixedBpm = null
        latestFixedPeriodicity = 0f
        latestFixedPeakRatio = 0f
        vmdHeartWorker.invalidate()
        heartModeTracker.clear()
        heartRateArbitrator.clear()
        _vmdHeartStatus.value = VmdHeartStatus()
        _heartRateDecision.value = HeartRateDecision()
        if (clearSleepModelHistory) {
            // Source/contact loss and physical saturation also invalidate the
            // visible causal-filter state. Ordinary body motion deliberately
            // leaves this independent UI path running.
            displayDcBlocker.clear()
            displayAcquisitionLPF.clear()
            displayRespFIR.clear()
            displayRespHighPass1.clear()
            displayRespHighPass2.clear()
            displayHeartHighPass1.clear()
            displayHeartHighPass2.clear()
            displayHeartLowPass1.clear()
            displayHeartLowPass2.clear()
            displayHeartSmooth.clear()
            sleepHeartBuf.clear()
            sleepRespBuf.clear()
        }
    }

    private fun setSignalState(state: SignalState) {
        signalState = state
        _signalQuality.value = state
    }

    /**
     * Require one complete clean VMD window after an invalid interval.  Filters
     * and estimators run during this phase so they settle on real samples, but
     * waveform/rate/sleep buffers and formal vital outputs remain suppressed.
     */
    private fun beginRecoveryWarmup(timeMs: Long) {
        recoveryWarmupUntilMs = timeMs + recoveryWarmupMs
        setSignalState(SignalState.RECOVERING)
    }

    private fun finishRecoveryWarmupIfReady(timeMs: Long) {
        if (signalState == SignalState.RECOVERING && timeMs >= recoveryWarmupUntilMs) {
            recoveryWarmupUntilMs = 0L
            setSignalState(SignalState.MEASURING)
        }
    }

    fun reset() {
       rawBuf.clear()
        preBuf.clear()
       respBuf.clear()
        respDisplayBuf.clear()
        respRateBuf.clear()
       hrBuf.clear()
        cleanHeartBuf.clear()
        trackedHeartBuf.clear()
        vmdHeartBuf.clear()
        vmdMorphologyBuf.clear()
        vmdHeartbeatStreamPlayer.clear()
        vmdMorphologyStreamPlayer.clear()
        _vmdDisplayReady.value = false
        sleepHeartBuf.clear()
        sleepRespBuf.clear()
        paperHeartBuf.clear()
        heartDetailBuf.clear()
        enhancedHeartBuf.clear()
        morphologyHeartBuf.clear()
        zeroCrossBpm = null
        prevHrRaw = 0f
        lastHrZeroT = 0L
        isHrValid = false
        normIdx = 0; normCount = 0
        smoothHrCenter = 0f; smoothHrP2P = 0f
        respNormIdx = 0; respNormCount = 0; smoothRespP2P = 0f; smoothRespCenter = 0f
        rawIdx = 0; rawCount = 0
        centeredIdx = 0; centeredCount = 0
        lastPreviewEmitMs = Long.MIN_VALUE
        _rawPreview.value = ""
        _centeredPreview.value = ""
        latestSleepPrediction = null
        // A signal/source reset must not silently detach an active CSV logger;
        // the UI owns the logger lifecycle through Start/Stop CSV Logging.
        hrEstimator.clear()
        hrRateSmoother.clear()
        respRateSmoother.clear()
        hrHPF.clear()
        hrHPF2.clear()
        hrPostLPF.clear()
        hrCalc.clear()
        hrDisplaySmooth.clear()
        dcBlocker.clear()
        acquisitionLPF.clear()
        displayDcBlocker.clear()
        displayAcquisitionLPF.clear()
        displayRespFIR.clear()
        displayRespHighPass1.clear()
        displayRespHighPass2.clear()
        displayHeartHighPass1.clear()
        displayHeartHighPass2.clear()
        displayHeartLowPass1.clear()
        displayHeartLowPass2.clear()
        displayHeartSmooth.clear()
        respFIR.clear()
        respHighPass1.clear()
        respHighPass2.clear()
        respRateHighPass1.clear()
        respRateHighPass2.clear()
        respRateLowPass1.clear()
        respRateLowPass2.clear()
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
        adaptiveTrackedHeartFilter.clear()
        templateDetailHPF1.clear()
        templateDetailHPF2.clear()
        templateDetailLPF1.clear()
        templateDetailLPF2.clear()
        heartbeatTemplateEnhancer.clear()
        heartRateSynchronousPeakTracker.clear()
        morphologyRateReferenceTracker.clear()
        _heartTemplateStatus.value = HeartTemplateEnhancement()
        heartbeatMorphologyBuilder.clear()
        heartbeatMorphologyPlayer.clear()
        _heartMorphologySnapshot.value = MorphologyWaveformSnapshot()
        _morphologyPlaybackBoundaryTimes.value = emptyList()
        heartSep.clear()
        hrPeakDetector.clear()
        windowedHeartPeakDetector.clear()
        heartPeriodicityEstimator.clear()
        cleanHeartPeriodicityEstimator.clear()
        cleanHeartSpectrumEstimator.clear()
        heartRateAmbiguityResolver.clear()
        cleanHeartBeatDetector.clear()
        latestIntervalBpm = null
        latestIntervalQuality = 0f
        latestIntervalTimeMs = 0L
        cleanPeakTimesList.clear()
        _cleanPeakTimes.value = emptyList()
        lastAcceptedFormalBpmTimeMs = 0L
        lastAcceptedRespRateTimeMs = 0L
        motionFlag = false; motionEndTime = 0L; motionAbsBuf.clear(); motionRawBuf.clear()
        consecutiveClippedSamples = 0; saturationActive = false; saturationHoldUntilMs = 0L
        nearSaturationDetector.clear(); nearSaturationActive = false; nearSaturationHoldUntilMs = 0L
        slowContactArtifactDetector.clear()
        vmdRawBuf.clear(); vmdRespBuf.clear(); vmdCleanSampleCount = 0; lastVmdScheduleMs = 0L
        vmdHeartWorker.invalidate()
        val invalidTracking = heartModeTracker.update(
            candidates = emptyList(),
            fixedBpm = null,
            fixedPeriodicity = 0f,
            fixedSpectrumPeakRatio = 0f,
            respirationRpm = null,
            invalidSignal = true
        )
        lastVmdCompletedWindowMs = 0L
        latestFixedBpm = null; latestFixedPeriodicity = 0f; latestFixedPeakRatio = 0f
        _vmdHeartStatus.value = VmdHeartStatus(tracking = invalidTracking)
        heartRateArbitrator.clear()
        _heartRateDecision.value = HeartRateDecision()
        presenceObserver.reset(); _presenceObservation.value = PresenceObserver.Observation()
        lastBypassMotionDetection = false
        _slowContactArtifact.value = SlowContactArtifactDetector.Observation()
        recoveryWarmupUntilMs = 0L
        setSignalState(SignalState.MEASURING)
        zeroCrossRpm.clear()
        respirationStartupRateDetector.clear()
        respCycleRateDetector.clear()
        lastHrEstimateMs = 0L
        _rates.value = Rates()
        _heartRespHarmonicRisk.value = false
        smoothBpm = null
        lastBpmEstimate = null
        lastRpmEstimate = null
    }

    private fun respRateBufHasEnoughData(windowSec: Int): Boolean {
        val (ts, vs) = respRateBuf.snapshot()
        if (vs.size < fsHz * windowSec / 2) return false
        val latestTs = ts.lastOrNull() ?: return false
        val minTs = latestTs - windowSec * 1000L
        var count = 0
        for (i in vs.indices) {
            if (ts[i] >= minTs) count++
        }
        return count >= fsHz * windowSec * 8 / 10
    }

    private fun measureRecentRespRateAmplitude(windowSec: Int): Float {
        val (ts, vs) = respRateBuf.snapshot()
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






























