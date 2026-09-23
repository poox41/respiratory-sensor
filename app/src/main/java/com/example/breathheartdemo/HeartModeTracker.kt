package com.example.breathheartdemo

import kotlin.math.abs
import kotlin.math.min

/** A VMD mode summarized for temporal identity tracking. */
data class HeartModeCandidate(
    val centerHz: Float,
    val periodicity: Float,
    val templateCorrelation: Float,
    val distinctHarmonicOrders: Int,
    val respirationCoherence: Float,
    val spectrumPeakRatio: Float,
    /** True only after the paper's K=4 -> K=5 cardiac-band confirmation. */
    val multilayerConfirmed: Boolean = false
) {
    val bpm: Float get() = centerHz * 60f
}

enum class HeartTrackingState { SEARCH, LOCKED, FUSED, LOST }

data class HeartTrackingResult(
    val state: HeartTrackingState = HeartTrackingState.SEARCH,
    /** A newly accepted measurement. Null in SEARCH/LOST. */
    val acceptedBpm: Float? = null,
    /** Last accepted value for UI continuity only; never a fresh measurement. */
    val displayBpm: Float? = null,
    val stale: Boolean = false,
    /** High-quality fixed-path estimate used only as a reacquisition reference. */
    val fixedReferenceBpm: Float? = null,
    val selectedCandidate: HeartModeCandidate? = null
)

/**
 * Online Viterbi mode-identity tracker plus SEARCH/LOCKED/FUSED/LOST control.
 *
 * This class does not perform VMD.  It consumes mode observations produced by
 * a block VMD backend and keeps a low-cost path across overlapping windows.
 * All quality features are expected in [0, 1], except spectrumPeakRatio.
 */
class HeartModeTracker(
    private val minBpm: Float = 48f,
    private val maxBpm: Float = 110f,
    private val agreementBpm: Float = 8f,
    private val minimumPeriodicity: Float = 0.30f,
    private val minimumTemplateCorrelation: Float = 0.48f,
    private val minimumSpectrumPeakRatio: Float = 3f,
    private val lockFrames: Int = 2,
    private val fixedReferenceFrames: Int = 3
) {
    private data class Path(val bpm: Float?, val cost: Float, val candidateIndex: Int?)

    private var paths = listOf(Path(bpm = null, cost = 0f, candidateIndex = null))
    private var state = HeartTrackingState.SEARCH
    private var consecutiveLockable = 0
    private var consecutiveFixedReference = 0
    private var lastAcceptedBpm: Float? = null

    @Synchronized
    fun update(
        candidates: List<HeartModeCandidate>,
        fixedBpm: Float?,
        fixedPeriodicity: Float,
        fixedSpectrumPeakRatio: Float,
        respirationRpm: Float?,
        invalidSignal: Boolean
    ): HeartTrackingResult {
        val fixedHighQuality = fixedBpm != null &&
            fixedBpm in minBpm..maxBpm &&
            fixedPeriodicity >= minimumPeriodicity &&
            fixedSpectrumPeakRatio >= minimumSpectrumPeakRatio

        if (invalidSignal) {
            enterLost()
            consecutiveFixedReference = 0
            return result(selected = null, fixedReference = null)
        }

        val selected = selectViterbiCandidate(candidates, fixedBpm)
        val lockable = selected != null && isLockable(selected, fixedBpm)
        val fused = selected != null && isRespiratoryHarmonicAmbiguous(selected, respirationRpm)

        when (state) {
            HeartTrackingState.SEARCH -> {
                consecutiveLockable = if (lockable) consecutiveLockable + 1 else 0
                if (consecutiveLockable >= lockFrames) {
                    state = if (fused) HeartTrackingState.FUSED else HeartTrackingState.LOCKED
                    lastAcceptedBpm = selected!!.bpm
                }
            }

            HeartTrackingState.LOCKED -> {
                if (!lockable) {
                    enterLost()
                } else {
                    state = if (fused) HeartTrackingState.FUSED else HeartTrackingState.LOCKED
                    lastAcceptedBpm = selected!!.bpm
                }
            }

            HeartTrackingState.FUSED -> {
                // In a fused respiratory-harmonic region, template evidence is
                // mandatory and the raw respiration-dominated signal is never
                // used as the matching input.
                if (!lockable || selected!!.templateCorrelation < minimumTemplateCorrelation) {
                    enterLost()
                } else {
                    state = if (fused) HeartTrackingState.FUSED else HeartTrackingState.LOCKED
                    lastAcceptedBpm = selected.bpm
                }
            }

            HeartTrackingState.LOST -> {
                consecutiveLockable = if (lockable) consecutiveLockable + 1 else 0
                if (consecutiveLockable >= lockFrames) {
                    state = if (fused) HeartTrackingState.FUSED else HeartTrackingState.LOCKED
                    lastAcceptedBpm = selected!!.bpm
                    consecutiveFixedReference = 0
                }
            }
        }

        consecutiveFixedReference = if (
            state == HeartTrackingState.LOST && fixedHighQuality
        ) consecutiveFixedReference + 1 else 0
        val fixedReference = if (
            state == HeartTrackingState.LOST &&
            consecutiveFixedReference >= fixedReferenceFrames
        ) fixedBpm else null

        return result(selected = selected, fixedReference = fixedReference)
    }

    private fun result(
        selected: HeartModeCandidate?,
        fixedReference: Float?
    ): HeartTrackingResult {
        val accepted = if (
            state == HeartTrackingState.LOCKED || state == HeartTrackingState.FUSED
        ) lastAcceptedBpm else null
        return HeartTrackingResult(
            state = state,
            acceptedBpm = accepted,
            displayBpm = lastAcceptedBpm,
            stale = state == HeartTrackingState.LOST && lastAcceptedBpm != null,
            fixedReferenceBpm = fixedReference,
            selectedCandidate = selected
        )
    }

    private fun isLockable(candidate: HeartModeCandidate, fixedBpm: Float?): Boolean {
        // A twice-selected multilayer IMF is the primary observation. The
        // legacy fixed path remains useful as a fallback, but a subharmonic
        // fixed estimate must not veto a strong multilayer cardiac mode.
        val identitySupported = candidate.multilayerConfirmed ||
            candidate.distinctHarmonicOrders >= 2
        val referenceSupported = if (candidate.multilayerConfirmed) {
            true
        } else {
            fixedBpm != null && abs(candidate.bpm - fixedBpm) <= agreementBpm
        }
        return candidate.bpm in minBpm..maxBpm &&
            candidate.periodicity >= minimumPeriodicity &&
            candidate.templateCorrelation >= minimumTemplateCorrelation &&
            identitySupported &&
            candidate.spectrumPeakRatio >= minimumSpectrumPeakRatio &&
            referenceSupported
    }

    private fun isRespiratoryHarmonicAmbiguous(
        candidate: HeartModeCandidate,
        respirationRpm: Float?
    ): Boolean {
        if (respirationRpm == null || respirationRpm <= 0f) return false
        val respirationHz = respirationRpm / 60f
        for (order in 2..6) {
            if (abs(candidate.centerHz - order * respirationHz) < 0.15f) return true
        }
        return false
    }

    private fun selectViterbiCandidate(
        candidates: List<HeartModeCandidate>,
        fixedBpm: Float?
    ): HeartModeCandidate? {
        if (candidates.isEmpty()) {
            paths = listOf(Path(bpm = null, cost = 0f, candidateIndex = null))
            return null
        }

        val newPaths = ArrayList<Path>(candidates.size + 1)
        candidates.forEachIndexed { index, candidate ->
            var bestCost = Float.POSITIVE_INFINITY
            for (previous in paths) {
                val transition = if (previous.bpm == null) {
                    0.30f
                } else {
                    0.32f * huber((candidate.bpm - previous.bpm) / agreementBpm)
                }
                bestCost = min(bestCost, previous.cost + transition)
            }
            newPaths += Path(
                bpm = candidate.bpm,
                cost = bestCost + emissionCost(candidate, fixedBpm),
                candidateIndex = index
            )
        }

        // LOST is an explicit Viterbi state. It is cheaper than forcing a poor
        // candidate, but carries a small persistence cost.
        val plausibleCandidatePresent = candidates.any { it.bpm in minBpm..maxBpm }
        val lostCost = paths.minOf { previous ->
            previous.cost + if (plausibleCandidatePresent) {
                0.70f
            } else if (previous.bpm == null) {
                0.08f
            } else {
                0.18f
            }
        }
        newPaths += Path(bpm = null, cost = lostCost, candidateIndex = null)

        val offset = newPaths.minOf { it.cost }
        paths = newPaths.map { it.copy(cost = it.cost - offset) }
        val best = paths.minByOrNull { it.cost } ?: return null
        return best.candidateIndex?.let(candidates::get)
    }

    private fun emissionCost(candidate: HeartModeCandidate, fixedBpm: Float?): Float {
        val periodicity = candidate.periodicity.coerceIn(0f, 1f)
        val template = candidate.templateCorrelation.coerceIn(0f, 1f)
        val harmonics = if (candidate.multilayerConfirmed) 1f
        else (candidate.distinctHarmonicOrders / 3f).coerceIn(0f, 1f)
        val sharpness = ((candidate.spectrumPeakRatio - 3f) / 5f).coerceIn(0f, 1f)
        val respiration = candidate.respirationCoherence.coerceIn(0f, 1f)
        val rangePenalty = if (candidate.bpm in minBpm..maxBpm) 0f else 1f
        val fixedPenalty = if (candidate.multilayerConfirmed) 0f
        else if (fixedBpm == null) 0.25f
        else min(abs(candidate.bpm - fixedBpm) / agreementBpm, 2f) * 0.16f
        return 0.28f * (1f - periodicity) +
            0.22f * (1f - template) +
            0.18f * (1f - harmonics) +
            0.10f * (1f - sharpness) +
            0.18f * respiration +
            0.40f * rangePenalty +
            fixedPenalty
    }

    private fun huber(value: Float, delta: Float = 1f): Float {
        val magnitude = abs(value)
        return if (magnitude <= delta) 0.5f * magnitude * magnitude
        else delta * (magnitude - 0.5f * delta)
    }

    private fun enterLost() {
        state = HeartTrackingState.LOST
        consecutiveLockable = 0
    }

    @Synchronized
    fun clear() {
        paths = listOf(Path(bpm = null, cost = 0f, candidateIndex = null))
        state = HeartTrackingState.SEARCH
        consecutiveLockable = 0
        consecutiveFixedReference = 0
        lastAcceptedBpm = null
    }
}
