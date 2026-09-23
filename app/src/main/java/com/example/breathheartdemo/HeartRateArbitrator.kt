package com.example.breathheartdemo

import kotlin.math.abs

enum class HeartRateSource { NONE, VMD, FIXED_REFERENCE }

data class HeartRateDecision(
    val bpm: Float? = null,
    val source: HeartRateSource = HeartRateSource.NONE,
    /** True means [bpm], when present, is held for display and was not updated. */
    val stale: Boolean = false,
    val ambiguityReason: HeartRateAmbiguityReason = HeartRateAmbiguityReason.NONE,
    val primaryCandidateBpm: Float? = null,
    val alternateCandidateBpm: Float? = null,
    val confidence: Float = 0f
)

/**
 * Final arbitration between the tracked VMD path and the causal fixed channel.
 *
 * A twice-confirmed K=4 -> K=5 VMD mode is the primary measurement. The fixed
 * channel becomes a labelled fallback only after VMD is lost or stale and
 * three mutually consistent high-quality fixed updates have accumulated.
 */
class HeartRateArbitrator(
    private val agreementBpm: Float = 8f,
    private val minimumFixedPeriodicity: Float = 0.30f,
    private val minimumFixedPeakRatio: Float = 3f,
    private val fixedReferenceFrames: Int = 3,
    private val maximumHoldFrames: Int = 8
) {
    private var consecutiveFixedHighQuality = 0
    private var pendingFixedBpm: Float? = null
    private var lastPublishedBpm: Float? = null
    private var heldFrames = 0

    fun update(
        tracking: HeartTrackingResult,
        vmdFresh: Boolean,
        fixedBpm: Float?,
        fixedPeriodicity: Float,
        fixedPeakRatio: Float,
        invalidSignal: Boolean
    ): HeartRateDecision {
        if (invalidSignal) {
            consecutiveFixedHighQuality = 0
            pendingFixedBpm = null
            return heldDecision()
        }

        val fixedHighQuality = fixedBpm != null && fixedBpm in 48f..110f &&
            fixedPeriodicity >= minimumFixedPeriodicity &&
            fixedPeakRatio >= minimumFixedPeakRatio
        val vmdBpm = tracking.acceptedBpm
        val vmdUsable = vmdFresh &&
            (tracking.state == HeartTrackingState.LOCKED ||
                tracking.state == HeartTrackingState.FUSED) &&
            vmdBpm != null
        val selectedCandidate = tracking.selectedCandidate
        val multilayerPrimary = vmdUsable && selectedCandidate != null &&
            selectedCandidate.multilayerConfirmed &&
            selectedCandidate.periodicity >= 0.30f &&
            selectedCandidate.templateCorrelation >= 0.48f &&
            selectedCandidate.spectrumPeakRatio >= minimumFixedPeakRatio

        if (multilayerPrimary ||
            (vmdUsable && fixedHighQuality && abs(vmdBpm!! - fixedBpm!!) <= agreementBpm)
        ) {
            consecutiveFixedHighQuality = 0
            pendingFixedBpm = null
            lastPublishedBpm = vmdBpm!!
            heldFrames = 0
            return HeartRateDecision(vmdBpm, HeartRateSource.VMD, stale = false)
        }

        val fixedFallbackAllowed = tracking.state == HeartTrackingState.LOST ||
            (!vmdFresh && (tracking.state == HeartTrackingState.LOCKED ||
                tracking.state == HeartTrackingState.FUSED))
        consecutiveFixedHighQuality = if (fixedFallbackAllowed && fixedHighQuality) {
            val previous = pendingFixedBpm
            if (previous == null || abs(previous - fixedBpm!!) <= agreementBpm) {
                pendingFixedBpm = fixedBpm
                (consecutiveFixedHighQuality + 1).coerceAtMost(fixedReferenceFrames)
            } else {
                // Three high-quality values are not enough if they represent
                // mutually inconsistent spectral locks.
                pendingFixedBpm = fixedBpm
                1
            }
        } else {
            pendingFixedBpm = null
            0
        }
        if (fixedFallbackAllowed && consecutiveFixedHighQuality >= fixedReferenceFrames) {
            lastPublishedBpm = fixedBpm
            heldFrames = 0
            return HeartRateDecision(fixedBpm, HeartRateSource.FIXED_REFERENCE, stale = false)
        }

        return heldDecision()
    }

    fun invalidate(): HeartRateDecision {
        consecutiveFixedHighQuality = 0
        pendingFixedBpm = null
        return heldDecision()
    }

    fun clear() {
        consecutiveFixedHighQuality = 0
        pendingFixedBpm = null
        lastPublishedBpm = null
        heldFrames = 0
    }

    private fun heldDecision(): HeartRateDecision {
        val held = lastPublishedBpm ?: run {
            heldFrames = 0
            return HeartRateDecision()
        }
        heldFrames++
        if (heldFrames > maximumHoldFrames) {
            lastPublishedBpm = null
            heldFrames = 0
            return HeartRateDecision()
        }
        return HeartRateDecision(
            bpm = held,
            source = HeartRateSource.NONE,
            stale = true
        )
    }
}
