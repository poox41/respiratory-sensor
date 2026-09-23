package com.example.breathheartdemo

import kotlin.math.abs

enum class MorphologyRateReferenceSource {
    NONE,
    FRESH_FORMAL,
    STALE_FORMAL,
    FIXED_FALLBACK,
    HELD
}

data class MorphologyRateReference(
    val bpm: Float? = null,
    val source: MorphologyRateReferenceSource = MorphologyRateReferenceSource.NONE,
    /** Milliseconds since the last non-stale formal/fixed measurement. */
    val ageMs: Long? = null
)

/**
 * Display-only heart-period reference used to keep morphology segmentation moving.
 *
 * The formal BPM path remains unchanged.  A recently accepted rate may be held
 * for at most [maximumHoldMs] while VMD reacquires, because heart rate cannot
 * physically vanish between adjacent clean windows.  A fixed-band estimate can
 * refresh the reference only when it is high quality and close to the previous
 * accepted rate.  Motion/saturation clears the reference immediately.
 */
class MorphologyRateReferenceTracker(
    private val maximumHoldMs: Long = 30_000L,
    private val fixedAgreementBpm: Float = 8f,
    private val staleAgreementBpm: Float = 12f,
    private val minimumFixedPeriodicity: Float = 0.30f,
    private val minimumFixedPeakRatio: Float = 3f
) {
    private var lastReliableBpm: Float? = null
    private var lastEvidenceTimeMs: Long = 0L

    fun update(
        timeMs: Long,
        signalValid: Boolean,
        decision: HeartRateDecision,
        tracking: HeartTrackingResult,
        vmdFresh: Boolean,
        fixedBpm: Float?,
        fixedPeriodicity: Float,
        fixedPeakRatio: Float
    ): MorphologyRateReference {
        if (!signalValid) {
            clear()
            return MorphologyRateReference()
        }

        val freshFormal = decision.bpm
            ?.takeIf { !decision.stale && it.isFinite() && it in 40f..110f }
            ?: tracking.acceptedBpm?.takeIf {
                vmdFresh && it.isFinite() && it in 40f..110f &&
                    (tracking.state == HeartTrackingState.LOCKED ||
                        tracking.state == HeartTrackingState.FUSED)
            }
        if (freshFormal != null) {
            lastReliableBpm = freshFormal
            lastEvidenceTimeMs = timeMs
            return MorphologyRateReference(
                bpm = freshFormal,
                source = MorphologyRateReferenceSource.FRESH_FORMAL,
                ageMs = 0L
            )
        }

        val previous = lastReliableBpm
        val staleFormal = decision.bpm?.takeIf {
            decision.stale && previous != null && it.isFinite() && it in 40f..110f &&
                abs(it - previous) <= staleAgreementBpm
        }
        if (staleFormal != null && timeMs - lastEvidenceTimeMs <= maximumHoldMs) {
            return MorphologyRateReference(
                bpm = staleFormal,
                source = MorphologyRateReferenceSource.STALE_FORMAL,
                ageMs = (timeMs - lastEvidenceTimeMs).coerceAtLeast(0L)
            )
        }

        val fixedUsable = fixedBpm != null && previous != null &&
            fixedBpm.isFinite() && fixedBpm in 48f..110f &&
            fixedPeriodicity >= minimumFixedPeriodicity &&
            fixedPeakRatio >= minimumFixedPeakRatio &&
            abs(fixedBpm - previous) <= fixedAgreementBpm
        if (fixedUsable) {
            // Bound a one-frame correction so a marginal fixed-band lock cannot
            // jerk the displayed beat boundaries even though it passed quality.
            val smoothed = previous + (fixedBpm!! - previous).coerceIn(-3f, 3f)
            lastReliableBpm = smoothed
            lastEvidenceTimeMs = timeMs
            return MorphologyRateReference(
                bpm = smoothed,
                source = MorphologyRateReferenceSource.FIXED_FALLBACK,
                ageMs = 0L
            )
        }

        if (previous != null) {
            val age = (timeMs - lastEvidenceTimeMs).coerceAtLeast(0L)
            if (age <= maximumHoldMs) {
                return MorphologyRateReference(
                    bpm = previous,
                    source = MorphologyRateReferenceSource.HELD,
                    ageMs = age
                )
            }
        }
        clear()
        return MorphologyRateReference()
    }

    fun clear() {
        lastReliableBpm = null
        lastEvidenceTimeMs = 0L
    }
}
