package com.example.breathheartdemo

import kotlin.math.abs

/** One rate hypothesis retained instead of discarding every peak except the largest one. */
data class HeartRateCandidateEvidence(
    val bpm: Float,
    /** Autocorrelation in [0, 1], or a spectrum peak/median ratio. */
    val strength: Float
)

enum class HeartRateAmbiguityReason {
    NONE,
    INSUFFICIENT_EVIDENCE,
    CONFIRMING,
    LOW_RATE_REVIEW,
    COMPETING_RATES
}

enum class HeartRateLockState { SEARCH, LOCKED, LOST }

data class HeartRateResolution(
    /** A new publishable rate. Null while evidence is ambiguous or still confirming. */
    val bpm: Float? = null,
    /** UI continuity only. A stale display value is never a new measurement. */
    val displayBpm: Float? = bpm,
    val stale: Boolean = false,
    val lockState: HeartRateLockState = HeartRateLockState.SEARCH,
    val primaryCandidateBpm: Float? = null,
    val alternateCandidateBpm: Float? = null,
    val confidence: Float = 0f,
    val reason: HeartRateAmbiguityReason = HeartRateAmbiguityReason.INSUFFICIENT_EVIDENCE
)

/**
 * Resolves fundamental/double-rate ambiguity using autocorrelation, spectrum,
 * grouped beat intervals and the consistency of real aligned beat morphology.
 *
 * A low rate is not rejected merely because it is below 55 bpm. It is withheld
 * only when a credible higher-rate competitor exists, or when the grouped beat
 * intervals do not support it. This keeps physiological bradycardia possible
 * while preventing a strong two-beat modulation from being published as HR.
 */
class HeartRateAmbiguityResolver(
    private val minBpm: Float = 40f,
    private val maxBpm: Float = 120f,
    private val lowRateReviewBpm: Float = 55f,
    private val clusterToleranceBpm: Float = 6f,
    private val minimumScore: Float = 0.40f,
    private val minimumMargin: Float = 0.15f,
    private val confirmationFrames: Int = 3,
    private val reacquisitionFrames: Int = 2,
    private val minimumFastReacquisitionTemplate: Float = 0.75f,
    private val minimumTrackedIntervalQuality: Float = 0.45f,
    private val minimumTrackedTemplateCorrelation: Float = 0.75f,
    private val maximumHoldFrames: Int = 8
) {
    private enum class Source { PERIODICITY, SPECTRUM, INTERVAL, TEMPLATE }

    private data class Cluster(
        var bpm: Float,
        var weight: Float = 0f,
        var periodicity: Float = 0f,
        var spectrum: Float = 0f,
        var interval: Float = 0f,
        var template: Float = 0f,
        var score: Float = 0f
    ) {
        val sourceCount: Int
            get() = listOf(periodicity, spectrum, interval, template).count { it > 0f }
        val hasFrequencyEvidence: Boolean
            get() = periodicity > 0f || spectrum > 0f
    }

    private var pendingBpm: Float? = null
    private var pendingFrames = 0
    private val pendingValues = ArrayDeque<Float>()
    private var lastAcceptedBpm: Float? = null
    private var lockState = HeartRateLockState.SEARCH
    private var rejectedFrames = 0

    fun update(
        periodicity: HeartPeriodicityResult,
        spectrum: HeartSpectrumResult,
        intervalBpm: Float?,
        intervalQuality: Float,
        respirationRpm: Float?,
        invalidSignal: Boolean,
        templateCorrelation: Float = 0f
    ): HeartRateResolution {
        if (invalidSignal) {
            return reject(
                reason = HeartRateAmbiguityReason.INSUFFICIENT_EVIDENCE,
                forceLost = true
            )
        }

        val clusters = mutableListOf<Cluster>()
        periodicity.candidates.ifEmpty {
            listOfNotNull(
                periodicity.bpm?.let {
                    HeartRateCandidateEvidence(it, periodicity.quality ?: 0f)
                }
            )
        }.forEach { addEvidence(clusters, Source.PERIODICITY, it.bpm, it.strength) }

        spectrum.candidates.ifEmpty {
            listOfNotNull(
                spectrum.bpm?.let {
                    HeartRateCandidateEvidence(it, spectrum.peakRatio ?: 0f)
                }
            )
        }.forEach { addEvidence(clusters, Source.SPECTRUM, it.bpm, it.strength) }

        if (intervalBpm != null && intervalQuality > 0f) {
            addEvidence(clusters, Source.INTERVAL, intervalBpm, intervalQuality)
            if (templateCorrelation > 0f) {
                // The template was built from these grouped beats, so its
                // evidence belongs to the interval-rate candidate.
                addEvidence(clusters, Source.TEMPLATE, intervalBpm, templateCorrelation)
            }
        }
        if (clusters.isEmpty()) return reject(HeartRateAmbiguityReason.INSUFFICIENT_EVIDENCE)

        for (cluster in clusters) {
            val sourceBonus = when (cluster.sourceCount) {
                4 -> 0.18f
                3 -> 0.15f
                2 -> 0.08f
                else -> 0f
            }
            val continuityBonus = lastAcceptedBpm?.let {
                if (abs(cluster.bpm - it) <= 8f) 0.10f else 0f
            } ?: 0f
            val respirationPenalty = if (nearRespirationHarmonic(cluster.bpm, respirationRpm)) {
                0.16f
            } else 0f
            cluster.score =
                0.38f * cluster.periodicity +
                0.32f * cluster.spectrum +
                0.30f * cluster.interval +
                0.18f * cluster.template +
                sourceBonus + continuityBonus - respirationPenalty
        }

        // Penalize a low candidate only when the higher competitor has an
        // interval observation plus a second, independent spectral/ACF clue.
        for (low in clusters.filter { it.bpm < lowRateReviewBpm }) {
            val high = clusters
                .filter { it.bpm / low.bpm in 1.55f..2.20f }
                .maxByOrNull { it.score }
            if (high != null &&
                high.interval >= 0.35f &&
                (high.periodicity >= 0.18f || high.spectrum >= 0.20f)
            ) {
                low.score -= 0.27f
                high.score += 0.08f
            }
        }

        val ranked = clusters.sortedByDescending { it.score }
        val primary = ranked.first()
        val alternate = ranked.drop(1).firstOrNull()
        val margin = primary.score - (alternate?.score ?: 0f)
        val confidence = (0.65f * primary.score + 0.35f * margin).coerceIn(0f, 1f)

        val strictFailure = when {
            primary.sourceCount < 2 ||
                !primary.hasFrequencyEvidence ||
                primary.score < minimumScore ->
                HeartRateAmbiguityReason.INSUFFICIENT_EVIDENCE
            alternate != null && margin < minimumMargin ->
                HeartRateAmbiguityReason.COMPETING_RATES
            primary.bpm < lowRateReviewBpm && primary.interval < 0.35f ->
                HeartRateAmbiguityReason.LOW_RATE_REVIEW
            else -> null
        }
        if (strictFailure != null && !canTrackLocked(primary, alternate, margin)) {
            return reject(
                strictFailure,
                primary.bpm,
                alternate?.bpm,
                confidence
            )
        }

        return acceptCandidate(primary, alternate, confidence)
    }

    private fun canTrackLocked(
        primary: Cluster,
        alternate: Cluster?,
        margin: Float
    ): Boolean {
        val lockedBpm = lastAcceptedBpm ?: return false
        if (lockState != HeartRateLockState.LOCKED) return false
        if (abs(primary.bpm - lockedBpm) > 12f) return false
        val morphologyBackedInterval =
            primary.interval >= minimumTrackedIntervalQuality &&
                primary.template >= minimumTrackedTemplateCorrelation
        if (primary.sourceCount < 2 ||
            (!primary.hasFrequencyEvidence && !morphologyBackedInterval)
        ) return false
        if (primary.score < 0.30f) return false
        if (primary.bpm < lowRateReviewBpm && primary.interval < 0.35f) return false
        if (margin >= 0.06f || alternate == null) return true
        // When two candidates are close in score, retain the one that preserves
        // the already locked physical trajectory.
        return abs(primary.bpm - lockedBpm) + 4f <
            abs(alternate.bpm - lockedBpm)
    }

    private fun acceptCandidate(
        primary: Cluster,
        alternate: Cluster?,
        confidence: Float
    ): HeartRateResolution {
        rejectedFrames = 0
        val fastReacquisition = lockState == HeartRateLockState.LOST &&
            lastAcceptedBpm?.let { abs(primary.bpm - it) <= 12f } == true &&
            primary.template >= minimumFastReacquisitionTemplate &&
            primary.interval > 0f &&
            primary.hasFrequencyEvidence
        val requiredFrames = if (fastReacquisition) {
            reacquisitionFrames
        } else {
            confirmationFrames
        }
        val previousPending = pendingBpm
        if (previousPending == null || abs(previousPending - primary.bpm) > 8f) {
            pendingValues.clear()
        }
        pendingValues.addLast(primary.bpm)
        while (pendingValues.size > requiredFrames) pendingValues.removeFirst()
        pendingFrames = pendingValues.size
        pendingBpm = pendingValues.average().toFloat()
        if (lockState != HeartRateLockState.LOCKED &&
            pendingFrames < requiredFrames
        ) {
            return HeartRateResolution(
                displayBpm = null,
                stale = false,
                lockState = lockState,
                primaryCandidateBpm = primary.bpm,
                alternateCandidateBpm = alternate?.bpm,
                confidence = confidence,
                reason = HeartRateAmbiguityReason.CONFIRMING
            )
        }

        lockState = HeartRateLockState.LOCKED
        lastAcceptedBpm = pendingBpm
        return HeartRateResolution(
            bpm = pendingBpm,
            displayBpm = pendingBpm,
            stale = false,
            lockState = lockState,
            primaryCandidateBpm = primary.bpm,
            alternateCandidateBpm = alternate?.bpm,
            confidence = confidence,
            reason = HeartRateAmbiguityReason.NONE
        )
    }

    private fun addEvidence(
        clusters: MutableList<Cluster>,
        source: Source,
        bpm: Float,
        rawStrength: Float
    ) {
        if (bpm !in minBpm..maxBpm || !bpm.isFinite() || !rawStrength.isFinite()) return
        val normalizedStrength = when (source) {
            Source.SPECTRUM -> ((rawStrength - 1f) / 5f).coerceIn(0f, 1f)
            else -> rawStrength.coerceIn(0f, 1f)
        }
        if (normalizedStrength <= 0f) return

        val cluster = clusters.minByOrNull { abs(it.bpm - bpm) }
            ?.takeIf { abs(it.bpm - bpm) <= clusterToleranceBpm }
            ?: Cluster(bpm = bpm).also(clusters::add)
        val updatedWeight = cluster.weight + normalizedStrength
        cluster.bpm = if (updatedWeight > 0f) {
            (cluster.bpm * cluster.weight + bpm * normalizedStrength) / updatedWeight
        } else bpm
        cluster.weight = updatedWeight
        when (source) {
            Source.PERIODICITY -> cluster.periodicity =
                maxOf(cluster.periodicity, normalizedStrength)
            Source.SPECTRUM -> cluster.spectrum =
                maxOf(cluster.spectrum, normalizedStrength)
            Source.INTERVAL -> cluster.interval =
                maxOf(cluster.interval, normalizedStrength)
            Source.TEMPLATE -> cluster.template =
                maxOf(cluster.template, normalizedStrength)
        }
    }

    private fun nearRespirationHarmonic(bpm: Float, respirationRpm: Float?): Boolean {
        if (respirationRpm == null || respirationRpm <= 0f) return false
        return (2..6).any { order -> abs(bpm - respirationRpm * order) <= 3f }
    }

    private fun reject(
        reason: HeartRateAmbiguityReason,
        primary: Float? = null,
        alternate: Float? = null,
        confidence: Float = 0f,
        forceLost: Boolean = false
    ): HeartRateResolution {
        if (forceLost) {
            lockState = HeartRateLockState.LOST
            pendingBpm = null
            pendingFrames = 0
            pendingValues.clear()
        }
        if (lastAcceptedBpm != null &&
            (lockState == HeartRateLockState.LOCKED || forceLost)
        ) {
            rejectedFrames++
            if (rejectedFrames <= maximumHoldFrames) {
                return HeartRateResolution(
                    displayBpm = lastAcceptedBpm,
                    stale = true,
                    lockState = lockState,
                    primaryCandidateBpm = primary,
                    alternateCandidateBpm = alternate,
                    confidence = confidence,
                    reason = reason
                )
            }
            lockState = HeartRateLockState.LOST
        }
        pendingBpm = null
        pendingFrames = 0
        pendingValues.clear()
        return HeartRateResolution(
            displayBpm = null,
            stale = false,
            lockState = lockState,
            primaryCandidateBpm = primary,
            alternateCandidateBpm = alternate,
            confidence = confidence,
            reason = reason
        )
    }

    fun clear() {
        pendingBpm = null
        pendingFrames = 0
        pendingValues.clear()
        lastAcceptedBpm = null
        lockState = HeartRateLockState.SEARCH
        rejectedFrames = 0
    }
}
