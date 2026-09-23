package com.example.breathheartdemo

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Reuses feature rows shared by two overlapping sleep-model windows.
 *
 * A prediction is made every 30 seconds while the model consumes eleven
 * consecutive 30-second epochs. Therefore ten rows normally overlap. The
 * cache only reuses rows when the timestamp advance is an integer number of
 * epochs; irregular gaps and clock rollbacks force a complete rebuild.
 */
class SleepFeatureSequenceCache(
    private val sequenceLength: Int,
    private val featureDimension: Int,
    private val epochDurationMs: Long
) {
    init {
        require(sequenceLength > 0) { "sequenceLength must be positive" }
        require(featureDimension > 0) { "featureDimension must be positive" }
        require(epochDurationMs > 0L) { "epochDurationMs must be positive" }
    }

    private var cachedTimestampMs: Long? = null
    private var cachedFeatures: FloatArray? = null

    @Synchronized
    fun build(timestampMs: Long, extractEpoch: (Int) -> FloatArray): FloatArray {
        val previousTimestamp = cachedTimestampMs
        val previous = cachedFeatures
        if (previousTimestamp == timestampMs && previous != null) return previous.copyOf()

        val shift = reusableEpochShift(previousTimestamp, timestampMs)
        val reusableShift = shift?.takeIf { it in 1 until sequenceLength }
        val result = FloatArray(sequenceLength * featureDimension)
        val firstEpochToExtract = if (previous != null && reusableShift != null) {
            val retainedEpochs = sequenceLength - reusableShift
            previous.copyInto(
                destination = result,
                destinationOffset = 0,
                startIndex = reusableShift * featureDimension,
                endIndex = sequenceLength * featureDimension
            )
            retainedEpochs
        } else {
            0
        }

        for (epoch in firstEpochToExtract until sequenceLength) {
            val row = extractEpoch(epoch)
            require(row.size == featureDimension) {
                "epoch $epoch returned ${row.size} features, expected $featureDimension"
            }
            require(row.all { it.isFinite() }) {
                "epoch $epoch returned NaN or infinite features"
            }
            row.copyInto(result, destinationOffset = epoch * featureDimension)
        }

        // Commit only after every requested epoch has been extracted and
        // validated, so an extractor failure cannot poison the previous cache.
        cachedTimestampMs = timestampMs
        cachedFeatures = result.copyOf()
        return result
    }

    private fun reusableEpochShift(previousTimestampMs: Long?, timestampMs: Long): Int? {
        previousTimestampMs ?: return null
        val deltaMs = timestampMs - previousTimestampMs
        if (deltaMs <= 0L) return null
        val nearestEpochs = (deltaMs.toDouble() / epochDurationMs.toDouble()).roundToInt()
        if (nearestEpochs <= 0) return null
        // Sensor timestamps normally differ by one 20 ms sample at most.
        // Keep a small absolute tolerance but never treat an arbitrary partial
        // epoch as reusable.
        val alignmentErrorMs = abs(deltaMs - nearestEpochs * epochDurationMs)
        val toleranceMs = minOf(100L, epochDurationMs / 100L)
        return nearestEpochs.takeIf { alignmentErrorMs <= toleranceMs }
    }

    @Synchronized
    fun clear() {
        cachedTimestampMs = null
        cachedFeatures = null
    }
}
