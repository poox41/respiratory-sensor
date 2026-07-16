package com.example.breathheartdemo

/**
 * Replays completed morphology snapshots sample by sample with a fixed delay.
 *
 * The delay keeps the playback cursor inside the latest completed-cycle
 * snapshot between two detected peaks. This makes the chart scroll at the
 * sensor sample rate instead of replacing the entire waveform once per beat.
 */
class HeartbeatMorphologyDelayedPlayer(
    private val fsHz: Int,
    val delayMs: Long = 3_000L,
    private val maximumDisplaySlopePerSecond: Float = 15f
) {
    private var snapshot = MorphologyWaveformSnapshot()
    private var lastValue = 0f
    private var hasLastValue = false

    val delayedBoundaryTimesMs: List<Long>
        get() = if (snapshot.ready) {
            snapshot.boundaryTimesMs.map { it + delayMs }
        } else {
            emptyList()
        }

    fun update(nextSnapshot: MorphologyWaveformSnapshot) {
        if (!nextSnapshot.ready || nextSnapshot.timesMs.isEmpty()) return
        snapshot = nextSnapshot
    }

    /** Returns one continuously replayed value for [timeMs], or null before ready. */
    fun next(timeMs: Long): Float? {
        if (!snapshot.ready || snapshot.timesMs.isEmpty()) return null
        val sourceTimeMs = timeMs - delayMs
        val times = snapshot.timesMs
        val values = snapshot.values
        if (sourceTimeMs < times.first() || sourceTimeMs > times.last()) return null

        var low = 0
        var high = times.lastIndex
        while (low < high) {
            val middle = (low + high) ushr 1
            if (times[middle] < sourceTimeMs) low = middle + 1 else high = middle
        }
        val upper = low
        val target = if (times[upper] == sourceTimeMs || upper == 0) {
            values[upper]
        } else {
            val lower = upper - 1
            val duration = (times[upper] - times[lower]).coerceAtLeast(1L)
            val fraction = (sourceTimeMs - times[lower]).toFloat() / duration.toFloat()
            values[lower] + (values[upper] - values[lower]) * fraction
        }
        return continuousValue(target)
    }

    private fun continuousValue(target: Float): Float {
        if (!hasLastValue) {
            hasLastValue = true
            lastValue = target
            return target
        }
        val maximumStep = (maximumDisplaySlopePerSecond / fsHz.toFloat()).coerceAtLeast(0.01f)
        lastValue += (target - lastValue).coerceIn(-maximumStep, maximumStep)
        return lastValue
    }

    fun clear() {
        snapshot = MorphologyWaveformSnapshot()
        lastValue = 0f
        hasLastValue = false
    }
}
