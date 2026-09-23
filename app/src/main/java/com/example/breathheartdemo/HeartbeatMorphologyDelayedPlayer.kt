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
    private val maximumHeldRepeatMs: Long = 20_000L,
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

    /**
     * Refill a display buffer after startup or a long playback gap.
     *
     * Only snapshot samples whose delayed presentation time is not in the
     * future are copied. [currentValue] closes a possible interpolation gap at
     * [timeMs]. This prevents a retained chart from collapsing to one point
     * when live playback resumes after several seconds.
     */
    fun seedDisplayHistory(
        buffer: RingBuffer,
        timeMs: Long,
        currentValue: Float
    ): Boolean {
        if (!snapshot.ready || snapshot.timesMs.size < 2) return false
        var sourceCount = 0
        while (sourceCount < snapshot.timesMs.size &&
            snapshot.timesMs[sourceCount] + delayMs <= timeMs
        ) {
            sourceCount++
        }
        if (sourceCount < 2) return false

        val lastDelayedTime = snapshot.timesMs[sourceCount - 1] + delayMs
        val appendCurrent = lastDelayedTime < timeMs
        val outputSize = sourceCount + if (appendCurrent) 1 else 0
        val times = LongArray(outputSize)
        val values = FloatArray(outputSize)
        for (index in 0 until sourceCount) {
            times[index] = snapshot.timesMs[index] + delayMs
            values[index] = snapshot.values[index]
        }
        if (appendCurrent) {
            times[outputSize - 1] = timeMs
            values[outputSize - 1] = currentValue
        }
        buffer.replace(times, values)
        return true
    }

    /**
     * Returns one continuously replayed value for [timeMs], or null before ready.
     *
     * During an explicitly invalid/recovery interval, [allowHeldRepeat] may
     * repeat only the final completed real cycle for a bounded time. This is a
     * display hold: it never enters BPM, VMD, sleep features, or any estimator.
     * It prevents the enhancement card from becoming empty while the strict
     * 16-second clean-window path reacquires.
     */
    fun next(timeMs: Long, allowHeldRepeat: Boolean = false): Float? {
        if (!snapshot.ready || snapshot.timesMs.isEmpty()) return null
        var sourceTimeMs = timeMs - delayMs
        val times = snapshot.timesMs
        val values = snapshot.values
        if (sourceTimeMs < times.first()) return null
        if (sourceTimeMs > times.last()) {
            if (!allowHeldRepeat || sourceTimeMs - times.last() > maximumHeldRepeatMs) {
                return null
            }
            val boundaries = snapshot.boundaryTimesMs
            if (boundaries.size < 2) return null
            val cycleStart = boundaries[boundaries.lastIndex - 1]
            val cycleEnd = boundaries.last()
            val cycleDuration = cycleEnd - cycleStart
            if (cycleDuration <= 0L) return null
            sourceTimeMs = cycleStart + (sourceTimeMs - cycleStart) % cycleDuration
        }

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
