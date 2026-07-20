package com.example.breathheartdemo
// Keep a fixed-capacity recent waveform history and overwrite the oldest samples.
class RingBuffer(private val capacity: Int) {
    private val t = LongArray(capacity)
    private val v = FloatArray(capacity)
    private var size = 0
    private var head = 0

    @Synchronized
    fun add(tMs: Long, value: Float) {
        addUnsafe(tMs, value)
    }

    private fun addUnsafe(tMs: Long, value: Float) {
        t[head] = tMs
        v[head] = value
        head = (head + 1) % capacity
        if (size < capacity) size++
    }

    @Synchronized
    fun snapshot(): Pair<LongArray, FloatArray> {
        val outT = LongArray(size)
        val outV = FloatArray(size)
        val start = if (size == capacity) head else 0
        for (i in 0 until size) {
            val idx = (start + i) % capacity
            outT[i] = t[idx]
            outV[i] = v[idx]
        }
        return outT to outV
    }

    /** Copy only the visible tail of the buffer instead of the full model history. */
    @Synchronized
    fun snapshotWindow(windowMs: Long): Pair<LongArray, FloatArray> {
        if (size == 0) return LongArray(0) to FloatArray(0)
        val latestIndex = logicalToPhysical(size - 1)
        val minimumTime = t[latestIndex] - windowMs.coerceAtLeast(0L)
        val first = lowerBoundTime(minimumTime)
        val count = size - first
        val outT = LongArray(count)
        val outV = FloatArray(count)
        for (offset in 0 until count) {
            val source = logicalToPhysical(first + offset)
            outT[offset] = t[source]
            outV[offset] = v[source]
        }
        return outT to outV
    }

    /** Return min/max for a recent window without allocating waveform arrays. */
    @Synchronized
    fun valueRange(windowMs: Long): ValueRange? {
        if (size == 0) return null
        val latestIndex = logicalToPhysical(size - 1)
        val latestTime = t[latestIndex]
        val first = lowerBoundTime(latestTime - windowMs.coerceAtLeast(0L))
        var minimum = Float.POSITIVE_INFINITY
        var maximum = Float.NEGATIVE_INFINITY
        for (logical in first until size) {
            val value = v[logicalToPhysical(logical)]
            if (!value.isFinite()) continue
            if (value < minimum) minimum = value
            if (value > maximum) maximum = value
        }
        return if (minimum.isFinite() && maximum.isFinite()) {
            ValueRange(minimum, maximum, latestTime, size - first)
        } else {
            null
        }
    }

    /** Timestamp-only copy used by model readiness checks. */
    @Synchronized
    fun timestampsSnapshot(): LongArray {
        val out = LongArray(size)
        for (logical in 0 until size) {
            out[logical] = t[logicalToPhysical(logical)]
        }
        return out
    }

    @Synchronized
    fun isEmpty() = size == 0

    @Synchronized
    fun clear() {
        size = 0
        head = 0
    }

    /** Atomically replace this UI buffer with an immutable reconstructed snapshot. */
    @Synchronized
    fun replace(timesMs: LongArray, values: FloatArray) {
        require(timesMs.size == values.size) { "times and values must have the same size" }
        size = 0
        head = 0
        val start = (timesMs.size - capacity).coerceAtLeast(0)
        for (index in start until timesMs.size) {
            addUnsafe(timesMs[index], values[index])
        }
    }

    private fun logicalToPhysical(logicalIndex: Int): Int {
        val start = if (size == capacity) head else 0
        return (start + logicalIndex) % capacity
    }

    private fun lowerBoundTime(targetMs: Long): Int {
        var low = 0
        var high = size
        while (low < high) {
            val middle = (low + high) ushr 1
            if (t[logicalToPhysical(middle)] < targetMs) low = middle + 1 else high = middle
        }
        return low
    }
}

data class ValueRange(
    val minimum: Float,
    val maximum: Float,
    val latestTimeMs: Long,
    val sampleCount: Int
)
