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
}
