package com.example.breathheartdemo
//存最近10秒波形，永远只存固定容量,新数据进来会覆盖最旧的数据.
class RingBuffer(private val capacity: Int) {
    private val t = LongArray(capacity)
    private val v = FloatArray(capacity)
    private var size = 0
    private var head = 0

    fun add(tMs: Long, value: Float) {
        t[head] = tMs
        v[head] = value
        head = (head + 1) % capacity
        if (size < capacity) size++
    }

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

    fun isEmpty() = size == 0

    fun clear() {
        size = 0
        head = 0
    }
}
