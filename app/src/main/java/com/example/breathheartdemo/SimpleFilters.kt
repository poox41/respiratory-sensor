package com.example.breathheartdemo
// Simple filters for separating respiration and heart signals.
class MovingAverage(private val windowSize: Int) {
    private val buf = FloatArray(windowSize)
    private var sum = 0f
    private var idx = 0
    private var filled = 0

    fun next(x: Float): Float {
        sum -= buf[idx]
        buf[idx] = x
        sum += x
        idx = (idx + 1) % windowSize
        if (filled < windowSize) filled++
        return sum / filled
    }
}

class ShortSmoother(windowSize: Int) {
    private val ma = MovingAverage(windowSize)
    fun next(x: Float) = ma.next(x)
}
