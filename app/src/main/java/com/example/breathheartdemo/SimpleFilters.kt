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


class DualSmoother(windowSize: Int) {
    private val ma1 = MovingAverage(windowSize)
    private val ma2 = MovingAverage(windowSize)
    fun next(x: Float): Float = ma2.next(ma1.next(x))
}

class HighPassFilter(private val alpha: Float) {
    private var yPrev = 0f
    private var xPrev = 0f

    fun next(x: Float): Float {
        // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        val y = alpha * (yPrev + x - xPrev)
        xPrev = x
        yPrev = y
        return y
    }

    fun clear() {
        yPrev = 0f
        xPrev = 0f
    }
}
