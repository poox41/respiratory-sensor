package com.example.breathheartdemo

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
//鎷嗗垎鍑簉esp/hr,骞剁畻bpm/rpm
class Processor(private val fsHz: Int) {
    private val cap = fsHz * 10
    val rawBuf = RingBuffer(cap)
    val respBuf = RingBuffer(cap)
    val hrBuf = RingBuffer(cap)

    private val respLP = MovingAverage(windowSize = fsHz * 2)
    private val hrSmooth = ShortSmoother(windowSize = maxOf(1, fsHz / 10))

    private val hrEstimator = PeakRateEstimator(fsHz, refractoryMs = 250, windowSec = 10)
    private val respEstimator = PeakRateEstimator(fsHz, refractoryMs = 1500, windowSec = 30)

    private val _rates = MutableStateFlow(Rates())
    val rates: StateFlow<Rates> = _rates

    private var lastEstimateMs = 0L

    fun onSample(sample: Sample) {
        val t = sample.tMs
        val x = sample.x

        val resp = respLP.next(x)
        val hrRaw = x - resp
        val hr = hrSmooth.next(hrRaw)

        rawBuf.add(t, x)
        respBuf.add(t, resp)
        hrBuf.add(t, hr)

        hrEstimator.add(t, hr)
        respEstimator.add(t, resp)

        if (t - lastEstimateMs >= 1000L) {
            lastEstimateMs = t
            _rates.value = Rates(
                bpm = hrEstimator.estimate(),
                rpm = respEstimator.estimate()
            )
        }
    }

    fun reset() {
        rawBuf.clear()
        respBuf.clear()
        hrBuf.clear()
        hrEstimator.clear()
        respEstimator.clear()
        lastEstimateMs = 0L
        _rates.value = Rates()
    }
}
