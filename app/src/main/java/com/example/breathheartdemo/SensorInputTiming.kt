package com.example.breathheartdemo

import kotlin.math.abs

/** Synthetic sample time follows the configured decoded-value rate, not BLE burst cadence. */
internal class SensorSampleClock(
    sampleRateHz: Int,
    private val receiveGapMs: Long = 2_000L
) {
    init { require(sampleRateHz > 0) }

    private val periodMs = 1_000.0 / sampleRateHz
    private var nextTimeMs: Double? = null
    private var lastReceiveTimeMs: Long? = null

    fun beginPacket(receiveTimeMs: Long) {
        val next = nextTimeMs
        val previousReceive = lastReceiveTimeMs
        if (next == null) {
            nextTimeMs = receiveTimeMs.toDouble()
        } else if (previousReceive != null && receiveTimeMs - previousReceive > receiveGapMs) {
            // A real receive gap can move the clock forward, never backward.
            nextTimeMs = maxOf(next, receiveTimeMs.toDouble())
        }
        lastReceiveTimeMs = receiveTimeMs
    }

    fun nextTimestampMs(): Long {
        val current = checkNotNull(nextTimeMs) { "beginPacket must precede a decoded sample" }
        nextTimeMs = current + periodMs
        return current.toLong()
    }

    fun leadMs(receiveTimeMs: Long): Long? = nextTimeMs?.let { it.toLong() - receiveTimeMs }

    fun clear() {
        nextTimeMs = null
        lastReceiveTimeMs = null
    }
}

internal data class ReceiveRateObservation(
    val totalDecodedValues: Long,
    val observedValuesPerSecond: Double?,
    val status: String
)

/** Independent receive-clock audit; it never drops samples or changes the rate. */
internal class BleReceiveRateMonitor(
    private val configuredSampleRateHz: Int,
    private val windowMs: Long = 10_000L
) {
    init { require(configuredSampleRateHz > 0 && windowMs > 0) }

    private var startedElapsedMs: Long? = null
    private var startedCount = 0L
    private var total = 0L
    private var rate: Double? = null

    fun observe(receiveElapsedMs: Long, decodedValues: Int): ReceiveRateObservation {
        require(decodedValues >= 0)
        total += decodedValues
        val start = startedElapsedMs
        if (start == null || receiveElapsedMs < start) {
            startedElapsedMs = receiveElapsedMs
            startedCount = total
            rate = null
        } else if (receiveElapsedMs - start >= windowMs) {
            rate = (total - startedCount) * 1_000.0 / (receiveElapsedMs - start)
            startedElapsedMs = receiveElapsedMs
            startedCount = total
        }
        val measured = rate
        val status = when {
            measured == null -> "WAITING"
            abs(measured - configuredSampleRateHz) <= configuredSampleRateHz * 0.20 -> "MATCH"
            else -> "MISMATCH"
        }
        return ReceiveRateObservation(total, measured, status)
    }

    fun clear() {
        startedElapsedMs = null
        startedCount = 0L
        total = 0L
        rate = null
    }
}
