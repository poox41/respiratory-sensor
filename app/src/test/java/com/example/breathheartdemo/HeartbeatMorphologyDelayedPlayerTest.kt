package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HeartbeatMorphologyDelayedPlayerTest {
    @Test
    fun replaysSnapshotContinuouslyAtSensorRateWithThreeSecondDelay() {
        val fsHz = 50
        val delayMs = 3_000L
        val player = HeartbeatMorphologyDelayedPlayer(fsHz = fsHz, delayMs = delayMs)
        val times = LongArray(321) { index -> index * 20L }
        val values = FloatArray(times.size) { index ->
            sin(2.0 * PI * times[index] / 800.0).toFloat()
        }
        val boundaries = (0L..6_400L step 800L).toList()
        val snapshot = MorphologyWaveformSnapshot(
            timesMs = times,
            values = values,
            boundaryTimesMs = boundaries,
            cycles = 8,
            quality = 0.9f,
            durationMs = 6_400L,
            medianRrMs = 800L,
            ready = true
        )
        player.update(snapshot)

        assertNull(player.next(delayMs - 20L))
        val output = ArrayList<Float>()
        for (timeMs in delayMs..delayMs + 1_600L step 20L) {
            output.add(player.next(timeMs)!!)
        }

        assertEquals(values[0], output[0], 1e-5f)
        assertTrue("playback did not change continuously", output.zipWithNext().any { (a, b) -> abs(b - a) > 0.01f })
        assertTrue(
            "playback step exceeded the slew limit",
            output.zipWithNext().all { (a, b) -> abs(b - a) <= 0.301f }
        )
        assertEquals(boundaries.map { it + delayMs }, player.delayedBoundaryTimesMs)
    }
}
