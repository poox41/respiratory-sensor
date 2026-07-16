package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HeartbeatMorphologySnapshotBuilderTest {
    @Test
    fun buildsEightRealCyclesWithActualIntervalsAndSmallVariation() {
        val fsHz = 50
        val builder = HeartbeatMorphologySnapshotBuilder(fsHz = fsHz)
        val intervals = listOf(800L, 780L, 820L, 760L, 840L, 800L, 780L, 820L)
        val boundaries = mutableListOf(1_000L)
        for (interval in intervals) boundaries.add(boundaries.last() + interval)
        val boundarySet = boundaries.toSet()
        var latest: MorphologyWaveformSnapshot? = null

        for (timeMs in 0L..boundaries.last() step 20L) {
            val cycleIndex = cycleIndex(timeMs, boundaries)
            val phase = cyclePhase(timeMs, boundaries, intervals)
            val jitter = ((cycleIndex % 3) - 1) * 0.006
            val amplitude = 1.0 + ((cycleIndex % 5) - 2) * 0.025
            val main = exp(-square((phase - (0.29 + jitter)) / 0.060))
            val smallPeak = exp(-square((phase - (0.54 - jitter)) / 0.055))
            val trough = exp(-square((phase - 0.78) / 0.095))
            val input = (
                amplitude * (1.10 * main + 0.26 * smallPeak - 0.55 * trough) +
                    0.06 * sin(2.0 * PI * phase)
                ).toFloat()
            builder.next(
                value = input,
                timeMs = timeMs,
                detectedPeakTimeMs = timeMs.takeIf { it in boundarySet }
            )?.let { latest = it }
        }

        val snapshot = assertNotNull(latest).let { latest!! }
        assertTrue("snapshot should be ready after eight completed cycles", snapshot.ready)
        assertEquals(8, snapshot.cycles)
        assertEquals(intervals.sum(), snapshot.durationMs)
        assertEquals(9, snapshot.boundaryTimesMs.size)
        assertEquals(snapshot.timesMs.size, snapshot.values.size)
        assertTrue(snapshot.values.isNotEmpty())
        assertTrue("main positive peak missing", snapshot.values.maxOrNull()!! >= 0.90f)
        assertTrue("negative trough was not softly limited", snapshot.values.minOrNull()!! >= -0.80f)
        assertTrue("template consistency too low: ${snapshot.quality}", (snapshot.quality ?: 0f) >= 0.75f)
        assertTrue("secondary peak became dominant", snapshot.secondaryPeakRatio <= 0.45f)
        assertTrue("all cycles became identical", snapshot.cycleVariationRms > 0.001f)
        assertTrue("cycle variation is too large", snapshot.cycleVariationRms < 0.15f)

        val exportedIntervals = snapshot.boundaryTimesMs.zipWithNext { first, second -> second - first }
        assertEquals(intervals, exportedIntervals)

        val priorTimes = snapshot.timesMs.copyOf()
        val invalidPeakMs = boundaries.last() + 300L
        var held: MorphologyWaveformSnapshot? = null
        for (timeMs in boundaries.last() + 20L..invalidPeakMs step 20L) {
            builder.next(
                value = 0f,
                timeMs = timeMs,
                detectedPeakTimeMs = timeMs.takeIf { it == invalidPeakMs }
            )?.let { held = it }
        }
        assertTrue("one invalid cycle should retain the last ready display", held?.ready == true)
        assertTrue(priorTimes.contentEquals(held!!.timesMs))
    }

    private fun cycleIndex(timeMs: Long, boundaries: List<Long>): Int {
        if (timeMs < boundaries.first()) return 0
        for (index in 0 until boundaries.lastIndex) {
            if (timeMs < boundaries[index + 1]) return index
        }
        return boundaries.lastIndex - 1
    }

    private fun cyclePhase(timeMs: Long, boundaries: List<Long>, intervals: List<Long>): Double {
        if (timeMs < boundaries.first()) {
            return (timeMs % 800L).toDouble() / 800.0
        }
        val index = cycleIndex(timeMs, boundaries)
        return ((timeMs - boundaries[index]).coerceAtLeast(0L) % intervals[index]).toDouble() /
            intervals[index].toDouble()
    }

    private fun square(value: Double): Double = value * value
}
