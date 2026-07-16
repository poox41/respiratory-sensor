package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SleepModelInputAdapterTest {
    private val fsHz = 50

    @Test
    fun reportsReadyOnlyAfterContinuousFiveAndHalfMinutes() {
        val samples = fsHz * 330
        val heart = RingBuffer(samples)
        val respiration = RingBuffer(samples)
        repeat(samples) { index ->
            val timeMs = index * 20L
            heart.add(timeMs, sin(2.0 * PI * index / 42.0).toFloat())
            respiration.add(timeMs, sin(2.0 * PI * index / 200.0).toFloat())
        }

        val adapter = SleepModelInputAdapter(fsHz)
        val progress = adapter.progress(heart, respiration)

        assertTrue(progress.ready)
        assertEquals(samples, progress.collectedSamples)
        assertTrue(adapter.prepare(heart, respiration) is SleepModelInputResult.Ready)
    }

    @Test
    fun tracksBriefGapWithoutRestartingElapsedProgress() {
        val heart = RingBuffer(2_000)
        val respiration = RingBuffer(2_000)
        repeat(1_000) { index ->
            val gapMs = if (index >= 800) 100L else 0L
            val timeMs = index * 20L + gapMs
            heart.add(timeMs, index.toFloat())
            respiration.add(timeMs, (index * 2).toFloat())
        }

        val progress = SleepModelInputAdapter(fsHz).progress(heart, respiration)

        assertFalse(progress.ready)
        assertFalse(progress.continuous)
        assertEquals(1_005, progress.collectedSamples)
        assertEquals(5, progress.imputedSamples)
        assertEquals(120L, progress.maxGapMs)
        assertTrue(progress.qualityAcceptable)
    }

    @Test
    fun reconstructsOneBriefMotionGapInsideReadyWindow() {
        val samples = fsHz * 330
        val heart = RingBuffer(samples)
        val respiration = RingBuffer(samples)
        repeat(samples) { index ->
            val motionGapMs = if (index >= 800) 3_000L else 0L
            val timeMs = index * 20L + motionGapMs
            heart.add(timeMs, sin(2.0 * PI * index / 42.0).toFloat())
            respiration.add(timeMs, sin(2.0 * PI * index / 200.0).toFloat())
        }

        val adapter = SleepModelInputAdapter(fsHz)
        val progress = adapter.progress(heart, respiration)
        val prepared = adapter.prepare(heart, respiration)

        assertTrue(progress.ready)
        assertTrue(progress.imputedSamples > 0)
        assertEquals(3_020L, progress.maxGapMs)
        assertTrue(prepared is SleepModelInputResult.Ready)
        assertTrue((prepared as SleepModelInputResult.Ready).window.imputedFraction < 0.05f)
    }

    @Test
    fun rejectsLongMotionGapButKeepsRollingCoverage() {
        val samples = fsHz * 330
        val heart = RingBuffer(samples)
        val respiration = RingBuffer(samples)
        repeat(samples) { index ->
            val motionGapMs = if (index >= 800) 5_000L else 0L
            val timeMs = index * 20L + motionGapMs
            heart.add(timeMs, sin(2.0 * PI * index / 42.0).toFloat())
            respiration.add(timeMs, sin(2.0 * PI * index / 200.0).toFloat())
        }

        val adapter = SleepModelInputAdapter(fsHz)
        val progress = adapter.progress(heart, respiration)
        val prepared = adapter.prepare(heart, respiration)

        assertEquals(samples, progress.collectedSamples)
        assertFalse(progress.ready)
        assertFalse(progress.qualityAcceptable)
        assertTrue(prepared is SleepModelInputResult.Invalid)
        assertEquals(1002, (prepared as SleepModelInputResult.Invalid).error.code)
    }

    @Test
    fun rejectsFlatSignalsEvenWhenWindowIsLongEnough() {
        val samples = fsHz * 330
        val heart = RingBuffer(samples)
        val respiration = RingBuffer(samples)
        repeat(samples) { index ->
            val timeMs = index * 20L
            heart.add(timeMs, 1f)
            respiration.add(timeMs, 2f)
        }

        val result = SleepModelInputAdapter(fsHz).prepare(heart, respiration)

        assertTrue(result is SleepModelInputResult.Invalid)
        assertEquals(1002, (result as SleepModelInputResult.Invalid).error.code)
    }
}
