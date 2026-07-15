package com.example.breathheartdemo

import org.junit.Assert.assertEquals
import org.junit.Test

class EnhancedHeartWindowTest {
    @Test
    fun defaultsToFourSecondsBeforeEnoughPeaksArrive() {
        assertEquals(4_000L, calculateEnhancedHeartWindowMs(emptyList()))
        assertEquals(4_000L, calculateEnhancedHeartWindowMs(listOf(1_000L)))
    }

    @Test
    fun usesFourMedianHeartbeatIntervals() {
        val peaks = listOf(0L, 760L, 1_540L, 2_300L, 3_060L)
        assertEquals(3_040L, calculateEnhancedHeartWindowMs(peaks))
    }

    @Test
    fun ignoresImplausibleIntervals() {
        val peaks = listOf(0L, 100L, 900L, 1_700L, 4_500L, 5_300L)
        assertEquals(3_200L, calculateEnhancedHeartWindowMs(peaks))
    }
}
