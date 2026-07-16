package com.example.breathheartdemo

import org.junit.Assert.assertEquals
import org.junit.Test

class EnhancedHeartWindowTest {
    @Test
    fun usesTenSecondPaperStyleMultiCycleWindow() {
        assertEquals(10_000L, calculateEnhancedHeartWindowMs())
    }
}
