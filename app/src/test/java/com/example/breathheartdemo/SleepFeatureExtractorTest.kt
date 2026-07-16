package com.example.breathheartdemo

import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SleepFeatureExtractorTest {
    @Test
    fun emitsFiniteFeaturesInTheDeclaredModelOrder() {
        val samples = 50 * 30
        val heart = FloatArray(samples) { index ->
            sin(2.0 * PI * 1.2 * index / 50.0).toFloat()
        }
        val respiration = FloatArray(samples) { index ->
            sin(2.0 * PI * 0.25 * index / 50.0).toFloat()
        }

        val features = SleepFeatureExtractor().extractEpoch(heart, respiration)

        assertEquals(SleepFeatureExtractor.FEATURE_DIM, SleepFeatureExtractor.FEATURE_NAMES.size)
        assertEquals(SleepFeatureExtractor.FEATURE_DIM, features.size)
        assertTrue(features.all { it.isFinite() })
        assertTrue("heart features were not populated", features[12] > 0f)
        assertTrue("respiration features were not populated", features[30] > 0f)
    }
}
