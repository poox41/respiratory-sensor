package com.example.breathheartdemo

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random

class MockDataSource(
    private val fsHz: Int = 50,
    private val startBpm: Float = 72f,
    private val startRpm: Float = 16f
) {
    fun samples(): Flow<Sample> = flow {
        val dtMs = (1000.0 / fsHz).toLong()
        val t0 = System.currentTimeMillis()
        var n = 0L

        while (true) {
            val tMs = t0 + n * dtMs
            val sec = (n.toFloat() / fsHz)
            val fHr = startBpm / 60f
            val fResp = startRpm / 60f
            // ADC: 2048=0V, breathing ~800pp, HR ~200pp
            val x = 2048f +
                800f * sin(2f * PI.toFloat() * fResp * sec) +
                200f * sin(2f * PI.toFloat() * fHr * sec) +
                Random.nextFloat() * 10f - 5f

            emit(Sample(tMs, x))
            n++
            delay(dtMs)
        }
    }
}
