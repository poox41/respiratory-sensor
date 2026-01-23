package com.example.breathheartdemo

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random
//模拟混合通道数据
class MockDataSource(
    private val fsHz: Int = 100,
    private val startBpm: Float = 75f,
    private val startRpm: Float = 15f
) {
    fun samples(): Flow<Sample> = flow {
        val dtMs = (1000.0 / fsHz).toLong()
        val t0 = System.currentTimeMillis()
        var n = 0L

        fun bpmAt(sec: Float) = startBpm
        fun rpmAt(sec: Float) = startRpm

        while (true) {
            val tMs = t0 + n * dtMs
            val sec = (n.toFloat() / fsHz)

            val bpm = bpmAt(sec)
            val rpm = rpmAt(sec)

            val fHr = bpm / 60f
            val fResp = rpm / 60f

            val resp = 1.0f * sin(2f * PI.toFloat() * fResp * sec)
            val hr = 0.6f * sin(2f * PI.toFloat() * fHr * sec)
            val x = resp + hr

            emit(Sample(tMs, x))
            n++
            delay(dtMs)
        }
    }
}
