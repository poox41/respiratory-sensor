package com.example.breathheartdemo

data class Sample(val tMs: Long, val x: Float)

data class Rates(
    val bpm: Float? = null,
    val rpm: Float? = null
)