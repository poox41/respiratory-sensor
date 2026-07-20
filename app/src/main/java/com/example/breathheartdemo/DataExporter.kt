package com.example.breathheartdemo

import android.content.Context
import android.os.Environment
import android.os.SystemClock
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class ExportedSample(
    val receiveTimeMs: Long,
    val sampleTimeMs: Long,
    val value: Int,
    val lo: Int,
    val hi: Int
)

class DataExporter(
    context: Context
) {
    private val sessionFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
    private val readableTimeFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private val rootDir: File =
        (context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir)
            .resolve("breathheart_exports")

    private var sessionDir: File? = null
    private var currentFile: File? = null
    private var rawLogFile: File? = null
    private var csvWriter: BufferedWriter? = null
    private var rawLogWriter: BufferedWriter? = null
    private var sampleIndex = 0L
    private var lastFlushElapsedMs = 0L

    fun startSession(device: BleDevice?): File {
        close()
        rootDir.mkdirs()
        val startedAt = System.currentTimeMillis()
        val safeDevice = sanitize(device?.address ?: "unknown")
        val safeName = sanitize(device?.name ?: "unnamed")
        val dir = rootDir.resolve(
            "session_${sessionFormat.format(Date(startedAt))}_${safeName}_$safeDevice"
        )
        dir.mkdirs()
        dir.resolve("session_info.txt").writeText(
            buildString {
                appendLine("started_at_ms=$startedAt")
                appendLine("started_at=${Date(startedAt)}")
                appendLine("device_name=${device?.name ?: ""}")
                appendLine("device_address=${device?.address ?: ""}")
            }
        )
        sessionDir = dir
        sampleIndex = 0L
        openSessionFile(dir, startedAt)
        return dir
    }

    fun appendSamples(samples: List<ExportedSample>) {
        if (samples.isEmpty()) return
        if (sessionDir == null || csvWriter == null) return
        for (sample in samples) {
            sampleIndex++
            csvWriter?.apply {
                write(sampleIndex.toString())
                write(",")
                write(readableTimeFormat.format(Date(sample.receiveTimeMs)))
                write(",")
                write(sample.receiveTimeMs.toString())
                write(",")
                write(readableTimeFormat.format(Date(sample.sampleTimeMs)))
                write(",")
                write(sample.sampleTimeMs.toString())
                write(",")
                write(sample.value.toString())
                write(",")
                write(sample.lo.toString())
                write(",")
                write(sample.hi.toString())
                newLine()
            }
        }
        flushIfDue()
    }

    fun appendRawPacket(receiveTimeMs: Long, hex: String, byteCount: Int) {
        if (sessionDir == null || rawLogWriter == null) return
        rawLogWriter?.apply {
            write("[${readableTimeFormat.format(Date(receiveTimeMs))}]")
            write(" receive_time_ms=")
            write(receiveTimeMs.toString())
            write(" bytes=")
            write(byteCount.toString())
            write(" hex=")
            write(hex)
            newLine()
        }
        flushIfDue()
    }

    fun currentSessionPath(): String? = sessionDir?.absolutePath

    fun currentFileName(): String? = currentFile?.name

    fun close() {
        csvWriter?.flush()
        csvWriter?.close()
        csvWriter = null
        rawLogWriter?.flush()
        rawLogWriter?.close()
        rawLogWriter = null
        currentFile = null
        rawLogFile = null
        sessionDir = null
        lastFlushElapsedMs = 0L
    }

    private fun openSessionFile(dir: File, startedAt: Long) {
        val file = dir.resolve(
            "samples_${sessionFormat.format(Date(startedAt))}.csv"
        )
        currentFile = file
        csvWriter = BufferedWriter(FileWriter(file, true)).apply {
            write("序号,接收时间,接收时间戳ms,采样时间,采样时间戳ms,采样值,原始低字节,原始高字节")
            newLine()
        }
        rawLogFile = dir.resolve(
            "raw_hex_${sessionFormat.format(Date(startedAt))}.txt"
        )
        rawLogWriter = BufferedWriter(FileWriter(rawLogFile!!, true)).apply {
            write("# 原始蓝牙HEX日志")
            newLine()
            write("# 格式: [接收时间] receive_time_ms=... bytes=... hex=...")
            newLine()
        }
        lastFlushElapsedMs = SystemClock.elapsedRealtime()
    }

    /** Batch disk flushes so BLE callbacks are not blocked on every packet. */
    private fun flushIfDue() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastFlushElapsedMs < FLUSH_INTERVAL_MS) return
        csvWriter?.flush()
        rawLogWriter?.flush()
        lastFlushElapsedMs = now
    }

    private fun sanitize(value: String): String {
        return value.replace(Regex("[^A-Za-z0-9._-]"), "_")
    }
}

private const val FLUSH_INTERVAL_MS = 1_000L
