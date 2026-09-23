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
    private var inputTimingWriter: BufferedWriter? = null
    private var configuredProcessorSampleRateHz = 50
    private var configuredHardwareSampleRateHz: Int? = 50
    private var configuredDecodedValueRateHz = 100
    private var configuredDecodedValuesPerProcessorSample = 2
    private var configuredRateEvidence = "unknown"
    private var configuredInputResamplingMode = "unknown"
    private var packetIndex = 0L
    private var sampleIndex = 0L
    private var lastFlushElapsedMs = 0L

    fun startSession(
        device: BleDevice?,
        acquisitionConfig: SensorAcquisitionConfig
    ): File {
        close()
        configuredProcessorSampleRateHz = acquisitionConfig.processingSampleRateHz
        configuredHardwareSampleRateHz = acquisitionConfig.hardwareSampleRateHz
        configuredDecodedValueRateHz = acquisitionConfig.decodedValueRateHz
        configuredDecodedValuesPerProcessorSample =
            acquisitionConfig.decodedValuesPerProcessorSample
        configuredRateEvidence = acquisitionConfig.evidence
        configuredInputResamplingMode = acquisitionConfig.inputResamplingMode
        packetIndex = 0L
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
                appendLine("configured_hardware_sample_rate_hz=${acquisitionConfig.hardwareSampleRateHz ?: "unknown"}")
                appendLine("sample_rate_evidence=${acquisitionConfig.evidence}")
                appendLine("processor_sample_rate_hz=${acquisitionConfig.processingSampleRateHz}")
                appendLine("decoded_values_per_processor_sample=${acquisitionConfig.decodedValuesPerProcessorSample}")
                appendLine("expected_decoded_value_rate_hz=${acquisitionConfig.decodedValueRateHz}")
                appendLine("input_mode=${acquisitionConfig.inputResamplingMode}")
                appendLine("samples_csv_mode=every_decoded_wire_value_before_pair_averaging")
                appendLine("decoder=little_endian_int16_stream")
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

    fun appendInputTiming(
        receiveTimeMs: Long,
        receiveElapsedMs: Long,
        payloadBytes: Int,
        decodedValues: Int,
        totalDecodedValues: Long,
        expectedDecodedValueRateHz: Int,
        observedValuesPerSecond: Double?,
        sampleClockLeadMs: Long?,
        rateStatus: String,
        processorSamplesProduced: Long,
        failedSampleEmissions: Long,
        callbackApi: String
    ) {
        val writer = inputTimingWriter ?: return
        packetIndex++
        writer.write(listOf(
            packetIndex, receiveTimeMs, receiveElapsedMs, payloadBytes, decodedValues,
            totalDecodedValues, configuredHardwareSampleRateHz ?: "",
            configuredRateEvidence, configuredProcessorSampleRateHz,
            configuredDecodedValuesPerProcessorSample, expectedDecodedValueRateHz,
            configuredInputResamplingMode,
            observedValuesPerSecond ?: "", sampleClockLeadMs ?: "", rateStatus,
            processorSamplesProduced, failedSampleEmissions, callbackApi
        ).joinToString(","))
        writer.newLine()
        flushIfDue()
    }

    fun currentFileName(): String? = currentFile?.name

    fun close() {
        csvWriter?.flush()
        csvWriter?.close()
        csvWriter = null
        rawLogWriter?.flush()
        rawLogWriter?.close()
        rawLogWriter = null
        inputTimingWriter?.flush()
        inputTimingWriter?.close()
        inputTimingWriter = null
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
        val timingFile = dir.resolve("input_timing_${sessionFormat.format(Date(startedAt))}.csv")
        inputTimingWriter = BufferedWriter(FileWriter(timingFile, true)).apply {
            write("packet_index,receive_time_ms,receive_elapsed_ms,payload_bytes,decoded_values," +
                "total_decoded_values,configured_hardware_sample_rate_hz,sample_rate_evidence," +
                "configured_processor_sample_rate_hz," +
                "decoded_values_per_processor_sample,expected_decoded_value_rate_hz," +
                "input_resampling_mode," +
                "observed_values_per_second,sample_clock_lead_ms,rate_status," +
                "processor_samples_produced,failed_sample_emissions,callback_api")
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
        inputTimingWriter?.flush()
        lastFlushElapsedMs = now
    }

    private fun sanitize(value: String): String {
        return value.replace(Regex("[^A-Za-z0-9._-]"), "_")
    }
}

private const val FLUSH_INTERVAL_MS = 1_000L
