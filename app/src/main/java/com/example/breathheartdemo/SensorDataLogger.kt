package com.example.breathheartdemo

import android.content.Context
import android.os.Environment
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SensorDataLogger(private val context: Context) {

    private var writer: BufferedWriter? = null
    private var logFile: File? = null
    var isLogging = false
        private set

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    fun startLogging(): File? {
        stopLogging()
        val dir = (context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: context.filesDir).resolve("sensor_logs")
        dir.mkdirs()
        val fileName = "sensor_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.csv"
        logFile = File(dir, fileName)
        writer = BufferedWriter(FileWriter(logFile!!, true))
        writer?.write("time_ms,time_str,rawX,x,dc,xF,resp,hr,normHr,bpm,rpm,rpm_zc")
        writer?.newLine()
        isLogging = true
        return logFile
    }

    fun log(tMs: Long, rawX: Float, x: Float, dc: Float, xF: Float, resp: Float, hr: Float, normHr: Float, bpm: Float? = null, rpm: Float? = null, rpmZc: Float? = null) {
        if (!isLogging || writer == null) return
        try {
            writer?.apply {
                write(tMs.toString()); write(",")
                write(dateFormat.format(Date(tMs))); write(",")
                write(rawX.toString()); write(",")
                write(x.toString()); write(",")
                write(dc.toString()); write(",")
                write(xF.toString()); write(",")
                write(resp.toString()); write(",")
                write(hr.toString()); write(",")
                write(normHr.toString()); write(",")
                write(bpm?.toString() ?: ""); write(",")
                write(rpm?.toString() ?: ""); write(",")
                write(rpmZc?.toString() ?: "")
                newLine()
            }
        } catch (_: Exception) {
            isLogging = false
        }
    }

    fun flush() {
        try { writer?.flush() } catch (_: Exception) {}
    }

    fun stopLogging(): File? {
        isLogging = false
        try {
            writer?.flush()
            writer?.close()
        } catch (_: Exception) {}
        writer = null
        val f = logFile
        logFile = null
        return f
    }
}
