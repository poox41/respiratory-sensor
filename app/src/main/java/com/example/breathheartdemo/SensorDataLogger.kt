package com.example.breathheartdemo

import android.content.Context
import android.os.Environment
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SensorDataLogger(
    private val context: Context,
    private val acquisitionConfig: SensorAcquisitionConfig =
        SensorAcquisitionConfig.OFFLINE_INFERRED_PROFILE
) {

    private var writer: BufferedWriter? = null
    private var logFile: File? = null
    @Volatile
    var isLogging = false
        private set

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    @Synchronized
    fun startLogging(): File? {
        stopLogging()
        val dir = (context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: context.filesDir).resolve("sensor_logs")
        dir.mkdirs()
        val fileName = "sensor_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.csv"
        logFile = File(dir, fileName)
        writer = BufferedWriter(FileWriter(logFile!!, true))
        writer?.write(
            "time_ms,time_str,hardwareSampleRateHz,decodedValueRateHz," +
                "processingSampleRateHz,sampleRateEvidence,rawX,xDc,dc,xF,resp," +
                "hrRaw,hrF1,hrF,hrFilt,hrCalc,hrDisplay,heartCandidate,heartEnvelope," +
                "cleanHeart,cleanHeartEnvelope,trackedHeart,trackedHeartCenterHz," +
                "vmdHeartbeat,vmdMorphology," +
                "heartTemplateInput," +
                "heartTemplateEnhanced,heartTemplateQuality," +
                "heartTemplateCycles,heartTemplateReady," +
                "heartMorphologyEnhanced,heartMorphologyReady," +
                "heartMorphologyCycles,heartMorphologyQuality," +
                "heartMorphologyDelayMs,heartMorphologyBoundary," +
                "heartMorphologyReferenceBpm,heartMorphologyReferenceSource," +
                "heartMorphologyReferenceAgeMs,normHr," +
                "windowedPeak,windowedBpm,periodicBpm,periodicQuality," +
                "cleanPeriodicBpm,cleanPeriodicQuality,cleanPeak,cleanPeakBpm," +
                "thresholdPeak,dualPeakBpm,bpm,heartRateSource,heartRateStale," +
                "heartRatePrimaryCandidate,heartRateAlternateCandidate," +
                "heartRateAmbiguityReason,heartRateConfidence," +
                "vmdTrackingState,vmdComputeTimeMs,signalQuality," +
                "slowContactRawRange,slowContactRmsRatio,rpm,rpm_zc," +
                "respCycle,cycleRpm," +
                "presenceState,presenceScore,heartEnvelope10s,heartStd10s," +
                "sleepPredictedState,sleepProbability,sleepResultCode," +
                "sleepModelVersion,sleepPredictionTimeMs,sleepInputOutlierRatio," +
                "sleepInputImputedFraction,sleepInputMaxGapMs"
        )
        writer?.newLine()
        isLogging = true
        return logFile
    }

    @Synchronized
    fun log(
        tMs: Long,
        rawX: Float,
        xDc: Float,
        dc: Float,
        xF: Float,
        resp: Float,
        hrRaw: Float,
        hrF1: Float,
        hrF: Float,
        hrFilt: Float,
        hrCalc: Float,
        hrDisplay: Float,
        heartCandidate: Float,
        heartEnvelope: Float,
        cleanHeart: Float,
        cleanHeartEnvelope: Float,
        trackedHeart: Float,
        trackedHeartCenterHz: Float,
        vmdHeartbeat: Float?,
        vmdMorphology: Float?,
        heartTemplateInput: Float,
        heartTemplateEnhanced: Float,
        heartTemplateQuality: Float?,
        heartTemplateCycles: Int,
        heartTemplateReady: Boolean,
        heartMorphologyEnhanced: Float?,
        heartMorphologyReady: Boolean,
        heartMorphologyCycles: Int,
        heartMorphologyQuality: Float?,
        heartMorphologyDelayMs: Long,
        heartMorphologyBoundary: Boolean,
        heartMorphologyReferenceBpm: Float?,
        heartMorphologyReferenceSource: String,
        heartMorphologyReferenceAgeMs: Long?,
        normHr: Float,
        windowedPeak: Boolean,
        windowedBpm: Float?,
        periodicBpm: Float?,
        periodicQuality: Float?,
        cleanPeriodicBpm: Float?,
        cleanPeriodicQuality: Float?,
        cleanPeak: Boolean,
        cleanPeakBpm: Float?,
        thresholdPeak: Boolean,
        dualPeakBpm: Float?,
        bpm: Float? = null,
        heartRateSource: String? = null,
        heartRateStale: Boolean = false,
        heartRatePrimaryCandidate: Float? = null,
        heartRateAlternateCandidate: Float? = null,
        heartRateAmbiguityReason: String? = null,
        heartRateConfidence: Float? = null,
        vmdTrackingState: String? = null,
        vmdComputeTimeMs: Long? = null,
        signalQuality: String? = null,
        slowContactRawRange: Float? = null,
        slowContactRmsRatio: Float? = null,
        rpm: Float? = null,
        rpmZc: Float? = null,
        respCycle: Boolean,
        cycleRpm: Float?,
        presenceState: String,
        presenceScore: Float?,
        heartEnvelope10s: Float?,
        heartStd10s: Float?,
        sleepPredictedState: String?,
        sleepProbability: Float?,
        sleepResultCode: Int?,
        sleepModelVersion: String?,
        sleepPredictionTimeMs: Long?,
        sleepInputOutlierRatio: Float?,
        sleepInputImputedFraction: Float?,
        sleepInputMaxGapMs: Long?
    ) {
        if (!isLogging || writer == null) return
        try {
            writer?.apply {
                write(tMs.toString()); write(",")
                write(dateFormat.format(Date(tMs))); write(",")
                write(acquisitionConfig.hardwareSampleRateHz?.toString() ?: ""); write(",")
                write(acquisitionConfig.decodedValueRateHz.toString()); write(",")
                write(acquisitionConfig.processingSampleRateHz.toString()); write(",")
                write(acquisitionConfig.evidence); write(",")
                write(rawX.toString()); write(",")
                write(xDc.toString()); write(",")
                write(dc.toString()); write(",")
                write(xF.toString()); write(",")
                write(resp.toString()); write(",")
                write(hrRaw.toString()); write(",")
                write(hrF1.toString()); write(",")
                write(hrF.toString()); write(",")
                write(hrFilt.toString()); write(",")
                write(hrCalc.toString()); write(",")
                write(hrDisplay.toString()); write(",")
                write(heartCandidate.toString()); write(",")
                write(heartEnvelope.toString()); write(",")
                write(cleanHeart.toString()); write(",")
                write(cleanHeartEnvelope.toString()); write(",")
                write(trackedHeart.toString()); write(",")
                write(trackedHeartCenterHz.toString()); write(",")
                write(vmdHeartbeat?.toString() ?: ""); write(",")
                write(vmdMorphology?.toString() ?: ""); write(",")
                write(heartTemplateInput.toString()); write(",")
                write(heartTemplateEnhanced.toString()); write(",")
                write(heartTemplateQuality?.toString() ?: ""); write(",")
                write(heartTemplateCycles.toString()); write(",")
                write(if (heartTemplateReady) "1" else "0"); write(",")
                write(heartMorphologyEnhanced?.toString() ?: ""); write(",")
                write(if (heartMorphologyReady) "1" else "0"); write(",")
                write(heartMorphologyCycles.toString()); write(",")
                write(heartMorphologyQuality?.toString() ?: ""); write(",")
                write(heartMorphologyDelayMs.toString()); write(",")
                write(if (heartMorphologyBoundary) "1" else "0"); write(",")
                write(heartMorphologyReferenceBpm?.toString() ?: ""); write(",")
                write(heartMorphologyReferenceSource); write(",")
                write(heartMorphologyReferenceAgeMs?.toString() ?: ""); write(",")
                write(normHr.toString()); write(",")
                write(if (windowedPeak) "1" else "0"); write(",")
                write(windowedBpm?.toString() ?: ""); write(",")
                write(periodicBpm?.toString() ?: ""); write(",")
                write(periodicQuality?.toString() ?: ""); write(",")
                write(cleanPeriodicBpm?.toString() ?: ""); write(",")
                write(cleanPeriodicQuality?.toString() ?: ""); write(",")
                write(if (cleanPeak) "1" else "0"); write(",")
                write(cleanPeakBpm?.toString() ?: ""); write(",")
                write(if (thresholdPeak) "1" else "0"); write(",")
                write(dualPeakBpm?.toString() ?: ""); write(",")
                write(bpm?.toString() ?: ""); write(",")
                write(heartRateSource ?: ""); write(",")
                write(if (heartRateStale) "1" else "0"); write(",")
                write(heartRatePrimaryCandidate?.toString() ?: ""); write(",")
                write(heartRateAlternateCandidate?.toString() ?: ""); write(",")
                write(heartRateAmbiguityReason ?: ""); write(",")
                write(heartRateConfidence?.toString() ?: ""); write(",")
                write(vmdTrackingState ?: ""); write(",")
                write(vmdComputeTimeMs?.toString() ?: ""); write(",")
                write(signalQuality ?: ""); write(",")
                write(slowContactRawRange?.toString() ?: ""); write(",")
                write(slowContactRmsRatio?.toString() ?: ""); write(",")
                write(rpm?.toString() ?: ""); write(",")
                write(rpmZc?.toString() ?: ""); write(",")
                write(if (respCycle) "1" else "0"); write(",")
                write(cycleRpm?.toString() ?: ""); write(",")
                write(presenceState); write(",")
                write(presenceScore?.toString() ?: ""); write(",")
                write(heartEnvelope10s?.toString() ?: ""); write(",")
                write(heartStd10s?.toString() ?: ""); write(",")
                write(sleepPredictedState ?: ""); write(",")
                write(sleepProbability?.toString() ?: ""); write(",")
                write(sleepResultCode?.toString() ?: ""); write(",")
                write(sleepModelVersion ?: ""); write(",")
                write(sleepPredictionTimeMs?.toString() ?: ""); write(",")
                write(sleepInputOutlierRatio?.toString() ?: ""); write(",")
                write(sleepInputImputedFraction?.toString() ?: ""); write(",")
                write(sleepInputMaxGapMs?.toString() ?: "")
                newLine()
            }
        } catch (_: Exception) {
            isLogging = false
        }
    }

    @Synchronized
    fun flush() {
        try { writer?.flush() } catch (_: Exception) {}
    }

    @Synchronized
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
