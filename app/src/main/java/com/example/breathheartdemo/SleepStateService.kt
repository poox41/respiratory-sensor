package com.example.breathheartdemo

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.exp

data class SleepStateResult(
    val code: Int,
    val state: String,
    val stateName: String,
    val confidence: Float,
    val modelVersion: String?,
    val message: String,
    val timestampMs: Long?
)

class SleepStateService(
    private val context: Context,
    private val fsHz: Int,
    private val windowSeconds: Int = 330
) {
    private val extractor = SleepFeatureExtractor()
    private val config: SleepModelConfig by lazy { loadConfig() }
    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val session: OrtSession by lazy {
        env.createSession(context.assets.open(MODEL_ASSET).readBytes(), OrtSession.SessionOptions())
    }

    fun predict(processor: Processor, rates: Rates): SleepStateResult {
        val raw = recent(processor.rawBuf, windowSeconds)
        val respiration = recent(processor.respBuf, windowSeconds)
        val heart = recent(processor.hrBuf, windowSeconds)
        val timestampMs = raw.timestamps.lastOrNull()

        val required = fsHz * windowSeconds
        val available = minOf(raw.values.size, respiration.values.size, heart.values.size)
        if (available < required) {
            val collectedSeconds = available / max(1f, fsHz.toFloat())
            return unknown(
                code = 1001,
                message = "data not enough: %.1fs/%.0fs collected".format(
                    collectedSeconds,
                    windowSeconds.toFloat()
                ),
                timestampMs = timestampMs
            )
        }

        return try {
            val features = buildFeatureTensor(heart.values, respiration.values)
            val normalized = normalize(features, config)
            val tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(normalized),
                longArrayOf(1L, config.seqLen.toLong(), config.featureDim.toLong())
            )
            val output = session.run(mapOf(INPUT_NAME to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                (result[0].value as Array<FloatArray>)[0]
            }
            tensor.close()

            val probSleep = softmaxSleep(output)
            val isSleep = probSleep >= config.threshold
            SleepStateResult(
                code = 0,
                state = if (isSleep) "sleep" else "awake",
                stateName = if (isSleep) "睡眠" else "清醒",
                confidence = probSleep,
                modelVersion = MODEL_VERSION,
                message = "success",
                timestampMs = timestampMs
            )
        } catch (e: Exception) {
            unknown(
                code = 1005,
                message = e.message ?: "model error",
                timestampMs = timestampMs
            )
        }
    }

    private fun buildFeatureTensor(heart: FloatArray, respiration: FloatArray): FloatArray {
        val cfg = config
        val epochSamples = fsHz * EPOCH_SECONDS
        val features = FloatArray(cfg.seqLen * cfg.featureDim)
        val start = heart.size - cfg.seqLen * epochSamples
        for (epoch in 0 until cfg.seqLen) {
            val from = start + epoch * epochSamples
            val to = from + epochSamples
            val epochFeatures = extractor.extractEpoch(
                heart50Hz = heart.copyOfRange(from, to),
                respiration50Hz = respiration.copyOfRange(from, to)
            )
            System.arraycopy(epochFeatures, 0, features, epoch * cfg.featureDim, cfg.featureDim)
        }
        return features
    }

    private fun normalize(features: FloatArray, cfg: SleepModelConfig): FloatArray {
        val out = FloatArray(features.size)
        for (i in features.indices) {
            val j = i % cfg.featureDim
            val std = if (cfg.std[j] == 0f) 1f else cfg.std[j]
            val value = (features[i] - cfg.mean[j]) / std
            out[i] = if (value.isFinite()) value else 0f
        }
        return out
    }

    private fun recent(buffer: RingBuffer, seconds: Int): SignalWindow {
        val (timestamps, values) = buffer.snapshot()
        if (values.isEmpty()) return SignalWindow(LongArray(0), FloatArray(0))

        val latest = timestamps.last()
        val minTimestamp = latest - seconds * 1000L
        var start = 0
        while (start < timestamps.size && timestamps[start] < minTimestamp) {
            start++
        }

        return SignalWindow(
            timestamps = timestamps.copyOfRange(start, timestamps.size),
            values = values.copyOfRange(start, values.size)
        )
    }

    private fun loadConfig(): SleepModelConfig {
        val json = JSONObject(context.assets.open(CONFIG_ASSET).bufferedReader().use { it.readText() })
        val meanArray = json.getJSONArray("mean")
        val stdArray = json.getJSONArray("std")
        val featureDim = json.getInt("feature_dim")
        return SleepModelConfig(
            mean = FloatArray(featureDim) { meanArray.getDouble(it).toFloat() },
            std = FloatArray(featureDim) { stdArray.getDouble(it).toFloat() },
            threshold = json.getDouble("threshold").toFloat(),
            seqLen = json.getInt("seq_len"),
            featureDim = featureDim
        )
    }

    private fun softmaxSleep(logits: FloatArray): Float {
        if (logits.size < 2) return 0f
        val maxLogit = max(logits[0], logits[1])
        val awake = exp((logits[0] - maxLogit).toDouble()).toFloat()
        val sleep = exp((logits[1] - maxLogit).toDouble()).toFloat()
        return sleep / (awake + sleep)
    }

    private fun unknown(code: Int, message: String, timestampMs: Long?): SleepStateResult {
        return SleepStateResult(
            code = code,
            state = "unknown",
            stateName = "无法判断",
            confidence = 0f,
            modelVersion = null,
            message = message,
            timestampMs = timestampMs
        )
    }
}

private data class SignalWindow(
    val timestamps: LongArray,
    val values: FloatArray
)

private data class SleepModelConfig(
    val mean: FloatArray,
    val std: FloatArray,
    val threshold: Float,
    val seqLen: Int,
    val featureDim: Int
)

private const val MODEL_ASSET = "sleep_bcg_v1.onnx"
private const val CONFIG_ASSET = "sleep_bcg_v1.json"
private const val MODEL_VERSION = "sleep_bcg_v1.0"
private const val INPUT_NAME = "input"
private const val EPOCH_SECONDS = 30
