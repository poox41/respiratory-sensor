package com.example.breathheartdemo

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.exp
import kotlin.system.measureTimeMillis

data class SleepStateResult(
    val code: Int,
    val state: String,
    val stateName: String,
    val confidence: Float,
    val modelVersion: String?,
    val message: String,
    val timestampMs: Long?,
    val inferenceTimeMs: Long? = null,
    val inputOutlierRatio: Float? = null,
    val inputImputedFraction: Float? = null,
    val inputMaxGapMs: Long? = null
)

data class SleepModelHealth(
    val ready: Boolean,
    val code: Int,
    val modelVersion: String,
    val message: String,
    val inputShape: List<Long> = emptyList(),
    val outputShape: List<Long> = emptyList(),
    val featureParityValidated: Boolean = false
)

class SleepStateService(
    private val context: Context,
    private val fsHz: Int,
    private val windowSeconds: Int = 330
) {
    private val extractor = SleepFeatureExtractor()
    private val inputAdapter = SleepModelInputAdapter(fsHz = fsHz, windowSeconds = windowSeconds)
    private val config: SleepModelConfig by lazy { loadConfig() }
    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val session: OrtSession by lazy {
        env.createSession(context.assets.open(MODEL_ASSET).readBytes(), OrtSession.SessionOptions())
    }

    fun inputProgress(processor: Processor): SleepModelInputProgress = inputAdapter.progress(
        heartBuffer = processor.cleanHeartBuf,
        respirationBuffer = processor.respBuf
    )

    suspend fun checkModel(): SleepModelHealth = withContext(Dispatchers.Default) {
        validateModelContract()
    }

    suspend fun predict(processor: Processor, @Suppress("UNUSED_PARAMETER") rates: Rates): SleepStateResult =
        withContext(Dispatchers.Default) {
            predictBlocking(processor)
        }

    private fun predictBlocking(processor: Processor): SleepStateResult {
        val modelHealth = validateModelContract()
        if (!modelHealth.ready) {
            return unknown(
                code = modelHealth.code,
                message = modelHealth.message,
                timestampMs = null
            )
        }
        if (windowSeconds != config.seqLen * EPOCH_SECONDS) {
            return unknown(
                code = 1005,
                message = "model window mismatch: ${config.seqLen} x ${EPOCH_SECONDS}s",
                timestampMs = null
            )
        }
        val modelInput = when (val prepared = inputAdapter.prepare(
            heartBuffer = processor.cleanHeartBuf,
            respirationBuffer = processor.respBuf
        )) {
            is SleepModelInputResult.Ready -> prepared.window
            is SleepModelInputResult.Invalid -> {
                return unknown(
                    code = prepared.error.code,
                    message = prepared.error.message,
                    timestampMs = prepared.error.timestampMs
                )
            }
        }

        return try {
            var prediction: SleepStateResult? = null
            val elapsedMs = measureTimeMillis {
                val features = buildFeatureTensor(modelInput.heart, modelInput.respiration)
                val normalized = normalize(features, config)
                val inputOutlierRatio = normalized.count { kotlin.math.abs(it) > 10f }
                    .toFloat() / normalized.size.coerceAtLeast(1)
                val output = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(normalized),
                    longArrayOf(1L, config.seqLen.toLong(), config.featureDim.toLong())
                ).use { tensor ->
                    session.run(mapOf(INPUT_NAME to tensor)).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        (result[0].value as Array<FloatArray>)[0]
                    }
                }

                val probSleep = softmaxSleep(output)
                val isSleep = probSleep >= config.threshold
                prediction = SleepStateResult(
                    code = 0,
                    state = if (isSleep) "sleep" else "awake",
                    stateName = if (isSleep) "Sleep" else "Awake",
                    confidence = probSleep,
                    modelVersion = MODEL_VERSION,
                    message = if (inputOutlierRatio >= INPUT_OUTLIER_WARNING_RATIO) {
                        "experimental output; input distribution differs from training data"
                    } else {
                        "experimental output: cleanHeart + resp, 11 x 30s"
                    },
                    timestampMs = modelInput.timestampMs,
                    inferenceTimeMs = null,
                    inputOutlierRatio = inputOutlierRatio,
                    inputImputedFraction = modelInput.imputedFraction,
                    inputMaxGapMs = modelInput.maxGapMs
                )
            }
            prediction!!.copy(inferenceTimeMs = elapsedMs)
        } catch (e: Exception) {
            unknown(
                code = 1005,
                message = e.message ?: "model error",
                timestampMs = modelInput.timestampMs
            )
        }
    }

    private fun validateModelContract(): SleepModelHealth {
        return try {
            val cfg = config
            require(cfg.seqLen == EXPECTED_SEQ_LEN) {
                "config seq_len=${cfg.seqLen}, expected $EXPECTED_SEQ_LEN"
            }
            require(cfg.featureDim == SleepFeatureExtractor.FEATURE_DIM) {
                "config feature_dim=${cfg.featureDim}, expected ${SleepFeatureExtractor.FEATURE_DIM}"
            }
            require(cfg.mean.size == cfg.featureDim && cfg.std.size == cfg.featureDim) {
                "normalization vector length does not match feature_dim"
            }
            require(cfg.featureCols == SleepFeatureExtractor.FEATURE_NAMES) {
                "feature order does not match the Android extractor"
            }

            val inputInfo = session.inputInfo[INPUT_NAME]?.info as? TensorInfo
                ?: error("model input '$INPUT_NAME' is missing or is not a tensor")
            val outputInfo = session.outputInfo[OUTPUT_NAME]?.info as? TensorInfo
                ?: error("model output '$OUTPUT_NAME' is missing or is not a tensor")
            val inputShape = inputInfo.shape.toList()
            val outputShape = outputInfo.shape.toList()
            require(matchesShape(inputShape, listOf(1L, cfg.seqLen.toLong(), cfg.featureDim.toLong()))) {
                "model input shape $inputShape does not match [1, ${cfg.seqLen}, ${cfg.featureDim}]"
            }
            require(matchesShape(outputShape, listOf(1L, 2L))) {
                "model output shape $outputShape does not match [1, 2]"
            }

            SleepModelHealth(
                ready = true,
                code = 0,
                modelVersion = MODEL_VERSION,
                message = "ONNX/config contract verified; real-sensor feature parity still requires validation",
                inputShape = inputShape,
                outputShape = outputShape,
                featureParityValidated = false
            )
        } catch (e: Exception) {
            SleepModelHealth(
                ready = false,
                code = 1004,
                modelVersion = MODEL_VERSION,
                message = e.message ?: "model initialization failed",
                featureParityValidated = false
            )
        }
    }

    private fun matchesShape(actual: List<Long>, expected: List<Long>): Boolean {
        if (actual.size != expected.size) return false
        return actual.indices.all { index -> actual[index] <= 0L || actual[index] == expected[index] }
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

    private fun loadConfig(): SleepModelConfig {
        val json = JSONObject(context.assets.open(CONFIG_ASSET).bufferedReader().use { it.readText() })
        val meanArray = json.getJSONArray("mean")
        val stdArray = json.getJSONArray("std")
        val featureColsArray = json.getJSONArray("feature_cols")
        val featureDim = json.getInt("feature_dim")
        return SleepModelConfig(
            mean = FloatArray(featureDim) { meanArray.getDouble(it).toFloat() },
            std = FloatArray(featureDim) { stdArray.getDouble(it).toFloat() },
            threshold = json.getDouble("threshold").toFloat(),
            seqLen = json.getInt("seq_len"),
            featureDim = featureDim,
            featureCols = List(featureColsArray.length()) { featureColsArray.getString(it) }
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
            stateName = "Undetermined",
            confidence = 0f,
            modelVersion = null,
            message = message,
            timestampMs = timestampMs
        )
    }
}

private data class SleepModelConfig(
    val mean: FloatArray,
    val std: FloatArray,
    val threshold: Float,
    val seqLen: Int,
    val featureDim: Int,
    val featureCols: List<String>
)

private const val MODEL_ASSET = "sleep_bcg_v1.onnx"
private const val CONFIG_ASSET = "sleep_bcg_v1.json"
private const val MODEL_VERSION = "sleep_bcg_v1.0"
private const val INPUT_NAME = "input"
private const val OUTPUT_NAME = "output"
private const val EPOCH_SECONDS = 30
private const val EXPECTED_SEQ_LEN = 11
private const val INPUT_OUTLIER_WARNING_RATIO = 0.05f
