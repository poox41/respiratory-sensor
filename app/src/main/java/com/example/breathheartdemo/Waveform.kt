package com.example.breathheartdemo

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import kotlin.math.max
import kotlin.math.min

// Draw the waveform.
@Composable
fun rememberWaveformFrame(refreshMs: Long = 50L): State<Long> {
    val tick = remember { mutableLongStateOf(0L) }
    LaunchedEffect(refreshMs) {
        while (true) {
            delay(refreshMs.coerceAtLeast(16L))
            tick.longValue++
        }
    }
    return tick
}

@Composable
fun Waveform(
    modifier: Modifier = Modifier,
    buffer: RingBuffer,
    color: Color = Color.Black,
    yMin: Float = -2f,
    yMax: Float = 2f,
    windowMs: Long = 6000L,
    gain: Float = 1f,
    windowMsProvider: (() -> Long)? = null,
    gainProvider: (() -> Float)? = null,
    offset: Float = 0f,
    showGrid: Boolean = true,
    showZeroLine: Boolean = true,
    zeroLineValue: Float? = null,
    peakTimes: List<Long> = emptyList(),
    peakTimesProvider: (() -> List<Long>)? = null,
    headroomFraction: Float = 1f,
    frameTick: State<Long>
) {
    key(frameTick.value) {
        Canvas(modifier = modifier.fillMaxWidth().height(140.dp)) {
            val effectiveWindowMs = (windowMsProvider?.invoke() ?: windowMs).coerceAtLeast(1L)
            val requestedGain = gainProvider?.invoke() ?: gain
            val effectivePeakTimes = peakTimesProvider?.invoke() ?: peakTimes
            val vMin = yMin
            val vMax = yMax
            val range = max(1e-6f, vMax - vMin)

            val w = size.width
            val h = size.height
            val path = Path()
            val minStepPx = 0.5f
            var lastX = Float.NEGATIVE_INFINITY
            var lastTimeMs = Long.MIN_VALUE
            var started = false

            if (showGrid) {
                val gridColor = color.copy(alpha = 0.12f)
                val vLines = 6
                val hLines = 4
                for (i in 1 until vLines) {
                    val x = w * i / vLines
                    drawLine(
                        gridColor,
                        start = androidx.compose.ui.geometry.Offset(x, 0f),
                        end = androidx.compose.ui.geometry.Offset(x, h),
                        strokeWidth = 1f
                    )
                }
                for (i in 1 until hLines) {
                    val y = h * i / hLines
                    drawLine(
                        gridColor,
                        start = androidx.compose.ui.geometry.Offset(0f, y),
                        end = androidx.compose.ui.geometry.Offset(w, y),
                        strokeWidth = 1f
                    )
                }
            }

            if (showZeroLine) {
                val zeroValue = zeroLineValue ?: 0f
                if (zeroValue >= vMin && zeroValue <= vMax) {
                    val y0 = h - ((zeroValue - vMin) / range) * h
                    drawLine(
                        color = Color.Black.copy(alpha = 0.4f),
                        start = androidx.compose.ui.geometry.Offset(0f, y0),
                        end = androidx.compose.ui.geometry.Offset(w, y0),
                        strokeWidth = 1.5f
                    )
                }
            }

            val (ts, vs) = buffer.snapshotWindow(effectiveWindowMs)
            if (vs.isEmpty()) return@Canvas

            val tMax = ts[vs.lastIndex]
            val tMin = tMax - effectiveWindowMs

            // Binary search for first visible sample
            var lo = 0
            var hi = vs.size - 1
            while (lo < hi) {
                val mid = (lo + hi) / 2
                if (ts[mid] < tMin) lo = mid + 1
                else hi = mid
            }
            if (ts[lo] < tMin) return@Canvas

            // The auto-gain coroutine samples the buffer every 250 ms, while
            // drawing happens more frequently.  Limit the requested gain again
            // using the samples visible in this exact frame so a newly arriving
            // large beat cannot be clamped into a flat top or bottom.
            val effectiveGain = displayGainWithHeadroom(
                values = vs,
                startIndex = lo,
                requestedGain = requestedGain,
                offset = offset,
                yMin = vMin,
                yMax = vMax,
                headroomFraction = headroomFraction
            )

            for (i in lo until vs.size) {
                val t = ts[i]
                val xNorm = (t - tMin).toFloat() / effectiveWindowMs.toFloat()
                val x = xNorm.coerceIn(0f, 1f) * w
                // Defensive segmentation: a source clock discontinuity must
                // never be rendered as a long diagonal or a vertical step.
                val discontinuity = lastTimeMs != Long.MIN_VALUE &&
                    (t <= lastTimeMs || t - lastTimeMs > 250L)
                lastTimeMs = t
                if (discontinuity) lastX = Float.NEGATIVE_INFINITY
                if (x - lastX < minStepPx) continue
                lastX = x
                val scaled = (vs[i] - offset) * effectiveGain + offset
//            val clamped = min(vMax, max(vMin, scaled))
                val yNorm = ((scaled - vMin) / range).coerceIn(0f, 1f)
                val y = h - yNorm * h
                if (!started || discontinuity) {
                    path.moveTo(x, y)
                    started = true
                } else {
                    path.lineTo(x, y)
                }
            }

            if (started) {
                drawPath(
                    path = path,
                    color = color,
                    alpha = 0.9f,
                    style = Stroke(width = 1.5f, cap = StrokeCap.Round, join = StrokeJoin.Round)
                )
            }

            // Draw dashed rectangles between consecutive peaks (marking complete heart cycles)
            if (effectivePeakTimes.size >= 2) {
                val dash = PathEffect.dashPathEffect(floatArrayOf(4f, 3f))
                val rectColor = Color.Red.copy(alpha = 0.35f)
                for (i in 0 until effectivePeakTimes.size - 1) {
                    val p1 = effectivePeakTimes[i]
                    val p2 = effectivePeakTimes[i + 1]
                    if (p1 >= tMin) {
                        val x1 = ((p1 - tMin).toFloat() / effectiveWindowMs.toFloat()).coerceIn(0f, 1f) * w
                        val x2 = ((p2 - tMin).toFloat() / effectiveWindowMs.toFloat()).coerceIn(0f, 1f) * w
                        val rectPath = Path()
                        rectPath.moveTo(x1, 0f)
                        rectPath.lineTo(x2, 0f)
                        rectPath.lineTo(x2, h)
                        rectPath.lineTo(x1, h)
                        rectPath.close()
                        drawPath(
                            path = rectPath,
                            color = rectColor,
                            style = Stroke(width = 1.2f, pathEffect = dash)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Return a display gain that keeps every visible sample inside the requested
 * fraction of the positive and negative plot ranges.  It only reduces the
 * caller's requested gain; it never amplifies a signal on its own.
 */
internal fun displayGainWithHeadroom(
    values: FloatArray,
    startIndex: Int,
    requestedGain: Float,
    offset: Float,
    yMin: Float,
    yMax: Float,
    headroomFraction: Float
): Float {
    if (values.isEmpty() || startIndex !in values.indices) return requestedGain.coerceAtLeast(0f)

    val fraction = headroomFraction.coerceIn(0.1f, 1f)
    val positiveRoom = max(0f, yMax - offset) * fraction
    val negativeRoom = max(0f, offset - yMin) * fraction
    var safeLimit = Float.POSITIVE_INFINITY

    for (index in startIndex until values.size) {
        val delta = values[index] - offset
        if (!delta.isFinite()) continue
        if (delta > 1e-6f) {
            safeLimit = min(safeLimit, positiveRoom / delta)
        } else if (delta < -1e-6f) {
            safeLimit = min(safeLimit, negativeRoom / -delta)
        }
    }

    val nonNegativeRequested = requestedGain.coerceAtLeast(0f)
    return if (safeLimit.isFinite()) min(nonNegativeRequested, safeLimit.coerceAtLeast(0f))
    else nonNegativeRequested
}

/** Draw one representative beat over normalized 0-100% cardiac phase. */
@Composable
fun RepresentativeHeartbeatCycle(
    modifier: Modifier = Modifier,
    valuesProvider: () -> List<Float>,
    color: Color,
    frameTick: State<Long>
) {
    key(frameTick.value) {
        Canvas(modifier = modifier.fillMaxWidth().height(140.dp)) {
            val values = valuesProvider()
            val w = size.width
            val h = size.height
            val gridColor = color.copy(alpha = 0.12f)
            for (index in 1 until 4) {
                val x = w * index / 4f
                drawLine(
                    color = gridColor,
                    start = androidx.compose.ui.geometry.Offset(x, 0f),
                    end = androidx.compose.ui.geometry.Offset(x, h),
                    strokeWidth = 1f
                )
            }
            for (index in 1 until 4) {
                val y = h * index / 4f
                drawLine(
                    color = gridColor,
                    start = androidx.compose.ui.geometry.Offset(0f, y),
                    end = androidx.compose.ui.geometry.Offset(w, y),
                    strokeWidth = 1f
                )
            }
            drawLine(
                color = color.copy(alpha = 0.35f),
                start = androidx.compose.ui.geometry.Offset(0f, h / 2f),
                end = androidx.compose.ui.geometry.Offset(w, h / 2f),
                strokeWidth = 1.2f
            )
            if (values.size < 2) return@Canvas

            val path = Path()
            for (index in values.indices) {
                val x = index.toFloat() / (values.size - 1).toFloat() * w
                val normalized = ((values[index] + 1.15f) / 2.30f).coerceIn(0f, 1f)
                val y = h - normalized * h
                if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(
                path = path,
                color = color,
                alpha = 0.9f,
                style = Stroke(width = 1.8f, cap = StrokeCap.Round, join = StrokeJoin.Round)
            )
        }
    }
}


