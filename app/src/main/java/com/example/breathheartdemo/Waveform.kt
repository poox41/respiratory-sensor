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
    frameTick: State<Long>
) {
    key(frameTick.value) {
        Canvas(modifier = modifier.fillMaxWidth().height(140.dp)) {
            val effectiveWindowMs = (windowMsProvider?.invoke() ?: windowMs).coerceAtLeast(1L)
            val effectiveGain = gainProvider?.invoke() ?: gain
            val effectivePeakTimes = peakTimesProvider?.invoke() ?: peakTimes
            val vMin = yMin
            val vMax = yMax
            val range = max(1e-6f, vMax - vMin)

            val w = size.width
            val h = size.height
            val path = Path()
            val minStepPx = 0.5f
            var lastX = Float.NEGATIVE_INFINITY
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

            for (i in lo until vs.size) {
                val t = ts[i]
                val xNorm = (t - tMin).toFloat() / effectiveWindowMs.toFloat()
                val x = xNorm.coerceIn(0f, 1f) * w
                if (x - lastX < minStepPx) continue
                lastX = x
                val scaled = (vs[i] - offset) * effectiveGain + offset
//            val clamped = min(vMax, max(vMin, scaled))
                val yNorm = ((scaled - vMin) / range).coerceIn(0f, 1f)
                val y = h - yNorm * h
                if (!started) {
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


