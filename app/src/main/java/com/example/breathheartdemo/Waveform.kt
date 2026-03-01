package com.example.breathheartdemo

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.key
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import kotlin.math.max
import kotlin.math.min

//画曲线
@Composable
fun Waveform(
    modifier: Modifier = Modifier,
    buffer: RingBuffer,
    color: Color = Color.Black,
    yMin: Float = -2f,
    yMax: Float = 2f,
    windowMs: Long = 6000L,
    gain: Float = 1f,
    offset: Float = 0f,
    showGrid: Boolean = true,
    showZeroLine: Boolean = true,
    zeroLineValue: Float? = null
) {
    var tick by remember { mutableLongStateOf(0L) }
    val effectiveGain = gain
    LaunchedEffect(Unit) {
        while (true) {
            delay(33L) // ~30 FPS redraw
            tick++
        }
    }
    key(tick) {
        Canvas(modifier = modifier.fillMaxWidth().height(140.dp)) {
            val vMin = yMin
            val vMax = yMax
            val range = max(1e-6f, vMax - vMin)

            val w = size.width
            val h = size.height
            val path = Path()
            val minStepPx = 1.5f
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

            if (buffer.isEmpty()) return@Canvas
            val (ts, vs) = buffer.snapshot()
            if (vs.isEmpty()) return@Canvas

            val tMax = ts[vs.lastIndex]
            val tMin = tMax - windowMs

            for (i in vs.indices) {
                val t = ts[i]
                if (t < tMin) continue
                val xNorm = (t - tMin).toFloat() / windowMs.toFloat()
                val x = xNorm.coerceIn(0f, 1f) * w
                if (x - lastX < minStepPx) continue
                lastX = x
                val scaled = (vs[i] - offset) * effectiveGain + offset
                val clamped = min(vMax, max(vMin, scaled))
                val yNorm = (clamped - vMin) / range
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
                    style = Stroke(width = 2f)
                )
            }
        }
    }
}
