package com.hinnka.mycamera.raw

import com.hinnka.mycamera.color.TransferCurve
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

/**
 * 评价测光与场景分析系统
 */
object MeteringSystem {
    private const val TAG = "MeteringSystem"
    private const val DISPLAY_TARGET_LUMA = 0.46f


    fun analyzeRenderedExposureEv(
        byteBuffer: ByteBuffer,
        width: Int,
        height: Int,
    ): Float {
        val pixelCount = width * height
        if (pixelCount == 0) return 0f

        val lumas = FloatArray(pixelCount)

        byteBuffer.position(0)
        for (y in 0 until height) {
            for (x in 0 until width) {

                val r = (byteBuffer.get().toInt() and 0xFF) / 255f
                val g = (byteBuffer.get().toInt() and 0xFF) / 255f
                val b = (byteBuffer.get().toInt() and 0xFF) / 255f
                byteBuffer.get() // skip alpha

                val luma = r * 0.2126f + g * 0.7152f + b * 0.0722f
                val idx = y * width + x
                lumas[idx] = luma
            }
        }

        // 1. Statistical analysis: Sort to find percentiles
        lumas.sort()
        val p998 = lumas[(pixelCount * 0.998f).toInt().coerceIn(0, pixelCount - 1)]
        
        val highlightAnchorGain = 1f / p998.coerceAtLeast(0.01f)

        // 3. Midtone Balance Logic (Spatial Average)
        val avgLuma = lumas.average().toFloat()
        val midToneGain = DISPLAY_TARGET_LUMA / avgLuma.coerceAtLeast(0.001f)
        val dynamicRangeGap = midToneGain / highlightAnchorGain

        val extra = 1f - smoothStep(0.66f, 3f, dynamicRangeGap)

        val baseGain = midToneGain * lerp(0.9f, 1.2f, extra)

        val maxAllowedGain = highlightAnchorGain * 1.1f

        val adaptiveGain = minOf(baseGain, maxAllowedGain)

        val meteredEv = log2(adaptiveGain.coerceIn(0.25f, 4.0f))
        
        PLog.d("MeteringSystem", "Smart AE: p998=$p998 avg=$avgLuma midToneGain=$midToneGain highlightAnchorGain=$highlightAnchorGain gain=$adaptiveGain ev=$meteredEv")
        
        return meteredEv.coerceIn(-2f, 2f)
    }

    private fun percentile(sortedValues: FloatArray, percentile: Float): Float {
        if (sortedValues.isEmpty()) {
            return 0f
        }

        val index = ((sortedValues.size - 1) * percentile.coerceIn(0f, 1f)).toInt()
        return sortedValues[index]
    }

    private fun log2(value: Float): Float {
        return (ln(value.toDouble()) / ln(2.0)).toFloat()
    }

    private fun lerp(start: Float, end: Float, fraction: Float): Float {
        return start + (end - start) * fraction.coerceIn(0f, 1f)
    }

    private fun smoothStep(edge0: Float, edge1: Float, x: Float): Float {
        val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }
}
