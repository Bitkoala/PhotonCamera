package com.hinnka.mycamera.raw

import android.util.Half
import com.hinnka.mycamera.color.TransferCurve
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

/**
 * 评价测光与场景分析系统
 */
object MeteringSystem {
    private const val TAG = "MeteringSystem"
    private const val DISPLAY_TARGET_LUMA = 0.18f
    private const val ACR3_AVERAGE_TONE_CURVE_EV = 1f


    data class MeteringResult(
        val meteredEv: Float,
        val dynamicRangeGap: Float,
        val avgLuma: Float,
        val p998: Float
    )

    fun analyzeLinearHalfFloatExposureEv(
        pixelBuffer: ShortBuffer,
        width: Int,
        height: Int,
        weightBuffer: ByteBuffer? = null, // Optional weight mask (e.g. depth map)
        droMode: RawProcessingPreferences.DROMode = RawProcessingPreferences.DROMode.OFF
    ): MeteringResult {
        return analyzeExposureEv(
            width = width,
            height = height,
            weightBuffer = weightBuffer,
            droMode = droMode,
            targetLuma = DISPLAY_TARGET_LUMA,
            toneCurveEvCompensation = -ACR3_AVERAGE_TONE_CURVE_EV,
            tag = "Linear RAW AE"
        ) { index ->
            val base = index * 4
            val r = Half.toFloat(pixelBuffer.get(base)).coerceAtLeast(0f)
            val g = Half.toFloat(pixelBuffer.get(base + 1)).coerceAtLeast(0f)
            val b = Half.toFloat(pixelBuffer.get(base + 2)).coerceAtLeast(0f)
            r * 0.2126f + g * 0.7152f + b * 0.0722f
        }
    }

    private inline fun analyzeExposureEv(
        width: Int,
        height: Int,
        weightBuffer: ByteBuffer?,
        droMode: RawProcessingPreferences.DROMode,
        targetLuma: Float,
        toneCurveEvCompensation: Float,
        tag: String,
        lumaAt: (Int) -> Float
    ): MeteringResult {
        val pixelCount = width * height
        if (pixelCount == 0) return MeteringResult(0f, 0f, 0f, 0f)

        val lumas = FloatArray(pixelCount)
        var weightedLumaSum = 0f
        var totalWeight = 0f

        weightBuffer?.position(0)

        for (y in 0 until height) {
            val ny = (y.toFloat() / height) - 0.5f
            for (x in 0 until width) {
                val nx = (x.toFloat() / width) - 0.5f
                val distSq = nx * nx + ny * ny
                val spatialWeight = lerp(1.0f, 0.5f, distSq * 2f)

                var depthWeight = 1.0f
                if (weightBuffer != null && weightBuffer.hasRemaining()) {
                    val wValue = (weightBuffer.get().toInt() and 0xFF) / 255f
                    depthWeight = lerp(1.0f, 10.0f, wValue)
                    if (weightBuffer.capacity() >= pixelCount * 4 && weightBuffer.remaining() >= 3) {
                        weightBuffer.get(); weightBuffer.get(); weightBuffer.get()
                    }
                }

                val idx = y * width + x
                val luma = lumaAt(idx)
                val highlightWeight = calculateHighlightWeight(luma, droMode)
                val finalWeight = spatialWeight * depthWeight * highlightWeight

                lumas[idx] = luma
                weightedLumaSum += luma * finalWeight
                totalWeight += finalWeight
            }
        }

        lumas.sort()
        val p998 = lumas[(pixelCount * 0.998f).toInt().coerceIn(0, pixelCount - 1)]

        val highlightAnchorGain = 1f / p998.coerceAtLeast(0.01f)
        val avgLuma = weightedLumaSum / totalWeight.coerceAtLeast(0.001f)
        val midToneGain = targetLuma / avgLuma.coerceAtLeast(0.001f)
        val dynamicRangeGap = midToneGain / highlightAnchorGain

        val extra = 1f - smoothStep(0.66f, 2.22f, dynamicRangeGap)
        val adaptiveGain = midToneGain * lerp(0.9f, 1.2f, extra)
        val rawMeteredEv = log2(adaptiveGain.coerceIn(0.25f, 4.0f))
        val meteredEv = rawMeteredEv + toneCurveEvCompensation

        PLog.d(
            TAG,
            "$tag: dro=$droMode p998=$p998 avg=$avgLuma target=$targetLuma " +
                "midToneGain=$midToneGain highlightAnchorGain=$highlightAnchorGain gain=$adaptiveGain " +
                "rawEv=$rawMeteredEv toneComp=$toneCurveEvCompensation ev=$meteredEv gap=$dynamicRangeGap"
        )

        return MeteringResult(
            meteredEv = meteredEv.coerceIn(-2f, 2f),
            dynamicRangeGap = dynamicRangeGap,
            avgLuma = avgLuma,
            p998 = p998
        )
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

    private fun srgbToLinear(value: Float): Float {
        val srgb = value.coerceIn(0f, 1f)
        return if (srgb <= 0.04045f) {
            srgb / 12.92f
        } else {
            ((srgb + 0.055f) / 1.055f).pow(2.4f)
        }
    }

    private fun calculateHighlightWeight(
        luma: Float,
        droMode: RawProcessingPreferences.DROMode
    ): Float {
        if (!droMode.isEnabled) {
            return 1f
        }

        val minWeight = when (droMode) {
            RawProcessingPreferences.DROMode.OFF -> 1f
            RawProcessingPreferences.DROMode.DR100 -> 0.5f
            RawProcessingPreferences.DROMode.DR200 -> 0.25f
            RawProcessingPreferences.DROMode.DR400 -> 0.12f
        }
        val highlightFraction = smoothStep(0.65f, 0.95f, luma)
        return lerp(1f, minWeight, highlightFraction)
    }

    private fun lerp(start: Float, end: Float, fraction: Float): Float {
        return start + (end - start) * fraction.coerceIn(0f, 1f)
    }

    private fun smoothStep(edge0: Float, edge1: Float, x: Float): Float {
        val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }
}
