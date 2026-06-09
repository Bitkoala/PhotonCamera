package com.hinnka.mycamera.raw

import android.annotation.SuppressLint
import android.util.Half
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
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
    private const val MATRIX_GRID_COLUMNS = 5
    private const val MATRIX_GRID_ROWS = 5
    private const val MIN_METERING_WEIGHT = 0.08f
    private const val MAX_METERING_WEIGHT = 3.0f
    private const val LUMA_FLOOR = 0.001f
    private const val MAX_LINEAR_LUMA = 16.0f
    private const val MIN_DEPTH_SEPARATION = 0.08f


    data class MeteringResult(
        val meteredEv: Float,
        val dynamicRangeGap: Float,
        val avgLuma: Float,
        val p998: Float
    )

    private data class DepthWeightMap(
        val weights: FloatArray,
        val separation: Float,
        val enabled: Boolean
    )

    @SuppressLint("HalfFloat")
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
            tag = "Linear RAW AE"
        ) { index ->
            val base = index * 4
            val scale = 2f.pow(ACR3_AVERAGE_TONE_CURVE_EV)
            val r = Half.toFloat(pixelBuffer.get(base)) * scale
            val g = Half.toFloat(pixelBuffer.get(base + 1)) * scale
            val b = Half.toFloat(pixelBuffer.get(base + 2)) * scale
            r * 0.2126f + g * 0.7152f + b * 0.0722f
        }
    }

    private inline fun analyzeExposureEv(
        width: Int,
        height: Int,
        weightBuffer: ByteBuffer?,
        droMode: RawProcessingPreferences.DROMode,
        targetLuma: Float,
        tag: String,
        lumaAt: (Int) -> Float
    ): MeteringResult {
        val pixelCount = width * height
        if (pixelCount == 0) return MeteringResult(0f, 0f, 0f, 0f)

        val depthWeights = decodeDepthWeights(weightBuffer, pixelCount)
        val lumas = FloatArray(pixelCount)
        var weightedLumaSum = 0.0
        var weightedLogLumaSum = 0.0
        var totalWeight = 0.0
        var sanitizedSampleCount = 0

        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x
                val rawLuma = lumaAt(idx)
                val luma = sanitizeLinearLuma(rawLuma)
                if (rawLuma != luma) {
                    sanitizedSampleCount++
                }
                val spatialWeight = calculateMatrixSpatialWeight(x, y, width, height)
                val toneWeight = calculateReflectiveMeterWeight(luma)
                val depthWeight = depthWeights?.weights?.get(idx) ?: 1f
                val finalWeight = sanitizeMeteringWeight(spatialWeight * toneWeight * depthWeight)

                lumas[idx] = luma
                weightedLumaSum += (luma * finalWeight).toDouble()
                weightedLogLumaSum += (log2(luma.coerceAtLeast(LUMA_FLOOR)) * finalWeight).toDouble()
                totalWeight += finalWeight.toDouble()
            }
        }

        lumas.sort()
        val p998 = lumas[(pixelCount * 0.998f).toInt().coerceIn(0, pixelCount - 1)]

        val highlightAnchorGain = 2f.pow(ACR3_AVERAGE_TONE_CURVE_EV) / p998.coerceAtLeast(0.01f)
        val safeTotalWeight = totalWeight.coerceAtLeast(0.001)
        val linearAvgLuma = sanitizeAverageLuma((weightedLumaSum / safeTotalWeight).toFloat(), targetLuma)
        val logAvgLuma = sanitizeAverageLuma(exp2((weightedLogLumaSum / safeTotalWeight).toFloat()), targetLuma)
        val avgLuma = sanitizeAverageLuma(lerp(logAvgLuma, linearAvgLuma, 0.25f), targetLuma)
        val midToneGain = targetLuma / avgLuma.coerceAtLeast(0.001f)
        val dynamicRangeGap = midToneGain / highlightAnchorGain

        val extra = smoothStep(0.66f, 2.2f, dynamicRangeGap)
        val adaptiveGain = lerp(midToneGain * 1.2f, highlightAnchorGain * 1.2f, extra)
        val rawMeteredEv = log2(adaptiveGain.coerceIn(0.25f, 4.0f))

        PLog.d(
            TAG,
            "$tag: dro=$droMode p998=$p998 avg=$avgLuma target=$targetLuma " +
                "logAvg=$logAvgLuma linearAvg=$linearAvgLuma " +
                "midToneGain=$midToneGain highlightAnchorGain=$highlightAnchorGain gain=$adaptiveGain " +
                "ev=$rawMeteredEv gap=$dynamicRangeGap " +
                "depth=${depthWeights?.enabled == true} depthSeparation=${depthWeights?.separation ?: 0f} " +
                "sanitized=$sanitizedSampleCount"
        )

        return MeteringResult(
            meteredEv = rawMeteredEv.coerceIn(-2f, 2f),
            dynamicRangeGap = dynamicRangeGap,
            avgLuma = avgLuma,
            p998 = p998
        )
    }

    private fun decodeDepthWeights(
        weightBuffer: ByteBuffer?,
        pixelCount: Int
    ): DepthWeightMap? {
        if (weightBuffer == null || pixelCount == 0 || weightBuffer.capacity() < pixelCount) {
            return null
        }

        val stride = if (weightBuffer.capacity() >= pixelCount * 4) 4 else 1
        val depthValues = FloatArray(pixelCount)
        for (i in 0 until pixelCount) {
            depthValues[i] = (weightBuffer.get(i * stride).toInt() and 0xFF) / 255f
        }

        val sortedDepth = depthValues.copyOf()
        sortedDepth.sort()
        val median = percentile(sortedDepth, 0.50f)
        val highAnchor = percentile(sortedDepth, 0.90f)
        val lowAnchor = percentile(sortedDepth, 0.10f)
        val depthRange = (highAnchor - lowAnchor).coerceAtLeast(0f)
        val hasUsableSeparation = depthRange >= MIN_DEPTH_SEPARATION

        val weights = FloatArray(pixelCount) { index ->
            if (!hasUsableSeparation) {
                1f
            } else {
                calculateDepthSeparationWeight(
                    depth = depthValues[index],
                    median = median,
                    depthRange = depthRange
                )
            }
        }

        return DepthWeightMap(
            weights = weights,
            separation = depthRange,
            enabled = hasUsableSeparation
        )
    }

    private fun calculateMatrixSpatialWeight(
        x: Int,
        y: Int,
        width: Int,
        height: Int
    ): Float {
        val u = (x + 0.5f) / width.coerceAtLeast(1)
        val v = (y + 0.5f) / height.coerceAtLeast(1)

        val matrixWeight = calculateZoneMeterWeight(u, v)
        val centerWeight = gaussian2d(u, v, 0.5f, 0.5f, 0.30f)
        val thirdsWeight = (
            gaussian2d(u, v, 1f / 3f, 1f / 3f, 0.18f) +
                gaussian2d(u, v, 2f / 3f, 1f / 3f, 0.18f) +
                gaussian2d(u, v, 1f / 3f, 2f / 3f, 0.18f) +
                gaussian2d(u, v, 2f / 3f, 2f / 3f, 0.18f)
            ) * 0.25f
        val edgeFalloff = lerp(0.72f, 1f, calculateEdgeConfidence(u, v))

        return (0.55f * matrixWeight + 0.30f * centerWeight + 0.15f * thirdsWeight) * edgeFalloff
    }

    private fun calculateZoneMeterWeight(u: Float, v: Float): Float {
        val zoneX = (u * MATRIX_GRID_COLUMNS).toInt().coerceIn(0, MATRIX_GRID_COLUMNS - 1)
        val zoneY = (v * MATRIX_GRID_ROWS).toInt().coerceIn(0, MATRIX_GRID_ROWS - 1)
        val centerX = (zoneX + 0.5f) / MATRIX_GRID_COLUMNS
        val centerY = (zoneY + 0.5f) / MATRIX_GRID_ROWS
        return 0.75f + gaussian2d(centerX, centerY, 0.5f, 0.5f, 0.34f) * 0.55f
    }

    private fun calculateReflectiveMeterWeight(luma: Float): Float {
        val safeLuma = luma.coerceAtLeast(0f)
        val shadowConfidence = smoothStep(0.002f, 0.030f, safeLuma)
        val highlightPenalty = 1f - smoothStep(0.72f, 1.08f, safeLuma)
        val midTonePreference = gaussian1d(log2(safeLuma.coerceAtLeast(0.002f) / DISPLAY_TARGET_LUMA), 0f, 2.1f)
        return lerp(0.70f, 1.18f, midTonePreference) * lerp(0.72f, 1f, shadowConfidence) *
            lerp(0.35f, 1f, highlightPenalty)
    }

    private fun calculateDepthSeparationWeight(
        depth: Float,
        median: Float,
        depthRange: Float
    ): Float {
        val normalizedSeparation = kotlin.math.abs(depth - median) / depthRange.coerceAtLeast(MIN_DEPTH_SEPARATION)
        val saliency = smoothStep(0.18f, 0.88f, normalizedSeparation)
        return lerp(0.88f, 1.55f, saliency)
    }

    private fun sanitizeLinearLuma(value: Float): Float {
        return when {
            value.isFinite() -> value.coerceIn(0f, MAX_LINEAR_LUMA)
            value == Float.POSITIVE_INFINITY -> MAX_LINEAR_LUMA
            else -> 0f
        }
    }

    private fun sanitizeAverageLuma(value: Float, fallback: Float): Float {
        return if (value.isFinite() && value > 0f) {
            value.coerceIn(LUMA_FLOOR, MAX_LINEAR_LUMA)
        } else {
            fallback.coerceIn(LUMA_FLOOR, MAX_LINEAR_LUMA)
        }
    }

    private fun sanitizeMeteringWeight(value: Float): Float {
        return if (value.isFinite()) {
            value.coerceIn(MIN_METERING_WEIGHT, MAX_METERING_WEIGHT)
        } else {
            MIN_METERING_WEIGHT
        }
    }

    private fun sanitizeMeteringGain(value: Float): Float {
        return if (value.isFinite() && value > 0f) {
            value
        } else {
            1f
        }
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

    private fun exp2(value: Float): Float {
        return exp(value * ln(2.0)).toFloat()
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

    private fun calculateEdgeConfidence(u: Float, v: Float): Float {
        val horizontal = 1f - kotlin.math.abs(u - 0.5f) * 2f
        val vertical = 1f - kotlin.math.abs(v - 0.5f) * 2f
        return smoothStep(0f, 0.75f, minOf(horizontal, vertical))
    }

    private fun gaussian1d(value: Float, mean: Float, sigma: Float): Float {
        if (!value.isFinite() || !mean.isFinite() || !sigma.isFinite()) {
            return 0f
        }
        val normalized = (value - mean) / sigma.coerceAtLeast(0.001f)
        return exp((-0.5f * normalized * normalized).toDouble()).toFloat()
    }

    private fun gaussian2d(
        u: Float,
        v: Float,
        centerU: Float,
        centerV: Float,
        sigma: Float
    ): Float {
        if (!u.isFinite() || !v.isFinite() || !centerU.isFinite() || !centerV.isFinite() || !sigma.isFinite()) {
            return 0f
        }
        val safeSigma = sigma.coerceAtLeast(0.001f)
        val du = (u - centerU) / safeSigma
        val dv = (v - centerV) / safeSigma
        return exp((-0.5f * (du * du + dv * dv)).toDouble()).toFloat()
    }

    private fun lerp(start: Float, end: Float, fraction: Float): Float {
        return start + (end - start) * fraction.coerceIn(0f, 1f)
    }

    private fun smoothStep(edge0: Float, edge1: Float, x: Float): Float {
        if (!edge0.isFinite() || !edge1.isFinite() || !x.isFinite()) {
            return 0f
        }
        val width = edge1 - edge0
        if (kotlin.math.abs(width) < 0.000001f) {
            return if (x >= edge1) 1f else 0f
        }
        val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }
}
