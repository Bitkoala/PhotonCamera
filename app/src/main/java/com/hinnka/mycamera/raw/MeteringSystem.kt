package com.hinnka.mycamera.raw

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

/**
 * 评价测光与场景分析系统
 */
object MeteringSystem {
    private const val DISPLAY_TARGET_LUMA = 0.18f
    private const val LUMA_FLOOR = 0.001f
    private const val MAX_LINEAR_LUMA = 16.0f
    private const val AUTO_DEVELOP_EPSILON = 0.0001f
    private const val SRGB_TRANSFER_THRESHOLD = 0.04045f
    private const val SRGB_LINEAR_SCALE = 12.92f
    private const val SRGB_TRANSFER_A = 0.055f
    private const val SRGB_TRANSFER_GAMMA = 2.4f
    private const val CENTER_WEIGHT_SIGMA = 0.32f
    private const val MIN_METERING_SAMPLE_COUNT = 32

    data class MeteringResult(
        val meteredEv: Float,
        val dynamicRangeGap: Float,
        val avgLuma: Float,
        val clipLow: Float,
        val clipHigh: Float,
        val curveWhitePoint: Float
    ) {
        companion object {
            val EMPTY = MeteringResult(
                meteredEv = 0f,
                dynamicRangeGap = 0f,
                avgLuma = 0f,
                clipLow = 0f,
                clipHigh = 0f,
                curveWhitePoint = 0f
            )
        }
    }

    data class ShadowsHighlightsParams(
        val highlights: Float,
        val shadows: Float,
    ) {
        companion object {
            val NEUTRAL = ShadowsHighlightsParams(
                highlights = 0f,
                shadows = 0f,
            )
        }
    }

    data class SrgbThumbnailMeteringStats(
        val weightedLogLumaSum: Double,
        val weightSum: Double,
        val sampleCount: Int,
        val sanitizedSampleCount: Int,
        val midToneLinearLuma: Float
    )

    fun hasManualRawDevelopAdjustments(
        rawExposureCompensation: Float,
        rawHighlightsAdjustment: Float,
        rawShadowsAdjustment: Float
    ): Boolean {
        return abs(rawExposureCompensation) > AUTO_DEVELOP_EPSILON ||
            abs(rawHighlightsAdjustment) > AUTO_DEVELOP_EPSILON ||
            abs(rawShadowsAdjustment) > AUTO_DEVELOP_EPSILON
    }

    fun analyzeSrgbThumbnail(
        width: Int,
        height: Int,
        argbPixels: IntArray
    ): SrgbThumbnailMeteringStats? {
        if (width <= 0 || height <= 0 || argbPixels.size < width * height) {
            return null
        }

        var weightedLogLumaSum = 0.0
        var weightSum = 0.0
        var sampleCount = 0
        var sanitizedSampleCount = 0

        for (y in 0 until height) {
            val yFraction = (y + 0.5f) / height.toFloat()
            for (x in 0 until width) {
                val pixel = argbPixels[y * width + x]
                val alpha = (pixel ushr 24) and 0xff
                if (alpha == 0) continue

                val alphaScale = alpha / 255f
                val r = srgbToLinear(((pixel ushr 16) and 0xff) / 255f)
                val g = srgbToLinear(((pixel ushr 8) and 0xff) / 255f)
                val b = srgbToLinear((pixel and 0xff) / 255f)
                val rawLuma = (0.2126f * r + 0.7152f * g + 0.0722f * b) * alphaScale
                val luma = sanitizeLinearLuma(rawLuma)
                if (luma != rawLuma) {
                    sanitizedSampleCount++
                }

                val xFraction = (x + 0.5f) / width.toFloat()
                val weight = centerWeight(xFraction, yFraction)
                weightedLogLumaSum += log2(luma.coerceAtLeast(LUMA_FLOOR)).toDouble() * weight.toDouble()
                weightSum += weight.toDouble()
                sampleCount++
            }
        }

        if (sampleCount < MIN_METERING_SAMPLE_COUNT || weightSum <= 0.0) {
            return null
        }

        val midToneLinearLuma = sanitizeAverageLuma(
            exp2((weightedLogLumaSum / weightSum).toFloat()),
            DISPLAY_TARGET_LUMA
        )
        return SrgbThumbnailMeteringStats(
            weightedLogLumaSum = weightedLogLumaSum,
            weightSum = weightSum,
            sampleCount = sampleCount,
            sanitizedSampleCount = sanitizedSampleCount,
            midToneLinearLuma = midToneLinearLuma
        )
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

    private fun centerWeight(xFraction: Float, yFraction: Float): Float {
        val dx = (xFraction - 0.5f) / CENTER_WEIGHT_SIGMA
        val dy = (yFraction - 0.5f) / CENTER_WEIGHT_SIGMA
        val gaussian = exp((-0.5f * (dx * dx + dy * dy)).toDouble()).toFloat()
        return 0.35f + gaussian * 0.65f
    }

    private fun srgbToLinear(value: Float): Float {
        val clamped = value.coerceIn(0f, 1f)
        return if (clamped <= SRGB_TRANSFER_THRESHOLD) {
            clamped / SRGB_LINEAR_SCALE
        } else {
            ((clamped + SRGB_TRANSFER_A) / (1f + SRGB_TRANSFER_A)).pow(SRGB_TRANSFER_GAMMA)
        }
    }

    private fun log2(value: Float): Float {
        return (ln(value.toDouble()) / ln(2.0)).toFloat()
    }

    private fun exp2(value: Float): Float {
        return exp(value * ln(2.0)).toFloat()
    }

}
