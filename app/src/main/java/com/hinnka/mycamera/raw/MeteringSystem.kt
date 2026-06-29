package com.hinnka.mycamera.raw

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 评价测光与场景分析系统
 */
object MeteringSystem {
    const val RAW_EXPOSURE_MIN_EV = -4f
    const val RAW_EXPOSURE_MAX_EV = 4f

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
    private const val LOG_LUMA_HISTOGRAM_BIN_COUNT = 512
    private const val MIDTONE_TRIM_LOW_QUANTILE = 0.25f
    private const val MIDTONE_TRIM_HIGH_QUANTILE = 0.75f
    private const val MIDTONE_MEDIAN_LOG_WEIGHT = 0.55f
    private const val CLIP_LOW_LUMA = 0.002f
    private const val CLIP_HIGH_LUMA = 0.995f
    private const val HIGHLIGHT_COMPRESSION_START_LUMA = 0.62f
    private const val HIGHLIGHT_COMPRESSION_FULL_LUMA = 0.96f
    private const val HIGHLIGHT_COMPRESSION_AMOUNT_MIN = 0.015f
    private const val HIGHLIGHT_COMPRESSION_AMOUNT_FULL = 0.18f
    private const val HIGHLIGHT_COMPRESSION_STRENGTH_MIN = 0.32f
    private const val HIGHLIGHT_COMPRESSION_STRENGTH_FULL = 0.86f
    private const val AUTO_HIGHLIGHTS_MAX_REDUCTION = 0.8f
    private val LOG_LUMA_HISTOGRAM_MIN = log2(LUMA_FLOOR)
    private val LOG_LUMA_HISTOGRAM_MAX = log2(MAX_LINEAR_LUMA)
    private val LOG_LUMA_HISTOGRAM_RANGE =
        (LOG_LUMA_HISTOGRAM_MAX - LOG_LUMA_HISTOGRAM_MIN).coerceAtLeast(0.000001f)

    data class MeteringResult(
        val meteredEv: Float,
        val dynamicRangeGap: Float,
        val avgLuma: Float,
        val clipLow: Float,
        val clipHigh: Float,
        val curveWhitePoint: Float,
        val highlightCompression: HighlightCompressionEstimate = HighlightCompressionEstimate.NEUTRAL
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

    data class HighlightCompressionEstimate(
        val amount: Float,
        val strength: Float,
        val reductionThreshold: Float,
        val autoHighlightsAdjustment: Float
    ) {
        companion object {
            val NEUTRAL = HighlightCompressionEstimate(
                amount = 0f,
                strength = 0f,
                reductionThreshold = 0f,
                autoHighlightsAdjustment = 0f
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
        val width: Int,
        val height: Int,
        val weightedLogLumaSum: Double,
        val weightSum: Double,
        val sampleCount: Int,
        val sanitizedSampleCount: Int,
        val midToneLinearLuma: Float,
        val clipLow: Float,
        val clipHigh: Float,
        val highlightCompression: HighlightCompressionEstimate
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
        var clipLowWeightSum = 0.0
        var clipHighWeightSum = 0.0
        var highlightCompressionWeightSum = 0.0
        var highlightCompressionStrengthSum = 0.0
        val logLumaHistogramWeightSums = DoubleArray(LOG_LUMA_HISTOGRAM_BIN_COUNT)
        val logLumaHistogramWeightedSums = DoubleArray(LOG_LUMA_HISTOGRAM_BIN_COUNT)

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
                val logLuma = log2(luma.coerceAtLeast(LUMA_FLOOR))
                weightedLogLumaSum += logLuma.toDouble() * weight.toDouble()
                weightSum += weight.toDouble()
                val histogramIndex = logLumaHistogramIndex(logLuma)
                logLumaHistogramWeightSums[histogramIndex] += weight.toDouble()
                logLumaHistogramWeightedSums[histogramIndex] += logLuma.toDouble() * weight.toDouble()
                if (luma <= CLIP_LOW_LUMA) {
                    clipLowWeightSum += weight.toDouble()
                }
                if (luma >= CLIP_HIGH_LUMA) {
                    clipHighWeightSum += weight.toDouble()
                }
                val compressionWeight = smoothStep(
                    HIGHLIGHT_COMPRESSION_START_LUMA,
                    HIGHLIGHT_COMPRESSION_FULL_LUMA,
                    luma
                )
                if (compressionWeight > 0f) {
                    val weightedCompression = weight.toDouble() * compressionWeight.toDouble()
                    val compressionStrength = smoothStep(
                        HIGHLIGHT_COMPRESSION_START_LUMA,
                        1f,
                        luma
                    )
                    highlightCompressionWeightSum += weightedCompression
                    highlightCompressionStrengthSum += weightedCompression * compressionStrength.toDouble()
                }
                sampleCount++
            }
        }

        if (sampleCount < MIN_METERING_SAMPLE_COUNT || weightSum <= 0.0) {
            return null
        }

        val midToneLinearLuma = sanitizeAverageLuma(
            robustMidToneLinearLuma(
                logLumaHistogramWeightSums = logLumaHistogramWeightSums,
                logLumaHistogramWeightedSums = logLumaHistogramWeightedSums,
                weightSum = weightSum,
                fallbackLogLuma = (weightedLogLumaSum / weightSum).toFloat()
            ),
            DISPLAY_TARGET_LUMA
        )
        val clipLow = (clipLowWeightSum / weightSum).toFloat().coerceIn(0f, 1f)
        val clipHigh = (clipHighWeightSum / weightSum).toFloat().coerceIn(0f, 1f)
        val highlightCompressionAmount =
            (highlightCompressionWeightSum / weightSum).toFloat().coerceIn(0f, 1f)
        val highlightCompressionStrength = if (highlightCompressionWeightSum > 0.0) {
            (highlightCompressionStrengthSum / highlightCompressionWeightSum).toFloat().coerceIn(0f, 1f)
        } else {
            0f
        }
        return SrgbThumbnailMeteringStats(
            width = width,
            height = height,
            weightedLogLumaSum = weightedLogLumaSum,
            weightSum = weightSum,
            sampleCount = sampleCount,
            sanitizedSampleCount = sanitizedSampleCount,
            midToneLinearLuma = midToneLinearLuma,
            clipLow = clipLow,
            clipHigh = clipHigh,
            highlightCompression = estimateHighlightCompression(
                amount = highlightCompressionAmount,
                strength = highlightCompressionStrength
            )
        )
    }

    private fun logLumaHistogramIndex(logLuma: Float): Int {
        val normalized = (logLuma - LOG_LUMA_HISTOGRAM_MIN) / LOG_LUMA_HISTOGRAM_RANGE
        return (normalized * (LOG_LUMA_HISTOGRAM_BIN_COUNT - 1))
            .toInt()
            .coerceIn(0, LOG_LUMA_HISTOGRAM_BIN_COUNT - 1)
    }

    private fun robustMidToneLinearLuma(
        logLumaHistogramWeightSums: DoubleArray,
        logLumaHistogramWeightedSums: DoubleArray,
        weightSum: Double,
        fallbackLogLuma: Float
    ): Float {
        val medianLogLuma = weightedLogLumaQuantile(
            logLumaHistogramWeightSums = logLumaHistogramWeightSums,
            logLumaHistogramWeightedSums = logLumaHistogramWeightedSums,
            weightSum = weightSum,
            quantile = 0.5f,
            fallbackLogLuma = fallbackLogLuma
        )
        val trimmedLogMean = weightedTrimmedLogLumaMean(
            logLumaHistogramWeightSums = logLumaHistogramWeightSums,
            logLumaHistogramWeightedSums = logLumaHistogramWeightedSums,
            weightSum = weightSum,
            lowQuantile = MIDTONE_TRIM_LOW_QUANTILE,
            highQuantile = MIDTONE_TRIM_HIGH_QUANTILE,
            fallbackLogLuma = fallbackLogLuma
        )
        val medianWeight = MIDTONE_MEDIAN_LOG_WEIGHT.coerceIn(0f, 1f)
        val robustLogLuma = medianLogLuma * medianWeight + trimmedLogMean * (1f - medianWeight)
        return exp2(robustLogLuma)
    }

    private fun weightedLogLumaQuantile(
        logLumaHistogramWeightSums: DoubleArray,
        logLumaHistogramWeightedSums: DoubleArray,
        weightSum: Double,
        quantile: Float,
        fallbackLogLuma: Float
    ): Float {
        if (weightSum <= 0.0) {
            return fallbackLogLuma
        }
        val targetWeight = weightSum * quantile.coerceIn(0f, 1f).toDouble()
        var cumulativeWeight = 0.0
        for (i in logLumaHistogramWeightSums.indices) {
            val binWeight = logLumaHistogramWeightSums[i]
            if (binWeight <= 0.0) continue
            cumulativeWeight += binWeight
            if (cumulativeWeight >= targetWeight) {
                return averageLogLumaForBin(
                    logLumaHistogramWeightedSums = logLumaHistogramWeightedSums,
                    binIndex = i,
                    binWeight = binWeight,
                    fallbackLogLuma = fallbackLogLuma
                )
            }
        }
        return fallbackLogLuma
    }

    private fun weightedTrimmedLogLumaMean(
        logLumaHistogramWeightSums: DoubleArray,
        logLumaHistogramWeightedSums: DoubleArray,
        weightSum: Double,
        lowQuantile: Float,
        highQuantile: Float,
        fallbackLogLuma: Float
    ): Float {
        if (weightSum <= 0.0) {
            return fallbackLogLuma
        }
        val lowWeight = weightSum * lowQuantile.coerceIn(0f, 1f).toDouble()
        val highWeight = weightSum * highQuantile.coerceIn(0f, 1f).toDouble()
        if (highWeight <= lowWeight) {
            return fallbackLogLuma
        }

        var cumulativeWeight = 0.0
        var trimmedWeightedLogSum = 0.0
        var trimmedWeightSum = 0.0
        for (i in logLumaHistogramWeightSums.indices) {
            val binWeight = logLumaHistogramWeightSums[i]
            if (binWeight <= 0.0) continue

            val binStartWeight = cumulativeWeight
            val binEndWeight = cumulativeWeight + binWeight
            val overlapWeight = (minOf(binEndWeight, highWeight) - maxOf(binStartWeight, lowWeight))
                .coerceAtLeast(0.0)
            if (overlapWeight > 0.0) {
                val binAverageLogLuma = averageLogLumaForBin(
                    logLumaHistogramWeightedSums = logLumaHistogramWeightedSums,
                    binIndex = i,
                    binWeight = binWeight,
                    fallbackLogLuma = fallbackLogLuma
                )
                trimmedWeightedLogSum += binAverageLogLuma.toDouble() * overlapWeight
                trimmedWeightSum += overlapWeight
            }
            cumulativeWeight = binEndWeight
        }

        return if (trimmedWeightSum > 0.0) {
            (trimmedWeightedLogSum / trimmedWeightSum).toFloat()
        } else {
            fallbackLogLuma
        }
    }

    private fun averageLogLumaForBin(
        logLumaHistogramWeightedSums: DoubleArray,
        binIndex: Int,
        binWeight: Double,
        fallbackLogLuma: Float
    ): Float {
        val averageLogLuma = (logLumaHistogramWeightedSums[binIndex] / binWeight).toFloat()
        return if (averageLogLuma.isFinite()) {
            averageLogLuma
        } else {
            fallbackLogLuma
        }
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

    fun estimateHighlightCompression(
        amount: Float,
        strength: Float
    ): HighlightCompressionEstimate {
        val safeAmount = sanitizeUnit(amount)
        val safeStrength = sanitizeUnit(strength)
        val amountPressure = smoothStep(
            HIGHLIGHT_COMPRESSION_AMOUNT_MIN,
            HIGHLIGHT_COMPRESSION_AMOUNT_FULL,
            safeAmount
        )
        val strengthPressure = smoothStep(
            HIGHLIGHT_COMPRESSION_STRENGTH_MIN,
            HIGHLIGHT_COMPRESSION_STRENGTH_FULL,
            safeStrength
        )
        val reductionThreshold = sqrt(amountPressure * strengthPressure).coerceIn(0f, 1f)
        val autoHighlightsAdjustment = if (reductionThreshold <= 0f) {
            0f
        } else {
            (-AUTO_HIGHLIGHTS_MAX_REDUCTION * reductionThreshold).coerceIn(-1f, 0f)
        }
        return HighlightCompressionEstimate(
            amount = safeAmount,
            strength = safeStrength,
            reductionThreshold = reductionThreshold,
            autoHighlightsAdjustment = autoHighlightsAdjustment
        )
    }

    private fun sanitizeUnit(value: Float): Float {
        return if (value.isFinite()) value.coerceIn(0f, 1f) else 0f
    }

    private fun smoothStep(edge0: Float, edge1: Float, value: Float): Float {
        val t = ((value - edge0) / (edge1 - edge0).coerceAtLeast(0.000001f)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun log2(value: Float): Float {
        return (ln(value.toDouble()) / ln(2.0)).toFloat()
    }

    private fun exp2(value: Float): Float {
        return exp(value * ln(2.0)).toFloat()
    }

}
