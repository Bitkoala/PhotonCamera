package com.hinnka.mycamera.raw

import com.hinnka.mycamera.utils.PLog
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * 评价测光与场景分析系统
 */
object MeteringSystem {
    private const val TAG = "MeteringSystem"
    private const val DISPLAY_TARGET_LUMA = 0.18f
    private const val MAX_METERING_COMPENSATION_EV = 4f
    private const val LUMA_FLOOR = 0.001f
    private const val MAX_LINEAR_LUMA = 16.0f
    private const val RAW_CURVE_NEUTRAL_WHITE_POINT = 1.0f
    private const val AUTO_DEVELOP_EPSILON = 0.0001f
    private const val AUTO_CLIP_FRACTION = 0.0002f
    private const val RAW_AUTO_HIGHLIGHT_LIMIT = 0.95f
    private const val RAW_AUTO_STANDARD_SHADOW_LIMIT = 0.5f
    private const val RAW_AUTO_LOW_KEY_P75_RATIO_FULL = 0.45f
    private const val RAW_AUTO_LOW_KEY_P75_RATIO_START = 0.85f
    private const val RAW_AUTO_LOW_KEY_P90_RATIO_FULL = 0.70f
    private const val RAW_AUTO_LOW_KEY_P90_RATIO_START = 1.25f
    private const val RAW_AUTO_LOW_KEY_MAX_SHADOW_ATTENUATION = 0.78f
    private const val MERTENS_MIN_TONE_WEIGHT = 0.0001f
    private const val MERTENS_TONE_STATS_MAX_SAMPLES = 4096


    data class MeteringResult(
        val meteredEv: Float,
        val dynamicRangeGap: Float,
        val avgLuma: Float,
        val clipLow: Float,
        val clipHigh: Float,
        val highlights: Float,
        val shadows: Float,
        val curveWhitePoint: Float
    ) {
        companion object {
            val EMPTY = MeteringResult(
                meteredEv = 0f,
                dynamicRangeGap = 0f,
                avgLuma = 0f,
                clipLow = 0f,
                clipHigh = 0f,
                highlights = 0f,
                shadows = 0f,
                curveWhitePoint = 0f
            )
        }
    }

    data class ShadowsHighlightsParams(
        val highlights: Float,
        val shadows: Float,
        val curveWhitePoint: Float,
    ) {
        companion object {
            val NEUTRAL = ShadowsHighlightsParams(
                highlights = 0f,
                shadows = 0f,
                curveWhitePoint = RAW_CURVE_NEUTRAL_WHITE_POINT,
            )
        }
    }

    data class GpuRawAutoExposureBaseStats(
        val histogram: IntArray,
        val histogramLogMin: Float,
        val histogramLogMax: Float,
        val weightedLogLumaSum: Double,
        val weightSum: Double,
        val sampleCount: Int,
        val sanitizedSampleCount: Int,
        val groupCount: Int
    )

    data class GpuRawAutoExposureToneStats(
        val highlightDeltaEnergySum: Double,
        val shadowDeltaEnergySum: Double,
        val highlightWeightSum: Double,
        val shadowWeightSum: Double,
        val mertensContrastSum: Double,
        val mertensSaturationSum: Double,
        val mertensWellExposednessSum: Double,
        val sampleCount: Int,
        val sampleStep: Int,
        val groupCount: Int
    )

    class GpuRawAutoExposurePlan internal constructor(
        val compensatedExposureScale: Float,
        val sampleStep: Int,
        val clipLow: Float,
        val clipHigh: Float,
        val p05: Float,
        val p25: Float,
        val p75: Float,
        val p90: Float,
        val p99: Float,
        val compensatedClipLow: Float,
        val compensatedClipHigh: Float,
        val compensatedP05: Float,
        val compensatedP25: Float,
        val compensatedP75: Float,
        val compensatedP90: Float,
        val compensatedP99: Float,
        val baselineExposure: Float,
        val baselineExposureScale: Float,
        val autoExposureScale: Float,
        val targetLuma: Float,
        val meteringCompensationEv: Float,
        val midToneLuma: Float,
        val midToneGain: Float,
        val highlightAnchorGain: Float,
        val adaptiveGain: Float,
        val rawMeteredEv: Float,
        val dynamicRangeGap: Float,
        val sanitizedSampleCount: Int,
        val histogramSampleCount: Int,
        val histogramGroupCount: Int,
        val tag: String
    )

    private data class ExposureFusionToneStats(
        val highlightFusionDelta: Float,
        val shadowFusionDelta: Float,
        val highlightRegionWeight: Float,
        val shadowRegionWeight: Float,
        val mertensContrastMean: Float,
        val mertensSaturationMean: Float,
        val mertensWellExposednessMean: Float,
        val sampleCount: Int,
        val sampleStep: Int
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

    fun prepareGpuLinearRawAutoExposure(
        stats: GpuRawAutoExposureBaseStats,
        baselineExposure: Float,
        meteringCompensationEv: Float = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        tag: String = "Linear RAW AE GPU"
    ): GpuRawAutoExposurePlan? {
        if (stats.histogram.isEmpty() || stats.sampleCount <= 0) {
            return null
        }

        val safeMeteringCompensationEv = sanitizeMeteringCompensationEv(meteringCompensationEv)
        val targetLuma = resolveDisplayTargetLuma(safeMeteringCompensationEv)
        val midToneLuma = if (stats.weightSum > 0.0) {
            exp2((stats.weightedLogLumaSum / stats.weightSum).toFloat())
        } else {
            targetLuma
        }.let { sanitizeAverageLuma(it, targetLuma) }

        val clipLow = percentileFromLogHistogram(
            histogram = stats.histogram,
            percentile = AUTO_CLIP_FRACTION,
            logMin = stats.histogramLogMin,
            logMax = stats.histogramLogMax
        )
        val p05 = percentileFromLogHistogram(stats.histogram, 0.05f, stats.histogramLogMin, stats.histogramLogMax)
        val p25 = percentileFromLogHistogram(stats.histogram, 0.25f, stats.histogramLogMin, stats.histogramLogMax)
        val p75 = percentileFromLogHistogram(stats.histogram, 0.75f, stats.histogramLogMin, stats.histogramLogMax)
        val p90 = percentileFromLogHistogram(stats.histogram, 0.90f, stats.histogramLogMin, stats.histogramLogMax)
        val p99 = percentileFromLogHistogram(stats.histogram, 0.99f, stats.histogramLogMin, stats.histogramLogMax)
        val clipHigh = percentileFromLogHistogram(
            histogram = stats.histogram,
            percentile = 1f - AUTO_CLIP_FRACTION,
            logMin = stats.histogramLogMin,
            logMax = stats.histogramLogMax
        )

        val baselineExposureScale = exposureScaleFromEv(baselineExposure)
        val meteringClipLow = (clipLow * baselineExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val meteringP05 = (p05 * baselineExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val meteringP25 = (p25 * baselineExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val meteringP75 = (p75 * baselineExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val meteringP90 = (p90 * baselineExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val meteringP99 = (p99 * baselineExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val meteringClipHigh = (clipHigh * baselineExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val meteringMidToneLuma = (midToneLuma * baselineExposureScale).coerceIn(LUMA_FLOOR, MAX_LINEAR_LUMA)

        val highlightAnchorGain = maxOf(1f, meteringClipHigh) / meteringClipHigh.coerceAtLeast(0.01f)
        val midToneGain = targetLuma / meteringMidToneLuma.coerceAtLeast(LUMA_FLOOR)
        val dynamicRangeGap = evDifference(meteringClipHigh, meteringClipLow)
        val extra = smoothStep(4f, 12f, dynamicRangeGap)
        val adaptiveGain = lerp(midToneGain * 1.2f, highlightAnchorGain * 1.1f, extra)
        val autoExposureScale = adaptiveGain.coerceIn(0.25f, 4.0f)
        val rawMeteredEv = log2(autoExposureScale)
        val compensatedExposureScale = baselineExposureScale * autoExposureScale

        val sampleStep = calculateMertensToneSampleStep(stats.sampleCount)
        return GpuRawAutoExposurePlan(
            compensatedExposureScale = compensatedExposureScale,
            sampleStep = sampleStep,
            clipLow = meteringClipLow,
            clipHigh = meteringClipHigh,
            p05 = meteringP05,
            p25 = meteringP25,
            p75 = meteringP75,
            p90 = meteringP90,
            p99 = meteringP99,
            compensatedClipLow = (meteringClipLow * autoExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedClipHigh = (meteringClipHigh * autoExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP05 = (meteringP05 * autoExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP25 = (meteringP25 * autoExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP75 = (meteringP75 * autoExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP90 = (meteringP90 * autoExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP99 = (meteringP99 * autoExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            baselineExposure = baselineExposure,
            baselineExposureScale = baselineExposureScale,
            autoExposureScale = autoExposureScale,
            targetLuma = targetLuma,
            meteringCompensationEv = safeMeteringCompensationEv,
            midToneLuma = meteringMidToneLuma,
            midToneGain = midToneGain,
            highlightAnchorGain = highlightAnchorGain,
            adaptiveGain = adaptiveGain,
            rawMeteredEv = rawMeteredEv,
            dynamicRangeGap = dynamicRangeGap,
            sanitizedSampleCount = stats.sanitizedSampleCount,
            histogramSampleCount = stats.sampleCount,
            histogramGroupCount = stats.groupCount,
            tag = tag
        )
    }

    fun finishGpuLinearRawAutoExposure(
        plan: GpuRawAutoExposurePlan,
        stats: GpuRawAutoExposureToneStats
    ): MeteringResult {
        val exposureFusionToneStats = exposureFusionToneStatsFromGpu(stats)
        val highlights = -calculateAutoHighlightCompression(plan.baselineExposure, exposureFusionToneStats.highlightFusionDelta)
        val unprotectedShadows = calculateAutoShadowLift(
            baselineExposure = plan.baselineExposure,
            stats = exposureFusionToneStats
        )
        val lowKeyShadowProtection = calculateLowKeyShadowProtection(
            compensatedP75 = plan.compensatedP75,
            compensatedP90 = plan.compensatedP90,
            targetLuma = plan.targetLuma
        )
        val shadows = applyLowKeyShadowProtection(
            shadows = unprotectedShadows,
            protection = lowKeyShadowProtection,
            shadowLimit = RAW_AUTO_STANDARD_SHADOW_LIMIT
        )
        val curveWhitePoint = if (plan.compensatedClipHigh > 1f) {
            plan.compensatedClipHigh.coerceIn(1.05f, 4.0f)
        } else {
            RAW_CURVE_NEUTRAL_WHITE_POINT
        }

        PLog.d(
            TAG,
            "${plan.tag}: clipLow=${plan.clipLow} clipHigh=${plan.clipHigh} " +
                "p05=${plan.p05} p25=${plan.p25} p75=${plan.p75} p90=${plan.p90} p99=${plan.p99} " +
                "baselineScale=${plan.baselineExposureScale} autoScale=${plan.autoExposureScale} " +
                "finalScale=${plan.compensatedExposureScale} " +
                "compClipLow=${plan.compensatedClipLow} compClipHigh=${plan.compensatedClipHigh} " +
                "compP05=${plan.compensatedP05} compP25=${plan.compensatedP25} " +
                "compP75=${plan.compensatedP75} compP90=${plan.compensatedP90} compP99=${plan.compensatedP99} " +
                "highlightFusionDelta=${exposureFusionToneStats.highlightFusionDelta} " +
                "shadowFusionDelta=${exposureFusionToneStats.shadowFusionDelta} " +
                "highlightRegionWeight=${exposureFusionToneStats.highlightRegionWeight} " +
                "shadowRegionWeight=${exposureFusionToneStats.shadowRegionWeight} " +
                "mertensContrastMean=${exposureFusionToneStats.mertensContrastMean} " +
                "mertensSaturationMean=${exposureFusionToneStats.mertensSaturationMean} " +
                "mertensWellExposednessMean=${exposureFusionToneStats.mertensWellExposednessMean} " +
                "mertensSamples=${exposureFusionToneStats.sampleCount} " +
                "mertensSampleStep=${exposureFusionToneStats.sampleStep} " +
                "lowKeyShadowProtection=$lowKeyShadowProtection " +
                "baselineExposure=${plan.baselineExposure} " +
                "highlights=$highlights shadows=$shadows unprotectedShadows=$unprotectedShadows " +
                "target=${plan.targetLuma} meteringCompensationEv=${plan.meteringCompensationEv} " +
                "midToneLuma=${plan.midToneLuma} " +
                "midToneGain=${plan.midToneGain} highlightAnchorGain=${plan.highlightAnchorGain} " +
                "gain=${plan.adaptiveGain} ev=${plan.rawMeteredEv} gap=${plan.dynamicRangeGap} " +
                "sanitizedSamples=${plan.sanitizedSampleCount} histogramSamples=${plan.histogramSampleCount} " +
                "histogramGroups=${plan.histogramGroupCount} mertensGroups=${stats.groupCount}"
        )

        return MeteringResult(
            meteredEv = plan.rawMeteredEv.coerceIn(-2f, 2f),
            dynamicRangeGap = plan.dynamicRangeGap,
            avgLuma = plan.midToneLuma,
            clipLow = plan.clipLow,
            clipHigh = plan.clipHigh,
            highlights = highlights,
            shadows = shadows,
            curveWhitePoint = curveWhitePoint
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

    private fun emptyExposureFusionToneStats(): ExposureFusionToneStats {
        return ExposureFusionToneStats(
            highlightFusionDelta = 0f,
            shadowFusionDelta = 0f,
            highlightRegionWeight = 0f,
            shadowRegionWeight = 0f,
            mertensContrastMean = 0f,
            mertensSaturationMean = 0f,
            mertensWellExposednessMean = 0f,
            sampleCount = 0,
            sampleStep = 1
        )
    }

    private fun exposureFusionToneStatsFromGpu(stats: GpuRawAutoExposureToneStats): ExposureFusionToneStats {
        if (stats.sampleCount <= 0) {
            return emptyExposureFusionToneStats()
        }

        val highlightFusionDelta = if (stats.highlightWeightSum > MERTENS_MIN_TONE_WEIGHT) {
            sqrt(stats.highlightDeltaEnergySum / stats.highlightWeightSum)
                .toFloat()
                .coerceAtLeast(0f)
        } else {
            0f
        }
        val shadowFusionDelta = if (stats.shadowWeightSum > MERTENS_MIN_TONE_WEIGHT) {
            sqrt(stats.shadowDeltaEnergySum / stats.shadowWeightSum)
                .toFloat()
                .coerceAtLeast(0f)
        } else {
            0f
        }
        val sampleCount = stats.sampleCount.toDouble().coerceAtLeast(1.0)
        return ExposureFusionToneStats(
            highlightFusionDelta = highlightFusionDelta,
            shadowFusionDelta = shadowFusionDelta,
            highlightRegionWeight = (stats.highlightWeightSum / sampleCount).toFloat().coerceIn(0f, 1f),
            shadowRegionWeight = (stats.shadowWeightSum / sampleCount).toFloat().coerceIn(0f, 1f),
            mertensContrastMean = (stats.mertensContrastSum / sampleCount).toFloat().coerceAtLeast(0f),
            mertensSaturationMean = (stats.mertensSaturationSum / sampleCount).toFloat().coerceAtLeast(0f),
            mertensWellExposednessMean = (stats.mertensWellExposednessSum / sampleCount).toFloat().coerceIn(0f, 1f),
            sampleCount = stats.sampleCount,
            sampleStep = stats.sampleStep
        )
    }

    private fun calculateMertensToneSampleStep(pixelCount: Int): Int {
        if (pixelCount <= MERTENS_TONE_STATS_MAX_SAMPLES) {
            return 1
        }

        var step = 1
        while ((pixelCount + step * step - 1) / (step * step) > MERTENS_TONE_STATS_MAX_SAMPLES) {
            step++
        }
        return step
    }

    private fun percentileFromLogHistogram(
        histogram: IntArray,
        percentile: Float,
        logMin: Float,
        logMax: Float
    ): Float {
        if (histogram.isEmpty()) {
            return 0f
        }

        var total = 0L
        for (count in histogram) {
            if (count > 0) {
                total += count.toLong()
            }
        }
        if (total <= 0L) {
            return 0f
        }

        val targetIndex = ((total - 1L) * percentile.coerceIn(0f, 1f)).toLong()
        var cumulative = 0L
        for (i in histogram.indices) {
            val count = histogram[i]
            if (count <= 0) continue
            cumulative += count.toLong()
            if (cumulative > targetIndex) {
                val fraction = (i + 0.5f) / histogram.size.toFloat()
                val logLuma = logMin + (logMax - logMin) * fraction
                return exp2(logLuma).coerceIn(0f, MAX_LINEAR_LUMA)
            }
        }

        return exp2(logMax).coerceIn(0f, MAX_LINEAR_LUMA)
    }

    private fun calculateAutoHighlightCompression(baselineExposure: Float, highlightFusionDelta: Float): Float {
        val min = 0.3f * baselineExposure.coerceIn(0f, 3f)
        return highlightFusionDelta.coerceIn(min, RAW_AUTO_HIGHLIGHT_LIMIT)
    }

    private fun calculateAutoShadowLift(
        baselineExposure: Float,
        stats: ExposureFusionToneStats
    ): Float {
        val min = 0.15f * baselineExposure.coerceIn(0f, 3f)
        return stats.shadowFusionDelta
            .coerceIn(min, RAW_AUTO_STANDARD_SHADOW_LIMIT)
    }

    private fun calculateLowKeyShadowProtection(
        compensatedP75: Float,
        compensatedP90: Float,
        targetLuma: Float
    ): Float {
        val safeTarget = targetLuma.coerceAtLeast(LUMA_FLOOR)
        val upperMidDarkness = 1f - smoothStep(
            RAW_AUTO_LOW_KEY_P75_RATIO_FULL,
            RAW_AUTO_LOW_KEY_P75_RATIO_START,
            compensatedP75 / safeTarget
        )
        val highToneDarkness = 1f - smoothStep(
            RAW_AUTO_LOW_KEY_P90_RATIO_FULL,
            RAW_AUTO_LOW_KEY_P90_RATIO_START,
            compensatedP90 / safeTarget
        )
        return minOf(upperMidDarkness, highToneDarkness).coerceIn(0f, 1f)
    }

    private fun applyLowKeyShadowProtection(
        shadows: Float,
        protection: Float,
        shadowLimit: Float
    ): Float {
        val attenuation = lerp(
            start = 1f,
            end = 1f - RAW_AUTO_LOW_KEY_MAX_SHADOW_ATTENUATION,
            fraction = protection
        )
        return (shadows * attenuation).coerceIn(0f, shadowLimit.coerceAtLeast(0f))
    }

    private fun percentile(sortedValues: FloatArray, percentile: Float): Float {
        if (sortedValues.isEmpty()) {
            return 0f
        }

        val index = ((sortedValues.size - 1) * percentile.coerceIn(0f, 1f)).toInt()
        return sortedValues[index]
    }

    private fun evDifference(brighter: Float, darker: Float): Float {
        val safeBrighter = sanitizeLinearLuma(brighter).coerceAtLeast(LUMA_FLOOR)
        val safeDarker = sanitizeLinearLuma(darker).coerceAtLeast(1e-5f)
        return log2(safeBrighter / safeDarker)
    }

    private fun log2(value: Float): Float {
        return (ln(value.toDouble()) / ln(2.0)).toFloat()
    }

    private fun exp2(value: Float): Float {
        return exp(value * ln(2.0)).toFloat()
    }

    private fun exposureScaleFromEv(ev: Float): Float {
        return if (ev.isFinite()) {
            exp2(ev.coerceIn(-MAX_METERING_COMPENSATION_EV, MAX_METERING_COMPENSATION_EV))
        } else {
            1f
        }
    }

    private fun resolveDisplayTargetLuma(meteringCompensationEv: Float): Float {
        return DISPLAY_TARGET_LUMA * exp2(-sanitizeMeteringCompensationEv(meteringCompensationEv))
    }

    private fun sanitizeMeteringCompensationEv(meteringCompensationEv: Float): Float {
        return if (meteringCompensationEv.isFinite()) {
            meteringCompensationEv.coerceIn(-MAX_METERING_COMPENSATION_EV, MAX_METERING_COMPENSATION_EV)
        } else {
            RAW_COLOR_ENGINE_METERING_COMPENSATION_EV
        }
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
