package com.hinnka.mycamera.raw

import android.annotation.SuppressLint
import android.util.Half
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.ShortBuffer
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 评价测光与场景分析系统
 */
object MeteringSystem {
    private const val TAG = "MeteringSystem"
    private const val DISPLAY_TARGET_LUMA = 0.18f
    private const val ACR3_AVERAGE_TONE_CURVE_EV = 1f
    private const val MID_TONE_GRID_COLUMNS = 7
    private const val MID_TONE_GRID_ROWS = 7
    private const val MID_TONE_ZONE_SAMPLE_TRIM_FRACTION = 0.15f
    private const val MID_TONE_ZONE_REJECT_FRACTION = 0.15f
    private const val MID_TONE_BUCKET_COUNT = 9
    private const val MID_TONE_MIN_BUCKET_RANGE_EV = 0.35f
    private const val MID_TONE_MIN_WEIGHT = 0.001f
    private const val LUMA_FLOOR = 0.001f
    private const val MAX_LINEAR_LUMA = 16.0f
    private const val MIN_DEPTH_SEPARATION = 0.08f
    private const val RAW_CURVE_NEUTRAL_WHITE_POINT = 1.0f
    private const val AUTO_DEVELOP_EPSILON = 0.0001f
    private const val AUTO_CLIP_FRACTION = 0.0002f
    private const val RAW_AUTO_HIGHLIGHT_LIMIT = 0.95f
    private const val RAW_AUTO_STANDARD_SHADOW_LIMIT = 0.45f
    private const val RAW_AUTO_HDR_SHADOW_LIMIT = 0.9f
    private const val RAW_AUTO_HDR_BASELINE_START = 1.0f
    private const val RAW_AUTO_HDR_BASELINE_FULL = 2.0f
    private const val RAW_AUTO_LOW_KEY_P75_RATIO_FULL = 0.45f
    private const val RAW_AUTO_LOW_KEY_P75_RATIO_START = 0.85f
    private const val RAW_AUTO_LOW_KEY_P90_RATIO_FULL = 0.70f
    private const val RAW_AUTO_LOW_KEY_P90_RATIO_START = 1.25f
    private const val RAW_AUTO_LOW_KEY_MAX_SHADOW_ATTENUATION = 0.78f
    private const val MERTENS_WELL_EXPOSED_CENTER = 0.5f
    private const val MERTENS_WELL_EXPOSED_SIGMA = 0.2f
    private const val MERTENS_TARGET_TO_MID_INTENSITY_SCALE = 2f
    private const val MERTENS_SHADOW_MASK_START = 0.04f
    private const val MERTENS_SHADOW_MASK_END = 0.45f
    private const val MERTENS_HIGHLIGHT_MASK_START = 0.55f
    private const val MERTENS_HIGHLIGHT_MASK_END = 0.92f
    private const val MERTENS_CONTRAST_WEIGHT_EXPONENT = 1f
    private const val MERTENS_SATURATION_WEIGHT_EXPONENT = 1f
    private const val MERTENS_EXPOSURE_WEIGHT_EXPONENT = 1f
    private const val MERTENS_MIN_TONE_WEIGHT = 0.0001f
    private const val MERTENS_TONE_STATS_MAX_SAMPLES = 4096
    private val MERTENS_VIRTUAL_EXPOSURE_EVS = floatArrayOf(-2f, -1f, 0f, 1f, 2f)


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
        fun scaleLuma(scale: Float): MeteringResult {
            val safeScale = if (scale.isFinite() && scale > 0f) scale else 1f
            return copy(
                avgLuma = (avgLuma * safeScale).coerceIn(0f, MAX_LINEAR_LUMA),
                clipLow = (clipLow * safeScale).coerceIn(0f, MAX_LINEAR_LUMA),
                clipHigh = (clipHigh * safeScale).coerceIn(0f, MAX_LINEAR_LUMA)
            )
        }

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
        val targetLuma: Float,
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

    private data class DepthWeightMap(
        val weights: FloatArray,
        val separation: Float,
        val enabled: Boolean
    )

    private data class MidToneZone(
        val luma: Float,
        val weight: Float
    )

    private data class MidToneReference(
        val luma: Float,
        val zoneMedianLuma: Float,
        val bucketMedianLuma: Float,
        val retainedZoneCount: Int,
        val retainedBucketCount: Int
    )

    private data class MidToneBucket(
        var weightedLogLumaSum: Double = 0.0,
        var weightSum: Double = 0.0,
        var zoneCount: Int = 0
    )

    private data class MidToneSample(
        val luma: Float,
        val weight: Float
    )

    private data class LinearRgbImage(
        val width: Int,
        val height: Int,
        val red: FloatArray,
        val green: FloatArray,
        val blue: FloatArray,
        val lumas: FloatArray,
        val sanitizedSampleCount: Int
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

    private data class MertensVirtualExposureResult(
        val intensity: Float,
        val contrastMean: Float,
        val saturationMean: Float,
        val wellExposednessMean: Float
    )

    private data class MertensFusionWeight(
        val weight: Float,
        val contrast: Float,
        val saturation: Float,
        val wellExposedness: Float
    )

    @SuppressLint("HalfFloat")
    fun analyzeLinearHalfFloatExposureEv(
        pixelBuffer: ShortBuffer,
        width: Int,
        height: Int,
        weightBuffer: ByteBuffer? = null, // Optional weight mask (e.g. depth map)
        baselineExposure: Float = 0f
    ): MeteringResult {
        val source = decodeLinearHalfFloatRgbImage(
            pixelBuffer = pixelBuffer,
            width = width,
            height = height
        )
        return analyzeExposureEv(
            width = width,
            height = height,
            weightBuffer = weightBuffer,
            targetLuma = DISPLAY_TARGET_LUMA * 2f.pow(-ACR3_AVERAGE_TONE_CURVE_EV),
            baselineExposure = baselineExposure,
            tag = "Linear RAW AE",
            source = source
        )
    }

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
        tag: String = "Linear RAW AE GPU"
    ): GpuRawAutoExposurePlan? {
        if (stats.histogram.isEmpty() || stats.sampleCount <= 0) {
            return null
        }

        val targetLuma = DISPLAY_TARGET_LUMA * 2f.pow(-ACR3_AVERAGE_TONE_CURVE_EV)
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

        val highlightAnchorGain = maxOf(1f, clipHigh) / clipHigh.coerceAtLeast(0.01f)
        val midToneGain = targetLuma / midToneLuma.coerceAtLeast(LUMA_FLOOR)
        val dynamicRangeGap = evDifference(clipHigh, clipLow)
        val extra = smoothStep(4f, 12f, dynamicRangeGap)
        val adaptiveGain = lerp(midToneGain * 1.2f, highlightAnchorGain * 1.1f, extra)
        val rawMeteredEv = log2(adaptiveGain.coerceIn(0.25f, 4.0f))
        val compensatedExposureScale = exp2(rawMeteredEv)

        val sampleStep = calculateMertensToneSampleStep(stats.sampleCount)
        return GpuRawAutoExposurePlan(
            compensatedExposureScale = compensatedExposureScale,
            sampleStep = sampleStep,
            clipLow = clipLow,
            clipHigh = clipHigh,
            p05 = p05,
            p25 = p25,
            p75 = p75,
            p90 = p90,
            p99 = p99,
            compensatedClipLow = (clipLow * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedClipHigh = (clipHigh * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP05 = (p05 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP25 = (p25 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP75 = (p75 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP90 = (p90 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            compensatedP99 = (p99 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA),
            baselineExposure = baselineExposure,
            targetLuma = targetLuma,
            midToneLuma = midToneLuma,
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
        val highlights = -calculateAutoHighlightCompression(exposureFusionToneStats.highlightFusionDelta)
        val shadowLimit = resolveAutoShadowLimit(plan.baselineExposure)
        val unprotectedShadows = calculateAutoShadowLift(
            stats = exposureFusionToneStats,
            shadowLimit = shadowLimit
        )
        val lowKeyShadowProtection = calculateLowKeyShadowProtection(
            compensatedP75 = plan.compensatedP75,
            compensatedP90 = plan.compensatedP90,
            targetLuma = plan.targetLuma
        )
        val shadows = applyLowKeyShadowProtection(
            shadows = unprotectedShadows,
            protection = lowKeyShadowProtection,
            shadowLimit = shadowLimit
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
                "compScale=${plan.compensatedExposureScale} " +
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
                "baselineExposure=${plan.baselineExposure} shadowLimit=$shadowLimit " +
                "highlights=$highlights shadows=$shadows unprotectedShadows=$unprotectedShadows " +
                "target=${plan.targetLuma} midToneLuma=${plan.midToneLuma} " +
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

    @SuppressLint("HalfFloat")
    private fun decodeLinearHalfFloatRgbImage(
        pixelBuffer: ShortBuffer,
        width: Int,
        height: Int
    ): LinearRgbImage {
        val pixelCount = (width * height).coerceAtLeast(0)
        val red = FloatArray(pixelCount)
        val green = FloatArray(pixelCount)
        val blue = FloatArray(pixelCount)
        val lumas = FloatArray(pixelCount)
        var sanitizedSampleCount = 0

        for (index in 0 until pixelCount) {
            val base = index * 4
            val rawR = Half.toFloat(pixelBuffer.get(base))
            val rawG = Half.toFloat(pixelBuffer.get(base + 1))
            val rawB = Half.toFloat(pixelBuffer.get(base + 2))
            val r = sanitizeLinearLuma(rawR)
            val g = sanitizeLinearLuma(rawG)
            val b = sanitizeLinearLuma(rawB)
            val luma = sanitizeLinearLuma(r * 0.2126f + g * 0.7152f + b * 0.0722f)

            if (rawR != r || rawG != g || rawB != b) {
                sanitizedSampleCount++
            }
            red[index] = r
            green[index] = g
            blue[index] = b
            lumas[index] = luma
        }

        return LinearRgbImage(
            width = width,
            height = height,
            red = red,
            green = green,
            blue = blue,
            lumas = lumas,
            sanitizedSampleCount = sanitizedSampleCount
        )
    }

    private fun analyzeExposureEv(
        width: Int,
        height: Int,
        weightBuffer: ByteBuffer?,
        targetLuma: Float,
        baselineExposure: Float,
        tag: String,
        source: LinearRgbImage
    ): MeteringResult {
        val pixelCount = width * height
        if (pixelCount == 0 || source.lumas.isEmpty()) return MeteringResult.EMPTY

        val depthWeights = decodeDepthWeights(weightBuffer, pixelCount)

        val midToneReference = calculateMidToneReferenceLuma(
            lumas = source.lumas,
            width = width,
            height = height,
            depthWeights = depthWeights?.weights,
            fallback = targetLuma
        )
        val midToneLuma = midToneReference.luma

        val sortedLumas = source.lumas.copyOf()
        sortedLumas.sort()
        val clipLow = percentile(sortedLumas, AUTO_CLIP_FRACTION)
        val p05 = percentile(sortedLumas, 0.05f)
        val p25 = percentile(sortedLumas, 0.25f)
        val p75 = percentile(sortedLumas, 0.75f)
        val p90 = percentile(sortedLumas, 0.90f)
        val p99 = percentile(sortedLumas, 0.99f)
        val clipHigh = percentile(sortedLumas, 1f - AUTO_CLIP_FRACTION)

        val highlightAnchorGain = maxOf(1f, clipHigh) / clipHigh.coerceAtLeast(0.01f)
        val midToneGain = targetLuma / midToneLuma.coerceAtLeast(LUMA_FLOOR)
        val dynamicRangeGap = evDifference(clipHigh, clipLow)

        val extra = smoothStep(4f, 12f, dynamicRangeGap)
        val adaptiveGain = lerp(midToneGain * 1.2f, highlightAnchorGain * 1.1f, extra)
        val rawMeteredEv = log2(adaptiveGain.coerceIn(0.25f, 4.0f))

        val compensatedExposureScale = exp2(rawMeteredEv)
        val compensatedClipLow = (clipLow * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val compensatedP05 = (p05 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val compensatedP25 = (p25 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val compensatedP75 = (p75 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val compensatedP90 = (p90 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val compensatedP99 = (p99 * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val compensatedClipHigh = (clipHigh * compensatedExposureScale).coerceIn(0f, MAX_LINEAR_LUMA)
        val exposureFusionToneStats = calculateExposureFusionToneStats(
            source = source,
            exposureScale = compensatedExposureScale,
            targetLuma = targetLuma
        )
        val highlights = -calculateAutoHighlightCompression(exposureFusionToneStats.highlightFusionDelta)
        val shadowLimit = resolveAutoShadowLimit(baselineExposure)
        val unprotectedShadows = calculateAutoShadowLift(
            stats = exposureFusionToneStats,
            shadowLimit = shadowLimit
        )
        val lowKeyShadowProtection = calculateLowKeyShadowProtection(
            compensatedP75 = compensatedP75,
            compensatedP90 = compensatedP90,
            targetLuma = targetLuma
        )
        val shadows = applyLowKeyShadowProtection(
            shadows = unprotectedShadows,
            protection = lowKeyShadowProtection,
            shadowLimit = shadowLimit
        )

        // Curve points
        val curveWhitePoint = if (compensatedClipHigh > 1f) {
            compensatedClipHigh.coerceIn(1.05f, 4.0f)
        } else {
            RAW_CURVE_NEUTRAL_WHITE_POINT
        }

        PLog.d(
            TAG,
            "$tag: clipLow=$clipLow clipHigh=$clipHigh " +
                "p05=$p05 p25=$p25 p75=$p75 p90=$p90 p99=$p99 " +
                "compScale=$compensatedExposureScale " +
                "compClipLow=$compensatedClipLow compClipHigh=$compensatedClipHigh " +
                "compP05=$compensatedP05 compP25=$compensatedP25 " +
                "compP75=$compensatedP75 compP90=$compensatedP90 compP99=$compensatedP99 " +
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
                "baselineExposure=$baselineExposure shadowLimit=$shadowLimit " +
                "highlights=$highlights shadows=$shadows unprotectedShadows=$unprotectedShadows " +
                "target=$targetLuma midToneLuma=$midToneLuma " +
                "midToneGain=$midToneGain highlightAnchorGain=$highlightAnchorGain gain=$adaptiveGain " +
                "ev=$rawMeteredEv gap=$dynamicRangeGap " +
                "sanitizedSamples=${source.sanitizedSampleCount}"
        )

        return MeteringResult(
            meteredEv = rawMeteredEv.coerceIn(-2f, 2f),
            dynamicRangeGap = dynamicRangeGap,
            avgLuma = midToneLuma,
            clipLow = clipLow,
            clipHigh = clipHigh,
            highlights = highlights,
            shadows = shadows,
            curveWhitePoint = curveWhitePoint
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

    private fun calculateMidToneReferenceLuma(
        lumas: FloatArray,
        width: Int,
        height: Int,
        depthWeights: FloatArray?,
        fallback: Float
    ): MidToneReference {
        if (lumas.isEmpty() || width <= 0 || height <= 0) {
            return fallbackMidToneReference(fallback)
        }

        val zones = ArrayList<MidToneZone>(MID_TONE_GRID_COLUMNS * MID_TONE_GRID_ROWS)
        for (zoneY in 0 until MID_TONE_GRID_ROWS) {
            val startY = zoneY * height / MID_TONE_GRID_ROWS
            val endY = (zoneY + 1) * height / MID_TONE_GRID_ROWS
            if (startY >= endY) continue

            for (zoneX in 0 until MID_TONE_GRID_COLUMNS) {
                val startX = zoneX * width / MID_TONE_GRID_COLUMNS
                val endX = (zoneX + 1) * width / MID_TONE_GRID_COLUMNS
                if (startX >= endX) continue

                val zoneLuma = calculateZoneTrimmedMeanLuma(lumas, width, startX, endX, startY, endY, fallback)
                val centerU = ((startX + endX) * 0.5f) / width
                val centerV = ((startY + endY) * 0.5f) / height
                zones += MidToneZone(
                    luma = zoneLuma,
                    weight = calculateMidToneZoneWeight(
                        depthWeights = depthWeights,
                        width = width,
                        startX = startX,
                        endX = endX,
                        startY = startY,
                        endY = endY,
                        centerU = centerU,
                        centerV = centerV
                    )
                )
            }
        }

        if (zones.isEmpty()) {
            return fallbackMidToneReference(fallback)
        }

        zones.sortBy { it.luma }
        val rejectZoneCount = (zones.size * MID_TONE_ZONE_REJECT_FRACTION)
            .toInt()
            .coerceAtMost((zones.size - 1) / 2)
        val startIndex = rejectZoneCount
        val endIndex = zones.size - rejectZoneCount
        if (startIndex >= endIndex) {
            val fallbackLuma = sanitizeAverageLuma(zones[zones.size / 2].luma, fallback)
            return MidToneReference(
                luma = fallbackLuma,
                zoneMedianLuma = fallbackLuma,
                bucketMedianLuma = fallbackLuma,
                retainedZoneCount = 1,
                retainedBucketCount = 1
            )
        }

        val zoneMedianLuma = sanitizeAverageLuma(
            weightedMedianMidToneLuma(zones, startIndex, endIndex),
            fallback
        )
        val bucketSamples = buildToneBalancedMidToneSamples(zones, startIndex, endIndex)
        val bucketMedianLuma = sanitizeAverageLuma(
            weightedMedianMidToneSampleLuma(bucketSamples).takeIf { it > 0f } ?: zoneMedianLuma,
            fallback
        )

        return MidToneReference(
            luma = bucketMedianLuma,
            zoneMedianLuma = zoneMedianLuma,
            bucketMedianLuma = bucketMedianLuma,
            retainedZoneCount = endIndex - startIndex,
            retainedBucketCount = bucketSamples.size
        )
    }

    private fun fallbackMidToneReference(fallback: Float): MidToneReference {
        return MidToneReference(
            luma = fallback,
            zoneMedianLuma = fallback,
            bucketMedianLuma = fallback,
            retainedZoneCount = 0,
            retainedBucketCount = 0
        )
    }

    private fun calculateZoneTrimmedMeanLuma(
        lumas: FloatArray,
        width: Int,
        startX: Int,
        endX: Int,
        startY: Int,
        endY: Int,
        fallback: Float,
    ): Float {
        val sampleCount = (endX - startX) * (endY - startY)
        if (sampleCount <= 0) {
            return fallback
        }

        val samples = FloatArray(sampleCount)
        var sampleIndex = 0
        for (y in startY until endY) {
            val rowOffset = y * width
            for (x in startX until endX) {
                samples[sampleIndex++] = lumas[rowOffset + x]
            }
        }

        return trimmedMean(samples, MID_TONE_ZONE_SAMPLE_TRIM_FRACTION, fallback)
    }

    private fun trimmedMean(
        values: FloatArray,
        trimFraction: Float,
        fallback: Float
    ): Float {
        if (values.isEmpty()) {
            return fallback
        }

        values.sort()
        val trimCount = (values.size * trimFraction.coerceIn(0f, 0.49f))
            .toInt()
            .coerceAtMost((values.size - 1) / 2)
        val startIndex = trimCount
        val endIndex = values.size - trimCount
        var sum = 0.0
        for (i in startIndex until endIndex) {
            sum += values[i].toDouble()
        }

        return (sum / (endIndex - startIndex).coerceAtLeast(1)).toFloat()
    }

    private fun weightedMedianMidToneLuma(
        sortedZones: List<MidToneZone>,
        startIndex: Int,
        endIndex: Int
    ): Float {
        var totalWeight = 0.0
        for (i in startIndex until endIndex) {
            val weight = sortedZones[i].weight
            if (weight.isFinite() && weight > 0f) {
                totalWeight += weight.toDouble()
            }
        }

        if (totalWeight <= 0.0) {
            return sortedZones[(startIndex + endIndex - 1) / 2].luma
        }

        val halfWeight = totalWeight * 0.5
        var cumulativeWeight = 0.0
        for (i in startIndex until endIndex) {
            val weight = sortedZones[i].weight
            if (weight.isFinite() && weight > 0f) {
                cumulativeWeight += weight.toDouble()
            }
            if (cumulativeWeight >= halfWeight) {
                return sortedZones[i].luma
            }
        }

        return sortedZones[endIndex - 1].luma
    }

    private fun buildToneBalancedMidToneSamples(
        sortedZones: List<MidToneZone>,
        startIndex: Int,
        endIndex: Int
    ): List<MidToneSample> {
        if (startIndex >= endIndex) {
            return emptyList()
        }

        val lowLogLuma = log2(sortedZones[startIndex].luma.coerceAtLeast(LUMA_FLOOR))
        val highLogLuma = log2(sortedZones[endIndex - 1].luma.coerceAtLeast(LUMA_FLOOR))
        val logRange = highLogLuma - lowLogLuma
        if (!logRange.isFinite() || logRange < MID_TONE_MIN_BUCKET_RANGE_EV) {
            return listOf(
                MidToneSample(
                    luma = weightedGeometricMeanMidToneLuma(sortedZones, startIndex, endIndex),
                    weight = 1f
                )
            )
        }

        val buckets = Array(MID_TONE_BUCKET_COUNT) { MidToneBucket() }
        for (i in startIndex until endIndex) {
            val zone = sortedZones[i]
            val weight = zone.weight
            if (weight.isFinite() && weight > 0f) {
                val logLuma = log2(zone.luma.coerceAtLeast(LUMA_FLOOR))
                val bucketIndex = (((logLuma - lowLogLuma) / logRange) * MID_TONE_BUCKET_COUNT)
                    .toInt()
                    .coerceIn(0, MID_TONE_BUCKET_COUNT - 1)
                val bucket = buckets[bucketIndex]
                bucket.weightedLogLumaSum += (logLuma * weight).toDouble()
                bucket.weightSum += weight.toDouble()
                bucket.zoneCount++
            }
        }

        val samples = ArrayList<MidToneSample>(MID_TONE_BUCKET_COUNT)
        for (bucket in buckets) {
            if (bucket.zoneCount > 0 && bucket.weightSum > 0.0) {
                val representativeLogLuma = (bucket.weightedLogLumaSum / bucket.weightSum).toFloat()
                samples += MidToneSample(
                    luma = exp2(representativeLogLuma),
                    weight = (bucket.weightSum / bucket.zoneCount).toFloat().coerceAtLeast(MID_TONE_MIN_WEIGHT)
                )
            }
        }
        return samples
    }

    private fun weightedMedianMidToneSampleLuma(samples: List<MidToneSample>): Float {
        if (samples.isEmpty()) {
            return 0f
        }

        var totalWeight = 0.0
        for (sample in samples) {
            if (sample.weight.isFinite() && sample.weight > 0f) {
                totalWeight += sample.weight.toDouble()
            }
        }

        if (totalWeight <= 0.0) {
            return samples[samples.size / 2].luma
        }

        val halfWeight = totalWeight * 0.5
        var cumulativeWeight = 0.0
        for (sample in samples) {
            if (sample.weight.isFinite() && sample.weight > 0f) {
                cumulativeWeight += sample.weight.toDouble()
            }
            if (cumulativeWeight >= halfWeight) {
                return sample.luma
            }
        }

        return samples.last().luma
    }

    private fun weightedGeometricMeanMidToneLuma(
        sortedZones: List<MidToneZone>,
        startIndex: Int,
        endIndex: Int
    ): Float {
        var weightedLogSum = 0.0
        var totalWeight = 0.0
        for (i in startIndex until endIndex) {
            val zone = sortedZones[i]
            val weight = zone.weight
            if (weight.isFinite() && weight > 0f) {
                weightedLogSum += (log2(zone.luma.coerceAtLeast(LUMA_FLOOR)) * weight).toDouble()
                totalWeight += weight.toDouble()
            }
        }

        return if (totalWeight > 0.0) {
            exp2((weightedLogSum / totalWeight).toFloat())
        } else {
            sortedZones[(startIndex + endIndex - 1) / 2].luma
        }
    }

    private fun calculateMidToneZoneWeight(
        depthWeights: FloatArray?,
        width: Int,
        startX: Int,
        endX: Int,
        startY: Int,
        endY: Int,
        centerU: Float,
        centerV: Float
    ): Float {
        val centerWeight = calculateMidToneCenterWeight(centerU, centerV)
        val depthWeight = calculateZoneDepthWeight(depthWeights, width, startX, endX, startY, endY)
        val weight = centerWeight * depthWeight
        return if (weight.isFinite()) {
            weight.coerceAtLeast(MID_TONE_MIN_WEIGHT)
        } else {
            MID_TONE_MIN_WEIGHT
        }
    }

    private fun calculateZoneDepthWeight(
        depthWeights: FloatArray?,
        width: Int,
        startX: Int,
        endX: Int,
        startY: Int,
        endY: Int
    ): Float {
        if (depthWeights == null) {
            return 1f
        }

        var sum = 0.0
        var count = 0
        for (y in startY until endY) {
            val rowOffset = y * width
            for (x in startX until endX) {
                val weight = depthWeights.getOrNull(rowOffset + x) ?: continue
                if (weight.isFinite() && weight > 0f) {
                    sum += weight.toDouble()
                    count++
                }
            }
        }

        return if (count > 0) {
            (sum / count).toFloat().coerceAtLeast(MID_TONE_MIN_WEIGHT)
        } else {
            1f
        }
    }

    private fun calculateMidToneCenterWeight(u: Float, v: Float): Float {
        val centerWeight = gaussian2d(u, v, 0.5f, 0.5f, 0.32f)
        return 0.35f + centerWeight * 0.65f
    }

    private fun sanitizeAverageLuma(value: Float, fallback: Float): Float {
        return if (value.isFinite() && value > 0f) {
            value.coerceIn(LUMA_FLOOR, MAX_LINEAR_LUMA)
        } else {
            fallback.coerceIn(LUMA_FLOOR, MAX_LINEAR_LUMA)
        }
    }

    private fun calculateExposureFusionToneStats(
        source: LinearRgbImage,
        exposureScale: Float,
        targetLuma: Float
    ): ExposureFusionToneStats {
        if (source.lumas.isEmpty()) {
            return emptyExposureFusionToneStats()
        }

        val safeExposureScale = if (exposureScale.isFinite() && exposureScale > 0f) {
            exposureScale
        } else {
            1f
        }
        val virtualExposureScales = buildMertensVirtualExposureScales(safeExposureScale)
        val sampleStep = calculateMertensToneSampleStep(source.lumas.size)
        val startOffset = sampleStep / 2
        var highlightDeltaEnergySum = 0.0
        var shadowDeltaEnergySum = 0.0
        var highlightWeightSum = 0.0
        var shadowWeightSum = 0.0
        var mertensContrastSum = 0.0
        var mertensSaturationSum = 0.0
        var mertensWellExposednessSum = 0.0
        var sampledCount = 0
        var y = startOffset.coerceAtMost(source.height - 1)
        while (y < source.height) {
            var x = startOffset.coerceAtMost(source.width - 1)
            val rowOffset = y * source.width
            while (x < source.width) {
                val index = rowOffset + x
                val baseIntensity = calculateExposureFusionIntensity(
                    source = source,
                    index = index,
                    exposureScale = safeExposureScale,
                    targetLuma = targetLuma
                )
                val fusionResult = calculateVirtualMertensFusionIntensity(
                    source = source,
                    index = index,
                    baseIntensity = baseIntensity,
                    virtualExposureScales = virtualExposureScales,
                    targetLuma = targetLuma
                )
                val fusionDelta = fusionResult.intensity - baseIntensity
                val highlightMask = smoothStep(
                    MERTENS_HIGHLIGHT_MASK_START,
                    MERTENS_HIGHLIGHT_MASK_END,
                    baseIntensity
                )
                val shadowMask = 1f - smoothStep(
                    MERTENS_SHADOW_MASK_START,
                    MERTENS_SHADOW_MASK_END,
                    baseIntensity
                )
                val highlightFusionDelta = if (fusionDelta < 0f) {
                    -fusionDelta
                } else {
                    0f
                }
                val shadowFusionDelta = if (fusionDelta > 0f) {
                    fusionDelta
                } else {
                    0f
                }
                highlightDeltaEnergySum += (highlightFusionDelta * highlightFusionDelta * highlightMask).toDouble()
                shadowDeltaEnergySum += (shadowFusionDelta * shadowFusionDelta * shadowMask).toDouble()
                highlightWeightSum += highlightMask.toDouble()
                shadowWeightSum += shadowMask.toDouble()
                mertensContrastSum += fusionResult.contrastMean.toDouble()
                mertensSaturationSum += fusionResult.saturationMean.toDouble()
                mertensWellExposednessSum += fusionResult.wellExposednessMean.toDouble()
                sampledCount++
                x += sampleStep
            }
            y += sampleStep
        }

        val sampleCount = sampledCount.toDouble().coerceAtLeast(1.0)
        val highlightFusionDelta = if (highlightWeightSum > MERTENS_MIN_TONE_WEIGHT) {
            sqrt(highlightDeltaEnergySum / highlightWeightSum)
                .toFloat()
                .coerceAtLeast(0f)
        } else {
            0f
        }
        val shadowFusionDelta = if (shadowWeightSum > MERTENS_MIN_TONE_WEIGHT) {
            sqrt(shadowDeltaEnergySum / shadowWeightSum)
                .toFloat()
                .coerceAtLeast(0f)
        } else {
            0f
        }

        return ExposureFusionToneStats(
            highlightFusionDelta = highlightFusionDelta,
            shadowFusionDelta = shadowFusionDelta,
            highlightRegionWeight = (highlightWeightSum / sampleCount).toFloat().coerceIn(0f, 1f),
            shadowRegionWeight = (shadowWeightSum / sampleCount).toFloat().coerceIn(0f, 1f),
            mertensContrastMean = (mertensContrastSum / sampleCount).toFloat().coerceAtLeast(0f),
            mertensSaturationMean = (mertensSaturationSum / sampleCount).toFloat().coerceAtLeast(0f),
            mertensWellExposednessMean = (mertensWellExposednessSum / sampleCount).toFloat().coerceIn(0f, 1f),
            sampleCount = sampledCount,
            sampleStep = sampleStep
        )
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

    private fun buildMertensVirtualExposureScales(exposureScale: Float): FloatArray {
        val scales = FloatArray(MERTENS_VIRTUAL_EXPOSURE_EVS.size)
        for (i in MERTENS_VIRTUAL_EXPOSURE_EVS.indices) {
            scales[i] = exposureScale * exp2(MERTENS_VIRTUAL_EXPOSURE_EVS[i])
        }
        return scales
    }

    private fun calculateVirtualMertensFusionIntensity(
        source: LinearRgbImage,
        index: Int,
        baseIntensity: Float,
        virtualExposureScales: FloatArray,
        targetLuma: Float
    ): MertensVirtualExposureResult {
        var weightedIntensitySum = 0.0
        var weightSum = 0.0
        var contrastSum = 0.0
        var saturationSum = 0.0
        var wellExposednessSum = 0.0
        for (virtualExposureScale in virtualExposureScales) {
            val virtualIntensity = calculateExposureFusionIntensity(
                source = source,
                index = index,
                exposureScale = virtualExposureScale,
                targetLuma = targetLuma
            )
            val weight = calculateMertensFusionWeight(
                source = source,
                index = index,
                exposureScale = virtualExposureScale,
                targetLuma = targetLuma
            )
            weightedIntensitySum += (virtualIntensity * weight.weight).toDouble()
            weightSum += weight.weight.toDouble()
            contrastSum += weight.contrast.toDouble()
            saturationSum += weight.saturation.toDouble()
            wellExposednessSum += weight.wellExposedness.toDouble()
        }

        val virtualExposureCount = MERTENS_VIRTUAL_EXPOSURE_EVS.size.toDouble().coerceAtLeast(1.0)
        val intensity = if (weightSum > 0.0) {
            (weightedIntensitySum / weightSum).toFloat().coerceIn(0f, 1f)
        } else {
            baseIntensity
        }

        return MertensVirtualExposureResult(
            intensity = intensity,
            contrastMean = (contrastSum / virtualExposureCount).toFloat().coerceAtLeast(0f),
            saturationMean = (saturationSum / virtualExposureCount).toFloat().coerceAtLeast(0f),
            wellExposednessMean = (wellExposednessSum / virtualExposureCount).toFloat().coerceIn(0f, 1f)
        )
    }

    private fun calculateMertensFusionWeight(
        source: LinearRgbImage,
        index: Int,
        exposureScale: Float,
        targetLuma: Float
    ): MertensFusionWeight {
        val r = linearLumaToExposureFusionIntensity(source.red[index] * exposureScale, targetLuma)
        val g = linearLumaToExposureFusionIntensity(source.green[index] * exposureScale, targetLuma)
        val b = linearLumaToExposureFusionIntensity(source.blue[index] * exposureScale, targetLuma)
        val contrast = calculateMertensContrast(
            source = source,
            index = index,
            exposureScale = exposureScale,
            targetLuma = targetLuma
        )
        val saturation = calculateMertensSaturation(r, g, b)
        val wellExposedness = (mertensWellExposedness(r) *
            mertensWellExposedness(g) *
            mertensWellExposedness(b))
            .coerceIn(0f, 1f)
        val contrastWeight = mertensWeightComponent(
            value = contrast,
            exponent = MERTENS_CONTRAST_WEIGHT_EXPONENT
        )
        val saturationWeight = mertensWeightComponent(
            value = saturation,
            exponent = MERTENS_SATURATION_WEIGHT_EXPONENT
        )
        val exposureWeight = mertensWeightComponent(
            value = wellExposedness,
            exponent = MERTENS_EXPOSURE_WEIGHT_EXPONENT
        )
        val weight = (contrastWeight * saturationWeight * exposureWeight)
            .takeIf { it.isFinite() && it > 0f }
            ?: 0f

        return MertensFusionWeight(
            weight = weight,
            contrast = contrast,
            saturation = saturation,
            wellExposedness = wellExposedness
        )
    }

    private fun mertensWeightComponent(
        value: Float,
        exponent: Float
    ): Float {
        if (exponent <= 0f) {
            return 1f
        }
        val safeValue = if (value.isFinite()) value.coerceAtLeast(0f) else 0f
        if (exponent == 1f) {
            return safeValue
        }
        return safeValue
            .pow(exponent)
            .takeIf { it.isFinite() && it >= 0f }
            ?: 0f
    }

    private fun calculateExposureFusionIntensity(
        source: LinearRgbImage,
        index: Int,
        exposureScale: Float,
        targetLuma: Float
    ): Float {
        val r = linearLumaToExposureFusionIntensity(source.red[index] * exposureScale, targetLuma)
        val g = linearLumaToExposureFusionIntensity(source.green[index] * exposureScale, targetLuma)
        val b = linearLumaToExposureFusionIntensity(source.blue[index] * exposureScale, targetLuma)
        return (r * 0.2126f + g * 0.7152f + b * 0.0722f).coerceIn(0f, 1f)
    }

    private fun calculateMertensContrast(
        source: LinearRgbImage,
        index: Int,
        exposureScale: Float,
        targetLuma: Float
    ): Float {
        if (source.width <= 0 || source.height <= 0) {
            return 0f
        }

        val x = index % source.width
        val y = index / source.width
        val center = calculateExposureFusionIntensityAt(source, x, y, exposureScale, targetLuma)
        val left = calculateExposureFusionIntensityAt(source, x - 1, y, exposureScale, targetLuma)
        val right = calculateExposureFusionIntensityAt(source, x + 1, y, exposureScale, targetLuma)
        val up = calculateExposureFusionIntensityAt(source, x, y - 1, exposureScale, targetLuma)
        val down = calculateExposureFusionIntensityAt(source, x, y + 1, exposureScale, targetLuma)
        return abs(left + right + up + down - center * 4f).coerceAtLeast(0f)
    }

    private fun calculateExposureFusionIntensityAt(
        source: LinearRgbImage,
        x: Int,
        y: Int,
        exposureScale: Float,
        targetLuma: Float
    ): Float {
        val sampleX = reflect101(x, source.width)
        val sampleY = reflect101(y, source.height)
        return calculateExposureFusionIntensity(
            source = source,
            index = sampleY * source.width + sampleX,
            exposureScale = exposureScale,
            targetLuma = targetLuma
        )
    }

    private fun calculateMertensSaturation(
        r: Float,
        g: Float,
        b: Float
    ): Float {
        val mean = (r + g + b) / 3f
        val dr = r - mean
        val dg = g - mean
        val db = b - mean
        return sqrt((dr * dr + dg * dg + db * db).toDouble())
            .toFloat()
            .coerceAtLeast(0f)
    }

    private fun reflect101(position: Int, size: Int): Int {
        if (size <= 1) {
            return 0
        }

        val period = size * 2 - 2
        var value = position
        if (value < 0) {
            value = -value
        }
        value %= period
        return if (value >= size) period - value else value
    }

    private fun linearLumaToExposureFusionIntensity(
        luma: Float,
        targetLuma: Float
    ): Float {
        val safeTarget = targetLuma.coerceAtLeast(LUMA_FLOOR)
        return (sanitizeLinearLuma(luma) / (safeTarget * MERTENS_TARGET_TO_MID_INTENSITY_SCALE))
            .coerceIn(0f, 1f)
    }

    private fun mertensWellExposedness(intensity: Float): Float {
        val offset = intensity.coerceIn(0f, 1f) - MERTENS_WELL_EXPOSED_CENTER
        val sigma2 = MERTENS_WELL_EXPOSED_SIGMA * MERTENS_WELL_EXPOSED_SIGMA
        return exp((-0.5f * offset * offset / sigma2).toDouble()).toFloat().coerceIn(0f, 1f)
    }

    private fun calculateAutoHighlightCompression(highlightFusionDelta: Float): Float {
        return highlightFusionDelta.coerceIn(0f, RAW_AUTO_HIGHLIGHT_LIMIT)
    }

    private fun resolveAutoShadowLimit(baselineExposure: Float): Float {
        val hdrAllowance = smoothStep(
            RAW_AUTO_HDR_BASELINE_START,
            RAW_AUTO_HDR_BASELINE_FULL,
            baselineExposure.takeIf { it.isFinite() } ?: 0f
        )
        return lerp(
            RAW_AUTO_STANDARD_SHADOW_LIMIT,
            RAW_AUTO_HDR_SHADOW_LIMIT,
            hdrAllowance
        )
    }

    private fun calculateAutoShadowLift(
        stats: ExposureFusionToneStats,
        shadowLimit: Float
    ): Float {
        return stats.shadowFusionDelta
            .coerceIn(0f, shadowLimit.coerceAtLeast(0f))
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
