package com.hinnka.mycamera.raw

import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

internal object DngAppleProRawExposureNormalizer {
    private const val TARGET_AVERAGE_EV = 0.7f
    private const val TARGET_TOLERANCE_EV = 0.05f
    private const val MIN_ADDITIONAL_OFFSET_EV = -2.75f
    private const val SOLVE_STEPS = 22
    private const val MIN_OUTPUT_FOR_EV = 1e-5f
    private const val MAX_OUTPUT_FOR_EV = 0.985f

    private const val DEFAULT_BLACK_RENDER_SHADOWS = 5f
    private const val DEFAULT_BLACK_RENDER_SHADOW_SCALE = 1f
    private const val DEFAULT_BLACK_RENDER_STAGE3_GAIN = 1f
    private const val PROFILE_HIGHLIGHT_SHOULDER_START = 0.58f
    private const val PROFILE_HIGHLIGHT_SHOULDER_SOFTNESS = 0.30f

    private val NEUTRAL_GRAY_SAMPLES = floatArrayOf(
        0.010f,
        0.015f,
        0.022f,
        0.032f,
        0.046f,
        0.065f,
        0.090f,
        0.125f,
        0.180f,
        0.250f,
        0.360f,
        0.500f,
        0.700f
    )

    data class Result(
        val additionalExposureOffsetEv: Float,
        val measuredAverageEv: Float,
        val normalizedAverageEv: Float,
        val targetAverageEv: Float
    )

    fun resolveExposureOffset(
        profileGainTableMap: DngProfileGainTableMap?,
        baselineExposureEv: Float,
        toneCurveLut: FloatArray?,
        existingExposureOffsetEv: Float = 0f,
        defaultBlackRender: DcpDefaultBlackRender = DcpDefaultBlackRender.None,
        shadowScale: Float = 1f,
        targetAverageEv: Float = TARGET_AVERAGE_EV
    ): Result {
        val target = targetAverageEv.takeIf { it.isFinite() } ?: TARGET_AVERAGE_EV
        val samples = neutralSamples(profileGainTableMap, baselineExposureEv)
        val sourceLut = toneCurveLut?.takeIf { it.isNotEmpty() }
        if (samples.isEmpty() || sourceLut == null) {
            return Result(
                additionalExposureOffsetEv = 0f,
                measuredAverageEv = Float.NaN,
                normalizedAverageEv = Float.NaN,
                targetAverageEv = target
            )
        }

        val baseOffset = existingExposureOffsetEv.takeIf { it.isFinite() } ?: 0f
        val measured = averageRenderedExposureEv(
            samples = samples,
            toneCurveLut = sourceLut,
            profileExposureOffsetEv = baseOffset,
            defaultBlackRender = defaultBlackRender,
            shadowScale = shadowScale
        ) ?: return Result(
            additionalExposureOffsetEv = 0f,
            measuredAverageEv = Float.NaN,
            normalizedAverageEv = Float.NaN,
            targetAverageEv = target
        )
        if (measured <= target + TARGET_TOLERANCE_EV) {
            return Result(
                additionalExposureOffsetEv = 0f,
                measuredAverageEv = measured,
                normalizedAverageEv = measured,
                targetAverageEv = target
            )
        }

        val darkestAverage = averageRenderedExposureEv(
            samples = samples,
            toneCurveLut = sourceLut,
            profileExposureOffsetEv = baseOffset + MIN_ADDITIONAL_OFFSET_EV,
            defaultBlackRender = defaultBlackRender,
            shadowScale = shadowScale
        ) ?: measured
        if (darkestAverage > target) {
            return Result(
                additionalExposureOffsetEv = MIN_ADDITIONAL_OFFSET_EV,
                measuredAverageEv = measured,
                normalizedAverageEv = darkestAverage,
                targetAverageEv = target
            )
        }

        var darkOffset = MIN_ADDITIONAL_OFFSET_EV
        var brightOffset = 0f
        repeat(SOLVE_STEPS) {
            val midOffset = (darkOffset + brightOffset) * 0.5f
            val midAverage = averageRenderedExposureEv(
                samples = samples,
                toneCurveLut = sourceLut,
                profileExposureOffsetEv = baseOffset + midOffset,
                defaultBlackRender = defaultBlackRender,
                shadowScale = shadowScale
            ) ?: measured
            if (midAverage > target) {
                brightOffset = midOffset
            } else {
                darkOffset = midOffset
            }
        }

        val additionalOffset = ((darkOffset + brightOffset) * 0.5f)
            .coerceIn(MIN_ADDITIONAL_OFFSET_EV, 0f)
            .let { if (it > -0.02f) 0f else it }
        val normalized = averageRenderedExposureEv(
            samples = samples,
            toneCurveLut = sourceLut,
            profileExposureOffsetEv = baseOffset + additionalOffset,
            defaultBlackRender = defaultBlackRender,
            shadowScale = shadowScale
        ) ?: measured
        return Result(
            additionalExposureOffsetEv = additionalOffset,
            measuredAverageEv = measured,
            normalizedAverageEv = normalized,
            targetAverageEv = target
        )
    }

    fun estimateAverageExposureEv(
        profileGainTableMap: DngProfileGainTableMap?,
        baselineExposureEv: Float,
        toneCurveLut: FloatArray?,
        profileExposureOffsetEv: Float = 0f,
        defaultBlackRender: DcpDefaultBlackRender = DcpDefaultBlackRender.None,
        shadowScale: Float = 1f
    ): Float? {
        val sourceLut = toneCurveLut?.takeIf { it.isNotEmpty() } ?: return null
        val samples = neutralSamples(profileGainTableMap, baselineExposureEv)
        if (samples.isEmpty()) return null
        return averageRenderedExposureEv(
            samples = samples,
            toneCurveLut = sourceLut,
            profileExposureOffsetEv = profileExposureOffsetEv,
            defaultBlackRender = defaultBlackRender,
            shadowScale = shadowScale
        )
    }

    private fun neutralSamples(
        profileGainTableMap: DngProfileGainTableMap?,
        baselineExposureEv: Float
    ): List<NeutralSample> {
        val map = profileGainTableMap?.takeIf { it.isValid } ?: return emptyList()
        val baselineGain = DngBaselineExposure.exactGain(baselineExposureEv)
        val weightSum = map.mapInputWeights.sum()
            .takeIf { it.isFinite() && it > 0f }
            ?: return emptyList()
        val gamma = map.gamma.coerceIn(0.125f, 8.0f)
        val samples = ArrayList<NeutralSample>(NEUTRAL_GRAY_SAMPLES.size)
        for (gray in NEUTRAL_GRAY_SAMPLES) {
            val tableInput = (gray * baselineGain * weightSum)
                .coerceIn(0f, 1f)
                .pow(gamma)
            val pgtmGain = medianGainAtTableInput(map, tableInput)
            val linearValue = gray * baselineGain * pgtmGain
            if (linearValue.isFinite() && linearValue > 0f) {
                samples += NeutralSample(gray = gray, linearValue = linearValue)
            }
        }
        return samples
    }

    private fun averageRenderedExposureEv(
        samples: List<NeutralSample>,
        toneCurveLut: FloatArray,
        profileExposureOffsetEv: Float,
        defaultBlackRender: DcpDefaultBlackRender,
        shadowScale: Float
    ): Float? {
        val exposure = profileExposureParams(
            exposureEv = profileExposureOffsetEv.takeIf { it.isFinite() } ?: 0f,
            defaultBlackRender = defaultBlackRender,
            shadowScale = shadowScale
        )
        var sum = 0f
        var count = 0
        samples.forEach { sample ->
            val exposed = applyProfileExposure(sample.linearValue, exposure)
            val output = sampleToneCurve(toneCurveLut, exposed)
            if (output.isFinite() && output in MIN_OUTPUT_FOR_EV..MAX_OUTPUT_FOR_EV) {
                sum += log2(output / sample.gray)
                count++
            }
        }
        if (count > 0) return sum / count.toFloat()

        val middle = samples.minByOrNull { kotlin.math.abs(it.gray - 0.18f) } ?: return null
        val exposed = applyProfileExposure(middle.linearValue, exposure)
        val output = sampleToneCurve(toneCurveLut, exposed)
        return if (output.isFinite() && output > 0f) log2(output / middle.gray) else null
    }

    private fun medianGainAtTableInput(map: DngProfileGainTableMap, tableInput: Float): Float {
        val pointCount = map.mapPointsN
        val cellCount = map.mapPointsH * map.mapPointsV
        if (pointCount <= 0 || cellCount <= 0) return 1f
        val tableIndex = (tableInput * pointCount.toFloat())
            .coerceIn(0f, (pointCount - 1).toFloat())
        val index0 = floor(tableIndex).toInt().coerceIn(0, pointCount - 1)
        val index1 = min(index0 + 1, pointCount - 1)
        val t = tableIndex - index0.toFloat()
        val values = FloatArray(cellCount) { cell ->
            val offset = cell * pointCount
            lerp(map.gains[offset + index0], map.gains[offset + index1], t)
        }
        values.sort()
        return values[(values.size - 1) / 2].takeIf { it.isFinite() && it > 0f } ?: 1f
    }

    private fun profileExposureParams(
        exposureEv: Float,
        defaultBlackRender: DcpDefaultBlackRender,
        shadowScale: Float
    ): ProfileExposureParams {
        val safeShadowScale = if (shadowScale.isFinite() && shadowScale > 0f && shadowScale <= 1f) {
            shadowScale
        } else {
            DEFAULT_BLACK_RENDER_SHADOW_SCALE
        }
        val positiveExposureEv = max(0f, exposureEv)
        val white = 1f / 2.0f.pow(positiveExposureEv)
        val black = when (defaultBlackRender) {
            DcpDefaultBlackRender.Auto -> DEFAULT_BLACK_RENDER_SHADOWS *
                safeShadowScale *
                DEFAULT_BLACK_RENDER_STAGE3_GAIN *
                0.001f

            DcpDefaultBlackRender.None -> 0f
        }.coerceIn(0f, 0.99f * white)
        val slope = 1f / max(white - black, 1e-6f)
        val radius = min(0.5f * black, (1f / 16f) / max(slope, 1e-6f))
        val qScale = if (radius > 0f) slope / (4f * radius) else 0f
        val toneEnabled = exposureEv < 0f
        val toneSlope = 2.0f.pow(exposureEv)
        val toneA = (16f / 9f) * (1f - toneSlope)
        val toneB = toneSlope - 0.5f * toneA
        val toneC = 1f - toneA - toneB
        return ProfileExposureParams(
            rampSlope = slope,
            rampBlack = black,
            rampRadius = radius,
            rampQScale = qScale,
            toneEnabled = toneEnabled,
            toneSlope = toneSlope,
            toneA = toneA,
            toneB = toneB,
            toneC = toneC
        )
    }

    private fun applyProfileExposure(value: Float, exposure: ProfileExposureParams): Float {
        val ramped = applyProfileExposureRamp(value, exposure)
        return applyProfileExposureTone(ramped, exposure)
    }

    private fun applyProfileExposureRamp(value: Float, exposure: ProfileExposureParams): Float {
        val black = exposure.rampBlack
        val radius = exposure.rampRadius
        val ramped = when {
            value <= black - radius -> 0f
            value >= black + radius -> max((value - black) * exposure.rampSlope, 0f)
            else -> {
                val y = value - (black - radius)
                exposure.rampQScale * y * y
            }
        }
        if (ramped <= PROFILE_HIGHLIGHT_SHOULDER_START) return ramped
        val shoulderInput = ramped - PROFILE_HIGHLIGHT_SHOULDER_START
        val shoulderAmount = shoulderInput / (shoulderInput + PROFILE_HIGHLIGHT_SHOULDER_SOFTNESS)
        return PROFILE_HIGHLIGHT_SHOULDER_START +
            (1f - PROFILE_HIGHLIGHT_SHOULDER_START) * shoulderAmount
    }

    private fun applyProfileExposureTone(value: Float, exposure: ProfileExposureParams): Float {
        if (!exposure.toneEnabled) return value
        return if (value <= 0.25f) {
            value * exposure.toneSlope
        } else {
            (exposure.toneA * value + exposure.toneB) * value + exposure.toneC
        }
    }

    private fun sampleToneCurve(lut: FloatArray, input: Float): Float {
        if (lut.isEmpty()) return input
        if (lut.size == 1) return lut[0]
        val x = input.coerceIn(0f, 1f) * (lut.size - 1).toFloat()
        val index0 = floor(x).toInt().coerceIn(0, lut.size - 1)
        val index1 = min(index0 + 1, lut.size - 1)
        return lerp(lut[index0], lut[index1], x - index0.toFloat())
    }

    private fun log2(value: Float): Float {
        return (ln(max(value, 1e-6f).toDouble()) / ln(2.0)).toFloat()
    }

    private fun lerp(a: Float, b: Float, t: Float): Float {
        return a + (b - a) * t.coerceIn(0f, 1f)
    }

    private data class NeutralSample(
        val gray: Float,
        val linearValue: Float
    )

    private data class ProfileExposureParams(
        val rampSlope: Float,
        val rampBlack: Float,
        val rampRadius: Float,
        val rampQScale: Float,
        val toneEnabled: Boolean,
        val toneSlope: Float,
        val toneA: Float,
        val toneB: Float,
        val toneC: Float
    )
}
