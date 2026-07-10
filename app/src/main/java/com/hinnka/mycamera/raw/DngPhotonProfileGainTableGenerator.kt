package com.hinnka.mycamera.raw

import com.hinnka.mycamera.utils.PLog
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

internal object DngPhotonProfileGainTableGenerator {
    private const val TAG = "DngPhotonProfileGainTableGenerator"

    private const val DEFAULT_TABLE_POINTS = 257
    private const val MIN_TABLE_POINTS = 257
    private const val MAX_TABLE_POINTS = 257
    private const val MIN_BASELINE_EV = 0f
    private const val MAX_BASELINE_EV = 8f
    private const val MIN_CURVE_INPUT = 1e-6f
    private const val MIN_SCENE_MAX = 1f
    private const val MAX_SCENE_MAX = 4.80f
    private const val MAX_GAIN_VALUE = 5.40f
    private const val MAP_INPUT_WEIGHT_COUNT = 5
    private const val MID_GRAY_INPUT = 0.18f
    private const val DEFAULT_MID_GRAY_LIFT_EV = 1.10f
    private const val MAX_LOCAL_MICRO_CONTRAST = 0.045f

    private val BASE_INPUT_WEIGHTS = floatArrayOf(
        DngHdrProfileGainTableGenerator.BASE_INPUT_WEIGHT_RED,
        DngHdrProfileGainTableGenerator.BASE_INPUT_WEIGHT_GREEN,
        DngHdrProfileGainTableGenerator.BASE_INPUT_WEIGHT_BLUE,
        DngHdrProfileGainTableGenerator.BASE_INPUT_WEIGHT_MIN,
        DngHdrProfileGainTableGenerator.BASE_INPUT_WEIGHT_MAX
    )

    fun forCellStats(
        width: Int,
        height: Int,
        baselineExposureEv: Float,
        packedCellStats: FloatArray,
        tablePointCount: Int = DEFAULT_TABLE_POINTS,
        diagnosticBand: DngHdrProfileGainTableGenerator.DiagnosticBand? = null,
    ): DngProfileGainTableMap? {
        if (width <= 0 || height <= 0 || !baselineExposureEv.isFinite() || baselineExposureEv < MIN_BASELINE_EV) {
            return null
        }
        val gridSize = DngHdrProfileGainTableGenerator.gridSizeFor(width, height)
        val gridWidth = gridSize.getOrElse(0) { 0 }
        val gridHeight = gridSize.getOrElse(1) { 0 }
        if (gridWidth <= 0 || gridHeight <= 0) return null
        val cellCount = gridWidth * gridHeight
        if (packedCellStats.size < cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE) {
            PLog.w(
                TAG,
                "Photon PGTM stats too small: ${packedCellStats.size}, " +
                    "expected=${cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE}"
            )
            return null
        }

        val cells = Array<PhotonPgtmCellStats?>(cellCount) { index ->
            val offset = index * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
            val sampleWeight = packedCellStats[offset + 5].takeIf { it.isFinite() && it > 0f } ?: 0f
            if (sampleWeight <= 0f) {
                null
            } else {
                packedStatsAt(packedCellStats, offset, sampleWeight)
            }
        }
        val global = weightedGlobalStats(cells)
        val safeBaselineEv = baselineExposureEv.coerceIn(0f, MAX_BASELINE_EV)
        val safePointCount = tablePointCount.coerceIn(MIN_TABLE_POINTS, MAX_TABLE_POINTS)
        val sceneMax = photonSceneMaxForStats(global)
        // Stats collection and the render shader both apply DNG baseline gain before
        // table lookup, so table input 1.0 corresponds to raw linear 1 / baselineGain.
        val inputScale = sanitizeInputScale(1f / sceneMax)
        val gains = FloatArray(cellCount * safePointCount)
        for (cellIndex in 0 until cellCount) {
            writePhotonCurve(
                output = gains,
                outputOffset = cellIndex * safePointCount,
                pointCount = safePointCount,
                inputScale = inputScale,
                global = global,
                local = cells[cellIndex] ?: global
            )
        }

        val map = DngProfileGainTableMap(
            mapPointsV = gridHeight,
            mapPointsH = gridWidth,
            mapSpacingV = if (gridHeight > 1) 1.0 / (gridHeight - 1).toDouble() else 1.0,
            mapSpacingH = if (gridWidth > 1) 1.0 / (gridWidth - 1).toDouble() else 1.0,
            mapOriginV = 0.0,
            mapOriginH = 0.0,
            mapPointsN = safePointCount,
            mapInputWeights = photonPgtmInputWeights(inputScale),
            gamma = 1f,
            gains = gains,
            sourceTag = DngProfileGainTableMap.TAG_PROFILE_GAIN_TABLE_MAP2
        )
        PLog.d(
            TAG,
            "Built Photon PGTM: grid=${gridWidth}x${gridHeight}x$safePointCount " +
                "baselineEv=$safeBaselineEv sceneMax=$sceneMax inputScale=$inputScale " +
                "p98=${global.p98} tail99=${global.tailP99} max=${global.maxInput}"
        )
        return DngHdrProfileGainTableGenerator.withDiagnosticBand(map, diagnosticBand)
    }

    private fun packedStatsAt(stats: FloatArray, offset: Int, sampleWeight: Float): PhotonPgtmCellStats {
        val p10 = safe01(stats[offset])
        val p50 = max(p10, safe01(stats[offset + 1]))
        val p90 = max(p50, safe01(stats[offset + 2]))
        val p98 = max(p90, safe01(stats[offset + 3]))
        val tailP95 = max(p98, safePositive(stats[offset + 6], p98))
        val tailP99 = max(tailP95, safePositive(stats[offset + 7], tailP95))
        return PhotonPgtmCellStats(
            p10 = p10,
            p50 = p50,
            p90 = p90,
            p98 = p98,
            highlightFraction = safe01(stats[offset + 4]),
            tailP95 = tailP95,
            tailP99 = tailP99,
            maxInput = tailP99,
            sampleWeight = sampleWeight
        )
    }

    private fun weightedGlobalStats(cells: Array<PhotonPgtmCellStats?>): PhotonPgtmCellStats {
        var weightSum = 0f
        var p10 = 0f
        var p50 = 0f
        var p90 = 0f
        var p98 = 0f
        var highlightFraction = 0f
        val validCells = ArrayList<PhotonPgtmCellStats>(cells.size)
        cells.forEach { cell ->
            if (cell != null && cell.sampleWeight > 0f) {
                validCells += cell
                weightSum += cell.sampleWeight
                p10 += cell.p10 * cell.sampleWeight
                p50 += cell.p50 * cell.sampleWeight
                p90 += cell.p90 * cell.sampleWeight
                p98 += cell.p98 * cell.sampleWeight
                highlightFraction += cell.highlightFraction * cell.sampleWeight
            }
        }
        if (weightSum <= 0f) {
            return PhotonPgtmCellStats(
                p10 = 0.02f,
                p50 = 0.18f,
                p90 = 0.55f,
                p98 = 0.82f,
                highlightFraction = 0f,
                tailP95 = 0.82f,
                tailP99 = 0.82f,
                maxInput = 0.82f,
                sampleWeight = 1f
            )
        }
        val tailP95 = weightedPercentile(validCells, 0.95f) { it.tailP95 }
        val tailP99 = weightedPercentile(validCells, 0.99f) { it.tailP99 }
        val maxInput = validCells.maxOfOrNull { it.maxInput } ?: tailP99
        return PhotonPgtmCellStats(
            p10 = p10 / weightSum,
            p50 = p50 / weightSum,
            p90 = p90 / weightSum,
            p98 = p98 / weightSum,
            highlightFraction = highlightFraction / weightSum,
            tailP95 = tailP95,
            tailP99 = tailP99,
            maxInput = maxInput,
            sampleWeight = weightSum
        )
    }

    private fun weightedPercentile(
        cells: List<PhotonPgtmCellStats>,
        percentile: Float,
        selector: (PhotonPgtmCellStats) -> Float,
    ): Float {
        if (cells.isEmpty()) return 0f
        val sorted = cells
            .mapNotNull { cell ->
                val value = selector(cell)
                if (value.isFinite() && cell.sampleWeight > 0f) {
                    value to cell.sampleWeight
                } else {
                    null
                }
            }
            .sortedBy { it.first }
        if (sorted.isEmpty()) return 0f
        val totalWeight = sorted.sumOf { it.second.toDouble() }.toFloat()
        val target = totalWeight * percentile.coerceIn(0f, 1f)
        var cumulative = 0f
        sorted.forEach { (value, weight) ->
            cumulative += weight
            if (cumulative >= target) return value
        }
        return sorted.last().first
    }

    private fun photonSceneMaxForStats(global: PhotonPgtmCellStats): Float {
        val tailP95 = max(global.tailP95, global.p98)
        val tailP99 = max(global.tailP99, tailP95)
        val maxInput = max(global.maxInput, tailP99)
        val tailRangeEv = log2((tailP99 + 0.04f) / (global.p50 + 0.04f)).coerceIn(0f, 8f)
        val highlightDensity = global.highlightFraction.coerceIn(0f, 1f)
        val hdrStrength = max(
            max(smoothStep(1.02f, 1.55f, tailP95), smoothStep(1.8f, 4.2f, tailRangeEv)),
            smoothStep(0.08f, 0.24f, highlightDensity) * smoothStep(0.86f, 1.18f, tailP95)
        )
        val outlierGapEv = log2((maxInput + 0.04f) / (tailP99 + 0.04f)).coerceIn(0f, 6f)
        val sparseOutlier = smoothStep(0.45f, 1.25f, outlierGapEv) *
            (1f - smoothStep(0.16f, 0.32f, highlightDensity))
        val denseHighlight = smoothStep(0.12f, 0.30f, highlightDensity)
        val percentileTarget = max(
            1f,
            lerp(tailP95 * 1.03f, tailP99 * 0.96f, 0.45f + 0.35f * denseHighlight)
        )
        val outlierTarget = lerp(
            percentileTarget,
            min(maxInput, max(percentileTarget, tailP99 * 1.24f)),
            sparseOutlier * 0.42f
        )
        return lerp(MIN_SCENE_MAX, outlierTarget, hdrStrength)
            .coerceIn(MIN_SCENE_MAX, MAX_SCENE_MAX)
    }

    private fun writePhotonCurve(
        output: FloatArray,
        outputOffset: Int,
        pointCount: Int,
        inputScale: Float,
        global: PhotonPgtmCellStats,
        local: PhotonPgtmCellStats,
    ) {
        val safeInputScale = sanitizeInputScale(inputScale)
        val sceneMax = (1f / safeInputScale).coerceIn(MIN_SCENE_MAX, MAX_SCENE_MAX)
        var previousOutput = 0f
        val lowTableInput = 1f / max(pointCount - 1, 1).toFloat()
        val lowSceneLinear = lowTableInput / safeInputScale
        val lowGain = (photonToneOutput(lowSceneLinear, sceneMax, global, local) / lowSceneLinear)
            .coerceIn(0.05f, MAX_GAIN_VALUE)
        for (index in 0 until pointCount) {
            val tableInput = tableInputForIndex(index, pointCount)
            if (tableInput <= MIN_CURVE_INPUT) {
                output[outputOffset + index] = lowGain
                continue
            }
            val sceneLinear = tableInput / safeInputScale
            val targetOutput = photonToneOutput(sceneLinear, sceneMax, global, local)
            val monotonicOutput = max(previousOutput, targetOutput)
            output[outputOffset + index] = (monotonicOutput / max(sceneLinear, MIN_CURVE_INPUT))
                .coerceIn(0.05f, MAX_GAIN_VALUE)
            previousOutput = monotonicOutput
        }
    }

    private fun photonToneOutput(
        sceneLinear: Float,
        sceneMax: Float,
        global: PhotonPgtmCellStats,
        local: PhotonPgtmCellStats,
    ): Float {
        if (sceneLinear <= 0f) return 0f
        val midGrayGain = photonMidGrayGain(global, local, sceneMax)
        val midGrayOutput = (MID_GRAY_INPUT * midGrayGain).coerceIn(0.20f, 0.72f)
        val baseOutput = if (sceneLinear <= MID_GRAY_INPUT) {
            photonAdaptiveShadowOutput(
                baseOutput = sceneLinear * midGrayGain,
                sceneLinear = sceneLinear,
                sceneMax = sceneMax,
                global = global,
                local = local
            )
        } else {
            // sceneMax is baseline-normalized. Its output 1 maps to raw 1 / baselineGain.
            val inputRange = max(sceneMax - MID_GRAY_INPUT, 1e-4f)
            val outputRange = max(1f - midGrayOutput, 1e-4f)
            val t = ((sceneLinear - MID_GRAY_INPUT) / inputRange).coerceIn(0f, 1f)
            val initialSlope = min(midGrayGain * inputRange, 2.4f * outputRange)
            val finalSlope = 0.22f * outputRange
            cubicHermite(
                start = midGrayOutput,
                end = 1f,
                startSlope = initialSlope,
                endSlope = finalSlope,
                t = t
            )
        }
        return photonLocalMicroContrastOutput(
            baseOutput = baseOutput,
            sceneLinear = sceneLinear,
            global = global,
            local = local
        )
    }

    private fun photonMidGrayGain(
        global: PhotonPgtmCellStats,
        local: PhotonPgtmCellStats,
        sceneMax: Float,
    ): Float {
        val rangeEv = log2((global.p98 + 0.008f) / (global.p10 + 0.008f)).coerceIn(0f, 10f)
        val highContrast = smoothStep(3.6f, 6.2f, rangeEv) *
            smoothStep(1.10f, 1.80f, sceneMax)
        val lowContrast = 1f - smoothStep(1.6f, 3.0f, rangeEv)
        val darkScene = 1f - smoothStep(0.14f, 0.30f, global.p50)
        val darkCell = 1f - smoothStep(0.12f, 0.28f, local.p50)
        val liftEv = DEFAULT_MID_GRAY_LIFT_EV +
            0.22f * highContrast * darkScene +
            0.08f * highContrast * darkCell -
            0.16f * highContrast * (1f - darkScene) -
            0.12f * lowContrast
        return 2.0f.pow(liftEv.coerceIn(0.82f, 1.38f))
    }

    private fun cubicHermite(
        start: Float,
        end: Float,
        startSlope: Float,
        endSlope: Float,
        t: Float,
    ): Float {
        val u = t.coerceIn(0f, 1f)
        val u2 = u * u
        val u3 = u2 * u
        val h00 = 2f * u3 - 3f * u2 + 1f
        val h10 = u3 - 2f * u2 + u
        val h01 = -2f * u3 + 3f * u2
        val h11 = u3 - u2
        return (h00 * start + h10 * startSlope + h01 * end + h11 * endSlope)
            .coerceIn(start, end)
    }

    /**
     * The fixed profile curve owns only the base S shape. PGTM adds local shadow
     * gain when the scene is both high contrast and globally dark.
     */
    private fun photonAdaptiveShadowOutput(
        baseOutput: Float,
        sceneLinear: Float,
        sceneMax: Float,
        global: PhotonPgtmCellStats,
        local: PhotonPgtmCellStats,
    ): Float {
        val x = sceneLinear.coerceIn(0f, MID_GRAY_INPUT)
        if (x <= 0f || x >= MID_GRAY_INPUT) return baseOutput
        val shadowWindow = smoothStep(0.010f, 0.036f, x) *
            (1f - smoothStep(0.135f, MID_GRAY_INPUT, x))
        val adaptiveLift = photonAdaptiveShadowLiftStrength(sceneMax, global, local)
        return baseOutput * (1f + adaptiveLift * shadowWindow)
    }

    private fun photonLocalMicroContrastOutput(
        baseOutput: Float,
        sceneLinear: Float,
        global: PhotonPgtmCellStats,
        local: PhotonPgtmCellStats,
    ): Float {
        val localRangeEv = log2((local.p90 + 0.008f) / (local.p10 + 0.008f))
            .coerceIn(0f, 8f)
        val localDetail = smoothStep(1.15f, 3.20f, localRangeEv)
        val highlightDamping = 1f - smoothStep(0.10f, 0.34f, global.highlightFraction)
        val strength = MAX_LOCAL_MICRO_CONTRAST * localDetail * highlightDamping
        if (strength <= 1e-4f) return baseOutput

        val pivot = local.p50.coerceIn(0.065f, 0.28f)
        val signedDistance = (log2((sceneLinear + 0.010f) / (pivot + 0.010f)) / 1.35f)
            .coerceIn(-1f, 1f)
        val tonalWindow = smoothStep(0.020f, 0.065f, sceneLinear) *
            (1f - smoothStep(0.55f, 0.85f, sceneLinear))
        val adjusted = baseOutput * (1f + strength * signedDistance * tonalWindow)
        return adjusted.coerceIn(0f, 1f)
    }

    private fun photonAdaptiveShadowLiftStrength(
        sceneMax: Float,
        global: PhotonPgtmCellStats,
        local: PhotonPgtmCellStats,
    ): Float {
        val globalRangeEv = log2((global.p98 + 0.008f) / (global.p10 + 0.008f))
            .coerceIn(0f, 10f)
        val highDynamicRange = smoothStep(3.2f, 5.8f, globalRangeEv) *
            smoothStep(1.10f, 1.80f, sceneMax)
        val globalDarkArea = max(
            1f - smoothStep(0.075f, 0.165f, global.p50),
            0.75f * (1f - smoothStep(0.24f, 0.50f, global.p90))
        )
        val localDarkArea = max(
            1f - smoothStep(0.060f, 0.150f, local.p50),
            0.60f * (1f - smoothStep(0.20f, 0.45f, local.p90))
        )
        val darkSceneLift = highDynamicRange * globalDarkArea.coerceIn(0f, 1f)
        val darkCellLift = darkSceneLift * localDarkArea.coerceIn(0f, 1f)
        return (0.068f * darkSceneLift + 0.022f * darkCellLift)
            .coerceIn(0f, 0.090f)
    }

    private fun photonPgtmInputWeights(inputScale: Float): FloatArray {
        val scale = sanitizeInputScale(inputScale)
        return FloatArray(MAP_INPUT_WEIGHT_COUNT) { index ->
            BASE_INPUT_WEIGHTS[index] * scale
        }
    }

    private fun tableInputForIndex(index: Int, pointCount: Int): Float {
        if (pointCount <= 1) return 0f
        return if (index == pointCount - 1) {
            1f
        } else {
            index.toFloat() / pointCount.toFloat()
        }
    }

    private fun sanitizeInputScale(inputScale: Float): Float {
        return inputScale.takeIf { it.isFinite() }
            ?.coerceIn(1f / MAX_SCENE_MAX, 1.0f)
            ?: 1.0f
    }

    private fun smoothStep(edge0: Float, edge1: Float, x: Float): Float {
        val t = ((x - edge0) / max(edge1 - edge0, 1e-6f)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun lerp(a: Float, b: Float, t: Float): Float {
        return a + (b - a) * min(max(t, 0f), 1f)
    }

    private fun log2(value: Float): Float {
        return (ln(max(value, 1e-6f).toDouble()) / ln(2.0)).toFloat()
    }

    private fun safe01(value: Float): Float {
        return value.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 0f
    }

    private fun safePositive(value: Float, fallback: Float): Float {
        return value.takeIf { it.isFinite() && it > 0f } ?: fallback
    }

    private data class PhotonPgtmCellStats(
        val p10: Float,
        val p50: Float,
        val p90: Float,
        val p98: Float,
        val highlightFraction: Float,
        val tailP95: Float,
        val tailP99: Float,
        val maxInput: Float,
        val sampleWeight: Float,
    )
}
