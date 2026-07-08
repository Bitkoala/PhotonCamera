package com.hinnka.mycamera.raw

import com.hinnka.mycamera.utils.PLog
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

internal object DngAppleProRawProfileGainTableGenerator {
    private const val TAG = "DngAppleProRawPgtmGenerator"

    private const val MAP_INPUT_WEIGHT_COUNT = 5
    private const val DEFAULT_TABLE_POINTS = 257
    private const val MIN_TABLE_POINTS = 257
    private const val MAX_TABLE_POINTS = 257
    private const val TARGET_TILE_PX = 64
    private const val GRID_MIN_H = 8
    private const val GRID_MIN_V = 6
    private const val GRID_MAX_H = 64
    private const val GRID_MAX_V = 48
    private const val MIN_INPUT_SCALE = 1f / 13.5f
    private const val MAX_INPUT_SCALE = 1f
    private const val MIN_GAIN_VALUE = 0.05f
    private const val MAX_GAIN_VALUE = 12.0f
    private const val MIN_CURVE_INPUT = 1e-6f
    private const val LOCAL_CURVE_MAX_BLEND = 0.22f
    private const val LOCAL_CURVE_SMOOTH_PASSES = 4

    internal const val BASE_INPUT_WEIGHT_RED = 0.1063f
    internal const val BASE_INPUT_WEIGHT_GREEN = 0.3576f
    internal const val BASE_INPUT_WEIGHT_BLUE = 0.0361f
    internal const val BASE_INPUT_WEIGHT_MIN = 0.0f
    internal const val BASE_INPUT_WEIGHT_MAX = 0.5f

    private val BASE_INPUT_WEIGHTS = floatArrayOf(
        BASE_INPUT_WEIGHT_RED,
        BASE_INPUT_WEIGHT_GREEN,
        BASE_INPUT_WEIGHT_BLUE,
        BASE_INPUT_WEIGHT_MIN,
        BASE_INPUT_WEIGHT_MAX
    )

    fun sceneInputFromLinearRgb(
        red: Float,
        green: Float,
        blue: Float,
        baselineGain: Float,
        colorCorrectionMatrix: FloatArray? = null,
    ): Float {
        val profileRgb = if (colorCorrectionMatrix != null && colorCorrectionMatrix.size >= 9) {
            floatArrayOf(
                colorCorrectionMatrix[0] * red + colorCorrectionMatrix[1] * green + colorCorrectionMatrix[2] * blue,
                colorCorrectionMatrix[3] * red + colorCorrectionMatrix[4] * green + colorCorrectionMatrix[5] * blue,
                colorCorrectionMatrix[6] * red + colorCorrectionMatrix[7] * green + colorCorrectionMatrix[8] * blue
            )
        } else {
            floatArrayOf(red, green, blue)
        }
        return sceneInputFromProfileRgb(
            red = profileRgb[0],
            green = profileRgb[1],
            blue = profileRgb[2],
            baselineGain = baselineGain
        )
    }

    fun sceneInputFromProfileRgb(
        red: Float,
        green: Float,
        blue: Float,
        baselineGain: Float,
    ): Float {
        val r = red * baselineGain
        val g = green * baselineGain
        val b = blue * baselineGain
        val rgbMin = min(r, min(g, b))
        val rgbMax = max(r, max(g, b))
        return max(
            BASE_INPUT_WEIGHT_RED * r +
                BASE_INPUT_WEIGHT_GREEN * g +
                BASE_INPUT_WEIGHT_BLUE * b +
                BASE_INPUT_WEIGHT_MIN * rgbMin +
                BASE_INPUT_WEIGHT_MAX * rgbMax,
            0f
        )
    }

    fun forCellStats(
        width: Int,
        height: Int,
        baselineExposureEv: Float,
        packedCellStats: FloatArray,
        tablePointCount: Int = DEFAULT_TABLE_POINTS,
        diagnosticBand: DngHdrProfileGainTableGenerator.DiagnosticBand? = null,
    ): DngProfileGainTableMap? {
        if (width <= 0 || height <= 0 || !baselineExposureEv.isFinite()) {
            return null
        }
        val grid = chooseGrid(width, height)
        val cellCount = grid.mapPointsH * grid.mapPointsV
        if (packedCellStats.size < cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE) {
            PLog.w(
                TAG,
                "Apple PGTM stats too small: ${packedCellStats.size}, " +
                    "expected=${cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE}"
            )
            return null
        }

        val safePointCount = tablePointCount.coerceIn(MIN_TABLE_POINTS, MAX_TABLE_POINTS)
        val cells = Array<HdrPgtmCellStats?>(cellCount) { index ->
            val offset = index * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
            val sampleWeight = packedCellStats[offset + 5].takeIf { it.isFinite() && it > 0f } ?: 0f
            if (sampleWeight <= 0f) {
                null
            } else {
                packedStatsAt(packedCellStats, offset, sampleWeight)
            }
        }
        val global = weightedGlobalStats(cells)
        val inputScale = appleInputScaleForStats(global)
        val sceneMax = 1f / sanitizeInputScale(inputScale)
        val globalParams = buildCurveParams(
            cell = global,
            global = global,
            inputScale = inputScale,
            sceneMax = sceneMax
        )
        val params = smoothCurveParams(
            source = Array(cellCount) { index ->
                val cell = cells[index]
                if (cell == null || cell.sampleWeight <= 0f) {
                    globalParams
                } else {
                    val localParams = buildCurveParams(
                        cell = cell,
                        global = global,
                        inputScale = inputScale,
                        sceneMax = sceneMax
                    )
                    blendCurveParams(
                        first = globalParams,
                        second = localParams,
                        amount = localCurveBlend(cell = cell, global = global, sceneMax = sceneMax)
                    )
                }
            },
            grid = grid
        )
        val map = buildMap(
            grid = grid,
            safePointCount = safePointCount,
            inputScale = inputScale,
            sceneMax = sceneMax,
            params = params
        )
        return DngHdrProfileGainTableGenerator.withDiagnosticBand(
            map,
            diagnosticBand?.sanitized()
        )
    }

    private fun packedStatsAt(stats: FloatArray, offset: Int, sampleWeight: Float): HdrPgtmCellStats {
        val p10 = safe01(stats[offset])
        val p50 = max(p10, safe01(stats[offset + 1]))
        val p90 = max(p50, safe01(stats[offset + 2]))
        val p98 = max(p90, safePositive(stats[offset + 3], p90))
        val p995Input = max(p98, safePositive(stats[offset + 6], p98))
        val p999Input = max(p995Input, safePositive(stats[offset + 7], p995Input))
        return HdrPgtmCellStats(
            p10 = p10,
            p50 = p50,
            p90 = p90,
            p98 = p98,
            highlightFraction = safe01(stats[offset + 4]),
            p995Input = p995Input,
            p999Input = p999Input,
            inputTailP95 = p995Input,
            inputTailP98 = p995Input,
            inputTailP99 = p999Input,
            maxInput = p999Input,
            sampleWeight = sampleWeight
        )
    }

    private fun weightedGlobalStats(cells: Array<HdrPgtmCellStats?>): HdrPgtmCellStats {
        var weightSum = 0f
        var p10 = 0f
        var p50 = 0f
        var p90 = 0f
        var p98 = 0f
        var highlightFraction = 0f
        var p995Input = 0f
        var p999Input = 0f
        val valid = ArrayList<HdrPgtmCellStats>(cells.size)
        cells.forEach { cell ->
            if (cell != null && cell.sampleWeight > 0f) {
                val weight = cell.sampleWeight
                valid += cell
                weightSum += weight
                p10 += cell.p10 * weight
                p50 += cell.p50 * weight
                p90 += cell.p90 * weight
                p98 += cell.p98 * weight
                highlightFraction += cell.highlightFraction * weight
                p995Input += cell.p995Input * weight
                p999Input += cell.p999Input * weight
            }
        }
        if (weightSum <= 0f) {
            return HdrPgtmCellStats(
                p10 = 0.03f,
                p50 = 0.18f,
                p90 = 0.48f,
                p98 = 0.72f,
                highlightFraction = 0f,
                p995Input = 0.82f,
                p999Input = 0.82f,
                inputTailP95 = 0.82f,
                inputTailP98 = 0.82f,
                inputTailP99 = 0.82f,
                maxInput = 0.82f,
                sampleWeight = 1f
            )
        }
        val inputTailP95 = weightedPercentile(valid, 0.95f) { it.p995Input }
        val inputTailP98 = weightedPercentile(valid, 0.98f) { it.p995Input }
        val inputTailP99 = weightedPercentile(valid, 0.99f) { it.p999Input }
        val maxInput = valid.maxOfOrNull { it.p999Input } ?: (p999Input / weightSum)
        return HdrPgtmCellStats(
            p10 = p10 / weightSum,
            p50 = p50 / weightSum,
            p90 = p90 / weightSum,
            p98 = p98 / weightSum,
            highlightFraction = highlightFraction / weightSum,
            p995Input = p995Input / weightSum,
            p999Input = p999Input / weightSum,
            inputTailP95 = inputTailP95,
            inputTailP98 = inputTailP98,
            inputTailP99 = inputTailP99,
            maxInput = maxInput,
            sampleWeight = weightSum
        )
    }

    private fun appleInputScaleForStats(global: HdrPgtmCellStats): Float {
        val directSceneMax = max(
            1f,
            max(global.inputTailP98, global.inputTailP99 * 0.98f).takeIf { it.isFinite() && it > 0f } ?: 1f
        )
        val brightMedian = smoothStep(0.16f, 0.24f, global.p50)
        val brightUpper = smoothStep(0.50f, 0.78f, global.p98)
        val denseHighlight = smoothStep(0.050f, 0.120f, global.highlightFraction)
        val highlightTail = smoothStep(1.30f, 4.50f, global.inputTailP95) *
            smoothStep(0.030f, 0.080f, global.highlightFraction)
        val expansion = max(max(brightMedian, brightUpper * max(denseHighlight, 0.45f)), highlightTail)
            .coerceIn(0f, 1f)
        val overBrightMedian = smoothStep(0.72f, 1.05f, global.p50)
        val expandedSceneMax = lerp(1f, 12.6f, expansion) *
            lerp(1f, 0.74f, overBrightMedian)
        val sceneMax = max(directSceneMax, expandedSceneMax).coerceIn(1f, 13.5f)
        return sanitizeInputScale(1f / sceneMax)
    }

    private fun buildCurveParams(
        cell: HdrPgtmCellStats,
        global: HdrPgtmCellStats,
        inputScale: Float,
        sceneMax: Float,
    ): AppleCurveParams {
        val expandedScene = smoothStep(2.0f, 8.5f, sceneMax)
        val darkCell = 1f - smoothStep(0.035f, 0.120f, cell.p10)
        val highlightTailPressure = (
            smoothStep(1.30f, 4.00f, global.inputTailP95) *
                smoothStep(0.035f, 0.090f, global.highlightFraction) +
                smoothStep(0.80f, 1.40f, global.p98) *
                smoothStep(0.040f, 0.120f, global.highlightFraction)
            ).coerceIn(0f, 1f)
        val compactLowGain = lerp(
            6.85f,
            4.95f,
            smoothStep(0.085f, 0.135f, global.p50)
        )
        val expandedLowGain = (
            1.30f +
                4.15f * darkCell * (1f - highlightTailPressure) +
                0.36f * (1f - smoothStep(0.22f, 0.42f, cell.p50))
            ).coerceIn(1.15f, 5.65f)
        val lowGain = lerp(compactLowGain, expandedLowGain, expandedScene)
            .coerceIn(1.05f, MAX_GAIN_VALUE)

        val compactKnee = lerp(0.178f, 0.225f, smoothStep(0.10f, 0.28f, global.p98))
        val expandedKnee = lerp(0.095f, 0.98f, smoothStep(0.11f, 0.36f, cell.p50)) *
            lerp(1f, 0.64f, highlightTailPressure)
        val knee = lerp(compactKnee, expandedKnee.coerceIn(0.055f, 1.10f), expandedScene)
            .coerceIn(0.035f, sceneMax)
        val power = (
            lerp(1.70f, 1.18f, expandedScene) +
                0.28f * highlightTailPressure +
                0.10f * smoothStep(2.0f, 5.0f, localRangeEv(cell))
            ).coerceIn(0.92f, 2.25f)
        return AppleCurveParams(
            lowGain = lowGain,
            knee = knee,
            power = power,
            terminalGain = inputScale
        )
    }

    private fun smoothCurveParams(
        source: Array<AppleCurveParams>,
        grid: HdrPgtmGrid,
    ): Array<AppleCurveParams> {
        var current = source
        repeat(LOCAL_CURVE_SMOOTH_PASSES) {
            current = Array(current.size) { index ->
                val x = index % grid.mapPointsH
                val y = index / grid.mapPointsH
                var weightSum = 0f
                var lowGain = 0f
                var knee = 0f
                var power = 0f
                var terminalGain = 0f
                for (dy in -1..1) {
                    val yy = y + dy
                    if (yy !in 0 until grid.mapPointsV) continue
                    for (dx in -1..1) {
                        val xx = x + dx
                        if (xx !in 0 until grid.mapPointsH) continue
                        val weight = when {
                            dx == 0 && dy == 0 -> 4f
                            dx == 0 || dy == 0 -> 2f
                            else -> 1f
                        }
                        val params = current[yy * grid.mapPointsH + xx]
                        weightSum += weight
                        lowGain += params.lowGain * weight
                        knee += params.knee * weight
                        power += params.power * weight
                        terminalGain += params.terminalGain * weight
                    }
                }
                AppleCurveParams(
                    lowGain = lowGain / weightSum,
                    knee = knee / weightSum,
                    power = power / weightSum,
                    terminalGain = terminalGain / weightSum
                )
            }
        }
        return current
    }

    private fun localCurveBlend(
        cell: HdrPgtmCellStats,
        global: HdrPgtmCellStats,
        sceneMax: Float,
    ): Float {
        val expandedScene = smoothStep(2.0f, 8.5f, sceneMax)
        if (expandedScene <= 0f) return 0f
        val confidence = smoothStep(64f, 192f, cell.sampleWeight)
        val localRange = localRangeEv(cell)
        val detailGate = smoothStep(0.70f, 2.20f, localRange)
        val toneDistance = (
            absLogRatio(cell.p50 + 0.006f, global.p50 + 0.006f) * 0.34f +
                absLogRatio(cell.p90 + 0.006f, global.p90 + 0.006f) * 0.28f +
                absLogRatio(cell.p98 + 0.006f, global.p98 + 0.006f) * 0.24f +
                absLogRatio(cell.p999Input + 0.04f, global.inputTailP99 + 0.04f) * 0.18f +
                kotlin.math.abs(cell.highlightFraction - global.highlightFraction) * 0.70f
            )
        val toneVariation = smoothStep(0.48f, 1.85f, toneDistance)
        val highlightPressure = smoothStep(0.025f, 0.16f, cell.highlightFraction) *
            smoothStep(0.45f, 1.40f, toneDistance)
        val flatAreaDamping = lerp(0.12f, 1f, detailGate)
        val localSignal = max(toneVariation * flatAreaDamping, highlightPressure * 0.55f)
        return (LOCAL_CURVE_MAX_BLEND * expandedScene * confidence * localSignal)
            .coerceIn(0f, LOCAL_CURVE_MAX_BLEND)
    }

    private fun blendCurveParams(
        first: AppleCurveParams,
        second: AppleCurveParams,
        amount: Float,
    ): AppleCurveParams {
        val t = amount.coerceIn(0f, 1f)
        return AppleCurveParams(
            lowGain = lerp(first.lowGain, second.lowGain, t),
            knee = lerp(first.knee, second.knee, t),
            power = lerp(first.power, second.power, t),
            terminalGain = lerp(first.terminalGain, second.terminalGain, t)
        )
    }

    private fun buildMap(
        grid: HdrPgtmGrid,
        safePointCount: Int,
        inputScale: Float,
        sceneMax: Float,
        params: Array<AppleCurveParams>,
    ): DngProfileGainTableMap {
        val gains = FloatArray(grid.mapPointsH * grid.mapPointsV * safePointCount)
        val safeInputScale = sanitizeInputScale(inputScale)
        for (cellIndex in 0 until grid.mapPointsH * grid.mapPointsV) {
            writeAppleCurve(
                output = gains,
                outputOffset = cellIndex * safePointCount,
                pointCount = safePointCount,
                inputScale = safeInputScale,
                sceneMax = sceneMax,
                params = params[cellIndex]
            )
        }
        return DngProfileGainTableMap(
            mapPointsV = grid.mapPointsV,
            mapPointsH = grid.mapPointsH,
            mapSpacingV = grid.mapSpacingV,
            mapSpacingH = grid.mapSpacingH,
            mapOriginV = 0.0,
            mapOriginH = 0.0,
            mapPointsN = safePointCount,
            mapInputWeights = appleInputWeights(safeInputScale),
            gamma = 1f,
            gains = gains,
            sourceTag = DngProfileGainTableMap.TAG_PROFILE_GAIN_TABLE_MAP2
        )
    }

    private fun writeAppleCurve(
        output: FloatArray,
        outputOffset: Int,
        pointCount: Int,
        inputScale: Float,
        sceneMax: Float,
        params: AppleCurveParams,
    ) {
        var previousOutput = 0f
        for (index in 0 until pointCount) {
            val tableInput = tableInputForIndex(index, pointCount)
            if (tableInput <= MIN_CURVE_INPUT) {
                output[outputOffset + index] = params.lowGain.coerceIn(MIN_GAIN_VALUE, MAX_GAIN_VALUE)
                continue
            }
            val scene = tableInput / inputScale
            val rawGain = appleGainForScene(scene, sceneMax, params)
            val targetOutput = (scene * rawGain).coerceIn(0f, 1f)
            val monotonicOutput = max(previousOutput, targetOutput)
            output[outputOffset + index] = (monotonicOutput / max(scene, MIN_CURVE_INPUT))
                .coerceIn(MIN_GAIN_VALUE, MAX_GAIN_VALUE)
            previousOutput = monotonicOutput
        }
    }

    private fun appleGainForScene(
        scene: Float,
        sceneMax: Float,
        params: AppleCurveParams,
    ): Float {
        val safeScene = max(scene, MIN_CURVE_INPUT)
        val rolloff = 1f / (1f + (safeScene / max(params.knee, 1e-4f)).pow(params.power))
        val baseGain = params.terminalGain + (params.lowGain - params.terminalGain) * rolloff
        val whiteGain = 1f / max(safeScene, MIN_CURVE_INPUT)
        val whiteBlend = smoothStep(sceneMax * 0.62f, sceneMax, safeScene)
        return lerp(baseGain, whiteGain, whiteBlend).coerceIn(MIN_GAIN_VALUE, MAX_GAIN_VALUE)
    }

    private fun appleInputWeights(inputScale: Float): FloatArray {
        val scale = sanitizeInputScale(inputScale)
        return FloatArray(MAP_INPUT_WEIGHT_COUNT) { index ->
            BASE_INPUT_WEIGHTS[index] * scale
        }
    }

    private fun chooseGrid(width: Int, height: Int): HdrPgtmGrid {
        val mapPointsH = ((width + TARGET_TILE_PX - 1) / TARGET_TILE_PX)
            .coerceIn(GRID_MIN_H, GRID_MAX_H)
        val mapPointsV = ((height + TARGET_TILE_PX - 1) / TARGET_TILE_PX)
            .coerceIn(GRID_MIN_V, GRID_MAX_V)
        return HdrPgtmGrid(
            mapPointsH = mapPointsH,
            mapPointsV = mapPointsV,
            mapSpacingH = if (mapPointsH > 1) 1.0 / (mapPointsH - 1).toDouble() else 1.0,
            mapSpacingV = if (mapPointsV > 1) 1.0 / (mapPointsV - 1).toDouble() else 1.0
        )
    }

    private fun weightedPercentile(
        cells: List<HdrPgtmCellStats>,
        percentile: Float,
        selector: (HdrPgtmCellStats) -> Float,
    ): Float {
        if (cells.isEmpty()) return 0f
        val sorted = cells
            .mapNotNull { cell ->
                val value = selector(cell)
                if (value.isFinite() && cell.sampleWeight > 0f) value to cell.sampleWeight else null
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

    private fun localRangeEv(cell: HdrPgtmCellStats): Float {
        return log2((cell.p98 + 0.006f) / (cell.p10 + 0.006f)).coerceIn(0f, 8f)
    }

    private fun sanitizeInputScale(inputScale: Float): Float {
        return inputScale.takeIf { it.isFinite() }
            ?.coerceIn(MIN_INPUT_SCALE, MAX_INPUT_SCALE)
            ?: MAX_INPUT_SCALE
    }

    private fun tableInputForIndex(index: Int, pointCount: Int): Float {
        if (pointCount <= 1) return 0f
        return if (index == pointCount - 1) 1f else index.toFloat() / pointCount.toFloat()
    }

    private fun safe01(value: Float): Float {
        return value.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 0f
    }

    private fun safePositive(value: Float, fallback: Float): Float {
        return value.takeIf { it.isFinite() && it > 0f } ?: fallback
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

    private fun absLogRatio(first: Float, second: Float): Float {
        return kotlin.math.abs(log2(first / max(second, 1e-6f)))
    }

    private data class AppleCurveParams(
        val lowGain: Float,
        val knee: Float,
        val power: Float,
        val terminalGain: Float,
    )
}
