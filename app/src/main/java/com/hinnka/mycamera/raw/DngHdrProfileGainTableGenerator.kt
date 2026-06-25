package com.hinnka.mycamera.raw

import com.hinnka.mycamera.utils.PLog
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

internal object DngHdrProfileGainTableGenerator {
    private const val TAG = "DngHdrProfileGainTableGenerator"

    const val CELL_STATS_FLOAT_STRIDE = 8

    private const val MAP_INPUT_WEIGHT_COUNT = 5
    private const val MIN_BASELINE_EV = 0.05f
    private const val MAX_BASELINE_EV = 8f
    private const val DEFAULT_TABLE_POINTS = 257
    private const val MIN_TABLE_POINTS = 257
    private const val MAX_TABLE_POINTS = 257
    private const val TARGET_TILE_PX = 128
    private const val GRID_MIN_H = 8
    private const val GRID_MIN_V = 6
    private const val GRID_MAX_H = 32
    private const val GRID_MAX_V = 24
    private const val INPUT_HEADROOM = 1.12f
    private const val MIN_HIGHLIGHT_GAIN = 0.08f
    private const val MAX_HIGHLIGHT_GAIN = 0.95f
    private const val MIN_CURVE_INPUT = 1e-6f
    private const val MIN_SHOULDER_PIVOT = 0.055f
    private const val MAX_SHOULDER_PIVOT = 0.24f
    private const val MIN_LOW_GAIN = 0.82f
    private const val MAX_LOW_GAIN = 1.72f
    private const val IPHONE_AVERAGE_SHOULDER_PIVOT = 0.102f
    private const val IPHONE_AVERAGE_SHOULDER_POWER = 1.60f
    private const val IPHONE_AVERAGE_TAIL_POWER = 1.9f
    private const val IPHONE_HIGHLIGHT_SHOULDER_PIVOT = 0.0485f
    private const val IPHONE_HIGHLIGHT_SHOULDER_POWER = 1.025f
    private const val IPHONE_HIGHLIGHT_TAIL_POWER = 1.3f
    private const val IPHONE_DODGE_SHOULDER_PIVOT = 0.080f
    private const val IPHONE_DODGE_SHOULDER_POWER = 1.60f
    private const val IPHONE_DODGE_TAIL_POWER = 1.8f
    private val BASE_INPUT_WEIGHTS = floatArrayOf(
        0.1063f,
        0.3576f,
        0.0361f,
        0f,
        0.5f
    )

    fun gridSizeFor(width: Int, height: Int): IntArray {
        val grid = chooseHdrPgtmGrid(width, height)
        return intArrayOf(grid.mapPointsH, grid.mapPointsV)
    }

    fun forHdrBaselineExposure(
        baselineExposureEv: Float,
        tablePointCount: Int = DEFAULT_TABLE_POINTS,
    ): DngProfileGainTableMap? {
        if (!baselineExposureEv.isFinite() || baselineExposureEv <= MIN_BASELINE_EV) {
            return null
        }
        val safeBaselineEv = baselineExposureEv.coerceIn(0f, MAX_BASELINE_EV)
        val safePointCount = tablePointCount.coerceIn(MIN_TABLE_POINTS, MAX_TABLE_POINTS)
        return buildMap(
            grid = HdrPgtmGrid(
                mapPointsH = 1,
                mapPointsV = 1,
                mapSpacingH = 1.0,
                mapSpacingV = 1.0
            ),
            safeBaselineEv = safeBaselineEv,
            safePointCount = safePointCount,
            curveParams = arrayOf(baselineHdrPgtmCurveParams(safeBaselineEv))
        )
    }

    fun forHdrCellStats(
        width: Int,
        height: Int,
        baselineExposureEv: Float,
        packedCellStats: FloatArray,
        tablePointCount: Int = DEFAULT_TABLE_POINTS,
    ): DngProfileGainTableMap? {
        if (width <= 0 || height <= 0 || !baselineExposureEv.isFinite() || baselineExposureEv <= MIN_BASELINE_EV) {
            return null
        }
        val grid = chooseHdrPgtmGrid(width, height)
        val cellCount = grid.mapPointsH * grid.mapPointsV
        if (packedCellStats.size < cellCount * CELL_STATS_FLOAT_STRIDE) {
            PLog.w(
                TAG,
                "GPU PGTM stats too small: ${packedCellStats.size}, expected=${cellCount * CELL_STATS_FLOAT_STRIDE}"
            )
            return forHdrBaselineExposure(baselineExposureEv, tablePointCount)
        }
        val safeBaselineEv = baselineExposureEv.coerceIn(0f, MAX_BASELINE_EV)
        val safePointCount = tablePointCount.coerceIn(MIN_TABLE_POINTS, MAX_TABLE_POINTS)
        val cells = Array<HdrPgtmCellStats?>(cellCount) { index ->
            val offset = index * CELL_STATS_FLOAT_STRIDE
            val sampleWeight = packedCellStats[offset + 5].takeIf { it.isFinite() && it > 0f } ?: 0f
            if (sampleWeight <= 0f) {
                null
            } else {
                packedStatsAt(packedCellStats, offset, sampleWeight)
            }
        }
        val globalStats = weightedGlobalStats(cells)
        val curveParams = smoothHdrPgtmCurveParams(
            stats = Array(cellCount) { index ->
                buildHdrPgtmCurveParams(
                    cell = cells[index] ?: globalStats,
                    global = globalStats,
                    baselineExposureEv = safeBaselineEv
                )
            },
            grid = grid
        )
        return buildMap(
            grid = grid,
            safeBaselineEv = safeBaselineEv,
            safePointCount = safePointCount,
            curveParams = curveParams
        )
    }

    private fun packedStatsAt(stats: FloatArray, offset: Int, sampleWeight: Float): HdrPgtmCellStats {
        val p10 = safe01(stats[offset])
        val p50 = max(p10, safe01(stats[offset + 1]))
        val p90 = max(p50, safe01(stats[offset + 2]))
        val p98 = max(p90, safe01(stats[offset + 3]))
        return HdrPgtmCellStats(
            p10 = p10,
            p50 = p50,
            p90 = p90,
            p98 = p98,
            highlightFraction = safe01(stats[offset + 4]),
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
        cells.forEach { cell ->
            if (cell != null && cell.sampleWeight > 0f) {
                val weight = cell.sampleWeight
                weightSum += weight
                p10 += cell.p10 * weight
                p50 += cell.p50 * weight
                p90 += cell.p90 * weight
                p98 += cell.p98 * weight
                highlightFraction += cell.highlightFraction * weight
            }
        }
        if (weightSum <= 0f) {
            return HdrPgtmCellStats(
                p10 = 0.02f,
                p50 = 0.18f,
                p90 = 0.55f,
                p98 = 0.82f,
                highlightFraction = 0f,
                sampleWeight = 1f
            )
        }
        return HdrPgtmCellStats(
            p10 = p10 / weightSum,
            p50 = p50 / weightSum,
            p90 = p90 / weightSum,
            p98 = p98 / weightSum,
            highlightFraction = highlightFraction / weightSum,
            sampleWeight = weightSum
        )
    }

    private fun buildMap(
        grid: HdrPgtmGrid,
        safeBaselineEv: Float,
        safePointCount: Int,
        curveParams: Array<HdrPgtmCurveParams>,
    ): DngProfileGainTableMap {
        val gains = FloatArray(grid.mapPointsH * grid.mapPointsV * safePointCount)
        for (y in 0 until grid.mapPointsV) {
            for (x in 0 until grid.mapPointsH) {
                val cellIndex = y * grid.mapPointsH + x
                writeHdrPgtmCurve(
                    output = gains,
                    outputOffset = cellIndex * safePointCount,
                    pointCount = safePointCount,
                    params = curveParams[cellIndex]
                )
            }
        }
        return DngProfileGainTableMap(
            mapPointsV = grid.mapPointsV,
            mapPointsH = grid.mapPointsH,
            mapSpacingV = grid.mapSpacingV,
            mapSpacingH = grid.mapSpacingH,
            mapOriginV = 0.0,
            mapOriginH = 0.0,
            mapPointsN = safePointCount,
            mapInputWeights = hdrPgtmInputWeights(safeBaselineEv),
            gamma = 1f,
            gains = gains,
            sourceTag = DngProfileGainTableMap.TAG_PROFILE_GAIN_TABLE_MAP2
        )
    }

    private fun chooseHdrPgtmGrid(width: Int, height: Int): HdrPgtmGrid {
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

    private fun buildHdrPgtmCurveParams(
        cell: HdrPgtmCellStats,
        global: HdrPgtmCellStats,
        baselineExposureEv: Float,
    ): HdrPgtmCurveParams {
        val globalDynamicRangeEv = log2((global.p98 + 0.006f) / (global.p10 + 0.006f))
            .coerceIn(0f, 8f)
        val sceneContrastStrength = smoothStep(1.8f, 5.5f, globalDynamicRangeEv)
        val hdrStrength = max(
            smoothStep(0.35f, 3.2f, baselineExposureEv),
            0.70f * sceneContrastStrength
        ).coerceIn(0f, 1f)
        val globalMedian = global.p50.coerceIn(0.025f, 0.85f)
        val cellMedian = cell.p50.coerceIn(0.002f, 1f)
        val medianEv = log2(cellMedian / globalMedian)
        val brightRegion = smoothStep(0.25f, 1.7f, medianEv)
        val darkRegion = smoothStep(0.25f, 2.4f, -medianEv)
        val highlightPressure = (
            0.45f * smoothStep(0.48f, 0.88f, cell.p90) +
                0.35f * smoothStep(0.70f, 0.985f, cell.p98) +
                0.20f * smoothStep(0.015f, 0.30f, cell.highlightFraction)
            ).coerceIn(0f, 1f) * hdrStrength
        val combinedPressure = max(
            highlightPressure,
            0.60f * brightRegion * hdrStrength
        ).coerceIn(0f, 1f)
        val deepShadowLiftEv = 0.42f *
            (1f - smoothStep(0.015f, 0.050f, cell.p90)) *
            hdrStrength
        val localDodgeEv = (
            deepShadowLiftEv +
                0.10f * darkRegion * (1f - smoothStep(0.05f, 0.18f, cell.p90))
            ).coerceIn(0f, 0.52f)
        val localBurnEv = (
            0.46f * smoothStep(0.18f, 0.88f, cell.p90) +
                0.20f * brightRegion +
                0.18f * combinedPressure
            ).coerceIn(0f, 0.72f) * hdrStrength
        val lowGain = hdrPgtmBaseLowGain(baselineExposureEv) *
            2.0f.pow((localDodgeEv - localBurnEv).coerceIn(-0.68f, 0.46f))
        return toneMappedHdrPgtmCurveParams(
            lowGain = lowGain,
            highlightGain = hdrPgtmHighlightGain(baselineExposureEv),
            highlightPressure = combinedPressure,
            statsShoulderPivot = hdrPgtmStatsShoulderPivot(
                cell = cell,
                global = global,
                highlightPressure = combinedPressure
            )
        )
    }

    private fun baselineHdrPgtmCurveParams(baselineExposureEv: Float): HdrPgtmCurveParams {
        val highlightPressure = smoothStep(0.35f, 3.2f, baselineExposureEv)
        return toneMappedHdrPgtmCurveParams(
            lowGain = hdrPgtmBaseLowGain(baselineExposureEv),
            highlightGain = hdrPgtmHighlightGain(baselineExposureEv),
            highlightPressure = highlightPressure,
            statsShoulderPivot = 0.12f
        )
    }

    private fun toneMappedHdrPgtmCurveParams(
        lowGain: Float,
        highlightGain: Float,
        highlightPressure: Float,
        statsShoulderPivot: Float,
    ): HdrPgtmCurveParams {
        val safeLowGain = lowGain.coerceIn(MIN_LOW_GAIN, MAX_LOW_GAIN)
        val safeHighlightGain = highlightGain.coerceIn(MIN_HIGHLIGHT_GAIN, MAX_HIGHLIGHT_GAIN)
        val pressure = highlightPressure.coerceIn(0f, 1f)
        val highlightRatio = safeHighlightGain / max(safeLowGain, 1e-6f)
        val dynamicRangePressure = 1f - smoothStep(0.10f, 0.45f, highlightRatio)
        val shoulderStrength = max(pressure, dynamicRangePressure).coerceIn(0f, 1f)
        val exposurePivot = lerp(0.18f, IPHONE_AVERAGE_SHOULDER_PIVOT, shoulderStrength)
        val requestedPivot = min(
            exposurePivot,
            statsShoulderPivot.takeIf { it.isFinite() && it > 0f } ?: exposurePivot
        )
        return HdrPgtmCurveParams(
            lowGain = safeLowGain,
            highlightGain = safeHighlightGain,
            shoulderPivot = requestedPivot.coerceIn(MIN_SHOULDER_PIVOT, MAX_SHOULDER_PIVOT),
            shoulderPower = lerp(1.40f, IPHONE_AVERAGE_SHOULDER_POWER, shoulderStrength),
            tailPower = lerp(2.8f, IPHONE_AVERAGE_TAIL_POWER, shoulderStrength),
            highlightPressure = pressure
        )
    }

    private fun hdrPgtmStatsShoulderPivot(
        cell: HdrPgtmCellStats,
        global: HdrPgtmCellStats,
        highlightPressure: Float,
    ): Float {
        val pressure = highlightPressure.coerceIn(0f, 1f)
        val localPivot = min(
            cell.p90 * lerp(0.56f, 0.34f, pressure),
            cell.p98 * lerp(0.40f, 0.25f, pressure)
        )
        val globalPivot = min(
            global.p90 * lerp(0.68f, 0.46f, pressure),
            global.p98 * lerp(0.50f, 0.32f, pressure)
        )
        val statsPivot = lerp(globalPivot, localPivot, 0.68f)
        return lerp(0.18f, statsPivot, pressure).coerceIn(MIN_SHOULDER_PIVOT, MAX_SHOULDER_PIVOT)
    }

    private fun smoothHdrPgtmCurveParams(
        stats: Array<HdrPgtmCurveParams>,
        grid: HdrPgtmGrid,
    ): Array<HdrPgtmCurveParams> {
        var current = stats
        repeat(2) {
            val next = Array(current.size) { current[it] }
            for (y in 0 until grid.mapPointsV) {
                for (x in 0 until grid.mapPointsH) {
                    var weightSum = 0f
                    var lowGain = 0f
                    var highlightGain = 0f
                    var shoulderPivot = 0f
                    var shoulderPower = 0f
                    var tailPower = 0f
                    var highlightPressure = 0f
                    for (dy in -1..1) {
                        for (dx in -1..1) {
                            val nx = (x + dx).coerceIn(0, grid.mapPointsH - 1)
                            val ny = (y + dy).coerceIn(0, grid.mapPointsV - 1)
                            val weight = if (dx == 0 && dy == 0) 4f else if (dx == 0 || dy == 0) 2f else 1f
                            val value = current[ny * grid.mapPointsH + nx]
                            weightSum += weight
                            lowGain += value.lowGain * weight
                            highlightGain += value.highlightGain * weight
                            shoulderPivot += value.shoulderPivot * weight
                            shoulderPower += value.shoulderPower * weight
                            tailPower += value.tailPower * weight
                            highlightPressure += value.highlightPressure * weight
                        }
                    }
                    val index = y * grid.mapPointsH + x
                    next[index] = HdrPgtmCurveParams(
                        lowGain = lowGain / weightSum,
                        highlightGain = highlightGain / weightSum,
                        shoulderPivot = shoulderPivot / weightSum,
                        shoulderPower = shoulderPower / weightSum,
                        tailPower = tailPower / weightSum,
                        highlightPressure = highlightPressure / weightSum,
                    )
                }
            }
            current = next
        }
        return current
    }

    private fun writeHdrPgtmCurve(
        output: FloatArray,
        outputOffset: Int,
        pointCount: Int,
        params: HdrPgtmCurveParams,
    ) {
        val lowGain = params.lowGain.coerceIn(MIN_LOW_GAIN, MAX_LOW_GAIN)
        val highlightGain = params.highlightGain.coerceIn(MIN_HIGHLIGHT_GAIN, MAX_HIGHLIGHT_GAIN)
        val localDodgeCompression = hdrPgtmLocalDodgeCompression(lowGain)
        val localHighlightCompression = hdrPgtmLocalHighlightCompression(
            lowGain = lowGain,
            highlightPressure = params.highlightPressure
        )
        val dodgePivot = lerp(
            params.shoulderPivot,
            IPHONE_DODGE_SHOULDER_PIVOT,
            localDodgeCompression
        )
        val dodgePower = lerp(
            params.shoulderPower,
            IPHONE_DODGE_SHOULDER_POWER,
            localDodgeCompression
        )
        val dodgeTailPower = lerp(
            params.tailPower,
            IPHONE_DODGE_TAIL_POWER,
            localDodgeCompression
        )
        val shoulderPivot = lerp(
            dodgePivot,
            IPHONE_HIGHLIGHT_SHOULDER_PIVOT,
            localHighlightCompression
        ).coerceIn(0.035f, MAX_SHOULDER_PIVOT)
        val shoulderPower = lerp(
            dodgePower,
            IPHONE_HIGHLIGHT_SHOULDER_POWER,
            localHighlightCompression
        ).coerceIn(0.95f, 2.20f)
        val tailPower = lerp(
            dodgeTailPower,
            IPHONE_HIGHLIGHT_TAIL_POWER,
            localHighlightCompression
        ).coerceIn(1.0f, 4.0f)
        for (index in 0 until pointCount) {
            val input = tableInputForIndex(index, pointCount)
            if (input <= MIN_CURVE_INPUT) {
                output[outputOffset + index] = lowGain
                continue
            }
            output[outputOffset + index] = iphoneStyleHdrPgtmGain(
                input = input,
                lowGain = lowGain,
                highlightGain = highlightGain,
                shoulderPivot = shoulderPivot,
                shoulderPower = shoulderPower,
                tailPower = tailPower
            ).coerceIn(
                0.05f,
                2.2f
            )
        }
    }

    private fun iphoneStyleHdrPgtmGain(
        input: Float,
        lowGain: Float,
        highlightGain: Float,
        shoulderPivot: Float,
        shoulderPower: Float,
        tailPower: Float,
    ): Float {
        val rawRoll = rationalShoulderRoll(input, shoulderPivot, shoulderPower)
        val endpointRoll = rationalShoulderRoll(1f, shoulderPivot, shoulderPower)
        val endpointCompensation = endpointRoll * input.coerceIn(0f, 1f).pow(tailPower)
        val shoulderRoll = (rawRoll - endpointCompensation).coerceIn(0f, 1f)
        return highlightGain + (lowGain - highlightGain) * shoulderRoll
    }

    private fun rationalShoulderRoll(input: Float, pivot: Float, power: Float): Float {
        val ratio = input.coerceAtLeast(0f) / max(pivot, MIN_CURVE_INPUT)
        return 1f / (1f + ratio.pow(power))
    }

    private fun hdrPgtmLocalHighlightCompression(
        lowGain: Float,
        highlightPressure: Float,
    ): Float {
        val localBurn = 1f - smoothStep(0.84f, 1.18f, lowGain)
        val pressure = smoothStep(0.20f, 0.85f, highlightPressure.coerceIn(0f, 1f))
        return (localBurn * pressure).coerceIn(0f, 1f)
    }

    private fun hdrPgtmLocalDodgeCompression(lowGain: Float): Float {
        return smoothStep(1.30f, 1.65f, lowGain)
    }

    private fun hdrPgtmInputWeights(baselineExposureEv: Float): FloatArray {
        val scale = hdrPgtmHighlightGain(baselineExposureEv)
        return FloatArray(MAP_INPUT_WEIGHT_COUNT) { index ->
            BASE_INPUT_WEIGHTS[index] * scale
        }
    }

    private fun hdrPgtmBaseLowGain(baselineExposureEv: Float): Float {
        return lerp(1.06f, 1.25f, smoothStep(1.0f, 3.35f, baselineExposureEv))
    }

    private fun hdrPgtmHighlightGain(baselineExposureEv: Float): Float {
        val baselineGain = 2.0f.pow(baselineExposureEv.coerceIn(0f, MAX_BASELINE_EV))
        return (INPUT_HEADROOM / baselineGain)
            .coerceIn(MIN_HIGHLIGHT_GAIN, MAX_HIGHLIGHT_GAIN)
    }

    private fun tableInputForIndex(index: Int, pointCount: Int): Float {
        if (pointCount <= 1) return 0f
        return if (index == pointCount - 1) {
            1f
        } else {
            index.toFloat() / pointCount.toFloat()
        }
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

    private data class HdrPgtmGrid(
        val mapPointsH: Int,
        val mapPointsV: Int,
        val mapSpacingH: Double,
        val mapSpacingV: Double,
    )

    private data class HdrPgtmCellStats(
        val p10: Float,
        val p50: Float,
        val p90: Float,
        val p98: Float,
        val highlightFraction: Float,
        val sampleWeight: Float,
    )

    private data class HdrPgtmCurveParams(
        val lowGain: Float,
        val highlightGain: Float,
        val shoulderPivot: Float,
        val shoulderPower: Float,
        val tailPower: Float,
        val highlightPressure: Float,
    )
}
