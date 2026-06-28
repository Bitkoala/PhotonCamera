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
    private const val TARGET_TILE_PX = 64
    private const val GRID_MIN_H = 8
    private const val GRID_MIN_V = 6
    private const val GRID_MAX_H = 64
    private const val GRID_MAX_V = 48
    private const val INPUT_HEADROOM = 1.12f
    private const val MIN_CURVE_INPUT = 1e-6f
    private const val SCENE_SHADOW_ANCHOR = 0.02f
    private const val SCENE_LOW_ANCHOR = 0.05f
    private const val SCENE_MID_ANCHOR = 0.18f
    private const val SCENE_HALF_ANCHOR = 0.50f
    private const val SCENE_WHITE_ANCHOR = 1.0f
    private const val SCENE_BRIGHT_ANCHOR = 1.5f
    private const val SCENE_SUPER_WHITE_ANCHOR = 2.0f
    private const val MIN_SHADOW_GAIN = 1.70f
    private const val MAX_SHADOW_GAIN = 3.50f
    private const val MIN_MID_GAIN = 1.45f
    private const val MAX_MID_GAIN = 2.40f
    private const val MIN_WHITE_OUTPUT = 0.82f
    private const val MAX_WHITE_OUTPUT = 0.93f
    private const val MIN_BRIGHT_OUTPUT = 0.90f
    private const val MAX_BRIGHT_OUTPUT = 0.995f
    private const val MIN_SUPER_WHITE_OUTPUT = 0.97f
    private const val MAX_SUPER_WHITE_OUTPUT = 0.999f
    private val BASE_INPUT_WEIGHTS = floatArrayOf(
        0.1495f,
        0.2935f,
        0.0570f,
        0.1250f,
        0.3750f
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

    fun forHdrBaselineExposureLike(
        baselineExposureEv: Float,
        template: DngProfileGainTableMap,
    ): DngProfileGainTableMap? {
        if (!template.isValid || !baselineExposureEv.isFinite() || baselineExposureEv <= MIN_BASELINE_EV) {
            return null
        }
        val safeBaselineEv = baselineExposureEv.coerceIn(0f, MAX_BASELINE_EV)
        val safePointCount = template.mapPointsN.coerceIn(MIN_TABLE_POINTS, MAX_TABLE_POINTS)
        val cellCount = template.mapPointsH * template.mapPointsV
        val params = baselineHdrPgtmCurveParams(safeBaselineEv)
        return buildMap(
            grid = HdrPgtmGrid(
                mapPointsH = template.mapPointsH,
                mapPointsV = template.mapPointsV,
                mapSpacingH = template.mapSpacingH,
                mapSpacingV = template.mapSpacingV
            ),
            safeBaselineEv = safeBaselineEv,
            safePointCount = safePointCount,
            curveParams = Array(cellCount) { params }
        ).copy(
            mapOriginH = template.mapOriginH,
            mapOriginV = template.mapOriginV
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
                    inputScale = hdrPgtmInputScale(safeBaselineEv),
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
        val shadowGain = hdrPgtmBaseShadowGain(baselineExposureEv) *
            2.0f.pow((localDodgeEv - 0.55f * localBurnEv).coerceIn(-0.42f, 0.58f))
        val midGain = hdrPgtmBaseMidGain(baselineExposureEv) *
            2.0f.pow((0.34f * localDodgeEv - 0.22f * localBurnEv).coerceIn(-0.22f, 0.24f))
        val halfOutput = (
            0.64f -
                0.035f * combinedPressure +
                0.010f * darkRegion * hdrStrength
            ).coerceIn(0.56f, 0.70f)
        val whiteOutput = (
            0.89f -
                0.045f * combinedPressure -
                0.010f * brightRegion * hdrStrength
            ).coerceIn(MIN_WHITE_OUTPUT, MAX_WHITE_OUTPUT)
        val brightOutput = (
            0.975f -
                0.045f * combinedPressure -
                0.010f * brightRegion * hdrStrength
            ).coerceIn(max(whiteOutput + 0.035f, MIN_BRIGHT_OUTPUT), MAX_BRIGHT_OUTPUT)
        val superWhiteOutput = lerp(0.997f, 0.985f, combinedPressure)
            .coerceIn(max(brightOutput + 0.012f, MIN_SUPER_WHITE_OUTPUT), MAX_SUPER_WHITE_OUTPUT)
        return HdrPgtmCurveParams(
            shadowGain = shadowGain.coerceIn(MIN_SHADOW_GAIN, MAX_SHADOW_GAIN),
            midGain = midGain.coerceIn(MIN_MID_GAIN, MAX_MID_GAIN),
            halfOutput = halfOutput,
            whiteOutput = whiteOutput,
            brightOutput = brightOutput,
            superWhiteOutput = superWhiteOutput,
            highlightPressure = combinedPressure
        )
    }

    private fun baselineHdrPgtmCurveParams(baselineExposureEv: Float): HdrPgtmCurveParams {
        val highlightPressure = smoothStep(0.35f, 3.2f, baselineExposureEv)
        return HdrPgtmCurveParams(
            shadowGain = hdrPgtmBaseShadowGain(baselineExposureEv),
            midGain = hdrPgtmBaseMidGain(baselineExposureEv),
            halfOutput = lerp(0.65f, 0.63f, highlightPressure),
            whiteOutput = lerp(0.90f, 0.875f, highlightPressure).coerceIn(MIN_WHITE_OUTPUT, MAX_WHITE_OUTPUT),
            brightOutput = lerp(0.98f, 0.945f, highlightPressure)
                .coerceIn(MIN_BRIGHT_OUTPUT, MAX_BRIGHT_OUTPUT),
            superWhiteOutput = lerp(0.997f, 0.988f, highlightPressure)
                .coerceIn(MIN_SUPER_WHITE_OUTPUT, MAX_SUPER_WHITE_OUTPUT),
            highlightPressure = highlightPressure.coerceIn(0f, 1f)
        )
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
                    var shadowGain = 0f
                    var midGain = 0f
                    var halfOutput = 0f
                    var whiteOutput = 0f
                    var brightOutput = 0f
                    var superWhiteOutput = 0f
                    var highlightPressure = 0f
                    for (dy in -1..1) {
                        for (dx in -1..1) {
                            val nx = (x + dx).coerceIn(0, grid.mapPointsH - 1)
                            val ny = (y + dy).coerceIn(0, grid.mapPointsV - 1)
                            val weight = if (dx == 0 && dy == 0) 4f else if (dx == 0 || dy == 0) 2f else 1f
                            val value = current[ny * grid.mapPointsH + nx]
                            weightSum += weight
                            shadowGain += value.shadowGain * weight
                            midGain += value.midGain * weight
                            halfOutput += value.halfOutput * weight
                            whiteOutput += value.whiteOutput * weight
                            brightOutput += value.brightOutput * weight
                            superWhiteOutput += value.superWhiteOutput * weight
                            highlightPressure += value.highlightPressure * weight
                        }
                    }
                    val index = y * grid.mapPointsH + x
                    next[index] = HdrPgtmCurveParams(
                        shadowGain = shadowGain / weightSum,
                        midGain = midGain / weightSum,
                        halfOutput = halfOutput / weightSum,
                        whiteOutput = whiteOutput / weightSum,
                        brightOutput = brightOutput / weightSum,
                        superWhiteOutput = superWhiteOutput / weightSum,
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
        inputScale: Float,
        params: HdrPgtmCurveParams,
    ) {
        val safeInputScale = inputScale.coerceIn(1e-4f, INPUT_HEADROOM)
        val sceneMax = (1f / safeInputScale).coerceAtLeast(SCENE_WHITE_ANCHOR + 1e-3f)
        val anchors = buildSceneAnchors(params, sceneMax)
        for (index in 0 until pointCount) {
            val input = tableInputForIndex(index, pointCount)
            if (input <= MIN_CURVE_INPUT) {
                output[outputOffset + index] = params.shadowGain
                continue
            }
            val sceneLinear = input / safeInputScale
            val outputLinear = sampleSceneCurve(anchors, sceneLinear)
            output[outputOffset + index] = (outputLinear / max(sceneLinear, MIN_CURVE_INPUT))
                .coerceIn(0.05f, 4.0f)
        }
    }

    private fun buildSceneAnchors(params: HdrPgtmCurveParams, sceneMax: Float): List<Pair<Float, Float>> {
        val anchors = ArrayList<Pair<Float, Float>>(7)
        fun addAnchor(scene: Float, output: Float) {
            if (!scene.isFinite() || scene <= 0f || scene >= sceneMax) return
            val previous = anchors.lastOrNull()?.second ?: 0f
            anchors += scene to output.coerceAtLeast(previous + 1e-5f)
        }
        anchors += 0f to 0f
        val shadowGain = params.shadowGain.coerceIn(MIN_SHADOW_GAIN, MAX_SHADOW_GAIN)
        val midGain = params.midGain.coerceIn(MIN_MID_GAIN, MAX_MID_GAIN)
        val baseHalfOutput = params.halfOutput.coerceIn(0.52f, 0.74f)
        val baseWhiteOutput = params.whiteOutput.coerceIn(MIN_WHITE_OUTPUT, MAX_WHITE_OUTPUT)
        val baseBrightOutput = params.brightOutput
            .coerceIn(max(baseWhiteOutput + 0.025f, MIN_BRIGHT_OUTPUT), MAX_BRIGHT_OUTPUT)
        val baseSuperWhiteOutput = params.superWhiteOutput
            .coerceIn(max(baseBrightOutput + 0.008f, MIN_SUPER_WHITE_OUTPUT), MAX_SUPER_WHITE_OUTPUT)
        val midOutput = SCENE_MID_ANCHOR * midGain
        addAnchor(SCENE_SHADOW_ANCHOR, SCENE_SHADOW_ANCHOR * shadowGain)
        addAnchor(SCENE_LOW_ANCHOR, SCENE_LOW_ANCHOR * shadowGain)
        addAnchor(SCENE_MID_ANCHOR, midOutput)
        addAnchor(SCENE_HALF_ANCHOR, baseHalfOutput.coerceAtLeast(midOutput + 0.08f))
        addAnchor(SCENE_WHITE_ANCHOR, baseWhiteOutput)
        addAnchor(SCENE_BRIGHT_ANCHOR, baseBrightOutput)
        addAnchor(SCENE_SUPER_WHITE_ANCHOR, baseSuperWhiteOutput)
        anchors += sceneMax to max(1f, (anchors.lastOrNull()?.second ?: 0f) + 1e-5f)
        return anchors
    }

    private fun sampleSceneCurve(anchors: List<Pair<Float, Float>>, sceneLinear: Float): Float {
        if (sceneLinear <= 0f) return 0f
        for (index in 0 until anchors.lastIndex) {
            val start = anchors[index]
            val end = anchors[index + 1]
            if (sceneLinear <= end.first) {
                val rawT = ((sceneLinear - start.first) / max(end.first - start.first, 1e-6f))
                    .coerceIn(0f, 1f)
                val t = if (end.first <= SCENE_MID_ANCHOR) {
                    rawT
                } else {
                    smoothStep(0f, 1f, rawT)
                }
                return lerp(start.second, end.second, t)
            }
        }
        return anchors.last().second
    }

    private fun hdrPgtmInputWeights(baselineExposureEv: Float): FloatArray {
        val scale = hdrPgtmInputScale(baselineExposureEv)
        return FloatArray(MAP_INPUT_WEIGHT_COUNT) { index ->
            BASE_INPUT_WEIGHTS[index] * scale
        }
    }

    private fun hdrPgtmBaseShadowGain(baselineExposureEv: Float): Float {
        return lerp(2.45f, 2.58f, smoothStep(0.8f, 3.35f, baselineExposureEv))
            .coerceIn(MIN_SHADOW_GAIN, MAX_SHADOW_GAIN)
    }

    private fun hdrPgtmBaseMidGain(baselineExposureEv: Float): Float {
        return lerp(1.82f, 1.93f, smoothStep(1.0f, 3.35f, baselineExposureEv))
            .coerceIn(MIN_MID_GAIN, MAX_MID_GAIN)
    }

    private fun hdrPgtmInputScale(baselineExposureEv: Float): Float {
        val baselineGain = 2.0f.pow(baselineExposureEv.coerceIn(0f, MAX_BASELINE_EV))
        return (INPUT_HEADROOM / baselineGain)
            .coerceIn(1e-4f, INPUT_HEADROOM)
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
        val shadowGain: Float,
        val midGain: Float,
        val halfOutput: Float,
        val whiteOutput: Float,
        val brightOutput: Float,
        val superWhiteOutput: Float,
        val highlightPressure: Float,
    )
}
