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
    private const val MIN_BASELINE_EV = 0f
    private const val MAX_BASELINE_EV = 8f
    private const val DEFAULT_TABLE_POINTS = 257
    private const val MIN_TABLE_POINTS = 257
    private const val MAX_TABLE_POINTS = 257
    private const val TARGET_TILE_PX = 64
    private const val GRID_MIN_H = 8
    private const val GRID_MIN_V = 6
    private const val GRID_MAX_H = 64
    private const val GRID_MAX_V = 48
    private const val MIN_EFFECTIVE_INPUT_HEADROOM = 0.28f
    private const val MAX_EFFECTIVE_INPUT_HEADROOM = 3.05f
    private const val MIN_SCENE_MAX = 1.02f
    private const val LOW_KEY_LOCAL_TAIL_SCENE_MAX_FACTOR = 2.75f
    private const val LOW_KEY_LOCAL_TAIL_HDR_STRENGTH = 0.58f
    private const val MAX_GAIN_VALUE = 5.40f
    private const val MIN_CURVE_INPUT = 1e-6f
    private const val SCENE_SHADOW_ANCHOR = 0.02f
    private const val SCENE_LOW_ANCHOR = 0.05f
    private const val SCENE_MID_ANCHOR = 0.18f
    private const val SCENE_HALF_ANCHOR = 0.50f
    private const val SCENE_WHITE_ANCHOR = 1.0f
    private const val SCENE_BRIGHT_ANCHOR = 1.5f
    private const val SCENE_SUPER_WHITE_ANCHOR = 2.0f
    private const val SCENE_EXTREME_WHITE_ANCHOR = 3.0f
    private const val MIN_SHADOW_GAIN = 1.65f
    private const val MAX_SHADOW_GAIN = 5.40f
    private const val MIN_MID_GAIN = 1.40f
    private const val MAX_MID_GAIN = 2.75f
    private const val MIN_HALF_OUTPUT = 0.56f
    private const val MAX_HALF_OUTPUT = 0.68f
    private const val MIN_WHITE_OUTPUT = 0.70f
    private const val MAX_WHITE_OUTPUT = 0.995f
    private const val MIN_BRIGHT_OUTPUT = 0.80f
    private const val MAX_BRIGHT_OUTPUT = 1.0f
    private const val MIN_SUPER_WHITE_OUTPUT = 0.88f
    private const val MAX_SUPER_WHITE_OUTPUT = 1.0f
    private const val MIN_EXTREME_WHITE_OUTPUT = 0.97f
    private const val MAX_EXTREME_WHITE_OUTPUT = 1.0f
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

    fun forCellStats(
        width: Int,
        height: Int,
        baselineExposureEv: Float,
        packedCellStats: FloatArray,
        tablePointCount: Int = DEFAULT_TABLE_POINTS,
    ): DngProfileGainTableMap? {
        if (width <= 0 || height <= 0 || !baselineExposureEv.isFinite() || baselineExposureEv < MIN_BASELINE_EV) {
            return null
        }
        val grid = chooseHdrPgtmGrid(width, height)
        val cellCount = grid.mapPointsH * grid.mapPointsV
        if (packedCellStats.size < cellCount * CELL_STATS_FLOAT_STRIDE) {
            PLog.w(
                TAG,
                "GPU PGTM stats too small: ${packedCellStats.size}, expected=${cellCount * CELL_STATS_FLOAT_STRIDE}"
            )
            return null
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
        val inputScale = hdrPgtmInputScaleForStats(globalStats, safeBaselineEv)
        val sceneHdrStrength = max(
            sceneHdrStrength(inputScale),
            LOW_KEY_LOCAL_TAIL_HDR_STRENGTH * lowKeyLocalTailStrength(globalStats)
        ).coerceIn(0f, 1f)
        val curveParams = smoothHdrPgtmCurveParams(
            stats = Array(cellCount) { index ->
                buildHdrPgtmCurveParams(
                    cell = cells[index] ?: globalStats,
                    global = globalStats,
                    baselineExposureEv = safeBaselineEv,
                    sceneHdrStrength = sceneHdrStrength
                )
            },
            grid = grid
        )
        return buildMap(
            grid = grid,
            safeBaselineEv = safeBaselineEv,
            safePointCount = safePointCount,
            curveParams = curveParams,
            inputScale = inputScale
        )
    }

    private fun packedStatsAt(stats: FloatArray, offset: Int, sampleWeight: Float): HdrPgtmCellStats {
        val p10 = safe01(stats[offset])
        val p50 = max(p10, safe01(stats[offset + 1]))
        val p90 = max(p50, safe01(stats[offset + 2]))
        val p98 = max(p90, safe01(stats[offset + 3]))
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
        val validCells = ArrayList<HdrPgtmCellStats>(cells.size)
        cells.forEach { cell ->
            if (cell != null && cell.sampleWeight > 0f) {
                val weight = cell.sampleWeight
                validCells += cell
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
                p10 = 0.02f,
                p50 = 0.18f,
                p90 = 0.55f,
                p98 = 0.82f,
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
        val inputTailP95 = weightedPercentile(validCells, 0.95f) { it.p995Input }
        val inputTailP98 = weightedPercentile(validCells, 0.98f) { it.p995Input }
        val inputTailP99 = weightedPercentile(validCells, 0.99f) { it.p999Input }
        val maxInput = validCells.maxOfOrNull { it.p999Input } ?: (p999Input / weightSum)
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

    private fun weightedPercentile(
        cells: List<HdrPgtmCellStats>,
        percentile: Float,
        selector: (HdrPgtmCellStats) -> Float,
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
            if (cumulative >= target) {
                return value
            }
        }
        return sorted.last().first
    }

    private fun buildMap(
        grid: HdrPgtmGrid,
        safeBaselineEv: Float,
        safePointCount: Int,
        curveParams: Array<HdrPgtmCurveParams>,
        inputScale: Float,
    ): DngProfileGainTableMap {
        val safeInputScale = sanitizeInputScale(inputScale)
        val gains = FloatArray(grid.mapPointsH * grid.mapPointsV * safePointCount)
        for (y in 0 until grid.mapPointsV) {
            for (x in 0 until grid.mapPointsH) {
                val cellIndex = y * grid.mapPointsH + x
                writeHdrPgtmCurve(
                    output = gains,
                    outputOffset = cellIndex * safePointCount,
                    pointCount = safePointCount,
                    inputScale = safeInputScale,
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
            mapInputWeights = hdrPgtmInputWeights(safeInputScale),
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
        sceneHdrStrength: Float,
    ): HdrPgtmCurveParams {
        val globalDynamicRangeEv = log2((global.p98 + 0.006f) / (global.p10 + 0.006f))
            .coerceIn(0f, 8f)
        val sceneContrastStrength = smoothStep(1.8f, 5.5f, globalDynamicRangeEv)
        val hdrStrength = (max(
            smoothStep(0.35f, 3.2f, baselineExposureEv),
            0.70f * sceneContrastStrength
        ) * sceneHdrStrength).coerceIn(0f, 1f)
        val shadowSceneStrength = shadowSceneStrength(global, sceneHdrStrength)
        val midSceneStrength = midSceneStrength(global, sceneHdrStrength, shadowSceneStrength)
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
            (1f - smoothStep(0.020f, 0.115f, cell.p90)) *
            shadowSceneStrength
        val localDodgeEv = (
            deepShadowLiftEv +
                0.16f * darkRegion *
                (1f - smoothStep(0.08f, 0.32f, cell.p90)) *
                shadowSceneStrength
            ).coerceIn(0f, 0.52f)
        val localBurnEv = (
            0.46f * smoothStep(0.18f, 0.88f, cell.p90) +
                0.20f * brightRegion +
                0.18f * combinedPressure
            ).coerceIn(0f, 0.72f) * hdrStrength
        val shadowGain = hdrPgtmBaseShadowGain(baselineExposureEv, shadowSceneStrength) *
            2.0f.pow((0.95f * localDodgeEv - 0.42f * localBurnEv).coerceIn(-0.34f, 0.42f))
        val midGain = hdrPgtmBaseMidGain(baselineExposureEv, midSceneStrength) *
            2.0f.pow((0.54f * localDodgeEv - 0.18f * localBurnEv).coerceIn(-0.16f, 0.28f))
        val halfOutput = (
            hdrPgtmBaseHalfOutput(baselineExposureEv, sceneHdrStrength) -
                0.032f * combinedPressure +
                0.018f * darkRegion * hdrStrength
            ).coerceIn(MIN_HALF_OUTPUT, MAX_HALF_OUTPUT)
        val whiteOutput = (
            hdrPgtmBaseWhiteOutput(baselineExposureEv, sceneHdrStrength) -
                0.040f * combinedPressure -
                0.014f * brightRegion * hdrStrength
            ).coerceIn(MIN_WHITE_OUTPUT, MAX_WHITE_OUTPUT)
        val brightOutput = (
            hdrPgtmBaseBrightOutput(baselineExposureEv, sceneHdrStrength) -
                0.045f * combinedPressure -
                0.012f * brightRegion * hdrStrength
            ).coerceInNonEmpty(max(whiteOutput + 0.035f, MIN_BRIGHT_OUTPUT), MAX_BRIGHT_OUTPUT)
        val superWhiteOutput = (
            hdrPgtmBaseSuperWhiteOutput(baselineExposureEv, sceneHdrStrength) -
                0.042f * combinedPressure -
                0.008f * brightRegion * hdrStrength
            )
            .coerceInNonEmpty(max(brightOutput + 0.012f, MIN_SUPER_WHITE_OUTPUT), MAX_SUPER_WHITE_OUTPUT)
        val extremeWhiteOutput = (
            hdrPgtmBaseExtremeWhiteOutput(baselineExposureEv, sceneHdrStrength) -
                0.010f * combinedPressure
            ).coerceInNonEmpty(max(superWhiteOutput + 0.020f, MIN_EXTREME_WHITE_OUTPUT), MAX_EXTREME_WHITE_OUTPUT)
        return HdrPgtmCurveParams(
            shadowGain = shadowGain.coerceIn(MIN_SHADOW_GAIN, MAX_SHADOW_GAIN),
            midGain = midGain.coerceIn(MIN_MID_GAIN, MAX_MID_GAIN),
            halfOutput = halfOutput,
            whiteOutput = whiteOutput,
            brightOutput = brightOutput,
            superWhiteOutput = superWhiteOutput,
            extremeWhiteOutput = extremeWhiteOutput,
            highlightPressure = combinedPressure
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
                    var extremeWhiteOutput = 0f
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
                            extremeWhiteOutput += value.extremeWhiteOutput * weight
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
                        extremeWhiteOutput = extremeWhiteOutput / weightSum,
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
        val safeInputScale = sanitizeInputScale(inputScale)
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
                .coerceIn(0.05f, MAX_GAIN_VALUE)
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
        val baseHalfOutput = params.halfOutput.coerceIn(MIN_HALF_OUTPUT, MAX_HALF_OUTPUT)
        val baseWhiteOutput = params.whiteOutput.coerceIn(MIN_WHITE_OUTPUT, MAX_WHITE_OUTPUT)
        val baseBrightOutput = params.brightOutput
            .coerceInNonEmpty(max(baseWhiteOutput + 0.025f, MIN_BRIGHT_OUTPUT), MAX_BRIGHT_OUTPUT)
        val baseSuperWhiteOutput = params.superWhiteOutput
            .coerceInNonEmpty(max(baseBrightOutput + 0.008f, MIN_SUPER_WHITE_OUTPUT), MAX_SUPER_WHITE_OUTPUT)
        val baseExtremeWhiteOutput = params.extremeWhiteOutput
            .coerceInNonEmpty(max(baseSuperWhiteOutput + 0.020f, MIN_EXTREME_WHITE_OUTPUT), MAX_EXTREME_WHITE_OUTPUT)
        val midOutput = SCENE_MID_ANCHOR * midGain
        addAnchor(SCENE_SHADOW_ANCHOR, SCENE_SHADOW_ANCHOR * shadowGain)
        addAnchor(SCENE_LOW_ANCHOR, SCENE_LOW_ANCHOR * shadowGain)
        addAnchor(SCENE_MID_ANCHOR, midOutput)
        addAnchor(SCENE_HALF_ANCHOR, baseHalfOutput.coerceAtLeast(midOutput + 0.08f))
        addAnchor(SCENE_WHITE_ANCHOR, baseWhiteOutput)
        addAnchor(SCENE_BRIGHT_ANCHOR, baseBrightOutput)
        addAnchor(SCENE_SUPER_WHITE_ANCHOR, baseSuperWhiteOutput)
        addAnchor(SCENE_EXTREME_WHITE_ANCHOR, baseExtremeWhiteOutput)
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

    private fun hdrPgtmInputWeights(inputScale: Float): FloatArray {
        val scale = sanitizeInputScale(inputScale)
        return FloatArray(MAP_INPUT_WEIGHT_COUNT) { index ->
            BASE_INPUT_WEIGHTS[index] * scale
        }
    }

    private fun hdrPgtmBaseShadowGain(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrGain = lerp(3.55f, 3.85f, smoothStep(0.8f, 2.2f, baselineExposureEv))
        return lerp(1.95f, hdrGain, sceneHdrStrength)
            .coerceIn(MIN_SHADOW_GAIN, MAX_SHADOW_GAIN)
    }

    private fun hdrPgtmBaseMidGain(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrGain = lerp(2.30f, 2.40f, smoothStep(0.8f, 2.2f, baselineExposureEv))
        return lerp(1.65f, hdrGain, sceneHdrStrength)
            .coerceIn(MIN_MID_GAIN, MAX_MID_GAIN)
    }

    private fun hdrPgtmBaseHalfOutput(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrOutput = lerp(0.64f, 0.615f, smoothStep(0.35f, 2.2f, baselineExposureEv))
        return lerp(0.655f, hdrOutput, sceneHdrStrength)
            .coerceIn(MIN_HALF_OUTPUT, MAX_HALF_OUTPUT)
    }

    private fun hdrPgtmBaseWhiteOutput(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrOutput = lerp(0.80f, 0.76f, smoothStep(0.35f, 2.2f, baselineExposureEv))
        return lerp(0.985f, hdrOutput, sceneHdrStrength)
            .coerceIn(MIN_WHITE_OUTPUT, MAX_WHITE_OUTPUT)
    }

    private fun hdrPgtmBaseBrightOutput(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrOutput = lerp(0.88f, 0.845f, smoothStep(0.35f, 2.2f, baselineExposureEv))
        return lerp(0.996f, hdrOutput, sceneHdrStrength)
            .coerceIn(MIN_BRIGHT_OUTPUT, MAX_BRIGHT_OUTPUT)
    }

    private fun hdrPgtmBaseSuperWhiteOutput(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrOutput = lerp(0.925f, 0.905f, smoothStep(0.35f, 2.2f, baselineExposureEv))
        return lerp(1.0f, hdrOutput, sceneHdrStrength)
            .coerceIn(MIN_SUPER_WHITE_OUTPUT, MAX_SUPER_WHITE_OUTPUT)
    }

    private fun hdrPgtmBaseExtremeWhiteOutput(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrOutput = lerp(0.985f, 0.990f, smoothStep(0.35f, 2.2f, baselineExposureEv))
        return lerp(1.0f, hdrOutput, sceneHdrStrength)
            .coerceIn(MIN_EXTREME_WHITE_OUTPUT, MAX_EXTREME_WHITE_OUTPUT)
    }

    private fun hdrPgtmInputScaleForStats(
        global: HdrPgtmCellStats,
        baselineExposureEv: Float,
    ): Float {
        val baselineGain = 2.0f.pow(baselineExposureEv.coerceIn(0f, MAX_BASELINE_EV))
        val tailP95 = global.inputTailP95.takeIf { it.isFinite() && it > 0f }
            ?: global.p995Input.takeIf { it.isFinite() && it > 0f }
            ?: global.p98.takeIf { it.isFinite() && it > 0f }
            ?: MIN_SCENE_MAX
        val tailP98 = max(tailP95, global.inputTailP98.takeIf { it.isFinite() && it > 0f } ?: tailP95)
        val tailP99 = max(tailP98, global.inputTailP99.takeIf { it.isFinite() && it > 0f } ?: tailP98)
        val maxInput = max(tailP99, global.maxInput.takeIf { it.isFinite() && it > 0f } ?: tailP99)
        val baseSceneMax = hdrPgtmSceneMaxFromTailStats(
            highlightFraction = global.highlightFraction,
            tailP95 = tailP95,
            tailP98 = tailP98,
            tailP99 = tailP99,
            maxInput = maxInput
        )
        val lowKeyTailStrength = lowKeyLocalTailStrength(global)
        val lowKeySceneMax = if (lowKeyTailStrength > 0f) {
            val localTailSceneMax = tailP99 * lerp(
                1.0f,
                LOW_KEY_LOCAL_TAIL_SCENE_MAX_FACTOR,
                lowKeyTailStrength
            )
            max(baseSceneMax, localTailSceneMax)
        } else {
            baseSceneMax
        }
        val sceneMax = lowKeySceneMax.coerceIn(
            max(MIN_SCENE_MAX, baselineGain / MAX_EFFECTIVE_INPUT_HEADROOM),
            baselineGain / MIN_EFFECTIVE_INPUT_HEADROOM
        )
        return sanitizeInputScale(1f / sceneMax)
    }

    private fun hdrPgtmSceneMaxFromTailStats(
        highlightFraction: Float,
        tailP95: Float,
        tailP98: Float,
        tailP99: Float,
        maxInput: Float,
    ): Float {
        val safeHighlightFraction = highlightFraction.coerceIn(0f, 1f)
        return when {
            safeHighlightFraction >= 0.20f -> {
                max(maxInput * 0.99f, tailP99 * 1.10f)
            }

            safeHighlightFraction >= 0.10f -> {
                max(tailP95, tailP98 * 0.98f)
            }

            safeHighlightFraction >= 0.03f -> {
                tailP99 * 1.06f
            }

            else -> {
                val outlierGapEv = log2((tailP99 + 0.04f) / (tailP98 + 0.04f))
                if (outlierGapEv >= 0.72f) {
                    tailP99
                } else {
                    min(tailP95 * 1.08f, tailP98)
                }
            }
        }
    }

    private fun sceneHdrStrength(inputScale: Float): Float {
        val sceneMax = 1f / sanitizeInputScale(inputScale)
        return smoothStep(1.12f, 2.35f, sceneMax)
    }

    private fun lowKeyLocalTailStrength(global: HdrPgtmCellStats): Float {
        val tailInput = max(global.inputTailP99, global.p999Input)
        if (!tailInput.isFinite() || tailInput <= 0f) return 0f
        val localTailGapEv = log2((tailInput + 0.04f) / (global.p98 + 0.04f))
        val lowUpperToneStrength = 1f - smoothStep(0.26f, 0.58f, global.p90)
        val lowHighlightToneStrength = 1f - smoothStep(0.24f, 0.52f, global.p98)
        val localTailStrength = smoothStep(0.82f, 1.55f, localTailGapEv)
        return (lowUpperToneStrength * lowHighlightToneStrength * localTailStrength)
            .coerceIn(0f, 1f)
    }

    private fun shadowSceneStrength(global: HdrPgtmCellStats, sceneHdrStrength: Float): Float {
        val darkMedianStrength = 1f - smoothStep(0.035f, 0.160f, global.p50)
        val deepShadowStrength = 1f - smoothStep(0.008f, 0.070f, global.p10)
        return max(
            sceneHdrStrength,
            max(0.82f * darkMedianStrength, 0.70f * deepShadowStrength)
        ).coerceIn(0f, 1f)
    }

    private fun midSceneStrength(
        global: HdrPgtmCellStats,
        sceneHdrStrength: Float,
        shadowSceneStrength: Float,
    ): Float {
        val darkMedianStrength = 1f - smoothStep(0.035f, 0.160f, global.p50)
        val deepShadowStrength = 1f - smoothStep(0.008f, 0.070f, global.p10)
        return max(
            0.82f * sceneHdrStrength,
            max(0.60f * darkMedianStrength, max(0.45f * deepShadowStrength, 0.88f * shadowSceneStrength))
        ).coerceIn(0f, 1f)
    }

    private fun sanitizeInputScale(inputScale: Float): Float {
        return inputScale.takeIf { it.isFinite() }
            ?.coerceIn(1e-4f, 1.0f)
            ?: 1e-4f
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

    private fun safePositive(value: Float, fallback: Float): Float {
        return value.takeIf { it.isFinite() && it > 0f } ?: fallback
    }

    private fun Float.coerceInNonEmpty(minimumValue: Float, maximumValue: Float): Float {
        return coerceIn(minimumValue.coerceAtMost(maximumValue), maximumValue)
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
        val p995Input: Float,
        val p999Input: Float,
        val inputTailP95: Float,
        val inputTailP98: Float,
        val inputTailP99: Float,
        val maxInput: Float,
        val sampleWeight: Float,
    )

    private data class HdrPgtmCurveParams(
        val shadowGain: Float,
        val midGain: Float,
        val halfOutput: Float,
        val whiteOutput: Float,
        val brightOutput: Float,
        val superWhiteOutput: Float,
        val extremeWhiteOutput: Float,
        val highlightPressure: Float,
    )
}
