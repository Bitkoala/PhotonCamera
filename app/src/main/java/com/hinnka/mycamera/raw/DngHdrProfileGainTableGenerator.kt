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
    private const val GCAM_HDR_LOWER_RANGE_EV = -1f
    private const val GCAM_HDR_UPPER_RANGE_EV = 8.5f
    private const val GCAM_HDR_EFFECT_INTENSITY = 1f
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
    private const val MIN_MID_GAIN = 1.40f
    private const val MAX_MID_GAIN = 2.75f
    private const val MIN_BLACK_GAIN = 1.65f
    private const val MAX_BLACK_GAIN = 5.40f
    private const val MIN_TOE_SLOPE = 0.58f
    private const val MAX_TOE_SLOPE = 0.88f
    private const val MIN_SHOULDER_POWER = 0.82f
    private const val MAX_SHOULDER_POWER = 1.28f
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
        val sceneMax = 1f / sanitizeInputScale(inputScale)
        val sceneHdrStrength = (max(
            sceneHdrStrength(inputScale),
            LOW_KEY_LOCAL_TAIL_HDR_STRENGTH * lowKeyLocalTailStrength(globalStats)
        ) * GCAM_HDR_EFFECT_INTENSITY).coerceIn(0f, 1f)
        val curveParams = smoothHdrPgtmCurveParams(
            stats = Array(cellCount) { index ->
                buildHdrPgtmCurveParams(
                    cell = cells[index] ?: globalStats,
                    global = globalStats,
                    baselineExposureEv = safeBaselineEv,
                    sceneHdrStrength = sceneHdrStrength,
                    sceneMax = sceneMax
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
        sceneMax: Float,
    ): HdrPgtmCurveParams {
        val globalDynamicRangeEv = log2((global.p98 + 0.006f) / (global.p10 + 0.006f))
            .coerceIn(0f, 8f)
        val sceneContrastStrength = smoothStep(1.8f, 5.5f, globalDynamicRangeEv)
        val hdrStrength = (max(
            smoothStep(0.35f, 3.2f, baselineExposureEv),
            0.70f * sceneContrastStrength
        ) * sceneHdrStrength).coerceIn(0f, 1f)
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
        val rangeCenterEv = hdrRangeCenterOffsetEv(
            cell = cell,
            darkRegion = darkRegion,
            brightRegion = brightRegion,
            highlightPressure = combinedPressure,
            hdrStrength = hdrStrength
        )
        val baseMidGain = hdrRangeMidGain(baselineExposureEv, sceneHdrStrength)
        val midGain = (baseMidGain * 2.0f.pow(rangeCenterEv))
            .coerceIn(MIN_MID_GAIN, MAX_MID_GAIN)
        val effectiveUpperRangeEv = log2(sceneMax / SCENE_MID_ANCHOR)
            .coerceIn(0.01f, GCAM_HDR_UPPER_RANGE_EV - GCAM_HDR_LOWER_RANGE_EV)
        val shoulderPower = hdrRangeShoulderPower(
            effectiveUpperRangeEv = effectiveUpperRangeEv,
            highlightPressure = combinedPressure,
            hdrStrength = hdrStrength
        )
        val toeSlope = hdrRangeToeSlope(sceneHdrStrength, combinedPressure)
        val blackGain = hdrRangeBlackGain(
            midGain = midGain,
            sceneHdrStrength = sceneHdrStrength,
            rangeCenterEv = rangeCenterEv,
            highlightPressure = combinedPressure
        )
        return HdrPgtmCurveParams(
            midOutput = SCENE_MID_ANCHOR * midGain,
            blackGain = blackGain,
            toeSlope = toeSlope,
            shoulderPower = shoulderPower,
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
                    var midOutput = 0f
                    var blackGain = 0f
                    var toeSlope = 0f
                    var shoulderPower = 0f
                    var highlightPressure = 0f
                    for (dy in -1..1) {
                        for (dx in -1..1) {
                            val nx = (x + dx).coerceIn(0, grid.mapPointsH - 1)
                            val ny = (y + dy).coerceIn(0, grid.mapPointsV - 1)
                            val weight = if (dx == 0 && dy == 0) 4f else if (dx == 0 || dy == 0) 2f else 1f
                            val value = current[ny * grid.mapPointsH + nx]
                            weightSum += weight
                            midOutput += value.midOutput * weight
                            blackGain += value.blackGain * weight
                            toeSlope += value.toeSlope * weight
                            shoulderPower += value.shoulderPower * weight
                            highlightPressure += value.highlightPressure * weight
                        }
                    }
                    val index = y * grid.mapPointsH + x
                    next[index] = HdrPgtmCurveParams(
                        midOutput = midOutput / weightSum,
                        blackGain = blackGain / weightSum,
                        toeSlope = toeSlope / weightSum,
                        shoulderPower = shoulderPower / weightSum,
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
                output[outputOffset + index] = params.blackGain
                continue
            }
            val sceneLinear = input / safeInputScale
            val outputLinear = sampleSceneCurve(anchors, sceneLinear)
            output[outputOffset + index] = (outputLinear / max(sceneLinear, MIN_CURVE_INPUT))
                .coerceIn(0.05f, MAX_GAIN_VALUE)
        }
    }

    private fun buildSceneAnchors(params: HdrPgtmCurveParams, sceneMax: Float): List<Pair<Float, Float>> {
        val anchors = ArrayList<Pair<Float, Float>>(10)
        fun addAnchor(scene: Float, output: Float) {
            if (!scene.isFinite() || scene <= 0f || scene >= sceneMax) return
            val previous = anchors.lastOrNull()?.second ?: 0f
            anchors += scene to output.coerceAtLeast(previous + 1e-5f)
        }
        anchors += 0f to 0f
        addAnchor(SCENE_SHADOW_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_SHADOW_ANCHOR))
        addAnchor(SCENE_LOW_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_LOW_ANCHOR))
        addAnchor(SCENE_MID_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_MID_ANCHOR))
        addAnchor(SCENE_HALF_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_HALF_ANCHOR))
        addAnchor(SCENE_WHITE_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_WHITE_ANCHOR))
        addAnchor(SCENE_BRIGHT_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_BRIGHT_ANCHOR))
        addAnchor(SCENE_SUPER_WHITE_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_SUPER_WHITE_ANCHOR))
        addAnchor(SCENE_EXTREME_WHITE_ANCHOR, sampleHdrRangeCompression(params, sceneMax, SCENE_EXTREME_WHITE_ANCHOR))
        anchors += sceneMax to max(1f, (anchors.lastOrNull()?.second ?: 0f) + 1e-5f)
        return anchors
    }

    private fun sampleHdrRangeCompression(
        params: HdrPgtmCurveParams,
        sceneMax: Float,
        sceneLinear: Float,
    ): Float {
        if (sceneLinear <= 0f) return 0f
        if (sceneLinear >= sceneMax) return 1f
        val safeScene = sceneLinear.coerceAtLeast(MIN_CURVE_INPUT)
        val midOutput = params.midOutput
            .coerceIn(SCENE_MID_ANCHOR * MIN_MID_GAIN, SCENE_MID_ANCHOR * MAX_MID_GAIN)
        if (safeScene <= SCENE_MID_ANCHOR) {
            val directGainOutput = safeScene * params.blackGain.coerceIn(MIN_BLACK_GAIN, MAX_BLACK_GAIN)
            val sceneEv = log2(safeScene / SCENE_MID_ANCHOR)
            val midOutputEv = log2(midOutput / SCENE_MID_ANCHOR)
            val toeEvOutput = SCENE_MID_ANCHOR *
                2.0f.pow(midOutputEv + sceneEv * params.toeSlope.coerceIn(MIN_TOE_SLOPE, MAX_TOE_SLOPE))
            val t = smoothStep(SCENE_SHADOW_ANCHOR, SCENE_MID_ANCHOR, safeScene)
            return lerp(directGainOutput, toeEvOutput, t)
                .coerceIn(0f, midOutput)
        }
        val sourceUpperEv = log2(sceneMax / SCENE_MID_ANCHOR).coerceAtLeast(0.01f)
        val sourceEv = log2(safeScene / SCENE_MID_ANCHOR).coerceIn(0f, sourceUpperEv)
        val midOutputEv = log2(midOutput / SCENE_MID_ANCHOR)
        val displayUpperEv = log2(1f / SCENE_MID_ANCHOR)
        val compressedRangeEv = max(displayUpperEv - midOutputEv, 0.01f)
        val t = (sourceEv / sourceUpperEv).coerceIn(0f, 1f)
        val shoulderT = hdrRangeShoulder(t, params.shoulderPower, params.highlightPressure)
        return (SCENE_MID_ANCHOR * 2.0f.pow(midOutputEv + compressedRangeEv * shoulderT))
            .coerceIn(midOutput, 1f)
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

    private fun hdrRangeMidGain(baselineExposureEv: Float, sceneHdrStrength: Float): Float {
        val hdrGain = lerp(2.30f, 2.40f, smoothStep(0.8f, 2.2f, baselineExposureEv))
        return lerp(1.65f, hdrGain, sceneHdrStrength)
            .coerceIn(MIN_MID_GAIN, MAX_MID_GAIN)
    }

    private fun hdrRangeCenterOffsetEv(
        cell: HdrPgtmCellStats,
        darkRegion: Float,
        brightRegion: Float,
        highlightPressure: Float,
        hdrStrength: Float,
    ): Float {
        val lowRangePressure = 1f - smoothStep(0.020f, 0.115f, cell.p90)
        val highRangePressure = smoothStep(0.18f, 0.88f, cell.p90)
        return (
            0.32f * lowRangePressure +
                0.20f * darkRegion -
                0.16f * brightRegion -
                0.20f * highlightPressure -
                0.08f * highRangePressure * hdrStrength
            ).coerceIn(-0.24f, 0.42f)
    }

    private fun hdrRangeBlackGain(
        midGain: Float,
        sceneHdrStrength: Float,
        rangeCenterEv: Float,
        highlightPressure: Float,
    ): Float {
        val rangeToeLift = lerp(0.18f, 0.78f, sceneHdrStrength)
        return (midGain * (1f + rangeToeLift) *
            2.0f.pow(0.48f * rangeCenterEv - 0.18f * highlightPressure))
            .coerceIn(MIN_BLACK_GAIN, MAX_BLACK_GAIN)
    }

    private fun hdrRangeToeSlope(sceneHdrStrength: Float, highlightPressure: Float): Float {
        return (0.74f + 0.08f * sceneHdrStrength - 0.05f * highlightPressure)
            .coerceIn(MIN_TOE_SLOPE, MAX_TOE_SLOPE)
    }

    private fun hdrRangeShoulderPower(
        effectiveUpperRangeEv: Float,
        highlightPressure: Float,
        hdrStrength: Float,
    ): Float {
        val gcamRangeEv = GCAM_HDR_UPPER_RANGE_EV - GCAM_HDR_LOWER_RANGE_EV
        val normalizedRange = (effectiveUpperRangeEv / gcamRangeEv).coerceIn(0f, 1f)
        return (1.02f +
            0.50f * normalizedRange +
            0.08f * hdrStrength -
            0.34f * highlightPressure)
            .coerceIn(MIN_SHOULDER_POWER, MAX_SHOULDER_POWER)
    }

    private fun hdrRangeShoulder(t: Float, shoulderPower: Float, highlightPressure: Float): Float {
        val safeT = t.coerceIn(0f, 1f)
        val safePower = shoulderPower.coerceIn(MIN_SHOULDER_POWER, MAX_SHOULDER_POWER)
        val safePressure = highlightPressure.coerceIn(0f, 1f)
        val base = 1f - (1f - safeT).pow(safePower)
        val kneeLift = 0.035f * safePressure *
            smoothStep(0.68f, 0.82f, safeT) *
            (1f - smoothStep(0.88f, 0.96f, safeT))
        val rolloff = 0.30f * safePressure *
            smoothStep(0.72f, 0.98f, safeT) *
            safeT
        val terminalRolloff = 0.028f *
            smoothStep(0.90f, 0.99f, safeT) *
            safeT
        return (base + kneeLift - rolloff - terminalRolloff).coerceIn(0f, 1f)
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
        val midOutput: Float,
        val blackGain: Float,
        val toeSlope: Float,
        val shoulderPower: Float,
        val highlightPressure: Float,
    )
}
