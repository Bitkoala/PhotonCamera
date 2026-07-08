package com.hinnka.mycamera.raw

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.ceil

class DngAppleProRawProfileGainTableGeneratorTest {
    @Test
    fun appleProRawFittedToneCurveKeepsLowAnchors() {
        val curve = DngProfileToneCurve.appleProRawFittedToneCurvePoints()

        assertEquals(514, curve.size)
        assertEquals(0f, curve[0], 0f)
        assertEquals(0f, curve[1], 0f)
        assertEquals(0.000012401f, toneCurveYAtIndex(curve, 1), 1e-8f)
        assertEquals(0.000553325f, toneCurveYAtIndex(curve, 4), 1e-7f)
        assertEquals(0.005831674f, toneCurveYAtIndex(curve, 8), 1e-7f)
        assertEquals(0.069945026f, toneCurveYAtIndex(curve, 32), 1e-6f)
        assertEquals(0.479043320f, toneCurveYAtIndex(curve, 128), 1e-6f)
        assertEquals(1f, curve[curve.lastIndex - 1], 0f)
        assertEquals(1f, curve[curve.lastIndex], 0f)
    }

    @Test
    fun compactAppleProRawStatsKeepShadowLift() {
        val map = DngAppleProRawProfileGainTableGenerator.forCellStats(
            width = 6048,
            height = 8064,
            baselineExposureEv = -0.491214842f,
            packedCellStats = packedStatsForAppleProfile(
                width = 6048,
                height = 8064,
                p10 = 0.05317f,
                p50 = 0.07299f,
                p90 = 0.10034f,
                p98 = 0.11416f,
                highlightFraction = 0f,
                tailP95 = 0.39022f,
                tailP98 = 0.52531f,
                tailP99 = 0.58001f,
                maxInput = 0.70506f
            )
        ) ?: error("Expected Apple ProRAW PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        assertAppleInputWeights(map)
        assertEquals(1f, map.mapInputWeights.sum(), 0.001f)
        assertGainAtIndexIn(map, 0, 5.6f..7.2f)
        assertGainAtIndexIn(map, 8, 5.5f..7.2f)
        assertGainAtIndexIn(map, 16, 4.8f..7.0f)
        assertGainAtIndexIn(map, 32, 3.7f..6.1f)
        assertGainAtIndexIn(map, 64, 1.7f..3.9f)
        assertGainAtIndexIn(map, 128, 1.0f..2.6f)
        assertGainAtIndexIn(map, 256, 0.96f..1.04f)
    }

    @Test
    fun brightAppleProRawStatsCompressLowAndWhitePoints() {
        val map = DngAppleProRawProfileGainTableGenerator.forCellStats(
            width = 6048,
            height = 8064,
            baselineExposureEv = -0.645403624f,
            packedCellStats = packedStatsForAppleProfile(
                width = 6048,
                height = 8064,
                p10 = 0.28629f,
                p50 = 0.31120f,
                p90 = 0.33687f,
                p98 = 0.34596f,
                highlightFraction = 0f,
                tailP95 = 0.43744f,
                tailP98 = 0.46330f,
                tailP99 = 0.48841f,
                maxInput = 0.60448f
            )
        ) ?: error("Expected Apple ProRAW PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        assertAppleInputWeights(map)
        assertTrue("weightSum=${map.mapInputWeights.sum()}", map.mapInputWeights.sum() in 0.070f..0.090f)
        assertGainAtIndexIn(map, 0, 1.2f..1.9f)
        assertGainAtIndexIn(map, 8, 1.1f..1.9f)
        assertGainAtIndexIn(map, 16, 0.8f..1.6f)
        assertGainAtIndexIn(map, 32, 0.35f..0.85f)
        assertGainAtIndexIn(map, 64, 0.18f..0.45f)
        assertGainAtIndexIn(map, 256, 0.070f..0.090f)
    }

    @Test
    fun appleProRawFlatGradientStatsAvoidVisibleTileSteps() {
        val map = DngAppleProRawProfileGainTableGenerator.forCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = 0.2f,
            packedCellStats = packedFlatGradientStats(
                width = 1024,
                height = 768
            )
        ) ?: error("Expected Apple ProRAW PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        assertTrue(
            "lowInputAdjacentDelta=${maxAdjacentGainDelta(map, 0.06f)}",
            maxAdjacentGainDelta(map, 0.06f) < 0.18f
        )
        assertTrue(
            "midInputAdjacentDelta=${maxAdjacentGainDelta(map, 0.18f)}",
            maxAdjacentGainDelta(map, 0.18f) < 0.10f
        )
    }

    @Test
    fun appleProRawExposureNormalizerReducesBrightProfileBias() {
        val map = DngAppleProRawProfileGainTableGenerator.forCellStats(
            width = 6048,
            height = 8064,
            baselineExposureEv = -0.491214842f,
            packedCellStats = packedStatsForAppleProfile(
                width = 6048,
                height = 8064,
                p10 = 0.05317f,
                p50 = 0.07299f,
                p90 = 0.10034f,
                p98 = 0.11416f,
                highlightFraction = 0f,
                tailP95 = 0.39022f,
                tailP98 = 0.52531f,
                tailP99 = 0.58001f,
                maxInput = 0.70506f
            )
        ) ?: error("Expected Apple ProRAW PGTM")
        val curve = DngProfileToneCurve.appleProRawFittedToneCurveLut()
        val sourceEv = DngAppleProRawExposureNormalizer.estimateAverageExposureEv(
            profileGainTableMap = map,
            baselineExposureEv = -0.491214842f,
            toneCurveLut = curve
        ) ?: error("Expected source exposure EV")
        val normalization = DngAppleProRawExposureNormalizer.resolveExposureOffset(
            profileGainTableMap = map,
            baselineExposureEv = -0.491214842f,
            toneCurveLut = curve
        )
        val normalizedEv = DngAppleProRawExposureNormalizer.estimateAverageExposureEv(
            profileGainTableMap = map,
            baselineExposureEv = -0.491214842f,
            toneCurveLut = curve,
            profileExposureOffsetEv = normalization.additionalExposureOffsetEv
        ) ?: error("Expected normalized exposure EV")

        assertTrue("sourceEv=$sourceEv", sourceEv > 1.2f)
        assertTrue(
            "offset=${normalization.additionalExposureOffsetEv}",
            normalization.additionalExposureOffsetEv < -0.25f
        )
        assertTrue("normalizedEv=$normalizedEv", normalizedEv in 0.85f..1.05f)
        assertEquals(normalizedEv, normalization.normalizedAverageEv, 0.02f)
    }

    @Test
    fun appleProRawExposureNormalizerDoesNotBrightenDarkProfileBias() {
        val map = DngAppleProRawProfileGainTableGenerator.forCellStats(
            width = 6048,
            height = 8064,
            baselineExposureEv = -0.645403624f,
            packedCellStats = packedStatsForAppleProfile(
                width = 6048,
                height = 8064,
                p10 = 0.28629f,
                p50 = 0.31120f,
                p90 = 0.33687f,
                p98 = 0.34596f,
                highlightFraction = 0f,
                tailP95 = 0.43744f,
                tailP98 = 0.46330f,
                tailP99 = 0.48841f,
                maxInput = 0.60448f
            )
        ) ?: error("Expected Apple ProRAW PGTM")
        val curve = DngProfileToneCurve.appleProRawFittedToneCurveLut()
        val sourceEv = DngAppleProRawExposureNormalizer.estimateAverageExposureEv(
            profileGainTableMap = map,
            baselineExposureEv = -0.645403624f,
            toneCurveLut = curve
        ) ?: error("Expected source exposure EV")
        val normalization = DngAppleProRawExposureNormalizer.resolveExposureOffset(
            profileGainTableMap = map,
            baselineExposureEv = -0.645403624f,
            toneCurveLut = curve
        )

        assertTrue("sourceEv=$sourceEv", sourceEv < normalization.targetAverageEv)
        assertEquals(0f, normalization.additionalExposureOffsetEv, 0f)
        assertEquals(sourceEv, normalization.normalizedAverageEv, 0f)
    }

    private fun packedStatsForAppleProfile(
        width: Int,
        height: Int,
        p10: Float,
        p50: Float,
        p90: Float,
        p98: Float,
        highlightFraction: Float,
        tailP95: Float,
        tailP98: Float,
        tailP99: Float,
        maxInput: Float,
    ): FloatArray {
        val grid = DngHdrProfileGainTableGenerator.gridSizeFor(width, height)
        val cellCount = grid[0] * grid[1]
        return FloatArray(cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE).also { stats ->
            for (cell in 0 until cellCount) {
                val rank = (cell + 1).toFloat() / cellCount.toFloat()
                val tail = when {
                    rank <= 0.95f -> lerp(p98, tailP95, rank / 0.95f)
                    rank <= 0.98f -> lerp(tailP95, tailP98, (rank - 0.95f) / 0.03f)
                    rank <= 0.99f -> lerp(tailP98, tailP99, (rank - 0.98f) / 0.01f)
                    else -> lerp(tailP99, maxInput, (rank - 0.99f) / 0.01f)
                }
                val offset = cell * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
                stats[offset] = p10
                stats[offset + 1] = p50
                stats[offset + 2] = p90
                stats[offset + 3] = p98.coerceIn(0f, 1f)
                stats[offset + 4] = highlightFraction
                stats[offset + 5] = 64f
                stats[offset + 6] = tail
                stats[offset + 7] = tail
            }
        }
    }

    private fun packedFlatGradientStats(width: Int, height: Int): FloatArray {
        val grid = DngHdrProfileGainTableGenerator.gridSizeFor(width, height)
        val gridWidth = grid[0]
        val gridHeight = grid[1]
        val cellCount = gridWidth * gridHeight
        return FloatArray(cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE).also { stats ->
            for (cell in 0 until cellCount) {
                val x = cell % gridWidth
                val t = if (gridWidth <= 1) 0f else x.toFloat() / (gridWidth - 1).toFloat()
                val p50 = lerp(0.055f, 0.315f, t)
                val p10 = (p50 - 0.018f).coerceAtLeast(0.005f)
                val p90 = p50 + 0.018f
                val p98 = p50 + 0.026f
                val tail = lerp(3.2f, 4.6f, t)
                val offset = cell * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
                stats[offset] = p10
                stats[offset + 1] = p50
                stats[offset + 2] = p90
                stats[offset + 3] = p98
                stats[offset + 4] = 0.055f
                stats[offset + 5] = 256f
                stats[offset + 6] = tail
                stats[offset + 7] = tail
            }
        }
    }

    private fun assertAppleInputWeights(map: DngProfileGainTableMap) {
        val normalizedWeights = map.mapInputWeights.map { it / map.mapInputWeights.sum() }
        assertEquals(0.1063f, normalizedWeights[0], 0.0001f)
        assertEquals(0.3576f, normalizedWeights[1], 0.0001f)
        assertEquals(0.0361f, normalizedWeights[2], 0.0001f)
        assertEquals(0.0f, normalizedWeights[3], 0.0001f)
        assertEquals(0.5f, normalizedWeights[4], 0.0001f)
    }

    private fun assertToneOutputMonotonic(map: DngProfileGainTableMap) {
        val pointCount = map.mapPointsN
        val cellCount = map.mapPointsH * map.mapPointsV
        for (cell in 0 until cellCount) {
            var previousOutput = 0f
            val offset = cell * pointCount
            for (index in 0 until pointCount) {
                val input = tableInputForIndex(index, pointCount)
                val output = input * map.gains[offset + index]
                assertTrue(
                    "cell=$cell index=$index output=$output previous=$previousOutput",
                    output + 0.0001f >= previousOutput
                )
                previousOutput = maxOf(previousOutput, output)
            }
        }
    }

    private fun assertGainAtIndexIn(
        map: DngProfileGainTableMap,
        index: Int,
        expectedRange: ClosedFloatingPointRange<Float>,
    ) {
        val actual = medianGainAtIndex(map, index)
        assertTrue("index=$index actual=$actual expected=$expectedRange", actual in expectedRange)
    }

    private fun medianGainAtIndex(map: DngProfileGainTableMap, index: Int): Float {
        val pointIndex = index.coerceIn(0, map.mapPointsN - 1)
        val cellCount = map.mapPointsH * map.mapPointsV
        val values = FloatArray(cellCount) { cell -> map.gains[cell * map.mapPointsN + pointIndex] }
        values.sort()
        val medianIndex = ceil(values.size * 0.5f).toInt().coerceIn(1, values.size) - 1
        return values[medianIndex]
    }

    private fun maxAdjacentGainDelta(map: DngProfileGainTableMap, tableInput: Float): Float {
        var maxDelta = 0f
        for (y in 0 until map.mapPointsV) {
            for (x in 0 until map.mapPointsH) {
                val cell = y * map.mapPointsH + x
                val gain = gainAtTableInput(map, cell, tableInput)
                if (x + 1 < map.mapPointsH) {
                    maxDelta = maxOf(maxDelta, kotlin.math.abs(gain - gainAtTableInput(map, cell + 1, tableInput)))
                }
                if (y + 1 < map.mapPointsV) {
                    maxDelta = maxOf(maxDelta, kotlin.math.abs(gain - gainAtTableInput(map, cell + map.mapPointsH, tableInput)))
                }
            }
        }
        return maxDelta
    }

    private fun gainAtTableInput(
        map: DngProfileGainTableMap,
        cell: Int,
        tableInput: Float
    ): Float {
        val tableIndex = (tableInput.coerceIn(0f, 1f) * map.mapPointsN)
            .coerceIn(0f, (map.mapPointsN - 1).toFloat())
        val index0 = kotlin.math.floor(tableIndex).toInt().coerceIn(0, map.mapPointsN - 1)
        val index1 = (index0 + 1).coerceAtMost(map.mapPointsN - 1)
        val t = tableIndex - index0.toFloat()
        val offset = cell * map.mapPointsN
        return lerp(map.gains[offset + index0], map.gains[offset + index1], t)
    }

    private fun toneCurveYAtIndex(points: FloatArray, pointIndex: Int): Float {
        return points[pointIndex * 2 + 1]
    }

    private fun tableInputForIndex(index: Int, pointCount: Int): Float {
        if (pointCount <= 1) return 0f
        return if (index == pointCount - 1) 1f else index.toFloat() / pointCount.toFloat()
    }

    private fun lerp(a: Float, b: Float, t: Float): Float {
        return a + (b - a) * t.coerceIn(0f, 1f)
    }
}
