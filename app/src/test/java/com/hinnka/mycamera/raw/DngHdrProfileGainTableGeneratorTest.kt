package com.hinnka.mycamera.raw

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.pow

class DngHdrProfileGainTableGeneratorTest {
    @Test
    fun googleHdrToneCurveMatchesEmbeddedDngSamples() {
        val curve = DngProfileToneCurve.googleHdrToneCurvePoints()

        assertEquals(514, curve.size)
        assertEquals(0f, curve[0], 0f)
        assertEquals(0f, curve[1], 0f)
        assertEquals(0.104f, googleToneCurve(0.18f), 0.002f)
        assertEquals(0.5f, googleToneCurve(0.5f), 0.0001f)
        assertEquals(0.825f, googleToneCurve(0.75f), 0.002f)
        assertEquals(1f, curve[curve.lastIndex - 1], 0f)
        assertEquals(1f, curve[curve.lastIndex], 0f)
    }

    @Test
    fun baselineToneCurveKeepsMidGrayAndReservesHighlightHeadroom() {
        val map = DngHdrProfileGainTableGenerator.forHdrBaselineExposure(2f)
            ?: error("Expected baseline PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        val midGray = renderedOutputForPostBaselineSignal(map, input = 0.18f)
        val white = renderedOutputForPostBaselineSignal(map, input = 1f)
        val superWhite = renderedOutputForPostBaselineSignal(map, input = 2f)
        val clippedHeadroom = renderedOutputForPostBaselineSignal(map, input = 4f)
        val toneMappedMidGray = googleToneCurve(midGray)
        val toneMappedWhite = googleToneCurve(white.coerceIn(0f, 1f))

        assertTrue("midGray=$midGray", midGray in 0.30f..0.36f)
        assertTrue("toneMappedMidGray=$toneMappedMidGray", toneMappedMidGray in 0.25f..0.37f)
        assertTrue("white=$white", white in 0.84f..0.92f)
        assertTrue("toneMappedWhite=$toneMappedWhite", toneMappedWhite in 0.90f..0.97f)
        assertTrue("superWhite=$superWhite", superWhite in 0.97f..1.01f)
        assertEquals(1.12f, clippedHeadroom, 0.02f)
        assertEquals(sampleGain(map, input = 1f), map.mapInputWeights.sum(), 0.0001f)
        assertEquals(sampleGainAtIndex(map, 256), map.mapInputWeights.sum(), 0.0001f)
        assertTrue(map.mapInputWeights[4] > map.mapInputWeights[1])
    }

    @Test
    fun highDynamicRangeBaselineUsesConservativeHdrTargets() {
        val map = DngHdrProfileGainTableGenerator.forHdrBaselineExposure(HIGH_HDR_BASELINE_EV)
            ?: error("Expected HDR baseline PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        val midGray = renderedOutputForPostBaselineSignal(map, input = 0.18f)
        val white = renderedOutputForPostBaselineSignal(map, input = 1f)
        val superWhite = renderedOutputForPostBaselineSignal(map, input = 2f)
        val farHighlight = renderedOutputForPostBaselineSignal(map, input = 4f)
        val sceneMax = 1f / map.mapInputWeights.sum()
        val toneMappedMidGray = googleToneCurve(midGray)
        val toneMappedWhite = googleToneCurve(white.coerceIn(0f, 1f))

        assertTrue("midGray=$midGray", midGray in 0.34f..0.42f)
        assertTrue("toneMappedMidGray=$toneMappedMidGray", toneMappedMidGray in 0.28f..0.38f)
        assertTrue("white=$white", white in 0.84f..0.92f)
        assertTrue("toneMappedWhite=$toneMappedWhite", toneMappedWhite in 0.90f..0.97f)
        assertTrue("superWhite=$superWhite", superWhite in 0.97f..1.01f)
        assertTrue("farHighlight=$farHighlight superWhite=$superWhite", farHighlight in superWhite..1.02f)
        assertEquals(1f, renderedOutputForPostBaselineSignal(map, input = sceneMax), 0.02f)
        assertTrue(sampleGainAtIndex(map, 0) in 2.45f..2.65f)
        assertTrue(sampleGainAtIndex(map, 256) < 0.12f)
        assertEquals(sampleGainAtIndex(map, 256), map.mapInputWeights.sum(), 0.0001f)
        assertEquals(1.12f, map.mapInputWeights.sum() * 2.0f.pow(HIGH_HDR_BASELINE_EV), 0.02f)
    }

    @Test
    fun baselineToneCurveCanBeRegeneratedWithExistingMapTopology() {
        val template = DngProfileGainTableMap(
            mapPointsV = 3,
            mapPointsH = 2,
            mapSpacingV = 0.5,
            mapSpacingH = 1.0,
            mapOriginV = 0.125,
            mapOriginH = 0.25,
            mapPointsN = 257,
            mapInputWeights = floatArrayOf(0.1f, 0.2f, 0.3f, 0f, 0.4f),
            gamma = 1f,
            gains = FloatArray(3 * 2 * 257) { 1f },
            sourceTag = DngProfileGainTableMap.TAG_PROFILE_GAIN_TABLE_MAP2
        )

        val map = DngHdrProfileGainTableGenerator.forHdrBaselineExposureLike(2f, template)
            ?: error("Expected topology-matched PGTM")

        assertEquals(template.mapPointsH, map.mapPointsH)
        assertEquals(template.mapPointsV, map.mapPointsV)
        assertEquals(template.mapSpacingH, map.mapSpacingH, 0.0)
        assertEquals(template.mapSpacingV, map.mapSpacingV, 0.0)
        assertEquals(template.mapOriginH, map.mapOriginH, 0.0)
        assertEquals(template.mapOriginV, map.mapOriginV, 0.0)
        assertEquals(template.mapPointsH * template.mapPointsV * template.mapPointsN, map.gains.size)
        assertToneOutputMonotonic(map)
    }

    @Test
    fun localToneMappingDodgesDarkTilesAndBurnsBrightTiles() {
        val dark = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = HIGH_HDR_BASELINE_EV,
            packedCellStats = packedStatsFor(
                width = 1024,
                height = 768,
                p10 = 0.0035f,
                p50 = 0.0054f,
                p90 = 0.0084f,
                p98 = 0.0114f,
                highlightFraction = 0f
            )
        ) ?: error("Expected dark PGTM")
        val mid = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = HIGH_HDR_BASELINE_EV,
            packedCellStats = packedStatsFor(
                width = 1024,
                height = 768,
                p10 = 0.0158f,
                p50 = 0.0276f,
                p90 = 0.0438f,
                p98 = 0.0610f,
                highlightFraction = 0f
            )
        ) ?: error("Expected mid PGTM")
        val bright = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = HIGH_HDR_BASELINE_EV,
            packedCellStats = packedStatsFor(
                width = 1024,
                height = 768,
                p10 = 0.6089f,
                p50 = 0.7216f,
                p90 = 0.8911f,
                p98 = 0.9102f,
                highlightFraction = 0.18f
            )
        ) ?: error("Expected bright PGTM")

        assertToneOutputMonotonic(dark)
        assertToneOutputMonotonic(mid)
        assertToneOutputMonotonic(bright)
        assertTrue(sampleGainAtIndex(dark, 0) > sampleGainAtIndex(mid, 0))
        assertTrue(sampleGainAtIndex(mid, 0) > sampleGainAtIndex(bright, 0))
        assertTrue(
            renderedOutputForPostBaselineSignal(dark, input = 0.18f) >
                renderedOutputForPostBaselineSignal(mid, input = 0.18f)
        )
        assertTrue(
            renderedOutputForPostBaselineSignal(mid, input = 0.18f) >
                renderedOutputForPostBaselineSignal(bright, input = 0.18f)
        )
        assertTrue(
            renderedOutputForPostBaselineSignal(bright, input = 1f) <
                renderedOutputForPostBaselineSignal(mid, input = 1f)
        )
        assertTrue(
            renderedOutputForPostBaselineSignal(dark, input = 0.8f) >
                renderedOutputForPostBaselineSignal(bright, input = 0.8f)
        )
        assertEquals(sampleGainAtIndex(dark, 256), sampleGainAtIndex(bright, 256), 0.002f)
    }

    @Test
    fun highlightStatsStartShoulderEarlierThanNeutralStats() {
        val neutral = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = 2f,
            packedCellStats = packedStatsFor(
                width = 1024,
                height = 768,
                p10 = 0.04f,
                p50 = 0.18f,
                p90 = 0.42f,
                p98 = 0.68f,
                highlightFraction = 0.04f
            )
        ) ?: error("Expected neutral PGTM")
        val highlight = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = 2f,
            packedCellStats = packedStatsFor(
                width = 1024,
                height = 768,
                p10 = 0.06f,
                p50 = 0.24f,
                p90 = 0.88f,
                p98 = 0.98f,
                highlightFraction = 0.42f
            )
        ) ?: error("Expected highlight PGTM")

        assertTrue(neutral.isValid)
        assertTrue(highlight.isValid)
        assertToneOutputMonotonic(neutral)
        assertToneOutputMonotonic(highlight)
        assertTrue(
            renderedOutputForPostBaselineSignal(highlight, input = 0.8f) <
                renderedOutputForPostBaselineSignal(neutral, input = 0.8f)
        )
    }

    private fun packedStatsFor(
        width: Int,
        height: Int,
        p10: Float,
        p50: Float,
        p90: Float,
        p98: Float,
        highlightFraction: Float,
    ): FloatArray {
        val grid = DngHdrProfileGainTableGenerator.gridSizeFor(width, height)
        val cellCount = grid[0] * grid[1]
        return FloatArray(cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE).also { stats ->
            for (cell in 0 until cellCount) {
                val offset = cell * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
                stats[offset] = p10
                stats[offset + 1] = p50
                stats[offset + 2] = p90
                stats[offset + 3] = p98
                stats[offset + 4] = highlightFraction
                stats[offset + 5] = 64f
            }
        }
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

    private fun sampleGain(map: DngProfileGainTableMap, input: Float): Float {
        val index = (input.coerceIn(0f, 1f) * map.mapPointsN).toInt()
            .coerceIn(0, map.mapPointsN - 1)
        return map.gains[index]
    }

    private fun sampleGainAtIndex(map: DngProfileGainTableMap, index: Int): Float {
        return map.gains[index.coerceIn(0, map.mapPointsN - 1)]
    }

    private fun renderedOutputForPostBaselineSignal(map: DngProfileGainTableMap, input: Float): Float {
        val tableInput = (input.coerceAtLeast(0f) * map.mapInputWeights.sum()).coerceIn(0f, 1f)
        return input * sampleGain(map, tableInput)
    }

    private fun googleToneCurve(input: Float): Float {
        return curveSample(DngProfileToneCurve.googleHdrToneCurvePoints(), input)
    }

    private fun curveSample(points: FloatArray, input: Float): Float {
        val x = input.coerceIn(0f, 1f)
        var segment = 0
        while (segment < points.size / 2 - 2 && x > points[(segment + 1) * 2]) {
            segment++
        }
        val x0 = points[segment * 2]
        val y0 = points[segment * 2 + 1]
        val x1 = points[(segment + 1) * 2]
        val y1 = points[(segment + 1) * 2 + 1]
        val t = if (x1 == x0) 0f else ((x - x0) / (x1 - x0)).coerceIn(0f, 1f)
        return y0 + (y1 - y0) * t
    }

    private fun tableInputForIndex(index: Int, pointCount: Int): Float {
        return if (index == pointCount - 1) {
            1f
        } else {
            index.toFloat() / pointCount.toFloat()
        }
    }

    private companion object {
        private const val HIGH_HDR_BASELINE_EV = 3.377331f
    }
}
