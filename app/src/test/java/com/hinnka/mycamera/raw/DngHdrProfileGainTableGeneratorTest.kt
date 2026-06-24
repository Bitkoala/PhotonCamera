package com.hinnka.mycamera.raw

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.pow

class DngHdrProfileGainTableGeneratorTest {
    @Test
    fun baselineToneCurvePreservesLowerRangeAndCompressesHighlights() {
        val map = DngHdrProfileGainTableGenerator.forHdrBaselineExposure(2f)
            ?: error("Expected baseline PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        assertEquals(0.18f, renderedOutputForPostBaselineSignal(map, input = 0.18f), 0.03f)
        assertTrue(renderedOutputForPostBaselineSignal(map, input = 1f) < 0.55f)
        assertEquals(1.12f, renderedOutputForPostBaselineSignal(map, input = 4f), 0.02f)
        assertEquals(sampleGain(map, input = 1f), map.mapInputWeights.sum(), 0.0001f)
        assertTrue(map.mapInputWeights[4] > map.mapInputWeights[1])
    }

    @Test
    fun highDynamicRangeBaselineTracksIphoneReferenceCurveShape() {
        val map = DngHdrProfileGainTableGenerator.forHdrBaselineExposure(IPHONE_REFERENCE_BASELINE_EV)
            ?: error("Expected iPhone-like baseline PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        assertEquals(1.25f, sampleGainAtIndex(map, 0), 0.04f)
        assertEquals(0.94f, sampleGainAtIndex(map, 16), 0.06f)
        assertEquals(0.32f, sampleGainAtIndex(map, 64), 0.04f)
        assertEquals(0.18f, sampleGainAtIndex(map, 128), 0.04f)
        assertEquals(0.108f, sampleGainAtIndex(map, 256), 0.004f)
        assertEquals(1.12f, map.mapInputWeights.sum() * 2.0f.pow(IPHONE_REFERENCE_BASELINE_EV), 0.02f)
    }

    @Test
    fun iphoneStyleLocalToneDodgeDarkTilesAndBurnBrightTiles() {
        val dark = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = IPHONE_REFERENCE_BASELINE_EV,
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
            baselineExposureEv = IPHONE_REFERENCE_BASELINE_EV,
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
            baselineExposureEv = IPHONE_REFERENCE_BASELINE_EV,
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
        assertTrue(sampleGainAtIndex(dark, 0) > 1.55f)
        assertTrue(sampleGainAtIndex(mid, 0) in 1.18f..1.36f)
        assertTrue(sampleGainAtIndex(bright, 0) < 0.95f)
        assertTrue(sampleGainAtIndex(dark, 16) > sampleGainAtIndex(mid, 16))
        assertTrue(sampleGainAtIndex(mid, 16) > sampleGainAtIndex(bright, 16))
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

    private fun tableInputForIndex(index: Int, pointCount: Int): Float {
        return if (index == pointCount - 1) {
            1f
        } else {
            index.toFloat() / pointCount.toFloat()
        }
    }

    private companion object {
        private const val IPHONE_REFERENCE_BASELINE_EV = 3.377331f
    }
}
