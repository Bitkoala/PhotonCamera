package com.hinnka.mycamera.raw

import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.floor

class DngPhotonProfileGainTableGeneratorTest {
    @Test
    fun sparseHdrTailKeepsHeadroomAndOpensDeepShadows() {
        val width = 2800
        val height = 2102
        val grid = DngHdrProfileGainTableGenerator.gridSizeFor(width, height)
        val cellCount = grid[0] * grid[1]
        val stats = FloatArray(cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE)
        repeat(cellCount) { cell ->
            val offset = cell * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
            stats[offset] = 0.045098f
            stats[offset + 1] = 0.113744f
            stats[offset + 2] = 0.243189f
            stats[offset + 3] = 0.277654f
            stats[offset + 4] = 0.023916f
            stats[offset + 5] = 64f
            stats[offset + 6] = 1.290788f
            stats[offset + 7] = 1.469952f
        }

        val map = DngPhotonProfileGainTableGenerator.forCellStats(
            width = width,
            height = height,
            baselineExposureEv = 0.64f,
            packedCellStats = stats,
            emitDiagnostics = false
        ) ?: error("Expected Photon PGTM")

        val inputScale = map.mapInputWeights.sum()
        val deepShadowGain = medianGain(map, tableInput = 0.01f)
        val upperShadowGain = medianGain(map, tableInput = 0.08f)

        assertTrue("inputScale=$inputScale", inputScale in 0.40f..0.50f)
        assertTrue("deepShadowGain=$deepShadowGain", deepShadowGain in 3.45f..4.55f)
        assertTrue("upperShadowGain=$upperShadowGain", upperShadowGain in 2.05f..3.15f)
        assertTrue("shadow curve must taper", deepShadowGain > upperShadowGain * 1.25f)
    }

    private fun medianGain(map: DngProfileGainTableMap, tableInput: Float): Float {
        val values = FloatArray(map.mapPointsH * map.mapPointsV) { cell ->
            val position = tableInput.coerceIn(0f, 1f) * map.mapPointsN
            val i0 = floor(position).toInt().coerceIn(0, map.mapPointsN - 1)
            val i1 = (i0 + 1).coerceAtMost(map.mapPointsN - 1)
            val fraction = position - i0
            val offset = cell * map.mapPointsN
            map.gains[offset + i0] * (1f - fraction) + map.gains[offset + i1] * fraction
        }
        values.sort()
        return values[values.size / 2]
    }
}
