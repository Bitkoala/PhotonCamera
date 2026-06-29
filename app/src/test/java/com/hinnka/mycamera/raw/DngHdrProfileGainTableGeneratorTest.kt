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
    fun baselineToneCurveMatchesPixelStyleShoulderTargets() {
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

        assertTrue("midGray=$midGray", midGray in 0.41f..0.46f)
        assertTrue("toneMappedMidGray=$toneMappedMidGray", toneMappedMidGray in 0.38f..0.46f)
        assertTrue("white=$white", white in 0.73f..0.80f)
        assertTrue("toneMappedWhite=$toneMappedWhite", toneMappedWhite in 0.78f..0.88f)
        assertTrue("superWhite=$superWhite", superWhite in 0.88f..0.93f)
        assertEquals(1f, clippedHeadroom, 0.025f)
        assertEquals(0.94f, map.mapInputWeights.sum() * 2.0f.pow(2f), 0.02f)
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

        assertTrue("midGray=$midGray", midGray in 0.41f..0.46f)
        assertTrue("toneMappedMidGray=$toneMappedMidGray", toneMappedMidGray in 0.38f..0.46f)
        assertTrue("white=$white", white in 0.73f..0.80f)
        assertTrue("toneMappedWhite=$toneMappedWhite", toneMappedWhite in 0.78f..0.88f)
        assertTrue("superWhite=$superWhite", superWhite in 0.88f..0.93f)
        assertTrue("farHighlight=$farHighlight superWhite=$superWhite", farHighlight in superWhite..1.02f)
        assertEquals(1f, renderedOutputForPostBaselineSignal(map, input = sceneMax), 0.02f)
        assertTrue(sampleGainAtIndex(map, 0) in 3.55f..3.95f)
        assertTrue(sampleGainAtIndex(map, 256) < 0.10f)
        assertEquals(sampleGainAtIndex(map, 256), map.mapInputWeights.sum(), 0.0001f)
        assertEquals(0.94f, map.mapInputWeights.sum() * 2.0f.pow(HIGH_HDR_BASELINE_EV), 0.02f)
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
    fun cellStatsDeriveSceneAdaptiveInputHeadroomLikePixelSamples() {
        val lowHdr = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 4096,
            height = 3072,
            baselineExposureEv = 1.55f,
            packedCellStats = packedStatsForTailProfile(
                width = 4096,
                height = 3072,
                p10 = 0.065f,
                p50 = 0.208f,
                p90 = 0.663f,
                p98 = 0.93f,
                highlightFraction = 0.010f,
                tailP95 = 0.952f,
                tailP98 = 1.127f,
                tailP99 = 1.465f,
                maxInput = 2.798f
            )
        ) ?: error("Expected low HDR PGTM")
        val typicalDenseHdr = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 2048,
            height = 1536,
            baselineExposureEv = 0.99f,
            packedCellStats = packedStatsForTailProfile(
                width = 2048,
                height = 1536,
                p10 = 0.038f,
                p50 = 0.059f,
                p90 = 0.111f,
                p98 = 1f,
                highlightFraction = 0.226f,
                tailP95 = 1.656f,
                tailP98 = 1.722f,
                tailP99 = 1.745f,
                maxInput = 1.986f
            )
        ) ?: error("Expected typical dense HDR PGTM")
        val smallHighlightHdr = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 4096,
            height = 3072,
            baselineExposureEv = 1.56f,
            packedCellStats = packedStatsForTailProfile(
                width = 4096,
                height = 3072,
                p10 = 0.005f,
                p50 = 0.03f,
                p90 = 0.48f,
                p98 = 1f,
                highlightFraction = 0.049f,
                tailP95 = 2.751f,
                tailP98 = 2.825f,
                tailP99 = 2.949f,
                maxInput = 2.949f
            )
        ) ?: error("Expected small highlight PGTM")
        val sparseOutlierHdr = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 4096,
            height = 3072,
            baselineExposureEv = 1.47f,
            packedCellStats = packedStatsForTailProfile(
                width = 4096,
                height = 3072,
                p10 = 0.015f,
                p50 = 0.12f,
                p90 = 0.55f,
                p98 = 0.77f,
                highlightFraction = 0.006f,
                tailP95 = 0.699f,
                tailP98 = 0.908f,
                tailP99 = 1.808f,
                maxInput = 2.770f
            )
        ) ?: error("Expected sparse outlier PGTM")
        val denseHighlightHdr = DngHdrProfileGainTableGenerator.forHdrCellStats(
            width = 4096,
            height = 3072,
            baselineExposureEv = 1.47f,
            packedCellStats = packedStatsForTailProfile(
                width = 4096,
                height = 3072,
                p10 = 0.12f,
                p50 = 0.45f,
                p90 = 0.95f,
                p98 = 1f,
                highlightFraction = 0.165f,
                tailP95 = 2.563f,
                tailP98 = 2.579f,
                tailP99 = 2.586f,
                maxInput = 2.694f
            )
        ) ?: error("Expected dense highlight PGTM")

        val normalizedWeights = lowHdr.mapInputWeights.map { it / lowHdr.mapInputWeights.sum() }
        assertEquals(0.1495f, normalizedWeights[0], 0.0001f)
        assertEquals(0.2935f, normalizedWeights[1], 0.0001f)
        assertEquals(0.0570f, normalizedWeights[2], 0.0001f)
        assertEquals(0.1250f, normalizedWeights[3], 0.0001f)
        assertEquals(0.3750f, normalizedWeights[4], 0.0001f)
        assertTrue(effectiveInputHeadroom(lowHdr, 1.55f) in 2.65f..3.05f)
        assertTrue(effectiveInputHeadroom(typicalDenseHdr, 0.99f) in 0.95f..1.08f)
        assertTrue(effectiveInputHeadroom(smallHighlightHdr, 1.56f) in 0.82f..1.05f)
        assertTrue(effectiveInputHeadroom(sparseOutlierHdr, 1.47f) in 1.55f..1.95f)
        assertTrue(effectiveInputHeadroom(denseHighlightHdr, 1.47f) in 1.05f..1.30f)
        assertTrue(sampleGainAtIndex(lowHdr, 0) < sampleGainAtIndex(smallHighlightHdr, 0))
        assertTrue(
            "typicalShadow=${sampleGainAtIndex(typicalDenseHdr, 0)} lowShadow=${sampleGainAtIndex(lowHdr, 0)}",
            sampleGainAtIndex(typicalDenseHdr, 0) > sampleGainAtIndex(lowHdr, 0)
        )
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
        assertTrue(
            "darkShadow=${sampleGainAtIndex(dark, 0)} midShadow=${sampleGainAtIndex(mid, 0)}",
            sampleGainAtIndex(dark, 0) > sampleGainAtIndex(mid, 0)
        )
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
        p995Input: Float = p98,
        p999Input: Float = p995Input,
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
                stats[offset + 6] = p995Input
                stats[offset + 7] = p999Input
            }
        }
    }

    private fun packedStatsForTailProfile(
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

    private fun effectiveInputHeadroom(map: DngProfileGainTableMap, baselineExposureEv: Float): Float {
        return map.mapInputWeights.sum() * 2.0f.pow(baselineExposureEv)
    }

    private fun googleToneCurve(input: Float): Float {
        return curveSample(DngProfileToneCurve.googleHdrToneCurvePoints(), input)
    }

    private fun lerp(a: Float, b: Float, t: Float): Float {
        return a + (b - a) * t.coerceIn(0f, 1f)
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
