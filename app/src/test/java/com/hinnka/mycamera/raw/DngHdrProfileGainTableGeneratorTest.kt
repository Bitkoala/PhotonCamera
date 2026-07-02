package com.hinnka.mycamera.raw

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
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
    fun cellStatsToneCurveMatchesPixelStyleShoulderTargets() {
        val map = DngHdrProfileGainTableGenerator.forCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = 2f,
            packedCellStats = packedStatsForTailProfile(
                width = 1024,
                height = 768,
                p10 = 0.04f,
                p50 = 0.12f,
                p90 = 0.55f,
                p98 = 0.85f,
                highlightFraction = 0.08f,
                tailP95 = 1.40f,
                tailP98 = 1.70f,
                tailP99 = 2.00f,
                maxInput = 2.40f
            )
        ) ?: error("Expected stats PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        val midGray = renderedOutputForPostBaselineSignal(map, input = 0.18f)
        val white = renderedOutputForPostBaselineSignal(map, input = 1f)
        val superWhite = renderedOutputForPostBaselineSignal(map, input = 2f)
        val sceneMax = 1f / map.mapInputWeights.sum()
        val sceneMaxOutput = renderedOutputForPostBaselineSignal(map, input = sceneMax)
        val toneMappedMidGray = googleToneCurve(midGray)
        val toneMappedWhite = googleToneCurve(white.coerceIn(0f, 1f))

        assertTrue("midGray=$midGray", midGray in 0.41f..0.46f)
        assertTrue("toneMappedMidGray=$toneMappedMidGray", toneMappedMidGray in 0.37f..0.46f)
        assertTrue("white=$white", white in 0.73f..0.80f)
        assertTrue("toneMappedWhite=$toneMappedWhite", toneMappedWhite in 0.78f..0.88f)
        assertTrue("superWhite=$superWhite", superWhite in 0.88f..0.93f)
        assertEquals(1f, sceneMaxOutput, 0.025f)
        val effectiveHeadroom = effectiveInputHeadroom(map, 2f)
        assertTrue("effectiveHeadroom=$effectiveHeadroom", effectiveHeadroom in 1.55f..2.05f)
        assertEquals(sampleGain(map, input = 1f), map.mapInputWeights.sum(), 0.0001f)
        assertEquals(sampleGainAtIndex(map, 256), map.mapInputWeights.sum(), 0.0001f)
        assertTrue(map.mapInputWeights[4] > map.mapInputWeights[1])
    }

    @Test
    fun highDynamicRangeCellStatsUseConservativeHdrTargets() {
        val map = DngHdrProfileGainTableGenerator.forCellStats(
            width = 1024,
            height = 768,
            baselineExposureEv = HIGH_HDR_BASELINE_EV,
            packedCellStats = packedStatsForTailProfile(
                width = 1024,
                height = 768,
                p10 = 0.012f,
                p50 = 0.055f,
                p90 = 0.48f,
                p98 = 1.0f,
                highlightFraction = 0.05f,
                tailP95 = 2.60f,
                tailP98 = 2.80f,
                tailP99 = 3.05f,
                maxInput = 3.30f
            )
        ) ?: error("Expected HDR stats PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        val midGray = renderedOutputForPostBaselineSignal(map, input = 0.18f)
        val white = renderedOutputForPostBaselineSignal(map, input = 1f)
        val superWhite = renderedOutputForPostBaselineSignal(map, input = 2f)
        val sceneMax = 1f / map.mapInputWeights.sum()
        val farHighlight = renderedOutputForPostBaselineSignal(map, input = 4f.coerceAtMost(sceneMax))
        val toneMappedMidGray = googleToneCurve(midGray)
        val toneMappedWhite = googleToneCurve(white.coerceIn(0f, 1f))

        assertTrue("midGray=$midGray", midGray in 0.40f..0.46f)
        assertTrue("toneMappedMidGray=$toneMappedMidGray", toneMappedMidGray in 0.36f..0.46f)
        assertTrue("white=$white", white in 0.72f..0.80f)
        assertTrue("toneMappedWhite=$toneMappedWhite", toneMappedWhite in 0.78f..0.88f)
        assertTrue("superWhite=$superWhite", superWhite in 0.88f..0.93f)
        assertTrue("farHighlight=$farHighlight superWhite=$superWhite", farHighlight in superWhite..1.02f)
        assertEquals(1f, renderedOutputForPostBaselineSignal(map, input = sceneMax), 0.02f)
        assertTrue(sampleGainAtIndex(map, 0) in 3.55f..3.95f)
        assertTrue(sampleGainAtIndex(map, 256) < 0.32f)
        assertEquals(sampleGainAtIndex(map, 256), map.mapInputWeights.sum(), 0.0001f)
        val effectiveHeadroom = effectiveInputHeadroom(map, HIGH_HDR_BASELINE_EV)
        assertTrue("effectiveHeadroom=$effectiveHeadroom", effectiveHeadroom in 2.45f..3.10f)
    }

    @Test
    fun dngDerivedSceneStatsMatchEmbeddedPixelPgtmSamples() {
        DNG_DERIVED_FIXTURES.forEach { fixture ->
            val map = DngHdrProfileGainTableGenerator.forCellStats(
                width = fixture.width,
                height = fixture.height,
                baselineExposureEv = fixture.baselineExposureEv,
                packedCellStats = packedStatsForTailProfile(
                    width = fixture.width,
                    height = fixture.height,
                    p10 = fixture.rimmP10,
                    p50 = fixture.rimmP50,
                    p90 = fixture.rimmP90,
                    p98 = fixture.rimmP98,
                    highlightFraction = fixture.rimmHighlightFraction,
                    tailP95 = fixture.rimmTailP95,
                    tailP98 = fixture.rimmTailP98,
                    tailP99 = fixture.rimmTailP99,
                    maxInput = fixture.rimmMaxInput
                )
            ) ?: error("Expected PGTM for ${fixture.sourceName}")

            assertTrue("${fixture.sourceName} invalid map", map.isValid)
            assertToneOutputMonotonic(map)
            assertOfficialPgtmInputWeights(map, fixture.sourceName)
            assertEquals(
                "${fixture.sourceName} embedded weight sum",
                fixture.embeddedWeightSum,
                map.mapInputWeights.sum(),
                fixture.weightTolerance
            )
            assertEquals(
                "${fixture.sourceName} embedded headroom",
                fixture.embeddedInputHeadroom,
                effectiveInputHeadroom(map, fixture.baselineExposureEv),
                fixture.headroomTolerance
            )
            assertTrue(
                "${fixture.sourceName} shadow gain=${sampleGainAtIndex(map, 0)}",
                sampleGainAtIndex(map, 0) in fixture.shadowGainRange
            )
        }
    }

    @Test
    fun googleKeycapDngStatsMatchEmbeddedPgtmLowRangeCurve() {
        val map = DngHdrProfileGainTableGenerator.forCellStats(
            width = 4096,
            height = 3072,
            baselineExposureEv = GOOGLE_KEYCAP_BASELINE_EV,
            packedCellStats = packedStatsForTailProfile(
                width = 4096,
                height = 3072,
                p10 = 0.272827f,
                p50 = 0.344414f,
                p90 = 0.406222f,
                p98 = 0.429191f,
                highlightFraction = 0.038032f,
                tailP95 = 0.998236f,
                tailP98 = 1.243947f,
                tailP99 = 1.443504f,
                maxInput = 2.448398f
            )
        ) ?: error("Expected Google keycap DNG PGTM")

        assertTrue(map.isValid)
        assertToneOutputMonotonic(map)
        assertEquals(GOOGLE_KEYCAP_PGTM_WEIGHT_SUM, map.mapInputWeights.sum(), 0.045f)
        assertGainNear(map, tableInput = 0.005f, expected = 2.95f, tolerance = 0.38f)
        assertGainNear(map, tableInput = 0.010f, expected = 2.95f, tolerance = 0.36f)
        assertGainNear(map, tableInput = 0.015f, expected = 2.95f, tolerance = 0.35f)
        assertGainNear(map, tableInput = 0.02f, expected = 2.95f, tolerance = 0.34f)
        assertGainNear(map, tableInput = 0.03f, expected = 2.86f, tolerance = 0.30f)
        assertGainNear(map, tableInput = 0.04f, expected = 2.73f, tolerance = 0.28f)
        assertGainNear(map, tableInput = 0.05f, expected = 2.61f, tolerance = 0.24f)
        assertGainNear(map, tableInput = 0.06f, expected = 2.50f, tolerance = 0.22f)
        assertGainNear(map, tableInput = 0.07f, expected = 2.41f, tolerance = 0.20f)
        assertGainNear(map, tableInput = 0.08f, expected = 2.32f, tolerance = 0.18f)
        assertGainNear(map, tableInput = 0.09f, expected = 2.24f, tolerance = 0.17f)
        assertGainNear(map, tableInput = 0.10f, expected = 2.16f, tolerance = 0.16f)
        assertGainNear(map, tableInput = 0.11f, expected = 2.09f, tolerance = 0.15f)
        assertGainNear(map, tableInput = 0.12f, expected = 2.03f, tolerance = 0.14f)
        assertGainNear(map, tableInput = 0.13f, expected = 1.97f, tolerance = 0.13f)
        assertGainNear(map, tableInput = 0.14f, expected = 1.92f, tolerance = 0.13f)
        assertGainNear(map, tableInput = 0.16f, expected = 1.82f, tolerance = 0.13f)
        assertGainNear(map, tableInput = 0.18f, expected = 1.73f, tolerance = 0.13f)
        assertGainNear(map, tableInput = 0.28f, expected = 1.41f, tolerance = 0.12f)
        assertGainNear(map, tableInput = 0.50f, expected = 1.03f, tolerance = 0.08f)
        assertGainNear(map, tableInput = 0.75f, expected = 0.80f, tolerance = 0.06f)
        assertGainNear(map, tableInput = 1.00f, expected = 0.666f, tolerance = 0.025f)
        assertGoogleKeycapLowRangeOutputProfile(map)
        assertTrue("darkOutput=${0.03f * sampleGain(map, 0.03f * map.mapInputWeights.sum())}",
            0.03f * sampleGain(map, 0.03f * map.mapInputWeights.sum()) > 0.075f
        )
        assertTrue("midOutput=${0.18f * sampleGain(map, 0.18f * map.mapInputWeights.sum())}",
            0.18f * sampleGain(map, 0.18f * map.mapInputWeights.sum()) in 0.33f..0.41f
        )
    }

    @Test
    fun sceneStatsProduceOrderedToneTargetsWithoutBrightnessReversal() {
        val dark = DngHdrProfileGainTableGenerator.forCellStats(
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
        val mid = DngHdrProfileGainTableGenerator.forCellStats(
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
        val bright = DngHdrProfileGainTableGenerator.forCellStats(
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
        val darkMid = renderedOutputForPostBaselineSignal(dark, input = 0.18f)
        val midMid = renderedOutputForPostBaselineSignal(mid, input = 0.18f)
        val brightMid = renderedOutputForPostBaselineSignal(bright, input = 0.18f)
        val brightWhite = renderedOutputForPostBaselineSignal(bright, input = 1f)
        val midWhite = renderedOutputForPostBaselineSignal(mid, input = 1f)
        val darkUpper = renderedOutputForPostBaselineSignal(dark, input = 0.8f)
        val brightUpper = renderedOutputForPostBaselineSignal(bright, input = 0.8f)
        assertTrue(
            "darkMid=$darkMid midMid=$midMid",
            darkMid + 0.002f >= midMid
        )
        assertTrue(
            "midMid=$midMid brightMid=$brightMid",
            midMid > brightMid
        )
        assertTrue(
            "brightWhite=$brightWhite midWhite=$midWhite",
            brightWhite < midWhite
        )
        assertTrue(
            "darkUpper=$darkUpper brightUpper=$brightUpper",
            darkUpper > brightUpper
        )
        assertEquals(sampleGainAtIndex(dark, 256), sampleGainAtIndex(bright, 256), 0.002f)
    }

    @Test
    fun highlightStatsStartShoulderEarlierThanNeutralStats() {
        val neutral = DngHdrProfileGainTableGenerator.forCellStats(
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
        val highlight = DngHdrProfileGainTableGenerator.forCellStats(
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
        val clamped = input.coerceIn(0f, 1f)
        val scaled = clamped * map.mapPointsN.coerceAtLeast(1)
        val i0 = scaled.toInt().coerceIn(0, map.mapPointsN - 1)
        val i1 = (i0 + 1).coerceIn(0, map.mapPointsN - 1)
        val t = scaled - i0.toFloat()
        return map.gains[i0] * (1f - t) + map.gains[i1] * t
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
        if (pointCount <= 1) return 0f
        return if (index == pointCount - 1) {
            1f
        } else {
            index.toFloat() / pointCount.toFloat()
        }
    }

    private fun assertGainNear(
        map: DngProfileGainTableMap,
        tableInput: Float,
        expected: Float,
        tolerance: Float,
    ) {
        val actual = sampleGain(map, tableInput)
        assertEquals("tableInput=$tableInput actual=$actual expected=$expected", expected, actual, tolerance)
    }

    private fun assertOfficialPgtmInputWeights(map: DngProfileGainTableMap, sourceName: String) {
        val normalizedWeights = map.mapInputWeights.map { it / map.mapInputWeights.sum() }
        assertEquals("$sourceName weight R", 0.1495f, normalizedWeights[0], 0.0001f)
        assertEquals("$sourceName weight G", 0.2935f, normalizedWeights[1], 0.0001f)
        assertEquals("$sourceName weight B", 0.0570f, normalizedWeights[2], 0.0001f)
        assertEquals("$sourceName weight min", 0.1250f, normalizedWeights[3], 0.0001f)
        assertEquals("$sourceName weight max", 0.3750f, normalizedWeights[4], 0.0001f)
    }

    private fun assertGoogleKeycapLowRangeOutputProfile(map: DngProfileGainTableMap) {
        val samples = floatArrayOf(
            0.080f, 0.085f, 0.090f, 0.095f, 0.100f, 0.105f, 0.110f,
            0.115f, 0.120f, 0.125f, 0.130f, 0.135f, 0.140f
        )
        val expectedOutputs = floatArrayOf(
            0.185364f, 0.193387f, 0.201181f, 0.208760f, 0.216135f, 0.223304f, 0.230243f,
            0.237050f, 0.243581f, 0.249959f, 0.256297f, 0.262396f, 0.268389f
        )
        val expectedSlopes = floatArrayOf(
            1.604631f, 1.558854f, 1.515782f, 1.474917f, 1.433899f, 1.387858f,
            1.361360f, 1.306116f, 1.275739f, 1.267608f, 1.219756f, 1.198601f
        )
        val outputs = samples.map { input -> input * sampleGain(map, input) }
        outputs.forEachIndexed { index, output ->
            assertEquals(
                "lowRangeOutput tableInput=${samples[index]} actual=$output expected=${expectedOutputs[index]}",
                expectedOutputs[index],
                output,
                0.007f
            )
        }
        val slopes = outputs.zipWithNext().mapIndexed { index, (a, b) ->
            (b - a) / (samples[index + 1] - samples[index])
        }
        slopes.forEachIndexed { index, slope ->
            assertEquals(
                "lowRangeSlope tableInput=${samples[index]}..${samples[index + 1]} " +
                    "actual=$slope expected=${expectedSlopes[index]}",
                expectedSlopes[index],
                slope,
                0.17f
            )
        }
    }

    private companion object {
        private const val HIGH_HDR_BASELINE_EV = 3.377331f
        private const val GOOGLE_KEYCAP_BASELINE_EV = 1.46f
        private const val GOOGLE_KEYCAP_PGTM_HEADROOM = 1.8273006f
        private const val GOOGLE_KEYCAP_PGTM_WEIGHT_SUM = 0.6642112f

        private val DNG_DERIVED_FIXTURES = listOf(
            DngDerivedPgtmFixture(
                sourceName = "PXL_20260628_175619159.RAW-02.ORIGINAL..dng",
                width = 2048,
                height = 1536,
                baselineExposureEv = 0.99f,
                embeddedWeightSum = 0.5074623f,
                embeddedInputHeadroom = 1.0079140f,
                rimmP10 = 0.337668f,
                rimmP50 = 0.428147f,
                rimmP90 = 0.556421f,
                rimmP98 = 0.622064f,
                rimmHighlightFraction = 0.237918f,
                rimmTailP95 = 1.769715f,
                rimmTailP98 = 1.848588f,
                rimmTailP99 = 1.960254f,
                rimmMaxInput = 3.247468f,
                shadowGainRange = 2.6f..4.2f
            ),
            DngDerivedPgtmFixture(
                sourceName = "PXL_20260630_100508552.RAW-02.ORIGINAL..dng",
                width = 2818,
                height = 2114,
                baselineExposureEv = 1.38f,
                embeddedWeightSum = 0.6550936f,
                embeddedInputHeadroom = 1.7050014f,
                rimmP10 = 0.322686f,
                rimmP50 = 0.387230f,
                rimmP90 = 0.493896f,
                rimmP98 = 0.552372f,
                rimmHighlightFraction = 0.213364f,
                rimmTailP95 = 1.592500f,
                rimmTailP98 = 1.797323f,
                rimmTailP99 = 1.881310f,
                rimmMaxInput = 1.991887f,
                shadowGainRange = 2.5f..3.8f
            ),
            DngDerivedPgtmFixture(
                sourceName = "PXL_20260629_213614190.RAW-02.ORIGINAL..dng",
                width = 4096,
                height = 3072,
                baselineExposureEv = 1.40f,
                embeddedWeightSum = 0.7948943f,
                embeddedInputHeadroom = 2.0977387f,
                rimmP10 = 0.270184f,
                rimmP50 = 0.340886f,
                rimmP90 = 0.418926f,
                rimmP98 = 0.456479f,
                rimmHighlightFraction = 0.012236f,
                rimmTailP95 = 0.920819f,
                rimmTailP98 = 1.325456f,
                rimmTailP99 = 1.543746f,
                rimmMaxInput = 4.766730f,
                shadowGainRange = 2.0f..3.4f
            ),
            DngDerivedPgtmFixture(
                sourceName = "PXL_20260702_144246096.RAW-02.ORIGINAL..dng",
                width = 4096,
                height = 3072,
                baselineExposureEv = GOOGLE_KEYCAP_BASELINE_EV,
                embeddedWeightSum = GOOGLE_KEYCAP_PGTM_WEIGHT_SUM,
                embeddedInputHeadroom = GOOGLE_KEYCAP_PGTM_HEADROOM,
                rimmP10 = 0.272827f,
                rimmP50 = 0.344414f,
                rimmP90 = 0.406222f,
                rimmP98 = 0.429191f,
                rimmHighlightFraction = 0.038032f,
                rimmTailP95 = 0.998236f,
                rimmTailP98 = 1.243947f,
                rimmTailP99 = 1.443504f,
                rimmMaxInput = 2.448398f,
                shadowGainRange = 2.3f..3.3f
            )
        )
    }

    private data class DngDerivedPgtmFixture(
        val sourceName: String,
        val width: Int,
        val height: Int,
        val baselineExposureEv: Float,
        val embeddedWeightSum: Float,
        val embeddedInputHeadroom: Float,
        val rimmP10: Float,
        val rimmP50: Float,
        val rimmP90: Float,
        val rimmP98: Float,
        val rimmHighlightFraction: Float,
        val rimmTailP95: Float,
        val rimmTailP98: Float,
        val rimmTailP99: Float,
        val rimmMaxInput: Float,
        val shadowGainRange: ClosedFloatingPointRange<Float>,
        val weightTolerance: Float = 0.035f,
        val headroomTolerance: Float = 0.10f,
    )
}
