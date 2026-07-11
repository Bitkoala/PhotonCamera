package com.hinnka.mycamera.processor

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin

class RawGlobalPreAlignerTest {
    @Test
    fun recoversIntegerTranslation() {
        val reference = proxy(pattern(WIDTH, HEIGHT), WIDTH, HEIGHT)
        val current = proxy(shift(reference.lowPass, WIDTH, HEIGHT, 4, -2), WIDTH, HEIGHT)

        val alignment = RawGlobalPreAligner.align(reference, current)

        assertNotNull(alignment)
        alignment!!
        assertEquals(4f, alignment.dxProxyPx, 1f)
        assertEquals(-2f, alignment.dyProxyPx, 1f)
        assertTrue(alignment.score < 0.2f)
        assertTrue(alignment.coverage > 0.75f)
    }

    @Test
    fun geometryCenterFrameIsSelectedAsReference() {
        val base = pattern(WIDTH, HEIGHT)
        val proxies = listOf(
            proxy(shift(base, WIDTH, HEIGHT, -5, 0), WIDTH, HEIGHT).copy(sourcePlaneWidth = WIDTH * 2),
            proxy(base, WIDTH, HEIGHT).copy(sourcePlaneWidth = WIDTH * 2),
            proxy(shift(base, WIDTH, HEIGHT, 5, 0), WIDTH, HEIGHT).copy(sourcePlaneWidth = WIDTH * 2),
        )

        val plan = RawBurstReferencePlanner.planProxies(
            proxies = proxies,
            gyroRisks = floatArrayOf(0.001f, 0.001f, 0.001f),
        )

        assertEquals(1, plan.referenceOriginalIndex)
        assertEquals(1, plan.orderedIndices.first())
        assertEquals(-10f, plan.preAlignmentsToReference[0]!!.translationXPlanePx, 2f)
        assertEquals(10f, plan.preAlignmentsToReference[2]!!.translationXPlanePx, 2f)
    }

    private fun proxy(values: FloatArray, width: Int, height: Int): RawAlignmentProxy {
        val band = FloatArray(values.size)
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val index = y * width + x
                band[index] = values[index] - 0.25f * (
                    values[index - 1] + values[index + 1] +
                        values[index - width] + values[index + width]
                    )
            }
        }
        return RawAlignmentProxy(width, height, values, band, sharpness = 1f)
    }

    private fun pattern(width: Int, height: Int): FloatArray {
        return FloatArray(width * height) { index ->
            val x = index % width
            val y = index / width
            val checker = if ((((x / 7) + (y / 5)) and 1) == 0) 0.7f else -0.5f
            checker + 0.25f * sin(x * 0.19f) + 0.2f * sin(y * 0.13f) +
                if (abs(x - 31) < 3 && abs(y - 57) < 5) 1.2f else 0f
        }
    }

    private fun shift(
        input: FloatArray,
        width: Int,
        height: Int,
        dx: Int,
        dy: Int,
    ): FloatArray {
        val output = FloatArray(input.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val sourceX = x - dx
                val sourceY = y - dy
                if (sourceX in 0 until width && sourceY in 0 until height) {
                    output[y * width + x] = input[sourceY * width + sourceX]
                }
            }
        }
        return output
    }

    private companion object {
        const val WIDTH = 96
        const val HEIGHT = 96
    }
}
