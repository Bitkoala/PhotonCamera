package com.hinnka.mycamera.processor

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RawNoiseModelTest {
    @Test
    fun legacyNoiseModelReplicatesScalarCoefficientsAcrossBayerChannels() {
        val model = RawNoiseModel.fromLegacyNoiseModel(floatArrayOf(1024f, 256f))

        assertFalse(model.hasValidCamera2Profile)
        assertArrayEquals(floatArrayOf(1024f, 1024f, 1024f, 1024f), model.shotNoise, 0f)
        assertArrayEquals(floatArrayOf(256f, 256f, 256f, 256f), model.readNoise, 0f)
        assertEquals(1024f, model.greenShotNoise, 0f)
        assertEquals(256f, model.greenReadNoise, 0f)
    }

    @Test
    fun camera2NoiseProfilePreservesFourChannelPairs() {
        val model = RawNoiseModel.fromCamera2NoiseProfile(
            floatArrayOf(
                1f, 2f,
                3f, 4f,
                5f, 6f,
                7f, 8f,
            ),
        )

        assertTrue(model.hasValidCamera2Profile)
        assertArrayEquals(floatArrayOf(1f, 3f, 5f, 7f), model.shotNoise, 0f)
        assertArrayEquals(floatArrayOf(2f, 4f, 6f, 8f), model.readNoise, 0f)
        assertEquals(4f, model.averageShotNoise, 0f)
        assertEquals(5f, model.averageReadNoise, 0f)
        assertEquals(4f, model.greenShotNoise, 0f)
        assertEquals(5f, model.greenReadNoise, 0f)
    }

    @Test
    fun noiseModelNormalizesSensorCoefficientsForShaderDomain() {
        val varianceScale = SENSOR_SCALE * SENSOR_SCALE
        val model = RawNoiseModel.fromCamera2NoiseProfile(
            floatArrayOf(
                SENSOR_SCALE, varianceScale,
                SENSOR_SCALE * 0.5f, varianceScale * 0.25f,
                0f, 0f,
                SENSOR_SCALE * 0.1f, varianceScale * 0.1f,
            ),
        )

        assertArrayEquals(
            floatArrayOf(1f, 0.5f, 0f, 0.1f),
            model.normalizedShotNoiseForShader(),
            0.000001f,
        )
        assertArrayEquals(
            floatArrayOf(1f, 0.25f, 0f, 0.1f),
            model.normalizedReadNoiseForShader(),
            0.000001f,
        )
    }

    @Test
    fun invalidNoiseProfilesCollapseToZeroModel() {
        val empty = RawNoiseModel.fromCamera2NoiseProfile(FloatArray(0))
        val negative = RawNoiseModel.fromCamera2NoiseProfile(
            floatArrayOf(
                -1f, -2f,
                Float.NaN, Float.POSITIVE_INFINITY,
            ),
        )

        assertFalse(empty.hasValidCamera2Profile)
        assertArrayEquals(FloatArray(4), empty.shotNoise, 0f)
        assertArrayEquals(FloatArray(4), empty.readNoise, 0f)

        assertFalse(negative.hasValidCamera2Profile)
        assertArrayEquals(FloatArray(4), negative.shotNoise, 0f)
        assertArrayEquals(FloatArray(4), negative.readNoise, 0f)
    }

    private companion object {
        const val SENSOR_SCALE = 65535.0f
    }
}
