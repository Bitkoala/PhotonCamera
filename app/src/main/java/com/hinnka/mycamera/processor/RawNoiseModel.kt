package com.hinnka.mycamera.processor

class RawNoiseModel private constructor(
    shotNoise: FloatArray,
    readNoise: FloatArray,
    val hasValidCamera2Profile: Boolean,
) {
    val shotNoise: FloatArray = normalizeChannels(shotNoise)
    val readNoise: FloatArray = normalizeChannels(readNoise)

    val averageShotNoise: Float = average(this.shotNoise)
    val averageReadNoise: Float = average(this.readNoise)
    val greenShotNoise: Float = 0.5f * (this.shotNoise[1] + this.shotNoise[2])
    val greenReadNoise: Float = 0.5f * (this.readNoise[1] + this.readNoise[2])

    fun normalizedShotNoiseForShader(): FloatArray {
        return FloatArray(CHANNEL_COUNT) { index ->
            shotNoise[index].coerceAtLeast(0f) / SENSOR_NORMALIZATION_SCALE
        }
    }

    fun normalizedReadNoiseForShader(): FloatArray {
        return FloatArray(CHANNEL_COUNT) { index ->
            readNoise[index].coerceAtLeast(0f) / SENSOR_VARIANCE_NORMALIZATION_SCALE
        }
    }

    companion object {
        private const val CHANNEL_COUNT = 4
        private const val SENSOR_NORMALIZATION_SCALE = 65535.0f
        private const val SENSOR_VARIANCE_NORMALIZATION_SCALE = SENSOR_NORMALIZATION_SCALE * SENSOR_NORMALIZATION_SCALE

        val EMPTY = RawNoiseModel(
            shotNoise = FloatArray(CHANNEL_COUNT),
            readNoise = FloatArray(CHANNEL_COUNT),
            hasValidCamera2Profile = false,
        )

        fun fromLegacyNoiseModel(noiseModel: FloatArray): RawNoiseModel {
            val s = sanitizeCoefficient(noiseModel.getOrElse(0) { 0f })
            val o = sanitizeCoefficient(noiseModel.getOrElse(1) { 0f })
            if (s <= 0f && o <= 0f) return EMPTY
            return RawNoiseModel(
                shotNoise = FloatArray(CHANNEL_COUNT) { s },
                readNoise = FloatArray(CHANNEL_COUNT) { o },
                hasValidCamera2Profile = false,
            )
        }

        fun fromCamera2NoiseProfile(channelPairs: FloatArray): RawNoiseModel {
            if (channelPairs.size < 2) return EMPTY
            if (channelPairs.size == 2) return fromLegacyNoiseModel(channelPairs)

            val pairCount = (channelPairs.size / 2).coerceAtLeast(1)
            val shotNoise = FloatArray(CHANNEL_COUNT)
            val readNoise = FloatArray(CHANNEL_COUNT)
            for (index in 0 until CHANNEL_COUNT) {
                val source = index.coerceAtMost(pairCount - 1)
                shotNoise[index] = sanitizeCoefficient(channelPairs[source * 2])
                readNoise[index] = sanitizeCoefficient(channelPairs[source * 2 + 1])
            }
            return RawNoiseModel(
                shotNoise = shotNoise,
                readNoise = readNoise,
                hasValidCamera2Profile = shotNoise.any { it > 0f } || readNoise.any { it > 0f },
            )
        }

        private fun normalizeChannels(values: FloatArray): FloatArray {
            if (values.isEmpty()) return FloatArray(CHANNEL_COUNT)
            return FloatArray(CHANNEL_COUNT) { index ->
                sanitizeCoefficient(values.getOrElse(index) { values.last() })
            }
        }

        private fun sanitizeCoefficient(value: Float): Float {
            return value.takeIf { it.isFinite() }?.coerceAtLeast(0f) ?: 0f
        }

        private fun average(values: FloatArray): Float {
            if (values.isEmpty()) return 0f
            return values.sum() / values.size.toFloat()
        }
    }
}
