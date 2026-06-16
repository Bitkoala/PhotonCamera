package com.hinnka.mycamera.raw

object RawCfaCorrection {
    const val MODE_DEFAULT = "Default"
    const val MODE_2X2_RGGB = "2x2_RGGB"
    const val MODE_2X2_GRBG = "2x2_GRBG"
    const val MODE_2X2_GBRG = "2x2_GBRG"
    const val MODE_2X2_BGGR = "2x2_BGGR"
    const val MODE_4X4_RGGB = "4x4_RGGB"
    const val MODE_4X4_GRBG = "4x4_GRBG"
    const val MODE_4X4_GBRG = "4x4_GBRG"
    const val MODE_4X4_BGGR = "4x4_BGGR"

    val allModes: List<String> = listOf(
        MODE_DEFAULT,
        MODE_2X2_RGGB,
        MODE_2X2_GRBG,
        MODE_2X2_GBRG,
        MODE_2X2_BGGR,
        MODE_4X4_RGGB,
        MODE_4X4_GRBG,
        MODE_4X4_GBRG,
        MODE_4X4_BGGR
    )

    fun resolveCfaPattern(defaultCfaPattern: Int, mode: String?): Int {
        return when (mode) {
            MODE_2X2_RGGB -> RawMetadata.CFA_RGGB
            MODE_2X2_GRBG -> RawMetadata.CFA_GRBG
            MODE_2X2_GBRG -> RawMetadata.CFA_GBRG
            MODE_2X2_BGGR -> RawMetadata.CFA_BGGR
            MODE_4X4_RGGB -> RawMetadata.CFA_QUAD_RGGB
            MODE_4X4_GRBG -> RawMetadata.CFA_QUAD_GRBG
            MODE_4X4_GBRG -> RawMetadata.CFA_QUAD_GBRG
            MODE_4X4_BGGR -> RawMetadata.CFA_QUAD_BGGR
            else -> defaultCfaPattern
        }
    }

    fun patternFromMode(mode: String?): Int? {
        return when (mode) {
            MODE_2X2_RGGB -> RawMetadata.CFA_RGGB
            MODE_2X2_GRBG -> RawMetadata.CFA_GRBG
            MODE_2X2_GBRG -> RawMetadata.CFA_GBRG
            MODE_2X2_BGGR -> RawMetadata.CFA_BGGR
            MODE_4X4_RGGB -> RawMetadata.CFA_QUAD_RGGB
            MODE_4X4_GRBG -> RawMetadata.CFA_QUAD_GRBG
            MODE_4X4_GBRG -> RawMetadata.CFA_QUAD_GBRG
            MODE_4X4_BGGR -> RawMetadata.CFA_QUAD_BGGR
            else -> null
        }
    }

    fun isOverrideMode(mode: String?): Boolean {
        return patternFromMode(mode) != null
    }

    fun repeatPatternDim(cfaPattern: Int): IntArray {
        return if (RawMetadata.isQuadBayer(cfaPattern)) {
            intArrayOf(4, 4)
        } else {
            intArrayOf(2, 2)
        }
    }

    fun cfaPatternBytes(cfaPattern: Int): ByteArray {
        return if (RawMetadata.isQuadBayer(cfaPattern)) {
            quadCfaPatternBytes(cfaPattern)
        } else {
            bayerCfaPatternBytes(baseBayerPattern(cfaPattern))
        }
    }

    fun channelIndexForPixel(cfaPattern: Int, x: Int, y: Int): Int {
        return if (RawMetadata.isQuadBayer(cfaPattern)) {
            quadChannelIndex(cfaPattern, x and 3, y and 3)
        } else {
            bayerChannelIndex(baseBayerPattern(cfaPattern), x and 1, y and 1)
        }
    }

    fun colorCodeForPixel(cfaPattern: Int, x: Int, y: Int): Int {
        return when (channelIndexForPixel(cfaPattern, x, y)) {
            0 -> 0
            3 -> 2
            else -> 1
        }
    }

    fun baseBayerPattern(cfaPattern: Int): Int {
        return if (RawMetadata.isQuadBayer(cfaPattern)) {
            cfaPattern - RawMetadata.CFA_QUAD_RGGB
        } else {
            cfaPattern
        }
    }

    private fun bayerCfaPatternBytes(cfaPattern: Int): ByteArray {
        return when (cfaPattern) {
            RawMetadata.CFA_GRBG -> byteArrayOf(1, 0, 2, 1)
            RawMetadata.CFA_GBRG -> byteArrayOf(1, 2, 0, 1)
            RawMetadata.CFA_BGGR -> byteArrayOf(2, 1, 1, 0)
            else -> byteArrayOf(0, 1, 1, 2)
        }
    }

    private fun quadCfaPatternBytes(cfaPattern: Int): ByteArray {
        val bytes = ByteArray(16)
        var index = 0
        for (y in 0 until 4) {
            for (x in 0 until 4) {
                bytes[index++] = colorCodeForPixel(cfaPattern, x, y).toByte()
            }
        }
        return bytes
    }

    private fun bayerChannelIndex(cfaPattern: Int, xParity: Int, yParity: Int): Int {
        return when (cfaPattern) {
            RawMetadata.CFA_GRBG -> when {
                yParity == 0 && xParity == 0 -> 1
                yParity == 0 && xParity == 1 -> 0
                yParity == 1 && xParity == 0 -> 3
                else -> 2
            }

            RawMetadata.CFA_GBRG -> when {
                yParity == 0 && xParity == 0 -> 2
                yParity == 0 && xParity == 1 -> 3
                yParity == 1 && xParity == 0 -> 0
                else -> 1
            }

            RawMetadata.CFA_BGGR -> when {
                yParity == 0 && xParity == 0 -> 3
                yParity == 0 && xParity == 1 -> 2
                yParity == 1 && xParity == 0 -> 1
                else -> 0
            }

            else -> when {
                yParity == 0 && xParity == 0 -> 0
                yParity == 0 && xParity == 1 -> 1
                yParity == 1 && xParity == 0 -> 2
                else -> 3
            }
        }
    }

    private fun quadChannelIndex(cfaPattern: Int, xMod4: Int, yMod4: Int): Int {
        return bayerChannelIndex(
            cfaPattern = baseBayerPattern(cfaPattern),
            xParity = xMod4 / 2,
            yParity = yMod4 / 2
        )
    }
}
