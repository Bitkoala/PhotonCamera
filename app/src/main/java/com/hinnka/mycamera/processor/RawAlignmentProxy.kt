package com.hinnka.mycamera.processor

import android.graphics.ImageFormat
import com.hinnka.mycamera.model.SafeImage
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

data class RawAlignmentProxyConfig(
    val cfaArrangement: Int,
    val blackLevelByPosition: FloatArray,
    val whiteLevel: Float,
    val maxLongEdge: Int = 128,
)

data class RawAlignmentProxy(
    val width: Int,
    val height: Int,
    val lowPass: FloatArray,
    val bandPass: FloatArray,
    val sharpness: Float,
    val sourcePlaneWidth: Int = width,
    val sourcePlaneHeight: Int = height,
)

object RawAlignmentProxyBuilder {
    fun build(image: SafeImage, config: RawAlignmentProxyConfig): RawAlignmentProxy? {
        if (image.format != ImageFormat.RAW_SENSOR || image.width < 4 || image.height < 4) return null
        val plane = image.planes.firstOrNull() ?: return null
        if (plane.pixelStride < 2 || plane.rowStride <= 0) return null
        val sourceCellWidth = image.width / 2
        val sourceCellHeight = image.height / 2
        if (sourceCellWidth < 2 || sourceCellHeight < 2) return null
        val scale = max(
            1.0,
            max(sourceCellWidth, sourceCellHeight).toDouble() / config.maxLongEdge.coerceAtLeast(32),
        )
        val width = max(2, (sourceCellWidth / scale).toInt())
        val height = max(2, (sourceCellHeight / scale).toInt())
        val buffer = plane.buffer.duplicate().order(ByteOrder.nativeOrder())
        val positionBlack = FloatArray(4) { index ->
            config.blackLevelByPosition.getOrElse(index) {
                config.blackLevelByPosition.firstOrNull() ?: 0f
            }
        }
        val greenOffsets = greenOffsets(config.cfaArrangement)
        val linear = FloatArray(width * height)
        for (y in 0 until height) {
            val sourceCellY = min(sourceCellHeight - 1, ((y + 0.5) * sourceCellHeight / height).toInt())
            val rawY = sourceCellY * 2
            for (x in 0 until width) {
                val sourceCellX = min(sourceCellWidth - 1, ((x + 0.5) * sourceCellWidth / width).toInt())
                val rawX = sourceCellX * 2
                var sum = 0f
                for (offset in greenOffsets) {
                    val px = rawX + offset.first
                    val py = rawY + offset.second
                    val positionIndex = offset.second * 2 + offset.first
                    val byteOffset = py * plane.rowStride + px * plane.pixelStride
                    if (byteOffset < 0 || byteOffset + 1 >= buffer.limit()) continue
                    val raw = buffer.getShort(byteOffset).toInt() and 0xffff
                    val black = positionBlack[positionIndex]
                    val range = max(config.whiteLevel - black, 1f)
                    sum += ((raw - black) / range).coerceIn(0f, 1f)
                }
                val value = sum * 0.5f
                linear[y * width + x] = (ln(1.0 + LOG_COMPRESSION * value) / LOG_COMPRESSION_DENOMINATOR).toFloat()
            }
        }

        val lowPass = gaussian5(linear, width, height)
        val wide = gaussian5(gaussian5(lowPass, width, height), width, height)
        val bandPass = FloatArray(linear.size) { lowPass[it] - wide[it] }
        normalizeZeroMeanUnitVariance(lowPass)
        normalizeZeroMeanUnitVariance(bandPass)
        val sharpness = laplacianSharpness(lowPass, width, height)
        return RawAlignmentProxy(
            width = width,
            height = height,
            lowPass = lowPass,
            bandPass = bandPass,
            sharpness = sharpness,
            sourcePlaneWidth = sourceCellWidth,
            sourcePlaneHeight = sourceCellHeight,
        )
    }

    private fun greenOffsets(cfaArrangement: Int): Array<Pair<Int, Int>> {
        return when (cfaArrangement) {
            1, 2 -> arrayOf(0 to 0, 1 to 1) // GRBG / GBRG
            else -> arrayOf(1 to 0, 0 to 1) // RGGB / BGGR and safe fallback
        }
    }

    private fun gaussian5(input: FloatArray, width: Int, height: Int): FloatArray {
        val temporary = FloatArray(input.size)
        val output = FloatArray(input.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                for (k in -2..2) {
                    val sx = (x + k).coerceIn(0, width - 1)
                    sum += input[y * width + sx] * GAUSSIAN_5[k + 2]
                }
                temporary[y * width + x] = sum * GAUSSIAN_5_NORMALIZER
            }
        }
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                for (k in -2..2) {
                    val sy = (y + k).coerceIn(0, height - 1)
                    sum += temporary[sy * width + x] * GAUSSIAN_5[k + 2]
                }
                output[y * width + x] = sum * GAUSSIAN_5_NORMALIZER
            }
        }
        return output
    }

    private fun normalizeZeroMeanUnitVariance(values: FloatArray) {
        if (values.isEmpty()) return
        val mean = values.sum().toDouble() / values.size
        var variance = 0.0
        values.forEach { value ->
            val delta = value - mean
            variance += delta * delta
        }
        val scale = 1.0 / kotlin.math.sqrt(max(variance / values.size, 1.0e-8))
        for (index in values.indices) values[index] = ((values[index] - mean) * scale).toFloat()
    }

    private fun laplacianSharpness(values: FloatArray, width: Int, height: Int): Float {
        if (width < 3 || height < 3) return 0f
        var sum = 0.0
        var count = 0
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val center = values[y * width + x]
                val laplacian = values[y * width + x - 1] + values[y * width + x + 1] +
                    values[(y - 1) * width + x] + values[(y + 1) * width + x] - 4f * center
                sum += abs(laplacian)
                count++
            }
        }
        return if (count > 0) (sum / count).toFloat() else 0f
    }

    private val GAUSSIAN_5 = floatArrayOf(1f, 4f, 6f, 4f, 1f)
    private const val GAUSSIAN_5_NORMALIZER = 1f / 16f
    private const val LOG_COMPRESSION = 8.0
    private val LOG_COMPRESSION_DENOMINATOR = ln(1.0 + LOG_COMPRESSION)
}
