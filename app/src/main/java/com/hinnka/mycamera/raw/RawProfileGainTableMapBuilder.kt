package com.hinnka.mycamera.raw

import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

internal object RawProfileGainTableMapBuilder {
    private const val TAG = "RawProfileGainTableMapBuilder"
    private const val SAMPLE_GRID = 16
    private const val HIST_BINS = 256

    fun build(
        rawData: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        metadata: RawMetadata,
        samplesPerPixel: Int = 1,
    ): DngProfileGainTableMap? {
        val stats = buildPackedCellStats(
            rawData = rawData,
            width = width,
            height = height,
            rowStride = rowStride,
            metadata = metadata,
            samplesPerPixel = samplesPerPixel
        ) ?: return null
        return DngHdrProfileGainTableGenerator.forCellStats(
            width = width,
            height = height,
            baselineExposureEv = DngBaselineExposure.sanitize(metadata.baselineExposure),
            packedCellStats = stats
        )
    }

    private fun buildPackedCellStats(
        rawData: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        metadata: RawMetadata,
        samplesPerPixel: Int,
    ): FloatArray? {
        if (width <= 0 || height <= 0 || metadata.whiteLevel <= 0f) return null
        val sampleCountPerPixel = samplesPerPixel.coerceAtLeast(1)
        val rowBytes = width * sampleCountPerPixel * 2
        val safeRowStride = rowStride.takeIf { it >= rowBytes } ?: rowBytes
        val source = rawData.duplicate().order(ByteOrder.nativeOrder())
        val requiredBytes = (height - 1).toLong() * safeRowStride.toLong() + rowBytes.toLong()
        if (requiredBytes > source.capacity().toLong()) {
            PLog.w(
                TAG,
                "RAW buffer too small for PGTM stats: required=$requiredBytes capacity=${source.capacity()} " +
                    "size=${width}x$height rowStride=$safeRowStride"
            )
            return null
        }

        val grid = DngHdrProfileGainTableGenerator.gridSizeFor(width, height)
        val gridWidth = grid.getOrElse(0) { 0 }
        val gridHeight = grid.getOrElse(1) { 0 }
        if (gridWidth <= 0 || gridHeight <= 0) return null

        val baselineGain = DngBaselineExposure.exactGain(metadata.baselineExposure)
        val stats = FloatArray(gridWidth * gridHeight * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE)
        val hist = IntArray(HIST_BINS)
        val inputSamples = FloatArray(SAMPLE_GRID * SAMPLE_GRID)
        for (cellY in 0 until gridHeight) {
            for (cellX in 0 until gridWidth) {
                java.util.Arrays.fill(hist, 0)
                var sampleCount = 0
                var highlightCount = 0
                var startX = (cellX * width) / gridWidth
                var endX = ((cellX + 1) * width + gridWidth - 1) / gridWidth
                var startY = (cellY * height) / gridHeight
                var endY = ((cellY + 1) * height + gridHeight - 1) / gridHeight
                startX -= startX and 1
                startY -= startY and 1
                endX = min(width, endX + (endX and 1))
                endY = min(height, endY + (endY and 1))

                val cellWidth = max(endX - startX, 2)
                val cellHeight = max(endY - startY, 2)
                for (localY in 0 until SAMPLE_GRID) {
                    for (localX in 0 until SAMPLE_GRID) {
                        var x = startX + ((localX * 2 + 1) * cellWidth) / (SAMPLE_GRID * 2)
                        var y = startY + ((localY * 2 + 1) * cellHeight) / (SAMPLE_GRID * 2)
                        x = (x - (x and 1)).coerceIn(startX, max(startX, endX - 2))
                        y = (y - (y and 1)).coerceIn(startY, max(startY, endY - 2))
                        val inputValue = pgtmInputAt(
                            rawData = source,
                            rowStride = safeRowStride,
                            metadata = metadata,
                            width = width,
                            height = height,
                            baseX = x,
                            baseY = y,
                            baselineGain = baselineGain,
                            samplesPerPixel = sampleCountPerPixel
                        )
                        inputSamples[sampleCount] = inputValue
                        val clampedInput = inputValue.coerceIn(0f, 1f)
                        val bin = (clampedInput * (HIST_BINS - 1).toFloat() + 0.5f)
                            .toInt()
                            .coerceIn(0, HIST_BINS - 1)
                        hist[bin] += 1
                        sampleCount += 1
                        if (inputValue >= 0.92f) {
                            highlightCount += 1
                        }
                    }
                }

                val offset = (cellY * gridWidth + cellX) *
                    DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
                stats[offset] = percentileFromHist(hist, sampleCount, 0.10f)
                stats[offset + 1] = percentileFromHist(hist, sampleCount, 0.50f)
                stats[offset + 2] = percentileFromHist(hist, sampleCount, 0.90f)
                stats[offset + 3] = percentileFromHist(hist, sampleCount, 0.98f)
                stats[offset + 4] = if (sampleCount > 0) {
                    highlightCount.toFloat() / sampleCount.toFloat()
                } else {
                    0f
                }
                stats[offset + 5] = sampleCount.toFloat()
                stats[offset + 6] = percentileFromSamples(inputSamples, sampleCount, 0.995f)
                stats[offset + 7] = percentileFromSamples(inputSamples, sampleCount, 0.999f)
            }
        }
        return stats
    }

    private fun pgtmInputAt(
        rawData: ByteBuffer,
        rowStride: Int,
        metadata: RawMetadata,
        width: Int,
        height: Int,
        baseX: Int,
        baseY: Int,
        baselineGain: Float,
        samplesPerPixel: Int,
    ): Float {
        if (samplesPerPixel >= 3) {
            return linearRgbPgtmInputAt(
                rawData = rawData,
                rowStride = rowStride,
                metadata = metadata,
                width = width,
                height = height,
                x = baseX,
                y = baseY,
                baselineGain = baselineGain,
                samplesPerPixel = samplesPerPixel
            )
        }
        var r = 0f
        var g = 0f
        var b = 0f
        var rc = 0f
        var gc = 0f
        var bc = 0f
        for (dy in 0..1) {
            for (dx in 0..1) {
                val x = (baseX + dx).coerceIn(0, width - 1)
                val y = (baseY + dy).coerceIn(0, height - 1)
                val value = normalizedRawAt(rawData, rowStride, metadata, x, y)
                when (RawCfaCorrection.colorCodeForPixel(metadata.cfaPattern, x, y)) {
                    0 -> {
                        r += value
                        rc += 1f
                    }

                    2 -> {
                        b += value
                        bc += 1f
                    }

                    else -> {
                        g += value
                        gc += 1f
                    }
                }
            }
        }
        val fallback = (r + g + b) / max(rc + gc + bc, 1f)
        val red = if (rc > 0f) r / rc else fallback
        val green = if (gc > 0f) g / gc else fallback
        val blue = if (bc > 0f) b / bc else fallback
        val luma = 0.2126f * red + 0.7152f * green + 0.0722f * blue
        val maxChannel = max(red, max(green, blue))
        return max((0.5f * luma + 0.5f * maxChannel) * baselineGain, 0f)
    }

    private fun linearRgbPgtmInputAt(
        rawData: ByteBuffer,
        rowStride: Int,
        metadata: RawMetadata,
        width: Int,
        height: Int,
        x: Int,
        y: Int,
        baselineGain: Float,
        samplesPerPixel: Int,
    ): Float {
        val clampedX = x.coerceIn(0, width - 1)
        val clampedY = y.coerceIn(0, height - 1)
        val pixelOffset = clampedY * rowStride + clampedX * samplesPerPixel * 2
        val red = normalizedLinearRgbAt(rawData, pixelOffset, metadata, 0)
        val green = normalizedLinearRgbAt(rawData, pixelOffset + 2, metadata, 1)
        val blue = normalizedLinearRgbAt(rawData, pixelOffset + 4, metadata, 2)
        val luma = 0.2126f * red + 0.7152f * green + 0.0722f * blue
        val maxChannel = max(red, max(green, blue))
        return max((0.5f * luma + 0.5f * maxChannel) * baselineGain, 0f)
    }

    private fun normalizedLinearRgbAt(
        rawData: ByteBuffer,
        offset: Int,
        metadata: RawMetadata,
        channelIndex: Int,
    ): Float {
        val raw = rawData.getShort(offset).toInt() and 0xFFFF
        val black = metadata.blackLevel.getOrElse(channelIndex) {
            metadata.blackLevel.firstOrNull() ?: 0f
        }
        val range = max(metadata.whiteLevel - black, 1f)
        return ((raw.toFloat() - black) / range).coerceIn(0f, 1f)
    }

    private fun normalizedRawAt(
        rawData: ByteBuffer,
        rowStride: Int,
        metadata: RawMetadata,
        x: Int,
        y: Int,
    ): Float {
        val offset = y * rowStride + x * 2
        val raw = rawData.getShort(offset).toInt() and 0xFFFF
        val channelIndex = RawCfaCorrection.channelIndexForPixel(metadata.cfaPattern, x, y)
        val black = metadata.blackLevel.getOrElse(channelIndex) {
            metadata.blackLevel.firstOrNull() ?: 0f
        }
        val range = max(metadata.whiteLevel - black, 1f)
        return ((raw.toFloat() - black) / range).coerceIn(0f, 1f)
    }

    private fun percentileFromHist(hist: IntArray, sampleCount: Int, percentile: Float): Float {
        if (sampleCount <= 0) return 0f
        val target = ceil(sampleCount.toFloat() * percentile.coerceIn(0f, 1f)).toInt()
            .coerceAtLeast(1)
        var cumulative = 0
        hist.forEachIndexed { index, count ->
            cumulative += count
            if (cumulative >= target) {
                return index.toFloat() / (hist.size - 1).toFloat()
            }
        }
        return 1f
    }

    private fun percentileFromSamples(samples: FloatArray, sampleCount: Int, percentile: Float): Float {
        if (sampleCount <= 0) return 0f
        java.util.Arrays.sort(samples, 0, sampleCount)
        val index = ceil(sampleCount.toFloat() * percentile.coerceIn(0f, 1f)).toInt()
            .coerceIn(1, sampleCount) - 1
        return samples[index].takeIf { it.isFinite() && it > 0f } ?: 0f
    }
}
