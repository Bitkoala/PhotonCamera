package com.hinnka.mycamera.raw

import com.hinnka.mycamera.utils.PLog
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

object DngProfileGainTableMapPatcher {
    private const val TAG = "DngProfileGainTableMapPatcher"

    private const val TIFF_CLASSIC_MAGIC = 42
    private const val TIFF_TAG_IMAGE_WIDTH = 256
    private const val TIFF_TAG_IMAGE_LENGTH = 257
    private const val TIFF_TAG_BITS_PER_SAMPLE = 258
    private const val TIFF_TAG_COMPRESSION = 259
    private const val TIFF_TAG_PHOTOMETRIC_INTERPRETATION = 262
    private const val TIFF_TAG_STRIP_OFFSETS = 273
    private const val TIFF_TAG_SAMPLES_PER_PIXEL = 277
    private const val TIFF_TAG_ROWS_PER_STRIP = 278
    private const val TIFF_TAG_STRIP_BYTE_COUNTS = 279
    private const val TIFF_TAG_SUB_IFDS = 330
    private const val TIFF_TAG_CFA_REPEAT_PATTERN_DIM = 33421
    private const val TIFF_TAG_CFA_PATTERN = 33422
    private const val TIFF_TAG_BLACK_LEVEL_REPEAT_DIM = 50713
    private const val TIFF_TAG_BLACK_LEVEL = 50714
    private const val TIFF_TAG_WHITE_LEVEL = 50717
    private const val TIFF_TAG_BASELINE_EXPOSURE = 50730
    private const val PHOTOMETRIC_CFA = 32803
    private const val COMPRESSION_UNCOMPRESSED = 1
    private const val BITS_PER_SAMPLE_16 = 16
    private const val RAW_SAMPLE_GRID = 16
    private const val HIST_BINS = 256
    private const val MAX_RAW_PIXEL_COUNT_FOR_PATCH = 80_000_000L
    private const val TYPE_BYTE = 1
    private const val TYPE_ASCII = 2
    private const val TYPE_SHORT = 3
    private const val TYPE_LONG = 4
    private const val TYPE_RATIONAL = 5
    private const val TYPE_SBYTE = 6
    private const val TYPE_UNDEFINED = 7
    private const val TYPE_SSHORT = 8
    private const val TYPE_SLONG = 9
    private const val TYPE_SRATIONAL = 10
    private const val TYPE_FLOAT = 11
    private const val TYPE_DOUBLE = 12
    private const val MAX_IFD_VISITS = 64

    fun rewriteExistingPgtm2WithCurrentGenerator(file: File): Boolean {
        if (!file.exists() || file.length() < 16L) return false

        val existingMap = DngProfileGainTableMap.readFrom(file)
            ?.takeIf { it.isValid && it.sourceTag == DngProfileGainTableMap.TAG_PROFILE_GAIN_TABLE_MAP2 }
            ?: return false

        return runCatching {
            RandomAccessFile(file, "rw").use { raf ->
                val header = ByteArray(8)
                raf.seek(0)
                raf.readFully(header)

                val byteOrder = when {
                    header[0] == 'I'.code.toByte() && header[1] == 'I'.code.toByte() -> ByteOrder.LITTLE_ENDIAN
                    header[0] == 'M'.code.toByte() && header[1] == 'M'.code.toByte() -> ByteOrder.BIG_ENDIAN
                    else -> return@use false
                }
                if (readUnsignedShort(header, 2, byteOrder) != TIFF_CLASSIC_MAGIC) {
                    return@use false
                }

                val firstIfdOffset = readUnsignedInt(header, 4, byteOrder)
                if (firstIfdOffset <= 0L || firstIfdOffset + 2L > raf.length()) {
                    return@use false
                }

                val scan = scanForPgtm2Entries(raf, firstIfdOffset, byteOrder)
                if (scan.pgtm2Entries.isEmpty()) {
                    return@use false
                }

                val baselineExposureEv = scan.baselineExposure ?: run {
                    PLog.w(TAG, "Skip PGTM2 rewrite for ${file.name}: missing BaselineExposure")
                    return@use false
                }
                val localStats = scan.rawImages.firstNotNullOfOrNull { rawImage ->
                    buildLocalStatsFromRaw(
                        raf = raf,
                        rawImage = rawImage,
                        baselineExposureEv = baselineExposureEv
                    )?.let { stats -> rawImage to stats }
                } ?: run {
                    PLog.w(TAG, "Skip PGTM2 rewrite for ${file.name}: no supported uncompressed CFA RAW IFD")
                    return@use false
                }
                val rawImage = localStats.first
                val replacementMap = DngHdrProfileGainTableGenerator.forCellStats(
                    width = rawImage.width,
                    height = rawImage.height,
                    baselineExposureEv = baselineExposureEv,
                    packedCellStats = localStats.second,
                    tablePointCount = existingMap.mapPointsN
                )?.let { map ->
                    if (map.mapPointsH == existingMap.mapPointsH && map.mapPointsV == existingMap.mapPointsV) {
                        map.copy(
                            mapOriginH = existingMap.mapOriginH,
                            mapOriginV = existingMap.mapOriginV
                        )
                    } else {
                        map
                    }
                } ?: return@use false
                val replacementBytes = replacementMap.encodeProfileGainTableMap2(byteOrder)
                val reusableEntry = scan.pgtm2Entries.firstOrNull { entry ->
                    entry.count >= replacementBytes.size.toLong() &&
                        entry.valueOffset >= 0L &&
                        entry.valueOffset + replacementBytes.size.toLong() <= raf.length()
                }
                val replacementOffset = reusableEntry?.valueOffset
                    ?: appendValue(raf, replacementBytes)
                    ?: return@use false
                raf.seek(replacementOffset)
                raf.write(replacementBytes)

                scan.pgtm2Entries.forEach { entry ->
                    writeUndefinedEntryPointer(
                        raf = raf,
                        entryOffset = entry.entryOffset,
                        count = replacementBytes.size.toLong(),
                        valueOffset = replacementOffset,
                        byteOrder = byteOrder
                    )
                }
                PLog.d(
                    TAG,
                    "Rewrote ${scan.pgtm2Entries.size} DNG PGTM2 entries in ${file.name}: " +
                        "baselineExposureEv=$baselineExposureEv " +
                        "grid=${replacementMap.mapPointsH}x${replacementMap.mapPointsV}x${replacementMap.mapPointsN} " +
                        "raw=${rawImage.width}x${rawImage.height} " +
                        "storage=${if (reusableEntry != null) "reused" else "appended"}"
                )
                true
            }
        }.onFailure {
            PLog.w(TAG, "Failed to rewrite DNG PGTM2 with current generator: ${file.absolutePath}", it)
        }.getOrDefault(false)
    }

    private fun scanForPgtm2Entries(
        raf: RandomAccessFile,
        firstIfdOffset: Long,
        byteOrder: ByteOrder,
    ): Pgtm2ScanResult {
        val pendingIfds = mutableListOf(firstIfdOffset)
        val visitedIfds = mutableSetOf<Long>()
        val pgtm2Entries = mutableListOf<Pgtm2Entry>()
        val rawImages = mutableListOf<RawImageInfo>()
        var baselineExposure: Float? = null

        while (pendingIfds.isNotEmpty() && visitedIfds.size < MAX_IFD_VISITS) {
            val ifdOffset = pendingIfds.removeAt(0)
            if (ifdOffset <= 0L || ifdOffset in visitedIfds) continue
            visitedIfds += ifdOffset

            val scan = scanIfd(raf, ifdOffset, byteOrder) ?: continue
            pgtm2Entries += scan.pgtm2Entries
            scan.rawImage?.let { rawImages += it }
            if (baselineExposure == null) {
                baselineExposure = scan.baselineExposure
            }
            if (scan.nextIfdOffset > 0L) {
                pendingIfds += scan.nextIfdOffset
            }
            scan.subIfdOffsets.forEach { subIfdOffset ->
                if (subIfdOffset > 0L && subIfdOffset !in visitedIfds) {
                    pendingIfds += subIfdOffset
                }
            }
        }

        return Pgtm2ScanResult(
            pgtm2Entries = pgtm2Entries,
            baselineExposure = baselineExposure,
            rawImages = rawImages
        )
    }

    private fun scanIfd(
        raf: RandomAccessFile,
        ifdOffset: Long,
        byteOrder: ByteOrder,
    ): IfdScanResult? {
        if (ifdOffset <= 0L || ifdOffset + 2L > raf.length()) return null

        raf.seek(ifdOffset)
        val entryCountBytes = ByteArray(2)
        raf.readFully(entryCountBytes)
        val entryCount = readUnsignedShort(entryCountBytes, 0, byteOrder)
        val entriesStart = ifdOffset + 2L
        val nextIfdPointerOffset = entriesStart + entryCount * 12L
        if (nextIfdPointerOffset + 4L > raf.length()) return null

        val pgtm2Entries = mutableListOf<Pgtm2Entry>()
        val subIfdOffsets = mutableListOf<Long>()
        val entries = mutableListOf<IfdEntryBytes>()
        var baselineExposure: Float? = null

        for (entryIndex in 0 until entryCount) {
            val entryOffset = entriesStart + entryIndex * 12L
            val entry = ByteArray(12)
            raf.seek(entryOffset)
            raf.readFully(entry)
            val tag = readUnsignedShort(entry, 0, byteOrder)
            val type = readUnsignedShort(entry, 2, byteOrder)
            val count = readUnsignedInt(entry, 4, byteOrder)
            entries += IfdEntryBytes(
                tag = tag,
                type = type,
                count = count,
                entryOffset = entryOffset,
                entry = entry
            )
            when (tag) {
                DngProfileGainTableMap.TAG_PROFILE_GAIN_TABLE_MAP2 -> {
                    if (type == TYPE_UNDEFINED) {
                        val valueOffset = valueOffsetForEntry(entry, entryOffset, count, byteOrder)
                        if (valueOffset >= 0L && valueOffset + count <= raf.length()) {
                            pgtm2Entries += Pgtm2Entry(
                                entryOffset = entryOffset,
                                count = count,
                                valueOffset = valueOffset
                            )
                        }
                    }
                }

                TIFF_TAG_BASELINE_EXPOSURE -> {
                    baselineExposure = baselineExposure
                        ?: readFirstRealValue(raf, entry, entryOffset, type, count, byteOrder)
                }

                TIFF_TAG_SUB_IFDS -> {
                    subIfdOffsets += readUnsignedValues(raf, entry, entryOffset, type, count, byteOrder)
                }
            }
        }

        raf.seek(nextIfdPointerOffset)
        val nextIfdBytes = ByteArray(4)
        raf.readFully(nextIfdBytes)
        return IfdScanResult(
            pgtm2Entries = pgtm2Entries,
            baselineExposure = baselineExposure,
            rawImage = parseRawImageInfo(raf, entries, byteOrder),
            subIfdOffsets = subIfdOffsets,
            nextIfdOffset = readUnsignedInt(nextIfdBytes, 0, byteOrder)
        )
    }

    private fun parseRawImageInfo(
        raf: RandomAccessFile,
        entries: List<IfdEntryBytes>,
        byteOrder: ByteOrder,
    ): RawImageInfo? {
        val byTag = entries.associateBy { it.tag }
        val width = readFirstUnsignedValue(raf, byTag[TIFF_TAG_IMAGE_WIDTH], byteOrder)
            ?.takeIf { it in 1..Int.MAX_VALUE.toLong() }
            ?.toInt()
            ?: return null
        val height = readFirstUnsignedValue(raf, byTag[TIFF_TAG_IMAGE_LENGTH], byteOrder)
            ?.takeIf { it in 1..Int.MAX_VALUE.toLong() }
            ?.toInt()
            ?: return null
        if (width.toLong() * height.toLong() > MAX_RAW_PIXEL_COUNT_FOR_PATCH) return null
        val bitsPerSample = readFirstUnsignedValue(raf, byTag[TIFF_TAG_BITS_PER_SAMPLE], byteOrder)
            ?.toInt()
            ?: return null
        val compression = readFirstUnsignedValue(raf, byTag[TIFF_TAG_COMPRESSION], byteOrder)
            ?.toInt()
            ?: COMPRESSION_UNCOMPRESSED
        val photometric = readFirstUnsignedValue(raf, byTag[TIFF_TAG_PHOTOMETRIC_INTERPRETATION], byteOrder)
            ?.toInt()
            ?: return null
        val samplesPerPixel = readFirstUnsignedValue(raf, byTag[TIFF_TAG_SAMPLES_PER_PIXEL], byteOrder)
            ?.toInt()
            ?: 1
        if (bitsPerSample != BITS_PER_SAMPLE_16 ||
            compression != COMPRESSION_UNCOMPRESSED ||
            photometric != PHOTOMETRIC_CFA ||
            samplesPerPixel != 1
        ) {
            return null
        }

        val stripOffsets = readUnsignedValues(
            raf = raf,
            entry = byTag[TIFF_TAG_STRIP_OFFSETS],
            byteOrder = byteOrder,
            maxCount = 4096
        ).takeIf { it.isNotEmpty() } ?: return null
        val stripByteCounts = readUnsignedValues(
            raf = raf,
            entry = byTag[TIFF_TAG_STRIP_BYTE_COUNTS],
            byteOrder = byteOrder,
            maxCount = stripOffsets.size.coerceAtLeast(1)
        ).takeIf { it.size == stripOffsets.size } ?: return null
        val rowsPerStrip = readFirstUnsignedValue(raf, byTag[TIFF_TAG_ROWS_PER_STRIP], byteOrder)
            ?.takeIf { it > 0L && it <= Int.MAX_VALUE.toLong() }
            ?.toInt()
            ?: height

        val cfaDim = readUnsignedValues(
            raf = raf,
            entry = byTag[TIFF_TAG_CFA_REPEAT_PATTERN_DIM],
            byteOrder = byteOrder,
            maxCount = 2
        )
        val cfaRows = cfaDim.getOrNull(0)?.toInt()?.coerceIn(1, 8) ?: 2
        val cfaCols = cfaDim.getOrNull(1)?.toInt()?.coerceIn(1, 8) ?: 2
        val cfaPattern = readEntryBytes(
            raf = raf,
            entry = byTag[TIFF_TAG_CFA_PATTERN],
            byteOrder = byteOrder,
            maxBytes = 64
        )?.takeIf { it.size >= cfaRows * cfaCols }
            ?.take(cfaRows * cfaCols)
            ?.map { it.toInt() and 0xFF }
            ?.toIntArray()
            ?: intArrayOf(0, 1, 1, 2)

        val blackDim = readUnsignedValues(
            raf = raf,
            entry = byTag[TIFF_TAG_BLACK_LEVEL_REPEAT_DIM],
            byteOrder = byteOrder,
            maxCount = 2
        )
        val blackRows = blackDim.getOrNull(0)?.toInt()?.coerceIn(1, 8) ?: cfaRows
        val blackCols = blackDim.getOrNull(1)?.toInt()?.coerceIn(1, 8) ?: cfaCols
        val blackCount = blackRows * blackCols
        val blackValues = readRealValues(
            raf = raf,
            entry = byTag[TIFF_TAG_BLACK_LEVEL],
            byteOrder = byteOrder,
            maxCount = blackCount.coerceAtLeast(1)
        )
        val blackLevel = FloatArray(blackCount) { index ->
            blackValues.getOrNull(index)
                ?: blackValues.firstOrNull()
                ?: 0f
        }
        val whiteLevel = readRealValues(
            raf = raf,
            entry = byTag[TIFF_TAG_WHITE_LEVEL],
            byteOrder = byteOrder,
            maxCount = 1
        ).firstOrNull() ?: 65535f

        return RawImageInfo(
            width = width,
            height = height,
            rowsPerStrip = rowsPerStrip,
            stripOffsets = stripOffsets,
            stripByteCounts = stripByteCounts,
            cfaRows = cfaRows,
            cfaCols = cfaCols,
            cfaPattern = cfaPattern,
            blackRows = blackRows,
            blackCols = blackCols,
            blackLevel = blackLevel,
            whiteLevel = whiteLevel.coerceAtLeast(1f),
            byteOrder = byteOrder
        )
    }

    private fun buildLocalStatsFromRaw(
        raf: RandomAccessFile,
        rawImage: RawImageInfo,
        baselineExposureEv: Float,
    ): FloatArray? {
        val rawPlane = readRawPlane(raf, rawImage) ?: return null
        val grid = DngHdrProfileGainTableGenerator.gridSizeFor(rawImage.width, rawImage.height)
        val gridWidth = grid.getOrElse(0) { 0 }
        val gridHeight = grid.getOrElse(1) { 0 }
        if (gridWidth <= 0 || gridHeight <= 0) return null

        val baselineGain = 2.0f.pow(baselineExposureEv.coerceIn(0f, 8f))
        val stats = FloatArray(gridWidth * gridHeight * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE)
        val hist = IntArray(HIST_BINS)
        val inputSamples = FloatArray(RAW_SAMPLE_GRID * RAW_SAMPLE_GRID)
        for (cellY in 0 until gridHeight) {
            for (cellX in 0 until gridWidth) {
                java.util.Arrays.fill(hist, 0)
                var sampleCount = 0
                var highlightCount = 0
                var startX = (cellX * rawImage.width) / gridWidth
                var endX = ((cellX + 1) * rawImage.width + gridWidth - 1) / gridWidth
                var startY = (cellY * rawImage.height) / gridHeight
                var endY = ((cellY + 1) * rawImage.height + gridHeight - 1) / gridHeight
                startX -= startX and 1
                startY -= startY and 1
                endX = min(rawImage.width, endX + (endX and 1))
                endY = min(rawImage.height, endY + (endY and 1))

                val cellWidth = max(endX - startX, 2)
                val cellHeight = max(endY - startY, 2)
                for (localY in 0 until RAW_SAMPLE_GRID) {
                    for (localX in 0 until RAW_SAMPLE_GRID) {
                        var x = startX + ((localX * 2 + 1) * cellWidth) / (RAW_SAMPLE_GRID * 2)
                        var y = startY + ((localY * 2 + 1) * cellHeight) / (RAW_SAMPLE_GRID * 2)
                        x = (x - (x and 1)).coerceIn(startX, max(startX, endX - 2))
                        y = (y - (y and 1)).coerceIn(startY, max(startY, endY - 2))
                        val inputValue = pgtmInputAt(rawPlane, rawImage, x, y, baselineGain)
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
                val offset = (cellY * gridWidth + cellX) * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
                stats[offset] = percentileFromHist(hist, sampleCount, 0.10f)
                stats[offset + 1] = percentileFromHist(hist, sampleCount, 0.50f)
                stats[offset + 2] = percentileFromHist(hist, sampleCount, 0.90f)
                stats[offset + 3] = percentileFromHist(hist, sampleCount, 0.98f)
                stats[offset + 4] = if (sampleCount > 0) highlightCount.toFloat() / sampleCount.toFloat() else 0f
                stats[offset + 5] = sampleCount.toFloat()
                stats[offset + 6] = percentileFromSamples(inputSamples, sampleCount, 0.995f)
                stats[offset + 7] = percentileFromSamples(inputSamples, sampleCount, 0.999f)
            }
        }
        return stats
    }

    private fun readRawPlane(
        raf: RandomAccessFile,
        rawImage: RawImageInfo,
    ): ShortArray? {
        val pixelCount = rawImage.width.toLong() * rawImage.height.toLong()
        if (pixelCount <= 0L || pixelCount > MAX_RAW_PIXEL_COUNT_FOR_PATCH) return null
        val output = ShortArray(pixelCount.toInt())
        rawImage.stripOffsets.forEachIndexed { stripIndex, stripOffset ->
            val startRow = stripIndex * rawImage.rowsPerStrip
            if (startRow >= rawImage.height) return@forEachIndexed
            val rows = min(rawImage.rowsPerStrip, rawImage.height - startRow)
            val expectedBytes = rows.toLong() * rawImage.width.toLong() * 2L
            val byteCount = rawImage.stripByteCounts.getOrNull(stripIndex) ?: return null
            if (stripOffset <= 0L || byteCount < expectedBytes || stripOffset + expectedBytes > raf.length()) {
                return null
            }
            if (expectedBytes > Int.MAX_VALUE.toLong()) return null
            val bytes = ByteArray(expectedBytes.toInt())
            raf.seek(stripOffset)
            raf.readFully(bytes)
            val rowStrideBytes = rawImage.width * 2
            for (row in 0 until rows) {
                var byteOffset = row * rowStrideBytes
                var outOffset = (startRow + row) * rawImage.width
                repeat(rawImage.width) {
                    val value = if (rawImage.byteOrder == ByteOrder.LITTLE_ENDIAN) {
                        (bytes[byteOffset].toInt() and 0xFF) or
                            ((bytes[byteOffset + 1].toInt() and 0xFF) shl 8)
                    } else {
                        ((bytes[byteOffset].toInt() and 0xFF) shl 8) or
                            (bytes[byteOffset + 1].toInt() and 0xFF)
                    }
                    output[outOffset++] = value.toShort()
                    byteOffset += 2
                }
            }
        }
        return output
    }

    private fun pgtmInputAt(
        rawPlane: ShortArray,
        rawImage: RawImageInfo,
        baseX: Int,
        baseY: Int,
        baselineGain: Float,
    ): Float {
        var r = 0f
        var g = 0f
        var b = 0f
        var rc = 0f
        var gc = 0f
        var bc = 0f
        for (dy in 0..1) {
            for (dx in 0..1) {
                val x = (baseX + dx).coerceIn(0, rawImage.width - 1)
                val y = (baseY + dy).coerceIn(0, rawImage.height - 1)
                val value = normalizedRawAt(rawPlane, rawImage, x, y)
                when (cfaColorAt(rawImage, x, y)) {
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

    private fun normalizedRawAt(
        rawPlane: ShortArray,
        rawImage: RawImageInfo,
        x: Int,
        y: Int,
    ): Float {
        val raw = rawPlane[y * rawImage.width + x].toInt() and 0xFFFF
        val blackIndex = (y % rawImage.blackRows) * rawImage.blackCols + (x % rawImage.blackCols)
        val black = rawImage.blackLevel.getOrElse(blackIndex) { rawImage.blackLevel.firstOrNull() ?: 0f }
        val range = max(rawImage.whiteLevel - black, 1f)
        return ((raw.toFloat() - black) / range).coerceIn(0f, 1f)
    }

    private fun cfaColorAt(rawImage: RawImageInfo, x: Int, y: Int): Int {
        val index = (y % rawImage.cfaRows) * rawImage.cfaCols + (x % rawImage.cfaCols)
        return rawImage.cfaPattern.getOrElse(index) { 1 }
    }

    private fun percentileFromHist(hist: IntArray, sampleCount: Int, percentile: Float): Float {
        if (sampleCount <= 0) return 0f
        val target = kotlin.math.ceil(sampleCount.toFloat() * percentile.coerceIn(0f, 1f)).toInt()
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
        val index = kotlin.math.ceil(sampleCount.toFloat() * percentile.coerceIn(0f, 1f)).toInt()
            .coerceIn(1, sampleCount) - 1
        return samples[index].takeIf { it.isFinite() && it > 0f } ?: 0f
    }

    private fun readFirstUnsignedValue(
        raf: RandomAccessFile,
        entry: IfdEntryBytes?,
        byteOrder: ByteOrder,
    ): Long? {
        return readUnsignedValues(raf, entry, byteOrder, maxCount = 1).firstOrNull()
    }

    private fun readUnsignedValues(
        raf: RandomAccessFile,
        entry: IfdEntryBytes?,
        byteOrder: ByteOrder,
        maxCount: Int,
    ): List<Long> {
        entry ?: return emptyList()
        if (entry.count <= 0L || entry.count > maxCount.toLong()) return emptyList()
        val valueBytes = valueByteCount(entry.type, entry.count) ?: return emptyList()
        val valueOffset = valueOffsetForEntry(entry.entry, entry.entryOffset, valueBytes, byteOrder)
        if (valueOffset < 0L || valueOffset + valueBytes > raf.length()) return emptyList()
        val typeSize = typeSize(entry.type) ?: return emptyList()
        raf.seek(valueOffset)
        return buildList {
            repeat(entry.count.toInt()) {
                when (entry.type) {
                    TYPE_BYTE, TYPE_ASCII, TYPE_UNDEFINED -> add(raf.readUnsignedByte().toLong())
                    TYPE_SHORT -> add(readUnsignedShort(raf.readBytes(2), 0, byteOrder).toLong())
                    TYPE_LONG -> add(readUnsignedInt(raf.readBytes(4), 0, byteOrder))
                    TYPE_SBYTE -> add(raf.readByte().toLong())
                    TYPE_SSHORT -> add(readSignedShort(raf.readBytes(2), 0, byteOrder).toLong())
                    TYPE_SLONG -> add(readSignedInt(raf.readBytes(4), 0, byteOrder).toLong())
                    else -> {
                        raf.skipBytes(typeSize.toInt())
                    }
                }
            }
        }
    }

    private fun readRealValues(
        raf: RandomAccessFile,
        entry: IfdEntryBytes?,
        byteOrder: ByteOrder,
        maxCount: Int,
    ): List<Float> {
        entry ?: return emptyList()
        if (entry.count <= 0L || entry.count > maxCount.toLong()) return emptyList()
        val valueBytes = valueByteCount(entry.type, entry.count) ?: return emptyList()
        val valueOffset = valueOffsetForEntry(entry.entry, entry.entryOffset, valueBytes, byteOrder)
        if (valueOffset < 0L || valueOffset + valueBytes > raf.length()) return emptyList()
        val typeSize = typeSize(entry.type) ?: return emptyList()
        raf.seek(valueOffset)
        return buildList {
            repeat(entry.count.toInt()) {
                val value = when (entry.type) {
                    TYPE_BYTE, TYPE_ASCII, TYPE_UNDEFINED -> raf.readUnsignedByte().toDouble()
                    TYPE_SBYTE -> raf.readByte().toDouble()
                    TYPE_SHORT -> readUnsignedShort(raf.readBytes(2), 0, byteOrder).toDouble()
                    TYPE_SSHORT -> readSignedShort(raf.readBytes(2), 0, byteOrder).toDouble()
                    TYPE_LONG -> readUnsignedInt(raf.readBytes(4), 0, byteOrder).toDouble()
                    TYPE_SLONG -> readSignedInt(raf.readBytes(4), 0, byteOrder).toDouble()
                    TYPE_RATIONAL -> {
                        val bytes = raf.readBytes(8)
                        val numerator = readUnsignedInt(bytes, 0, byteOrder).toDouble()
                        val denominator = readUnsignedInt(bytes, 4, byteOrder).toDouble()
                        if (denominator == 0.0) Double.NaN else numerator / denominator
                    }

                    TYPE_SRATIONAL -> {
                        val bytes = raf.readBytes(8)
                        val numerator = readSignedInt(bytes, 0, byteOrder).toDouble()
                        val denominator = readSignedInt(bytes, 4, byteOrder).toDouble()
                        if (denominator == 0.0) Double.NaN else numerator / denominator
                    }

                    TYPE_FLOAT -> ByteBuffer.wrap(raf.readBytes(4)).order(byteOrder).float.toDouble()
                    TYPE_DOUBLE -> ByteBuffer.wrap(raf.readBytes(8)).order(byteOrder).double
                    else -> {
                        raf.skipBytes(typeSize.toInt())
                        Double.NaN
                    }
                }
                if (value.isFinite()) {
                    add(value.toFloat())
                }
            }
        }
    }

    private fun readEntryBytes(
        raf: RandomAccessFile,
        entry: IfdEntryBytes?,
        byteOrder: ByteOrder,
        maxBytes: Int,
    ): ByteArray? {
        entry ?: return null
        val valueBytes = valueByteCount(entry.type, entry.count) ?: return null
        if (valueBytes <= 0L || valueBytes > maxBytes.toLong()) return null
        val valueOffset = valueOffsetForEntry(entry.entry, entry.entryOffset, valueBytes, byteOrder)
        if (valueOffset < 0L || valueOffset + valueBytes > raf.length()) return null
        raf.seek(valueOffset)
        return raf.readBytes(valueBytes.toInt())
    }

    private fun readUnsignedValues(
        raf: RandomAccessFile,
        entry: ByteArray,
        entryOffset: Long,
        type: Int,
        count: Long,
        byteOrder: ByteOrder,
    ): List<Long> {
        if (count <= 0L || count > 64L) return emptyList()
        val valueBytes = valueByteCount(type, count) ?: return emptyList()
        val valueOffset = valueOffsetForEntry(entry, entryOffset, valueBytes, byteOrder)
        if (valueOffset < 0L || valueOffset + valueBytes > raf.length()) return emptyList()

        return buildList {
            raf.seek(valueOffset)
            repeat(count.toInt()) {
                when (type) {
                    TYPE_SHORT -> add(readUnsignedShort(raf.readBytes(2), 0, byteOrder).toLong())
                    TYPE_LONG -> add(readUnsignedInt(raf.readBytes(4), 0, byteOrder))
                }
            }
        }
    }

    private fun readFirstRealValue(
        raf: RandomAccessFile,
        entry: ByteArray,
        entryOffset: Long,
        type: Int,
        count: Long,
        byteOrder: ByteOrder,
    ): Float? {
        if (count <= 0L) return null
        val typeSize = typeSize(type) ?: return null
        val valueBytes = valueByteCount(type, count) ?: return null
        val valueOffset = valueOffsetForEntry(entry, entryOffset, valueBytes, byteOrder)
        if (valueOffset < 0L || valueOffset + typeSize > raf.length()) return null

        raf.seek(valueOffset)
        val value = when (type) {
            TYPE_BYTE, TYPE_ASCII, TYPE_UNDEFINED -> raf.readUnsignedByte().toDouble()
            TYPE_SBYTE -> raf.readByte().toDouble()
            TYPE_SHORT -> readUnsignedShort(raf.readBytes(2), 0, byteOrder).toDouble()
            TYPE_SSHORT -> readSignedShort(raf.readBytes(2), 0, byteOrder).toDouble()
            TYPE_LONG -> readUnsignedInt(raf.readBytes(4), 0, byteOrder).toDouble()
            TYPE_SLONG -> readSignedInt(raf.readBytes(4), 0, byteOrder).toDouble()
            TYPE_RATIONAL -> {
                val bytes = raf.readBytes(8)
                val numerator = readUnsignedInt(bytes, 0, byteOrder).toDouble()
                val denominator = readUnsignedInt(bytes, 4, byteOrder).toDouble()
                if (denominator == 0.0) return null else numerator / denominator
            }

            TYPE_SRATIONAL -> {
                val bytes = raf.readBytes(8)
                val numerator = readSignedInt(bytes, 0, byteOrder).toDouble()
                val denominator = readSignedInt(bytes, 4, byteOrder).toDouble()
                if (denominator == 0.0) return null else numerator / denominator
            }

            TYPE_FLOAT -> ByteBuffer.wrap(raf.readBytes(4)).order(byteOrder).float.toDouble()
            TYPE_DOUBLE -> ByteBuffer.wrap(raf.readBytes(8)).order(byteOrder).double
            else -> return null
        }.toFloat()
        return value.takeIf { it.isFinite() }
    }

    private fun writeUndefinedEntryPointer(
        raf: RandomAccessFile,
        entryOffset: Long,
        count: Long,
        valueOffset: Long,
        byteOrder: ByteOrder,
    ) {
        raf.seek(entryOffset + 2L)
        raf.write(shortBytes(TYPE_UNDEFINED, byteOrder))
        raf.write(intBytes(count, byteOrder))
        raf.write(intBytes(valueOffset, byteOrder))
    }

    private fun appendValue(raf: RandomAccessFile, value: ByteArray): Long? {
        var offset = raf.length()
        if ((offset and 1L) != 0L) {
            raf.seek(offset)
            raf.write(0)
            offset += 1L
        }
        if (offset + value.size.toLong() > 0xFFFFFFFFL) return null
        raf.seek(offset)
        raf.write(value)
        return offset
    }

    private fun valueOffsetForEntry(
        entry: ByteArray,
        entryOffset: Long,
        valueBytes: Long,
        byteOrder: ByteOrder,
    ): Long {
        return if (valueBytes <= 4L) {
            entryOffset + 8L
        } else {
            readUnsignedInt(entry, 8, byteOrder)
        }
    }

    private fun valueByteCount(type: Int, count: Long): Long? {
        if (count <= 0L || count > Int.MAX_VALUE.toLong()) return null
        val size = typeSize(type) ?: return null
        return size * count
    }

    private fun typeSize(type: Int): Long? {
        return when (type) {
            TYPE_BYTE, TYPE_ASCII, TYPE_SBYTE, TYPE_UNDEFINED -> 1L
            TYPE_SHORT, TYPE_SSHORT -> 2L
            TYPE_LONG, TYPE_SLONG, TYPE_FLOAT -> 4L
            TYPE_RATIONAL, TYPE_SRATIONAL, TYPE_DOUBLE -> 8L
            else -> null
        }
    }

    private fun RandomAccessFile.readBytes(count: Int): ByteArray {
        val bytes = ByteArray(count)
        readFully(bytes)
        return bytes
    }

    private fun readUnsignedShort(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Int {
        return ByteBuffer.wrap(bytes, offset, 2).order(byteOrder).short.toInt() and 0xFFFF
    }

    private fun readSignedShort(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Short {
        return ByteBuffer.wrap(bytes, offset, 2).order(byteOrder).short
    }

    private fun readUnsignedInt(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Long {
        return ByteBuffer.wrap(bytes, offset, 4).order(byteOrder).int.toLong() and 0xFFFFFFFFL
    }

    private fun readSignedInt(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Int {
        return ByteBuffer.wrap(bytes, offset, 4).order(byteOrder).int
    }

    private fun shortBytes(value: Int, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(2).order(byteOrder).putShort((value and 0xFFFF).toShort()).array()

    private fun intBytes(value: Long, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(4).order(byteOrder).putInt((value and 0xFFFFFFFFL).toInt()).array()

    private data class Pgtm2ScanResult(
        val pgtm2Entries: List<Pgtm2Entry>,
        val baselineExposure: Float?,
        val rawImages: List<RawImageInfo>,
    )

    private data class IfdScanResult(
        val pgtm2Entries: List<Pgtm2Entry>,
        val baselineExposure: Float?,
        val rawImage: RawImageInfo?,
        val subIfdOffsets: List<Long>,
        val nextIfdOffset: Long,
    )

    private data class IfdEntryBytes(
        val tag: Int,
        val type: Int,
        val count: Long,
        val entryOffset: Long,
        val entry: ByteArray,
    )

    private data class RawImageInfo(
        val width: Int,
        val height: Int,
        val rowsPerStrip: Int,
        val stripOffsets: List<Long>,
        val stripByteCounts: List<Long>,
        val cfaRows: Int,
        val cfaCols: Int,
        val cfaPattern: IntArray,
        val blackRows: Int,
        val blackCols: Int,
        val blackLevel: FloatArray,
        val whiteLevel: Float,
        val byteOrder: ByteOrder,
    )

    private data class Pgtm2Entry(
        val entryOffset: Long,
        val count: Long,
        val valueOffset: Long,
    )
}
