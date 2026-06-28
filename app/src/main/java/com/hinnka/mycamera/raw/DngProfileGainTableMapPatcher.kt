package com.hinnka.mycamera.raw

import com.hinnka.mycamera.utils.PLog
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln

object DngProfileGainTableMapPatcher {
    private const val TAG = "DngProfileGainTableMapPatcher"

    private const val TIFF_CLASSIC_MAGIC = 42
    private const val TIFF_TAG_SUB_IFDS = 330
    private const val TIFF_TAG_BASELINE_EXPOSURE = 50730
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
    private const val HDR_PGTM_INPUT_HEADROOM = 1.12f

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

                val baselineExposureEv = scan.baselineExposure
                    ?: inferBaselineExposureFromInputWeights(existingMap)
                    ?: return@use false
                val replacementMap = DngHdrProfileGainTableGenerator.forHdrBaselineExposureLike(
                    baselineExposureEv = baselineExposureEv,
                    template = existingMap
                ) ?: return@use false
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
        var baselineExposure: Float? = null

        while (pendingIfds.isNotEmpty() && visitedIfds.size < MAX_IFD_VISITS) {
            val ifdOffset = pendingIfds.removeAt(0)
            if (ifdOffset <= 0L || ifdOffset in visitedIfds) continue
            visitedIfds += ifdOffset

            val scan = scanIfd(raf, ifdOffset, byteOrder) ?: continue
            pgtm2Entries += scan.pgtm2Entries
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
            baselineExposure = baselineExposure
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
        var baselineExposure: Float? = null

        for (entryIndex in 0 until entryCount) {
            val entryOffset = entriesStart + entryIndex * 12L
            val entry = ByteArray(12)
            raf.seek(entryOffset)
            raf.readFully(entry)
            val tag = readUnsignedShort(entry, 0, byteOrder)
            val type = readUnsignedShort(entry, 2, byteOrder)
            val count = readUnsignedInt(entry, 4, byteOrder)
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
            subIfdOffsets = subIfdOffsets,
            nextIfdOffset = readUnsignedInt(nextIfdBytes, 0, byteOrder)
        )
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

    private fun inferBaselineExposureFromInputWeights(map: DngProfileGainTableMap): Float? {
        val weightSum = map.mapInputWeights.sum().takeIf { it.isFinite() && it > 0f } ?: return null
        val ev = (ln((HDR_PGTM_INPUT_HEADROOM / weightSum).coerceAtLeast(1e-6f).toDouble()) / ln(2.0)).toFloat()
        return ev.takeIf { it.isFinite() && it > 0f }
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
    )

    private data class IfdScanResult(
        val pgtm2Entries: List<Pgtm2Entry>,
        val baselineExposure: Float?,
        val subIfdOffsets: List<Long>,
        val nextIfdOffset: Long,
    )

    private data class Pgtm2Entry(
        val entryOffset: Long,
        val count: Long,
        val valueOffset: Long,
    )
}
