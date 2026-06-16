package com.hinnka.mycamera.utils

import com.hinnka.mycamera.raw.RawCfaCorrection
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

object DngCfaPatternPatcher {
    private const val TAG = "DngCfaPatternPatcher"
    private const val TIFF_TAG_CFA_REPEAT_PATTERN_DIM = 33421
    private const val TIFF_TAG_CFA_PATTERN = 33422
    private const val TYPE_BYTE = 1
    private const val TYPE_SHORT = 3

    fun patchFromMode(file: File, mode: String?): Boolean {
        val cfaPattern = RawCfaCorrection.patternFromMode(mode) ?: return false
        return patchCfaPattern(file, cfaPattern)
    }

    private fun patchCfaPattern(file: File, cfaPattern: Int): Boolean {
        if (!file.exists() || file.length() < 16L) {
            return false
        }

        return runCatching {
            RandomAccessFile(file, "rw").use { raf ->
                val header = ByteArray(8)
                raf.seek(0)
                raf.readFully(header)

                val byteOrder = when {
                    header[0] == 'I'.code.toByte() && header[1] == 'I'.code.toByte() -> ByteOrder.LITTLE_ENDIAN
                    header[0] == 'M'.code.toByte() && header[1] == 'M'.code.toByte() -> ByteOrder.BIG_ENDIAN
                    else -> return false
                }
                val magic = readUnsignedShort(header, 2, byteOrder)
                if (magic != 42) {
                    return false
                }

                val firstIfdOffset = readUnsignedInt(header, 4, byteOrder)
                if (firstIfdOffset <= 0L || firstIfdOffset + 2L > raf.length()) {
                    return false
                }

                raf.seek(firstIfdOffset)
                val entryCountBytes = ByteArray(2)
                raf.readFully(entryCountBytes)
                val entryCount = readUnsignedShort(entryCountBytes, 0, byteOrder)

                var cfaDimEntryOffset = -1L
                var cfaPatternEntryOffset = -1L
                for (entryIndex in 0 until entryCount) {
                    val entryOffset = firstIfdOffset + 2L + entryIndex * 12L
                    if (entryOffset + 12L > raf.length()) {
                        break
                    }

                    val entry = ByteArray(12)
                    raf.seek(entryOffset)
                    raf.readFully(entry)
                    when (readUnsignedShort(entry, 0, byteOrder)) {
                        TIFF_TAG_CFA_REPEAT_PATTERN_DIM -> cfaDimEntryOffset = entryOffset
                        TIFF_TAG_CFA_PATTERN -> cfaPatternEntryOffset = entryOffset
                    }
                }

                if (cfaDimEntryOffset < 0L || cfaPatternEntryOffset < 0L) {
                    PLog.w(TAG, "DNG CFA tags missing in ${file.name}; dim=$cfaDimEntryOffset pattern=$cfaPatternEntryOffset")
                    return false
                }

                val dim = RawCfaCorrection.repeatPatternDim(cfaPattern)
                writeEntry(
                    raf = raf,
                    entryOffset = cfaDimEntryOffset,
                    type = TYPE_SHORT,
                    count = 2,
                    value = shortBytes(dim[0], byteOrder) + shortBytes(dim[1], byteOrder),
                    byteOrder = byteOrder
                )

                val patternBytes = RawCfaCorrection.cfaPatternBytes(cfaPattern)
                writeEntry(
                    raf = raf,
                    entryOffset = cfaPatternEntryOffset,
                    type = TYPE_BYTE,
                    count = patternBytes.size.toLong(),
                    value = patternBytes,
                    byteOrder = byteOrder
                )

                PLog.d(
                    TAG,
                    "Patched DNG CFA pattern cfa=$cfaPattern dim=${dim.joinToString("x")} " +
                        "bytes=${patternBytes.joinToString()} in ${file.name}"
                )
                true
            }
        }.onFailure {
            PLog.w(TAG, "Failed to patch DNG CFA pattern: ${file.absolutePath}", it)
        }.getOrDefault(false)
    }

    private fun writeEntry(
        raf: RandomAccessFile,
        entryOffset: Long,
        type: Int,
        count: Long,
        value: ByteArray,
        byteOrder: ByteOrder
    ) {
        raf.seek(entryOffset + 2L)
        raf.write(shortBytes(type, byteOrder))
        raf.write(intBytes(count, byteOrder))
        val valueOrOffset = if (value.size <= 4) {
            ByteArray(4).also { value.copyInto(it, endIndex = value.size) }
        } else {
            val valueOffset = appendValue(raf, value)
            intBytes(valueOffset, byteOrder)
        }
        raf.seek(entryOffset + 8L)
        raf.write(valueOrOffset)
    }

    private fun appendValue(raf: RandomAccessFile, value: ByteArray): Long {
        var offset = raf.length()
        if ((offset and 1L) != 0L) {
            raf.seek(offset)
            raf.write(0)
            offset += 1L
        }
        raf.seek(offset)
        raf.write(value)
        return offset
    }

    private fun readUnsignedShort(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Int {
        return ByteBuffer.wrap(bytes, offset, 2).order(byteOrder).short.toInt() and 0xFFFF
    }

    private fun readUnsignedInt(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Long {
        return ByteBuffer.wrap(bytes, offset, 4).order(byteOrder).int.toLong() and 0xFFFFFFFFL
    }

    private fun shortBytes(value: Int, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(2).order(byteOrder).putShort((value and 0xFFFF).toShort()).array()

    private fun intBytes(value: Long, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(4).order(byteOrder).putInt((value and 0xFFFFFFFFL).toInt()).array()
}
