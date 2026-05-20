package com.hinnka.mycamera.utils

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

object DngBlackLevelPatcher {
    private const val TAG = "DngBlackLevelPatcher"
    private const val TIFF_TAG_BLACK_LEVEL = 50714

    fun patchFromMode(file: File, mode: String?, customBlackLevel: Float?): Boolean {
        val level = resolveBlackLevel(mode, customBlackLevel) ?: return false
        return patchUniformBlackLevel(file, level)
    }

    private fun resolveBlackLevel(mode: String?, customBlackLevel: Float?): Float? {
        return when (mode) {
            "0" -> 0f
            "16" -> 16f
            "64" -> 64f
            "256" -> 256f
            "512" -> 512f
            "Custom" -> customBlackLevel ?: 0f
            else -> null
        }
    }

    private fun patchUniformBlackLevel(file: File, level: Float): Boolean {
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

                // Detect if the DNG uses a normalized WhiteLevel (65535) and skip patching to prevent dark clipping
                var isNormalizedDng = false
                for (entryIndex in 0 until entryCount) {
                    val entryOffset = firstIfdOffset + 2L + entryIndex * 12L
                    if (entryOffset + 12L > raf.length()) break
                    raf.seek(entryOffset)
                    val entry = ByteArray(12)
                    raf.readFully(entry)
                    val tag = readUnsignedShort(entry, 0, byteOrder)
                    if (tag == 50717) { // TAG_WHITE_LEVEL
                        val type = readUnsignedShort(entry, 2, byteOrder)
                        val count = readUnsignedInt(entry, 4, byteOrder)
                        if (count == 1L) {
                            val value = if (type == 3) {
                                readUnsignedShort(entry, 8, byteOrder).toLong()
                            } else {
                                readUnsignedInt(entry, 8, byteOrder)
                            }
                            if (value == 65535L) {
                                isNormalizedDng = true
                            }
                        }
                        break
                    }
                }
                if (isNormalizedDng) {
                    PLog.i(TAG, "DNG has normalized WhiteLevel (65535), skipping physical BlackLevel patch to prevent color clipping")
                    return false
                }

                for (entryIndex in 0 until entryCount) {
                    val entryOffset = firstIfdOffset + 2L + entryIndex * 12L
                    if (entryOffset + 12L > raf.length()) {
                        break
                    }

                    val entry = ByteArray(12)
                    raf.seek(entryOffset)
                    raf.readFully(entry)
                    val tag = readUnsignedShort(entry, 0, byteOrder)
                    if (tag != TIFF_TAG_BLACK_LEVEL) {
                        continue
                    }

                    val type = readUnsignedShort(entry, 2, byteOrder)
                    val count = readUnsignedInt(entry, 4, byteOrder)
                    val valueBytes = valueByteCount(type, count) ?: return false
                    val valueOffset = if (valueBytes <= 4L) {
                        entryOffset + 8L
                    } else {
                        readUnsignedInt(entry, 8, byteOrder)
                    }
                    if (valueOffset < 0L || valueOffset + valueBytes > raf.length()) {
                        return false
                    }

                    writeBlackLevelValues(raf, valueOffset, type, count, byteOrder, level)
                    PLog.d(TAG, "Patched DNG BlackLevel to $level in ${file.name}")
                    return true
                }
                false
            }
        }.onFailure {
            PLog.w(TAG, "Failed to patch DNG BlackLevel: ${file.absolutePath}", it)
        }.getOrDefault(false)
    }

    private fun writeBlackLevelValues(
        raf: RandomAccessFile,
        valueOffset: Long,
        type: Int,
        count: Long,
        byteOrder: ByteOrder,
        level: Float
    ) {
        raf.seek(valueOffset)
        val rounded = level.roundToInt().coerceAtLeast(0)
        repeat(count.coerceAtMost(4096L).toInt()) {
            when (type) {
                3 -> raf.write(shortBytes(rounded.coerceAtMost(0xFFFF), byteOrder))
                4 -> raf.write(intBytes(rounded, byteOrder))
                5 -> {
                    raf.write(intBytes(rounded, byteOrder))
                    raf.write(intBytes(1, byteOrder))
                }
                9 -> raf.write(intBytes(rounded, byteOrder))
                10 -> {
                    raf.write(intBytes(rounded, byteOrder))
                    raf.write(intBytes(1, byteOrder))
                }
                11 -> raf.write(floatBytes(level, byteOrder))
                12 -> raf.write(doubleBytes(level.toDouble(), byteOrder))
                else -> return
            }
        }
    }

    private fun valueByteCount(type: Int, count: Long): Long? {
        val typeSize = when (type) {
            1, 2, 6, 7 -> 1L
            3, 8 -> 2L
            4, 9, 11 -> 4L
            5, 10, 12 -> 8L
            else -> return null
        }
        return typeSize * count
    }

    private fun readUnsignedShort(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Int {
        return ByteBuffer.wrap(bytes, offset, 2).order(byteOrder).short.toInt() and 0xFFFF
    }

    private fun readUnsignedInt(bytes: ByteArray, offset: Int, byteOrder: ByteOrder): Long {
        return ByteBuffer.wrap(bytes, offset, 4).order(byteOrder).int.toLong() and 0xFFFFFFFFL
    }

    private fun shortBytes(value: Int, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(2).order(byteOrder).putShort(value.toShort()).array()

    private fun intBytes(value: Int, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(4).order(byteOrder).putInt(value).array()

    private fun floatBytes(value: Float, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(4).order(byteOrder).putFloat(value).array()

    private fun doubleBytes(value: Double, byteOrder: ByteOrder): ByteArray =
        ByteBuffer.allocate(8).order(byteOrder).putDouble(value).array()
}
