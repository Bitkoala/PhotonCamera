package com.hinnka.mycamera.utils

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.LensShadingMap
import android.os.Build
import com.hinnka.mycamera.raw.DngProfileGainTableMap
import com.hinnka.mycamera.raw.RawCfaCorrection
import com.hinnka.mycamera.raw.RawWhiteLevelCorrection
import java.io.ByteArrayOutputStream
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.Channels
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt
import kotlin.math.roundToLong

object SuperResolutionDngWriter {
    private const val TAG = "SuperResolutionDngWriter"

    private const val TYPE_BYTE = 1
    private const val TYPE_ASCII = 2
    private const val TYPE_SHORT = 3
    private const val TYPE_LONG = 4
    private const val TYPE_RATIONAL = 5
    private const val TYPE_UNDEFINED = 7
    private const val TYPE_SRATIONAL = 10

    private const val TAG_NEW_SUBFILE_TYPE = 254
    private const val TAG_IMAGE_WIDTH = 256
    private const val TAG_IMAGE_LENGTH = 257
    private const val TAG_BITS_PER_SAMPLE = 258
    private const val TAG_COMPRESSION = 259
    private const val TAG_PHOTOMETRIC_INTERPRETATION = 262
    private const val TAG_MAKE = 271
    private const val TAG_MODEL = 272
    private const val TAG_STRIP_OFFSETS = 273
    private const val TAG_ORIENTATION = 274
    private const val TAG_SAMPLES_PER_PIXEL = 277
    private const val TAG_ROWS_PER_STRIP = 278
    private const val TAG_STRIP_BYTE_COUNTS = 279
    private const val TAG_PLANAR_CONFIGURATION = 284
    private const val TAG_SOFTWARE = 305
    private const val TAG_DATETIME = 306
    private const val TAG_CFA_REPEAT_PATTERN_DIM = 33421
    private const val TAG_CFA_PATTERN = 33422
    private const val TAG_DNG_VERSION = 50706
    private const val TAG_DNG_BACKWARD_VERSION = 50707
    private const val TAG_UNIQUE_CAMERA_MODEL = 50708
    private const val TAG_CFA_PLANE_COLOR = 50710
    private const val TAG_CFA_LAYOUT = 50711
    private const val TAG_BLACK_LEVEL_REPEAT_DIM = 50713
    private const val TAG_BLACK_LEVEL = 50714
    private const val TAG_WHITE_LEVEL = 50717
    private const val TAG_DEFAULT_SCALE = 50718
    private const val TAG_DEFAULT_CROP_ORIGIN = 50719
    private const val TAG_DEFAULT_CROP_SIZE = 50720
    private const val TAG_COLOR_MATRIX_1 = 50721
    private const val TAG_COLOR_MATRIX_2 = 50722
    private const val TAG_FORWARD_MATRIX_1 = 50964
    private const val TAG_FORWARD_MATRIX_2 = 50965
    private const val TAG_AS_SHOT_NEUTRAL = 50728
    private const val TAG_BASELINE_EXPOSURE = 50730
    private const val TAG_CALIBRATION_ILLUMINANT_1 = 50778
    private const val TAG_CALIBRATION_ILLUMINANT_2 = 50779
    private const val TAG_ACTIVE_AREA = 50829
    private const val TAG_OPCODE_LIST_2 = 51009
    private const val TAG_PROFILE_GAIN_TABLE_MAP_2 = DngProfileGainTableMap.TAG_PROFILE_GAIN_TABLE_MAP2

    fun write(
        outputStream: OutputStream,
        rawBuffer: ByteBuffer,
        width: Int,
        height: Int,
        characteristics: CameraCharacteristics,
        captureResult: CaptureResult,
        orientation: Int,
        cfaPattern: Int,
        blackLevel: FloatArray,
        whiteLevel: Int,
        valueDomain: RawProcessor.RawBufferValueDomain,
        blackLevelMode: String? = null,
        customBlackLevel: Float? = null,
        whiteLevelMode: String? = null,
        baselineExposureEv: Float = 0f,
        profileGainTableMap: DngProfileGainTableMap? = null,
    ): Boolean {
        if (width <= 0 || height <= 0) return false

        return runCatching {
            val input = rawBuffer.duplicate().order(ByteOrder.nativeOrder())
            input.rewind()

            val resolvedBlackLevel = RawProcessor.resolveBlackLevelForMode(
                defaultBlackLevel = blackLevel,
                blackLevelMode = blackLevelMode,
                customBlackLevel = customBlackLevel
            )
            val hasBlackLevelOverride = blackLevelMode != null && !blackLevel.contentEquals(resolvedBlackLevel)
            val resolvedWhiteLevel = RawWhiteLevelCorrection.resolveWhiteLevel(
                defaultWhiteLevel = whiteLevel.toFloat(),
                mode = whiteLevelMode
            ).toInt()

            val encodedBlackLevel = if (valueDomain == RawProcessor.RawBufferValueDomain.NORMALIZED_SENSOR_RANGE) {
                if (hasBlackLevelOverride) {
                    FloatArray(blackLevel.size) { i ->
                        val defaultBL = blackLevel[i]
                        val correctedBL = resolvedBlackLevel.getOrElse(i) { resolvedBlackLevel.firstOrNull() ?: defaultBL }
                        val defaultWL = whiteLevel.toFloat()
                        if (defaultWL > defaultBL) {
                            ((correctedBL - defaultBL) / (defaultWL - defaultBL) * 65535f).coerceIn(0f, 65535f)
                        } else {
                            0f
                        }
                    }
                } else {
                    floatArrayOf(0f, 0f, 0f, 0f)
                }
            } else {
                if (hasBlackLevelOverride) {
                    resolvedBlackLevel
                } else {
                    blackLevel
                }
            }

            val encodedWhiteLevel = if (valueDomain == RawProcessor.RawBufferValueDomain.NORMALIZED_SENSOR_RANGE) {
                65535
            } else {
                resolvedWhiteLevel
            }

            val imageByteCount = width.toLong() * height.toLong() * 2L
            require(imageByteCount <= Int.MAX_VALUE) { "DNG image is too large for classic TIFF: ${width}x${height}" }

            val entries = buildEntries(
                width = width,
                height = height,
                characteristics = characteristics,
                captureResult = captureResult,
                orientation = orientation,
                cfaPattern = cfaPattern,
                blackLevel = encodedBlackLevel,
                whiteLevel = encodedWhiteLevel,
                imageByteCount = imageByteCount,
                baselineExposureEv = baselineExposureEv,
                profileGainTableMap = profileGainTableMap?.takeIf { it.isValid }
            )
            val header = buildHeader(entries)
            outputStream.write(header)
            writeRawImage(outputStream, input, imageByteCount.toInt())
            outputStream.flush()
            PLog.i(
                TAG,
                "Wrote super-resolution DNG ${width}x${height} " +
                    "blackLevel=${encodedBlackLevel.joinToString()} whiteLevel=$encodedWhiteLevel " +
                    "baselineExposureEv=$baselineExposureEv " +
                    "profileGainTable=${profileGainTableMap?.takeIf { it.isValid }?.let { "${it.mapPointsH}x${it.mapPointsV}x${it.mapPointsN}" } ?: "none"}"
            )
            true
        }.onFailure {
            PLog.w(TAG, "Failed to write super-resolution DNG ${width}x${height}", it)
        }.getOrDefault(false)
    }

    private fun buildEntries(
        width: Int,
        height: Int,
        characteristics: CameraCharacteristics,
        captureResult: CaptureResult,
        orientation: Int,
        cfaPattern: Int,
        blackLevel: FloatArray,
        whiteLevel: Int,
        imageByteCount: Long,
        baselineExposureEv: Float,
        profileGainTableMap: DngProfileGainTableMap?,
    ): List<TiffEntry> {
        // The custom writer is used when the fused RAW dimensions no longer
        // match the camera sensor. The fused buffer already lives in its final
        // Bayer pixel grid, so DefaultScale must stay 1:1. Scaling it back to
        // the physical sensor size makes RAW decoders downsample the >1x DNG
        // before demosaic, which looks blurrier than the original 1x frame.
        val defaultScaleX = 1.0
        val defaultScaleY = 1.0
        val cameraModel = buildCameraModel(characteristics)
        val illuminant1 = characteristics.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1) ?: 21
        val illuminant2 = characteristics.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2)?.toInt()
        val colorMatrix1 = characteristics.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1)
        val colorMatrix2 = characteristics.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2)
        val forwardMatrix1 = characteristics.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX1)
        val forwardMatrix2 = characteristics.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX2)
        val opcodeList2 = buildOpcodeList2(
            captureResult = captureResult,
            cfaPattern = cfaPattern,
            width = width,
            height = height
        )

        return buildList {
            add(long(TAG_NEW_SUBFILE_TYPE, 0))
            add(long(TAG_IMAGE_WIDTH, width.toLong()))
            add(long(TAG_IMAGE_LENGTH, height.toLong()))
            add(short(TAG_BITS_PER_SAMPLE, 16))
            add(short(TAG_COMPRESSION, 1))
            add(short(TAG_PHOTOMETRIC_INTERPRETATION, 32803))
            add(ascii(TAG_MAKE, Build.MANUFACTURER.ifBlank { "Android" }))
            add(ascii(TAG_MODEL, Build.MODEL.ifBlank { cameraModel }))
            add(long(TAG_STRIP_OFFSETS, 0))
            add(short(TAG_ORIENTATION, orientation))
            add(short(TAG_SAMPLES_PER_PIXEL, 1))
            add(long(TAG_ROWS_PER_STRIP, height.toLong()))
            add(long(TAG_STRIP_BYTE_COUNTS, imageByteCount))
            add(short(TAG_PLANAR_CONFIGURATION, 1))
            add(ascii(TAG_SOFTWARE, "PhotonCamera"))
            add(ascii(TAG_DATETIME, SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Locale.US).format(Date())))
            add(shortArray(TAG_CFA_REPEAT_PATTERN_DIM, RawCfaCorrection.repeatPatternDim(cfaPattern)))
            add(byteArray(TAG_CFA_PATTERN, RawCfaCorrection.cfaPatternBytes(cfaPattern)))
            add(byteArray(TAG_DNG_VERSION, if (profileGainTableMap != null) {
                byteArrayOf(1, 7, 0, 0)
            } else {
                byteArrayOf(1, 4, 0, 0)
            }))
            add(byteArray(TAG_DNG_BACKWARD_VERSION, byteArrayOf(1, 1, 0, 0)))
            add(ascii(TAG_UNIQUE_CAMERA_MODEL, cameraModel))
            add(byteArray(TAG_CFA_PLANE_COLOR, byteArrayOf(0, 1, 2)))
            add(short(TAG_CFA_LAYOUT, 1))
            add(shortArray(TAG_BLACK_LEVEL_REPEAT_DIM, RawCfaCorrection.repeatPatternDim(cfaPattern)))
            add(rationalArray(TAG_BLACK_LEVEL, blackLevelByCfaPosition(cfaPattern, blackLevel).map { it.toDouble() }))
            add(long(TAG_WHITE_LEVEL, whiteLevel.coerceAtLeast(1).toLong()))
            add(rationalArray(TAG_DEFAULT_SCALE, listOf(defaultScaleX, defaultScaleY)))
            add(rationalArray(TAG_DEFAULT_CROP_ORIGIN, listOf(0.0, 0.0)))
            add(rationalArray(TAG_DEFAULT_CROP_SIZE, listOf(width.toDouble(), height.toDouble())))
            colorMatrix1?.let { add(sRationalArray(TAG_COLOR_MATRIX_1, colorTransformToDngMatrix(it))) }
            colorMatrix2?.let { add(sRationalArray(TAG_COLOR_MATRIX_2, colorTransformToDngMatrix(it))) }
            forwardMatrix1?.let { add(sRationalArray(TAG_FORWARD_MATRIX_1, colorTransformToDngMatrix(it))) }
            forwardMatrix2?.let { add(sRationalArray(TAG_FORWARD_MATRIX_2, colorTransformToDngMatrix(it))) }
            add(rationalArray(TAG_AS_SHOT_NEUTRAL, asShotNeutral(captureResult)))
            add(sRationalArray(TAG_BASELINE_EXPOSURE, listOf(baselineExposureEv.toDouble())))
            add(short(TAG_CALIBRATION_ILLUMINANT_1, illuminant1))
            if (illuminant2 != null && colorMatrix2 != null) {
                add(short(TAG_CALIBRATION_ILLUMINANT_2, illuminant2))
            }
            add(longArray(TAG_ACTIVE_AREA, longArrayOf(0, 0, height.toLong(), width.toLong())))
            opcodeList2?.let { add(undefined(TAG_OPCODE_LIST_2, it)) }
            profileGainTableMap?.let {
                add(undefined(TAG_PROFILE_GAIN_TABLE_MAP_2, it.encodeProfileGainTableMap2(ByteOrder.LITTLE_ENDIAN)))
            }
        }.sortedBy { it.tag }
    }

    private fun buildHeader(entries: List<TiffEntry>): ByteArray {
        val dataArea = ByteArrayOutputStream()
        val entryCount = entries.size
        val dataBaseOffset = 8 + 2 + entryCount * 12 + 4

        val encodedEntries = entries.map { entry ->
            if (entry.value.size <= 4) {
                EncodedEntry(entry, inlineValue(entry.value))
            } else {
                if ((dataArea.size() and 1) != 0) dataArea.write(0)
                val offset = dataBaseOffset + dataArea.size()
                dataArea.write(entry.value)
                EncodedEntry(entry, uintBytes(offset.toLong()))
            }
        }

        if ((dataArea.size() and 1) != 0) dataArea.write(0)
        val rawOffset = dataBaseOffset + dataArea.size()
        val out = ByteArrayOutputStream(rawOffset)
        out.write(byteArrayOf('I'.code.toByte(), 'I'.code.toByte()))
        out.write(ushortBytes(42))
        out.write(uintBytes(8))
        out.write(ushortBytes(entryCount))
        for (encoded in encodedEntries) {
            out.write(ushortBytes(encoded.entry.tag))
            out.write(ushortBytes(encoded.entry.type))
            out.write(uintBytes(encoded.entry.count))
            val value = if (encoded.entry.tag == TAG_STRIP_OFFSETS) {
                uintBytes(rawOffset.toLong())
            } else {
                encoded.valueOrOffset
            }
            out.write(value)
        }
        out.write(uintBytes(0))
        out.write(dataArea.toByteArray())
        return out.toByteArray()
    }

    private fun writeRawImage(outputStream: OutputStream, rawBuffer: ByteBuffer, byteCount: Int) {
        val imageBytes = rawBuffer.duplicate()
        imageBytes.position(0)
        imageBytes.limit(byteCount.coerceAtMost(imageBytes.capacity()))

        if (ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN) {
            if (outputStream is FileOutputStream) {
                outputStream.channel.writeFully(imageBytes)
                return
            }
            Channels.newChannel(outputStream).writeFully(imageBytes)
            return
        }

        val shorts = rawBuffer.order(ByteOrder.nativeOrder()).asShortBuffer()
        val chunk = ByteArray(1024 * 1024)
        var remainingShorts = byteCount / 2
        while (remainingShorts > 0) {
            val shortsThisChunk = minOf(remainingShorts, chunk.size / 2)
            var out = 0
            repeat(shortsThisChunk) {
                val value = shorts.get().toInt() and 0xFFFF
                chunk[out++] = (value and 0xFF).toByte()
                chunk[out++] = ((value ushr 8) and 0xFF).toByte()
            }
            outputStream.write(chunk, 0, shortsThisChunk * 2)
            remainingShorts -= shortsThisChunk
        }
    }

    private fun java.nio.channels.WritableByteChannel.writeFully(buffer: ByteBuffer) {
        while (buffer.hasRemaining()) {
            write(buffer)
        }
    }

    private fun buildCameraModel(characteristics: CameraCharacteristics): String {
        val hardwareLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        return listOfNotNull(
            Build.MANUFACTURER.takeIf { it.isNotBlank() },
            Build.MODEL.takeIf { it.isNotBlank() },
            hardwareLevel?.let { "Camera2-$it" }
        ).joinToString(" ").ifBlank { "Photon Camera" }
    }

    private fun blackLevelByCfaPosition(cfaPattern: Int, blackLevel: FloatArray): List<Float> {
        fun channel(index: Int): Float = blackLevel.getOrElse(index) { blackLevel.firstOrNull() ?: 0f }
        val dim = RawCfaCorrection.repeatPatternDim(cfaPattern)
        return buildList(dim[0] * dim[1]) {
            for (y in 0 until dim[0]) {
                for (x in 0 until dim[1]) {
                    add(channel(RawCfaCorrection.channelIndexForPixel(cfaPattern, x, y)))
                }
            }
        }
    }

    private fun buildOpcodeList2(
        captureResult: CaptureResult,
        cfaPattern: Int,
        width: Int,
        height: Int,
    ): ByteArray? {
        val lensShadingMap = captureResult.get(CaptureResult.STATISTICS_LENS_SHADING_CORRECTION_MAP)
            ?: return null
        if (lensShadingMap.columnCount <= 0 || lensShadingMap.rowCount <= 0) return null

        val dim = RawCfaCorrection.repeatPatternDim(cfaPattern)
        val opcodeCount = dim[0] * dim[1]
        val opcodes = ByteArrayOutputStream()
        opcodes.write(beUInt(opcodeCount.toLong()))
        for (top in 0 until dim[0]) {
            for (left in 0 until dim[1]) {
                val channel = RawCfaCorrection.channelIndexForPixel(cfaPattern, left, top)
                opcodes.write(buildGainMapOpcode(lensShadingMap, channel, top, left, dim[0], dim[1], width, height))
            }
        }
        return opcodes.toByteArray()
    }

    private fun buildGainMapOpcode(
        lensShadingMap: LensShadingMap,
        channel: Int,
        top: Int,
        left: Int,
        rowPitch: Int,
        colPitch: Int,
        width: Int,
        height: Int,
    ): ByteArray {
        val payload = ByteArrayOutputStream()
        payload.write(beInt(top))
        payload.write(beInt(left))
        payload.write(beInt(height))
        payload.write(beInt(width))
        payload.write(beUInt(0))
        payload.write(beUInt(1))
        payload.write(beUInt(rowPitch.toLong()))
        payload.write(beUInt(colPitch.toLong()))
        payload.write(beUInt(lensShadingMap.rowCount.toLong()))
        payload.write(beUInt(lensShadingMap.columnCount.toLong()))
        payload.write(beDouble(if (lensShadingMap.rowCount > 1) 1.0 / (lensShadingMap.rowCount - 1).toDouble() else 1.0))
        payload.write(beDouble(if (lensShadingMap.columnCount > 1) 1.0 / (lensShadingMap.columnCount - 1).toDouble() else 1.0))
        payload.write(beDouble(0.0))
        payload.write(beDouble(0.0))
        payload.write(beUInt(1))

        for (row in 0 until lensShadingMap.rowCount) {
            for (col in 0 until lensShadingMap.columnCount) {
                payload.write(beFloat(lensShadingMap.getGainFactor(channel, col, row)))
            }
        }

        val opcode = ByteArrayOutputStream()
        opcode.write(beUInt(9))
        opcode.write(beUInt(0x01030000L))
        opcode.write(beUInt(0))
        opcode.write(beUInt(payload.size().toLong()))
        opcode.write(payload.toByteArray())
        return opcode.toByteArray()
    }

    private fun asShotNeutral(captureResult: CaptureResult): List<Double> {
        val gains = captureResult.get(CaptureResult.COLOR_CORRECTION_GAINS)
            ?: return listOf(1.0, 1.0, 1.0)
        val green = ((gains.greenEven + gains.greenOdd) * 0.5f).takeIf { it > 0f } ?: 1f
        val r = gains.red.takeIf { it > 0f } ?: green
        val b = gains.blue.takeIf { it > 0f } ?: green
        return listOf((green / r).toDouble(), 1.0, (green / b).toDouble())
    }

    private fun colorTransformToDngMatrix(transform: ColorSpaceTransform): List<Double> {
        val values = ArrayList<Double>(9)
        for (row in 0 until 3) {
            for (col in 0 until 3) {
                values.add(transform.getElement(col, row).toDouble())
            }
        }
        return values
    }

    private fun byteArray(tag: Int, values: ByteArray): TiffEntry =
        TiffEntry(tag, TYPE_BYTE, values.size.toLong(), values)

    private fun undefined(tag: Int, values: ByteArray): TiffEntry =
        TiffEntry(tag, TYPE_UNDEFINED, values.size.toLong(), values)

    private fun ascii(tag: Int, value: String): TiffEntry {
        val bytes = (value + "\u0000").toByteArray(Charsets.US_ASCII)
        return TiffEntry(tag, TYPE_ASCII, bytes.size.toLong(), bytes)
    }

    private fun short(tag: Int, value: Int): TiffEntry =
        TiffEntry(tag, TYPE_SHORT, 1, ushortBytes(value))

    private fun shortArray(tag: Int, values: IntArray): TiffEntry =
        TiffEntry(tag, TYPE_SHORT, values.size.toLong(), ByteArrayOutputStream().apply {
            values.forEach { write(ushortBytes(it)) }
        }.toByteArray())

    private fun long(tag: Int, value: Long): TiffEntry =
        TiffEntry(tag, TYPE_LONG, 1, uintBytes(value))

    private fun longArray(tag: Int, values: LongArray): TiffEntry =
        TiffEntry(tag, TYPE_LONG, values.size.toLong(), ByteArrayOutputStream().apply {
            values.forEach { write(uintBytes(it)) }
        }.toByteArray())

    private fun rationalArray(tag: Int, values: List<Double>): TiffEntry =
        TiffEntry(tag, TYPE_RATIONAL, values.size.toLong(), ByteArrayOutputStream().apply {
            values.forEach { value ->
                val rational = toUnsignedRational(value)
                write(uintBytes(rational.first))
                write(uintBytes(rational.second))
            }
        }.toByteArray())

    private fun sRationalArray(tag: Int, values: List<Double>): TiffEntry =
        TiffEntry(tag, TYPE_SRATIONAL, values.size.toLong(), ByteArrayOutputStream().apply {
            values.forEach { value ->
                val rational = toSignedRational(value)
                write(intBytes(rational.first))
                write(intBytes(rational.second))
            }
        }.toByteArray())

    private fun toUnsignedRational(value: Double): Pair<Long, Long> {
        val safe = value.takeIf { it.isFinite() && it >= 0.0 } ?: 0.0
        val denominator = if (safe >= 4096.0) 1L else 1_000_000L
        val numerator = (safe * denominator).roundToLong().coerceAtLeast(0L)
        return numerator to denominator
    }

    private fun toSignedRational(value: Double): Pair<Int, Int> {
        val safe = value.takeIf { it.isFinite() } ?: 0.0
        val denominator = 1_000_000
        val numerator = (safe * denominator).roundToInt()
        return numerator to denominator
    }

    private fun inlineValue(value: ByteArray): ByteArray =
        ByteArray(4).also { value.copyInto(it, endIndex = value.size.coerceAtMost(4)) }

    private fun ushortBytes(value: Int): ByteArray =
        ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((value and 0xFFFF).toShort()).array()

    private fun uintBytes(value: Long): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt((value and 0xFFFFFFFFL).toInt()).array()

    private fun intBytes(value: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()

    private fun beUInt(value: Long): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putInt((value and 0xFFFFFFFFL).toInt()).array()

    private fun beInt(value: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putInt(value).array()

    private fun beFloat(value: Float): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putFloat(value).array()

    private fun beDouble(value: Double): ByteArray =
        ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN).putDouble(value).array()

    private data class TiffEntry(
        val tag: Int,
        val type: Int,
        val count: Long,
        val value: ByteArray,
    )

    private data class EncodedEntry(
        val entry: TiffEntry,
        val valueOrOffset: ByteArray,
    )
}
