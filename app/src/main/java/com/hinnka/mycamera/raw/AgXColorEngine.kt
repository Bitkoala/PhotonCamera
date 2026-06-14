package com.hinnka.mycamera.raw

import android.content.Context
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

data class AgxLut(
    val name: String,
    val sourceKey: String,
    val size: Int,
    val rgbaFloatBuffer: FloatBuffer
)

object AgXColorEngine {
    private const val TAG = "AgXColorEngine"
    private const val BASE_SRGB_ASSET = "agx/AgX_Base_sRGB.plut"
    private const val MAGIC_PLUT = 0x54554C50
    private const val DATA_TYPE_UINT8 = 0
    private const val DATA_TYPE_UINT16 = 1

    @Volatile
    private var cachedBaseSrgb: AgxLut? = null

    fun loadBaseSrgbLut(context: Context): AgxLut? {
        cachedBaseSrgb?.let { return it }
        return synchronized(this) {
            cachedBaseSrgb ?: loadPlut(context, BASE_SRGB_ASSET)?.also { cachedBaseSrgb = it }
        }
    }

    private fun loadPlut(context: Context, assetPath: String): AgxLut? {
        val startTime = System.currentTimeMillis()
        return try {
            val bytes = context.assets.open(assetPath).use { it.readBytes() }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            if (buffer.remaining() < 16) {
                PLog.e(TAG, "PLUT $assetPath is too short: ${bytes.size} bytes")
                return null
            }

            val magic = buffer.int
            val version = buffer.int
            val size = buffer.int
            val dataType = buffer.int
            if (magic != MAGIC_PLUT || size <= 1 || dataType !in setOf(DATA_TYPE_UINT8, DATA_TYPE_UINT16)) {
                PLog.e(
                    TAG,
                    "Invalid AgX PLUT $assetPath: magic=$magic version=$version size=$size dataType=$dataType"
                )
                return null
            }

            if (version >= 2 && buffer.remaining() >= 4) {
                buffer.int
            }
            if (version >= 3 && buffer.remaining() >= 4) {
                buffer.int
            }

            val texelCount = size * size * size
            val componentCount = texelCount * 3
            val payloadBytes = componentCount * if (dataType == DATA_TYPE_UINT16) 2 else 1
            if (buffer.remaining() < payloadBytes) {
                PLog.e(TAG, "AgX PLUT $assetPath payload too short: remaining=${buffer.remaining()} expected=$payloadBytes")
                return null
            }

            val uploadBuffer = ByteBuffer
                .allocateDirect(texelCount * 4 * java.lang.Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()

            repeat(texelCount) {
                uploadBuffer.put(readNormalizedComponent(buffer, dataType))
                uploadBuffer.put(readNormalizedComponent(buffer, dataType))
                uploadBuffer.put(readNormalizedComponent(buffer, dataType))
                uploadBuffer.put(1f)
            }
            uploadBuffer.position(0)

            PLog.d(
                TAG,
                "Loaded AgX PLUT ${assetPath.substringAfterLast('/')} ($size^3) in " +
                    "${System.currentTimeMillis() - startTime}ms"
            )
            AgxLut(
                name = assetPath.substringAfterLast('/').substringBeforeLast('.'),
                sourceKey = "$assetPath:$version:$dataType:$size:$payloadBytes",
                size = size,
                rgbaFloatBuffer = uploadBuffer.apply { position(0) }
            )
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to load AgX PLUT from $assetPath", e)
            null
        }
    }

    private fun readNormalizedComponent(buffer: ByteBuffer, dataType: Int): Float {
        return if (dataType == DATA_TYPE_UINT16) {
            (buffer.short.toInt() and 0xFFFF) / 65535f
        } else {
            (buffer.get().toInt() and 0xFF) / 255f
        }
    }
}
