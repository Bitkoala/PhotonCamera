package com.hinnka.mycamera.processor

import android.graphics.Bitmap
import android.graphics.ColorSpace
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import androidx.core.graphics.createBitmap
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.model.SafeImage
import com.hinnka.mycamera.utils.BitmapUtils
import java.nio.ByteOrder
import kotlin.math.roundToInt

data class RawStackResult(
    var fusedBayerBuffer: ByteBuffer?,
    val width: Int,
    val height: Int,
    val isNormalizedSensorData: Boolean,
)

fun calculateRawSuperResolutionOutputSize(inputSize: Int, scale: Float): Int {
    return ((inputSize.toFloat() * scale / 2f).roundToInt() * 2).coerceAtLeast(2)
}

/**
 * Multi-Frame Stacker
 * 
 * Manages the native stacking process for burst captures.
 * Aligns and merges multiple frames to reduce noise and improve quality.
 */
object MultiFrameStacker {
    private const val TAG = "MultiFrameStacker"

    private data class CachedVulkanStacker(
        val ptr: Long,
        val width: Int,
        val height: Int,
        val enableSuperResolution: Boolean,
    )

    private var cachedVulkanStacker: CachedVulkanStacker? = null

    init {
        try {
            System.loadLibrary("my-native-lib")
        } catch (e: UnsatisfiedLinkError) {
            PLog.e(TAG, "Failed to load native library", e)
        }
    }

    /**
     * Process a burst of images and return a stacked Bitmap.
     * 
     * @param images List of captured Images (YUV_420_888).
     * @return Stacked Bitmap (ARGB_8888), or null if failed.
     */
    @Synchronized
    fun processBurst(
        images: List<SafeImage>,
        rotation: Int,
        aspectRatio: AspectRatio?,
        enableSuperResolution: Boolean = false,
        useVulkan: Boolean = true,
        colorSpace: ColorSpace,
    ): Bitmap? {
        if (images.isEmpty()) return null

        val width = images[0].width
        val height = images[0].height

        val scale = if (enableSuperResolution) 2 else 1
        val startTime = System.currentTimeMillis()

        if (useVulkan) {
            PLog.i(
                TAG,
                "Starting Vulkan stacking process for ${images.size} frames ($width x $height). SR=$enableSuperResolution"
            )
            val stackerPtr = obtainVulkanStacker(width, height, enableSuperResolution, resetForUse = true)
            if (stackerPtr != 0L) {
                try {
                    for (image in images) {
                        image.use {
                            val hardwareBuffer = it.image.hardwareBuffer
                            if (hardwareBuffer != null) {
                                addVulkanFrameNative(stackerPtr, hardwareBuffer)
                            } else {
                                PLog.w(TAG, "Image has no hardware buffer, skipping")
                            }
                        }
                    }
                    PLog.d(TAG, "Stack frames processed")

                    val dimensions = BitmapUtils.calculateProcessedRect(width, height, aspectRatio, null, rotation)
                    val targetW = dimensions.width() * scale
                    val targetH = dimensions.height() * scale
                    val previewBitmap = try {
                        createBitmap(targetW, targetH, colorSpace = colorSpace)
                    } catch (e: OutOfMemoryError) {
                        PLog.e(TAG, "OOM creating Vulkan stack bitmap ($targetW x $targetH)", e)
                        return null
                    }

                    val processOk = processVulkanStackNative(stackerPtr, previewBitmap, rotation)
                    if (!processOk) {
                        PLog.w(TAG, "Vulkan stack processing failed, invalidating cached stacker")
                        invalidateCachedVulkanStacker(stackerPtr)
                        return null
                    } else {
                        PLog.i(TAG, "Vulkan stacking completed in ${System.currentTimeMillis() - startTime}ms")
                        return previewBitmap
                    }
                } catch (e: Exception) {
                    PLog.e(TAG, "Error during Vulkan stacking", e)
                    invalidateCachedVulkanStacker(stackerPtr)
                    return null
                }
            }
            PLog.e(TAG, "Failed to obtain Vulkan stacker")
            return null
        }

        PLog.e(TAG, "YUV CPU stacking path is disabled")
        return null
    }

    @Synchronized
    fun prewarmVulkanStacker(
        width: Int,
        height: Int,
        enableSuperResolution: Boolean = false,
    ): Boolean {
        val stackerPtr = obtainVulkanStacker(width, height, enableSuperResolution, resetForUse = false)
        val success = stackerPtr != 0L
        if (success) {
            PLog.i(
                TAG,
                "Prewarmed Vulkan stacker for ${width}x$height SR=$enableSuperResolution"
            )
        }
        return success
    }

    @Synchronized
    fun releaseCachedVulkanStacker() {
        cachedVulkanStacker?.let {
            releaseVulkanStackerNative(it.ptr)
            PLog.i(
                TAG,
                "Released cached Vulkan stacker for ${it.width}x${it.height} SR=${it.enableSuperResolution}"
            )
        }
        cachedVulkanStacker = null
    }

    private fun obtainVulkanStacker(
        width: Int,
        height: Int,
        enableSuperResolution: Boolean,
        resetForUse: Boolean,
    ): Long {
        val cached = cachedVulkanStacker
        if (cached != null &&
            cached.width == width &&
            cached.height == height &&
            cached.enableSuperResolution == enableSuperResolution
        ) {
            if (!resetForUse || resetVulkanStackerNative(cached.ptr)) {
                return cached.ptr
            }
            PLog.w(TAG, "Failed to reset cached Vulkan stacker, recreating")
            releaseVulkanStackerNative(cached.ptr)
            cachedVulkanStacker = null
        } else if (cached != null) {
            releaseVulkanStackerNative(cached.ptr)
            cachedVulkanStacker = null
        }

        val stackerPtr = createVulkanStackerNative(width, height, enableSuperResolution)
        if (stackerPtr != 0L) {
            cachedVulkanStacker = CachedVulkanStacker(
                ptr = stackerPtr,
                width = width,
                height = height,
                enableSuperResolution = enableSuperResolution,
            )
        }
        return stackerPtr
    }

    private fun invalidateCachedVulkanStacker(stackerPtr: Long) {
        val cached = cachedVulkanStacker
        if (cached != null && cached.ptr == stackerPtr) {
            releaseVulkanStackerNative(stackerPtr)
            cachedVulkanStacker = null
        }
    }

    fun processBurstRaw(
        images: List<SafeImage>,
        cfaPattern: Int,
        enableSuperResolution: Boolean = false,
        superResolutionScale: Float = 1.2f,
        useVulkan: Boolean = true,
        masterBlackLevel: FloatArray = floatArrayOf(0f, 0f, 0f, 0f),
        whiteLevel: Int = 1023,
        whiteBalanceGains: FloatArray = floatArrayOf(1f, 1f, 1f, 1f),
        noiseModel: FloatArray = floatArrayOf(0f, 0f),
        lensShading: FloatArray? = null,
        lensShadingWidth: Int = 0,
        lensShadingHeight: Int = 0,
    ): RawStackResult? {
        val width = images[0].width
        val height = images[0].height

        PLog.d(
            TAG,
            "Starting RAW stacking for ${images.size} frames. Pattern=$cfaPattern SR=$enableSuperResolution scale=$superResolutionScale Vulkan=$useVulkan WL=$whiteLevel"
        )
        val outputScale = if (enableSuperResolution) superResolutionScale.coerceIn(1.0f, 2.0f) else 1.0f
        if (!useVulkan) {
            PLog.e(TAG, "RAW CPU stacking path is disabled")
            return null
        }

        val vulkanStackerPtr = createVulkanRawStackerNative(
            width, height, enableSuperResolution, outputScale,
            masterBlackLevel, whiteLevel, whiteBalanceGains, noiseModel,
            lensShading, lensShadingWidth, lensShadingHeight
        )
        if (vulkanStackerPtr != 0L) {
            PLog.i(TAG, "Using Vulkan RAW stacker")
            var vulkanFusedBayer: ByteBuffer? = null
            try {
                for (image in images) {
                    image.use {
                        if (image.width != width || image.height != height) return@use
                        val buffer = image.planes[0].buffer
                        val rowStride = image.planes[0].rowStride
                        addVulkanRawFrameNative(vulkanStackerPtr, buffer, rowStride, cfaPattern)
                    }
                }

                val outWidth = calculateRawSuperResolutionOutputSize(width, outputScale)
                val outHeight = calculateRawSuperResolutionOutputSize(height, outputScale)
                vulkanFusedBayer = try {
                    ByteBuffer.allocateDirect(outWidth * outHeight * 2)
                        .order(ByteOrder.nativeOrder())
                } catch (e: OutOfMemoryError) {
                    PLog.e(TAG, "OOM allocating Vulkan fused Bayer buffer", e)
                    return null
                }

                val fusedOk = processVulkanRawStackNative(vulkanStackerPtr, vulkanFusedBayer)
                if (fusedOk) {
                    vulkanFusedBayer.rewind()
                    PLog.i(TAG, "Vulkan RAW stacking completed successfully")
                    return RawStackResult(
                        fusedBayerBuffer = vulkanFusedBayer,
                        width = outWidth,
                        height = outHeight,
                        isNormalizedSensorData = true
                    )
                } else {
                    PLog.e(TAG, "Vulkan RAW stacking failed")
                    return null
                }
            } catch (e: Exception) {
                PLog.e(TAG, "Vulkan RAW stacking error: ${e.message}", e)
                return null
            } finally {
                releaseVulkanRawStackerNative(vulkanStackerPtr)
            }
        } else {
            PLog.e(TAG, "Failed to create Vulkan RAW stacker")
            return null
        }
    }

    // --- Native Methods ---

    private external fun createVulkanStackerNative(width: Int, height: Int, enableSuperRes: Boolean): Long
    private external fun addVulkanFrameNative(
        stackerPtr: Long,
        hardwareBuffer: android.hardware.HardwareBuffer
    ): Boolean

    private external fun processVulkanStackNative(stackerPtr: Long, outBitmap: Bitmap?, rotation: Int): Boolean
    private external fun releaseVulkanStackerNative(stackerPtr: Long)
    private external fun resetVulkanStackerNative(stackerPtr: Long): Boolean

    // Vulkan RAW Stacker
    private external fun createVulkanRawStackerNative(
        width: Int, height: Int, enableSuperRes: Boolean, superResScale: Float,
        blackLevel: FloatArray, whiteLevel: Int, wbGains: FloatArray, noiseModel: FloatArray,
        lensShadingMap: FloatArray?, shadingMapWidth: Int, shadingMapHeight: Int
    ): Long

    private external fun addVulkanRawFrameNative(
        stackerPtr: Long,
        rawData: ByteBuffer,
        rowStride: Int,
        cfaPattern: Int
    ): Boolean

    private external fun processVulkanRawStackNative(stackerPtr: Long, outputBuffer: ByteBuffer): Boolean
    private external fun releaseVulkanRawStackerNative(stackerPtr: Long)
}
