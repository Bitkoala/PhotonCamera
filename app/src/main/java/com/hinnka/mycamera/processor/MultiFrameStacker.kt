package com.hinnka.mycamera.processor

import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.media.Image
import android.util.Log
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
    val blackLevel: FloatArray = floatArrayOf(0f, 0f, 0f, 0f),
)

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

    private data class CachedVulkanRawStacker(
        val ptr: Long,
        val width: Int,
        val height: Int,
        val enableSuperResolution: Boolean,
        val superResolutionScale: Float,
        val blackLevel: FloatArray,
        val whiteLevel: Int,
        val wbGains: FloatArray,
        val noiseModel: FloatArray,
        val lensShading: FloatArray?,
        val lensShadingWidth: Int,
        val lensShadingHeight: Int,
    ) {
        fun matches(
            width: Int,
            height: Int,
            enableSuperResolution: Boolean,
            superResolutionScale: Float,
            blackLevel: FloatArray,
            whiteLevel: Int,
            wbGains: FloatArray,
            noiseModel: FloatArray,
            lensShading: FloatArray?,
            lensShadingWidth: Int,
            lensShadingHeight: Int,
        ): Boolean {
            return this.width == width &&
                this.height == height &&
                this.enableSuperResolution == enableSuperResolution &&
                this.superResolutionScale == superResolutionScale &&
                this.whiteLevel == whiteLevel &&
                this.lensShadingWidth == lensShadingWidth &&
                this.lensShadingHeight == lensShadingHeight &&
                this.blackLevel.contentEquals(blackLevel) &&
                this.wbGains.contentEquals(wbGains) &&
                this.noiseModel.contentEquals(noiseModel) &&
                this.lensShading.contentEqualsNullable(lensShading)
        }
    }

    private var cachedVulkanStacker: CachedVulkanStacker? = null
    private var cachedVulkanRawStacker: CachedVulkanRawStacker? = null

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
        outputPath: String? = null,
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
                    } else {
                        PLog.i(TAG, "Vulkan stacking completed in ${System.currentTimeMillis() - startTime}ms")
                        return previewBitmap
                    }
                } catch (e: Exception) {
                    PLog.e(TAG, "Error during Vulkan stacking", e)
                    invalidateCachedVulkanStacker(stackerPtr)
                }
            }
        }

        // Fallback or legacy path
        PLog.i(
            TAG,
            "Starting legacy stacking process for ${images.size} frames ($width x $height). SR=$enableSuperResolution"
        )
        val stackerPtr = createStackerNative(width, height, enableSuperResolution)
        if (stackerPtr == 0L) return null

        try {
            val stagedIndices = mutableListOf<Int>()
            for (image in images) {
                image.use {
                    val planes = image.planes
                    stageFrameNative(
                        stackerPtr,
                        planes[0].buffer, planes[1].buffer, planes[2].buffer,
                        planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride,
                        image.format
                    )
                    stagedIndices.add(stagedIndices.size)
                }
            }

            for (idx in stagedIndices) {
                processFrameNative(stackerPtr, idx)
            }
            clearStagedFramesNative(stackerPtr)

            val dimensions = BitmapUtils.calculateProcessedRect(width, height, aspectRatio, null, rotation)
            val targetW = dimensions.width() * scale
            val targetH = dimensions.height() * scale
            val previewBitmap = try {
                createBitmap(targetW, targetH, colorSpace = colorSpace)
            } catch (e: OutOfMemoryError) {
                PLog.e(TAG, "OOM creating legacy stack bitmap ($targetW x $targetH)", e)
                return null
            }

            processStackNative(
                stackerPtr,
                previewBitmap,
                rotation,
                aspectRatio?.widthRatio ?: width,
                aspectRatio?.heightRatio ?: height,
                outputPath
            )

            PLog.i(TAG, "Legacy stacking completed in ${System.currentTimeMillis() - startTime}ms")
            return previewBitmap
        } finally {
            releaseStackerNative(stackerPtr)
        }
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
        cachedVulkanRawStacker?.let {
            releaseVulkanRawStackerNative(it.ptr)
            PLog.i(
                TAG,
                "Released cached Vulkan RAW stacker for ${it.width}x${it.height} SR=${it.enableSuperResolution} scale=${it.superResolutionScale}"
            )
        }
        cachedVulkanRawStacker = null
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

    private fun obtainVulkanRawStacker(
        width: Int,
        height: Int,
        enableSuperResolution: Boolean,
        superResolutionScale: Float,
        blackLevel: FloatArray,
        whiteLevel: Int,
        wbGains: FloatArray,
        noiseModel: FloatArray,
        lensShading: FloatArray?,
        lensShadingWidth: Int,
        lensShadingHeight: Int,
    ): Long {
        val cached = cachedVulkanRawStacker
        if (cached != null &&
            cached.matches(
                width, height, enableSuperResolution, superResolutionScale,
                blackLevel, whiteLevel, wbGains, noiseModel,
                lensShading, lensShadingWidth, lensShadingHeight
            )
        ) {
            if (resetVulkanRawStackerNative(cached.ptr)) {
                PLog.d(TAG, "Reusing cached Vulkan RAW stacker")
                return cached.ptr
            }
            PLog.w(TAG, "Failed to reset cached Vulkan RAW stacker, recreating")
            releaseVulkanRawStackerNative(cached.ptr)
            cachedVulkanRawStacker = null
        } else if (cached != null) {
            releaseVulkanRawStackerNative(cached.ptr)
            cachedVulkanRawStacker = null
        }

        val stackerPtr = createVulkanRawStackerNative(
            width, height, enableSuperResolution, superResolutionScale,
            blackLevel, whiteLevel, wbGains, noiseModel,
            lensShading, lensShadingWidth, lensShadingHeight
        )
        if (stackerPtr != 0L) {
            cachedVulkanRawStacker = CachedVulkanRawStacker(
                ptr = stackerPtr,
                width = width,
                height = height,
                enableSuperResolution = enableSuperResolution,
                superResolutionScale = superResolutionScale,
                blackLevel = blackLevel.copyOf(),
                whiteLevel = whiteLevel,
                wbGains = wbGains.copyOf(),
                noiseModel = noiseModel.copyOf(),
                lensShading = lensShading?.copyOf(),
                lensShadingWidth = lensShadingWidth,
                lensShadingHeight = lensShadingHeight,
            )
        }
        return stackerPtr
    }

    private fun invalidateCachedVulkanRawStacker(stackerPtr: Long) {
        val cached = cachedVulkanRawStacker
        if (cached != null && cached.ptr == stackerPtr) {
            releaseVulkanRawStackerNative(stackerPtr)
            cachedVulkanRawStacker = null
        }
    }

    private fun FloatArray?.contentEqualsNullable(other: FloatArray?): Boolean {
        return when {
            this == null && other == null -> true
            this == null || other == null -> false
            else -> this.contentEquals(other)
        }
    }

    fun processBurstRaw(
        images: List<SafeImage>,
        cfaPattern: Int,
        enableSuperResolution: Boolean = false,
        superResolutionScale: Float = 1.5f,
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
            "Starting RAW stacking for ${images.size} frames. Pattern=$cfaPattern SR=$enableSuperResolution scale=$superResolutionScale Vulkan=$useVulkan BL=${masterBlackLevel.joinToString()} WL=$whiteLevel"
        )
        val outputScale = if (enableSuperResolution) superResolutionScale.coerceIn(1.0f, 2.0f) else 1.0f
        val useNativeSuperResolution = outputScale > 1.0f

        if (useVulkan) {
            val vulkanStackerPtr = obtainVulkanRawStacker(
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

                    val outWidth = (width * outputScale).roundToInt()
                    val outHeight = (height * outputScale).roundToInt()
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
                            isNormalizedSensorData = true,
                            blackLevel = masterBlackLevel.copyOf(),
                        )
                    } else {
                        PLog.w(TAG, "Vulkan RAW stacking failed, falling back to CPU")
                        invalidateCachedVulkanRawStacker(vulkanStackerPtr)
                    }
                } catch (e: Exception) {
                    PLog.e(TAG, "Vulkan RAW stacking error: ${e.message}, falling back to CPU")
                    invalidateCachedVulkanRawStacker(vulkanStackerPtr)
                }
                // fallback 到 CPU 路径前显式释放 Vulkan buffer 引用，避免与 CPU 路径的分配叠加
                // 仅 fused Bayer 持续到 DNG 保存完成。
                vulkanFusedBayer = null
                @Suppress("ExplicitGarbageCollectionCall")
                System.gc()
            } else {
                PLog.w(TAG, "Failed to create Vulkan RAW stacker, falling back to CPU")
            }
        }

        PLog.i(TAG, "Using CPU RAW stacker")
        val stackerPtr = createRawStackerNative(width, height, useNativeSuperResolution)
        if (stackerPtr == 0L) {
            PLog.e(TAG, "Failed to create CPU raw stacker")
            return null
        }

        try {
            val stagedIndices = mutableListOf<Int>()
            for (image in images) {
                image.use {
                    if (image.width != width || image.height != height) return@use
                    val buffer = image.planes[0].buffer
                    val rowStride = image.planes[0].rowStride
                    stageRawFrameNative(stackerPtr, buffer, rowStride, cfaPattern)
                    stagedIndices.add(stagedIndices.size)
                }
            }
            for (idx in stagedIndices) {
                processRawFrameNative(stackerPtr, idx)
            }
            clearStagedRawFramesNative(stackerPtr)

            val stackedWidth = if (useNativeSuperResolution) width * 2 else width
            val stackedHeight = if (useNativeSuperResolution) height * 2 else height
            val fusedBayerBuffer = try {
                ByteBuffer.allocateDirect(stackedWidth * stackedHeight * 2)
                    .order(ByteOrder.nativeOrder())
            } catch (e: OutOfMemoryError) {
                PLog.e(TAG, "OOM allocating fused Bayer buffer", e)
                return null
            }
            processRawStackWithBufferNative(stackerPtr, fusedBayerBuffer)

            fusedBayerBuffer.rewind()
            PLog.i(TAG, "CPU RAW stacking completed successfully")
            return RawStackResult(
                fusedBayerBuffer = fusedBayerBuffer,
                width = stackedWidth,
                height = stackedHeight,
                isNormalizedSensorData = false,
                blackLevel = masterBlackLevel.copyOf(),
            )

        } finally {
            releaseRawStackerNative(stackerPtr)
        }
    }

    // --- Native Methods ---

    private external fun createStackerNative(width: Int, height: Int, enableSuperRes: Boolean): Long

    private external fun stageFrameNative(
        stackerPtr: Long,
        yBuffer: ByteBuffer, uBuffer: ByteBuffer, vBuffer: ByteBuffer,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        format: Int
    )

    private external fun processFrameNative(stackerPtr: Long, index: Int)
    private external fun clearStagedFramesNative(stackerPtr: Long)

    private external fun processStackNative(
        stackerPtr: Long,
        outBitmap: Bitmap?,
        rotation: Int,
        targetWR: Int,
        targetHR: Int,
        outputPath: String?
    )

    private external fun releaseStackerNative(stackerPtr: Long)

    private external fun createVulkanStackerNative(width: Int, height: Int, enableSuperRes: Boolean): Long
    private external fun addVulkanFrameNative(
        stackerPtr: Long,
        hardwareBuffer: android.hardware.HardwareBuffer
    ): Boolean

    private external fun processVulkanStackNative(stackerPtr: Long, outBitmap: Bitmap?, rotation: Int): Boolean
    private external fun releaseVulkanStackerNative(stackerPtr: Long)
    private external fun resetVulkanStackerNative(stackerPtr: Long): Boolean

    private external fun createRawStackerNative(width: Int, height: Int, enableSuperRes: Boolean): Long
    private external fun stageRawFrameNative(stackerPtr: Long, rawData: ByteBuffer, rowStride: Int, cfaPattern: Int)
    private external fun processRawFrameNative(stackerPtr: Long, index: Int)
    private external fun clearStagedRawFramesNative(stackerPtr: Long)
    private external fun processRawStackWithBufferNative(stackerPtr: Long, outputBuffer: ByteBuffer)
    private external fun releaseRawStackerNative(stackerPtr: Long)

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
    private external fun resetVulkanRawStackerNative(stackerPtr: Long): Boolean
}
