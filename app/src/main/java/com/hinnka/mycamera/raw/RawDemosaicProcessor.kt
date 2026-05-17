package com.hinnka.mycamera.raw

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.media.Image
import android.opengl.*
import androidx.core.graphics.createBitmap
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.data.ContentRepository
import com.hinnka.mycamera.ml.SharedDepthEstimator
import com.hinnka.mycamera.raw.RawProcessingPreferences.DROMode
import com.hinnka.mycamera.utils.BitmapUtils
import com.hinnka.mycamera.utils.PLog
import com.hinnka.mycamera.utils.RawProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt
import android.opengl.Matrix as GlMatrix

/**
 * RAW 图像解马赛克处理器
 *
 * 使用 OpenGL ES 3.0 离屏渲染实现 GPU 加速的 RAW 处理管线：
 * Capture One 风格处理流程:
 * 1. 黑电平扣除
 * 2. 线性白平衡增益
 * 3. 输入锐化/反卷积 (Richardson-Lucy Deconvolution)
 * 4. 解马赛克 (RCD - Ratio Corrected Demosaicing)
 * 5. 色彩转换 (CCM)
 * 6. Gamma 曲线 (Filmic: 短趾部 + Gamma 2.2 + 长肩部)
 * 7. 结构增强 (Structure/Clarity - L通道高通滤波)
 * 8. 最终锐化 (Unsharp Mask)
 */
class RawDemosaicProcessor {

    /**
     * DNG 数据容器（包含原始 DngRawData 用于清理）
     */

    /**
     * 将 DngRawData 转换为 RawMetadata
     */
    private fun convertDngRawDataToMetadata(
        dngRawData: DngRawData,
        exposureBias: Float,
        baseMetadata: RawMetadata? = null
    ): RawMetadata {
        // CFA 模式：使用从 JNI 传递过来的实际值
        val cfaPattern = dngRawData.cfaPattern

        // 黑电平：DngRawData 提供的是 [R, Gr, Gb, B] 四通道
        val blackLevel = dngRawData.blackLevel
        val preMul = dngRawData.preMul

        // 白电平
        val whiteLevel = dngRawData.whiteLevel

        // 白平衡增益：DngRawData 提供的是 [R, Gr, Gb, B]
        val whiteBalanceGains = dngRawData.whiteBalance

        // 色彩校正矩阵：DNG 提供的是 3x3 矩阵（行主序）
        val colorCorrectionMatrix = if (dngRawData.colorMatrix.size == 9) {
            dngRawData.colorMatrix
        } else {
            // 默认单位矩阵
            floatArrayOf(
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f
            )
        }

        val activeArray = if (dngRawData.activeArray != null && dngRawData.activeArray.size == 4) {
            Rect(
                dngRawData.activeArray[0],
                dngRawData.activeArray[1],
                dngRawData.activeArray[2],
                dngRawData.activeArray[3]
            )
        } else baseMetadata?.activeArray

        return RawMetadata(
            width = dngRawData.width,
            height = dngRawData.height,
            cfaPattern = cfaPattern,
            blackLevel = blackLevel,
            whiteLevel = whiteLevel,
            whiteBalanceGains = whiteBalanceGains,
            preMul = preMul,
            colorCorrectionMatrix = colorCorrectionMatrix,
            lensShadingMap = dngRawData.lensShadingMap,
            lensShadingMapWidth = dngRawData.lensShadingMapWidth,
            lensShadingMapHeight = dngRawData.lensShadingMapHeight,
            baselineExposure = if (dngRawData.baselineExposure == 0f) (baseMetadata?.baselineExposure
                ?: 0f) else dngRawData.baselineExposure,
            exposureBias = if (dngRawData.exposureBias == 0f) {
                if (baseMetadata != null && baseMetadata.exposureBias != 0f) baseMetadata.exposureBias else exposureBias
            } else dngRawData.exposureBias,
            iso = if (dngRawData.iso == 0) (baseMetadata?.iso ?: 100) else dngRawData.iso,
            shutterSpeed = if (dngRawData.shutterSpeed == 0L) (baseMetadata?.shutterSpeed
                ?: 0L) else dngRawData.shutterSpeed,
            aperture = if (dngRawData.aperture == 0f) (baseMetadata?.aperture ?: 0f) else dngRawData.aperture,
            activeArray = activeArray,
            noiseProfile = dngRawData.noiseProfile ?: baseMetadata?.noiseProfile ?: floatArrayOf(0f, 0f),
            postRawSensitivityBoost = baseMetadata?.postRawSensitivityBoost ?: 1.0f,
            exposureCompensation = baseMetadata?.exposureCompensation ?: 0f,
            aeMode = baseMetadata?.aeMode ?: 1,
            afRegions = baseMetadata?.afRegions,
            frameCount = baseMetadata?.frameCount ?: 1
        )
    }

    /**
     * Native 方法：使用 LibRaw 处理 DNG 文件
     */
    private external fun processDngNative(
        filePath: String,
        xr: Float, yr: Float,
        xg: Float, yg: Float,
        xb: Float, yb: Float,
        xw: Float, yw: Float,
        useRawAutoWhiteBalanceEstimate: Boolean
    ): DngRawData?

    companion object {
        private const val TAG = "RawDemosaicProcessor"
        private const val RAW_HDR_HIGHLIGHT_START = 0.72f
        private const val RAW_HDR_WHITE_POINT_SCENE_LUMA = 2.4f

        init {
            // 加载 JNI 库
            System.loadLibrary("my-native-lib")
        }

        @Volatile
        private var instance: RawDemosaicProcessor? = null

        fun getInstance(): RawDemosaicProcessor {
            return instance ?: synchronized(this) {
                instance ?: RawDemosaicProcessor().also { instance = it }
            }
        }
    }

    // 单线程调度器，确保所有 EGL 操作在同一线程
    private val glDispatcher = Executors.newSingleThreadExecutor { r ->
        Thread(r, "RawDemosaicProcessor-GL").apply { isDaemon = true }
    }.asCoroutineDispatcher()

    // EGL 资源
    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE

    // GL 资源
    private var combinedProgram = 0
    private var sharpenProgram = 0
    private var passthroughProgram = 0
    private var hdrReferenceProgram = 0

    private var rawTextureId = 0

    private var demosaicFramebufferId = 0
    private var demosaicTextureId = 0
    private var demosaicWidth = 0
    private var demosaicHeight = 0

    private var combinedFramebufferId = 0
    private var combinedTextureId = 0
    private var combinedWidth = 0
    private var combinedHeight = 0

    private var hdrReferenceFramebufferId = 0
    private var hdrReferenceTextureId = 0
    private var hdrReferenceWidth = 0
    private var hdrReferenceHeight = 0

    private var sharpenFramebufferId = 0
    private var sharpenTextureId = 0
    private var sharpenWidth = 0
    private var sharpenHeight = 0
    private var outputFramebufferId = 0
    private var outputTextureId = 0

    private var curveTextureId = 0
    private var dcpToneCurveTextureId = 0
    private var dcpHueSatTextureId = 0
    private var dcpLookTableTextureId = 0
    private var dummyDcp3DTextureId = 0
    private var dummyDcpToneCurveTextureId = 0

    // (Chroma) & NLM 降噪资源
    private var gfPass0Program = 0
    private var nlmPassHProgram = 0
    private var nlmPassVProgram = 0

    // NLM 中间纹理: ping-pong (RGBA16F)
    private var gfTexId = intArrayOf(0, 0)
    private var gfFboId = intArrayOf(0, 0)
    private var gfWidth = 0
    private var gfHeight = 0
    private var gfChromaTexId = 0

    suspend fun prewarmDepthEstimator(context: Context) = withContext(Dispatchers.Default) {
        val start = System.currentTimeMillis()
        SharedDepthEstimator.prewarm(context.applicationContext)
        PLog.d(TAG, "RAW DepthEstimator prewarmed, took=${System.currentTimeMillis() - start}ms")
    }

    private var gfChromaFboId = 0

    // 缓冲区
    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null
    private var indexBuffer: ShortBuffer? = null
    private var pboId = 0

    // Linear Program (New)
    private var linearProgram = 0

    private var lensShadingTextureId = 0
    private var dummyShadingTextureId = 0

    data class SceneStats(
        val exposureGain: Float,
        val curveLut: FloatArray? = null
    )

    private fun SceneStats.toRenderPlan(): RawRenderPlan {
        return RawRenderPlan(
            sceneNormalizationGain = exposureGain,
            sdrCurveLut = curveLut
        )
    }

    private fun resolveWorkingColorSpace(): android.graphics.ColorSpace =
        android.graphics.ColorSpace.get(android.graphics.ColorSpace.Named.SRGB)


    private var isInitialized = false
    private var maxTextureSize = 8192 // default, queried at init

    fun getRawColorSpace(): ColorSpace {
        return ColorSpace.ProPhoto
    }

    /**
     * 处理 DNG 文件
     *
     * @param dngFilePath DNG 文件路径
     * @param aspectRatio 目标宽高比
     * @param cropRegion 可选裁切区域（在 RAW 纹理空间）
     * @param sharpeningValue 锐化强度 (0.0-1.0)
     * @return 处理后的 Bitmap，失败返回 null
     */
    suspend fun process(
        context: Context,
        dngFilePath: String,
        aspectRatio: AspectRatio?,
        cropRegion: Rect?,
        rotation: Int,
        exposureBias: Float = 0f,
        rawExposureCompensation: Float = 0f,
        rawAutoExposure: Boolean = true,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        rawDROMode: DROMode = DROMode.fromPersistedName(null),
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
        onMetadata: ((RawMetadata) -> Unit)? = null
    ): Bitmap? = withContext(glDispatcher) {
        val dngFile = File(dngFilePath)
        if (!dngFile.exists() || !dngFile.canRead()) {
            PLog.e(TAG, "DNG file not found or not readable: $dngFilePath")
            return@withContext null
        }

        try {
            processInternal(
                context = context,
                aspectRatio = aspectRatio,
                cropRegion = cropRegion,
                rotation = rotation,
                exposureBias = exposureBias,
                rawExposureCompensation = rawExposureCompensation,
                rawAutoExposure = rawAutoExposure,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                rawAutoWhiteBalanceEstimate = rawAutoWhiteBalanceEstimate,
                rawDROMode = rawDROMode,
                sharpeningValue = sharpeningValue,
                denoiseValue = denoiseValue,
                rawDcpId = rawDcpId,
                dcpRenderPlan = dcpRenderPlan,
                dngFile = dngFile,
                onMetadata = onMetadata
            )?.sdrBitmap
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to process DNG file: $dngFilePath", e)
            null
        }
    }

    /**
     * 处理 RAW Buffer (例如来自 MultiFrameStacker 的输出)
     */
    suspend fun process(
        context: Context,
        rawData: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        metadata: RawMetadata,
        aspectRatio: AspectRatio,
        cropRegion: Rect?,
        rotation: Int,
        rawExposureCompensation: Float = 0f,
        rawAutoExposure: Boolean = true,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        rawDROMode: DROMode = DROMode.fromPersistedName(null),
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        chromaDenoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
    ): Bitmap? = withContext(glDispatcher) {
        try {
            if (!isInitialized) {
                if (!initializeOnGlThread()) {
                    PLog.e(TAG, "Failed to initialize processor")
                    return@withContext null
                }
            }

            processInternal(
                context = context,
                rawData = rawData,
                width = width,
                height = height,
                rowStride = rowStride,
                metadata = metadata,
                aspectRatio = aspectRatio,
                cropRegion = cropRegion,
                rotation = rotation,
                rawExposureCompensation = rawExposureCompensation,
                rawAutoExposure = rawAutoExposure,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                rawAutoWhiteBalanceEstimate = rawAutoWhiteBalanceEstimate,
                rawDROMode = rawDROMode,
                sharpeningValue = sharpeningValue,
                denoiseValue = denoiseValue,
                rawDcpId = rawDcpId,
                dcpRenderPlan = dcpRenderPlan
            )?.sdrBitmap
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to process RAW buffer", e)
            null
        }
    }

    suspend fun processForHdrSources(
        context: Context,
        dngFilePath: String,
        aspectRatio: AspectRatio?,
        cropRegion: Rect?,
        rotation: Int,
        exposureBias: Float = 0f,
        rawExposureCompensation: Float = 0f,
        rawAutoExposure: Boolean = true,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        rawDROMode: DROMode = DROMode.fromPersistedName(null),
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
        onMetadata: ((RawMetadata) -> Unit)? = null
    ): RawHdrRenderResult? = withContext(glDispatcher) {
        val dngFile = File(dngFilePath)
        if (!dngFile.exists() || !dngFile.canRead()) {
            PLog.e(TAG, "DNG file not found or not readable: $dngFilePath")
            return@withContext null
        }

        try {
            processInternal(
                context = context,
                aspectRatio = aspectRatio,
                cropRegion = cropRegion,
                rotation = rotation,
                exposureBias = exposureBias,
                rawExposureCompensation = rawExposureCompensation,
                rawAutoExposure = rawAutoExposure,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                rawAutoWhiteBalanceEstimate = rawAutoWhiteBalanceEstimate,
                rawDROMode = rawDROMode,
                sharpeningValue = sharpeningValue,
                denoiseValue = denoiseValue,
                rawDcpId = rawDcpId,
                dcpRenderPlan = dcpRenderPlan,
                dngFile = dngFile,
                onMetadata = onMetadata,
                includeHdrReference = true
            )
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to process RAW HDR sources: $dngFilePath", e)
            null
        }
    }

    /**
     * 内部处理方法（共享的核心处理逻辑）
     */
    private suspend fun processInternal(
        context: Context,
        rawData: ByteBuffer? = null,
        width: Int = 0,
        height: Int = 0,
        rowStride: Int = 0,
        metadata: RawMetadata? = null,
        aspectRatio: AspectRatio?,
        cropRegion: Rect?,
        rotation: Int,
        exposureBias: Float = 0f,
        rawExposureCompensation: Float = 0f,
        rawAutoExposure: Boolean = true,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        rawDROMode: DROMode = DROMode.fromPersistedName(null),
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
        dngFile: File? = null,
        onMetadata: ((RawMetadata) -> Unit)? = null,
        includeHdrReference: Boolean = false
    ): RawHdrRenderResult? = withContext(glDispatcher) {
        var actualRawData = rawData
        var actualWidth = width
        var actualHeight = height
        var actualRowStride = rowStride
        var actualMetadata = metadata
        var actualRotation = rotation
        var dngRawDataCleanup: DngRawData? = null

        if (dngFile != null) {
            val dngRawData = processDngNative(
                dngFile.absolutePath,
                ColorSpace.ProPhoto.xr, ColorSpace.ProPhoto.yr,
                ColorSpace.ProPhoto.xg, ColorSpace.ProPhoto.yg,
                ColorSpace.ProPhoto.xb, ColorSpace.ProPhoto.yb,
                ColorSpace.ProPhoto.xw, ColorSpace.ProPhoto.yw,
                rawAutoWhiteBalanceEstimate
            )
            if (dngRawData == null) {
                return@withContext RawProcessor.processAndToBitmap(dngFile, aspectRatio, cropRegion, rotation)?.let {
                    RawHdrRenderResult(
                        sdrBitmap = it,
                        hdrReferenceBitmap = null,
                    )
                }
            }
            dngRawDataCleanup = dngRawData
            actualRawData = dngRawData.rawData
            actualWidth = dngRawData.width
            actualHeight = dngRawData.height
            actualRowStride = dngRawData.rowStride
            actualMetadata = convertDngRawDataToMetadata(dngRawData, exposureBias, actualMetadata)
            actualRotation = dngRawData.rotation
            onMetadata?.invoke(actualMetadata)
        }

        if (actualRawData == null || actualMetadata == null) {
            PLog.e(TAG, "Missing source data or metadata")
            return@withContext null
        }

        val resolvedDcpRenderPlan = dcpRenderPlan ?: rawDcpId?.let { dcpId ->
            val dcpInfo = ContentRepository.getInstance(context).getAvailableDcps().firstOrNull { it.id == dcpId }
            if (dcpInfo == null) {
                PLog.w(TAG, "RAW DCP not found: $dcpId")
                null
            } else {
                DcpProfileParser.resolveRenderPlan(context, dcpInfo, actualMetadata).also { plan ->
                    if (plan == null) {
                        PLog.w(TAG, "Failed to resolve RAW DCP render plan: $dcpId")
                    } else {
                        PLog.d(TAG, "Resolved RAW DCP plan: ${plan.profileName}")
                    }
                }
            }
        }

        PLog.d(TAG, "Processing RAW image: ${actualWidth}x${actualHeight}")

        try {
            if (!isInitialized) {
                if (!initializeOnGlThread()) {
                    PLog.e(TAG, "Failed to initialize processor")
                    return@withContext null
                }
            }

            // Check GL_MAX_TEXTURE_SIZE and downscale if needed
            var downscaledBuffer: ByteBuffer? = null
            if (actualWidth > maxTextureSize || actualHeight > maxTextureSize) {
                PLog.w(
                    TAG,
                    "Input ${actualWidth}x${actualHeight} exceeds GL_MAX_TEXTURE_SIZE=$maxTextureSize, downscaling"
                )
                val scaleX = maxTextureSize.toFloat() / actualWidth
                val scaleY = maxTextureSize.toFloat() / actualHeight
                val scaleFactor = minOf(scaleX, scaleY, 1f)
                val newWidth = (actualWidth * scaleFactor).toInt() and 0xFFFFFFFE.toInt() // align to even
                val newHeight = (actualHeight * scaleFactor).toInt() and 0xFFFFFFFE.toInt()

                val isLinearRGB = actualMetadata.cfaPattern == RawMetadata.CFA_LINEAR_RGB
                if (isLinearRGB) {
                    // CPU bilinear downscale for interleaved RGB16 buffer
                    val srcBuf = actualRawData.duplicate().order(java.nio.ByteOrder.nativeOrder())
                    srcBuf.position(0)
                    val src = srcBuf.asShortBuffer()
                    val dstSize = newWidth * newHeight * 3 * 2 // 3 channels * 2 bytes
                    val dstByteBuf = com.hinnka.mycamera.utils.DirectBufferAllocator.allocateNative(dstSize.toLong())
                        ?.order(java.nio.ByteOrder.nativeOrder())
                        ?: throw OutOfMemoryError("Failed to allocate native direct buffer")
                    val dst = dstByteBuf.asShortBuffer()

                    val srcChannels = 3
                    for (dy in 0 until newHeight) {
                        val sy = dy.toFloat() / newHeight * actualHeight
                        val sy0 = sy.toInt().coerceIn(0, actualHeight - 1)
                        val sy1 = (sy0 + 1).coerceIn(0, actualHeight - 1)
                        val fy = sy - sy0
                        for (dx in 0 until newWidth) {
                            val sx = dx.toFloat() / newWidth * actualWidth
                            val sx0 = sx.toInt().coerceIn(0, actualWidth - 1)
                            val sx1 = (sx0 + 1).coerceIn(0, actualWidth - 1)
                            val fx = sx - sx0
                            for (c in 0 until srcChannels) {
                                val v00 =
                                    (src.get((sy0 * actualWidth + sx0) * srcChannels + c).toInt() and 0xFFFF).toFloat()
                                val v10 =
                                    (src.get((sy0 * actualWidth + sx1) * srcChannels + c).toInt() and 0xFFFF).toFloat()
                                val v01 =
                                    (src.get((sy1 * actualWidth + sx0) * srcChannels + c).toInt() and 0xFFFF).toFloat()
                                val v11 =
                                    (src.get((sy1 * actualWidth + sx1) * srcChannels + c).toInt() and 0xFFFF).toFloat()
                                val v =
                                    v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy
                                dst.put((dy * newWidth + dx) * srcChannels + c, v.toInt().coerceIn(0, 65535).toShort())
                            }
                        }
                    }
                    actualRawData = dstByteBuf
                    downscaledBuffer = dstByteBuf
                    actualRowStride = newWidth * 6
                    PLog.d(TAG, "Downscaled to ${newWidth}x${newHeight}")
                }
                actualWidth = newWidth
                actualHeight = newHeight
                // Update metadata dimensions
                actualMetadata = actualMetadata.copy(width = newWidth, height = newHeight)
            }

            val bounds =
                BitmapUtils.calculateProcessedRect(actualWidth, actualHeight, aspectRatio, cropRegion, actualRotation)
            val finalWidth = bounds.width()
            val finalHeight = bounds.height()

            // 4. 第一步：全分辨率处理 (Linear CCM)
            setupFullResFramebuffer(actualWidth, actualHeight)
            uploadRawTextureFromBuffer(
                actualRawData,
                actualWidth,
                actualHeight,
                actualRowStride,
                RawMetadata.CFA_LINEAR_RGB
            )
            // GPU 已消费 rawData，立即释放 CPU 侧引用，帮助 GC 回收（超分时约 288 MB）
            actualRawData = null
            downscaledBuffer?.let {
                com.hinnka.mycamera.utils.DirectBufferAllocator.freeNative(it)
            }
            renderLinearPass(
                metadata = actualMetadata,
                rawExposureCompensation = rawExposureCompensation,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                dcpRenderPlan = resolvedDcpRenderPlan
            )
            var shadowLift = 0f
            var effectiveExposureCompensation = rawExposureCompensation
            if (rawAutoExposure) {
                val meteringResult = resolveRawAutoExposureEv(
                    context = context,
                    metadata = actualMetadata,
                    sourceTextureId = demosaicTextureId,
                    rawDROMode = rawDROMode,
                    dcpRenderPlan = resolvedDcpRenderPlan
                )
                val autoExposureEv = meteringResult.meteredEv
                if (rawDROMode.isEnabled && autoExposureEv > 0f && meteringResult.dynamicRangeGap > 1.2f) {
                    val maxLinearGainEv = ln(1.0f / meteringResult.p998.coerceAtLeast(0.01f)) / ln(2.0f)
                    val linearEv = autoExposureEv.coerceAtMost(maxLinearGainEv).coerceAtLeast(0f)
                    val remainingEv = autoExposureEv - linearEv
                    shadowLift = ((2.0f.pow(remainingEv) - 1.0f) / 0.67f).coerceIn(0f, 5.0f)

                    PLog.d(TAG, "Raw DRO ${rawDROMode.name} linearEv=$linearEv shadowLift=$shadowLift")

                    effectiveExposureCompensation += linearEv
                } else {
                    effectiveExposureCompensation += autoExposureEv
                }

                if (effectiveExposureCompensation != rawExposureCompensation) {
                    PLog.d(TAG, "final auto exposure compensation = $effectiveExposureCompensation")
                    renderLinearPass(
                        metadata = actualMetadata,
                        rawExposureCompensation = effectiveExposureCompensation,
                        rawBlackPointCorrection = rawBlackPointCorrection,
                        rawWhitePointCorrection = rawWhitePointCorrection,
                        dcpRenderPlan = resolvedDcpRenderPlan
                    )
                }
            }
            // rawTextureId 已被 linearPass 消费，提前释放 GPU 显存
            if (rawTextureId != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(rawTextureId), 0)
                rawTextureId = 0
            }
            val workingColorSpace = resolveWorkingColorSpace()

            // NLM 降噪
            renderNLMPass(
                sourceTextureId = demosaicTextureId,
                width = actualWidth,
                height = actualHeight,
                metadata = actualMetadata,
                denoiseValue = denoiseValue,
                chromaDenoiseValue = denoiseValue,
            )
            // demosaicTextureId / gfChromaTexId / gfTexId[0] 已被 NLM Pass 消费，提前释放
            if (demosaicTextureId != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(demosaicTextureId), 0)
                demosaicTextureId = 0
            }
            if (demosaicFramebufferId != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(demosaicFramebufferId), 0)
                demosaicFramebufferId = 0
            }
            demosaicWidth = 0; demosaicHeight = 0
            if (gfChromaTexId != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(gfChromaTexId), 0)
                gfChromaTexId = 0
            }
            if (gfChromaFboId != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(gfChromaFboId), 0)
                gfChromaFboId = 0
            }
            if (gfTexId[0] != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(gfTexId[0]), 0)
                gfTexId[0] = 0
            }
            if (gfFboId[0] != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(gfFboId[0]), 0)
                gfFboId[0] = 0
            }

            val outputTexture = gfTexId[1]

            val hdrReferenceBitmap = if (includeHdrReference) {
                setupHdrReferenceFramebuffer(actualWidth, actualHeight)
                renderHdrReferencePass(
                    metadata = actualMetadata,
                    inputTextureId = outputTexture
                )
                setupOutputFramebuffer(finalWidth, finalHeight)
                renderOutputPass(
                    actualRotation,
                    actualWidth,
                    actualHeight,
                    bounds,
                    hdrReferenceTextureId
                )
                val hdrPixels = readPixels(
                    finalWidth,
                    finalHeight,
                    android.graphics.ColorSpace.get(android.graphics.ColorSpace.Named.EXTENDED_SRGB)
                )
                // hdrReferenceTextureId 已被 outputPass 消费
                if (hdrReferenceTextureId != 0) {
                    GLES30.glDeleteTextures(1, intArrayOf(hdrReferenceTextureId), 0)
                    hdrReferenceTextureId = 0
                }
                if (hdrReferenceFramebufferId != 0) {
                    GLES30.glDeleteFramebuffers(1, intArrayOf(hdrReferenceFramebufferId), 0)
                    hdrReferenceFramebufferId = 0
                }
                hdrReferenceWidth = 0; hdrReferenceHeight = 0
                hdrPixels
            } else {
                null
            }

            // 5. 第二步：Combined Pass (HDR Linear -> LDR sRGB + LUT)
            setupCombinedFramebuffer(actualWidth, actualHeight)
            val combinedStart = System.currentTimeMillis()
            renderCombinedPass(
                metadata = actualMetadata,
                inputTextureId = outputTexture,
                dcpRenderPlan = resolvedDcpRenderPlan,
                shadowLift = shadowLift
            )
            PLog.d(TAG, "Combined Pass took: ${System.currentTimeMillis() - combinedStart}ms")
            // outputTexture (gfTexId[1]) 已被 combinedPass 消费，提前释放
            if (gfTexId[1] != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(gfTexId[1]), 0)
                gfTexId[1] = 0
            }
            if (gfFboId[1] != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(gfFboId[1]), 0)
                gfFboId[1] = 0
            }
            gfWidth = 0; gfHeight = 0

            // 6. 第三步：锐化 (Sharpen Pass)
            setupSharpenFramebuffer(actualWidth, actualHeight)
            val sharpenStart = System.currentTimeMillis()
            renderSharpenPass(actualMetadata, sharpeningValue, combinedTextureId)
            PLog.d(TAG, "Sharpen Pass took: ${System.currentTimeMillis() - sharpenStart}ms")
            // combinedTextureId 已被 sharpenPass 消费，提前释放
            if (combinedTextureId != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(combinedTextureId), 0)
                combinedTextureId = 0
            }
            if (combinedFramebufferId != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(combinedFramebufferId), 0)
                combinedFramebufferId = 0
            }
            combinedWidth = 0; combinedHeight = 0

            val sourceTextureForOutput = sharpenTextureId

            // 7. 第四步：输出旋转 (Output Pass)
            setupOutputFramebuffer(finalWidth, finalHeight)
            val outputStart = System.currentTimeMillis()
            renderOutputPass(
                actualRotation,
                actualWidth,
                actualHeight,
                bounds,
                sourceTextureForOutput
            )
            PLog.d(TAG, "Output Pass took: ${System.currentTimeMillis() - outputStart}ms")
            // sharpenTextureId 已被 outputPass 消费，在 readPixels 前释放以降低峰值内存
            if (sharpenTextureId != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(sharpenTextureId), 0)
                sharpenTextureId = 0
            }
            if (sharpenFramebufferId != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(sharpenFramebufferId), 0)
                sharpenFramebufferId = 0
            }
            sharpenWidth = 0; sharpenHeight = 0

            // 8. 读取结果
            val readStart = System.currentTimeMillis()
            val finalBitmap = readPixels(finalWidth, finalHeight, workingColorSpace)
            PLog.d(TAG, "readPixels took: ${System.currentTimeMillis() - readStart}ms")

            PLog.d(TAG, "RAW processing complete: ${finalBitmap?.width}x${finalBitmap?.height}")
            finalBitmap?.let {
                RawHdrRenderResult(
                    sdrBitmap = it,
                    hdrReferenceBitmap = hdrReferenceBitmap,
                )
            }
        } finally {
            dngRawDataCleanup?.close()
        }
    }

    private suspend fun initializeOnGlThread(): Boolean = withContext(glDispatcher) {
        initialize()
    }

    /**
     * 初始化 EGL 环境
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        try {
            // 获取 EGL Display
            eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
                PLog.e(TAG, "Unable to get EGL display")
                return false
            }

            // 初始化 EGL
            val version = IntArray(2)
            if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
                PLog.e(TAG, "Unable to initialize EGL")
                return false
            }

            // 配置属性
            val configAttribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
            )

            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)) {
                PLog.e(TAG, "Unable to choose EGL config")
                return false
            }

            val config = configs[0] ?: return false

            // 创建 EGL Context (ES 3.0)
            val contextAttribs = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
            )
            eglContext = EGL14.eglCreateContext(eglDisplay, config, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
            if (eglContext == EGL14.EGL_NO_CONTEXT) {
                PLog.e(TAG, "Unable to create EGL context")
                return false
            }

            // 创建 PBuffer Surface（1x1 占位，实际使用 FBO）
            val surfaceAttribs = intArrayOf(
                EGL14.EGL_WIDTH, 1,
                EGL14.EGL_HEIGHT, 1,
                EGL14.EGL_NONE
            )
            eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, config, surfaceAttribs, 0)
            if (eglSurface == EGL14.EGL_NO_SURFACE) {
                PLog.e(TAG, "Unable to create EGL surface")
                return false
            }

            // 激活上下文
            if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                PLog.e(TAG, "Unable to make EGL current")
                return false
            }

            // 初始化着色器和缓冲区
            initShaderProgram()
            initBuffers()

            // 创建静默遮挡图
            dummyShadingTextureId = createDummyShadingTexture()

            // Query hardware texture size limit
            val maxTexSizeArr = IntArray(1)
            GLES30.glGetIntegerv(GLES30.GL_MAX_TEXTURE_SIZE, maxTexSizeArr, 0)
            maxTextureSize = maxTexSizeArr[0]
            PLog.d(TAG, "GL_MAX_TEXTURE_SIZE = $maxTextureSize")

            isInitialized = true
            PLog.d(TAG, "RawDemosaicProcessor initialized")
            return true

        } catch (e: Exception) {
            PLog.e(TAG, "Failed to initialize", e)
            return false
        }
    }

    private fun initShaderProgram() {
        val vShader = compileShader(GLES30.GL_VERTEX_SHADER, RawShaders.VERTEX_SHADER)

        // 1. DHT Multi-Pass Programs (替代旧的单 pass AHD)
        // initDhtPrograms(vShader)

        // 1.5 Linear Program (For Stacked RGB, 保留)
        val fShaderLinear = compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.FRAGMENT_SHADER_LINEAR)
        if (vShader != 0 && fShaderLinear != 0) {
            linearProgram = GLES30.glCreateProgram()
            GLES30.glAttachShader(linearProgram, vShader)
            GLES30.glAttachShader(linearProgram, fShaderLinear)
            GLES30.glLinkProgram(linearProgram)
            if (!logProgramLinkResult(linearProgram, "linearProgram")) {
                linearProgram = 0
            }

            GLES30.glDeleteShader(fShaderLinear)
        }

        // 2. Combined Processing Program
        val fShaderCombined = compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.COMBINED_FRAGMENT_SHADER)
        if (vShader != 0 && fShaderCombined != 0) {
            combinedProgram = GLES30.glCreateProgram()
            GLES30.glAttachShader(combinedProgram, vShader)
            GLES30.glAttachShader(combinedProgram, fShaderCombined)
            GLES30.glLinkProgram(combinedProgram)
            if (!logProgramLinkResult(combinedProgram, "combinedProgram")) {
                combinedProgram = 0
            }

            GLES30.glDeleteShader(fShaderCombined)
        }

        val fShaderHdrReference = compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.HDR_REFERENCE_FRAGMENT_SHADER)
        if (vShader != 0 && fShaderHdrReference != 0) {
            hdrReferenceProgram = GLES30.glCreateProgram()
            GLES30.glAttachShader(hdrReferenceProgram, vShader)
            GLES30.glAttachShader(hdrReferenceProgram, fShaderHdrReference)
            GLES30.glLinkProgram(hdrReferenceProgram)
            if (!logProgramLinkResult(hdrReferenceProgram, "hdrReferenceProgram")) {
                hdrReferenceProgram = 0
            }

            GLES30.glDeleteShader(fShaderHdrReference)
        }

        // 2.2 Sharpen Program
        val fShaderSharpen = compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.SHARPEN_FRAGMENT_SHADER)
        if (vShader != 0 && fShaderSharpen != 0) {
            sharpenProgram = GLES30.glCreateProgram()
            GLES30.glAttachShader(sharpenProgram, vShader)
            GLES30.glAttachShader(sharpenProgram, fShaderSharpen)
            GLES30.glLinkProgram(sharpenProgram)
            if (!logProgramLinkResult(sharpenProgram, "sharpenProgram")) {
                sharpenProgram = 0
            }

            GLES30.glDeleteShader(fShaderSharpen)
        }

        // 2.7 NLM Programs
        initNLMPrograms(vShader)

        // 3. Passthrough Program
        val fShaderPass = compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.PASSTHROUGH_FRAGMENT_SHADER)
        if (vShader != 0 && fShaderPass != 0) {
            passthroughProgram = GLES30.glCreateProgram()
            GLES30.glAttachShader(passthroughProgram, vShader)
            GLES30.glAttachShader(passthroughProgram, fShaderPass)
            GLES30.glLinkProgram(passthroughProgram)
            if (!logProgramLinkResult(passthroughProgram, "passthroughProgram")) {
                passthroughProgram = 0
            }

            GLES30.glDeleteShader(fShaderPass)
        }

        GLES30.glDeleteShader(vShader)
        PLog.d(
            TAG,
            "Shader programs created: combined=$combinedProgram, passthrough=$passthroughProgram"
        )
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)

        val compiled = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val error = GLES30.glGetShaderInfoLog(shader)
            PLog.e(TAG, "Shader compilation failed: $error")
            GLES30.glDeleteShader(shader)
            return 0
        }
        return shader
    }

    private fun initBuffers() {
        vertexBuffer = ByteBuffer.allocateDirect(RawShaders.FULL_QUAD_VERTICES.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(RawShaders.FULL_QUAD_VERTICES)
        vertexBuffer?.position(0)

        texCoordBuffer = ByteBuffer.allocateDirect(RawShaders.TEXTURE_COORDS.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(RawShaders.TEXTURE_COORDS)
        texCoordBuffer?.position(0)

        indexBuffer = ByteBuffer.allocateDirect(RawShaders.DRAW_ORDER.size * 2)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
            .put(RawShaders.DRAW_ORDER)
        indexBuffer?.position(0)
    }

    /**
     * 初始化 Guided Filter Pass 0 和 NLM 着色器
     */
    private fun initNLMPrograms(vShader: Int) {
        fun createGfProgram(vShader: Int, fSource: String, name: String): Int {
            val fShader = compileShader(GLES30.GL_FRAGMENT_SHADER, fSource)
            if (vShader == 0 || fShader == 0) return 0
            val program = GLES30.glCreateProgram()
            GLES30.glAttachShader(program, vShader)
            GLES30.glAttachShader(program, fShader)
            GLES30.glLinkProgram(program)
            if (!logProgramLinkResult(program, name)) {
                GLES30.glDeleteShader(fShader)
                return 0
            }
            GLES30.glDeleteShader(fShader)
            return program
        }

        gfPass0Program = createGfProgram(vShader, BM3DShaders.PASS0_CHROMA_DENOISE, "BM3D_Pass0")
        nlmPassHProgram = createGfProgram(vShader, BM3DShaders.PASS1_BASIC_ESTIMATE, "BM3D_Pass1")
        nlmPassVProgram = createGfProgram(vShader, BM3DShaders.PASS2_WIENER, "BM3D_Pass2")

        PLog.d(
            TAG,
            "Denoise programs: BM3D_Pass0=$gfPass0Program BM3D_Pass1=$nlmPassHProgram BM3D_Pass2=$nlmPassVProgram"
        )
    }

    private fun setupNLMFramebuffers(width: Int, height: Int) {
        if (gfWidth == width && gfHeight == height && gfTexId[0] != 0) return
        gfWidth = width
        gfHeight = height

        // 清理旧资源
        for (i in 0..1) {
            if (gfTexId[i] != 0) GLES30.glDeleteTextures(1, intArrayOf(gfTexId[i]), 0)
            if (gfFboId[i] != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(gfFboId[i]), 0)
        }
        if (gfChromaTexId != 0) GLES30.glDeleteTextures(1, intArrayOf(gfChromaTexId), 0)
        if (gfChromaFboId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(gfChromaFboId), 0)

        // 创建独立纹理用于色度降噪结果
        val ct = IntArray(1)
        val cf = IntArray(1)
        GLES30.glGenTextures(1, ct, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, ct[0])
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RGBA16F,
            width,
            height,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_HALF_FLOAT,
            null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glGenFramebuffers(1, cf, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, cf[0])
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            ct[0],
            0
        )
        gfChromaTexId = ct[0]; gfChromaFboId = cf[0]

        // 创建双缓冲 (RGBA16F) 用于中间 pass

        for (i in 0..1) {
            val t = IntArray(1)
            val f = IntArray(1)
            GLES30.glGenTextures(1, t, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0])
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, width, height, 0,
                GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glGenFramebuffers(1, f, 0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, f[0])
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER,
                GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D,
                t[0],
                0
            )
            gfTexId[i] = t[0]; gfFboId[i] = f[0]
        }
        checkGlError("setupGuidedFilterFramebuffers")
    }


    /**
     * 渲染 BM3D 降噪
     *
     * 管线: tonemapTexture → [Pass0 Chroma] → [BM3D Basic] → [BM3D Wiener] → gfFboId[1]
     *
     * @param sourceTextureId 输入纹理 (ToneMap 输出)
     */
    private fun renderNLMPass(
        sourceTextureId: Int,
        width: Int,
        height: Int,
        metadata: RawMetadata,
        denoiseValue: Float?,
        chromaDenoiseValue: Float?,
    ) {
        setupNLMFramebuffers(width, height)

        if (gfPass0Program == 0 || nlmPassHProgram == 0 || nlmPassVProgram == 0) {
            PLog.w(TAG, "Denoise programs not initialized, skipping denoise")
            return
        }

        val texelW = 1.0f / width
        val texelH = 1.0f / height

        // 动态 NLM 参数计算
        // 增益由三部分组成：传感器 ISO、ISP 数字增益、后期 Tonemap 增益
        val isoGain = metadata.iso / 100.0f
        val digitalGain = metadata.postRawSensitivityBoost
        val postGain = ExposureNormalization.compute(metadata)
        val totalGain = (isoGain * digitalGain * postGain).coerceAtLeast(0f)

        var (s, o) = resolveDenoiseNoiseModel(metadata, totalGain)

        if (s <= 0 || o <= 0) {
            s = 1E-4f * totalGain
            o = 4.5E-7f * sqrt(totalGain)
        }

        val denoiseValue = denoiseValue ?: 0f
        val chromaDenoiseValue = chromaDenoiseValue ?: 0f

        val h = denoiseValue
        val ch = chromaDenoiseValue + 0.5f
        PLog.d(TAG, "Dynamic BM3D: s=$s o=$o, h=$h ch=$ch iso=${metadata.iso}")

        val identityMatrix = FloatArray(16)
        GlMatrix.setIdentityM(identityMatrix, 0)

        // ===== Pass 0: 色度降噪 (输出到 gfChromaFboId) =====
        GLES30.glUseProgram(gfPass0Program)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, gfChromaFboId)
        GLES30.glViewport(0, 0, width, height)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(gfPass0Program, "uInputTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(gfPass0Program, "uTexelSize"), texelW, texelH)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(gfPass0Program, "uTexMatrix"),
            1, false, identityMatrix, 0
        )
        GLES30.glUniform1f(GLES30.glGetUniformLocation(gfPass0Program, "uH"), ch)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(gfPass0Program, "uNoiseModel"), s, o)
        drawQuad(gfPass0Program)

        // ===== BM3D Pass 1: Basic Estimate (gfChromaTexId -> gfFboId[0]) =====
        GLES30.glUseProgram(nlmPassHProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, gfFboId[0])
        GLES30.glViewport(0, 0, width, height)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, gfChromaTexId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(nlmPassHProgram, "uInputTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(nlmPassHProgram, "uTexelSize"), texelW, texelH)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(nlmPassHProgram, "uTexMatrix"),
            1, false, identityMatrix, 0
        )
        GLES30.glUniform1f(GLES30.glGetUniformLocation(nlmPassHProgram, "uH"), h)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(nlmPassHProgram, "uNoiseModel"), s, o)
        drawQuad(nlmPassHProgram)

        // ===== BM3D Pass 2: Wiener Refinement (gfChromaTexId + gfTexId[0] -> gfFboId[1]) =====
        GLES30.glUseProgram(nlmPassVProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, gfFboId[1])
        GLES30.glViewport(0, 0, width, height)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, gfChromaTexId) // Original noisy (chroma-denoised)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(nlmPassVProgram, "uInputTexture"), 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, gfTexId[0])    // Pass-1 basic estimate
        GLES30.glUniform1i(GLES30.glGetUniformLocation(nlmPassVProgram, "uBasicTexture"), 1)

        GLES30.glUniform2f(GLES30.glGetUniformLocation(nlmPassVProgram, "uTexelSize"), texelW, texelH)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(nlmPassVProgram, "uTexMatrix"),
            1, false, identityMatrix, 0
        )
        GLES30.glUniform1f(GLES30.glGetUniformLocation(nlmPassVProgram, "uH"), h)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(nlmPassVProgram, "uNoiseModel"), s, o)
        drawQuad(nlmPassVProgram)

        checkGlError("renderNLMDenoise")
    }

    private fun resolveDenoiseNoiseModel(metadata: RawMetadata, totalGain: Float): Pair<Float, Float> {
        var s = metadata.noiseProfile.getOrElse(0) { 0f }
        var o = metadata.noiseProfile.getOrElse(1) { 0f }

        if (s <= 0f || o <= 0f) {
            s = 1E-4f * totalGain
            o = 4.5E-7f * sqrt(totalGain)
        }

        return s.coerceAtLeast(1e-10f) to o.coerceAtLeast(1e-10f)
    }

    private fun dhtSetCommonUniforms(program: Int, metadata: RawMetadata) {
        val loc = GLES30.glGetUniformLocation(program, "uImageSize")
        if (loc >= 0) GLES30.glUniform2f(loc, metadata.width.toFloat(), metadata.height.toFloat())
        val cfaLoc = GLES30.glGetUniformLocation(program, "uCfaPattern")
        if (cfaLoc >= 0) GLES30.glUniform1i(cfaLoc, metadata.cfaPattern)
        val tmLoc = GLES30.glGetUniformLocation(program, "uTexMatrix")
        if (tmLoc >= 0) {
            val id = FloatArray(16); GlMatrix.setIdentityM(id, 0)
            GLES30.glUniformMatrix4fv(tmLoc, 1, false, id, 0)
        }
    }

    /**
     * 从 ByteBuffer 上传 RAW 数据到纹理
     */
    private fun uploadRawTextureFromBuffer(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        cfaPattern: Int
    ) {
        if (rawTextureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            rawTextureId = textures[0]
        }

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, rawTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        // 确保 buffer 位置从 0 开始
        buffer.position(0)

        // 关键优化：使用 GL_UNPACK_ROW_LENGTH 处理 padding
        val isLinearRGB = cfaPattern == RawMetadata.CFA_LINEAR_RGB
        val bytesPerPixel = if (isLinearRGB) 6 else 2 // 16-bit (x3 for RGB)
        val rowLength = rowStride / bytesPerPixel

        val internalFormat = if (isLinearRGB) GLES30.GL_RGB16UI else GLES30.GL_R16UI
        val format = if (isLinearRGB) GLES30.GL_RGB_INTEGER else GLES30.GL_RED_INTEGER

        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 2)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, rowLength)

        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            internalFormat,
            width,
            height,
            0,
            format,
            GLES30.GL_UNSIGNED_SHORT,
            buffer
        )

        // 恢复默认设置
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)

        checkGlError("uploadRawTextureFromBuffer")
    }

    /**
     * 上传 RAW 数据到纹理（从 Image 对象）
     *
     * RAW_SENSOR 格式通常是 16 位（或 10/12 位打包为 16 位）的单通道数据
     */
    private fun uploadRawTexture(image: Image, width: Int, height: Int, rowStride: Int) {
        if (rawTextureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            rawTextureId = textures[0]
        }

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, rawTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        // 获取 RAW 数据
        val plane = image.planes[0]
        val buffer = plane.buffer
        buffer.position(0)

        // 关键优化：使用 GL_UNPACK_ROW_LENGTH 处理 padding，避免 CPU 逐行复制
        val bytesPerPixel = 2 // 16-bit
        val rowLength = rowStride / bytesPerPixel

        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 2)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, rowLength)

        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_R16UI,
            width,
            height,
            0,
            GLES30.GL_RED_INTEGER,
            GLES30.GL_UNSIGNED_SHORT,
            buffer
        )

        // 恢复默认设置
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)

        checkGlError("uploadRawTexture")
    }

    private fun uploadLensShadingTexture(metadata: RawMetadata) {
        if (metadata.lensShadingMap == null) return

        if (lensShadingTextureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            lensShadingTextureId = textures[0]
        }

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, lensShadingTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        val buffer = ByteBuffer.allocateDirect(metadata.lensShadingMap.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(metadata.lensShadingMap)
        buffer.position(0)

        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32F,
            metadata.lensShadingMapWidth, metadata.lensShadingMapHeight, 0,
            GLES30.GL_RGBA, GLES30.GL_FLOAT, buffer
        )
    }

    private fun createDummyShadingTexture(): Int {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)

        val buffer = ByteBuffer.allocateDirect(4 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        buffer.put(floatArrayOf(1f, 1f, 1f, 1f))
        buffer.position(0)

        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32F,
            1, 1, 0, GLES30.GL_RGBA, GLES30.GL_FLOAT, buffer
        )
        return textures[0]
    }

    private fun setupFullResFramebuffer(width: Int, height: Int) {
        if (demosaicFramebufferId != 0 && demosaicTextureId != 0) {
            // Check if size matches, if not, recreate
            if (demosaicWidth == width && demosaicHeight == height) {
                return
            }
            // Size mismatch, destroy and recreate
            GLES30.glDeleteTextures(1, intArrayOf(demosaicTextureId), 0)
            GLES30.glDeleteFramebuffers(1, intArrayOf(demosaicFramebufferId), 0)
        }

        demosaicWidth = width
        demosaicHeight = height

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        demosaicTextureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, demosaicTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RGBA16F,
            width,
            height,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_HALF_FLOAT,
            null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        val fbos = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        demosaicFramebufferId = fbos[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, demosaicFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            demosaicTextureId,
            0
        )
        checkGlError("setupFullResFramebuffer")
    }

    private fun setupCombinedFramebuffer(width: Int, height: Int) {
        if (combinedWidth == width && combinedHeight == height && combinedFramebufferId != 0) {
            return
        }

        if (combinedTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(combinedTextureId), 0)
        }
        if (combinedFramebufferId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(combinedFramebufferId), 0)
        }

        combinedWidth = width
        combinedHeight = height

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        combinedTextureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, combinedTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, width, height, 0,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        val framebuffers = IntArray(1)
        GLES30.glGenFramebuffers(1, framebuffers, 0)
        combinedFramebufferId = framebuffers[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, combinedFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            combinedTextureId,
            0
        )
        checkGlError("setupCombinedFramebuffer")
    }

    private fun setupHdrReferenceFramebuffer(width: Int, height: Int) {
        if (hdrReferenceWidth == width && hdrReferenceHeight == height && hdrReferenceFramebufferId != 0) {
            return
        }

        if (hdrReferenceTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(hdrReferenceTextureId), 0)
        }
        if (hdrReferenceFramebufferId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(hdrReferenceFramebufferId), 0)
        }

        hdrReferenceWidth = width
        hdrReferenceHeight = height

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        hdrReferenceTextureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, hdrReferenceTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, width, height, 0,
            GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        val framebuffers = IntArray(1)
        GLES30.glGenFramebuffers(1, framebuffers, 0)
        hdrReferenceFramebufferId = framebuffers[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, hdrReferenceFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            hdrReferenceTextureId,
            0
        )
        checkGlError("setupHdrReferenceFramebuffer")
    }

    private fun setupSharpenFramebuffer(width: Int, height: Int) {
        if (sharpenWidth == width && sharpenHeight == height && sharpenFramebufferId != 0) {
            return
        }

        if (sharpenTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(sharpenTextureId), 0)
        }
        if (sharpenFramebufferId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(sharpenFramebufferId), 0)
        }

        sharpenWidth = width
        sharpenHeight = height

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        sharpenTextureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sharpenTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, width, height, 0,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        val framebuffers = IntArray(1)
        GLES30.glGenFramebuffers(1, framebuffers, 0)
        sharpenFramebufferId = framebuffers[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, sharpenFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            sharpenTextureId,
            0
        )
        checkGlError("setupSharpenFramebuffer")
    }

    private fun setupOutputFramebuffer(width: Int, height: Int) {
        if (outputFramebufferId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(outputFramebufferId), 0)
            GLES30.glDeleteTextures(1, intArrayOf(outputTextureId), 0)
        }

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        outputTextureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, outputTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RGBA16F,
            width,
            height,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_HALF_FLOAT,
            null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        val fbos = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        outputFramebufferId = fbos[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, outputFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            outputTextureId,
            0
        )
        checkGlError("setupOutputFramebuffer")
    }

    // 辅助函数: 3x3 矩阵转置 (行主序 -> 列主序)
    private fun transposeMatrix3x3(matrix: FloatArray): FloatArray {
        require(matrix.size >= 9) { "Matrix must have at least 9 elements" }
        return floatArrayOf(
            matrix[0], matrix[3], matrix[6],
            matrix[1], matrix[4], matrix[7],
            matrix[2], matrix[5], matrix[8]
        )
    }

    private fun uploadCurveTexture(curveLut: FloatArray) {
        if (curveTextureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            curveTextureId = textures[0]
        }

        val buffer = ByteBuffer.allocateDirect(curveLut.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(curveLut)
        buffer.position(0)

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, curveTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R16F,
            curveLut.size, 1, 0, GLES30.GL_RED, GLES30.GL_FLOAT, buffer
        )
    }

    private fun uploadDcpToneCurveTexture(curveLut: FloatArray) {
        if (dcpToneCurveTextureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            dcpToneCurveTextureId = textures[0]
        }

        val buffer = ByteBuffer.allocateDirect(curveLut.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(curveLut)
        buffer.position(0)

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, dcpToneCurveTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R16F,
            curveLut.size, 1, 0, GLES30.GL_RED, GLES30.GL_FLOAT, buffer
        )
        checkGlError("uploadDcpToneCurveTexture")
    }

    private fun ensureDummyDcp3DTexture(): Int {
        if (dummyDcp3DTextureId != 0) return dummyDcp3DTextureId
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        dummyDcp3DTextureId = textures[0]
        val buffer = ByteBuffer.allocateDirect(4 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(floatArrayOf(0f, 1f, 1f, 1f))
        buffer.position(0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, dummyDcp3DTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_WRAP_R, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexImage3D(
            GLES30.GL_TEXTURE_3D,
            0,
            GLES30.GL_RGBA16F,
            1,
            1,
            1,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_FLOAT,
            buffer
        )
        checkGlError("ensureDummyDcp3DTexture")
        return dummyDcp3DTextureId
    }

    private fun ensureDummyDcpToneCurveTexture(): Int {
        if (dummyDcpToneCurveTextureId != 0) return dummyDcpToneCurveTextureId
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        dummyDcpToneCurveTextureId = textures[0]
        val buffer = ByteBuffer.allocateDirect(4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(floatArrayOf(0f))
        buffer.position(0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, dummyDcpToneCurveTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_R16F,
            1,
            1,
            0,
            GLES30.GL_RED,
            GLES30.GL_FLOAT,
            buffer
        )
        checkGlError("ensureDummyDcpToneCurveTexture")
        return dummyDcpToneCurveTextureId
    }

    private fun uploadDcp3DTexture(
        textureIdProvider: () -> Int,
        assignTextureId: (Int) -> Unit,
        table: DcpHueSatMap
    ): Int {
        var textureId = textureIdProvider()
        if (textureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            textureId = textures[0]
            assignTextureId(textureId)
        }

        val rgbaValues = FloatArray(table.hueDivisions * table.satDivisions * table.valueDivisions * 4)
        var srcIndex = 0
        var dstIndex = 0
        while (srcIndex < table.values.size && dstIndex < rgbaValues.size) {
            rgbaValues[dstIndex++] = table.values[srcIndex++]
            rgbaValues[dstIndex++] = table.values[srcIndex++]
            rgbaValues[dstIndex++] = table.values[srcIndex++]
            rgbaValues[dstIndex++] = 1.0f
        }

        val buffer = ByteBuffer.allocateDirect(rgbaValues.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(rgbaValues)
        buffer.position(0)

        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, textureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_WRAP_R, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexImage3D(
            GLES30.GL_TEXTURE_3D,
            0,
            GLES30.GL_RGBA16F,
            table.satDivisions,
            table.hueDivisions,
            table.valueDivisions,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_FLOAT,
            buffer
        )
        checkGlError("uploadDcp3DTexture")
        return textureId
    }

    private fun bindDcpCombinedResources(dcpRenderPlan: DcpRenderPlan?) {
        val hueSatMap = dcpRenderPlan?.hueSatMap?.takeIf { it.isValid }
        val lookTable = dcpRenderPlan?.lookTable?.takeIf { it.isValid }

        PLog.d(TAG, "bindDcpCombinedResources: hueSatMap=${hueSatMap?.values?.size}")
        PLog.d(TAG, "bindDcpCombinedResources: lookTable=${lookTable?.values?.size}")

        GLES30.glUniform1i(GLES30.glGetUniformLocation(combinedProgram, "uDcpHueSatTexture"), 2)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(combinedProgram, "uDcpLookTableTexture"), 3)
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uDcpHueSatEnabled"),
            if (hueSatMap != null) 1 else 0
        )
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uDcpLookTableEnabled"),
            if (lookTable != null) 1 else 0
        )

        if (hueSatMap != null) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            val textureId = uploadDcp3DTexture({ dcpHueSatTextureId }, { dcpHueSatTextureId = it }, hueSatMap)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, textureId)
            GLES30.glUniform3i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpHueSatDivisions"),
                hueSatMap.hueDivisions,
                hueSatMap.satDivisions,
                hueSatMap.valueDivisions
            )
            GLES30.glUniform1i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpHueSatEncoding"),
                hueSatMap.encoding
            )
        } else {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, ensureDummyDcp3DTexture())
            GLES30.glUniform3i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpHueSatDivisions"),
                1,
                1,
                1
            )
            GLES30.glUniform1i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpHueSatEncoding"),
                DcpHueSatMap.ENCODING_LINEAR
            )
        }

        if (lookTable != null) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
            val textureId = uploadDcp3DTexture({ dcpLookTableTextureId }, { dcpLookTableTextureId = it }, lookTable)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, textureId)
            GLES30.glUniform3i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpLookTableDivisions"),
                lookTable.hueDivisions,
                lookTable.satDivisions,
                lookTable.valueDivisions
            )
            GLES30.glUniform1i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpLookTableEncoding"),
                lookTable.encoding
            )
        } else {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, ensureDummyDcp3DTexture())
            GLES30.glUniform3i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpLookTableDivisions"),
                1,
                1,
                1
            )
            GLES30.glUniform1i(
                GLES30.glGetUniformLocation(combinedProgram, "uDcpLookTableEncoding"),
                DcpHueSatMap.ENCODING_LINEAR
            )
        }
        checkGlError("bindDcpCombinedResources")
    }


    /**
     * Combined Processing Pass: ToneMap + LUT + Sharpening
     */
    private fun renderCombinedPass(
        metadata: RawMetadata,
        inputTextureId: Int = demosaicTextureId,
        dcpRenderPlan: DcpRenderPlan? = null,
        viewportWidth: Int = metadata.width,
        viewportHeight: Int = metadata.height,
        shadowLift: Float = 0f
    ) {
        val baseCurve = dcpRenderPlan?.toneCurveLut ?: ACR3Curve.samples()
        val combinedLut = if (shadowLift > 0f) {
            FloatArray(baseCurve.size) { i ->
                val v = baseCurve[i]
                val x = shadowLift.coerceIn(0f, 5f)
                val boost = smoothstep(1f, 5f, x)
                val gammaPower = 0.75f + 0.13f * boost
                val shadowRangeEnd = 0.6f + 0.25f * boost
                val liftStrength = 1.35f - 0.25f * boost
                val blackProtectStart = 0.02f
                val black = 0.004f
                val liftBase = (v.pow(gammaPower) - v).coerceAtLeast(0f)
                val shadowMask = smoothstep(shadowRangeEnd, 0.02f, v)
                val blackProtect = smoothstep(black, blackProtectStart, v)
                val lift = x * liftBase * liftStrength * shadowMask * blackProtect
                (v + lift - black).coerceIn(0f, 1f)
            }
        } else {
            baseCurve
        }
        val outputTransform = computeWorkingToOutputTransform(ColorSpace.ProPhoto, ColorSpace.SRGB)

        GLES30.glUseProgram(combinedProgram)
        checkGlError("renderCombinedPass glUseProgram")
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, combinedFramebufferId)
        checkGlError("renderCombinedPass glBindFramebuffer")

        GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        checkGlError("renderCombinedPass clear")

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(combinedProgram, "uInputTexture"), 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        uploadCurveTexture(combinedLut)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, curveTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(combinedProgram, "uCurveTexture"), 1)
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(combinedProgram, "uCurveSize"),
            combinedLut.size.toFloat()
        )
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uCurveEnabled"),
            1
        )
        checkGlError("renderCombinedPass base uniforms")

        bindDcpCombinedResources(dcpRenderPlan)

        GLES30.glUniformMatrix3fv(
            GLES30.glGetUniformLocation(combinedProgram, "uOutputTransform"),
            1, false, transposeMatrix3x3(outputTransform), 0
        )

        val identityMatrix = FloatArray(16)
        GlMatrix.setIdentityM(identityMatrix, 0)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(combinedProgram, "uTexMatrix"),
            1, false, identityMatrix, 0
        )
        checkGlError("renderCombinedPass matrices")
        drawQuad(combinedProgram)
        checkGlError("renderCombinedPass")
    }

    private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
        val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun logProgramLinkResult(program: Int, name: String): Boolean {
        if (program == 0) {
            PLog.e(TAG, "$name creation failed")
            return false
        }
        val linked = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            PLog.e(TAG, "$name link failed: ${GLES30.glGetProgramInfoLog(program)}")
            GLES30.glDeleteProgram(program)
            return false
        } else {
            PLog.d(TAG, "$name link ok")
            return true
        }
    }

    private fun computeWorkingToOutputTransform(
        workingSpace: ColorSpace,
        outputSpace: ColorSpace
    ): FloatArray {
        val workingFromXyz = computeXyzD50ToGamut(workingSpace) ?: return identityMatrix3x3()
        val xyzFromWorking = invertMatrix3x3(workingFromXyz) ?: return identityMatrix3x3()
        val outputFromXyz = computeXyzD50ToGamut(outputSpace) ?: return identityMatrix3x3()
        return multiplyMatrix3x3(outputFromXyz, xyzFromWorking)
    }

    private fun computeXyzD50ToGamut(colorSpace: ColorSpace): FloatArray? {
        val primaries = colorSpace.primaries
        val whitePoint = colorSpace.whitePoint
        if (primaries.size != 6 || whitePoint.size != 2) return null

        val xr = primaries[0]
        val yr = primaries[1]
        val xg = primaries[2]
        val yg = primaries[3]
        val xb = primaries[4]
        val yb = primaries[5]
        val xw = whitePoint[0]
        val yw = whitePoint[1]

        val mS = floatArrayOf(
            xr / yr, xg / yg, xb / yb,
            1f, 1f, 1f,
            (1 - xr - yr) / yr, (1 - xg - yg) / yg, (1 - xb - yb) / yb
        )
        val invS = invertMatrix3x3(mS) ?: return null

        val xWhite = xw / yw
        val yWhite = 1f
        val zWhite = (1 - xw - yw) / yw

        val sR = invS[0] * xWhite + invS[1] * yWhite + invS[2] * zWhite
        val sG = invS[3] * xWhite + invS[4] * yWhite + invS[5] * zWhite
        val sB = invS[6] * xWhite + invS[7] * yWhite + invS[8] * zWhite

        val gamutToXyzNative = floatArrayOf(
            mS[0] * sR, mS[1] * sG, mS[2] * sB,
            mS[3] * sR, mS[4] * sG, mS[5] * sB,
            mS[6] * sR, mS[7] * sG, mS[8] * sB
        )

        val bradfordD65ToD50 = floatArrayOf(
            1.0478112f, 0.0228866f, -0.0501270f,
            0.0295424f, 0.9904844f, -0.0170491f,
            -0.0092345f, 0.0150436f, 0.7521316f
        )

        val gamutToXyzD50 = if (isD50WhitePoint(xw, yw)) {
            gamutToXyzNative
        } else {
            multiplyMatrix3x3(bradfordD65ToD50, gamutToXyzNative)
        }
        return invertMatrix3x3(gamutToXyzD50)
    }

    private fun isD50WhitePoint(x: Float, y: Float): Boolean {
        return abs(x - 0.3457f) < 0.002f && abs(y - 0.3585f) < 0.002f
    }

    private fun multiplyMatrix3x3(lhs: FloatArray, rhs: FloatArray): FloatArray {
        return FloatArray(9) { index ->
            val row = index / 3
            val col = index % 3
            lhs[row * 3] * rhs[col] +
                    lhs[row * 3 + 1] * rhs[3 + col] +
                    lhs[row * 3 + 2] * rhs[6 + col]
        }
    }

    private fun invertMatrix3x3(matrix: FloatArray): FloatArray? {
        val det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
                matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
                matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])

        if (abs(det) < 1e-12f) {
            PLog.e(TAG, "Matrix is singular, cannot invert")
            return null
        }

        val invDet = 1.0f / det
        return floatArrayOf(
            (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * invDet,
            (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * invDet,
            (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * invDet,
            (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * invDet,
            (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * invDet,
            (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * invDet,
            (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * invDet,
            (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * invDet,
            (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * invDet
        )
    }

    private fun identityMatrix3x3(): FloatArray = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    )

    private fun renderHdrReferencePass(
        metadata: RawMetadata,
        inputTextureId: Int
    ) {
        GLES30.glUseProgram(hdrReferenceProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, hdrReferenceFramebufferId)
        GLES30.glViewport(0, 0, metadata.width, metadata.height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(hdrReferenceProgram, "uInputTexture"), 0)
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(hdrReferenceProgram, "uHighlightStart"),
            RAW_HDR_HIGHLIGHT_START
        )
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(hdrReferenceProgram, "uWhitePointSceneLuma"),
            RAW_HDR_WHITE_POINT_SCENE_LUMA
        )

        val identityMatrix = FloatArray(16)
        GlMatrix.setIdentityM(identityMatrix, 0)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(hdrReferenceProgram, "uTexMatrix"),
            1, false, identityMatrix, 0
        )

        drawQuad(hdrReferenceProgram)
        checkGlError("renderHdrReferencePass")
    }

    /**
     * Sharpen Pass
     */
    private fun renderSharpenPass(
        metadata: RawMetadata,
        sharpeningValue: Float,
        inputTextureId: Int
    ) {
        GLES30.glUseProgram(sharpenProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, sharpenFramebufferId)
        GLES30.glViewport(0, 0, metadata.width, metadata.height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(sharpenProgram, "uInputTexture"), 0)

        GLES30.glUniform2f(
            GLES30.glGetUniformLocation(sharpenProgram, "uTexelSize"),
            1.0f / metadata.width, 1.0f / metadata.height
        )
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(sharpenProgram, "uSharpening"),
            sharpeningValue
        )

        val identityMatrix = FloatArray(16)
        GlMatrix.setIdentityM(identityMatrix, 0)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(sharpenProgram, "uTexMatrix"),
            1, false, identityMatrix, 0
        )

        drawQuad(sharpenProgram)
        checkGlError("renderSharpenPass")
    }

    private fun renderLinearPass(
        metadata: RawMetadata,
        rawExposureCompensation: Float,
        rawBlackPointCorrection: Float,
        rawWhitePointCorrection: Float,
        dcpRenderPlan: DcpRenderPlan? = null
    ) {
        GLES30.glUseProgram(linearProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, demosaicFramebufferId)
        GLES30.glViewport(0, 0, metadata.width, metadata.height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, rawTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(linearProgram, "uRawTexture"), 0)
        GLES30.glUniform2f(
            GLES30.glGetUniformLocation(linearProgram, "uImageSize"),
            metadata.width.toFloat(), metadata.height.toFloat()
        )
        val transposedCCM = transposeMatrix3x3(dcpRenderPlan?.colorCorrectionMatrix ?: metadata.colorCorrectionMatrix)

        GLES30.glUniformMatrix3fv(
            GLES30.glGetUniformLocation(linearProgram, "uColorCorrectionMatrix"),
            1, false, transposedCCM, 0
        )
        val identity = FloatArray(16)
        GlMatrix.setIdentityM(identity, 0)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(linearProgram, "uTexMatrix"),
            1, false, identity, 0
        )
        val normalizationGain = ExposureNormalization.compute(metadata)
        val dcpBaselineExposureOffset = dcpRenderPlan?.baselineExposureOffset ?: 0f
        PLog.d(
            TAG,
            "compute: normalizationGain=$normalizationGain baseline=${metadata.baselineExposure} dcpBaselineOffset=$dcpBaselineExposureOffset"
        )
        val exposureGain = normalizationGain * 2.0f.pow(rawExposureCompensation + dcpBaselineExposureOffset)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(linearProgram, "uExposureGain"), exposureGain)
        val (blackLevel, whiteLevel) = resolveLinearInputLevels(
            metadata = metadata,
            rawBlackPointCorrection = rawBlackPointCorrection,
            rawWhitePointCorrection = rawWhitePointCorrection
        )
        GLES30.glUniform3f(
            GLES30.glGetUniformLocation(linearProgram, "uBlackLevel"),
            blackLevel[0], blackLevel[1], blackLevel[2]
        )
        GLES30.glUniform3f(
            GLES30.glGetUniformLocation(linearProgram, "uWhiteLevel"),
            whiteLevel[0], whiteLevel[1], whiteLevel[2]
        )
        drawQuad(linearProgram)
    }

    private suspend fun resolveRawAutoExposureEv(
        context: Context,
        metadata: RawMetadata,
        sourceTextureId: Int,
        rawDROMode: DROMode,
        dcpRenderPlan: DcpRenderPlan?
    ): MeteringSystem.MeteringResult {
        val meteringWidth = minOf(metadata.width, 256).coerceAtLeast(1)
        val meteringHeight = minOf(metadata.height, 256).coerceAtLeast(1)
        return try {
            setupCombinedFramebuffer(meteringWidth, meteringHeight)
            renderCombinedPass(
                metadata = metadata,
                inputTextureId = sourceTextureId,
                dcpRenderPlan = dcpRenderPlan,
                viewportWidth = meteringWidth,
                viewportHeight = meteringHeight
            )
            val buffer = ByteBuffer
                .allocateDirect(meteringWidth * meteringHeight * 4)
                .order(ByteOrder.nativeOrder())
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, combinedFramebufferId)
            GLES30.glReadPixels(
                0,
                0,
                meteringWidth,
                meteringHeight,
                GLES30.GL_RGBA,
                GLES30.GL_UNSIGNED_BYTE,
                buffer
            )
            checkGlError("resolveRawAutoExposureEv")
            buffer.position(0)

            // Calculate depth weighting
            val bitmap = Bitmap.createBitmap(meteringWidth, meteringHeight, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(buffer)
            buffer.position(0) // Reset for MeteringSystem

            val depthMap = SharedDepthEstimator.estimateDepth(context, bitmap)
            val weightBuffer = depthMap?.let {
                val wb = ByteBuffer.allocateDirect(it.byteCount)
                it.copyPixelsToBuffer(wb)
                wb.position(0)
                wb
            }

            val meteringResult = MeteringSystem.analyzeRenderedExposureEv(
                byteBuffer = buffer,
                width = meteringWidth,
                height = meteringHeight,
                weightBuffer = weightBuffer,
                droMode = rawDROMode
            )

            // Clean up temporary bitmaps
            if (depthMap != null && !depthMap.isRecycled) {
                depthMap.recycle()
            }
            if (!bitmap.isRecycled) {
                bitmap.recycle()
            }
            meteringResult
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to resolve RAW auto exposure", e)
            MeteringSystem.MeteringResult(0f, 0f, 0f, 0f)
        }
    }

    private fun resolveLinearInputLevels(
        metadata: RawMetadata,
        rawBlackPointCorrection: Float,
        rawWhitePointCorrection: Float
    ): Pair<FloatArray, FloatArray> {
        val fixedRawBlackPoint = (rawBlackPointCorrection + 0.004f).coerceAtMost(1f)
        if (metadata.cfaPattern == RawMetadata.CFA_LINEAR_RGB) {
            val encodedMax = 65535f
            val blackLevel = FloatArray(3) { (encodedMax * fixedRawBlackPoint).coerceAtLeast(0f) }
            val whiteBase = encodedMax * (1f + rawWhitePointCorrection)
            val whiteLevel = FloatArray(3) { channel -> maxOf(whiteBase, blackLevel[channel] + 1f) }
            return blackLevel to whiteLevel
        }

        val sensorRange = metadata.whiteLevel - metadata.blackLevel.average().toFloat()
        val blackPointOffset = sensorRange * fixedRawBlackPoint
        val whitePointOffset = sensorRange * rawWhitePointCorrection
        val blackLevel = FloatArray(3) { channel ->
            val sourceIndex = channelToCfaChannelIndex(channel)
            (metadata.blackLevel.getOrElse(sourceIndex) { metadata.blackLevel.firstOrNull() ?: 0f } + blackPointOffset)
                .coerceAtLeast(0f)
        }
        val whiteLevel = FloatArray(3) { channel ->
            val level = metadata.whiteLevel + whitePointOffset
            maxOf(level, blackLevel[channel] + 1f)
        }
        return blackLevel to whiteLevel
    }

    private fun channelToCfaChannelIndex(channel: Int): Int {
        return when (channel) {
            0 -> 0
            1 -> 1
            else -> 3
        }
    }

    private fun renderOutputPass(rotation: Int, width: Int, height: Int, bounds: Rect, sourceTextureId: Int) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, outputFramebufferId)
        GLES30.glViewport(0, 0, bounds.width(), bounds.height())
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(passthroughProgram)
        val isSwapped = rotation == 90 || rotation == 270
        val cropW: Float
        val cropH: Float
        val cropCenterX: Float
        val cropCenterY: Float
        if (isSwapped) {
            cropW = bounds.height().toFloat()
            cropH = bounds.width().toFloat()
            cropCenterX = (bounds.top + bounds.height() / 2f)
            cropCenterY = (bounds.left + bounds.width() / 2f)
        } else {
            cropW = bounds.width().toFloat()
            cropH = bounds.height().toFloat()
            cropCenterX = bounds.centerX().toFloat()
            cropCenterY = bounds.centerY().toFloat()
        }
        val texMatrix = FloatArray(16)
        GlMatrix.setIdentityM(texMatrix, 0)
        GlMatrix.translateM(texMatrix, 0, cropCenterX / width, cropCenterY / height, 0f)
        GlMatrix.scaleM(texMatrix, 0, cropW / width, cropH / height, 1.0f)
        GlMatrix.rotateM(texMatrix, 0, -rotation.toFloat(), 0f, 0f, 1f)
        GlMatrix.translateM(texMatrix, 0, -0.5f, -0.5f, 0f)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(passthroughProgram, "uTexMatrix"),
            1, false, texMatrix, 0
        )
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(passthroughProgram, "uTexture"), 0)
        drawQuad(passthroughProgram)
        checkGlError("renderOutputPass")
    }

    private fun drawQuad(program: Int) {
        val positionHandle = GLES30.glGetAttribLocation(program, "aPosition")
        val texCoordHandle = GLES30.glGetAttribLocation(program, "aTexCoord")
        if (positionHandle >= 0) {
            vertexBuffer?.let {
                GLES30.glEnableVertexAttribArray(positionHandle)
                it.position(0)
                GLES30.glVertexAttribPointer(positionHandle, 2, GLES30.GL_FLOAT, false, 0, it)
            }
        }
        if (texCoordHandle >= 0) {
            texCoordBuffer?.let {
                GLES30.glEnableVertexAttribArray(texCoordHandle)
                it.position(0)
                GLES30.glVertexAttribPointer(texCoordHandle, 2, GLES30.GL_FLOAT, false, 0, it)
            }
        }
        indexBuffer?.let {
            it.position(0)
            GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, it)
        }
        if (positionHandle >= 0) GLES30.glDisableVertexAttribArray(positionHandle)
        if (texCoordHandle >= 0) GLES30.glDisableVertexAttribArray(texCoordHandle)
    }

    /**
     * 从当前 outputFramebuffer 读取像素并创建 Bitmap。
     *
     * 优先使用 PBO（Pixel Buffer Object）：像素数据存放在 GPU 内存（通过 glMapBufferRange 映射为
     * native ByteBuffer），完全不占用 Java 堆，避免超分时 fusedBayerBuffer +
     * pixelBuffer 同时存活导致 Java 堆 OOM（512 MB 设备上三者合计可达 768 MB）。
     * 若 PBO 分配或 map 失败则降级为直接分配 ByteBuffer。
     */
    private fun readPixels(width: Int, height: Int, colorSpace: android.graphics.ColorSpace): Bitmap? {
        val pixelSize = width * height * 8

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, outputFramebufferId)
        GLES30.glPixelStorei(GLES30.GL_PACK_ALIGNMENT, 8)

        // --- PBO 路径（避免 Java 堆分配）---
        if (pboId == 0) {
            val ids = IntArray(1)
            GLES30.glGenBuffers(1, ids, 0)
            pboId = ids[0]
        }
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pboId)
        // glBufferData(null) 重新分配 GPU 缓冲区（旧数据自动孤立释放）
        GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, pixelSize, null, GLES30.GL_STREAM_READ)
        val pboReady = GLES30.glGetError() == GLES30.GL_NO_ERROR
        if (pboReady) {
            // offset=0：读取写入已绑定的 PBO（GPU→GPU DMA，不阻塞 Java 堆）
            GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, 0)
            checkGlError("readPixels PBO glReadPixels")
            // 映射 PBO 为 native ByteBuffer（不占用 Java 堆）
            val mappedBuffer = GLES30.glMapBufferRange(
                GLES30.GL_PIXEL_PACK_BUFFER, 0, pixelSize, GLES30.GL_MAP_READ_BIT
            ) as? ByteBuffer
            if (mappedBuffer != null) {
                return try {
                    createBitmap(width, height, Bitmap.Config.RGBA_F16, colorSpace = colorSpace).also { bmp ->
                        bmp.copyPixelsFromBuffer(mappedBuffer.order(ByteOrder.nativeOrder()))
                    }
                } catch (e: OutOfMemoryError) {
                    PLog.e(TAG, "OOM creating output bitmap ($width x $height)", e)
                    null
                } finally {
                    GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)
                    GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
                }
            }
            PLog.w(TAG, "glMapBufferRange returned null, falling back to direct readPixels")
        } else {
            PLog.w(TAG, "PBO glBufferData failed for ${pixelSize}B, falling back to direct readPixels")
        }
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)

        // --- 降级路径：直接分配 ByteBuffer（Java 堆）---
        val pixelBuffer = try {
            com.hinnka.mycamera.utils.DirectBufferAllocator.allocateNative(pixelSize.toLong())
                ?.order(ByteOrder.nativeOrder())
                ?: throw OutOfMemoryError("Failed to allocate native direct buffer")
        } catch (e: OutOfMemoryError) {
            PLog.e(TAG, "OOM allocating pixel buffer ($width x $height, ${pixelSize}B)", e)
            return null
        }

        try {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, pixelBuffer)
            pixelBuffer.position(0)
            checkGlError("readPixels direct")
            return try {
                createBitmap(width, height, Bitmap.Config.RGBA_F16, colorSpace = colorSpace).also { bitmap ->
                    bitmap.copyPixelsFromBuffer(pixelBuffer)
                }
            } catch (e: OutOfMemoryError) {
                PLog.e(TAG, "OOM creating output bitmap ($width x $height)", e)
                null
            }
        } finally {
            com.hinnka.mycamera.utils.DirectBufferAllocator.freeNative(pixelBuffer)
        }
    }

    /**
     * 裁切 Bitmap 到目标宽高比（居中裁切）
     * GPU 已经处理了裁切，此方法作为降级参考
     */
    private fun cropToAspectRatio(bitmap: Bitmap, aspectRatio: AspectRatio): Bitmap {
        val srcWidth = bitmap.width
        val srcHeight = bitmap.height
        val srcRatio = srcWidth.toFloat() / srcHeight.toFloat()
        val targetRatio = aspectRatio.getValue(false)

        if (abs(srcRatio - targetRatio) < 0.01f) {
            return bitmap
        }

        val cropWidth: Int
        val cropHeight: Int
        val cropX: Int
        val cropY: Int

        if (srcRatio > targetRatio) {
            // 原图更宽，裁切左右
            cropHeight = srcHeight
            cropWidth = (srcHeight * targetRatio).toInt()
            cropX = (srcWidth - cropWidth) / 2
            cropY = 0
        } else {
            // 原图更高，裁切上下
            cropWidth = srcWidth
            cropHeight = (srcWidth / targetRatio).toInt()
            cropX = 0
            cropY = (srcHeight - cropHeight) / 2
        }

        return Bitmap.createBitmap(bitmap, cropX, cropY, cropWidth, cropHeight)
    }

    private fun checkGlError(tag: String) {
        var error: Int
        while (GLES30.glGetError().also { error = it } != GLES30.GL_NO_ERROR) {
            PLog.e(TAG, "$tag: glError $error")
        }
    }

    /**
     * 释放资源
     */
    fun release() {
        if (!isInitialized) return

        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

        if (combinedProgram != 0) GLES30.glDeleteProgram(combinedProgram)
        if (sharpenProgram != 0) GLES30.glDeleteProgram(sharpenProgram)
        if (passthroughProgram != 0) GLES30.glDeleteProgram(passthroughProgram)
        if (hdrReferenceProgram != 0) GLES30.glDeleteProgram(hdrReferenceProgram)

        // Guided Filter & NLM programs
        if (nlmPassHProgram != 0) GLES30.glDeleteProgram(nlmPassHProgram)
        if (nlmPassVProgram != 0) GLES30.glDeleteProgram(nlmPassVProgram)
        // Guided Filter textures and FBOs
        for (i in 0..1) {
            if (gfTexId[i] != 0) GLES30.glDeleteTextures(1, intArrayOf(gfTexId[i]), 0)
            if (gfFboId[i] != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(gfFboId[i]), 0)
        }

        if (rawTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(rawTextureId), 0)
        if (demosaicTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(demosaicTextureId), 0)
        if (demosaicFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(demosaicFramebufferId),
            0
        )
        if (combinedTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(combinedTextureId), 0)
        if (combinedFramebufferId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(combinedFramebufferId), 0)
        if (hdrReferenceTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(hdrReferenceTextureId), 0)
        if (hdrReferenceFramebufferId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(hdrReferenceFramebufferId), 0)
        if (sharpenTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(sharpenTextureId), 0)
        if (sharpenFramebufferId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(sharpenFramebufferId), 0)
        if (curveTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(curveTextureId), 0)
        if (dcpToneCurveTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(dcpToneCurveTextureId), 0)
        if (dcpHueSatTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(dcpHueSatTextureId), 0)
        if (dcpLookTableTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(dcpLookTableTextureId), 0)
        if (dummyDcp3DTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(dummyDcp3DTextureId), 0)
        if (dummyDcpToneCurveTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(dummyDcpToneCurveTextureId), 0)
        if (outputTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(outputTextureId), 0)

        if (outputFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(outputFramebufferId),
            0
        )
        if (pboId != 0) GLES30.glDeleteBuffers(1, intArrayOf(pboId), 0)

        if (lensShadingTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(lensShadingTextureId),
            0
        )
        if (dummyShadingTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(dummyShadingTextureId),
            0
        )

        EGL14.eglMakeCurrent(
            eglDisplay,
            EGL14.EGL_NO_SURFACE,
            EGL14.EGL_NO_SURFACE,
            EGL14.EGL_NO_CONTEXT
        )
        EGL14.eglDestroySurface(eglDisplay, eglSurface)
        EGL14.eglDestroyContext(eglDisplay, eglContext)
        EGL14.eglTerminate(eglDisplay)

        isInitialized = false
        instance = null
        PLog.d(TAG, "RawDemosaicProcessor released")
    }
}
