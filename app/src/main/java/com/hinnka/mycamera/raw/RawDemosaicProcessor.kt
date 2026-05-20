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
import kotlin.math.max
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
        private const val BM3D_REFERENCE_MID_GRAY_SIGMA = 0.012f

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

    // RCD Compute Shader Programs
    private var rcdPopulateProgram = 0
    private var rcdStep1Program = 0
    private var rcdStep2Program = 0
    private var rcdStep3Program = 0
    private var rcdStep40Program = 0
    private var rcdStep41Program = 0
    private var rcdStep42Program = 0
    private var rcdStep43Program = 0
    private var rcdWriteOutputProgram = 0
    private var linearRcdProgram = 0

    private var rawTextureId = 0

    private var demosaicFramebufferId = 0
    private var demosaicTextureId = 0
    private var demosaicWidth = 0
    private var demosaicHeight = 0
    private var linearOutputFramebufferId = 0
    private var linearOutputTextureId = 0

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

    private var lensShadingTextureId = 0
    private var dummyShadingTextureId = 0

    private val defaultUsmRadius = 2.0f
    private val defaultUsmAmount = 0.5f
    private val defaultUsmThreshold = 0.005f

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
            actualRotation = if (dngRawData.rotation != 0) dngRawData.rotation else rotation
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

            // Check GL_MAX_TEXTURE_SIZE. Oversized Bayer RCD inputs are rejected.
            if (actualWidth > maxTextureSize || actualHeight > maxTextureSize) {
                PLog.e(
                    TAG,
                    "Input ${actualWidth}x${actualHeight} exceeds GL_MAX_TEXTURE_SIZE=$maxTextureSize"
                )
                return@withContext null
            }

            val bounds =
                BitmapUtils.calculateProcessedRect(actualWidth, actualHeight, aspectRatio, cropRegion, actualRotation)
            val finalWidth = bounds.width()
            val finalHeight = bounds.height()

            // 4. 第一步：全分辨率处理 (Linear CCM / RCD Compute Shader Demosaic)
            setupFullResFramebuffer(actualWidth, actualHeight)
            uploadRawTextureFromBuffer(
                actualRawData,
                actualWidth,
                actualHeight,
                actualRowStride
            )
            // GPU 已消费 rawData，立即释放 CPU 侧引用，帮助 GC 回收（超分时约 288 MB）
            actualRawData = null

            var shadowLift = 0f
            var effectiveExposureCompensation = rawExposureCompensation

            // Bayer RCD Compute Shader 处理路径 (1:1 直接映射自 darktable RCD)
                val ssboIds = IntArray(9)
                GLES31.glGenBuffers(9, ssboIds, 0)
                val extraMargin = 1024 * 1024 // 1MB 额外余量，彻底防止移动端 GPU 推测性越界读取越界崩溃
                val fullSize = actualWidth * actualHeight * 4 + extraMargin
                val sizes = intArrayOf(
                    fullSize, // CFA_Buf (0)
                    fullSize, // RGB0_Buf (1)
                    fullSize, // RGB1_Buf (2)
                    fullSize, // RGB2_Buf (3)
                    fullSize, // VH_Dir_Buf (4)
                    fullSize, // LPF_Buf (5)
                    fullSize, // P_Diff_Buf (6)
                    fullSize, // Q_Diff_Buf (7)
                    fullSize  // PQ_Dir_Buf (8)
                )
                for (i in 0 until 9) {
                    GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, ssboIds[i])
                    GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, sizes[i], null, GLES31.GL_DYNAMIC_DRAW)
                    GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, i, ssboIds[i])
                }
                GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

                // 2.0 Populate (黑电平扣除与通道归一化)
                val blackLevel4 = FloatArray(4) { idx ->
                    actualMetadata.blackLevel.getOrElse(idx) { actualMetadata.blackLevel.firstOrNull() ?: 0f }
                        .coerceAtLeast(0f)
                }

                GLES31.glUseProgram(rcdPopulateProgram)
                GLES31.glActiveTexture(GLES31.GL_TEXTURE10)
                GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, rawTextureId)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdPopulateProgram, "uRawTexture"), 10) // 对应 RcdShaders.POPULATE 中的 binding = 10
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdPopulateProgram, "uImageSize"), actualWidth, actualHeight)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdPopulateProgram, "uCfaPattern"), actualMetadata.cfaPattern)
                GLES31.glUniform4fv(GLES31.glGetUniformLocation(rcdPopulateProgram, "uBlackLevel"), 1, blackLevel4, 0)
                GLES31.glUniform1f(GLES31.glGetUniformLocation(rcdPopulateProgram, "uWhiteLevel"), actualMetadata.whiteLevel)
                val wbGains = actualMetadata.whiteBalanceGains
                PLog.d(
                    TAG,
                    "RCD populate: cfa=${actualMetadata.cfaPattern} black=${blackLevel4.contentToString()} " +
                        "white=${actualMetadata.whiteLevel} wb=${wbGains.contentToString()} " +
                        "linearBlackPoint=${rawBlackPointCorrection.coerceIn(0f, 0.99f)} " +
                        "linearWhitePoint=${(1f + rawWhitePointCorrection).coerceAtLeast(rawBlackPointCorrection.coerceIn(0f, 0.99f) + 0.01f)}"
                )
                GLES31.glUniform4fv(GLES31.glGetUniformLocation(rcdPopulateProgram, "uWhiteBalanceGains"), 1, wbGains, 0)
                GLES31.glDispatchCompute((actualWidth + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Populate")

                // 2.1 Step 1 (共享内存垂直与水平梯度估计)
                GLES31.glUseProgram(rcdStep1Program)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdStep1Program, "uImageSize"), actualWidth, actualHeight)
                GLES31.glDispatchCompute((actualWidth + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Step 1")

                // 2.2 Step 2 (低通滤波 LPF 计算)
                GLES31.glUseProgram(rcdStep2Program)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdStep2Program, "uImageSize"), actualWidth, actualHeight)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdStep2Program, "uCfaPattern"), actualMetadata.cfaPattern)
                GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Step 2")

                // 2.3 Step 3 (绿通道在红蓝位置的边缘自适应插值)
                GLES31.glUseProgram(rcdStep3Program)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdStep3Program, "uImageSize"), actualWidth, actualHeight)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdStep3Program, "uCfaPattern"), actualMetadata.cfaPattern)
                GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Step 3")

                // 2.4 Step 4_0 (对角线高通滤波差分计算)
                GLES31.glUseProgram(rcdStep40Program)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdStep40Program, "uImageSize"), actualWidth, actualHeight)
                GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Step 4_0")

                // 2.5 Step 4_1 (对角线方向强弱度选择)
                GLES31.glUseProgram(rcdStep41Program)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdStep41Program, "uImageSize"), actualWidth, actualHeight)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdStep41Program, "uCfaPattern"), actualMetadata.cfaPattern)
                GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Step 4_1")

                // 2.6 Step 4_2 (红蓝通道在红蓝位置色差引导插值)
                GLES31.glUseProgram(rcdStep42Program)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdStep42Program, "uImageSize"), actualWidth, actualHeight)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdStep42Program, "uCfaPattern"), actualMetadata.cfaPattern)
                GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Step 4_2")

                // 2.7 Step 4_3 (红蓝通道在绿色位置插值)
                GLES31.glUseProgram(rcdStep43Program)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdStep43Program, "uImageSize"), actualWidth, actualHeight)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdStep43Program, "uCfaPattern"), actualMetadata.cfaPattern)
                GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
                checkGlError("RCD Step 4_3")

                // 2.8 Write Output (组合输出到 RGBA16F 纹理)
                GLES31.glUseProgram(rcdWriteOutputProgram)
                GLES31.glUniform2i(GLES31.glGetUniformLocation(rcdWriteOutputProgram, "uImageSize"), actualWidth, actualHeight)
                GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdWriteOutputProgram, "uBorder"), 4)
                GLES31.glBindImageTexture(11, demosaicTextureId, 0, false, 0, GLES31.GL_WRITE_ONLY, GLES31.GL_RGBA16F) // 对应 RcdShaders.WRITE_OUTPUT 中的 binding = 11
                GLES31.glDispatchCompute((actualWidth + 15) / 16, (actualHeight + 15) / 16, 1)
                GLES31.glMemoryBarrier(GLES31.GL_ALL_BARRIER_BITS)
                checkGlError("RCD Write Output")

                // 解绑 Image Texture 11，避免后续与常规纹理采样单元（glBindTexture）发生绑定冲突
                GLES31.glBindImageTexture(11, 0, 0, false, 0, GLES31.GL_WRITE_ONLY, GLES31.GL_RGBA16F)

                // 强制等待 GPU 彻底完成所有之前的渲染和计算指令，确保 SSBO 被 GPU 完全读取完毕后再进行安全删除
                GLES30.glFinish()

                // 清理 SSBO
                GLES31.glDeleteBuffers(9, ssboIds, 0)
                for (i in 0 until 9) {
                    GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, i, 0)
                }

                // 处理自动曝光测光
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
                        effectiveExposureCompensation += linearEv
                    } else {
                        effectiveExposureCompensation += autoExposureEv
                    }
                }

                // 运行 linearRcdProgram 将 CCM & Exposure 应用在已解马赛克的浮点 RCD 图像上
                checkGlError("Before LinearRcdPass")

                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, linearOutputFramebufferId)
                checkGlError("LinearRcdPass setup framebuffer")

                GLES30.glUseProgram(linearRcdProgram)
                GLES30.glViewport(0, 0, actualWidth, actualHeight)
                GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
                GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, demosaicTextureId)
                GLES30.glUniform1i(GLES30.glGetUniformLocation(linearRcdProgram, "uDemosaickedTexture"), 0)

                val transposedCCM = transposeMatrix3x3(resolvedDcpRenderPlan?.colorCorrectionMatrix ?: actualMetadata.colorCorrectionMatrix)
                GLES30.glUniformMatrix3fv(
                    GLES30.glGetUniformLocation(linearRcdProgram, "uColorCorrectionMatrix"),
                    1, false, transposedCCM, 0
                )
                val exposureGain = computeLinearExposureGain(actualMetadata, effectiveExposureCompensation, resolvedDcpRenderPlan)
                GLES30.glUniform1f(GLES30.glGetUniformLocation(linearRcdProgram, "uExposureGain"), exposureGain)
                GLES30.glUniform1f(
                    GLES30.glGetUniformLocation(linearRcdProgram, "uBlackPoint"),
                    rawBlackPointCorrection.coerceIn(0f, 0.99f)
                )
                GLES30.glUniform1f(
                    GLES30.glGetUniformLocation(linearRcdProgram, "uWhitePoint"),
                    (1f + rawWhitePointCorrection).coerceAtLeast(rawBlackPointCorrection.coerceIn(0f, 0.99f) + 0.01f)
                )

                val identityMatrix = FloatArray(16)
                android.opengl.Matrix.setIdentityM(identityMatrix, 0)
                GLES30.glUniformMatrix4fv(
                    GLES30.glGetUniformLocation(linearRcdProgram, "uTexMatrix"),
                    1, false, identityMatrix, 0
                )

                drawQuad(linearRcdProgram)
                checkGlError("LinearRcdPass drawQuad")

                // 重点：使用双缓冲交换 (Swap)，既不销毁任何纹理，也不需要 glGenTextures/glDeleteTextures
                val tempTex = demosaicTextureId
                demosaicTextureId = linearOutputTextureId
                linearOutputTextureId = tempTex

                val tempFbo = demosaicFramebufferId
                demosaicFramebufferId = linearOutputFramebufferId
                linearOutputFramebufferId = tempFbo

                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
                checkGlError("After LinearRcdPass Swap")
            // rawTextureId 已被 RCD populate 消费，提前释放 GPU 显存
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
                linearExposureGain = computeLinearExposureGain(actualMetadata, effectiveExposureCompensation, resolvedDcpRenderPlan),
                denoiseValue = denoiseValue,
                chromaDenoiseValue = denoiseValue,
            )
            val outputTexture = gfTexId[1]

            // 重点：不要在此处销毁常驻双缓冲的 framebuffer，由 setupFullResFramebuffer 或 release() 统一管理其生命周期
            // if (demosaicFramebufferId != 0) {
            //     GLES30.glDeleteFramebuffers(1, intArrayOf(demosaicFramebufferId), 0)
            //     demosaicFramebufferId = 0
            // }
            // demosaicWidth = 0; demosaicHeight = 0
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
            // outputTexture 已被 combinedPass 消费，提前释放
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
            if (combinedProgram == 0 || sharpenProgram == 0 || passthroughProgram == 0 ||
                rcdPopulateProgram == 0 || rcdStep1Program == 0 || rcdStep2Program == 0 ||
                rcdStep3Program == 0 || rcdStep40Program == 0 || rcdStep41Program == 0 ||
                rcdStep42Program == 0 || rcdStep43Program == 0 || rcdWriteOutputProgram == 0 ||
                linearRcdProgram == 0) {
                PLog.e(TAG, "Critical shader programs failed to compile or link. " +
                        "combined=$combinedProgram sharpen=$sharpenProgram pass=$passthroughProgram " +
                        "populate=$rcdPopulateProgram step1=$rcdStep1Program step2=$rcdStep2Program " +
                        "step3=$rcdStep3Program step40=$rcdStep40Program step41=$rcdStep41Program " +
                        "step42=$rcdStep42Program step43=$rcdStep43Program write=$rcdWriteOutputProgram " +
                        "linearRcd=$linearRcdProgram")
                return false
            }
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

        // 2.8 RCD Programs
        initRcdPrograms(vShader)

        GLES30.glDeleteShader(vShader)
        PLog.d(
            TAG,
            "Shader programs created: combined=$combinedProgram, passthrough=$passthroughProgram"
        )
    }

    private val FRAGMENT_SHADER_LINEAR_RCD = """
        #version 300 es
        precision highp float;
        
        in vec2 vTexCoord;
        out vec4 fragColor;
        
        uniform sampler2D uDemosaickedTexture;
        uniform mat3 uColorCorrectionMatrix;
        uniform float uExposureGain;
        uniform float uBlackPoint;
        uniform float uWhitePoint;
        
        void main() {
            vec3 rgb = texture(uDemosaickedTexture, vTexCoord).rgb;
            float blackPoint = clamp(uBlackPoint, 0.0, 0.99);
            float whitePoint = max(uWhitePoint, blackPoint + 0.01);
            rgb = clamp((rgb - vec3(blackPoint)) / max(whitePoint - blackPoint, 1e-5), 0.0, 1.0);
            // 应用色彩转换 CCM 矩阵和曝光值增益
            rgb = uColorCorrectionMatrix * rgb * uExposureGain;
            fragColor = vec4(rgb, 1.0);
        }
    """.trimIndent()

    private fun initRcdPrograms(vShader: Int) {
        rcdPopulateProgram = compileComputeProgram(RcdShaders.POPULATE, "POPULATE")
        rcdStep1Program = compileComputeProgram(RcdShaders.STEP_1, "STEP_1")
        rcdStep2Program = compileComputeProgram(RcdShaders.STEP_2, "STEP_2")
        rcdStep3Program = compileComputeProgram(RcdShaders.STEP_3, "STEP_3")
        rcdStep40Program = compileComputeProgram(RcdShaders.STEP_4_0, "STEP_4_0")
        rcdStep41Program = compileComputeProgram(RcdShaders.STEP_4_1, "STEP_4_1")
        rcdStep42Program = compileComputeProgram(RcdShaders.STEP_4_2, "STEP_4_2")
        rcdStep43Program = compileComputeProgram(RcdShaders.STEP_4_3, "STEP_4_3")
        rcdWriteOutputProgram = compileComputeProgram(RcdShaders.WRITE_OUTPUT, "WRITE_OUTPUT")

        val fShaderLinearRcd = compileShader(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER_LINEAR_RCD)
        if (vShader != 0 && fShaderLinearRcd != 0) {
            linearRcdProgram = GLES30.glCreateProgram()
            GLES30.glAttachShader(linearRcdProgram, vShader)
            GLES30.glAttachShader(linearRcdProgram, fShaderLinearRcd)
            GLES30.glLinkProgram(linearRcdProgram)
            if (!logProgramLinkResult(linearRcdProgram, "linearRcdProgram")) {
                linearRcdProgram = 0
            }
            GLES30.glDeleteShader(fShaderLinearRcd)
        }
    }

    private fun compileComputeProgram(source: String, name: String): Int {
        val shader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER)
        GLES31.glShaderSource(shader, source)
        GLES31.glCompileShader(shader)

        val compiled = IntArray(1)
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val error = GLES31.glGetShaderInfoLog(shader)
            PLog.e(TAG, "Compute Shader $name compilation failed: $error")
            GLES31.glDeleteShader(shader)
            return 0
        }

        val program = GLES31.glCreateProgram()
        GLES31.glAttachShader(program, shader)
        GLES31.glLinkProgram(program)

        val linked = IntArray(1)
        GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            val error = GLES31.glGetProgramInfoLog(program)
            PLog.e(TAG, "Compute Program $name linking failed: $error")
            GLES31.glDeleteProgram(program)
            GLES31.glDeleteShader(shader)
            return 0
        }

        GLES31.glDeleteShader(shader)
        PLog.d(TAG, "Compute Program $name created: $program")
        return program
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
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
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
        linearExposureGain: Float,
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

        // Camera2 SENSOR_NOISE_PROFILE is defined in the linear RAW domain.
        // Convert it into the linear RGB texture domain consumed by BM3D, so
        // the user strength remains a dimensionless multiplier of measured sigma.
        val isoGain = metadata.iso / 100.0f
        val digitalGain = metadata.postRawSensitivityBoost
        val fallbackGain = (isoGain * digitalGain).coerceAtLeast(1f)
        val (s, o) = resolveDenoiseNoiseModel(metadata, linearExposureGain, fallbackGain)

        val denoiseValue = denoiseValue ?: 0f
        val chromaDenoiseValue = chromaDenoiseValue ?: 0f

        val noiseAdaptiveScale = computeDenoiseStrengthScale(s, o)
        val h = denoiseValue.coerceAtLeast(0f) * BM3DShaders.SIGMA_STRENGTH_AT_SLIDER_ONE * noiseAdaptiveScale
        val ch = chromaDenoiseValue.coerceAtLeast(0f) * BM3DShaders.SIGMA_STRENGTH_AT_SLIDER_ONE * noiseAdaptiveScale + 0.5f
        PLog.d(
            TAG,
            "Dynamic BM3D: s=$s o=$o, h=$h ch=$ch scale=$noiseAdaptiveScale iso=${metadata.iso} linearGain=$linearExposureGain"
        )

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

    private fun resolveDenoiseNoiseModel(
        metadata: RawMetadata,
        linearExposureGain: Float,
        fallbackGain: Float
    ): Pair<Float, Float> {
        var s = metadata.noiseProfile.getOrElse(0) { 0f }
        var o = metadata.noiseProfile.getOrElse(1) { 0f }

        if (s <= 0f || o <= 0f) {
            s = 1E-4f * fallbackGain
            o = 4.5E-7f * sqrt(fallbackGain)
        }

        val frameNoiseScale = 1f / metadata.frameCount.coerceAtLeast(1).toFloat()
        val gain = linearExposureGain.coerceAtLeast(1e-6f)
        val transformedS = s * frameNoiseScale * gain
        val transformedO = o * frameNoiseScale * gain * gain
        return transformedS.coerceAtLeast(1e-10f) to transformedO.coerceAtLeast(1e-10f)
    }

    private fun computeDenoiseStrengthScale(s: Float, o: Float): Float {
        val midGraySigma = sqrt((s * 0.18f + o).coerceAtLeast(1e-10f))
        return (midGraySigma / BM3D_REFERENCE_MID_GRAY_SIGMA).coerceIn(1f, 2.8f)
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
        rowStride: Int
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
        val bytesPerPixel = 2 // 16-bit single-channel Bayer
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
            GLES30.glDeleteTextures(2, intArrayOf(demosaicTextureId, linearOutputTextureId), 0)
            GLES30.glDeleteFramebuffers(2, intArrayOf(demosaicFramebufferId, linearOutputFramebufferId), 0)
            demosaicTextureId = 0
            linearOutputTextureId = 0
            demosaicFramebufferId = 0
            linearOutputFramebufferId = 0
        }

        demosaicWidth = width
        demosaicHeight = height

        val textures = IntArray(2)
        GLES30.glGenTextures(2, textures, 0)
        demosaicTextureId = textures[0]
        linearOutputTextureId = textures[1]

        val fbos = IntArray(2)
        GLES30.glGenFramebuffers(2, fbos, 0)
        demosaicFramebufferId = fbos[0]
        linearOutputFramebufferId = fbos[1]

        // 分配并配置第一个 Immutable 纹理
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, demosaicTextureId)
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA16F, width, height)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, demosaicFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, demosaicTextureId, 0
        )

        // 分配并配置第二个 Immutable 纹理
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, linearOutputTextureId)
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA16F, width, height)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, linearOutputFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, linearOutputTextureId, 0
        )

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        checkGlError("setupFullResFramebuffer Double Buffered")
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

        val hasLensShading = metadata.lensShadingMap != null &&
            metadata.lensShadingMapWidth > 0 &&
            metadata.lensShadingMapHeight > 0
        GLES30.glActiveTexture(GLES30.GL_TEXTURE4)
        if (hasLensShading) {
            uploadLensShadingTexture(metadata)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, lensShadingTextureId)
        } else {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, createDummyShadingTexture())
        }
        GLES30.glUniform1i(GLES30.glGetUniformLocation(combinedProgram, "uLensShadingMap"), 4)
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uLensShadingEnabled"),
            if (hasLensShading) 1 else 0
        )
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(combinedProgram, "uLensShadingPower"),
            computeLensShadingPower(metadata)
        )

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

    private fun computeLensShadingPower(metadata: RawMetadata): Float {
        val map = metadata.lensShadingMap ?: return 1f
        val width = metadata.lensShadingMapWidth
        val height = metadata.lensShadingMapHeight
        if (width <= 0 || height <= 0 || map.size < width * height * 4) return 1f

        fun lumaGainAt(index: Int): Float {
            val base = index * 4
            val r = map.getOrElse(base) { 1f }
            val gr = map.getOrElse(base + 1) { 1f }
            val gb = map.getOrElse(base + 2) { gr }
            val b = map.getOrElse(base + 3) { 1f }
            return (0.2126f * r + 0.7152f * ((gr + gb) * 0.5f) + 0.0722f * b)
                .coerceAtLeast(1e-4f)
        }

        val centerIndex = (height / 2) * width + (width / 2)
        val centerGain = lumaGainAt(centerIndex)
        var maxRelativeGain = 1f
        for (i in 0 until width * height) {
            maxRelativeGain = max(maxRelativeGain, lumaGainAt(i) / centerGain)
        }

        // Limit the strongest post-denoise LSC lift to one stop:
        // applied = relativeGain^power, so power = log(2) / log(maxRelativeGain).
        val maxAllowedLift = 2f
        return if (maxRelativeGain <= maxAllowedLift) {
            1f
        } else {
            (ln(maxAllowedLift) / ln(maxRelativeGain)).coerceIn(0f, 1f)
        }
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
            if (sharpeningValue > 0f) sharpeningValue else defaultUsmAmount
        )
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(sharpenProgram, "uRadius"),
            defaultUsmRadius
        )
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(sharpenProgram, "uThreshold"),
            defaultUsmThreshold
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

    private fun computeLinearExposureGain(
        metadata: RawMetadata,
        rawExposureCompensation: Float,
        dcpRenderPlan: DcpRenderPlan?
    ): Float {
        val normalizationGain = ExposureNormalization.compute(metadata)
        val dcpBaselineExposureOffset = dcpRenderPlan?.baselineExposureOffset ?: 0f
        return normalizationGain * 2.0f.pow(rawExposureCompensation + dcpBaselineExposureOffset)
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

        // RCD Compute Programs
        if (rcdPopulateProgram != 0) GLES31.glDeleteProgram(rcdPopulateProgram)
        if (rcdStep1Program != 0) GLES31.glDeleteProgram(rcdStep1Program)
        if (rcdStep2Program != 0) GLES31.glDeleteProgram(rcdStep2Program)
        if (rcdStep3Program != 0) GLES31.glDeleteProgram(rcdStep3Program)
        if (rcdStep40Program != 0) GLES31.glDeleteProgram(rcdStep40Program)
        if (rcdStep41Program != 0) GLES31.glDeleteProgram(rcdStep41Program)
        if (rcdStep42Program != 0) GLES31.glDeleteProgram(rcdStep42Program)
        if (rcdStep43Program != 0) GLES31.glDeleteProgram(rcdStep43Program)
        if (rcdWriteOutputProgram != 0) GLES31.glDeleteProgram(rcdWriteOutputProgram)
        if (linearRcdProgram != 0) GLES31.glDeleteProgram(linearRcdProgram)

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
        if (linearOutputTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(linearOutputTextureId), 0)
        if (demosaicFramebufferId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(demosaicFramebufferId), 0)
        if (linearOutputFramebufferId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(linearOutputFramebufferId), 0)
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
