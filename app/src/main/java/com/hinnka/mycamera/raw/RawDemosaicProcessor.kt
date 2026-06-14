package com.hinnka.mycamera.raw

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.media.Image
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLES31
import android.opengl.GLUtils
import android.util.Half
import androidx.core.graphics.createBitmap
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.data.ContentRepository
import com.hinnka.mycamera.lut.ChromaDenoiseShaders
import com.hinnka.mycamera.lut.ShadowsHighlightsShader
import com.hinnka.mycamera.ml.SharedDepthEstimator
import com.hinnka.mycamera.raw.MeteringSystem.ShadowsHighlightsParams
import com.hinnka.mycamera.utils.BitmapUtils
import com.hinnka.mycamera.utils.LargeDirectBuffer
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

private typealias ShadowsHighlightsParams = MeteringSystem.ShadowsHighlightsParams

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
            lensShadingMapGrid = dngRawData.lensShadingMapGrid,
            baselineExposure = if (dngRawData.baselineExposure == 0f) (baseMetadata?.baselineExposure
                ?: 0f) else dngRawData.baselineExposure,
            exposureBias = if (dngRawData.exposureBias == 0f) {
                if (baseMetadata != null && baseMetadata.exposureBias != 0f) baseMetadata.exposureBias else exposureBias
            } else dngRawData.exposureBias,
            iso = if (dngRawData.iso == 0) (baseMetadata?.iso ?: 100) else dngRawData.iso,
            shutterSpeed = if (dngRawData.shutterSpeed == 0L) (baseMetadata?.shutterSpeed
                ?: 0L) else dngRawData.shutterSpeed,
            aperture = if (dngRawData.aperture == 0f) (baseMetadata?.aperture
                ?: 0f) else dngRawData.aperture,
            activeArray = activeArray,
            noiseProfile = dngRawData.noiseProfile ?: baseMetadata?.noiseProfile ?: floatArrayOf(
                0f,
                0f
            ),
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
        private const val DEFAULT_RAW_CHROMA_DENOISE_VALUE = 0.5f
        private const val RCD_RAW_TEXTURE_UNIT = 0
        private const val RCD_LENS_SHADING_TEXTURE_UNIT = 1
        private const val RCD_OUTPUT_IMAGE_UNIT = 0
        private const val RCD_PQ_WRITE_BINDING = 5
        private const val RCD_PQ_READ_BINDING = 4
        private const val RCD_VH_DIR_BINDING = 4
        private const val RCD_HIGHLIGHT_RECONSTRUCTION_THRESHOLD = 0.985f
        private const val RCD_HIGHLIGHT_RECONSTRUCTION_CEILING = 8.0f
        private const val RAW_AE_LOCAL_SIZE_X = 16
        private const val RAW_AE_LOCAL_SIZE_Y = 16
        private const val RAW_AE_HISTOGRAM_BINS = 256
        private const val RAW_AE_HISTOGRAM_LOG_MIN = -10f
        private const val RAW_AE_HISTOGRAM_LOG_MAX = 4f
        private const val RAW_AE_HISTOGRAM_BINDING = 0
        private const val RAW_AE_BASE_STATS_BINDING = 1
        private const val RAW_AE_TONE_STATS_BINDING = 0
        private const val AGX_BASE_SRGB_TEXTURE_UNIT = 7

        private val BRADFORD_D65_TO_D50 = floatArrayOf(
            1.0478112f, 0.0228866f, -0.0501270f,
            0.0295424f, 0.9904844f, -0.0170491f,
            -0.0092345f, 0.0150436f, 0.7521316f
        )

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
    private var chromaDenoiseProgram = 0
    private var loggedShadowsHighlightsUniforms = false

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
    private var rawAeBaseProgram = 0
    private var rawAeMertensProgram = 0

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

    private var linearMeteringFramebufferId = 0
    private var linearMeteringTextureId = 0
    private var linearMeteringWidth = 0
    private var linearMeteringHeight = 0

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
    private var readbackPboSize = 0
    private var readbackBuffer: ByteBuffer? = null
    private var readbackBufferSize = 0

    private var curveTextureId = 0
    private var dcpToneCurveTextureId = 0
    private var dcpHueSatTextureId = 0
    private var dcpLookTableTextureId = 0
    private var spectralFilmTextureId = 0
    private var spectralFilmTextureKey: String? = null
    private var agxLutTextureId = 0
    private var agxLutTextureKey: String? = null
    private var dummyDcp3DTextureId = 0
    private var dummyDcpToneCurveTextureId = 0

    // darktable denoiseprofile 降噪资源
    private var denoisePreconditionY0U0V0Program = 0
    private var denoisePreconditionV2Program = 0
    private var denoiseDecomposeProgram = 0
    private var denoiseSynthesizeProgram = 0
    private var denoiseReduceFirstProgram = 0
    private var denoiseReduceSecondProgram = 0
    private var denoiseBacktransformY0U0V0Program = 0
    private var denoiseBacktransformV2Program = 0

    // denoiseprofile 中间纹理: ping-pong (RGBA16F)
    private var gfTexId = intArrayOf(0, 0)
    private var gfFboId = intArrayOf(0, 0)
    private var gfWidth = 0
    private var gfHeight = 0

    suspend fun prewarmDepthEstimator(context: Context) = withContext(Dispatchers.Default) {
        val start = System.currentTimeMillis()
        SharedDepthEstimator.prewarm(context.applicationContext)
        PLog.d(TAG, "RAW DepthEstimator prewarmed, took=${System.currentTimeMillis() - start}ms")
    }

    private val denoiseDetailTexIds = IntArray(DenoiseProfileShaders.BANDS)
    private var denoiseReduceMBufferId = 0
    private var denoiseReduceRBufferId = 0
    private var denoiseReduceBufSize = 0
    private var denoiseReduceSize = 0

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

    data class RawAutoAdjustments(
        val exposureCompensation: Float,
        val highlights: Float,
        val shadows: Float
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

    fun getRawColorSpace(rawColorEngine: RawColorEngine = RawColorEngine.AdobeCurve): ColorSpace {
        return rawColorEngine.workingColorSpace
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
        rawHighlightsAdjustment: Float = 0f,
        rawShadowsAdjustment: Float = 0f,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
        spectralFilmStock: String? = null,
        spectralFilmPrint: String? = null,
        spectralFilmTuning: SpectralFilmTuning = SpectralFilmTuning.DEFAULT,
        rawColorEngine: RawColorEngine = RawColorEngine.AgX,
        onRawAutoAdjustments: ((RawAutoAdjustments) -> Unit)? = null,
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
                rawHighlightsAdjustment = rawHighlightsAdjustment,
                rawShadowsAdjustment = rawShadowsAdjustment,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                rawAutoWhiteBalanceEstimate = rawAutoWhiteBalanceEstimate,
                sharpeningValue = sharpeningValue,
                denoiseValue = denoiseValue,
                rawDcpId = rawDcpId,
                dcpRenderPlan = dcpRenderPlan,
                spectralFilmStock = spectralFilmStock,
                spectralFilmPrint = spectralFilmPrint,
                spectralFilmTuning = spectralFilmTuning,
                rawColorEngine = rawColorEngine,
                dngFile = dngFile,
                onRawAutoAdjustments = onRawAutoAdjustments,
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
        rawHighlightsAdjustment: Float = 0f,
        rawShadowsAdjustment: Float = 0f,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        chromaDenoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
        spectralFilmStock: String? = null,
        spectralFilmPrint: String? = null,
        spectralFilmTuning: SpectralFilmTuning = SpectralFilmTuning.DEFAULT,
        rawColorEngine: RawColorEngine = RawColorEngine.AgX,
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
                rawHighlightsAdjustment = rawHighlightsAdjustment,
                rawShadowsAdjustment = rawShadowsAdjustment,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                rawAutoWhiteBalanceEstimate = rawAutoWhiteBalanceEstimate,
                sharpeningValue = sharpeningValue,
                denoiseValue = denoiseValue,
                chromaDenoiseValue = chromaDenoiseValue,
                rawDcpId = rawDcpId,
                dcpRenderPlan = dcpRenderPlan,
                spectralFilmStock = spectralFilmStock,
                spectralFilmPrint = spectralFilmPrint,
                spectralFilmTuning = spectralFilmTuning,
                rawColorEngine = rawColorEngine
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
        rawHighlightsAdjustment: Float = 0f,
        rawShadowsAdjustment: Float = 0f,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
        spectralFilmStock: String? = null,
        spectralFilmPrint: String? = null,
        spectralFilmTuning: SpectralFilmTuning = SpectralFilmTuning.DEFAULT,
        rawColorEngine: RawColorEngine = RawColorEngine.AgX,
        onRawAutoAdjustments: ((RawAutoAdjustments) -> Unit)? = null,
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
                rawHighlightsAdjustment = rawHighlightsAdjustment,
                rawShadowsAdjustment = rawShadowsAdjustment,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                rawAutoWhiteBalanceEstimate = rawAutoWhiteBalanceEstimate,
                sharpeningValue = sharpeningValue,
                denoiseValue = denoiseValue,
                rawDcpId = rawDcpId,
                dcpRenderPlan = dcpRenderPlan,
                spectralFilmStock = spectralFilmStock,
                spectralFilmPrint = spectralFilmPrint,
                spectralFilmTuning = spectralFilmTuning,
                rawColorEngine = rawColorEngine,
                dngFile = dngFile,
                onRawAutoAdjustments = onRawAutoAdjustments,
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
        rawHighlightsAdjustment: Float = 0f,
        rawShadowsAdjustment: Float = 0f,
        rawBlackPointCorrection: Float = 0f,
        rawWhitePointCorrection: Float = 0f,
        rawAutoWhiteBalanceEstimate: Boolean = false,
        sharpeningValue: Float = 0f,
        denoiseValue: Float? = null,
        chromaDenoiseValue: Float? = null,
        rawDcpId: String? = null,
        dcpRenderPlan: DcpRenderPlan? = null,
        spectralFilmStock: String? = null,
        spectralFilmPrint: String? = null,
        spectralFilmTuning: SpectralFilmTuning = SpectralFilmTuning.DEFAULT,
        rawColorEngine: RawColorEngine = RawColorEngine.AdobeCurve,
        dngFile: File? = null,
        onRawAutoAdjustments: ((RawAutoAdjustments) -> Unit)? = null,
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
        val requestedColorEngine = rawColorEngine
        val rawWorkingColorSpace = requestedColorEngine.workingColorSpace

        if (dngFile != null) {
            val dngRawData = processDngNative(
                dngFile.absolutePath,
                rawWorkingColorSpace.xr, rawWorkingColorSpace.yr,
                rawWorkingColorSpace.xg, rawWorkingColorSpace.yg,
                rawWorkingColorSpace.xb, rawWorkingColorSpace.yb,
                rawWorkingColorSpace.xw, rawWorkingColorSpace.yw,
                rawAutoWhiteBalanceEstimate
            )
            if (dngRawData == null) {
                return@withContext RawProcessor.processAndToBitmap(
                    dngFile,
                    aspectRatio,
                    cropRegion,
                    rotation
                )?.let {
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

        val resolvedDcpRenderPlan =
            if (requestedColorEngine == RawColorEngine.AdobeCurve) {
                dcpRenderPlan ?: rawDcpId?.let { dcpId ->
                    val dcpInfo = ContentRepository.getInstance(context).getAvailableDcps()
                        .firstOrNull { it.id == dcpId }
                    if (dcpInfo == null) {
                        PLog.w(TAG, "RAW DCP not found: $dcpId")
                        null
                    } else {
                        DcpProfileParser.resolveRenderPlan(
                            context,
                            dcpInfo,
                            actualMetadata,
                            rawWorkingColorSpace
                        ).also { plan ->
                            if (plan == null) {
                                PLog.w(TAG, "Failed to resolve RAW DCP render plan: $dcpId")
                            } else {
                                PLog.d(TAG, "Resolved RAW DCP plan: ${plan.profileName}")
                            }
                        }
                    }
                }
            } else {
                null
            }
        val spectralFilmLut =
            if (requestedColorEngine == RawColorEngine.SpectralFilm &&
                spectralFilmStock != null && spectralFilmPrint != null
            ) {
                SpectralFilmProfile.loadCombinedLut(
                    context,
                    spectralFilmStock,
                    spectralFilmPrint,
                    spectralFilmTuning
                )
            } else {
                null
            }
        val agxLut =
            if (requestedColorEngine == RawColorEngine.AgX) {
                AgXColorEngine.loadBaseSrgbLut(context)
            } else {
                null
            }
        val colorEngine = when {
            requestedColorEngine == RawColorEngine.AgX && agxLut == null -> {
                PLog.w(TAG, "AgX LUT unavailable, falling back to AdobeCurve")
                RawColorEngine.AdobeCurve
            }

            requestedColorEngine == RawColorEngine.SpectralFilm && spectralFilmLut == null -> {
                PLog.w(TAG, "SpectralFilm LUT unavailable, falling back to AdobeCurve")
                RawColorEngine.AdobeCurve
            }

            else -> requestedColorEngine
        }

        PLog.d(
            TAG,
            "Processing RAW image: ${actualWidth}x${actualHeight}, " +
                "colorEngine=$colorEngine workingColorSpace=$rawWorkingColorSpace"
        )

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
                BitmapUtils.calculateProcessedRect(
                    actualWidth,
                    actualHeight,
                    aspectRatio,
                    cropRegion,
                    actualRotation
                )
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
                GLES31.glBufferData(
                    GLES31.GL_SHADER_STORAGE_BUFFER,
                    sizes[i],
                    null,
                    GLES31.GL_DYNAMIC_DRAW
                )
                if (i < 8) {
                    GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, i, ssboIds[i])
                }
            }
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

            // 2.0 Populate (黑电平扣除与通道归一化)
            val blackLevel4 = FloatArray(4) { idx ->
                actualMetadata.blackLevel.getOrElse(idx) {
                    actualMetadata.blackLevel.firstOrNull() ?: 0f
                }
                    .coerceAtLeast(0f)
            }

            GLES31.glUseProgram(rcdPopulateProgram)
            GLES31.glActiveTexture(GLES31.GL_TEXTURE0 + RCD_RAW_TEXTURE_UNIT)
            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, rawTextureId)
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rcdPopulateProgram, "uRawTexture"),
                RCD_RAW_TEXTURE_UNIT
            )
            bindLensShadingForRcdPopulate(actualMetadata)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdPopulateProgram, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rcdPopulateProgram, "uCfaPattern"),
                actualMetadata.cfaPattern
            )
            GLES31.glUniform4fv(
                GLES31.glGetUniformLocation(rcdPopulateProgram, "uBlackLevel"),
                1,
                blackLevel4,
                0
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(rcdPopulateProgram, "uWhiteLevel"),
                actualMetadata.whiteLevel
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(rcdPopulateProgram, "uHighlightClipThreshold"),
                RCD_HIGHLIGHT_RECONSTRUCTION_THRESHOLD
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(rcdPopulateProgram, "uHighlightCeiling"),
                RCD_HIGHLIGHT_RECONSTRUCTION_CEILING
            )
            val wbGains = actualMetadata.whiteBalanceGains
            val lscSize = if (hasValidLensShadingMap(actualMetadata)) {
                "${actualMetadata.lensShadingMapWidth}x${actualMetadata.lensShadingMapHeight}"
            } else {
                "none"
            }
            PLog.d(
                TAG,
                "RCD populate: cfa=${actualMetadata.cfaPattern} black=${blackLevel4.contentToString()} " +
                        "white=${actualMetadata.whiteLevel} wb=${wbGains.contentToString()} " +
                        "lsc=$lscSize " +
                        "highlightThreshold=$RCD_HIGHLIGHT_RECONSTRUCTION_THRESHOLD " +
                        "highlightCeiling=$RCD_HIGHLIGHT_RECONSTRUCTION_CEILING " +
                        "linearBlackPoint=${rawBlackPointCorrection.coerceIn(0f, 0.99f)} " +
                        "linearWhitePoint=${
                            (1f + rawWhitePointCorrection).coerceAtLeast(
                                rawBlackPointCorrection.coerceIn(0f, 0.99f) + 0.01f
                            )
                        }"
            )
            GLES31.glUniform4fv(
                GLES31.glGetUniformLocation(
                    rcdPopulateProgram,
                    "uWhiteBalanceGains"
                ), 1, wbGains, 0
            )
            GLES31.glDispatchCompute((actualWidth + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Populate")

            // 2.1 Step 1 (共享内存垂直与水平梯度估计)
            GLES31.glUseProgram(rcdStep1Program)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdStep1Program, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glDispatchCompute((actualWidth + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Step 1")

            // 2.2 Step 2 (低通滤波 LPF 计算)
            GLES31.glUseProgram(rcdStep2Program)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdStep2Program, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rcdStep2Program, "uCfaPattern"),
                actualMetadata.cfaPattern
            )
            GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Step 2")

            // 2.3 Step 3 (绿通道在红蓝位置的边缘自适应插值)
            GLES31.glUseProgram(rcdStep3Program)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdStep3Program, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rcdStep3Program, "uCfaPattern"),
                actualMetadata.cfaPattern
            )
            GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Step 3")

            // 2.4 Step 4_0 (对角线高通滤波差分计算)
            GLES31.glUseProgram(rcdStep40Program)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdStep40Program, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Step 4_0")

            // 2.5 Step 4_1 (对角线方向强弱度选择)
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                RCD_PQ_WRITE_BINDING,
                ssboIds[8]
            )
            GLES31.glUseProgram(rcdStep41Program)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdStep41Program, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rcdStep41Program, "uCfaPattern"),
                actualMetadata.cfaPattern
            )
            GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Step 4_1")

            // 2.6 Step 4_2 (红蓝通道在红蓝位置色差引导插值)
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                RCD_PQ_READ_BINDING,
                ssboIds[8]
            )
            GLES31.glUseProgram(rcdStep42Program)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdStep42Program, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rcdStep42Program, "uCfaPattern"),
                actualMetadata.cfaPattern
            )
            GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Step 4_2")

            // 2.7 Step 4_3 (红蓝通道在绿色位置插值)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, RCD_VH_DIR_BINDING, ssboIds[4])
            GLES31.glUseProgram(rcdStep43Program)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdStep43Program, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rcdStep43Program, "uCfaPattern"),
                actualMetadata.cfaPattern
            )
            GLES31.glDispatchCompute((actualWidth / 2 + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
            checkGlError("RCD Step 4_3")

            // 2.8 Write Output (组合输出到 RGBA16F 纹理)
            GLES31.glUseProgram(rcdWriteOutputProgram)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rcdWriteOutputProgram, "uImageSize"),
                actualWidth,
                actualHeight
            )
            GLES31.glUniform1i(GLES31.glGetUniformLocation(rcdWriteOutputProgram, "uBorder"), 4)
            GLES31.glBindImageTexture(
                RCD_OUTPUT_IMAGE_UNIT,
                demosaicTextureId,
                0,
                false,
                0,
                GLES31.GL_WRITE_ONLY,
                GLES31.GL_RGBA16F
            )
            GLES31.glDispatchCompute((actualWidth + 15) / 16, (actualHeight + 15) / 16, 1)
            GLES31.glMemoryBarrier(GLES31.GL_ALL_BARRIER_BITS)
            checkGlError("RCD Write Output")

            GLES31.glBindImageTexture(
                RCD_OUTPUT_IMAGE_UNIT,
                0,
                0,
                false,
                0,
                GLES31.GL_WRITE_ONLY,
                GLES31.GL_RGBA16F
            )

            // 强制等待 GPU 彻底完成所有之前的渲染和计算指令，确保 SSBO 被 GPU 完全读取完毕后再进行安全删除
            GLES30.glFinish()

            // 清理 SSBO
            GLES31.glDeleteBuffers(9, ssboIds, 0)
            for (i in 0 until 8) {
                GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, i, 0)
            }

            val linearExposureGain = computeLinearExposureGain(
                actualMetadata,
                0f,
                resolvedDcpRenderPlan
            )

            // 复用 RAW AE 的线性 metering 分布统计，AE 与 Shadows/Highlights 使用同一份结果。
            val meteringResult = resolveRawAutoExposureEv(
                context = context,
                metadata = actualMetadata,
                sourceTextureId = demosaicTextureId,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                dcpRenderPlan = resolvedDcpRenderPlan,
                meteringCompensationEv = colorEngine.meteringCompensationEv,
                useDepthWeighting = rawAutoExposure
            )

            val useAutoDevelopAdjustments = rawAutoExposure && !MeteringSystem.hasManualRawDevelopAdjustments(
                rawExposureCompensation = rawExposureCompensation,
                rawHighlightsAdjustment = rawHighlightsAdjustment,
                rawShadowsAdjustment = rawShadowsAdjustment
            )
            val effectiveExposureCompensation = if (useAutoDevelopAdjustments) {
                meteringResult.meteredEv
            } else {
                rawExposureCompensation
            }
            val renderExposureCompensation =
                effectiveExposureCompensation + colorEngine.defaultExposureCompensationEv
            val shadowsHighlightsParams = if (useAutoDevelopAdjustments) {
                ShadowsHighlightsParams(
                    highlights = meteringResult.highlights,
                    shadows = meteringResult.shadows,
                    curveWhitePoint = meteringResult.curveWhitePoint
                )
            } else {
                ShadowsHighlightsParams(
                    highlights = rawHighlightsAdjustment,
                    shadows = rawShadowsAdjustment,
                    curveWhitePoint = meteringResult.curveWhitePoint
                )
            }
            if (useAutoDevelopAdjustments) {
                RawAutoAdjustments(
                    exposureCompensation = effectiveExposureCompensation.coerceIn(-2f, 2f),
                    highlights = shadowsHighlightsParams.highlights,
                    shadows = shadowsHighlightsParams.shadows
                ).also { adjustments ->
                    PLog.d(
                        TAG,
                        "RAW auto develop sliders: exposure=${adjustments.exposureCompensation} " +
                            "highlights=${adjustments.highlights} shadows=${adjustments.shadows} " +
                            "engineDefaultEv=${colorEngine.defaultExposureCompensationEv} " +
                            "engineMeteringEv=${colorEngine.meteringCompensationEv} " +
                            "engineCompensationDomain=${colorEngine.exposureCompensationDomain}"
                    )
                    onRawAutoAdjustments?.invoke(adjustments)
                }
            }
            PLog.d(
                TAG,
                "RAW render exposure: userOrAutoEv=$effectiveExposureCompensation " +
                    "engineDefaultEv=${colorEngine.defaultExposureCompensationEv} " +
                    "engineMeteringEv=${colorEngine.meteringCompensationEv} " +
                    "engineCompensationDomain=${colorEngine.exposureCompensationDomain} " +
                    "renderEv=$renderExposureCompensation"
            )

            // 运行 linearRcdProgram 将 CCM & Exposure 应用在已解马赛克的浮点 RCD 图像上
            checkGlError("Before LinearRcdPass")

            renderLinearRcdPass(
                metadata = actualMetadata,
                sourceTextureId = demosaicTextureId,
                targetFramebufferId = linearOutputFramebufferId,
                viewportWidth = actualWidth,
                viewportHeight = actualHeight,
                rawExposureCompensation = renderExposureCompensation,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                dcpRenderPlan = resolvedDcpRenderPlan,
                label = "LinearRcdPass"
            )

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
            val denoiseSourceTextureId = renderDefaultChromaDenoiseBeforeDenoiseProfile(
                sourceTextureId = demosaicTextureId,
                width = actualWidth,
                height = actualHeight,
                metadata = actualMetadata,
                linearExposureGain = linearExposureGain,
            )

            // darktable denoiseprofile 降噪
            renderDenoiseProfilePass(
                sourceTextureId = denoiseSourceTextureId,
                width = actualWidth,
                height = actualHeight,
                metadata = actualMetadata,
                linearExposureGain = linearExposureGain,
                denoiseValue = denoiseValue,
            )
            val outputTexture = gfTexId[1]

            // 重点：不要在此处销毁常驻双缓冲的 framebuffer，由 setupFullResFramebuffer 或 release() 统一管理其生命周期
            // if (demosaicFramebufferId != 0) {
            //     GLES30.glDeleteFramebuffers(1, intArrayOf(demosaicFramebufferId), 0)
            //     demosaicFramebufferId = 0
            // }
            // demosaicWidth = 0; demosaicHeight = 0
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
                spectralFilmLut = spectralFilmLut,
                agxLut = agxLut,
                colorEngine = colorEngine,
                shadowsHighlightsParams = shadowsHighlightsParams
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
            if (!EGL14.eglChooseConfig(
                    eglDisplay,
                    configAttribs,
                    0,
                    configs,
                    0,
                    1,
                    numConfigs,
                    0
                )
            ) {
                PLog.e(TAG, "Unable to choose EGL config")
                return false
            }

            val config = configs[0] ?: return false

            // 创建 EGL Context (ES 3.0)
            val contextAttribs = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
            )
            eglContext =
                EGL14.eglCreateContext(eglDisplay, config, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
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
                chromaDenoiseProgram == 0 ||
                rcdPopulateProgram == 0 || rcdStep1Program == 0 || rcdStep2Program == 0 ||
                rcdStep3Program == 0 || rcdStep40Program == 0 || rcdStep41Program == 0 ||
                rcdStep42Program == 0 || rcdStep43Program == 0 || rcdWriteOutputProgram == 0 ||
                linearRcdProgram == 0
            ) {
                PLog.e(
                    TAG, "Critical shader programs failed to compile or link. " +
                            "combined=$combinedProgram sharpen=$sharpenProgram pass=$passthroughProgram " +
                            "chromaDenoise=$chromaDenoiseProgram " +
                            "populate=$rcdPopulateProgram step1=$rcdStep1Program step2=$rcdStep2Program " +
                            "step3=$rcdStep3Program step40=$rcdStep40Program step41=$rcdStep41Program " +
                            "step42=$rcdStep42Program step43=$rcdStep43Program write=$rcdWriteOutputProgram " +
                            "linearRcd=$linearRcdProgram"
                )
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
            logGlResourceLimits()

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
        val fShaderCombined =
            compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.COMBINED_FRAGMENT_SHADER)
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

        val fShaderHdrReference =
            compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.HDR_REFERENCE_FRAGMENT_SHADER)
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
        val fShaderSharpen =
            compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.SHARPEN_FRAGMENT_SHADER)
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

        // 2.75 RAW 默认色度降噪 Program
        val fShaderChromaDenoise =
            compileShader(GLES30.GL_FRAGMENT_SHADER, ChromaDenoiseShaders.PASS_CHROMA_DENOISE)
        if (vShader != 0 && fShaderChromaDenoise != 0) {
            chromaDenoiseProgram = GLES30.glCreateProgram()
            GLES30.glAttachShader(chromaDenoiseProgram, vShader)
            GLES30.glAttachShader(chromaDenoiseProgram, fShaderChromaDenoise)
            GLES30.glLinkProgram(chromaDenoiseProgram)
            if (!logProgramLinkResult(chromaDenoiseProgram, "rawChromaDenoiseProgram")) {
                chromaDenoiseProgram = 0
            }

            GLES30.glDeleteShader(fShaderChromaDenoise)
        }

        // 3. Passthrough Program
        val fShaderPass =
            compileShader(GLES30.GL_FRAGMENT_SHADER, RawShaders.PASSTHROUGH_FRAGMENT_SHADER)
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
            rgb = max((rgb - vec3(blackPoint)) / max(whitePoint - blackPoint, 1e-5), vec3(0.0));
            // 应用色彩转换 CCM 矩阵和曝光值增益
            rgb = uColorCorrectionMatrix * rgb * uExposureGain;
            fragColor = vec4(rgb, 1.0);
        }
    """.trimIndent()

    private val RAW_AE_BASE_COMPUTE_SHADER = """
        #version 310 es
        precision highp float;
        precision highp int;

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        const int HISTOGRAM_BINS = 256;
        const int LOCAL_SIZE = 256;
        const float LUMA_FLOOR = 0.001;
        const float MAX_LINEAR_LUMA = 16.0;
        const float LOG_LUMA_MIN = -10.0;
        const float LOG_LUMA_MAX = 4.0;

        uniform sampler2D uLinearTexture;
        uniform sampler2D uDepthTexture;
        uniform ivec2 uImageSize;
        uniform int uGroupsX;
        uniform int uUseDepthWeight;

        layout(std430, binding = 0) buffer HistogramOut {
            uint histogram[];
        };

        layout(std430, binding = 1) buffer BaseStatsOut {
            vec4 baseStats[];
        };

        shared uint localHistogram[HISTOGRAM_BINS];
        shared vec4 localStats[LOCAL_SIZE];

        float smoothStepRaw(float edge0, float edge1, float x) {
            float width = edge1 - edge0;
            if (abs(width) < 0.000001) {
                return x >= edge1 ? 1.0 : 0.0;
            }
            float t = clamp((x - edge0) / width, 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        }

        float sanitizeLuma(float value, out float sanitized) {
            sanitized = 0.0;
            if (!(value >= 0.0)) {
                sanitized = 1.0;
                return 0.0;
            }
            if (value > MAX_LINEAR_LUMA) {
                sanitized = 1.0;
                return MAX_LINEAR_LUMA;
            }
            return value;
        }

        float centerWeight(vec2 uv) {
            vec2 d = (uv - vec2(0.5)) / 0.32;
            float gaussian = exp(-0.5 * dot(d, d));
            return 0.35 + gaussian * 0.65;
        }

        float depthWeight(vec2 uv) {
            if (uUseDepthWeight == 0) {
                return 1.0;
            }
            float depth = texture(uDepthTexture, uv).r;
            float separation = abs(depth - 0.5) / 0.5;
            float saliency = smoothStepRaw(0.18, 0.88, separation);
            return mix(0.88, 1.55, saliency);
        }

        void main() {
            int localIndex = int(gl_LocalInvocationIndex);
            if (localIndex < HISTOGRAM_BINS) {
                localHistogram[localIndex] = 0u;
            }
            localStats[localIndex] = vec4(0.0);
            barrier();

            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            if (coord.x < uImageSize.x && coord.y < uImageSize.y) {
                vec3 rgb = texelFetch(uLinearTexture, coord, 0).rgb;
                float sanitized = 0.0;
                float luma = sanitizeLuma(dot(rgb, vec3(0.2126, 0.7152, 0.0722)), sanitized);
                float safeLuma = max(luma, LUMA_FLOOR);
                float logLuma = log2(safeLuma);
                float histogramT = clamp((logLuma - LOG_LUMA_MIN) / (LOG_LUMA_MAX - LOG_LUMA_MIN), 0.0, 0.999999);
                int bin = int(histogramT * float(HISTOGRAM_BINS));
                atomicAdd(localHistogram[bin], 1u);

                vec2 uv = (vec2(coord) + vec2(0.5)) / vec2(uImageSize);
                float weight = centerWeight(uv) * depthWeight(uv);
                localStats[localIndex] = vec4(logLuma * weight, weight, 1.0, sanitized);
            }
            barrier();

            for (int stride = LOCAL_SIZE / 2; stride > 0; stride = stride / 2) {
                if (localIndex < stride) {
                    localStats[localIndex] += localStats[localIndex + stride];
                }
                barrier();
            }

            int groupIndex = int(gl_WorkGroupID.y) * uGroupsX + int(gl_WorkGroupID.x);
            if (localIndex == 0) {
                baseStats[groupIndex] = localStats[0];
            }
            if (localIndex < HISTOGRAM_BINS) {
                histogram[groupIndex * HISTOGRAM_BINS + localIndex] = localHistogram[localIndex];
            }
        }
    """.trimIndent()

    private val RAW_AE_MERTENS_COMPUTE_SHADER = """
        #version 310 es
        precision highp float;
        precision highp int;

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        const int LOCAL_SIZE = 256;
        const float LUMA_FLOOR = 0.001;
        const float MAX_LINEAR_LUMA = 16.0;
        const float MERTENS_TARGET_TO_MID_INTENSITY_SCALE = 2.0;
        const float MERTENS_WELL_EXPOSED_CENTER = 0.5;
        const float MERTENS_WELL_EXPOSED_SIGMA = 0.2;
        const float MERTENS_SHADOW_MASK_START = 0.04;
        const float MERTENS_SHADOW_MASK_END = 0.45;
        const float MERTENS_HIGHLIGHT_MASK_START = 0.55;
        const float MERTENS_HIGHLIGHT_MASK_END = 0.92;

        uniform sampler2D uLinearTexture;
        uniform ivec2 uImageSize;
        uniform int uGroupsX;
        uniform int uSampleStep;
        uniform int uSampleOffset;
        uniform float uExposureScale;
        uniform float uTargetLuma;

        layout(std430, binding = 0) buffer ToneStatsOut {
            vec4 toneStats[];
        };

        shared vec4 localToneA[LOCAL_SIZE];
        shared vec4 localToneB[LOCAL_SIZE];

        float smoothStepRaw(float edge0, float edge1, float x) {
            float width = edge1 - edge0;
            if (abs(width) < 0.000001) {
                return x >= edge1 ? 1.0 : 0.0;
            }
            float t = clamp((x - edge0) / width, 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        }

        float sanitizeLinearLuma(float value) {
            if (!(value >= 0.0)) {
                return 0.0;
            }
            return clamp(value, 0.0, MAX_LINEAR_LUMA);
        }

        int reflect101(int position, int size) {
            if (size <= 1) {
                return 0;
            }
            int period = size * 2 - 2;
            int value = position;
            if (value < 0) {
                value = -value;
            }
            value = value - (value / period) * period;
            return value >= size ? period - value : value;
        }

        vec3 exposureFusionRgb(ivec2 coord, float exposureScale) {
            vec3 rgb = texelFetch(uLinearTexture, coord, 0).rgb * exposureScale;
            float divisor = max(uTargetLuma, LUMA_FLOOR) * MERTENS_TARGET_TO_MID_INTENSITY_SCALE;
            return clamp(vec3(
                sanitizeLinearLuma(rgb.r) / divisor,
                sanitizeLinearLuma(rgb.g) / divisor,
                sanitizeLinearLuma(rgb.b) / divisor
            ), vec3(0.0), vec3(1.0));
        }

        float exposureFusionIntensityAt(int x, int y, float exposureScale) {
            ivec2 coord = ivec2(
                reflect101(x, uImageSize.x),
                reflect101(y, uImageSize.y)
            );
            vec3 rgb = exposureFusionRgb(coord, exposureScale);
            return clamp(dot(rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        }

        float exposureFusionIntensity(ivec2 coord, float exposureScale) {
            vec3 rgb = exposureFusionRgb(coord, exposureScale);
            return clamp(dot(rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        }

        float mertensContrast(ivec2 coord, float exposureScale) {
            int x = coord.x;
            int y = coord.y;
            float center = exposureFusionIntensityAt(x, y, exposureScale);
            float left = exposureFusionIntensityAt(x - 1, y, exposureScale);
            float right = exposureFusionIntensityAt(x + 1, y, exposureScale);
            float up = exposureFusionIntensityAt(x, y - 1, exposureScale);
            float down = exposureFusionIntensityAt(x, y + 1, exposureScale);
            return max(abs(left + right + up + down - center * 4.0), 0.0);
        }

        float mertensSaturation(vec3 rgb) {
            float mean = (rgb.r + rgb.g + rgb.b) / 3.0;
            vec3 delta = rgb - vec3(mean);
            return max(sqrt(dot(delta, delta)), 0.0);
        }

        float mertensWellExposedness(float intensity) {
            float offset = clamp(intensity, 0.0, 1.0) - MERTENS_WELL_EXPOSED_CENTER;
            float sigma2 = MERTENS_WELL_EXPOSED_SIGMA * MERTENS_WELL_EXPOSED_SIGMA;
            return clamp(exp(-0.5 * offset * offset / sigma2), 0.0, 1.0);
        }

        vec4 mertensWeight(ivec2 coord, float exposureScale) {
            vec3 rgb = exposureFusionRgb(coord, exposureScale);
            float contrast = mertensContrast(coord, exposureScale);
            float saturation = mertensSaturation(rgb);
            float wellExposedness = clamp(
                mertensWellExposedness(rgb.r) *
                mertensWellExposedness(rgb.g) *
                mertensWellExposedness(rgb.b),
                0.0,
                1.0
            );
            float weight = contrast * saturation * wellExposedness;
            if (!(weight > 0.0)) {
                weight = 0.0;
            }
            return vec4(weight, contrast, saturation, wellExposedness);
        }

        void main() {
            int localIndex = int(gl_LocalInvocationIndex);
            localToneA[localIndex] = vec4(0.0);
            localToneB[localIndex] = vec4(0.0);

            ivec2 sampleCoord = ivec2(gl_GlobalInvocationID.xy) * uSampleStep + ivec2(uSampleOffset);
            if (sampleCoord.x < uImageSize.x && sampleCoord.y < uImageSize.y) {
                float baseIntensity = exposureFusionIntensity(sampleCoord, uExposureScale);
                float weightedIntensitySum = 0.0;
                float weightSum = 0.0;
                float contrastSum = 0.0;
                float saturationSum = 0.0;
                float wellExposednessSum = 0.0;

                for (int i = 0; i < 5; i++) {
                    float ev = float(i) - 2.0;
                    float virtualExposureScale = uExposureScale * exp2(ev);
                    float virtualIntensity = exposureFusionIntensity(sampleCoord, virtualExposureScale);
                    vec4 weight = mertensWeight(sampleCoord, virtualExposureScale);
                    weightedIntensitySum += virtualIntensity * weight.x;
                    weightSum += weight.x;
                    contrastSum += weight.y;
                    saturationSum += weight.z;
                    wellExposednessSum += weight.w;
                }

                float fusedIntensity = weightSum > 0.0
                    ? clamp(weightedIntensitySum / weightSum, 0.0, 1.0)
                    : baseIntensity;
                float fusionDelta = fusedIntensity - baseIntensity;
                float highlightMask = smoothStepRaw(
                    MERTENS_HIGHLIGHT_MASK_START,
                    MERTENS_HIGHLIGHT_MASK_END,
                    baseIntensity
                );
                float shadowMask = 1.0 - smoothStepRaw(
                    MERTENS_SHADOW_MASK_START,
                    MERTENS_SHADOW_MASK_END,
                    baseIntensity
                );
                float highlightFusionDelta = fusionDelta < 0.0 ? -fusionDelta : 0.0;
                float shadowFusionDelta = fusionDelta > 0.0 ? fusionDelta : 0.0;

                localToneA[localIndex] = vec4(
                    highlightFusionDelta * highlightFusionDelta * highlightMask,
                    shadowFusionDelta * shadowFusionDelta * shadowMask,
                    highlightMask,
                    shadowMask
                );
                localToneB[localIndex] = vec4(
                    contrastSum / 5.0,
                    saturationSum / 5.0,
                    wellExposednessSum / 5.0,
                    1.0
                );
            }
            barrier();

            for (int stride = LOCAL_SIZE / 2; stride > 0; stride = stride / 2) {
                if (localIndex < stride) {
                    localToneA[localIndex] += localToneA[localIndex + stride];
                    localToneB[localIndex] += localToneB[localIndex + stride];
                }
                barrier();
            }

            if (localIndex == 0) {
                int groupIndex = int(gl_WorkGroupID.y) * uGroupsX + int(gl_WorkGroupID.x);
                toneStats[groupIndex * 2] = localToneA[0];
                toneStats[groupIndex * 2 + 1] = localToneB[0];
            }
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
        rawAeBaseProgram = compileComputeProgram(RAW_AE_BASE_COMPUTE_SHADER, "RAW_AE_BASE")
        rawAeMertensProgram = compileComputeProgram(RAW_AE_MERTENS_COMPUTE_SHADER, "RAW_AE_MERTENS")

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
     * 初始化 darktable denoiseprofile compute 着色器。
     */
    private fun initNLMPrograms(vShader: Int) {
        denoisePreconditionY0U0V0Program = compileComputeProgram(
            DenoiseProfileShaders.PRECONDITION_Y0U0V0,
            "DenoiseProfile_Precondition_Y0U0V0"
        )
        denoisePreconditionV2Program = compileComputeProgram(
            DenoiseProfileShaders.PRECONDITION_V2,
            "DenoiseProfile_Precondition_V2"
        )
        denoiseDecomposeProgram =
            compileComputeProgram(DenoiseProfileShaders.DECOMPOSE, "DenoiseProfile_Decompose")
        denoiseSynthesizeProgram =
            compileComputeProgram(DenoiseProfileShaders.SYNTHESIZE, "DenoiseProfile_Synthesize")
        denoiseReduceFirstProgram =
            compileComputeProgram(DenoiseProfileShaders.REDUCE_FIRST, "DenoiseProfile_ReduceFirst")
        denoiseReduceSecondProgram = compileComputeProgram(
            DenoiseProfileShaders.REDUCE_SECOND,
            "DenoiseProfile_ReduceSecond"
        )
        denoiseBacktransformY0U0V0Program = compileComputeProgram(
            DenoiseProfileShaders.BACKTRANSFORM_Y0U0V0,
            "DenoiseProfile_Backtransform_Y0U0V0"
        )
        denoiseBacktransformV2Program = compileComputeProgram(
            DenoiseProfileShaders.BACKTRANSFORM_V2,
            "DenoiseProfile_Backtransform_V2"
        )

        PLog.d(
            TAG,
            "DenoiseProfile programs: preY=$denoisePreconditionY0U0V0Program preRgb=$denoisePreconditionV2Program " +
                    "decompose=$denoiseDecomposeProgram synthesize=$denoiseSynthesizeProgram " +
                    "reduce1=$denoiseReduceFirstProgram reduce2=$denoiseReduceSecondProgram " +
                    "backY=$denoiseBacktransformY0U0V0Program backRgb=$denoiseBacktransformV2Program"
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

        // 创建双缓冲 (RGBA16F) 用于 denoiseprofile 中间 pass
        for (i in 0..1) {
            val t = IntArray(1)
            val f = IntArray(1)
            GLES30.glGenTextures(1, t, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0])
            GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA16F, width, height)
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MIN_FILTER,
                GLES30.GL_NEAREST
            )
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MAG_FILTER,
                GLES30.GL_NEAREST
            )
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_S,
                GLES30.GL_CLAMP_TO_EDGE
            )
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_T,
                GLES30.GL_CLAMP_TO_EDGE
            )
            GLES30.glGenFramebuffers(1, f, 0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, f[0])
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER,
                GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D,
                t[0],
                0
            )
            val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
            if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
                PLog.e(TAG, "DenoiseProfile ping-pong FBO $i incomplete: $status")
            }
            gfTexId[i] = t[0]; gfFboId[i] = f[0]
        }
        setupDenoiseProfileResources(width, height)
        checkGlError("setupGuidedFilterFramebuffers")
    }

    private fun setupDenoiseProfileResources(width: Int, height: Int) {
        for (i in denoiseDetailTexIds.indices) {
            if (denoiseDetailTexIds[i] != 0) GLES30.glDeleteTextures(
                1,
                intArrayOf(denoiseDetailTexIds[i]),
                0
            )
            val t = IntArray(1)
            GLES30.glGenTextures(1, t, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0])
            GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA16F, width, height)
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MIN_FILTER,
                GLES30.GL_NEAREST
            )
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MAG_FILTER,
                GLES30.GL_NEAREST
            )
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_S,
                GLES30.GL_CLAMP_TO_EDGE
            )
            GLES30.glTexParameteri(
                GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_T,
                GLES30.GL_CLAMP_TO_EDGE
            )
            denoiseDetailTexIds[i] = t[0]
        }

        val groupsX = roundUp(
            width,
            DenoiseProfileShaders.REDUCE_FIRST_LOCAL_X
        ) / DenoiseProfileShaders.REDUCE_FIRST_LOCAL_X
        val groupsY = roundUp(
            height,
            DenoiseProfileShaders.REDUCE_FIRST_LOCAL_Y
        ) / DenoiseProfileShaders.REDUCE_FIRST_LOCAL_Y
        denoiseReduceBufSize = groupsX * groupsY
        denoiseReduceSize = kotlin.math.min(
            DenoiseProfileShaders.REDUCE_SIZE,
            roundUp(
                denoiseReduceBufSize,
                DenoiseProfileShaders.REDUCE_SECOND_LOCAL_X
            ) / DenoiseProfileShaders.REDUCE_SECOND_LOCAL_X
        ).coerceAtLeast(1)

        if (denoiseReduceMBufferId != 0) GLES31.glDeleteBuffers(
            1,
            intArrayOf(denoiseReduceMBufferId),
            0
        )
        if (denoiseReduceRBufferId != 0) GLES31.glDeleteBuffers(
            1,
            intArrayOf(denoiseReduceRBufferId),
            0
        )
        val buffers = IntArray(2)
        GLES31.glGenBuffers(2, buffers, 0)
        denoiseReduceMBufferId = buffers[0]
        denoiseReduceRBufferId = buffers[1]
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, denoiseReduceMBufferId)
        GLES31.glBufferData(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            denoiseReduceBufSize * 4 * 4,
            null,
            GLES31.GL_DYNAMIC_DRAW
        )
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, denoiseReduceRBufferId)
        GLES31.glBufferData(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            denoiseReduceSize * 4 * 4,
            null,
            GLES31.GL_DYNAMIC_READ
        )
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
    }

    private fun renderDefaultChromaDenoiseBeforeDenoiseProfile(
        sourceTextureId: Int,
        width: Int,
        height: Int,
        metadata: RawMetadata,
        linearExposureGain: Float,
    ): Int {
        val strength = DEFAULT_RAW_CHROMA_DENOISE_VALUE
        if (strength <= 0f || width * height < 2) {
            return sourceTextureId
        }

        if (chromaDenoiseProgram == 0 || linearOutputFramebufferId == 0 || linearOutputTextureId == 0) {
            PLog.w(
                TAG,
                "RAW chroma denoise program not initialized, falling back to denoiseprofile source"
            )
            return sourceTextureId
        }

        val profileGain =
            (metadata.iso / 100.0f * metadata.postRawSensitivityBoost).coerceAtLeast(1f)
        val (noiseA, noiseB) = resolveDenoiseNoiseModel(metadata, linearExposureGain, profileGain)
        val h = strength * strength * ChromaDenoiseShaders.SIGMA_STRENGTH_AT_SLIDER_ONE
        val identityMatrix = FloatArray(16)
        GlMatrix.setIdentityM(identityMatrix, 0)

        GLES30.glUseProgram(chromaDenoiseProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, linearOutputFramebufferId)
        GLES30.glViewport(0, 0, width, height)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(chromaDenoiseProgram, "uInputTexture"), 0)
        GLES30.glUniform2f(
            GLES30.glGetUniformLocation(chromaDenoiseProgram, "uTexelSize"),
            1.0f / width,
            1.0f / height
        )
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(chromaDenoiseProgram, "uTexMatrix"),
            1,
            false,
            identityMatrix,
            0
        )
        GLES30.glUniform1f(GLES30.glGetUniformLocation(chromaDenoiseProgram, "uH"), h)
        GLES30.glUniform2f(
            GLES30.glGetUniformLocation(chromaDenoiseProgram, "uNoiseModel"),
            noiseA,
            noiseB
        )
        drawQuad(chromaDenoiseProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        checkGlError("RAW chroma denoise before denoiseprofile")

        PLog.d(
            TAG,
            "RAW chroma denoise before denoiseprofile: strength=$strength h=$h a=$noiseA b=$noiseB"
        )
        return linearOutputTextureId
    }


    /**
     * 渲染 darktable denoiseprofile wavelets 降噪。
     *
     * 管线: linear RGB → variance-stabilizing transform → wavelet decompose/synthesize → inverse transform → gfFboId[1]
     */
    private fun renderDenoiseProfilePass(
        sourceTextureId: Int,
        width: Int,
        height: Int,
        metadata: RawMetadata,
        linearExposureGain: Float,
        denoiseValue: Float?,
    ) {
        setupNLMFramebuffers(width, height)

        if (!isDenoiseProfileReady()) {
            PLog.w(TAG, "DenoiseProfile programs not initialized, falling back to passthrough")
            renderPassthroughToTexture(sourceTextureId, width, height, gfFboId[1])
            return
        }

        val params = buildDenoiseProfileParams(metadata, linearExposureGain, denoiseValue ?: 0f)
        if (params.strength <= 0f || width * height < 2) {
            renderPassthroughToTexture(sourceTextureId, width, height, gfFboId[1])
            return
        }

        PLog.d(
            TAG,
            "DenoiseProfile wavelets: strength=${params.strength} a=${params.a} b=${params.b} " +
                    "shadows=${params.shadows} bias=${params.bias} maxScale=${params.maxScale} wb=${params.wb.contentToString()}"
        )

        dispatchDenoisePrecondition(sourceTextureId, gfTexId[0], width, height, params)
        var buf1 = gfTexId[0]
        var buf2 = gfTexId[1]

        for (scale in 0 until params.maxScale) {
            dispatchDenoiseDecompose(buf1, buf2, denoiseDetailTexIds[scale], width, height, scale)
            val tmp = buf2
            buf2 = buf1
            buf1 = tmp
        }

        for (scale in params.maxScale - 1 downTo 0) {
            val threshold = computeDenoiseProfileThreshold(
                textureId = denoiseDetailTexIds[scale],
                width = width,
                height = height,
                scale = scale,
                params = params
            )
            dispatchDenoiseSynthesize(
                buf1,
                denoiseDetailTexIds[scale],
                buf2,
                width,
                height,
                threshold
            )
            val tmp = buf2
            buf2 = buf1
            buf1 = tmp
        }

        val finalInput = if (buf1 == gfTexId[1]) {
            renderPassthroughToTexture(buf1, width, height, gfFboId[0])
            gfTexId[0]
        } else {
            buf1
        }
        dispatchDenoiseBacktransform(finalInput, gfTexId[1], width, height, params)
        checkGlError("renderDenoiseProfile")
    }

    private data class DenoiseProfileParams(
        val strength: Float,
        val a: Float,
        val b: Float,
        val shadows: Float,
        val bias: Float,
        val scale: Float,
        val maxScale: Int,
        val p: FloatArray,
        val wb: FloatArray,
        val aa: FloatArray,
        val bb: FloatArray,
        val toY0U0V0: FloatArray,
        val toRgb: FloatArray
    )

    private fun isDenoiseProfileReady(): Boolean {
        return denoisePreconditionY0U0V0Program != 0 &&
                denoiseDecomposeProgram != 0 &&
                denoiseSynthesizeProgram != 0 &&
                denoiseReduceFirstProgram != 0 &&
                denoiseReduceSecondProgram != 0 &&
                denoiseBacktransformY0U0V0Program != 0
    }

    private fun buildDenoiseProfileParams(
        metadata: RawMetadata,
        linearExposureGain: Float,
        strengthValue: Float
    ): DenoiseProfileParams {
        val profileGain =
            (metadata.iso / 100.0f * metadata.postRawSensitivityBoost).coerceAtLeast(1f)
        val (noiseA, noiseB) = resolveDenoiseNoiseModel(metadata, linearExposureGain, profileGain)
        val a = noiseA.coerceAtLeast(1e-10f)
        val b = noiseB.coerceAtLeast(1e-10f)
        val strength = strengthValue.coerceAtLeast(0f)
        val scale = 1.0f
        val shadows = inferDenoiseProfileShadows(a)
        val bias = inferDenoiseProfileBias(a)
        val wb = computeDenoiseProfileWb(metadata)
        val p = floatArrayOf(
            max(shadows + 0.1f * ln(scale / wb[0]), 0.0f),
            max(shadows + 0.1f * ln(scale / wb[1]), 0.0f),
            max(shadows + 0.1f * ln(scale / wb[2]), 0.0f),
            1.0f
        )
        val compensateP = 0.05f / 0.05f.pow(shadows)
        val compensateStrength = 2.5f
        val scaledWb = floatArrayOf(
            wb[0] * strength * compensateStrength * scale,
            wb[1] * strength * compensateStrength * scale,
            wb[2] * strength * compensateStrength * scale,
            0.0f
        )
        val aa = floatArrayOf(a * compensateP, a * compensateP, a * compensateP, 1.0f)
        val bb = floatArrayOf(b, b, b, 1.0f)
        val (toY, toRgbBase) = computeDenoiseProfileY0U0V0Matrices(wb)
        val toYScaled = FloatArray(9)
        val toRgbScaled = FloatArray(9)
        for (k in 0 until 3) {
            for (c in 0 until 3) {
                val idx = k * 3 + c
                toYScaled[idx] =
                    toY[idx] / (strength * compensateStrength * scale).coerceAtLeast(1e-10f)
                toRgbScaled[idx] = toRgbBase[idx] * strength * compensateStrength * scale
            }
        }

        return DenoiseProfileParams(
            strength = strength,
            a = a,
            b = b,
            shadows = shadows,
            bias = bias,
            scale = scale,
            maxScale = computeDenoiseProfileMaxScale(
                metadata.width,
                metadata.height,
                metadata.width,
                metadata.height
            ),
            p = p,
            wb = scaledWb,
            aa = aa,
            bb = bb,
            toY0U0V0 = toYScaled,
            toRgb = toRgbScaled
        )
    }

    private fun inferDenoiseProfileShadows(a: Float): Float {
        return max(0.1f - 0.1f * ln(a), 0.7f).coerceAtMost(1.8f)
    }

    private fun inferDenoiseProfileBias(a: Float): Float {
        return -max(5f + 0.5f * ln(a), 0.0f)
    }

    private fun computeDenoiseProfileWb(metadata: RawMetadata): FloatArray {
        val r = metadata.whiteBalanceGains.getOrElse(0) { 1f }.coerceAtLeast(1e-6f)
        val g =
            ((metadata.whiteBalanceGains.getOrElse(1) { 1f } + metadata.whiteBalanceGains.getOrElse(
                2
            ) { 1f }) * 0.5f)
                .coerceAtLeast(1e-6f)
        val b = metadata.whiteBalanceGains.getOrElse(3) { 1f }.coerceAtLeast(1e-6f)
        return floatArrayOf(r, g, b, 0f)
    }

    private fun computeDenoiseProfileMaxScale(
        width: Int,
        height: Int,
        fullWidth: Int,
        fullHeight: Int
    ): Int {
        val scale = 1.0f
        val maxMaxScale = DenoiseProfileShaders.BANDS
        val supp0 = kotlin.math.min(
            (2 * (2 shl (maxMaxScale - 1)) + 1).toFloat(),
            max(fullHeight.toFloat(), fullWidth.toFloat()) * 0.2f
        )
        val i0 = ln((supp0 - 1.0f) * 0.5f) / ln(2.0f)
        var maxScale = 0
        while (maxScale < maxMaxScale) {
            val supp = (2 * (2 shl maxScale) + 1).toFloat()
            val suppIn = supp * (1.0f / scale)
            val iIn = ln((suppIn - 1.0f) * 0.5f) / ln(2.0f) - 1.0f
            val t = 1.0f - (iIn + 0.5f) / i0
            if (t < 0.0f) break
            maxScale++
        }
        return maxScale.coerceIn(1, DenoiseProfileShaders.BANDS)
    }

    private fun computeDenoiseProfileY0U0V0Matrices(wb: FloatArray): Pair<FloatArray, FloatArray> {
        val toY = floatArrayOf(
            1f / 3f, 1f / 3f, 1f / 3f,
            0.5f, 0.0f, -0.5f,
            0.25f, -0.5f, 0.25f
        )
        var sumInvWb = 1.0f / wb[0] + 1.0f / wb[1] + 1.0f / wb[2]
        sumInvWb *= sqrt(3.0f)
        toY[0] = sumInvWb / wb[0]
        toY[1] = sumInvWb / wb[1]
        toY[2] = sumInvWb / wb[2]
        val stddevU0 = sqrt(0.25f * wb[0] * wb[0] + 0.25f * wb[2] * wb[2]).coerceAtLeast(1e-6f)
        val stddevV0 =
            sqrt(0.0625f * wb[0] * wb[0] + 0.25f * wb[1] * wb[1] + 0.0625f * wb[2] * wb[2]).coerceAtLeast(
                1e-6f
            )
        for (i in 3..5) toY[i] /= stddevU0
        for (i in 6..8) toY[i] /= stddevV0
        val toRgb = invertMatrix3x3RowMajor(toY) ?: run {
            val fallback = toY.copyOf()
            val stddevY0 =
                sqrt((wb[0] * wb[0] + wb[1] * wb[1] + wb[2] * wb[2]) / 9.0f).coerceAtLeast(1e-6f)
            fallback[0] = 1.0f / (3.0f * stddevY0)
            fallback[1] = fallback[0]
            fallback[2] = fallback[0]
            invertMatrix3x3RowMajor(fallback) ?: identityMatrix3x3()
        }
        return toY to toRgb
    }

    private fun invertMatrix3x3RowMajor(m: FloatArray): FloatArray? {
        val bigA = m[4] * m[8] - m[5] * m[7]
        val bigB = -m[3] * m[8] + m[5] * m[6]
        val bigC = m[3] * m[7] - m[4] * m[6]
        val bigD = -m[1] * m[8] + m[2] * m[7]
        val bigE = m[0] * m[8] - m[2] * m[6]
        val bigF = -m[0] * m[7] + m[1] * m[6]
        val bigG = m[1] * m[5] - m[2] * m[4]
        val bigH = -m[0] * m[5] + m[2] * m[3]
        val bigI = m[0] * m[4] - m[1] * m[3]
        val det = m[0] * bigA + m[1] * bigB + m[2] * bigC
        if (det == 0f) return null
        val invDet = 1.0f / det
        return floatArrayOf(
            invDet * bigA, invDet * bigD, invDet * bigG,
            invDet * bigB, invDet * bigE, invDet * bigH,
            invDet * bigC, invDet * bigF, invDet * bigI
        )
    }

    private fun matrix3RowMajorToGlColumnMajor(m: FloatArray): FloatArray {
        return floatArrayOf(
            m[0], m[3], m[6],
            m[1], m[4], m[7],
            m[2], m[5], m[8]
        )
    }

    private fun dispatchDenoisePrecondition(
        sourceTextureId: Int,
        outputTextureId: Int,
        width: Int,
        height: Int,
        params: DenoiseProfileParams
    ) {
        val program = denoisePreconditionY0U0V0Program
        GLES31.glUseProgram(program)
        bindComputeSampler(program, "uInput", 0, sourceTextureId)
        GLES31.glBindImageTexture(
            1,
            outputTextureId,
            0,
            false,
            0,
            GLES31.GL_WRITE_ONLY,
            GLES31.GL_RGBA16F
        )
        setDenoiseCommonUniforms(program, width, height, params)
        GLES31.glUniformMatrix3fv(
            GLES31.glGetUniformLocation(program, "uToY0U0V0"),
            1, false, matrix3RowMajorToGlColumnMajor(params.toY0U0V0), 0
        )
        dispatchDenoiseImage(width, height, "DenoiseProfile precondition")
    }

    private fun dispatchDenoiseDecompose(
        inputTextureId: Int,
        coarseTextureId: Int,
        detailTextureId: Int,
        width: Int,
        height: Int,
        scale: Int
    ) {
        val program = denoiseDecomposeProgram
        GLES31.glUseProgram(program)
        bindComputeSampler(program, "uInput", 0, inputTextureId)
        GLES31.glBindImageTexture(
            1,
            coarseTextureId,
            0,
            false,
            0,
            GLES31.GL_WRITE_ONLY,
            GLES31.GL_RGBA16F
        )
        GLES31.glBindImageTexture(
            2,
            detailTextureId,
            0,
            false,
            0,
            GLES31.GL_WRITE_ONLY,
            GLES31.GL_RGBA16F
        )
        GLES31.glUniform2i(GLES31.glGetUniformLocation(program, "uImageSize"), width, height)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(program, "uScale"), scale)
        val varf = sqrt(2.0f + 2.0f * 4.0f * 4.0f + 6.0f * 6.0f) / 16.0f
        val sigmaBand = varf.pow(scale)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uInvSigma2"),
            1.0f / (sigmaBand * sigmaBand)
        )
        dispatchDenoiseImage(width, height, "DenoiseProfile decompose $scale")
    }

    private fun dispatchDenoiseSynthesize(
        coarseTextureId: Int,
        detailTextureId: Int,
        outputTextureId: Int,
        width: Int,
        height: Int,
        threshold: FloatArray
    ) {
        val program = denoiseSynthesizeProgram
        GLES31.glUseProgram(program)
        bindComputeSampler(program, "uCoarse", 0, coarseTextureId)
        bindComputeSampler(program, "uDetail", 1, detailTextureId)
        GLES31.glBindImageTexture(
            2,
            outputTextureId,
            0,
            false,
            0,
            GLES31.GL_WRITE_ONLY,
            GLES31.GL_RGBA16F
        )
        GLES31.glUniform2i(GLES31.glGetUniformLocation(program, "uImageSize"), width, height)
        GLES31.glUniform4fv(GLES31.glGetUniformLocation(program, "uThreshold"), 1, threshold, 0)
        GLES31.glUniform4f(GLES31.glGetUniformLocation(program, "uBoost"), 1f, 1f, 1f, 1f)
        dispatchDenoiseImage(width, height, "DenoiseProfile synthesize")
    }

    private fun dispatchDenoiseBacktransform(
        inputTextureId: Int,
        outputTextureId: Int,
        width: Int,
        height: Int,
        params: DenoiseProfileParams
    ) {
        val program = denoiseBacktransformY0U0V0Program
        GLES31.glUseProgram(program)
        bindComputeSampler(program, "uInput", 0, inputTextureId)
        GLES31.glBindImageTexture(
            1,
            outputTextureId,
            0,
            false,
            0,
            GLES31.GL_WRITE_ONLY,
            GLES31.GL_RGBA16F
        )
        setDenoiseCommonUniforms(program, width, height, params)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uBias"),
            params.bias - 0.5f * ln(params.scale)
        )
        GLES31.glUniformMatrix3fv(
            GLES31.glGetUniformLocation(program, "uToRgb"),
            1, false, matrix3RowMajorToGlColumnMajor(params.toRgb), 0
        )
        dispatchDenoiseImage(width, height, "DenoiseProfile backtransform")
    }

    private fun setDenoiseCommonUniforms(
        program: Int,
        width: Int,
        height: Int,
        params: DenoiseProfileParams
    ) {
        GLES31.glUniform2i(GLES31.glGetUniformLocation(program, "uImageSize"), width, height)
        GLES31.glUniform4fv(GLES31.glGetUniformLocation(program, "uA"), 1, params.aa, 0)
        GLES31.glUniform4fv(GLES31.glGetUniformLocation(program, "uP"), 1, params.p, 0)
        GLES31.glUniform4fv(GLES31.glGetUniformLocation(program, "uB"), 1, params.bb, 0)
        GLES31.glUniform4fv(GLES31.glGetUniformLocation(program, "uWb"), 1, params.wb, 0)
    }

    private fun computeDenoiseProfileThreshold(
        textureId: Int,
        width: Int,
        height: Int,
        scale: Int,
        params: DenoiseProfileParams
    ): FloatArray {
        val groupsX = roundUp(
            width,
            DenoiseProfileShaders.REDUCE_FIRST_LOCAL_X
        ) / DenoiseProfileShaders.REDUCE_FIRST_LOCAL_X
        val groupsY = roundUp(
            height,
            DenoiseProfileShaders.REDUCE_FIRST_LOCAL_Y
        ) / DenoiseProfileShaders.REDUCE_FIRST_LOCAL_Y
        val bWidth = groupsX * DenoiseProfileShaders.REDUCE_FIRST_LOCAL_X
        val bHeight = groupsY * DenoiseProfileShaders.REDUCE_FIRST_LOCAL_Y

        GLES31.glUseProgram(denoiseReduceFirstProgram)
        bindComputeSampler(denoiseReduceFirstProgram, "uInput", 0, textureId)
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, denoiseReduceMBufferId)
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(denoiseReduceFirstProgram, "uImageSize"),
            width,
            height
        )
        GLES31.glDispatchCompute(groupsX, groupsY, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)
        checkGlError("DenoiseProfile reduce_first")

        GLES31.glUseProgram(denoiseReduceSecondProgram)
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, denoiseReduceMBufferId)
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, denoiseReduceRBufferId)
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(denoiseReduceSecondProgram, "uLength"),
            denoiseReduceBufSize
        )
        GLES31.glDispatchCompute(denoiseReduceSize, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT)
        checkGlError("DenoiseProfile reduce_second")

        val sumY2 = readDenoiseReduceResult(denoiseReduceSize)
        val npixels = (width * height).toFloat()
        val varf = sqrt(2.0f + 2.0f * 4.0f * 4.0f + 6.0f * 6.0f) / 16.0f
        val sigmaBand = varf.pow(scale)
        val sb2 = sigmaBand * sigmaBand
        val varY0 = sumY2[0] / (npixels - 1.0f)
        val varY1 = sumY2[1] / (npixels - 1.0f)
        val varY2 = sumY2[2] / (npixels - 1.0f)
        val stdX0 = sqrt(max(1e-6f, varY0 - sb2))
        val stdX1 = sqrt(max(1e-6f, varY1 - sb2))
        val stdX2 = sqrt(max(1e-6f, varY2 - sb2))

        val bandIndex =
            DenoiseProfileShaders.BANDS - (scale + (DenoiseProfileShaders.BANDS - params.maxScale) + 1)
        val forceY0 = 0.5f
        val forceU0V0 = 0.5f
        val adjY = 8.0f * forceY0 * forceY0 * 4.0f
        val adjC = 8.0f * forceU0V0 * forceU0V0 * 4.0f
        return floatArrayOf(
            adjY * sb2 / stdX0,
            adjC * sb2 / stdX1,
            adjC * sb2 / stdX2,
            0.0f
        ).also {
            PLog.d(
                TAG,
                "DenoiseProfile threshold scale=$scale band=$bandIndex thrs=${it.contentToString()} b=${bWidth}x$bHeight"
            )
        }
    }

    private fun readDenoiseReduceResult(reduceSize: Int): FloatArray {
        val byteCount = reduceSize * 4 * 4
        val out = FloatArray(4)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, denoiseReduceRBufferId)
        val mapped = GLES31.glMapBufferRange(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
            byteCount,
            GLES31.GL_MAP_READ_BIT
        ) as? ByteBuffer
        if (mapped != null) {
            mapped.order(ByteOrder.nativeOrder())
            val floats = mapped.asFloatBuffer()
            for (i in 0 until reduceSize) {
                out[0] += floats.get(i * 4)
                out[1] += floats.get(i * 4 + 1)
                out[2] += floats.get(i * 4 + 2)
            }
            GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
        } else {
            PLog.w(TAG, "DenoiseProfile reduce result map failed")
        }
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
        return out
    }

    private fun bindComputeSampler(program: Int, name: String, unit: Int, textureId: Int) {
        GLES31.glActiveTexture(GLES31.GL_TEXTURE0 + unit)
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureId)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(program, name), unit)
    }

    private fun dispatchDenoiseImage(width: Int, height: Int, tag: String) {
        GLES31.glDispatchCompute((width + 15) / 16, (height + 15) / 16, 1)
        GLES31.glMemoryBarrier(
            GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or
                    GLES31.GL_TEXTURE_FETCH_BARRIER_BIT or
                    GLES31.GL_FRAMEBUFFER_BARRIER_BIT
        )
        checkGlError(tag)
    }

    private fun renderPassthroughToTexture(
        sourceTextureId: Int,
        width: Int,
        height: Int,
        framebufferId: Int
    ) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, framebufferId)
        GLES30.glViewport(0, 0, width, height)
        GLES30.glUseProgram(passthroughProgram)
        val identityMatrix = FloatArray(16)
        GlMatrix.setIdentityM(identityMatrix, 0)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(passthroughProgram, "uTexMatrix"),
            1,
            false,
            identityMatrix,
            0
        )
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(passthroughProgram, "uTexture"), 0)
        drawQuad(passthroughProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        checkGlError("DenoiseProfile passthrough")
    }

    private fun roundUp(value: Int, multiple: Int): Int {
        return ((value + multiple - 1) / multiple) * multiple
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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )

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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )

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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )

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

    private fun hasValidLensShadingMap(metadata: RawMetadata): Boolean {
        val map = metadata.lensShadingMap ?: return false
        val width = metadata.lensShadingMapWidth
        val height = metadata.lensShadingMapHeight
        return width > 0 && height > 0 && map.size >= width * height * 4
    }

    private fun bindLensShadingForRcdPopulate(metadata: RawMetadata) {
        val enabled = hasValidLensShadingMap(metadata)
        GLES31.glActiveTexture(GLES31.GL_TEXTURE0 + RCD_LENS_SHADING_TEXTURE_UNIT)
        if (enabled) {
            uploadLensShadingTexture(metadata)
            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, lensShadingTextureId)
        } else {
            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, 0)
        }
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(rcdPopulateProgram, "uLensShadingMap"),
            RCD_LENS_SHADING_TEXTURE_UNIT
        )
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(rcdPopulateProgram, "uLensShadingEnabled"),
            if (enabled) 1 else 0
        )
        GLES31.glUniform2f(
            GLES31.glGetUniformLocation(rcdPopulateProgram, "uLensShadingMapSize"),
            metadata.lensShadingMapWidth.toFloat(),
            metadata.lensShadingMapHeight.toFloat()
        )
        val grid = metadata.lensShadingMapGrid
        val usesDngGrid = enabled && grid != null && grid.size >= 4
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(rcdPopulateProgram, "uLensShadingUsesDngGrid"),
            if (usesDngGrid) 1 else 0
        )
        GLES31.glUniform4f(
            GLES31.glGetUniformLocation(rcdPopulateProgram, "uLensShadingGrid"),
            grid?.getOrElse(0) { 0f } ?: 0f,
            grid?.getOrElse(1) { 0f } ?: 0f,
            grid?.getOrElse(2) { 1f } ?: 1f,
            grid?.getOrElse(3) { 1f } ?: 1f
        )
    }

    private fun logGlResourceLimits() {
        val value = IntArray(1)
        GLES30.glGetIntegerv(GLES30.GL_MAX_TEXTURE_IMAGE_UNITS, value, 0)
        val textureImageUnits = value[0]
        GLES30.glGetIntegerv(GLES31.GL_MAX_IMAGE_UNITS, value, 0)
        val imageUnits = value[0]
        GLES30.glGetIntegerv(GLES31.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, value, 0)
        val ssboBindings = value[0]
        GLES30.glGetIntegerv(GLES31.GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, value, 0)
        val computeSsboBlocks = value[0]
        PLog.d(
            TAG,
            "GL limits: textureImageUnits=$textureImageUnits imageUnits=$imageUnits " +
                    "ssboBindings=$ssboBindings computeSsboBlocks=$computeSsboBlocks"
        )
    }

    private fun createDummyShadingTexture(): Int {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )

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
            GLES30.glDeleteFramebuffers(
                2,
                intArrayOf(demosaicFramebufferId, linearOutputFramebufferId),
                0
            )
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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, demosaicFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, demosaicTextureId, 0
        )

        // 分配并配置第二个 Immutable 纹理
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, linearOutputTextureId)
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA16F, width, height)
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )

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

    private fun setupLinearMeteringFramebuffer(width: Int, height: Int) {
        if (linearMeteringWidth == width && linearMeteringHeight == height && linearMeteringFramebufferId != 0) {
            return
        }

        if (linearMeteringTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(linearMeteringTextureId), 0)
        }
        if (linearMeteringFramebufferId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(linearMeteringFramebufferId), 0)
        }

        linearMeteringWidth = width
        linearMeteringHeight = height

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        linearMeteringTextureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, linearMeteringTextureId)
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA16F, width, height)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )

        val framebuffers = IntArray(1)
        GLES30.glGenFramebuffers(1, framebuffers, 0)
        linearMeteringFramebufferId = framebuffers[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, linearMeteringFramebufferId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            linearMeteringTextureId,
            0
        )
        checkGlError("setupLinearMeteringFramebuffer")
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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )

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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )
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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_R,
            GLES30.GL_CLAMP_TO_EDGE
        )
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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )
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

        val rgbaValues =
            FloatArray(table.hueDivisions * table.satDivisions * table.valueDivisions * 4)
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
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_MIN_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_MAG_FILTER,
            GLES30.GL_NEAREST
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_R,
            GLES30.GL_CLAMP_TO_EDGE
        )
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
            val textureId =
                uploadDcp3DTexture({ dcpHueSatTextureId }, { dcpHueSatTextureId = it }, hueSatMap)
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
            val textureId = uploadDcp3DTexture(
                { dcpLookTableTextureId },
                { dcpLookTableTextureId = it },
                lookTable
            )
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

    private fun uploadSpectralFilmTexture(lut: SpectralFilmLut): Int {
        val key = "${lut.sourceKey}:${lut.size}:${lut.values.size}"
        if (spectralFilmTextureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            spectralFilmTextureId = textures[0]
            spectralFilmTextureKey = null
        }
        if (spectralFilmTextureKey == key) {
            return spectralFilmTextureId
        }

        val buffer = ByteBuffer.allocateDirect(lut.values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(lut.values)
        buffer.position(0)

        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, spectralFilmTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_R,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexImage3D(
            GLES30.GL_TEXTURE_3D,
            0,
            GLES30.GL_RGBA16F,
            lut.size,
            lut.size,
            lut.size,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_FLOAT,
            buffer
        )
        spectralFilmTextureKey = key
        PLog.d(
            TAG,
            "Uploaded spectral film LUT: ${lut.name}, type=${lut.type}, refLight=${lut.referenceIlluminant}, viewLight=${lut.viewingIlluminant}"
        )
        checkGlError("uploadSpectralFilmTexture")
        return spectralFilmTextureId
    }

    private fun bindSpectralFilmCombinedResource(lut: SpectralFilmLut?) {
        GLES30.glUniform1i(GLES30.glGetUniformLocation(combinedProgram, "uSpectralFilmTexture"), 6)
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uSpectralFilmSize"),
            lut?.size ?: 1
        )
        GLES30.glActiveTexture(GLES30.GL_TEXTURE6)
        if (lut != null) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, uploadSpectralFilmTexture(lut))
        } else {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, ensureDummyDcp3DTexture())
        }
        checkGlError("bindSpectralFilmCombinedResource")
    }

    private fun uploadAgxTexture(lut: AgxLut): Int {
        val key = "${lut.sourceKey}:${lut.rgbaFloatBuffer.capacity()}"
        if (agxLutTextureId == 0) {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            agxLutTextureId = textures[0]
            agxLutTextureKey = null
        }
        if (agxLutTextureKey == key) {
            return agxLutTextureId
        }

        val buffer = lut.rgbaFloatBuffer.duplicate().apply { position(0) }

        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, agxLutTextureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_3D,
            GLES30.GL_TEXTURE_WRAP_R,
            GLES30.GL_CLAMP_TO_EDGE
        )
        GLES30.glTexImage3D(
            GLES30.GL_TEXTURE_3D,
            0,
            GLES30.GL_RGBA16F,
            lut.size,
            lut.size,
            lut.size,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_FLOAT,
            buffer
        )
        agxLutTextureKey = key
        PLog.d(TAG, "Uploaded AgX LUT: ${lut.name}, size=${lut.size}")
        checkGlError("uploadAgxTexture")
        return agxLutTextureId
    }

    private fun bindAgxCombinedResource(agxLut: AgxLut?) {
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uAgxBaseSrgbTexture"),
            AGX_BASE_SRGB_TEXTURE_UNIT
        )
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uAgxLutSize"),
            agxLut?.size ?: 1
        )
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + AGX_BASE_SRGB_TEXTURE_UNIT)
        if (agxLut != null) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, uploadAgxTexture(agxLut))
        } else {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, ensureDummyDcp3DTexture())
        }
        checkGlError("bindAgxCombinedResource")
    }

    /**
     * Combined Processing Pass: ToneMap + LUT + Sharpening
     */
    private fun renderCombinedPass(
        metadata: RawMetadata,
        inputTextureId: Int = demosaicTextureId,
        dcpRenderPlan: DcpRenderPlan? = null,
        spectralFilmLut: SpectralFilmLut? = null,
        agxLut: AgxLut? = null,
        colorEngine: RawColorEngine = RawColorEngine.AdobeCurve,
        shadowsHighlightsParams: ShadowsHighlightsParams = ShadowsHighlightsParams.NEUTRAL,
        viewportWidth: Int = metadata.width,
        viewportHeight: Int = metadata.height
    ) {
        val baseCurve = dcpRenderPlan?.toneCurveLut ?: ACR3Curve.samples()
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
        uploadCurveTexture(baseCurve)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, curveTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(combinedProgram, "uCurveTexture"), 1)
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(combinedProgram, "uCurveSize"),
            baseCurve.size.toFloat()
        )
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uCurveEnabled"),
            1
        )
        GLES30.glUniform2f(
            GLES30.glGetUniformLocation(combinedProgram, "uTexelSize"),
            1.0f / maxOf(1, viewportWidth).toFloat(),
            1.0f / maxOf(1, viewportHeight).toFloat()
        )
        bindShadowsHighlightsUniforms(shadowsHighlightsParams)
        checkGlError("renderCombinedPass base uniforms")

        bindDcpCombinedResources(dcpRenderPlan)
        bindSpectralFilmCombinedResource(
            if (colorEngine == RawColorEngine.SpectralFilm) spectralFilmLut else null
        )
        bindAgxCombinedResource(
            agxLut = if (colorEngine == RawColorEngine.AgX) agxLut else null
        )
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(combinedProgram, "uRawColorEngine"),
            colorEngine.shaderId
        )

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

    private fun bindShadowsHighlightsUniforms(params: ShadowsHighlightsParams) {
        val highlightsLocation = GLES30.glGetUniformLocation(combinedProgram, "uHighlights")
        val shadowsLocation = GLES30.glGetUniformLocation(combinedProgram, "uShadows")
        ShadowsHighlightsShader.bindUniformLocations(
            highlightsLocation = highlightsLocation,
            shadowsLocation = shadowsLocation,
            highlights = params.highlights,
            shadows = params.shadows
        )
        if (!loggedShadowsHighlightsUniforms) {
            loggedShadowsHighlightsUniforms = true
            PLog.d(
                TAG,
                "RAW Shadows/Highlights uniforms: " +
                    "uHighlightsLoc=$highlightsLocation uShadowsLoc=$shadowsLocation " +
                    "highlights=${params.highlights} shadows=${params.shadows} " +
                    "curveWhitePoint=${params.curveWhitePoint}"
            )
        }
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(combinedProgram, "uCurveWhitePoint"),
            params.curveWhitePoint
        )
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(combinedProgram, "uCurveShoulderStart"),
            0.4f
        )
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

        val gamutToXyzD50 = if (isD50WhitePoint(xw, yw)) {
            gamutToXyzNative
        } else {
            multiplyMatrix3x3(BRADFORD_D65_TO_D50, gamutToXyzNative)
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

    private fun renderLinearRcdPass(
        metadata: RawMetadata,
        sourceTextureId: Int,
        targetFramebufferId: Int,
        viewportWidth: Int,
        viewportHeight: Int,
        rawExposureCompensation: Float,
        rawBlackPointCorrection: Float,
        rawWhitePointCorrection: Float,
        dcpRenderPlan: DcpRenderPlan?,
        label: String
    ) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, targetFramebufferId)
        checkGlError("$label setup framebuffer")

        GLES30.glUseProgram(linearRcdProgram)
        GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(linearRcdProgram, "uDemosaickedTexture"), 0)

        val transposedCCM = transposeMatrix3x3(
            dcpRenderPlan?.colorCorrectionMatrix ?: metadata.colorCorrectionMatrix
        )
        GLES30.glUniformMatrix3fv(
            GLES30.glGetUniformLocation(linearRcdProgram, "uColorCorrectionMatrix"),
            1,
            false,
            transposedCCM,
            0
        )
        val exposureGain = computeLinearExposureGain(metadata, rawExposureCompensation, dcpRenderPlan)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(linearRcdProgram, "uExposureGain"), exposureGain)

        val blackPoint = rawBlackPointCorrection.coerceIn(0f, 0.99f)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(linearRcdProgram, "uBlackPoint"), blackPoint)
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(linearRcdProgram, "uWhitePoint"),
            (1f + rawWhitePointCorrection).coerceAtLeast(blackPoint + 0.01f)
        )

        val identityMatrix = FloatArray(16)
        GlMatrix.setIdentityM(identityMatrix, 0)
        GLES30.glUniformMatrix4fv(
            GLES30.glGetUniformLocation(linearRcdProgram, "uTexMatrix"),
            1,
            false,
            identityMatrix,
            0
        )

        drawQuad(linearRcdProgram)
        checkGlError("$label drawQuad")
    }

    private suspend fun resolveRawAutoExposureEv(
        context: Context,
        metadata: RawMetadata,
        sourceTextureId: Int,
        rawBlackPointCorrection: Float,
        rawWhitePointCorrection: Float,
        dcpRenderPlan: DcpRenderPlan?,
        meteringCompensationEv: Float,
        useDepthWeighting: Boolean = true
    ): MeteringSystem.MeteringResult {
        val meteringWidth = minOf(metadata.width, 256).coerceAtLeast(1)
        val meteringHeight = minOf(metadata.height, 256).coerceAtLeast(1)
        return try {
            setupLinearMeteringFramebuffer(meteringWidth, meteringHeight)
            renderLinearRcdPass(
                metadata = metadata,
                sourceTextureId = sourceTextureId,
                targetFramebufferId = linearMeteringFramebufferId,
                viewportWidth = meteringWidth,
                viewportHeight = meteringHeight,
                rawExposureCompensation = 0f,
                rawBlackPointCorrection = rawBlackPointCorrection,
                rawWhitePointCorrection = rawWhitePointCorrection,
                dcpRenderPlan = dcpRenderPlan,
                label = "LinearMeteringPass"
            )
            var bitmap: Bitmap? = null
            var depthMap: Bitmap? = null
            var depthTextureId = 0
            try {
                if (useDepthWeighting) {
                    val depthPreviewBuffer = LargeDirectBuffer.allocate(
                        meteringWidth.toLong() * meteringHeight.toLong() * 4L * 2L,
                        "RAW auto exposure depth preview"
                    ) ?: return MeteringSystem.MeteringResult.EMPTY
                    try {
                        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, linearMeteringFramebufferId)
                        GLES30.glReadPixels(
                            0,
                            0,
                            meteringWidth,
                            meteringHeight,
                            GLES30.GL_RGBA,
                            GLES30.GL_HALF_FLOAT,
                            depthPreviewBuffer
                        )
                        checkGlError("resolveRawAutoExposureEv depth preview")
                        depthPreviewBuffer.position(0)
                        bitmap = createLinearMeteringPreviewBitmap(
                            depthPreviewBuffer.asShortBuffer(),
                            meteringWidth,
                            meteringHeight
                        )
                    } finally {
                        LargeDirectBuffer.free(depthPreviewBuffer)
                    }
                    depthMap = SharedDepthEstimator.estimateDepth(context, bitmap)
                    depthTextureId = depthMap?.let {
                        uploadRawAutoExposureDepthTexture(it)
                    } ?: 0
                    if (depthTextureId == 0) {
                        PLog.w(TAG, "RAW AE depth weighting skipped: depth texture unavailable")
                    }
                }

                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
                GLES31.glMemoryBarrier(
                    GLES31.GL_FRAMEBUFFER_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT
                )
                val meteringResult = resolveRawAutoExposureEvGpu(
                    linearTextureId = linearMeteringTextureId,
                    width = meteringWidth,
                    height = meteringHeight,
                    depthTextureId = depthTextureId,
                    baselineExposure = metadata.baselineExposure,
                    meteringCompensationEv = meteringCompensationEv
                ) ?: MeteringSystem.MeteringResult.EMPTY

                if (depthMap != null && !depthMap.isRecycled) {
                    depthMap.recycle()
                }
                if (bitmap != null && !bitmap.isRecycled) {
                    bitmap.recycle()
                }
                meteringResult
            } finally {
                if (depthTextureId != 0) {
                    GLES30.glDeleteTextures(1, intArrayOf(depthTextureId), 0)
                }
                if (depthMap != null && !depthMap.isRecycled) {
                    depthMap.recycle()
                }
                if (bitmap != null && !bitmap.isRecycled) {
                    bitmap.recycle()
                }
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to resolve RAW auto exposure", e)
            MeteringSystem.MeteringResult.EMPTY
        }
    }

    private fun resolveRawAutoExposureEvGpu(
        linearTextureId: Int,
        width: Int,
        height: Int,
        depthTextureId: Int,
        baselineExposure: Float,
        meteringCompensationEv: Float
    ): MeteringSystem.MeteringResult? {
        if (rawAeBaseProgram == 0 || rawAeMertensProgram == 0) {
            PLog.e(
                TAG,
                "RAW AE GPU programs unavailable: base=$rawAeBaseProgram mertens=$rawAeMertensProgram"
            )
            return null
        }

        val baseStats = dispatchRawAeBaseStats(
            linearTextureId = linearTextureId,
            width = width,
            height = height,
            depthTextureId = depthTextureId
        ) ?: return null
        val plan = MeteringSystem.prepareGpuLinearRawAutoExposure(
            stats = baseStats,
            baselineExposure = baselineExposure,
            meteringCompensationEv = meteringCompensationEv,
            tag = "Linear RAW AE GPU"
        ) ?: return null
        val toneStats = dispatchRawAeMertensStats(
            linearTextureId = linearTextureId,
            width = width,
            height = height,
            plan = plan
        ) ?: return null
        return MeteringSystem.finishGpuLinearRawAutoExposure(plan, toneStats)
    }

    private fun dispatchRawAeBaseStats(
        linearTextureId: Int,
        width: Int,
        height: Int,
        depthTextureId: Int
    ): MeteringSystem.GpuRawAutoExposureBaseStats? {
        val groupsX = roundUp(width, RAW_AE_LOCAL_SIZE_X) / RAW_AE_LOCAL_SIZE_X
        val groupsY = roundUp(height, RAW_AE_LOCAL_SIZE_Y) / RAW_AE_LOCAL_SIZE_Y
        val groupCount = (groupsX * groupsY).coerceAtLeast(1)
        val histogramByteCount = groupCount * RAW_AE_HISTOGRAM_BINS * 4
        val baseStatsByteCount = groupCount * 4 * 4
        val buffers = IntArray(2)
        GLES31.glGenBuffers(2, buffers, 0)
        return try {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[0])
            GLES31.glBufferData(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                histogramByteCount,
                null,
                GLES31.GL_DYNAMIC_READ
            )
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                RAW_AE_HISTOGRAM_BINDING,
                buffers[0]
            )

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[1])
            GLES31.glBufferData(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                baseStatsByteCount,
                null,
                GLES31.GL_DYNAMIC_READ
            )
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                RAW_AE_BASE_STATS_BINDING,
                buffers[1]
            )
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

            GLES31.glUseProgram(rawAeBaseProgram)
            bindComputeSampler(rawAeBaseProgram, "uLinearTexture", 0, linearTextureId)
            bindComputeSampler(rawAeBaseProgram, "uDepthTexture", 1, depthTextureId)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rawAeBaseProgram, "uImageSize"),
                width,
                height
            )
            GLES31.glUniform1i(GLES31.glGetUniformLocation(rawAeBaseProgram, "uGroupsX"), groupsX)
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rawAeBaseProgram, "uUseDepthWeight"),
                if (depthTextureId != 0) 1 else 0
            )
            GLES31.glDispatchCompute(groupsX, groupsY, 1)
            GLES31.glMemoryBarrier(
                GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT
            )
            checkGlError("RAW AE base stats")

            readRawAeBaseStats(
                histogramBufferId = buffers[0],
                baseStatsBufferId = buffers[1],
                histogramByteCount = histogramByteCount,
                baseStatsByteCount = baseStatsByteCount,
                groupCount = groupCount
            )
        } finally {
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, RAW_AE_HISTOGRAM_BINDING, 0)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, RAW_AE_BASE_STATS_BINDING, 0)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            GLES31.glDeleteBuffers(2, buffers, 0)
        }
    }

    private fun readRawAeBaseStats(
        histogramBufferId: Int,
        baseStatsBufferId: Int,
        histogramByteCount: Int,
        baseStatsByteCount: Int,
        groupCount: Int
    ): MeteringSystem.GpuRawAutoExposureBaseStats? {
        val histogram = IntArray(RAW_AE_HISTOGRAM_BINS)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, histogramBufferId)
        val mappedHistogram = GLES31.glMapBufferRange(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
            histogramByteCount,
            GLES31.GL_MAP_READ_BIT
        ) as? ByteBuffer
        if (mappedHistogram == null) {
            PLog.e(TAG, "RAW AE base histogram map failed")
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            return null
        }
        mappedHistogram.order(ByteOrder.nativeOrder())
        val histogramInts = mappedHistogram.asIntBuffer()
        for (group in 0 until groupCount) {
            val groupOffset = group * RAW_AE_HISTOGRAM_BINS
            for (bin in 0 until RAW_AE_HISTOGRAM_BINS) {
                histogram[bin] += histogramInts.get(groupOffset + bin)
            }
        }
        GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)

        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, baseStatsBufferId)
        val mappedBaseStats = GLES31.glMapBufferRange(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
            baseStatsByteCount,
            GLES31.GL_MAP_READ_BIT
        ) as? ByteBuffer
        if (mappedBaseStats == null) {
            PLog.e(TAG, "RAW AE base stats map failed")
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            return null
        }
        mappedBaseStats.order(ByteOrder.nativeOrder())
        val baseFloats = mappedBaseStats.asFloatBuffer()
        var weightedLogLumaSum = 0.0
        var weightSum = 0.0
        var sampleCount = 0
        var sanitizedSampleCount = 0
        for (group in 0 until groupCount) {
            val base = group * 4
            weightedLogLumaSum += baseFloats.get(base).toDouble()
            weightSum += baseFloats.get(base + 1).toDouble()
            sampleCount += baseFloats.get(base + 2).toInt()
            sanitizedSampleCount += baseFloats.get(base + 3).toInt()
        }
        GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

        return MeteringSystem.GpuRawAutoExposureBaseStats(
            histogram = histogram,
            histogramLogMin = RAW_AE_HISTOGRAM_LOG_MIN,
            histogramLogMax = RAW_AE_HISTOGRAM_LOG_MAX,
            weightedLogLumaSum = weightedLogLumaSum,
            weightSum = weightSum,
            sampleCount = sampleCount,
            sanitizedSampleCount = sanitizedSampleCount,
            groupCount = groupCount
        )
    }

    private fun dispatchRawAeMertensStats(
        linearTextureId: Int,
        width: Int,
        height: Int,
        plan: MeteringSystem.GpuRawAutoExposurePlan
    ): MeteringSystem.GpuRawAutoExposureToneStats? {
        val sampleStep = plan.sampleStep.coerceAtLeast(1)
        val sampleOffset = (sampleStep / 2).coerceAtMost(minOf(width, height) - 1)
        val sampleGridWidth = sampledGridSize(width, sampleStep, sampleOffset)
        val sampleGridHeight = sampledGridSize(height, sampleStep, sampleOffset)
        if (sampleGridWidth <= 0 || sampleGridHeight <= 0) {
            PLog.e(TAG, "RAW AE Mertens sample grid is empty: ${sampleGridWidth}x$sampleGridHeight")
            return null
        }

        val groupsX = roundUp(sampleGridWidth, RAW_AE_LOCAL_SIZE_X) / RAW_AE_LOCAL_SIZE_X
        val groupsY = roundUp(sampleGridHeight, RAW_AE_LOCAL_SIZE_Y) / RAW_AE_LOCAL_SIZE_Y
        val groupCount = (groupsX * groupsY).coerceAtLeast(1)
        val toneStatsByteCount = groupCount * 2 * 4 * 4
        val buffers = IntArray(1)
        GLES31.glGenBuffers(1, buffers, 0)
        return try {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, buffers[0])
            GLES31.glBufferData(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                toneStatsByteCount,
                null,
                GLES31.GL_DYNAMIC_READ
            )
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                RAW_AE_TONE_STATS_BINDING,
                buffers[0]
            )
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

            GLES31.glUseProgram(rawAeMertensProgram)
            bindComputeSampler(rawAeMertensProgram, "uLinearTexture", 0, linearTextureId)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(rawAeMertensProgram, "uImageSize"),
                width,
                height
            )
            GLES31.glUniform1i(GLES31.glGetUniformLocation(rawAeMertensProgram, "uGroupsX"), groupsX)
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rawAeMertensProgram, "uSampleStep"),
                sampleStep
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(rawAeMertensProgram, "uSampleOffset"),
                sampleOffset
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(rawAeMertensProgram, "uExposureScale"),
                plan.compensatedExposureScale
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(rawAeMertensProgram, "uTargetLuma"),
                plan.targetLuma
            )
            GLES31.glDispatchCompute(groupsX, groupsY, 1)
            GLES31.glMemoryBarrier(
                GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT
            )
            checkGlError("RAW AE Mertens stats")

            readRawAeMertensStats(
                toneStatsBufferId = buffers[0],
                toneStatsByteCount = toneStatsByteCount,
                groupCount = groupCount,
                sampleStep = sampleStep
            )
        } finally {
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, RAW_AE_TONE_STATS_BINDING, 0)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            GLES31.glDeleteBuffers(1, buffers, 0)
        }
    }

    private fun readRawAeMertensStats(
        toneStatsBufferId: Int,
        toneStatsByteCount: Int,
        groupCount: Int,
        sampleStep: Int
    ): MeteringSystem.GpuRawAutoExposureToneStats? {
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, toneStatsBufferId)
        val mappedToneStats = GLES31.glMapBufferRange(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
            toneStatsByteCount,
            GLES31.GL_MAP_READ_BIT
        ) as? ByteBuffer
        if (mappedToneStats == null) {
            PLog.e(TAG, "RAW AE Mertens stats map failed")
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            return null
        }
        mappedToneStats.order(ByteOrder.nativeOrder())
        val toneFloats = mappedToneStats.asFloatBuffer()
        var highlightDeltaEnergySum = 0.0
        var shadowDeltaEnergySum = 0.0
        var highlightWeightSum = 0.0
        var shadowWeightSum = 0.0
        var mertensContrastSum = 0.0
        var mertensSaturationSum = 0.0
        var mertensWellExposednessSum = 0.0
        var sampleCount = 0
        for (group in 0 until groupCount) {
            val base = group * 8
            highlightDeltaEnergySum += toneFloats.get(base).toDouble()
            shadowDeltaEnergySum += toneFloats.get(base + 1).toDouble()
            highlightWeightSum += toneFloats.get(base + 2).toDouble()
            shadowWeightSum += toneFloats.get(base + 3).toDouble()
            mertensContrastSum += toneFloats.get(base + 4).toDouble()
            mertensSaturationSum += toneFloats.get(base + 5).toDouble()
            mertensWellExposednessSum += toneFloats.get(base + 6).toDouble()
            sampleCount += toneFloats.get(base + 7).toInt()
        }
        GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

        return MeteringSystem.GpuRawAutoExposureToneStats(
            highlightDeltaEnergySum = highlightDeltaEnergySum,
            shadowDeltaEnergySum = shadowDeltaEnergySum,
            highlightWeightSum = highlightWeightSum,
            shadowWeightSum = shadowWeightSum,
            mertensContrastSum = mertensContrastSum,
            mertensSaturationSum = mertensSaturationSum,
            mertensWellExposednessSum = mertensWellExposednessSum,
            sampleCount = sampleCount,
            sampleStep = sampleStep,
            groupCount = groupCount
        )
    }

    private fun sampledGridSize(size: Int, sampleStep: Int, sampleOffset: Int): Int {
        if (size <= 0 || sampleStep <= 0 || sampleOffset >= size) {
            return 0
        }
        return ((size - 1 - sampleOffset) / sampleStep) + 1
    }

    private fun uploadRawAutoExposureDepthTexture(depthMap: Bitmap): Int {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        val textureId = textures[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, depthMap, 0)
        checkGlError("uploadRawAutoExposureDepthTexture")
        return textureId
    }

    @SuppressLint("HalfFloat")
    private fun createLinearMeteringPreviewBitmap(
        linearPixels: ShortBuffer,
        width: Int,
        height: Int
    ): Bitmap {
        val argbPixels = IntArray(width * height)
        for (i in argbPixels.indices) {
            val base = i * 4
            val r = linearToSrgb8(Half.toFloat(linearPixels.get(base)))
            val g = linearToSrgb8(Half.toFloat(linearPixels.get(base + 1)))
            val b = linearToSrgb8(Half.toFloat(linearPixels.get(base + 2)))
            argbPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(argbPixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun linearToSrgb8(value: Float): Int {
        val linear = value.coerceIn(0f, 1f)
        val srgb = if (linear <= 0.0031308f) {
            linear * 12.92f
        } else {
            1.055f * linear.pow(1f / 2.4f) - 0.055f
        }
        return (srgb * 255f + 0.5f).toInt().coerceIn(0, 255)
    }

    private fun renderOutputPass(
        rotation: Int,
        width: Int,
        height: Int,
        bounds: Rect,
        sourceTextureId: Int
    ) {
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
    private fun readPixels(
        width: Int,
        height: Int,
        colorSpace: android.graphics.ColorSpace
    ): Bitmap? {
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
        if (readbackPboSize != pixelSize) {
            GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, pixelSize, null, GLES30.GL_STREAM_READ)
            readbackPboSize = if (GLES30.glGetError() == GLES30.GL_NO_ERROR) pixelSize else 0
        }
        val pboReady = readbackPboSize == pixelSize
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
                    createBitmap(
                        width,
                        height,
                        Bitmap.Config.RGBA_F16,
                        colorSpace = colorSpace
                    ).also { bmp ->
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
            PLog.w(
                TAG,
                "PBO glBufferData failed for ${pixelSize}B, falling back to direct readPixels"
            )
        }
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)

        // --- 降级路径：直接读到复用的 native ByteBuffer ---
        val pixelBuffer = try {
            obtainReadbackBuffer(pixelSize)
        } catch (e: OutOfMemoryError) {
            PLog.e(TAG, "OOM allocating pixel buffer ($width x $height, ${pixelSize}B)", e)
            return null
        }

        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
        GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, pixelBuffer)
        pixelBuffer.position(0)
        checkGlError("readPixels direct")
        return try {
            createBitmap(
                width,
                height,
                Bitmap.Config.RGBA_F16,
                colorSpace = colorSpace
            ).also { bitmap ->
                bitmap.copyPixelsFromBuffer(pixelBuffer)
            }
        } catch (e: OutOfMemoryError) {
            PLog.e(TAG, "OOM creating output bitmap ($width x $height)", e)
            null
        }
    }

    private fun obtainReadbackBuffer(pixelSize: Int): ByteBuffer {
        val current = readbackBuffer
        if (current != null && readbackBufferSize >= pixelSize) {
            current.clear()
            current.limit(pixelSize)
            return current
        }
        releaseReadbackBuffer()
        return (com.hinnka.mycamera.utils.DirectBufferAllocator.allocateNative(pixelSize.toLong())
            ?.order(ByteOrder.nativeOrder())
            ?: throw OutOfMemoryError("Failed to allocate native direct buffer")).also {
            readbackBuffer = it
            readbackBufferSize = pixelSize
        }
    }

    private fun releaseReadbackBuffer() {
        readbackBuffer?.let { com.hinnka.mycamera.utils.DirectBufferAllocator.freeNative(it) }
        readbackBuffer = null
        readbackBufferSize = 0
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
        if (chromaDenoiseProgram != 0) GLES30.glDeleteProgram(chromaDenoiseProgram)

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
        if (rawAeBaseProgram != 0) GLES31.glDeleteProgram(rawAeBaseProgram)
        if (rawAeMertensProgram != 0) GLES31.glDeleteProgram(rawAeMertensProgram)

        // darktable denoiseprofile compute programs
        if (denoisePreconditionY0U0V0Program != 0) GLES31.glDeleteProgram(
            denoisePreconditionY0U0V0Program
        )
        if (denoisePreconditionV2Program != 0) GLES31.glDeleteProgram(denoisePreconditionV2Program)
        if (denoiseDecomposeProgram != 0) GLES31.glDeleteProgram(denoiseDecomposeProgram)
        if (denoiseSynthesizeProgram != 0) GLES31.glDeleteProgram(denoiseSynthesizeProgram)
        if (denoiseReduceFirstProgram != 0) GLES31.glDeleteProgram(denoiseReduceFirstProgram)
        if (denoiseReduceSecondProgram != 0) GLES31.glDeleteProgram(denoiseReduceSecondProgram)
        if (denoiseBacktransformY0U0V0Program != 0) GLES31.glDeleteProgram(
            denoiseBacktransformY0U0V0Program
        )
        if (denoiseBacktransformV2Program != 0) GLES31.glDeleteProgram(denoiseBacktransformV2Program)
        // denoiseprofile textures and FBOs
        for (i in 0..1) {
            if (gfTexId[i] != 0) GLES30.glDeleteTextures(1, intArrayOf(gfTexId[i]), 0)
            if (gfFboId[i] != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(gfFboId[i]), 0)
        }
        for (i in denoiseDetailTexIds.indices) {
            if (denoiseDetailTexIds[i] != 0) GLES30.glDeleteTextures(
                1,
                intArrayOf(denoiseDetailTexIds[i]),
                0
            )
            denoiseDetailTexIds[i] = 0
        }
        if (denoiseReduceMBufferId != 0) GLES31.glDeleteBuffers(
            1,
            intArrayOf(denoiseReduceMBufferId),
            0
        )
        if (denoiseReduceRBufferId != 0) GLES31.glDeleteBuffers(
            1,
            intArrayOf(denoiseReduceRBufferId),
            0
        )
        denoiseReduceMBufferId = 0
        denoiseReduceRBufferId = 0

        if (rawTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(rawTextureId), 0)
        if (demosaicTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(demosaicTextureId), 0)
        if (linearOutputTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(linearOutputTextureId),
            0
        )
        if (demosaicFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(demosaicFramebufferId),
            0
        )
        if (linearOutputFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(linearOutputFramebufferId),
            0
        )
        if (combinedTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(combinedTextureId), 0)
        if (combinedFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(combinedFramebufferId),
            0
        )
        if (linearMeteringTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(linearMeteringTextureId),
            0
        )
        if (linearMeteringFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(linearMeteringFramebufferId),
            0
        )
        if (hdrReferenceTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(hdrReferenceTextureId),
            0
        )
        if (hdrReferenceFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(hdrReferenceFramebufferId),
            0
        )
        if (sharpenTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(sharpenTextureId), 0)
        if (sharpenFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(sharpenFramebufferId),
            0
        )
        if (curveTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(curveTextureId), 0)
        if (dcpToneCurveTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(dcpToneCurveTextureId),
            0
        )
        if (dcpHueSatTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(dcpHueSatTextureId), 0)
        if (dcpLookTableTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(dcpLookTableTextureId),
            0
        )
        if (spectralFilmTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(spectralFilmTextureId), 0)
            spectralFilmTextureId = 0
            spectralFilmTextureKey = null
        }
        if (agxLutTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(agxLutTextureId), 0)
            agxLutTextureId = 0
            agxLutTextureKey = null
        }
        if (dummyDcp3DTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(dummyDcp3DTextureId), 0)
        if (dummyDcpToneCurveTextureId != 0) GLES30.glDeleteTextures(
            1,
            intArrayOf(dummyDcpToneCurveTextureId),
            0
        )
        if (outputTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(outputTextureId), 0)

        if (outputFramebufferId != 0) GLES30.glDeleteFramebuffers(
            1,
            intArrayOf(outputFramebufferId),
            0
        )
        if (pboId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(pboId), 0)
            pboId = 0
            readbackPboSize = 0
        }
        releaseReadbackBuffer()

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
