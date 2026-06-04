package com.hinnka.mycamera.lut

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.SurfaceTexture
import android.opengl.*
import com.hinnka.mycamera.livephoto.LivePhotoRecorder
import com.hinnka.mycamera.raw.ColorSpace
import com.hinnka.mycamera.color.TransferCurve
import com.hinnka.mycamera.screencapture.PhantomPipCrop
import com.hinnka.mycamera.camera.MeteringMode
import com.hinnka.mycamera.utils.PLog
import com.hinnka.mycamera.video.VideoLogProfile
import com.hinnka.mycamera.video.VideoRecorder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.hypot

/**
 * LUT 渲染器
 * 
 * 实现 GLSurfaceView.Renderer 接口，负责：
 * 1. 从相机 SurfaceTexture 获取帧
 * 2. 应用 3D LUT 颜色变换
 * 3. 渲染到屏幕
 */
class LutRenderer : GLSurfaceView.Renderer {

    private data class ColorPassLocations(
        val programId: Int,
        val uMVPMatrixLocation: Int,
        val uSTMatrixLocation: Int,
        val uCameraTextureLocation: Int,
        val uLutTextureLocation: Int,
        val uLutSizeLocation: Int,
        val uLutIntensityLocation: Int,
        val uLutEnabledLocation: Int,
        val uLutMaskTypeLocation: Int,
        val uLutCurveLocation: Int,
        val uLutColorSpaceLocation: Int,
        val uVideoLogEnabledLocation: Int,
        val uVideoLogCurveLocation: Int,
        val uVideoColorSpaceLocation: Int,
        val uIsHlgInputLocation: Int,
        val uColorRecipeEnabledLocation: Int,
        val uExposureLocation: Int,
        val uContrastLocation: Int,
        val uSaturationLocation: Int,
        val uTemperatureLocation: Int,
        val uTintLocation: Int,
        val uFadeLocation: Int,
        val uVibranceLocation: Int,
        val uHighlightsLocation: Int,
        val uShadowsLocation: Int,
        val uToneToeLocation: Int,
        val uToneShoulderLocation: Int,
        val uTonePivotLocation: Int,
        val uFilmGrainLocation: Int,
        val uVignetteLocation: Int,
        val uBleachBypassLocation: Int,
        val uChromaticAberrationLocation: Int,
        val uNoiseLocation: Int,
        val uNoiseSeedLocation: Int,
        val uLowResLocation: Int,
        val uAspectRatioLocation: Int,
        val uLchHueAdjustmentsLocation: Int,
        val uLchChromaAdjustmentsLocation: Int,
        val uLchLightnessAdjustmentsLocation: Int,
        val uPrimaryCalibrationMatrixLocation: Int,
        val uSTMatrixFragLocation: Int,
        val uCropRectLocation: Int,
        val uApertureLocation: Int,
        val uFocusPointLocation: Int,
        val uCurveTextureLocation: Int,
        val uCurveEnabledLocation: Int,
        val aPositionLocation: Int,
        val aTexCoordLocation: Int,
    )

    companion object {
        private const val TAG = "LutRenderer"
        private const val LCH_COLOR_BAND_COUNT = 9

        // 属性位置
        private const val POSITION_COMPONENT_COUNT = 2
        private const val TEXTURE_COORD_COMPONENT_COUNT = 2
        private const val BYTES_PER_FLOAT = 4
        private const val BYTES_PER_SHORT = 2
    }

    // 着色器程序 ID
    private var programId: Int = 0

    // 纹理 ID
    private var cameraTextureId: Int = 0
    private var lutTextureId: Int = 0
    private var baselineLutTextureId: Int = 0

    // 缓冲区
    private var vertexBufferId: Int = 0
    private var texCoordBufferId: Int = 0
    private var indexBufferId: Int = 0
    private var pboId: Int = 0
    private var meteringPboId: Int = 0

    // 测光相关纹理和 FBO
    private var meteringFboId: Int = 0
    private var meteringTextureId: Int = 0
    private val METERING_SIZE = 32
    private val meteringExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var captureFboId: Int = 0
    private var captureTextureId: Int = 0
    private var recordFboId: Int = 0
    private var recordTextureId: Int = 0
    private var recordFboWidth: Int = 0
    private var recordFboHeight: Int = 0
    private var passthroughProgramId: Int = 0
    private var uPassMVPMatrixLocation: Int = 0
    private var uPassSTMatrixLocation: Int = 0
    private var uPassCropRectLocation: Int = 0
    private var uPassCameraTextureLocation: Int = 0
    private var aPassPositionLocation: Int = 0
    private var aPassTexCoordLocation: Int = 0
    
    // 深度估计输入采集
    private var depthInputFboId: Int = 0
    private var depthInputTextureId: Int = 0
    private var depthInputPboId: Int = 0
    private val DEPTH_INPUT_SIZE = 256
    private val depthInputBuffer = ByteBuffer.allocateDirect(DEPTH_INPUT_SIZE * DEPTH_INPUT_SIZE * 4)
    private var lastRunDepthInputTime: Long = 0
    var onDepthInputAvailable: ((Bitmap) -> Unit)? = null
    private var aiFocusInputFboId: Int = 0
    private var aiFocusInputTextureId: Int = 0
    private val aiFocusInputPboIds = IntArray(2)
    private val aiFocusInputPboFences = LongArray(2)
    private var aiFocusInputPboIndex = 0
    @Volatile
    var isAiFocusBusy = false
    private val AI_FOCUS_INPUT_SIZE = 640
    private val aiFocusInputBuffer = ByteBuffer.allocateDirect(AI_FOCUS_INPUT_SIZE * AI_FOCUS_INPUT_SIZE * 4)
    private var lastRunAiFocusInputTime: Long = 0
    var onAiFocusInputAvailable: ((Bitmap) -> Unit)? = null

    // FBO 相关
    private var fboId: Int = 0
    private var fboTextureId: Int = 0
    private var fboWidth: Int = 0
    private var fboHeight: Int = 0
    private var stackFboId: Int = 0
    private var stackTextureId: Int = 0
    private var stackFboWidth: Int = 0
    private var stackFboHeight: Int = 0

    // Copy Shader (FBO -> Screen)
    private var copyProgramId: Int = 0
    private var secondPassProgramId: Int = 0
    private var uCopyTextureLoc: Int = 0
    private var uCopyMVPMatrixLoc: Int = 0
    private var uCopySTMatrixLoc: Int = 0
    private var uCopyCropRectLoc: Int = 0
    private var aCopyPositionLoc: Int = 0
    private var aCopyTexCoordLoc: Int = 0
    private var mainPassLocations: ColorPassLocations? = null
    private var secondPassLocations: ColorPassLocations? = null

    // Uniform 位置
    private var uMVPMatrixLocation: Int = 0
    private var uSTMatrixLocation: Int = 0
    private var uCameraTextureLocation: Int = 0
    private var uLutTextureLocation: Int = 0
    private var uLutSizeLocation: Int = 0
    private var uLutIntensityLocation: Int = 0
    private var uLutEnabledLocation: Int = 0
    private var uLutCurveLocation: Int = 0
    private var uLutColorSpaceLocation: Int = 0
    private var uVideoLogEnabledLocation: Int = 0
    private var uVideoLogCurveLocation: Int = 0
    private var uVideoColorSpaceLocation: Int = 0
    private var uIsHlgInputLocation: Int = 0

    // 是否以 HLG10 动态范围采集（Log LUT 兼容性方案）
    var isHlgInput: Boolean = false

    // 色彩配方 Uniform 位置
    private var uColorRecipeEnabledLocation: Int = 0
    private var uExposureLocation: Int = 0
    private var uContrastLocation: Int = 0
    private var uSaturationLocation: Int = 0
    private var uTemperatureLocation: Int = 0
    private var uTintLocation: Int = 0
    private var uFadeLocation: Int = 0
    private var uVibranceLocation: Int = 0
    private var uHighlightsLocation: Int = 0
    private var uShadowsLocation: Int = 0
    private var uToneToeLocation: Int = 0
    private var uToneShoulderLocation: Int = 0
    private var uTonePivotLocation: Int = 0
    private var uFilmGrainLocation: Int = 0
    private var uVignetteLocation: Int = 0
    private var uBleachBypassLocation: Int = 0
    private var uChromaticAberrationLocation: Int = 0
    private var uNoiseLocation: Int = 0
    private var uNoiseSeedLocation: Int = 0
    private var uLowResLocation: Int = 0
    private var uAspectRatioLocation: Int = 0
    private var uLchHueAdjustmentsLocation: Int = 0
    private var uLchChromaAdjustmentsLocation: Int = 0
    private var uLchLightnessAdjustmentsLocation: Int = 0
    private var uPrimaryCalibrationMatrixLocation: Int = 0
    private var uSTMatrixFragLocation: Int = 0
    private var uCropRectLocation: Int = 0
    private var uApertureLocation: Int = 0
    private var uFocusPointLocation: Int = 0
    private var uCurveTextureLocation: Int = 0
    private var uCurveEnabledLocation: Int = 0

    // 曲线纹理
    private var curveTextureId: Int = 0
    private var baselineCurveTextureId: Int = 0

    @Volatile
    var curveEnabled: Boolean = false

    @Volatile
    var baselineCurveEnabled: Boolean = false

    @Volatile
    var pendingCurveBuffer: java.nio.ByteBuffer? = null

    @Volatile
    var baselinePendingCurveBuffer: java.nio.ByteBuffer? = null

    @Volatile
    var masterCurvePoints: FloatArray? = null

    @Volatile
    var redCurvePoints: FloatArray? = null

    @Volatile
    var greenCurvePoints: FloatArray? = null

    @Volatile
    var blueCurvePoints: FloatArray? = null

    @Volatile
    var baselineRecipeParams: com.hinnka.mycamera.model.ColorRecipeParams =
        com.hinnka.mycamera.model.ColorRecipeParams.DEFAULT

    // HDF 实时预览资源
    private var hdfExtractBlurHProgram: Int = 0
    private var hdfBlurVProgram: Int = 0
    private var hdfCompositeProgram: Int = 0
    private var hdfTexId = IntArray(2)
    private var hdfFboId = IntArray(2)
    private var hdfWidth: Int = 0
    private var hdfHeight: Int = 0
    private var halationExtractBlurHProgram: Int = 0
    private var halationBlurVProgram: Int = 0
    private var halationTexId = IntArray(2)
    private var halationFboId = IntArray(2)
    private var halationWidth: Int = 0
    private var halationHeight: Int = 0
    
    // Bokeh 实时预览资源
    private var bokehProgramId: Int = 0
    private var uBokehInputTexLoc: Int = 0
    private var uBokehDepthTexLoc: Int = 0
    private var uBokehMaxBlurRadiusLoc: Int = 0
    private var uBokehApertureLoc: Int = 0
    private var uBokehFocusDepthLoc: Int = 0
    private var uBokehTexelSizeLoc: Int = 0
    private var aBokehPositionLoc: Int = 0
    private var aBokehTexCoordLoc: Int = 0
    
    private var depthTextureId: Int = 0
    @Volatile
    var depthMap: Bitmap? = null
    private var lastDepthMap: Bitmap? = null
    private var bokehFboId: Int = 0
    private var bokehTextureId: Int = 0
    private var bokehFboWidth: Int = 0
    private var bokehFboHeight: Int = 0
    private var bokehRenderScale: Float = 0.5f // 降采样比例，0.5 代表 1/4 像素量

    // Focus Peaking 实时预览资源
    private var focusPeakingProgramId: Int = 0
    private var uPeakInputTexLoc: Int = 0
    private var uPeakTexelSizeLoc: Int = 0
    private var uPeakThresholdLoc: Int = 0
    private var uPeakColorLoc: Int = 0
    private var aPeakPositionLoc: Int = 0
    private var aPeakTexCoordLoc: Int = 0
    private var focusPeakingFboId: Int = 0
    private var focusPeakingTextureId: Int = 0
    private var focusPeakingFboWidth: Int = 0
    private var focusPeakingFboHeight: Int = 0

    // Attribute 位置
    private var aPositionLocation: Int = 0
    private var aTexCoordLocation: Int = 0

    // SurfaceTexture 变换矩阵
    private val stMatrix = FloatArray(16).apply { Matrix.setIdentityM(this, 0) }

    // MVP 变换矩阵（用于 center crop）
    private val mvpMatrix = FloatArray(16)

    // 相机 SurfaceTexture
    private var surfaceTexture: SurfaceTexture? = null
    private var frameAvailable = false
    private val frameSyncObject = Any()

    // 标记 Surface 是否已创建（GL 上下文是否可用）
    private var surfaceReady = false
    @Volatile
    private var renderingPaused = false

    // 待处理的 LUT 配置（Surface 创建前设置的 LUT）
    private var pendingLutConfig: LutConfig? = null
    private var pendingBaselineLutConfig: LutConfig? = null

    // LUT 配置
    private var currentLutConfig: LutConfig? = null
    private var currentBaselineLutConfig: LutConfig? = null
    private var lutSize: Float = 32f
    private var baselineLutSize: Float = 32f

    // LUT 强度 (0.0 - 1.0)
    @Volatile
    var lutIntensity: Float = 1.0f

    // LUT 是否启用
    @Volatile
    var lutEnabled: Boolean = false

    @Volatile
    var isAutoFocus: Boolean = true

    @Volatile
    var focusPeakingEnabled: Boolean = true

    @Volatile
    var baselineLutEnabled: Boolean = false

    // 色彩配方参数
    @Volatile
    var colorRecipeEnabled: Boolean = false

    @Volatile
    var baselineColorRecipeEnabled: Boolean = false

    @Volatile
    var focusPoint: PointF? = null

    @Volatile
    var meteringMode: MeteringMode = MeteringMode.CENTER_WEIGHTED

    @Volatile
    var aperture: Float = 0f
    private val cropRect = floatArrayOf(0f, 0f, 1f, 1f)

    @Volatile
    var meteringEnabled: Boolean = true

    @Volatile
    var exposure: Float = 0f // -2.0 ~ +2.0

    @Volatile
    var contrast: Float = 1f // 0.5 ~ 1.5

    @Volatile
    var saturation: Float = 1f // 0.0 ~ 2.0

    @Volatile
    var temperature: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var tint: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var fade: Float = 0f // 0.0 ~ 1.0

    @Volatile
    var vibrance: Float = 1f // 0.0 ~ 2.0

    @Volatile
    var highlights: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var shadows: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var toneToe: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var toneShoulder: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var tonePivot: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var filmGrain: Float = 0f // 0.0 ~ 1.0

    @Volatile
    var vignette: Float = 0f // -1.0 ~ +1.0

    @Volatile
    var bleachBypass: Float = 0f // 0.0 ~ 1.0

    @Volatile
    var chromaticAberration: Float = 0f // 0.0 ~ 1.0

    @Volatile
    var noise: Float = 0f // 0.0 ~ 1.0

    @Volatile
    var lowRes: Float = 0f // 0.0 ~ 1.0

    @Volatile
    var halation: Float = 0f // 0.0 ~ 1.0

    @Volatile
    var redHalation: Float = 0f // 0.0 ~ 1.0

    @Volatile var primaryRedHue: Float = 0f
    @Volatile var primaryRedSaturation: Float = 0f
    @Volatile var primaryRedLightness: Float = 0f
    @Volatile var primaryGreenHue: Float = 0f
    @Volatile var primaryGreenSaturation: Float = 0f
    @Volatile var primaryGreenLightness: Float = 0f
    @Volatile var primaryBlueHue: Float = 0f
    @Volatile var primaryBlueSaturation: Float = 0f
    @Volatile var primaryBlueLightness: Float = 0f

    private val lchHueAdjustments = FloatArray(LCH_COLOR_BAND_COUNT)
    private val lchChromaAdjustments = FloatArray(LCH_COLOR_BAND_COUNT)
    private val lchLightnessAdjustments = FloatArray(LCH_COLOR_BAND_COUNT)

    // 渲染尺寸
    private var viewportWidth: Int = 0
    private var viewportHeight: Int = 0
    private var photoCaptureWidth: Int = 0
    private var photoCaptureHeight: Int = 0
    private var lastLoggedSpatialEffectScale: Float = -1f

    // 历史高光点记录（用于增加稳定性，减少跳动）
    private var lastBestX = -1
    private var lastBestY = -1

    // 预览尺寸
    private var previewWidth: Int = 1920
    private var previewHeight: Int = 1080
    private var sensorOrientation: Int = 0
    private var calibrationOffset: Int = 0
    private var deviceRotation: Int = 0
    private var lensFacing: Int = 1 // CameraCharacteristics.LENS_FACING_BACK

    // 回调
    var onSurfaceTextureAvailable: ((SurfaceTexture) -> Unit)? = null
    var onRequestRender: (() -> Unit)? = null
    var onPreviewFrameCaptured: ((Bitmap) -> Unit)? = null
    var onHistogramUpdated: ((IntArray) -> Unit)? = null
    var onMeteringUpdated: ((Double, Double) -> Unit)? = null
    var onHighlightPointUpdated: ((Float, Float) -> Unit)? = null

    // Live Photo 录制器
    var livePhotoRecorder: LivePhotoRecorder? = null
    @Volatile
    var videoRecorder: VideoRecorder? = null
    @Volatile
    var videoLogProfile: VideoLogProfile = VideoLogProfile.OFF
    private var videoRenderStatsWindowStartMs: Long = 0L
    private var videoRenderStatsFrames: Int = 0

    // 预览帧捕获标志
    private var shouldCapturePreview = false
    private var captureWidth = 512
    private var captureHeight = 512
    private var captureAspectRatio = 0f
    private val maxCaptureSize = 1080 // 预览图最大尺寸
    private var lastCaptureWidth = 0
    private var lastCaptureHeight = 0

    /**
     * Surface 创建时调用
     */
    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        PLog.d(TAG, "onSurfaceCreated")

        // Context lost or new, reset all resource IDs and cached state
        resetGlResourceState()

        // 设置清屏颜色
        GLES30.glClearColor(0f, 0f, 0f, 1f)

        // 创建着色器程序
        initShaderProgram()

        // 创建顶点缓冲
        initBuffers()

        // 创建相机纹理
        cameraTextureId = GlUtils.createOESTexture()

        // 创建 SurfaceTexture
        surfaceTexture = SurfaceTexture(cameraTextureId).apply {
            setDefaultBufferSize(previewWidth, previewHeight)
            setOnFrameAvailableListener {
                // 先设置标志（需要同步）
                synchronized(frameSyncObject) {
                    frameAvailable = true
                }
                // 在 synchronized 块外调用回调，避免潜在死锁
                onRequestRender?.invoke()
            }
        }

        // 初始化测光 FBO
        initMeteringFbo()

        // 标记 Surface 已创建
        surfaceReady = true

        // 如果有 LUT 配置，现在设置它（恢复由于 Context 丢失失效的纹理）
        (pendingLutConfig ?: currentLutConfig)?.let { config ->
            pendingLutConfig = null
            setLutInternal(config)
        }
        (pendingBaselineLutConfig ?: currentBaselineLutConfig)?.let { config ->
            pendingBaselineLutConfig = null
            setBaselineLutInternal(config)
        }

        // 通知调用者 SurfaceTexture 已准备好
        surfaceTexture?.let { onSurfaceTextureAvailable?.invoke(it) }

        GlUtils.checkGlError("initShaderProgram")
    }

    /**
     * 重置所有 GL 资源 ID 和缓存状态。
     * 当 GL Context 被重新创建时（例如 App 切回前台），旧的资源 ID 已失效。
     * 重置状态可确保在后续渲染中按需重新创建有效资源。
     */
    private fun resetGlResourceState() {
        programId = 0
        secondPassProgramId = 0
        mainPassLocations = null
        secondPassLocations = null
        cameraTextureId = 0
        lutTextureId = 0
        baselineLutTextureId = 0
        vertexBufferId = 0
        texCoordBufferId = 0
        indexBufferId = 0
        pboId = 0
        meteringPboId = 0
        meteringFboId = 0
        meteringTextureId = 0
        captureFboId = 0
        captureTextureId = 0
        recordFboId = 0
        recordTextureId = 0
        recordFboWidth = 0
        recordFboHeight = 0
        passthroughProgramId = 0
        depthInputFboId = 0
        depthInputTextureId = 0
        depthInputPboId = 0
        aiFocusInputFboId = 0
        aiFocusInputTextureId = 0
        aiFocusInputPboIds[0] = 0
        aiFocusInputPboIds[1] = 0
        aiFocusInputPboFences[0] = 0L
        aiFocusInputPboFences[1] = 0L
        aiFocusInputPboIndex = 0
        isAiFocusBusy = false
        fboId = 0
        fboTextureId = 0
        fboWidth = 0
        fboHeight = 0
        stackFboId = 0
        stackTextureId = 0
        stackFboWidth = 0
        stackFboHeight = 0
        copyProgramId = 0
        hdfExtractBlurHProgram = 0
        hdfBlurVProgram = 0
        hdfCompositeProgram = 0
        hdfTexId = IntArray(2)
        hdfFboId = IntArray(2)
        hdfWidth = 0
        hdfHeight = 0
        halationExtractBlurHProgram = 0
        halationBlurVProgram = 0
        halationTexId = IntArray(2)
        halationFboId = IntArray(2)
        halationWidth = 0
        halationHeight = 0
        bokehProgramId = 0
        depthTextureId = 0
        lastDepthMap = null
        bokehFboId = 0
        bokehTextureId = 0
        bokehFboWidth = 0
        bokehFboHeight = 0
        focusPeakingProgramId = 0
        focusPeakingFboId = 0
        focusPeakingTextureId = 0
        focusPeakingFboWidth = 0
        focusPeakingFboHeight = 0
        lastCaptureWidth = 0
        lastCaptureHeight = 0
        viewportWidth = 0
        viewportHeight = 0
        curveTextureId = 0
        baselineCurveTextureId = 0
    }

    /**
     * Surface 尺寸变化时调用
     */
    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        // PLog.d(TAG, "onSurfaceChanged: ${width}x${height}")

        viewportWidth = width
        viewportHeight = height

        GLES30.glViewport(0, 0, width, height)

        initMeteringFbo()

        // 更新 MVP 矩阵以处理 center crop
        updateMVPMatrix()

        GlUtils.checkGlError("onSurfaceChanged")
    }

    private fun initFbo(width: Int, height: Int) {
        if (fboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
            fboId = 0
        }
        if (fboTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(fboTextureId), 0)
            fboTextureId = 0
        }
        if (stackFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(stackFboId), 0)
            stackFboId = 0
        }
        if (stackTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(stackTextureId), 0)
            stackTextureId = 0
        }
        fboWidth = 0
        fboHeight = 0
        stackFboWidth = 0
        stackFboHeight = 0

        val ids = IntArray(1)
        GLES30.glGenFramebuffers(1, ids, 0)
        fboId = ids[0]

        GLES30.glGenTextures(1, ids, 0)
        fboTextureId = ids[0]

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, width, height,
            0, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, fboTextureId, 0
        )

        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            PLog.e(TAG, "FBO init failed: $status")
        }

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        fboWidth = width
        fboHeight = height

        GLES30.glGenFramebuffers(1, ids, 0)
        stackFboId = ids[0]
        GLES30.glGenTextures(1, ids, 0)
        stackTextureId = ids[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, stackTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, width, height,
            0, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, stackFboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, stackTextureId, 0
        )
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        stackFboWidth = width
        stackFboHeight = height
    }

    private fun isMainFboReady(width: Int, height: Int): Boolean {
        return fboId != 0 &&
            fboTextureId != 0 &&
            stackFboId != 0 &&
            stackTextureId != 0 &&
            fboWidth == width &&
            fboHeight == height &&
            stackFboWidth == width &&
            stackFboHeight == height
    }

    private fun hasBaselineLayer(): Boolean {
        return (baselineLutEnabled && currentBaselineLutConfig != null) ||
            baselineColorRecipeEnabled ||
            !baselineRecipeParams.isDefault()
    }

    private fun hasCreativeLayer(): Boolean {
        return (lutEnabled && currentLutConfig != null) || colorRecipeEnabled
    }

    private fun ensureCurveTextureUploaded(
        textureId: Int,
        pendingBuffer: ByteBuffer?
    ): Int {
        var currentTextureId = textureId
        pendingBuffer?.let { buf ->
            if (currentTextureId == 0) {
                val ids = IntArray(1)
                GLES30.glGenTextures(1, ids, 0)
                currentTextureId = ids[0]
            }
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, currentTextureId)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D,
                0,
                GLES30.GL_RGBA8,
                256,
                1,
                0,
                GLES30.GL_RGBA,
                GLES30.GL_UNSIGNED_BYTE,
                buf
            )
        }
        return currentTextureId
    }

    private fun drawColorPass(
        locations: ColorPassLocations,
        targetFboId: Int,
        width: Int,
        height: Int,
        sourceTextureTarget: Int,
        sourceTextureId: Int,
        sourceStMatrix: FloatArray,
        sourceCropRect: FloatArray,
        targetMvpMatrix: FloatArray,
        lutConfig: LutConfig?,
        lutTextureId: Int,
        lutSize: Float,
        lutEnabled: Boolean,
        params: com.hinnka.mycamera.model.ColorRecipeParams,
        curveTextureId: Int,
        curveEnabled: Boolean,
        enableVideoLog: Boolean,
        treatSourceAsHlgInput: Boolean,
        apertureOverride: Float = aperture,
        focusPointOverride: PointF? = focusPoint,
    ) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, targetFboId)
        GLES30.glViewport(0, 0, width, height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(locations.programId)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(sourceTextureTarget, sourceTextureId)
        GLES30.glUniform1i(locations.uCameraTextureLocation, 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(
            GLES30.GL_TEXTURE_3D,
            if (lutEnabled && lutTextureId != 0) lutTextureId else 0
        )
        GLES30.glUniform1i(locations.uLutTextureLocation, 1)

        GLES30.glUniformMatrix4fv(locations.uMVPMatrixLocation, 1, false, targetMvpMatrix, 0)
        GLES30.glUniformMatrix4fv(locations.uSTMatrixLocation, 1, false, sourceStMatrix, 0)
        GLES30.glUniformMatrix4fv(locations.uSTMatrixFragLocation, 1, false, sourceStMatrix, 0)
        GLES30.glUniform4f(
            locations.uCropRectLocation,
            sourceCropRect[0],
            sourceCropRect[1],
            sourceCropRect[2],
            sourceCropRect[3]
        )
        GLES30.glUniform1f(locations.uLutSizeLocation, lutSize)
        GLES30.glUniform1f(locations.uLutIntensityLocation, params.lutIntensity)
        GLES30.glUniform1i(locations.uLutEnabledLocation, if (lutEnabled && lutTextureId != 0) 1 else 0)
        GLES30.glUniform1i(locations.uLutMaskTypeLocation, 0)
        GLES30.glUniform1i(locations.uLutCurveLocation, mapLutCurve(lutConfig?.curve))
        GLES30.glUniform1i(locations.uLutColorSpaceLocation, mapRawColorSpace(lutConfig?.colorSpace ?: ColorSpace.SRGB))
        GLES30.glUniform1i(locations.uVideoLogEnabledLocation, if (enableVideoLog && videoLogProfile.isEnabled) 1 else 0)
        GLES30.glUniform1i(locations.uVideoLogCurveLocation, mapLogCurve(videoLogProfile.logCurve))
        GLES30.glUniform1i(locations.uVideoColorSpaceLocation, mapRawColorSpace(videoLogProfile.colorSpace))
        GLES30.glUniform1i(locations.uIsHlgInputLocation, if (treatSourceAsHlgInput) 1 else 0)

//        PLog.d(TAG, "uIsHlgInputLocation=$treatSourceAsHlgInput")

        val recipeEnabled = !params.isDefault()
        GLES30.glUniform1i(locations.uColorRecipeEnabledLocation, if (recipeEnabled) 1 else 0)
        if (recipeEnabled) {
            GLES30.glUniform1f(locations.uExposureLocation, params.exposure)
            GLES30.glUniform1f(locations.uContrastLocation, params.contrast)
            GLES30.glUniform1f(locations.uSaturationLocation, params.saturation)
            GLES30.glUniform1f(locations.uTemperatureLocation, params.temperature)
            GLES30.glUniform1f(locations.uTintLocation, params.tint)
            GLES30.glUniform1f(locations.uFadeLocation, params.fade)
            GLES30.glUniform1f(locations.uVibranceLocation, params.color)
            GLES30.glUniform1f(locations.uHighlightsLocation, params.highlights)
            GLES30.glUniform1f(locations.uShadowsLocation, params.shadows)
            GLES30.glUniform1f(locations.uToneToeLocation, params.toneToe)
            GLES30.glUniform1f(locations.uToneShoulderLocation, params.toneShoulder)
            GLES30.glUniform1f(locations.uTonePivotLocation, params.tonePivot)
            GLES30.glUniform1f(locations.uFilmGrainLocation, params.filmGrain)
            GLES30.glUniform1f(locations.uVignetteLocation, params.vignette)
            GLES30.glUniform1f(locations.uBleachBypassLocation, params.bleachBypass)
            GLES30.glUniform1f(locations.uNoiseLocation, params.noise)
            GLES30.glUniform1f(locations.uNoiseSeedLocation, (System.currentTimeMillis() % 10000) / 1000f)
            GLES30.glUniform1f(locations.uLowResLocation, params.lowRes)
            GLES30.glUniform1f(locations.uAspectRatioLocation, width.toFloat() / maxOf(1, height).toFloat())
            val lch = buildLchArrays(params)
            val primaryCalibrationMatrix = CameraRawCalibrationMatrix.build(params)
            GLES30.glUniform1fv(locations.uLchHueAdjustmentsLocation, LCH_COLOR_BAND_COUNT, lch.first, 0)
            GLES30.glUniform1fv(locations.uLchChromaAdjustmentsLocation, LCH_COLOR_BAND_COUNT, lch.second, 0)
            GLES30.glUniform1fv(locations.uLchLightnessAdjustmentsLocation, LCH_COLOR_BAND_COUNT, lch.third, 0)
            GLES30.glUniformMatrix3fv(locations.uPrimaryCalibrationMatrixLocation, 1, false, primaryCalibrationMatrix, 0)
        }

        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, if (curveEnabled && curveTextureId != 0) curveTextureId else 0)
        GLES30.glUniform1i(locations.uCurveTextureLocation, 2)
        GLES30.glUniform1i(locations.uCurveEnabledLocation, if (curveEnabled && curveTextureId != 0) 1 else 0)

        GLES30.glUniform1f(locations.uApertureLocation, apertureOverride)
        val fp = focusPointOverride ?: PointF(0.5f, 0.5f)
        GLES30.glUniform2f(locations.uFocusPointLocation, fp.x, 1.0f - fp.y)
        GLES30.glUniform1f(locations.uChromaticAberrationLocation, params.chromaticAberration)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(locations.aPositionLocation)
        GLES30.glVertexAttribPointer(locations.aPositionLocation, POSITION_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(locations.aTexCoordLocation)
        GLES30.glVertexAttribPointer(locations.aTexCoordLocation, TEXTURE_COORD_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, Shaders.DRAW_ORDER.size, GLES30.GL_UNSIGNED_SHORT, 0)

        GLES30.glDisableVertexAttribArray(locations.aPositionLocation)
        GLES30.glDisableVertexAttribArray(locations.aTexCoordLocation)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)
    }

    private fun buildLchArrays(params: com.hinnka.mycamera.model.ColorRecipeParams): Triple<FloatArray, FloatArray, FloatArray> {
        return Triple(
            floatArrayOf(
                params.skinHue, params.redHue, params.orangeHue, params.yellowHue,
                params.greenHue, params.cyanHue, params.blueHue, params.purpleHue, params.magentaHue,
            ),
            floatArrayOf(
                params.skinChroma, params.redChroma, params.orangeChroma, params.yellowChroma,
                params.greenChroma, params.cyanChroma, params.blueChroma, params.purpleChroma, params.magentaChroma,
            ),
            floatArrayOf(
                params.skinLightness, params.redLightness, params.orangeLightness, params.yellowLightness,
                params.greenLightness, params.cyanLightness, params.blueLightness, params.purpleLightness, params.magentaLightness,
            )
        )
    }

    private fun uploadPendingCurveTextures() {
        if (pendingCurveBuffer != null) {
            curveTextureId = ensureCurveTextureUploaded(curveTextureId, pendingCurveBuffer)
            pendingCurveBuffer = null
        }
        if (baselinePendingCurveBuffer != null) {
            baselineCurveTextureId = ensureCurveTextureUploaded(baselineCurveTextureId, baselinePendingCurveBuffer)
            baselinePendingCurveBuffer = null
        }
    }

    /**
     * 绘制帧
     */
    override fun onDrawFrame(gl: GL10?) {
        if (renderingPaused) return
        if (viewportWidth <= 0 || viewportHeight <= 0) return

        // 更新 SurfaceTexture
        var hasFreshCameraFrame = false
        synchronized(frameSyncObject) {
            if (frameAvailable) {
                try {
                    surfaceTexture?.updateTexImage()
                    surfaceTexture?.getTransformMatrix(stMatrix)
                    hasFreshCameraFrame = true
                } catch (e: RuntimeException) {
                    PLog.e(
                        TAG,
                        "updateTexImage failed, surfaceReady=$surfaceReady, cameraTextureId=$cameraTextureId",
                        e
                    )
                }
                frameAvailable = false
            }
        }

        val liveRecorder = livePhotoRecorder
        val activeVideoRecorder = videoRecorder?.takeIf { it.isRecording() }
        if (activeVideoRecorder == null) {
            videoRenderStatsWindowStartMs = 0L
            videoRenderStatsFrames = 0
        }
        val hdfEnabled = halation > 0.001f
        val halationEnabled = redHalation > 0.001f
        val postProcessEffectEnabled = hdfEnabled || halationEnabled
        val bokehNeeded = aperture > 0f && depthMap != null
        val aiFocusInputNeeded = onAiFocusInputAvailable != null && isAutoFocus && !isAiFocusBusy
        val suppressBaselineLayerForVideoLog = videoLogProfile.isEnabled
        val hasBaselineLayer = hasBaselineLayer() && !suppressBaselineLayerForVideoLog
        val hasCreativeLayer = hasCreativeLayer()
        val hasDualLayer = hasBaselineLayer && hasCreativeLayer
        uploadPendingCurveTextures()
        val requestedFbo = liveRecorder != null ||
            activeVideoRecorder != null ||
            postProcessEffectEnabled ||
            bokehNeeded ||
            aiFocusInputNeeded ||
            hasDualLayer ||
            (!isAutoFocus && focusPeakingEnabled)
        if (requestedFbo && !isMainFboReady(viewportWidth, viewportHeight)) {
            initFbo(viewportWidth, viewportHeight)
        }
        val needsFbo = requestedFbo && fboId != 0 && fboTextureId != 0

        if (needsFbo) {
            val identityMatrix = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
            val fullCropRect = floatArrayOf(0f, 0f, 1f, 1f)

            var currentWidth = viewportWidth
            var currentHeight = viewportHeight

            // 1. 渲染色彩链路到 FBO
            drawInternal(
                fboId = fboId,
                width = currentWidth,
                height = currentHeight,
                preferBaselineLayer = hasDualLayer,
                suppressBaselineLayer = suppressBaselineLayerForVideoLog
            )

            var currentTexId = fboTextureId
            if (hasDualLayer && stackFboId != 0 && stackTextureId != 0 && secondPassLocations != null) {
                drawColorPass(
                    locations = secondPassLocations ?: return,
                    targetFboId = stackFboId,
                    width = viewportWidth,
                    height = viewportHeight,
                    sourceTextureTarget = GLES30.GL_TEXTURE_2D,
                    sourceTextureId = fboTextureId,
                    sourceStMatrix = identityMatrix,
                    sourceCropRect = fullCropRect,
                    targetMvpMatrix = identityMatrix,
                    lutConfig = currentLutConfig,
                    lutTextureId = lutTextureId,
                    lutSize = lutSize,
                    lutEnabled = lutEnabled && currentLutConfig != null,
                    params = getCurrentRecipeParams(),
                    curveTextureId = curveTextureId,
                    curveEnabled = curveEnabled && curveTextureId != 0,
                    enableVideoLog = false,
                    treatSourceAsHlgInput = false,
                    apertureOverride = 0f,
                    focusPointOverride = null
                )
                currentTexId = stackTextureId
                currentWidth = viewportWidth
                currentHeight = viewportHeight
            }

            // 深度采集（按需，从刚刚渲染好的 FBO 纹理读取）
            if (aperture > 0f) {
                runDepthInputCaptureInternal(currentTexId)
            }
            if (aiFocusInputNeeded) {
                runAiFocusInputCaptureInternal(currentTexId)
            }

            // 2. Bokeh 处理
            if (bokehNeeded) {
                currentTexId = renderBokehPreview(currentTexId, viewportWidth, viewportHeight)
                currentWidth = viewportWidth
                currentHeight = viewportHeight
            }

            // 3. HDF 多 Pass 处理
            var outputTexId = currentTexId
            if (hdfEnabled) {
                renderHdfPreviewBlur(currentTexId, currentWidth, currentHeight)
            }
            if (halationEnabled) {
                renderHalationPreviewBlur(currentTexId, currentWidth, currentHeight)
            }


            // 确保 FBO 内容已刷入显存
            GLES30.glFlush()

            // 4. Live Photo 录制
            if (liveRecorder != null) {
                val applyRotation = getApplyRotation()
                val isSwapped = applyRotation % 180 != 0
                val targetWidth = if (isSwapped) viewportHeight else viewportWidth
                val targetHeight = if (isSwapped) viewportWidth else viewportHeight
                val rotationMatrix = FloatArray(16)
                Matrix.setIdentityM(rotationMatrix, 0)
                if (applyRotation != 0) {
                    Matrix.translateM(rotationMatrix, 0, 0.5f, 0.5f, 0f)
                    Matrix.rotateM(rotationMatrix, 0, applyRotation.toFloat(), 0f, 0f, 1f)
                    Matrix.translateM(rotationMatrix, 0, -0.5f, -0.5f, 0f)
                }
                liveRecorder.onPreviewFrame(
                    textureId = outputTexId,
                    transformMatrix = rotationMatrix,
                    width = targetWidth,
                    height = targetHeight,
                    timestampNs = surfaceTexture?.timestamp ?: 0L,
                    lutConfig = currentLutConfig,
                    params = getCurrentRecipeParams(),
                    sharedContext = EGL14.eglGetCurrentContext(),
                    sharedDisplay = EGL14.eglGetCurrentDisplay()
                )
            }

            // 5. 视频录制输出
            if (hasFreshCameraFrame) {
                activeVideoRecorder?.targetSize?.let { targetSize ->
                    ensureRecordFbo(targetSize.width, targetSize.height)
                    if (recordFboId != 0 && recordTextureId != 0) {
                        if (postProcessEffectEnabled) {
                            drawPostProcessComposite(recordFboId, targetSize.width, targetSize.height, currentTexId)
                        } else if (bokehNeeded) {
                            drawFboToScreen(recordFboId, targetSize.width, targetSize.height, currentTexId)
                        } else {
                            drawFboToScreen(
                                fboId = recordFboId,
                                width = targetSize.width,
                                height = targetSize.height,
                                sourceTextureId = currentTexId,
                                targetMvpMatrix = buildTextureMvpMatrix(
                                    sourceWidth = viewportWidth,
                                    sourceHeight = viewportHeight,
                                    targetWidth = targetSize.width,
                                    targetHeight = targetSize.height
                                )
                            )
                        }
                        GLES30.glFlush()
                        activeVideoRecorder.onPreviewFrame(
                            textureId = recordTextureId,
                            transformMatrix = identityMatrix,
                            timestampNs = surfaceTexture?.timestamp ?: 0L,
                            sharedContext = EGL14.eglGetCurrentContext(),
                            sharedDisplay = EGL14.eglGetCurrentDisplay()
                        )
                    }
                }
            }

            // 3.5. Focus Peaking (Only for preview, not for recording)
            if (!isAutoFocus && focusPeakingEnabled) {
                currentTexId = renderFocusPeaking(currentTexId, currentWidth, currentHeight)
            }

            // 6. 显示到屏幕
            if (postProcessEffectEnabled) {
                drawPostProcessComposite(0, viewportWidth, viewportHeight, currentTexId)
            } else {
                drawFboToScreen(0, viewportWidth, viewportHeight, currentTexId)
            }
            val finalDisplayTextureId = currentTexId
            val finalDisplayWidth = viewportWidth
            val finalDisplayHeight = viewportHeight
            val needsHdfCompositeForSampling = postProcessEffectEnabled
            if (shouldCapturePreview) {
                shouldCapturePreview = false
                capturePreviewFrameInternal(
                    sourceTextureId = finalDisplayTextureId,
                    sourceWidth = finalDisplayWidth,
                    sourceHeight = finalDisplayHeight,
                    compositeWithHdf = needsHdfCompositeForSampling
                )
            }
            if (meteringEnabled && activeVideoRecorder == null) {
                runMeteringInternal(
                    sourceTextureId = finalDisplayTextureId,
                    sourceWidth = finalDisplayWidth,
                    sourceHeight = finalDisplayHeight,
                    compositeWithHdf = needsHdfCompositeForSampling
                )
            }
        } else {
            // 直接渲染到屏幕
            drawInternal(
                fboId = 0,
                width = viewportWidth,
                height = viewportHeight,
                suppressBaselineLayer = suppressBaselineLayerForVideoLog
            )
        }
    }

    /**
     * 将 FBO 纹理绘制到屏幕 (Copy Shader)
     */
    private fun drawFboToScreen(
        fboId: Int,
        width: Int,
        height: Int,
        sourceTextureId: Int,
        targetMvpMatrix: FloatArray? = null
    ) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
        GLES30.glViewport(0, 0, width, height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(copyProgramId)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(uCopyTextureLoc, 0)

        // FBO 纹理已经是正向的 (经过 stMatrix/MVP 处理后)，所以这里可能不需要再变换
        // 或者需要 Identity
        val identity = FloatArray(16)
        Matrix.setIdentityM(identity, 0)
        GLES30.glUniformMatrix4fv(uCopyMVPMatrixLoc, 1, false, targetMvpMatrix ?: identity, 0)

        // stMatrix 也设为 Identity，因为 FBO 纹理坐标是标准的
        GLES30.glUniformMatrix4fv(uCopySTMatrixLoc, 1, false, identity, 0)
        GLES30.glUniform4f(uCopyCropRectLoc, 0f, 0f, 1f, 1f)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(aCopyPositionLoc)
        GLES30.glVertexAttribPointer(aCopyPositionLoc, POSITION_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(aCopyTexCoordLoc)
        GLES30.glVertexAttribPointer(aCopyTexCoordLoc, TEXTURE_COORD_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, Shaders.DRAW_ORDER.size, GLES30.GL_UNSIGNED_SHORT, 0)

        GLES30.glDisableVertexAttribArray(aCopyPositionLoc)
        GLES30.glDisableVertexAttribArray(aCopyTexCoordLoc)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)

    }

    /**
     * 内部绘图逻辑，支持渲染到 FBO 或屏幕
     */
    private fun drawInternal(
        fboId: Int,
        width: Int,
        height: Int,
        targetMvpMatrix: FloatArray = mvpMatrix,
        preferBaselineLayer: Boolean = false,
        suppressBaselineLayer: Boolean = false
    ) {
        val locations = mainPassLocations ?: return
        val baselineLayerAvailable = hasBaselineLayer() && !suppressBaselineLayer
        val creativeLayerAvailable = hasCreativeLayer()
        val useCreativeLayer = creativeLayerAvailable && (!preferBaselineLayer || !baselineLayerAvailable)
        val useBaselineLayer = !useCreativeLayer && baselineLayerAvailable
        val layerLutConfig = when {
            useCreativeLayer -> currentLutConfig
            useBaselineLayer -> currentBaselineLutConfig
            else -> null
        }
        val layerLutTextureId = when {
            useCreativeLayer -> lutTextureId
            useBaselineLayer -> baselineLutTextureId
            else -> 0
        }
        val layerLutSize = if (useBaselineLayer) baselineLutSize else lutSize
        val layerLutEnabled = if (useCreativeLayer) {
            lutEnabled && currentLutConfig != null
        } else if (useBaselineLayer) {
            baselineLutEnabled && currentBaselineLutConfig != null
        } else {
            false
        }
        val layerParams = when {
            useCreativeLayer -> getCurrentRecipeParams()
            useBaselineLayer -> baselineRecipeParams
            else -> com.hinnka.mycamera.model.ColorRecipeParams.DEFAULT
        }
        val layerCurveTextureId = if (useBaselineLayer) baselineCurveTextureId else curveTextureId
        val layerCurveEnabled = if (useCreativeLayer) {
            curveEnabled && curveTextureId != 0
        } else if (useBaselineLayer) {
            baselineCurveEnabled && baselineCurveTextureId != 0
        } else {
            false
        }
        drawColorPass(
            locations = locations,
            targetFboId = fboId,
            width = width,
            height = height,
            sourceTextureTarget = GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            sourceTextureId = cameraTextureId,
            sourceStMatrix = stMatrix,
            sourceCropRect = cropRect,
            targetMvpMatrix = targetMvpMatrix,
            lutConfig = layerLutConfig,
            lutTextureId = layerLutTextureId,
            lutSize = layerLutSize,
            lutEnabled = layerLutEnabled,
            params = layerParams,
            curveTextureId = layerCurveTextureId,
            curveEnabled = layerCurveEnabled,
            enableVideoLog = true,
            treatSourceAsHlgInput = isHlgInput
        )

        // 捕获预览帧（如果需要）
        if (fboId == 0 && shouldCapturePreview) {
            shouldCapturePreview = false
            capturePreviewFrameInternal()
        }

        // 测光和直方图（按需）
        if (meteringEnabled && videoRecorder?.isRecording() != true) {
            runMeteringInternal()
        }

        GlUtils.checkGlError("onDrawFrame")
    }

    /**
     * 初始化着色器程序
     */
    private fun queryColorPassLocations(program: Int): ColorPassLocations {
        return ColorPassLocations(
            programId = program,
            uMVPMatrixLocation = GLES30.glGetUniformLocation(program, "uMVPMatrix"),
            uSTMatrixLocation = GLES30.glGetUniformLocation(program, "uSTMatrix"),
            uCameraTextureLocation = GLES30.glGetUniformLocation(program, "uCameraTexture"),
            uLutTextureLocation = GLES30.glGetUniformLocation(program, "uLutTexture"),
            uLutSizeLocation = GLES30.glGetUniformLocation(program, "uLutSize"),
            uLutIntensityLocation = GLES30.glGetUniformLocation(program, "uLutIntensity"),
            uLutEnabledLocation = GLES30.glGetUniformLocation(program, "uLutEnabled"),
            uLutMaskTypeLocation = GLES30.glGetUniformLocation(program, "uLutMaskType"),
            uLutCurveLocation = GLES30.glGetUniformLocation(program, "uLutCurve"),
            uLutColorSpaceLocation = GLES30.glGetUniformLocation(program, "uLutColorSpace"),
            uVideoLogEnabledLocation = GLES30.glGetUniformLocation(program, "uVideoLogEnabled"),
            uVideoLogCurveLocation = GLES30.glGetUniformLocation(program, "uVideoLogCurve"),
            uVideoColorSpaceLocation = GLES30.glGetUniformLocation(program, "uVideoColorSpace"),
            uIsHlgInputLocation = GLES30.glGetUniformLocation(program, "uIsHlgInput"),
            uColorRecipeEnabledLocation = GLES30.glGetUniformLocation(program, "uColorRecipeEnabled"),
            uExposureLocation = GLES30.glGetUniformLocation(program, "uExposure"),
            uContrastLocation = GLES30.glGetUniformLocation(program, "uContrast"),
            uSaturationLocation = GLES30.glGetUniformLocation(program, "uSaturation"),
            uTemperatureLocation = GLES30.glGetUniformLocation(program, "uTemperature"),
            uTintLocation = GLES30.glGetUniformLocation(program, "uTint"),
            uFadeLocation = GLES30.glGetUniformLocation(program, "uFade"),
            uVibranceLocation = GLES30.glGetUniformLocation(program, "uVibrance"),
            uHighlightsLocation = GLES30.glGetUniformLocation(program, "uHighlights"),
            uShadowsLocation = GLES30.glGetUniformLocation(program, "uShadows"),
            uToneToeLocation = GLES30.glGetUniformLocation(program, "uToneToe"),
            uToneShoulderLocation = GLES30.glGetUniformLocation(program, "uToneShoulder"),
            uTonePivotLocation = GLES30.glGetUniformLocation(program, "uTonePivot"),
            uFilmGrainLocation = GLES30.glGetUniformLocation(program, "uFilmGrain"),
            uVignetteLocation = GLES30.glGetUniformLocation(program, "uVignette"),
            uBleachBypassLocation = GLES30.glGetUniformLocation(program, "uBleachBypass"),
            uChromaticAberrationLocation = GLES30.glGetUniformLocation(program, "uChromaticAberration"),
            uNoiseLocation = GLES30.glGetUniformLocation(program, "uNoise"),
            uNoiseSeedLocation = GLES30.glGetUniformLocation(program, "uNoiseSeed"),
            uLowResLocation = GLES30.glGetUniformLocation(program, "uLowRes"),
            uAspectRatioLocation = GLES30.glGetUniformLocation(program, "uAspectRatio"),
            uLchHueAdjustmentsLocation = GLES30.glGetUniformLocation(program, "uLchHueAdjustments"),
            uLchChromaAdjustmentsLocation = GLES30.glGetUniformLocation(program, "uLchChromaAdjustments"),
            uLchLightnessAdjustmentsLocation = GLES30.glGetUniformLocation(program, "uLchLightnessAdjustments"),
            uPrimaryCalibrationMatrixLocation = GLES30.glGetUniformLocation(program, "uPrimaryCalibrationMatrix"),
            uSTMatrixFragLocation = GLES30.glGetUniformLocation(program, "uSTMatrix"),
            uCropRectLocation = GLES30.glGetUniformLocation(program, "uCropRect"),
            uApertureLocation = GLES30.glGetUniformLocation(program, "uAperture"),
            uFocusPointLocation = GLES30.glGetUniformLocation(program, "uFocusPoint"),
            uCurveTextureLocation = GLES30.glGetUniformLocation(program, "uCurveTexture"),
            uCurveEnabledLocation = GLES30.glGetUniformLocation(program, "uCurveEnabled"),
            aPositionLocation = GLES30.glGetAttribLocation(program, "aPosition"),
            aTexCoordLocation = GLES30.glGetAttribLocation(program, "aTexCoord"),
        )
    }

    private fun initShaderProgram() {
        val vertexShader = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.VERTEX_SHADER)
        val fragmentShader = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.FRAGMENT_SHADER_COLOR_RECIPE)

        if (vertexShader == 0 || fragmentShader == 0) {
            PLog.e(TAG, "Failed to compile shaders")
            return
        }

        programId = GlUtils.linkProgram(vertexShader, fragmentShader)

        // 着色器已链接到程序，可以删除
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)

        if (programId == 0) {
            PLog.e(TAG, "Failed to link program")
            return
        }

        // 获取 Uniform 位置
        uMVPMatrixLocation = GLES30.glGetUniformLocation(programId, "uMVPMatrix")
        uSTMatrixLocation = GLES30.glGetUniformLocation(programId, "uSTMatrix")
        uCameraTextureLocation = GLES30.glGetUniformLocation(programId, "uCameraTexture")
        uLutTextureLocation = GLES30.glGetUniformLocation(programId, "uLutTexture")
        uLutSizeLocation = GLES30.glGetUniformLocation(programId, "uLutSize")
        uLutIntensityLocation = GLES30.glGetUniformLocation(programId, "uLutIntensity")
        uLutEnabledLocation = GLES30.glGetUniformLocation(programId, "uLutEnabled")
        uLutCurveLocation = GLES30.glGetUniformLocation(programId, "uLutCurve")
        uLutColorSpaceLocation = GLES30.glGetUniformLocation(programId, "uLutColorSpace")
        uVideoLogEnabledLocation = GLES30.glGetUniformLocation(programId, "uVideoLogEnabled")
        uVideoLogCurveLocation = GLES30.glGetUniformLocation(programId, "uVideoLogCurve")
        uVideoColorSpaceLocation = GLES30.glGetUniformLocation(programId, "uVideoColorSpace")
        uIsHlgInputLocation = GLES30.glGetUniformLocation(programId, "uIsHlgInput")

        // 获取色彩配方 Uniform 位置
        uColorRecipeEnabledLocation = GLES30.glGetUniformLocation(programId, "uColorRecipeEnabled")
        uExposureLocation = GLES30.glGetUniformLocation(programId, "uExposure")
        uContrastLocation = GLES30.glGetUniformLocation(programId, "uContrast")
        uSaturationLocation = GLES30.glGetUniformLocation(programId, "uSaturation")
        uTemperatureLocation = GLES30.glGetUniformLocation(programId, "uTemperature")
        uTintLocation = GLES30.glGetUniformLocation(programId, "uTint")
        uFadeLocation = GLES30.glGetUniformLocation(programId, "uFade")
        uVibranceLocation = GLES30.glGetUniformLocation(programId, "uVibrance")
        uHighlightsLocation = GLES30.glGetUniformLocation(programId, "uHighlights")
        uShadowsLocation = GLES30.glGetUniformLocation(programId, "uShadows")
        uToneToeLocation = GLES30.glGetUniformLocation(programId, "uToneToe")
        uToneShoulderLocation = GLES30.glGetUniformLocation(programId, "uToneShoulder")
        uTonePivotLocation = GLES30.glGetUniformLocation(programId, "uTonePivot")
        uFilmGrainLocation = GLES30.glGetUniformLocation(programId, "uFilmGrain")
        uVignetteLocation = GLES30.glGetUniformLocation(programId, "uVignette")
        uBleachBypassLocation = GLES30.glGetUniformLocation(programId, "uBleachBypass")
        uChromaticAberrationLocation = GLES30.glGetUniformLocation(programId, "uChromaticAberration")
        uNoiseLocation = GLES30.glGetUniformLocation(programId, "uNoise")
        uNoiseSeedLocation = GLES30.glGetUniformLocation(programId, "uNoiseSeed")
        uLowResLocation = GLES30.glGetUniformLocation(programId, "uLowRes")
        uAspectRatioLocation = GLES30.glGetUniformLocation(programId, "uAspectRatio")
        uLchHueAdjustmentsLocation = GLES30.glGetUniformLocation(programId, "uLchHueAdjustments")
        uLchChromaAdjustmentsLocation = GLES30.glGetUniformLocation(programId, "uLchChromaAdjustments")
        uLchLightnessAdjustmentsLocation = GLES30.glGetUniformLocation(programId, "uLchLightnessAdjustments")
        uPrimaryCalibrationMatrixLocation = GLES30.glGetUniformLocation(programId, "uPrimaryCalibrationMatrix")
        uSTMatrixFragLocation = GLES30.glGetUniformLocation(programId, "uSTMatrix")
        uCropRectLocation = GLES30.glGetUniformLocation(programId, "uCropRect")
        uApertureLocation = GLES30.glGetUniformLocation(programId, "uAperture")
        uFocusPointLocation = GLES30.glGetUniformLocation(programId, "uFocusPoint")
        uCurveTextureLocation = GLES30.glGetUniformLocation(programId, "uCurveTexture")
        uCurveEnabledLocation = GLES30.glGetUniformLocation(programId, "uCurveEnabled")

        // Attribute 位置
        aPositionLocation = GLES30.glGetAttribLocation(programId, "aPosition")
        aTexCoordLocation = GLES30.glGetAttribLocation(programId, "aTexCoord")
        mainPassLocations = queryColorPassLocations(programId)

        val secondVertexShader = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.VERTEX_SHADER)
        val secondFragmentShader = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.FRAGMENT_SHADER_COLOR_RECIPE_2D)
        secondPassProgramId = GlUtils.linkProgram(secondVertexShader, secondFragmentShader)
        GLES30.glDeleteShader(secondVertexShader)
        GLES30.glDeleteShader(secondFragmentShader)
        if (secondPassProgramId == 0) {
            PLog.e(TAG, "Failed to link second pass color recipe program")
        } else {
            secondPassLocations = queryColorPassLocations(secondPassProgramId)
        }

        // === 初始化 Passthrough Shader (用于深度采集) ===
        if (passthroughProgramId == 0) {
            val passVs = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.VERTEX_SHADER)
            val passFs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.FRAGMENT_SHADER_PASSTHROUGH)
            passthroughProgramId = GlUtils.linkProgram(passVs, passFs)
            GLES30.glDeleteShader(passVs)
            GLES30.glDeleteShader(passFs)
            
            if (passthroughProgramId != 0) {
                uPassMVPMatrixLocation = GLES30.glGetUniformLocation(passthroughProgramId, "uMVPMatrix")
                uPassSTMatrixLocation = GLES30.glGetUniformLocation(passthroughProgramId, "uSTMatrix")
                uPassCropRectLocation = GLES30.glGetUniformLocation(passthroughProgramId, "uCropRect")
                uPassCameraTextureLocation = GLES30.glGetUniformLocation(passthroughProgramId, "uCameraTexture")
                aPassPositionLocation = GLES30.glGetAttribLocation(passthroughProgramId, "aPosition")
                aPassTexCoordLocation = GLES30.glGetAttribLocation(passthroughProgramId, "aTexCoord")
            }
        }

        // === 初始化 Copy Shader (用于 FBO 上屏) ===
        val copyVs = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.VERTEX_SHADER)
        val copyFs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.FRAGMENT_SHADER_COPY_2D)
        copyProgramId = GlUtils.linkProgram(copyVs, copyFs)
        GLES30.glDeleteShader(copyVs)
        GLES30.glDeleteShader(copyFs)

        if (copyProgramId == 0) {
            PLog.e(TAG, "Failed to link copy program")
        } else {
            uCopyTextureLoc = GLES30.glGetUniformLocation(copyProgramId, "uCameraTexture")
            uCopyMVPMatrixLoc = GLES30.glGetUniformLocation(copyProgramId, "uMVPMatrix")
            uCopySTMatrixLoc = GLES30.glGetUniformLocation(copyProgramId, "uSTMatrix")
            uCopyCropRectLoc = GLES30.glGetUniformLocation(copyProgramId, "uCropRect")
            aCopyPositionLoc = GLES30.glGetAttribLocation(copyProgramId, "aPosition")
            aCopyTexCoordLoc = GLES30.glGetAttribLocation(copyProgramId, "aTexCoord")
        }
//        PLog.d(TAG, "Shader program created: $programId")
    }

    private fun initHdfPrograms() {
        val simpleVs = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.SIMPLE_VERTEX_SHADER)
        // Extract + Blur H
        val extractFs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.HDF_PREVIEW_EXTRACT_BLUR_H)
        hdfExtractBlurHProgram = GlUtils.linkProgram(simpleVs, extractFs)
        GLES30.glDeleteShader(extractFs)
        // Blur V
        val blurVFs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.HDF_PREVIEW_BLUR_V)
        hdfBlurVProgram = GlUtils.linkProgram(simpleVs, blurVFs)
        GLES30.glDeleteShader(blurVFs)
        // Composite
        val compositeFs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.HDF_PREVIEW_COMPOSITE)
        hdfCompositeProgram = GlUtils.linkProgram(simpleVs, compositeFs)
        GLES30.glDeleteShader(compositeFs)
        val halationExtractHFs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.HALATION_PREVIEW_EXTRACT_BLUR_H)
        val halationBlurVFs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.HALATION_PREVIEW_BLUR_V)
        halationExtractBlurHProgram = GlUtils.linkProgram(simpleVs, halationExtractHFs)
        halationBlurVProgram = GlUtils.linkProgram(simpleVs, halationBlurVFs)
        GLES30.glDeleteShader(halationExtractHFs)
        GLES30.glDeleteShader(halationBlurVFs)
        GLES30.glDeleteShader(simpleVs)

        if (hdfExtractBlurHProgram == 0 || hdfBlurVProgram == 0 || hdfCompositeProgram == 0 || halationExtractBlurHProgram == 0 || halationBlurVProgram == 0) {
            PLog.e(TAG, "Failed to link HDF preview programs")
        }
    }

    private fun setupHalationFbos(width: Int, height: Int) {
        val dsW = width / 4; val dsH = height / 4
        if (halationWidth == dsW && halationHeight == dsH && halationTexId[0] != 0) return
        halationWidth = dsW; halationHeight = dsH
        for (i in 0..1) {
            if (halationTexId[i] != 0) GLES30.glDeleteTextures(1, intArrayOf(halationTexId[i]), 0)
            if (halationFboId[i] != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(halationFboId[i]), 0)
            val t = IntArray(1); val f = IntArray(1)
            GLES30.glGenTextures(1, t, 0); GLES30.glGenFramebuffers(1, f, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0])
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, dsW, dsH, 0, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, f[0])
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, t[0], 0)
            halationTexId[i] = t[0]; halationFboId[i] = f[0]
        }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun setupHdfFbos(width: Int, height: Int) {
        val dsW = width / 4
        val dsH = height / 4
        if (hdfWidth == dsW && hdfHeight == dsH && hdfTexId[0] != 0) return
        hdfWidth = dsW
        hdfHeight = dsH
        for (i in 0..1) {
            if (hdfTexId[i] != 0) GLES30.glDeleteTextures(1, intArrayOf(hdfTexId[i]), 0)
            if (hdfFboId[i] != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(hdfFboId[i]), 0)
            val t = IntArray(1);
            val f = IntArray(1)
            GLES30.glGenTextures(1, t, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0])
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D,
                0,
                GLES30.GL_RGBA16F,
                dsW,
                dsH,
                0,
                GLES30.GL_RGBA,
                GLES30.GL_HALF_FLOAT,
                null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
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
            hdfTexId[i] = t[0]; hdfFboId[i] = f[0]
        }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun renderHdfPreviewBlur(sourceTexId: Int, width: Int, height: Int) {
        if (!ensureHdfPrograms()) return
        setupHdfFbos(width, height)
        if (hdfExtractBlurHProgram == 0 || hdfBlurVProgram == 0) return
        val dsW = width / 4;
        val dsH = height / 4
        val spatialScale = getPreviewSpatialEffectScale(width, height)
        val texelW = spatialScale / dsW;
        val texelH = spatialScale / dsH
        val threshold = 0.9f - halation * 0.3f
        // Pass 1: Extract + Horizontal Blur
        GLES30.glUseProgram(hdfExtractBlurHProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, hdfFboId[0])
        GLES30.glViewport(0, 0, dsW, dsH)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTexId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(hdfExtractBlurHProgram, "uInputTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(hdfExtractBlurHProgram, "uTexelSize"), texelW, texelH)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(hdfExtractBlurHProgram, "uThreshold"), threshold)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(hdfExtractBlurHProgram, "uStrength"), halation)
        drawSimpleQuad(hdfExtractBlurHProgram)
        // Pass 2: Vertical Blur
        GLES30.glUseProgram(hdfBlurVProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, hdfFboId[1])
        GLES30.glViewport(0, 0, dsW, dsH)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, hdfTexId[0])
        GLES30.glUniform1i(GLES30.glGetUniformLocation(hdfBlurVProgram, "uInputTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(hdfBlurVProgram, "uTexelSize"), texelW, texelH)
        drawSimpleQuad(hdfBlurVProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun renderHalationPreviewBlur(sourceTexId: Int, width: Int, height: Int) {
        if (!ensureHdfPrograms()) return
        setupHalationFbos(width, height)
        if (halationExtractBlurHProgram == 0 || halationBlurVProgram == 0) return
        val dsW = width / 4; val dsH = height / 4
        val spatialScale = getPreviewSpatialEffectScale(width, height)
        val texelW = spatialScale / dsW; val texelH = spatialScale / dsH
        val threshold = 0.72f - redHalation.coerceIn(0f, 1f) * 0.22f
        
        GLES30.glUseProgram(halationExtractBlurHProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, halationFboId[0])
        GLES30.glViewport(0, 0, dsW, dsH)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTexId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(halationExtractBlurHProgram, "uInputTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(halationExtractBlurHProgram, "uTexelSize"), texelW, texelH)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(halationExtractBlurHProgram, "uThreshold"), threshold)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(halationExtractBlurHProgram, "uStrength"), redHalation)
        drawSimpleQuad(halationExtractBlurHProgram)
        
        GLES30.glUseProgram(halationBlurVProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, halationFboId[1])
        GLES30.glViewport(0, 0, dsW, dsH)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, halationTexId[0])
        GLES30.glUniform1i(GLES30.glGetUniformLocation(halationBlurVProgram, "uInputTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(halationBlurVProgram, "uTexelSize"), texelW, texelH)
        drawSimpleQuad(halationBlurVProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun drawPostProcessComposite(targetFboId: Int, width: Int, height: Int, sourceTextureId: Int) {
        if (!ensureHdfPrograms()) return
        if (hdfCompositeProgram == 0) return
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, targetFboId)
        GLES30.glViewport(0, 0, width, height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(hdfCompositeProgram)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(hdfCompositeProgram, "uOriginalTexture"), 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, hdfTexId[1])
        GLES30.glUniform1i(GLES30.glGetUniformLocation(hdfCompositeProgram, "uBloomTexture"), 1)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(hdfCompositeProgram, "uHalation"), halation)
        
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, if (redHalation > 0f) halationTexId[1] else 0)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(hdfCompositeProgram, "uRedHalationTexture"), 2)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(hdfCompositeProgram, "uRedHalation"), redHalation)
        drawSimpleQuad(hdfCompositeProgram)
    }

    private fun getPreviewSpatialEffectScale(width: Int, height: Int): Float {
        val previewLongEdge = maxOf(width, height).coerceAtLeast(1)
        val captureLongEdge = maxOf(photoCaptureWidth, photoCaptureHeight).coerceAtLeast(previewLongEdge)
        val scale = (previewLongEdge.toFloat() / captureLongEdge.toFloat()).coerceIn(0.25f, 1f)
        if (abs(scale - lastLoggedSpatialEffectScale) > 0.01f) {
            lastLoggedSpatialEffectScale = scale
            PLog.d(
                TAG,
                "Preview spatial effect scale=$scale preview=${width}x${height} capture=${photoCaptureWidth}x${photoCaptureHeight}"
            )
        }
        return scale
    }

    /** 使用 VBO 绘制全屏四边形（用于 HDF Pass，使用 SIMPLE_VERTEX_SHADER） */
    private fun drawSimpleQuad(program: Int) {
        val posLoc = GLES30.glGetAttribLocation(program, "aPosition")
        val texLoc = GLES30.glGetAttribLocation(program, "aTexCoord")
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(posLoc)
        GLES30.glVertexAttribPointer(posLoc, POSITION_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(texLoc)
        GLES30.glVertexAttribPointer(texLoc, TEXTURE_COORD_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, Shaders.DRAW_ORDER.size, GLES30.GL_UNSIGNED_SHORT, 0)
        GLES30.glDisableVertexAttribArray(posLoc)
        GLES30.glDisableVertexAttribArray(texLoc)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)
    }

    private fun initPassthroughProgram() {
        val vertexShader = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.VERTEX_SHADER)
        val fragmentShader = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.FRAGMENT_SHADER_PASSTHROUGH)

        if (vertexShader == 0 || fragmentShader == 0) {
            PLog.e(TAG, "Failed to compile passthrough shaders")
            return
        }

        passthroughProgramId = GlUtils.linkProgram(vertexShader, fragmentShader)
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)

        if (passthroughProgramId != 0) {
            uPassMVPMatrixLocation = GLES30.glGetUniformLocation(passthroughProgramId, "uMVPMatrix")
            uPassSTMatrixLocation = GLES30.glGetUniformLocation(passthroughProgramId, "uSTMatrix")
            uPassCameraTextureLocation = GLES30.glGetUniformLocation(passthroughProgramId, "uCameraTexture")
            aPassPositionLocation = GLES30.glGetAttribLocation(passthroughProgramId, "aPosition")
            aPassTexCoordLocation = GLES30.glGetAttribLocation(passthroughProgramId, "aTexCoord")
        }
    }

    private fun initMeteringFbo() {
        val fbos = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        meteringFboId = fbos[0]

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        meteringTextureId = textures[0]

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, meteringTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
            METERING_SIZE, METERING_SIZE, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, meteringFboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, meteringTextureId, 0
        )

        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            PLog.e(TAG, "Failed to create metering FBO: $status")
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
    }

    private fun initDepthInputFbo() {
        if (depthInputFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(depthInputFboId), 0)
            depthInputFboId = 0
        }
        if (depthInputTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(depthInputTextureId), 0)
            depthInputTextureId = 0
        }

        val fbos = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        depthInputFboId = fbos[0]

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        depthInputTextureId = textures[0]

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthInputTextureId)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, depthInputFboId)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, depthInputTextureId, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun initAiFocusInputFbo() {
        if (aiFocusInputFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(aiFocusInputFboId), 0)
            aiFocusInputFboId = 0
        }
        if (aiFocusInputTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(aiFocusInputTextureId), 0)
            aiFocusInputTextureId = 0
        }

        val fbos = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        aiFocusInputFboId = fbos[0]

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        aiFocusInputTextureId = textures[0]

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, aiFocusInputTextureId)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, AI_FOCUS_INPUT_SIZE, AI_FOCUS_INPUT_SIZE, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, aiFocusInputFboId)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, aiFocusInputTextureId, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun initCaptureFbo() {
        if (captureFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(captureFboId), 0)
            captureFboId = 0
        }
        if (captureTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(captureTextureId), 0)
            captureTextureId = 0
        }

        val fbos = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        captureFboId = fbos[0]

        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        captureTextureId = textures[0]

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, captureTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
            captureWidth, captureHeight, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, captureFboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, captureTextureId, 0
        )
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun ensureRecordFbo(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        if (recordFboId != 0 && recordFboWidth == width && recordFboHeight == height) return

        if (recordFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(recordFboId), 0)
            recordFboId = 0
        }
        if (recordTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(recordTextureId), 0)
            recordTextureId = 0
        }

        val fbos = IntArray(1)
        val textures = IntArray(1)
        GLES30.glGenFramebuffers(1, fbos, 0)
        GLES30.glGenTextures(1, textures, 0)
        recordFboId = fbos[0]
        recordTextureId = textures[0]
        recordFboWidth = width
        recordFboHeight = height

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, recordTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F,
            width, height, 0, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, recordFboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            recordTextureId,
            0
        )
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    /**
     * 初始化顶点缓冲
     */
    private fun initBuffers() {
        // 顶点位置缓冲
        val vertexBuffer = GlUtils.createFloatBuffer(Shaders.FULL_QUAD_VERTICES)
        val vertexBufferIds = IntArray(1)
        GLES30.glGenBuffers(1, vertexBufferIds, 0)
        vertexBufferId = vertexBufferIds[0]
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            Shaders.FULL_QUAD_VERTICES.size * BYTES_PER_FLOAT,
            vertexBuffer,
            GLES30.GL_STATIC_DRAW
        )

        // 纹理坐标缓冲
        val texCoordBuffer = GlUtils.createFloatBuffer(Shaders.TEXTURE_COORDS)
        val texCoordBufferIds = IntArray(1)
        GLES30.glGenBuffers(1, texCoordBufferIds, 0)
        texCoordBufferId = texCoordBufferIds[0]
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            Shaders.TEXTURE_COORDS.size * BYTES_PER_FLOAT,
            texCoordBuffer,
            GLES30.GL_STATIC_DRAW
        )

        // 索引缓冲
        val indexBuffer = ByteBuffer.allocateDirect(Shaders.DRAW_ORDER.size * BYTES_PER_SHORT)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
            .put(Shaders.DRAW_ORDER)
        indexBuffer.position(0)

        val indexBufferIds = IntArray(1)
        GLES30.glGenBuffers(1, indexBufferIds, 0)
        indexBufferId = indexBufferIds[0]
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glBufferData(
            GLES30.GL_ELEMENT_ARRAY_BUFFER,
            Shaders.DRAW_ORDER.size * BYTES_PER_SHORT,
            indexBuffer,
            GLES30.GL_STATIC_DRAW
        )

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)
    }

    /**
     * 设置 LUT
     * 注意：需要在 GL 线程中调用
     */
    fun setLut(lutConfig: LutConfig?) {
        // 如果 Surface 尚未创建，保存配置稍后处理
        if (!surfaceReady) {
            currentLutConfig = lutConfig
            pendingLutConfig = lutConfig
            return
        }

        setLutInternal(lutConfig)
    }

    fun setBaselineLut(lutConfig: LutConfig?) {
        if (!surfaceReady) {
            currentBaselineLutConfig = lutConfig
            pendingBaselineLutConfig = lutConfig
            return
        }
        setBaselineLutInternal(lutConfig)
    }

    /**
     * 内部方法：实际设置 LUT
     * 仅在 Surface 已创建后调用
     */
    private fun setLutInternal(lutConfig: LutConfig?) {
        // 删除旧的 LUT 纹理
        if (lutTextureId != 0) {
            GlUtils.deleteTexture(lutTextureId)
            lutTextureId = 0
        }

        currentLutConfig = lutConfig

        if (lutConfig != null && lutConfig.isValid()) {
            lutTextureId = GlUtils.create3DTexture(lutConfig)
            lutSize = lutConfig.size.toFloat()
            lutEnabled = true
//            PLog.d(TAG, "LUT set: ${lutConfig.title}, size: ${lutConfig.size}")
        } else {
            lutEnabled = false
        }
    }

    private fun setBaselineLutInternal(lutConfig: LutConfig?) {
        if (baselineLutTextureId != 0) {
            GlUtils.deleteTexture(baselineLutTextureId)
            baselineLutTextureId = 0
        }

        currentBaselineLutConfig = lutConfig

        if (lutConfig != null && lutConfig.isValid()) {
            baselineLutTextureId = GlUtils.create3DTexture(lutConfig)
            baselineLutSize = lutConfig.size.toFloat()
            baselineLutEnabled = true
        } else {
            baselineLutEnabled = false
        }
    }

    /**
     * App 从后台恢复时，GLSurfaceView 可能保留了 Kotlin 层的 LUT 选择状态，
     * 但底层 GL texture 已随 Surface/Context 生命周期失效。按当前配置强制重建，
     * 避免 UI 仍显示已选 LUT 而预览实际走无 LUT 路径。
     */
    fun restoreLutTexturesAfterResume() {
        if (!surfaceReady) {
            // Surface 还没重建时，保留已经排队的配置，不要被当前缓存的 null 覆盖掉。
            pendingLutConfig = pendingLutConfig ?: currentLutConfig
            pendingBaselineLutConfig = pendingBaselineLutConfig ?: currentBaselineLutConfig
            PLog.d(TAG, "restore LUT deferred: surface not ready")
            return
        }

        currentBaselineLutConfig?.let { config ->
            PLog.d(TAG, "restore baseline LUT texture after resume: ${config.title}")
            setBaselineLutInternal(config)
        }

        currentLutConfig?.let { config ->
            PLog.d(TAG, "restore LUT texture after resume: ${config.title}")
            setLutInternal(config)
        }
    }

    /**
     * 设置预览尺寸
     */
    fun setPreviewSize(width: Int, height: Int) {
        previewWidth = width
        previewHeight = height
        surfaceTexture?.setDefaultBufferSize(width, height)
        // 更新 MVP 矩阵以处理 center crop
        updateMVPMatrix()
        updateCaptureSize()
    }

    fun setCaptureSize(width: Int, height: Int) {
        photoCaptureWidth = width.coerceAtLeast(0)
        photoCaptureHeight = height.coerceAtLeast(0)
    }

    fun setSourceCrop(crop: PhantomPipCrop) {
        val normalized = crop.normalized()
        cropRect[0] = normalized.left
        cropRect[1] = normalized.top
        cropRect[2] = normalized.right
        cropRect[3] = normalized.bottom
        updateMVPMatrix()
    }

    /**
     * 设置传感器方向 (硬件属性)
     */
    fun setSensorOrientation(orientation: Int) {
        if (sensorOrientation != orientation) {
            sensorOrientation = orientation
            updateMVPMatrix()
            updateCaptureSize()
        }
    }

    /**
     * 设置校正偏移 (用户设置)
     */
    fun setCalibrationOffset(offset: Int) {
        if (calibrationOffset != offset) {
            calibrationOffset = offset
            updateMVPMatrix()
            updateCaptureSize()
        }
    }

    /**
     * 设置设备旋转方向 (0, 90, 180, 270)
     */
    fun setDeviceRotation(degrees: Int) {
        if (deviceRotation != degrees) {
            deviceRotation = degrees
            updateMVPMatrix()
            updateCaptureSize()
        }
    }

    /**
     * 设置镜头朝向
     */
    fun setLensFacing(facing: Int) {
        if (lensFacing != facing) {
            lensFacing = facing
            updateMVPMatrix()
            updateCaptureSize()
        }
    }

    fun setCaptureAspectRatio(aspectRatio: Float) {
        val safeAspectRatio = aspectRatio.coerceAtLeast(0f)
        if (kotlin.math.abs(captureAspectRatio - safeAspectRatio) > 0.0001f) {
            captureAspectRatio = safeAspectRatio
            updateCaptureSize()
        }
    }

    /**
     * 计算相对于传感器的总旋转角度 (用于确定最终图片的宽高比)
     * 参考 CameraViewModel.saveImage 的逻辑
     */
    private fun calculateTotalRotation(): Int {
        val baseRotation = if (lensFacing == 0 /* CameraCharacteristics.LENS_FACING_FRONT */) {
            (sensorOrientation - deviceRotation + 360) % 360
        } else {
            (sensorOrientation + deviceRotation) % 360
        }
        return (baseRotation + calibrationOffset) % 360
    }

    /**
     * 计算相对于“竖屏正向”状态需要额外应用的旋转角度
     * 因为 stMatrix 已经处理了 sensorOrientation，所以我们只需要根据设备旋转和校正量进行增量旋转
     */
    private fun getApplyRotation(): Int {
        // 对于后置摄像头，设备旋转 90 (Landscape Left) 需要将画面顺时针旋转 90 度
        // 对于前置摄像头，由于镜像关系，设备旋转 90 (Landscape Left) 需要将画面逆时针旋转 90 度 (即 CW 270)
        val rotation = if (lensFacing == 0 /* FRONT */) {
            (360 - deviceRotation) % 360
        } else {
            deviceRotation
        }
        return (rotation + calibrationOffset) % 360
    }

    /**
     * 更新 MVP 矩阵以实现 center crop 效果
     * 
     * 当预览尺寸与显示区域比例不匹配时，放大画面并裁切超出部分
     */
    private fun updateMVPMatrix() {
        val computedMatrix = buildMvpMatrix(viewportWidth, viewportHeight)
        System.arraycopy(computedMatrix, 0, mvpMatrix, 0, mvpMatrix.size)

//        PLog.d(
//            TAG,
//            "MVP matrix updated: viewport=${viewportWidth}x${viewportHeight}, preview=${previewWidth}x${previewHeight}"
//        )
    }

    private fun buildMvpMatrix(targetWidth: Int, targetHeight: Int): FloatArray {
        val matrix = FloatArray(16)
        Matrix.setIdentityM(matrix, 0)
        if (targetWidth <= 0 || targetHeight <= 0) {
            return matrix
        }

        val isSwapped = (sensorOrientation + calibrationOffset) % 180 != 0
        val cropWidth = (cropRect[2] - cropRect[0]).coerceAtLeast(0.05f)
        val cropHeight = (cropRect[3] - cropRect[1]).coerceAtLeast(0.05f)
        val previewAspect = if (isSwapped) {
            (previewHeight.toFloat() * cropHeight) / (previewWidth.toFloat() * cropWidth)
        } else {
            (previewWidth.toFloat() * cropWidth) / (previewHeight.toFloat() * cropHeight)
        }
        val viewAspect = targetWidth.toFloat() / targetHeight.toFloat()

        if (calibrationOffset != 0) {
            Matrix.rotateM(matrix, 0, (-calibrationOffset).toFloat(), 0f, 0f, 1f)
        }

        if (previewAspect != viewAspect) {
            val scaleX: Float
            val scaleY: Float
            if (viewAspect > previewAspect) {
                scaleX = 1f
                scaleY = viewAspect / previewAspect
            } else {
                scaleX = previewAspect / viewAspect
                scaleY = 1f
            }
            Matrix.scaleM(matrix, 0, scaleX, scaleY, 1f)
        }
        return matrix
    }

    private fun buildTextureMvpMatrix(
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ): FloatArray {
        val matrix = FloatArray(16)
        Matrix.setIdentityM(matrix, 0)
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
            return matrix
        }

        val sourceAspect = sourceWidth.toFloat() / sourceHeight.toFloat()
        val targetAspect = targetWidth.toFloat() / targetHeight.toFloat()

        if (sourceAspect != targetAspect) {
            val scaleX: Float
            val scaleY: Float
            if (targetAspect > sourceAspect) {
                scaleX = 1f
                scaleY = targetAspect / sourceAspect
            } else {
                scaleX = sourceAspect / targetAspect
                scaleY = 1f
            }
            Matrix.scaleM(matrix, 0, scaleX, scaleY, 1f)
        }

        return matrix
    }

    private fun updateCaptureSize() {
        val targetAspectRatio = captureAspectRatio.takeIf { it > 0f } ?: run {
            val totalRotation = calculateTotalRotation()
            val isSwapped = totalRotation % 180 != 0
            val actualWidth = if (isSwapped) previewHeight else previewWidth
            val actualHeight = if (isSwapped) previewWidth else previewHeight
            actualWidth.toFloat() / actualHeight.coerceAtLeast(1).toFloat()
        }

        if (targetAspectRatio >= 1f) {
            captureWidth = maxCaptureSize
            captureHeight = (maxCaptureSize / targetAspectRatio).toInt()
        } else {
            captureHeight = maxCaptureSize
            captureWidth = (maxCaptureSize * targetAspectRatio).toInt()
        }
        captureWidth = captureWidth.coerceAtLeast(1)
        captureHeight = captureHeight.coerceAtLeast(1)
//        PLog.d(TAG, "Update capture size: ${captureWidth}x${captureHeight}, totalRotation: $totalRotation")
    }

    /**
     * 请求捕获预览帧
     * 会在下一帧渲染后捕获并通过回调返回
     */
    fun capturePreviewFrame() {
        shouldCapturePreview = true
    }

    /**
     * 内部方法：捕获当前帧为小尺寸 Bitmap
     * 必须在 GL 线程中调用
     */
    private fun capturePreviewFrameInternal(
        sourceTextureId: Int? = null,
        sourceWidth: Int = viewportWidth,
        sourceHeight: Int = viewportHeight,
        compositeWithHdf: Boolean = false
    ) {
        try {
            if (captureWidth != lastCaptureWidth || captureHeight != lastCaptureHeight) {
                initCaptureFbo()
                lastCaptureWidth = captureWidth
                lastCaptureHeight = captureHeight
            }

            if (sourceTextureId != null && sourceTextureId != 0) {
                if (compositeWithHdf) {
                    drawPostProcessComposite(captureFboId, captureWidth, captureHeight, sourceTextureId)
                } else {
                    drawFboToScreen(
                        fboId = captureFboId,
                        width = captureWidth,
                        height = captureHeight,
                        sourceTextureId = sourceTextureId,
                        targetMvpMatrix = buildTextureMvpMatrix(
                            sourceWidth = sourceWidth,
                            sourceHeight = sourceHeight,
                            targetWidth = captureWidth,
                            targetHeight = captureHeight
                        )
                    )
                }
            } else {
                drawInternal(
                    fboId = captureFboId,
                    width = captureWidth,
                    height = captureHeight,
                    targetMvpMatrix = buildMvpMatrix(captureWidth, captureHeight)
                )
            }

            // 2. 读取像素
            val pixelSize = captureWidth * captureHeight * 4
            if (pboId == 0) {
                val pbos = IntArray(1)
                GLES30.glGenBuffers(1, pbos, 0)
                pboId = pbos[0]
            }

            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pboId)
            GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, pixelSize, null, GLES30.GL_STREAM_READ)
            GLES30.glReadPixels(0, 0, captureWidth, captureHeight, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0)

            val mappedBuffer = GLES30.glMapBufferRange(
                GLES30.GL_PIXEL_PACK_BUFFER, 0, pixelSize, GLES30.GL_MAP_READ_BIT
            ) as? ByteBuffer

            if (mappedBuffer == null) {
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
                return
            }

            val bitmap = Bitmap.createBitmap(captureWidth, captureHeight, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(mappedBuffer)
            GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)

            // 翻转 Y 轴并返回
            val matrix = android.graphics.Matrix()
            matrix.preScale(1f, -1f)
            val finalBitmap = Bitmap.createBitmap(bitmap, 0, 0, captureWidth, captureHeight, matrix, false)
            bitmap.recycle()
            onPreviewFrameCaptured?.invoke(finalBitmap)

            // 3. 恢复环境
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)

        } catch (e: Exception) {
            PLog.e(TAG, "Failed to capture preview frame", e)
        }
    }

    private var lastRunMeteringTime = 0L

    private fun runMeteringInternal(
        sourceTextureId: Int? = null,
        sourceWidth: Int = viewportWidth,
        sourceHeight: Int = viewportHeight,
        compositeWithHdf: Boolean = false
    ) {
        val now = System.currentTimeMillis()
        if (now - lastRunMeteringTime < 100) return // 限制频率，每秒约 10 次
        lastRunMeteringTime = now

        if (sourceTextureId != null && sourceTextureId != 0) {
            if (compositeWithHdf) {
                drawPostProcessComposite(meteringFboId, METERING_SIZE, METERING_SIZE, sourceTextureId)
            } else {
                drawFboToScreen(
                    fboId = meteringFboId,
                    width = METERING_SIZE,
                    height = METERING_SIZE,
                    sourceTextureId = sourceTextureId,
                    targetMvpMatrix = buildTextureMvpMatrix(
                        sourceWidth = sourceWidth,
                        sourceHeight = sourceHeight,
                        targetWidth = METERING_SIZE,
                        targetHeight = METERING_SIZE
                    )
                )
            }
        } else {
            drawInternal(
                fboId = meteringFboId,
                width = METERING_SIZE,
                height = METERING_SIZE,
                targetMvpMatrix = buildMvpMatrix(METERING_SIZE, METERING_SIZE)
            )
        }

        // 2. 使用 PBO 读取像素 (减少 CPU 阻塞)
        val pixelSize = METERING_SIZE * METERING_SIZE * 4
        if (meteringPboId == 0) {
            val pbos = IntArray(1)
            GLES30.glGenBuffers(1, pbos, 0)
            meteringPboId = pbos[0]
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, meteringPboId)
            GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, pixelSize, null, GLES30.GL_STREAM_READ)
        } else {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, meteringPboId)
        }

        GLES30.glReadPixels(0, 0, METERING_SIZE, METERING_SIZE, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0)

        val mappedBuffer = GLES30.glMapBufferRange(
            GLES30.GL_PIXEL_PACK_BUFFER, 0, pixelSize, GLES30.GL_MAP_READ_BIT
        ) as? ByteBuffer

        if (mappedBuffer != null) {
            val bytesCopy = ByteArray(pixelSize)
            mappedBuffer.rewind()
            mappedBuffer.get(bytesCopy)
            GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)

            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)

            val currentFocus = focusPoint
            val currentMode = meteringMode

            try {
                if (!meteringExecutor.isShutdown) {
                    meteringExecutor.execute {
                        calculateMeteringResults(bytesCopy, currentFocus, currentMode)
                    }
                }
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to dispatch metering task", e)
            }
        } else {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
        }
    }

    private fun runDepthInputCaptureInternal(sourceTextureId: Int) {
        if (onDepthInputAvailable == null || sourceTextureId == 0) return
        if (depthInputFboId == 0 || depthInputTextureId == 0) {
            initDepthInputFbo()
        }
        if (depthInputFboId == 0 || depthInputTextureId == 0) return
        val now = System.currentTimeMillis()
        if (now - lastRunDepthInputTime < 50) return
        lastRunDepthInputTime = now

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, depthInputFboId)
        GLES30.glViewport(0, 0, DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        // 使用 Copy Shader 采集，源是已校正好的 FBO 纹理
        GLES30.glUseProgram(copyProgramId)
        
        // 计算居中裁剪矩阵 (Center Crop)
        val captureMatrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(captureMatrix, 0)
        
        // MVP 矩阵处理 glReadPixels 的 Y 轴翻转
        // 标准 Quad 是 [-1, 1]，直接 Scale -1 即可垂直镜像，无需位移
        val flipMatrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(flipMatrix, 0)
        android.opengl.Matrix.scaleM(flipMatrix, 0, 1f, -1f, 1f)
        
        GLES30.glUniformMatrix4fv(uCopyMVPMatrixLoc, 1, false, flipMatrix, 0)
        GLES30.glUniformMatrix4fv(uCopySTMatrixLoc, 1, false, captureMatrix, 0) 
        GLES30.glUniform4f(uCopyCropRectLoc, 0f, 0f, 1f, 1f)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(uCopyTextureLoc, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(aCopyPositionLoc)
        GLES30.glVertexAttribPointer(aCopyPositionLoc, POSITION_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)
        
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(aCopyTexCoordLoc)
        GLES30.glVertexAttribPointer(aCopyTexCoordLoc, TEXTURE_COORD_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, 0)

        // 确保渲染已完成
        GLES30.glFinish()

        val pixelSize = DEPTH_INPUT_SIZE * DEPTH_INPUT_SIZE * 4
        if (depthInputPboId == 0) {
            val pbos = IntArray(1)
            GLES30.glGenBuffers(1, pbos, 0)
            depthInputPboId = pbos[0]
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, depthInputPboId)
            GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, pixelSize, null, GLES30.GL_STREAM_READ)
        } else {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, depthInputPboId)
        }

        GLES30.glReadPixels(0, 0, DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0)
        val mappedBuffer = GLES30.glMapBufferRange(GLES30.GL_PIXEL_PACK_BUFFER, 0, pixelSize, GLES30.GL_MAP_READ_BIT) as? ByteBuffer
        if (mappedBuffer != null) {
            depthInputBuffer.rewind()
            depthInputBuffer.put(mappedBuffer)
            GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)
            
            val bitmap = Bitmap.createBitmap(DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, Bitmap.Config.ARGB_8888)
            depthInputBuffer.rewind()
            bitmap.copyPixelsFromBuffer(depthInputBuffer)
            onDepthInputAvailable?.invoke(bitmap)
        }

        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
    }

    private fun runAiFocusInputCaptureInternal(sourceTextureId: Int) {
        if (renderingPaused || onAiFocusInputAvailable == null || sourceTextureId == 0) return
        if (aiFocusInputFboId == 0 || aiFocusInputTextureId == 0) {
            initAiFocusInputFbo()
        }
        if (aiFocusInputFboId == 0 || aiFocusInputTextureId == 0 || copyProgramId == 0) return
        val now = System.currentTimeMillis()
        if (now - lastRunAiFocusInputTime < 300) return
        lastRunAiFocusInputTime = now

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, aiFocusInputFboId)
        GLES30.glViewport(0, 0, AI_FOCUS_INPUT_SIZE, AI_FOCUS_INPUT_SIZE)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(copyProgramId)

        val captureMatrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(captureMatrix, 0)
        val flipMatrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(flipMatrix, 0)
        android.opengl.Matrix.scaleM(flipMatrix, 0, 1f, -1f, 1f)

        GLES30.glUniformMatrix4fv(uCopyMVPMatrixLoc, 1, false, flipMatrix, 0)
        GLES30.glUniformMatrix4fv(uCopySTMatrixLoc, 1, false, captureMatrix, 0)
        GLES30.glUniform4f(uCopyCropRectLoc, 0f, 0f, 1f, 1f)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sourceTextureId)
        GLES30.glUniform1i(uCopyTextureLoc, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(aCopyPositionLoc)
        GLES30.glVertexAttribPointer(aCopyPositionLoc, POSITION_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(aCopyTexCoordLoc)
        GLES30.glVertexAttribPointer(aCopyTexCoordLoc, TEXTURE_COORD_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, 0)

        val pixelSize = AI_FOCUS_INPUT_SIZE * AI_FOCUS_INPUT_SIZE * 4
        val writeIndex = aiFocusInputPboIndex % 2
        val readIndex = (aiFocusInputPboIndex + 1) % 2

        if (aiFocusInputPboIds[0] == 0) {
            GLES30.glGenBuffers(2, aiFocusInputPboIds, 0)
            for (i in 0 until 2) {
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, aiFocusInputPboIds[i])
                GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, pixelSize, null, GLES30.GL_STREAM_READ)
            }
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
        }

        // 异步将当前帧读取到 writeIndex PBO
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, aiFocusInputPboIds[writeIndex])
        GLES30.glReadPixels(0, 0, AI_FOCUS_INPUT_SIZE, AI_FOCUS_INPUT_SIZE, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0)
        deleteAiFocusInputFence(writeIndex)
        aiFocusInputPboFences[writeIndex] = GLES30.glFenceSync(GLES30.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

        // 只在上一帧 PBO 已完成时读取，避免 GL 线程在暂停流程中被同步等待卡住。
        if (aiFocusInputPboIndex > 0 && isAiFocusInputFenceReady(readIndex)) {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, aiFocusInputPboIds[readIndex])
            val mappedBuffer = GLES30.glMapBufferRange(GLES30.GL_PIXEL_PACK_BUFFER, 0, pixelSize, GLES30.GL_MAP_READ_BIT) as? ByteBuffer
            if (mappedBuffer != null) {
                aiFocusInputBuffer.rewind()
                aiFocusInputBuffer.put(mappedBuffer)
                GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)

                val bitmap = Bitmap.createBitmap(AI_FOCUS_INPUT_SIZE, AI_FOCUS_INPUT_SIZE, Bitmap.Config.ARGB_8888)
                aiFocusInputBuffer.rewind()
                bitmap.copyPixelsFromBuffer(aiFocusInputBuffer)
                onAiFocusInputAvailable?.invoke(bitmap)
            }
            deleteAiFocusInputFence(readIndex)
        }

        aiFocusInputPboIndex++

        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
    }

    private fun isAiFocusInputFenceReady(index: Int): Boolean {
        val fence = aiFocusInputPboFences[index]
        if (fence == 0L) return false
        val result = GLES30.glClientWaitSync(fence, 0, 0)
        return result == GLES30.GL_ALREADY_SIGNALED || result == GLES30.GL_CONDITION_SATISFIED
    }

    private fun deleteAiFocusInputFence(index: Int) {
        val fence = aiFocusInputPboFences[index]
        if (fence != 0L) {
            GLES30.glDeleteSync(fence)
            aiFocusInputPboFences[index] = 0L
        }
    }

    fun setRenderingPaused(paused: Boolean) {
        renderingPaused = paused
    }

    private fun calculateMeteringResults(meteringBytes: ByteArray, focus: PointF?, mode: MeteringMode) {
        val histogram = IntArray(256)
        var weightedSumLuminance = 0.0
        var totalWeight = 0.0
        val lumaGrid = IntArray(METERING_SIZE * METERING_SIZE)

        // 根据测光模式设置权重参数
        val (weightCenter, weightEdge, radiusSq) = when (mode) {
            MeteringMode.SPOT -> Triple(100.0, 0.0, (METERING_SIZE * METERING_SIZE) / 64.0)   // 绝对统治权重，半径 METERING_SIZE/8
            MeteringMode.CENTER_WEIGHTED -> Triple(20.0, 1.0, (METERING_SIZE * METERING_SIZE) / 16.0) // 统治级权重，半径 METERING_SIZE/4
            MeteringMode.AVERAGE -> Triple(1.0, 1.0, 0.0)  // 所有像素等权
            MeteringMode.HIGHLIGHT_PRIORITY -> Triple(2.0, 1.0, (METERING_SIZE * METERING_SIZE) / 8.0) // 大区域，亮部加权在下方处理
        }

        for (y in 0 until METERING_SIZE) {
            for (x in 0 until METERING_SIZE) {
                val idx = (y * METERING_SIZE + x) * 4
                val r = meteringBytes[idx].toInt() and 0xFF
                val g = meteringBytes[idx + 1].toInt() and 0xFF
                val b = meteringBytes[idx + 2].toInt() and 0xFF

                // 计算亮度 (Rec.709)
                val luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).toInt().coerceIn(0, 255)
                histogram[luma]++
                lumaGrid[y * METERING_SIZE + x] = luma

                // 权重计算
                val spatialWeight = if (mode == MeteringMode.AVERAGE) {
                    weightEdge
                } else if (focus != null) {
                    val fx = focus.x * METERING_SIZE
                    val fy = (1.0f - focus.y) * METERING_SIZE
                    val dx = x.toDouble() - fx.toDouble()
                    val dy = y.toDouble() - fy.toDouble()
                    if (dx * dx + dy * dy < radiusSq) weightCenter else weightEdge
                } else {
                    // 无对焦点时，点测光回退到中心
                    val cx = METERING_SIZE / 2.0
                    val cy = METERING_SIZE / 2.0
                    val dx = x.toDouble() - cx
                    val dy = y.toDouble() - cy
                    if (dx * dx + dy * dy < radiusSq) weightCenter else weightEdge
                }
                // 高光优先：用亮度^2 加权，亮部像素对测光影响更大
                val weight = if (mode == MeteringMode.HIGHLIGHT_PRIORITY) {
                    val lumaNorm = luma / 255.0
                    spatialWeight * (0.1 + 0.9 * lumaNorm * lumaNorm)
                } else {
                    spatialWeight
                }

                weightedSumLuminance += luma * weight
                totalWeight += weight
            }
        }

        onHistogramUpdated?.invoke(histogram)
        onMeteringUpdated?.invoke(totalWeight, weightedSumLuminance)

        // 高光优先模式：通过寻找最亮区域的“核”来精确定位高光
        if (mode == MeteringMode.HIGHLIGHT_PRIORITY) {
            // 1. 根据直方图计算动态阈值 (Top 2% 像素，且不低于 128)
            var countP98 = 0
            var dynamicThreshold = 128
            for (i in 255 downTo 0) {
                countP98 += histogram[i]
                if (countP98 >= 20) { // 1024 * 0.02 ≈ 20 pixels
                    dynamicThreshold = i.coerceAtLeast(128)
                    break
                }
            }

            // 2. 寻找最亮“簇”的中心
            // 我们通过在 5x5 窗口内寻找最大亮度总和来定位最亮的块
            // 这能有效解决“两个相距较远的光源导致质心落在中间暗处”的问题
            var maxClusterLuma = -1.0
            var bestX = -1
            var bestY = -1
            
            val kernelSize = 2 // 半径 2 代表 5x5 窗口
            
            // 遍历所有可能的中心点
            for (y in kernelSize until METERING_SIZE - kernelSize) {
                for (x in kernelSize until METERING_SIZE - kernelSize) {
                    val centerLuma = lumaGrid[y * METERING_SIZE + x]
                    
                    // 只考虑中心像素达到阈值的候选点，减少计算量并过滤背景
                    if (centerLuma < dynamicThreshold) continue
                    
                    var clusterSum = 0.0
                    for (ky in -kernelSize..kernelSize) {
                        for (kx in -kernelSize..kernelSize) {
                            clusterSum += lumaGrid[(y + ky) * METERING_SIZE + (x + kx)]
                        }
                    }
                    
                    // 也可以加入距离权重，优先选择靠近对焦点或中心的高光
                    val bias: Double = if (focus != null) {
                        val fx = focus.x * METERING_SIZE
                        val fy = (1.0f - focus.y) * METERING_SIZE
                        val dist = hypot(x.toDouble() - fx, y.toDouble() - fy)
                        1.0 / (1.0 + dist * 0.05) // 距离越近权重越高
                    } else 1.0

                    // 引入历史偏置（滞后逻辑）：如果当前点靠近上一帧的高光点，给予额外权重加成
                    val historyBias = if (lastBestX != -1 && lastBestY != -1) {
                        val dx = x.toDouble() - lastBestX
                        val dy = y.toDouble() - lastBestY
                        if (dx * dx + dy * dy < 4.0) 1.2 else 1.0 // 半径 2.0 以内给予 20% 加成
                    } else 1.0

                    val score = clusterSum * bias * historyBias
                    if (score > maxClusterLuma) {
                        maxClusterLuma = score
                        bestX = x
                        bestY = y
                    }
                }
            }

            if (bestX != -1) {
                lastBestX = bestX
                lastBestY = bestY
                // 归一化到 0-1，注意 GL 坐标 Y 轴翻转
                val hx = bestX.toFloat() / METERING_SIZE
                val hy = 1.0f - (bestY.toFloat() / METERING_SIZE)
                onHighlightPointUpdated?.invoke(hx, hy)
            }
        }
    }

    /**
     * 获取 SurfaceTexture
     */
    fun getSurfaceTexture(): SurfaceTexture? = surfaceTexture

    /**
     * 释放资源
     */
    fun release() {
        try {
            meteringExecutor.shutdown()
        } catch (e: Exception) {
            PLog.e(TAG, "Error shutting down metering executor", e)
        }
        // 删除纹理
        if (cameraTextureId != 0) {
            GlUtils.deleteTexture(cameraTextureId)
            cameraTextureId = 0
        }
        if (lutTextureId != 0) {
            GlUtils.deleteTexture(lutTextureId)
            lutTextureId = 0
        }
        if (baselineLutTextureId != 0) {
            GlUtils.deleteTexture(baselineLutTextureId)
            baselineLutTextureId = 0
        }
        if (curveTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(curveTextureId), 0)
            curveTextureId = 0
        }
        if (baselineCurveTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(baselineCurveTextureId), 0)
            baselineCurveTextureId = 0
        }

        // 释放 HDF 实时预览资源
        if (hdfExtractBlurHProgram != 0) GLES30.glDeleteProgram(hdfExtractBlurHProgram)
        if (hdfBlurVProgram != 0) GLES30.glDeleteProgram(hdfBlurVProgram)
        if (hdfCompositeProgram != 0) GLES30.glDeleteProgram(hdfCompositeProgram)
        hdfExtractBlurHProgram = 0; hdfBlurVProgram = 0; hdfCompositeProgram = 0
        for (i in 0..1) {
            if (hdfTexId[i] != 0) GLES30.glDeleteTextures(1, intArrayOf(hdfTexId[i]), 0)
            if (hdfFboId[i] != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(hdfFboId[i]), 0)
        }
        hdfTexId = IntArray(2); hdfFboId = IntArray(2)
        hdfWidth = 0; hdfHeight = 0

        // 删除缓冲
        if (vertexBufferId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(vertexBufferId), 0)
            vertexBufferId = 0
        }
        if (texCoordBufferId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(texCoordBufferId), 0)
            texCoordBufferId = 0
        }
        if (indexBufferId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(indexBufferId), 0)
            indexBufferId = 0
        }
        if (pboId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(pboId), 0)
            pboId = 0
        }
        if (meteringPboId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(meteringPboId), 0)
            meteringPboId = 0
        }
        if (depthInputPboId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(depthInputPboId), 0)
            depthInputPboId = 0
        }
        if (aiFocusInputPboIds[0] != 0) {
            deleteAiFocusInputFence(0)
            deleteAiFocusInputFence(1)
            GLES30.glDeleteBuffers(2, aiFocusInputPboIds, 0)
            aiFocusInputPboIds[0] = 0
            aiFocusInputPboIds[1] = 0
        }

        if (meteringFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(meteringFboId), 0)
            meteringFboId = 0
        }
        if (meteringTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(meteringTextureId), 0)
            meteringTextureId = 0
        }
        if (captureFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(captureFboId), 0)
            captureFboId = 0
        }
        if (captureTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(captureTextureId), 0)
            captureTextureId = 0
        }
        if (depthInputFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(depthInputFboId), 0)
            depthInputFboId = 0
        }
        if (depthInputTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(depthInputTextureId), 0)
            depthInputTextureId = 0
        }
        if (aiFocusInputFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(aiFocusInputFboId), 0)
            aiFocusInputFboId = 0
        }
        if (aiFocusInputTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(aiFocusInputTextureId), 0)
            aiFocusInputTextureId = 0
        }
        if (recordFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(recordFboId), 0)
            recordFboId = 0
        }
        if (recordTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(recordTextureId), 0)
            recordTextureId = 0
        }
        if (fboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
            fboId = 0
        }
        if (fboTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(fboTextureId), 0)
            fboTextureId = 0
        }
        if (stackFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(stackFboId), 0)
            stackFboId = 0
        }
        if (stackTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(stackTextureId), 0)
            stackTextureId = 0
        }
        recordFboWidth = 0
        recordFboHeight = 0

        // 删除程序
        GlUtils.deleteProgram(programId)
        programId = 0
        GlUtils.deleteProgram(secondPassProgramId)
        secondPassProgramId = 0
        GlUtils.deleteProgram(passthroughProgramId)
        passthroughProgramId = 0
        GlUtils.deleteProgram(copyProgramId)
        copyProgramId = 0

        // 释放 SurfaceTexture
        surfaceTexture?.release()
        surfaceTexture = null

        // 重置状态
        surfaceReady = false
        pendingLutConfig = null
        pendingBaselineLutConfig = null
        currentLutConfig = null
        currentBaselineLutConfig = null
        mainPassLocations = null
        secondPassLocations = null
    }

    /**
     * 获取当前色彩配方参数对象
     */
    private fun getCurrentRecipeParams(): com.hinnka.mycamera.model.ColorRecipeParams {
        return com.hinnka.mycamera.model.ColorRecipeParams(
            lutIntensity = lutIntensity,
            exposure = exposure,
            contrast = contrast,
            saturation = saturation,
            temperature = temperature,
            tint = tint,
            color = vibrance,
            highlights = highlights,
            shadows = shadows,
            toneToe = toneToe,
            toneShoulder = toneShoulder,
            tonePivot = tonePivot,
            fade = fade,
            filmGrain = filmGrain,
            vignette = vignette,
            bleachBypass = bleachBypass,
            chromaticAberration = chromaticAberration,
            halation = halation,
            noise = noise,
            lowRes = lowRes,
            skinHue = lchHueAdjustments[0],
            skinChroma = lchChromaAdjustments[0],
            skinLightness = lchLightnessAdjustments[0],
            redHue = lchHueAdjustments[1],
            redChroma = lchChromaAdjustments[1],
            redLightness = lchLightnessAdjustments[1],
            orangeHue = lchHueAdjustments[2],
            orangeChroma = lchChromaAdjustments[2],
            orangeLightness = lchLightnessAdjustments[2],
            yellowHue = lchHueAdjustments[3],
            yellowChroma = lchChromaAdjustments[3],
            yellowLightness = lchLightnessAdjustments[3],
            greenHue = lchHueAdjustments[4],
            greenChroma = lchChromaAdjustments[4],
            greenLightness = lchLightnessAdjustments[4],
            cyanHue = lchHueAdjustments[5],
            cyanChroma = lchChromaAdjustments[5],
            cyanLightness = lchLightnessAdjustments[5],
            blueHue = lchHueAdjustments[6],
            blueChroma = lchChromaAdjustments[6],
            blueLightness = lchLightnessAdjustments[6],
            purpleHue = lchHueAdjustments[7],
            purpleChroma = lchChromaAdjustments[7],
            purpleLightness = lchLightnessAdjustments[7],
            magentaHue = lchHueAdjustments[8],
            magentaChroma = lchChromaAdjustments[8],
            magentaLightness = lchLightnessAdjustments[8],
            masterCurvePoints = masterCurvePoints,
            redCurvePoints = redCurvePoints,
            greenCurvePoints = greenCurvePoints,
            blueCurvePoints = blueCurvePoints,
            primaryRedHue = primaryRedHue,
            primaryRedSaturation = primaryRedSaturation,
            primaryRedLightness = primaryRedLightness,
            primaryGreenHue = primaryGreenHue,
            primaryGreenSaturation = primaryGreenSaturation,
            primaryGreenLightness = primaryGreenLightness,
            primaryBlueHue = primaryBlueHue,
            primaryBlueSaturation = primaryBlueSaturation,
            primaryBlueLightness = primaryBlueLightness,
        )
    }

    private fun mapLutCurve(curve: TransferCurve?): Int {
        return (curve ?: TransferCurve.SRGB).shaderId
    }

    private fun mapLogCurve(curve: TransferCurve): Int {
        return curve.shaderId
    }

    private fun mapRawColorSpace(colorSpace: ColorSpace): Int {
        return when (colorSpace) {
            ColorSpace.SRGB -> 0
            ColorSpace.DCI_P3 -> 1
            ColorSpace.BT2020 -> 2
            ColorSpace.ARRI4 -> 3
            ColorSpace.AppleLog2 -> 4
            ColorSpace.ProPhoto -> 0
            ColorSpace.S_GAMUT3_CINE -> 5
            ColorSpace.ACES_AP1 -> 6
            ColorSpace.VGamut -> 7
        }
    }

    fun setLchAdjustments(hue: FloatArray, chroma: FloatArray, lightness: FloatArray) {
        for (i in 0 until LCH_COLOR_BAND_COUNT) {
            lchHueAdjustments[i] = hue.getOrElse(i) { 0f }
            lchChromaAdjustments[i] = chroma.getOrElse(i) { 0f }
            lchLightnessAdjustments[i] = lightness.getOrElse(i) { 0f }
        }
    }

    /**
     * 从 ColorRecipeParams 统一设置所有渲染参数
     */
    fun setRecipeParams(params: com.hinnka.mycamera.model.ColorRecipeParams) {
        lutIntensity = params.lutIntensity
        exposure = params.exposure
        contrast = params.contrast
        saturation = params.saturation
        temperature = params.temperature
        tint = params.tint
        fade = params.fade
        vibrance = params.color
        highlights = params.highlights
        shadows = params.shadows
        toneToe = params.toneToe
        toneShoulder = params.toneShoulder
        tonePivot = params.tonePivot
        filmGrain = params.filmGrain
        vignette = params.vignette
        bleachBypass = params.bleachBypass
        chromaticAberration = params.chromaticAberration
        halation = params.halation
        redHalation = params.redHalation
        noise = params.noise
        lowRes = params.lowRes
        primaryRedHue = params.primaryRedHue
        primaryRedSaturation = params.primaryRedSaturation
        primaryRedLightness = params.primaryRedLightness
        primaryGreenHue = params.primaryGreenHue
        primaryGreenSaturation = params.primaryGreenSaturation
        primaryGreenLightness = params.primaryGreenLightness
        primaryBlueHue = params.primaryBlueHue
        primaryBlueSaturation = params.primaryBlueSaturation
        primaryBlueLightness = params.primaryBlueLightness
        setLchAdjustments(
            floatArrayOf(
                params.skinHue, params.redHue, params.orangeHue, params.yellowHue,
                params.greenHue, params.cyanHue, params.blueHue, params.purpleHue, params.magentaHue,
            ),
            floatArrayOf(
                params.skinChroma, params.redChroma, params.orangeChroma, params.yellowChroma,
                params.greenChroma, params.cyanChroma, params.blueChroma, params.purpleChroma, params.magentaChroma,
            ),
            floatArrayOf(
                params.skinLightness, params.redLightness, params.orangeLightness, params.yellowLightness,
                params.greenLightness, params.cyanLightness, params.blueLightness, params.purpleLightness, params.magentaLightness,
            )
        )
        // 更新曲线纹理
        val masterPts = params.masterCurvePoints
        val redPts = params.redCurvePoints
        val greenPts = params.greenCurvePoints
        val bluePts = params.blueCurvePoints
        masterCurvePoints = masterPts
        redCurvePoints = redPts
        greenCurvePoints = greenPts
        blueCurvePoints = bluePts
        curveEnabled = !CurveUtils.isIdentity(masterPts, redPts, greenPts, bluePts)
        if (curveEnabled) {
            pendingCurveBuffer = CurveUtils.buildCurveTextureBuffer(masterPts, redPts, greenPts, bluePts)
        } else {
            pendingCurveBuffer = null
        }
    }

    fun updateBaselineRecipeParams(params: com.hinnka.mycamera.model.ColorRecipeParams) {
        baselineRecipeParams = params
        baselineColorRecipeEnabled = !params.isDefault()
        val masterPts = params.masterCurvePoints
        val redPts = params.redCurvePoints
        val greenPts = params.greenCurvePoints
        val bluePts = params.blueCurvePoints
        baselineCurveEnabled = !CurveUtils.isIdentity(masterPts, redPts, greenPts, bluePts)
        baselinePendingCurveBuffer = if (baselineCurveEnabled) {
            CurveUtils.buildCurveTextureBuffer(masterPts, redPts, greenPts, bluePts)
        } else {
            null
        }
    }

    private fun initBokehProgram() {
        val vs = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.SIMPLE_VERTEX_SHADER)
        val fs = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.BOKEH_FRAGMENT_SHADER)
        bokehProgramId = GlUtils.linkProgram(vs, fs)
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)

        if (bokehProgramId != 0) {
            uBokehInputTexLoc = GLES30.glGetUniformLocation(bokehProgramId, "uInputTexture")
            uBokehDepthTexLoc = GLES30.glGetUniformLocation(bokehProgramId, "uDepthTexture")
            uBokehDepthMatrixLoc = GLES30.glGetUniformLocation(bokehProgramId, "uDepthMatrix")
            uBokehMaxBlurRadiusLoc = GLES30.glGetUniformLocation(bokehProgramId, "uMaxBlurRadius")
            uBokehApertureLoc = GLES30.glGetUniformLocation(bokehProgramId, "uAperture")
            uBokehFocusDepthLoc = GLES30.glGetUniformLocation(bokehProgramId, "uFocusDepth")
            uBokehTexelSizeLoc = GLES30.glGetUniformLocation(bokehProgramId, "uTexelSize")
            aBokehPositionLoc = GLES30.glGetAttribLocation(bokehProgramId, "aPosition")
            aBokehTexCoordLoc = GLES30.glGetAttribLocation(bokehProgramId, "aTexCoord")
        }
    }

    private fun ensureHdfPrograms(): Boolean {
        if (hdfExtractBlurHProgram != 0 &&
            hdfBlurVProgram != 0 &&
            hdfCompositeProgram != 0 &&
            halationExtractBlurHProgram != 0 &&
            halationBlurVProgram != 0
        ) {
            return true
        }
        initHdfPrograms()
        return hdfExtractBlurHProgram != 0 &&
            hdfBlurVProgram != 0 &&
            hdfCompositeProgram != 0 &&
            halationExtractBlurHProgram != 0 &&
            halationBlurVProgram != 0
    }

    private fun ensureBokehProgram(): Boolean {
        if (bokehProgramId != 0) return true
        initBokehProgram()
        return bokehProgramId != 0
    }

    private fun ensureFocusPeakingProgram(): Boolean {
        if (focusPeakingProgramId != 0) return true

        val vs = GlUtils.compileShader(GLES30.GL_VERTEX_SHADER, Shaders.SIMPLE_VERTEX_SHADER)
        val peakFrag = GlUtils.compileShader(GLES30.GL_FRAGMENT_SHADER, Shaders.FRAGMENT_SHADER_FOCUS_PEAKING)
        focusPeakingProgramId = GlUtils.linkProgram(vs, peakFrag)
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(peakFrag)
        if (focusPeakingProgramId != 0) {
            uPeakInputTexLoc = GLES30.glGetUniformLocation(focusPeakingProgramId, "uInputTexture")
            uPeakTexelSizeLoc = GLES30.glGetUniformLocation(focusPeakingProgramId, "uTexelSize")
            uPeakThresholdLoc = GLES30.glGetUniformLocation(focusPeakingProgramId, "uThreshold")
            uPeakColorLoc = GLES30.glGetUniformLocation(focusPeakingProgramId, "uPeakColor")
            aPeakPositionLoc = GLES30.glGetAttribLocation(focusPeakingProgramId, "aPosition")
            aPeakTexCoordLoc = GLES30.glGetAttribLocation(focusPeakingProgramId, "aTexCoord")
        }
        return focusPeakingProgramId != 0
    }
    
    private var uBokehDepthMatrixLoc: Int = 0

    private fun initBokehFbo(width: Int, height: Int) {
        if (bokehFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(bokehFboId), 0)
            GLES30.glDeleteTextures(1, intArrayOf(bokehTextureId), 0)
        }

        val fbo = IntArray(1)
        val tex = IntArray(1)
        GLES30.glGenFramebuffers(1, fbo, 0)
        GLES30.glGenTextures(1, tex, 0)
        bokehFboId = fbo[0]
        bokehTextureId = tex[0]

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, bokehTextureId)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, width, height, 0, GLES30.GL_RGBA, GLES30.GL_HALF_FLOAT, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, bokehFboId)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, bokehTextureId, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

        bokehFboWidth = width
        bokehFboHeight = height
    }

    private fun updateDepthTexture() {
        val bitmap = depthMap
        if (bitmap != null && bitmap != lastDepthMap) {
            if (depthTextureId == 0) {
                val tex = IntArray(1)
                GLES30.glGenTextures(1, tex, 0)
                depthTextureId = tex[0]
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            }
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
            GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
            lastDepthMap = bitmap
        }
    }

    private fun getFocusDepth(): Float {
        val bitmap = depthMap ?: return 0.5f
        val fp = focusPoint ?: PointF(0.5f, 0.5f)
        val px = (fp.x * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
        val py = (fp.y * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
        val color = try { bitmap.getPixel(px, py) } catch (e: Exception) { 0 }
        return (color shr 16 and 0xFF) / 255.0f
    }

    private fun renderBokehPreview(inputTexId: Int, width: Int, height: Int): Int {
        if (depthMap == null || !ensureBokehProgram()) {
            return inputTexId
        }

        // 降采样优化：虚化计算不需要全分辨率
        val renderWidth = (width * bokehRenderScale).toInt()
        val renderHeight = (height * bokehRenderScale).toInt()

        if (renderWidth != bokehFboWidth || renderHeight != bokehFboHeight) {
            initBokehFbo(renderWidth, renderHeight)
            bokehFboWidth = renderWidth
            bokehFboHeight = renderHeight
        }

        updateDepthTexture()
        if (depthTextureId == 0) return inputTexId

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, bokehFboId)
        GLES30.glViewport(0, 0, renderWidth, renderHeight)
        
        GLES30.glUseProgram(bokehProgramId)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexId)
        GLES30.glUniform1i(uBokehInputTexLoc, 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
        GLES30.glUniform1i(uBokehDepthTexLoc, 1)

        val depthMatrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(depthMatrix, 0)
        // Y-flip: depth texture top is at Y=0 (GLUtils convention)
        android.opengl.Matrix.translateM(depthMatrix, 0, 0f, 1f, 0f)
        android.opengl.Matrix.scaleM(depthMatrix, 0, 1f, -1f, 1f)
        GLES30.glUniformMatrix4fv(uBokehDepthMatrixLoc, 1, false, depthMatrix, 0)

        // 性能优化建议：对于高分辨率预览，在 1/2 或更低分辨率 FBO 进行虚化计算
        // 这里根据实验将模糊半径根据分辨率对齐
        GLES30.glUniform1f(uBokehMaxBlurRadiusLoc, renderWidth.toFloat() / 25.0f) 
        GLES30.glUniform1f(uBokehApertureLoc, aperture)
        GLES30.glUniform1f(uBokehFocusDepthLoc, getFocusDepth())
        GLES30.glUniform2f(uBokehTexelSizeLoc, 1.0f / renderWidth, 1.0f / renderHeight)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(aBokehPositionLoc)
        GLES30.glVertexAttribPointer(aBokehPositionLoc, 2, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(aBokehTexCoordLoc)
        GLES30.glVertexAttribPointer(aBokehTexCoordLoc, 2, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, 0)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        return bokehTextureId
    }

    private fun renderDepthDebug(width: Int, height: Int) {
        updateDepthTexture()
        if (depthTextureId == 0) return

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, width, height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        // 使用 Copy Shader 绘制深度纹理
        GLES30.glUseProgram(copyProgramId)

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
        GLES30.glUniform1i(uCopyTextureLoc, 0)

        // 调试预览也要应用 Center Crop，否则 256x256 的正方形强行拉伸到屏幕会长得很难看
        val captureMatrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(captureMatrix, 0)
        android.opengl.Matrix.translateM(captureMatrix, 0, 0.5f, 0.5f, 0f)
        val aspect = width.toFloat() / height.toFloat()
        if (aspect > 1f) {
            android.opengl.Matrix.scaleM(captureMatrix, 0, 1f / aspect, 1f, 1f)
        } else {
            android.opengl.Matrix.scaleM(captureMatrix, 0, 1f, aspect, 1f)
        }
        android.opengl.Matrix.translateM(captureMatrix, 0, -0.5f, -0.5f, 0f)

        val flipMatrix = FloatArray(16)
        android.opengl.Matrix.setIdentityM(flipMatrix, 0)
        // 渲染到屏幕时，应用 Y 轴翻转，使调试图看起来是正的
        android.opengl.Matrix.scaleM(flipMatrix, 0, 1f, -1f, 1f)
        
        GLES30.glUniformMatrix4fv(uCopyMVPMatrixLoc, 1, false, flipMatrix, 0)
        GLES30.glUniformMatrix4fv(uCopySTMatrixLoc, 1, false, captureMatrix, 0)
        GLES30.glUniform4f(uCopyCropRectLoc, 0f, 0f, 1f, 1f)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(aCopyPositionLoc)
        GLES30.glVertexAttribPointer(aCopyPositionLoc, POSITION_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(aCopyTexCoordLoc)
        GLES30.glVertexAttribPointer(aCopyTexCoordLoc, TEXTURE_COORD_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, 0)
    }

    private fun renderFocusPeaking(inputTextureId: Int, width: Int, height: Int): Int {
        if (!ensureFocusPeakingProgram()) return inputTextureId

        ensureFocusPeakingFbo(width, height)
        if (focusPeakingFboId == 0) return inputTextureId

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, focusPeakingFboId)
        GLES30.glViewport(0, 0, width, height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(focusPeakingProgramId)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTextureId)
        GLES30.glUniform1i(uPeakInputTexLoc, 0)

        GLES30.glUniform2f(uPeakTexelSizeLoc, 1.0f / width, 1.0f / height)
        GLES30.glUniform1f(uPeakThresholdLoc, 0.8f)
        GLES30.glUniform3f(uPeakColorLoc, 1.0f, 0.1f, 0.1f)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vertexBufferId)
        GLES30.glEnableVertexAttribArray(aPeakPositionLoc)
        GLES30.glVertexAttribPointer(aPeakPositionLoc, POSITION_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, texCoordBufferId)
        GLES30.glEnableVertexAttribArray(aPeakTexCoordLoc)
        GLES30.glVertexAttribPointer(aPeakTexCoordLoc, TEXTURE_COORD_COMPONENT_COUNT, GLES30.GL_FLOAT, false, 0, 0)

        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, indexBufferId)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, 0)

        GLES30.glDisableVertexAttribArray(aPeakPositionLoc)
        GLES30.glDisableVertexAttribArray(aPeakTexCoordLoc)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

        return focusPeakingTextureId
    }

    private fun ensureFocusPeakingFbo(width: Int, height: Int) {
        if (focusPeakingFboWidth == width && focusPeakingFboHeight == height && focusPeakingFboId != 0) return

        if (focusPeakingFboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(focusPeakingFboId), 0)
        }
        if (focusPeakingTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(focusPeakingTextureId), 0)
        }

        val ids = IntArray(1)
        GLES30.glGenFramebuffers(1, ids, 0)
        focusPeakingFboId = ids[0]

        GLES30.glGenTextures(1, ids, 0)
        focusPeakingTextureId = ids[0]

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, focusPeakingTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, width, height,
            0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, focusPeakingFboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, focusPeakingTextureId, 0
        )

        focusPeakingFboWidth = width
        focusPeakingFboHeight = height
    }
}
