package com.hinnka.mycamera.processor

import android.graphics.ImageFormat
import android.graphics.Rect
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLES31
import com.hinnka.mycamera.model.SafeImage
import com.hinnka.mycamera.raw.DngHdrProfileGainTableGenerator
import com.hinnka.mycamera.raw.DngPhotonProfileGainTableGenerator
import com.hinnka.mycamera.raw.DngPgtmDiagnostic
import com.hinnka.mycamera.raw.RawProfileToneMapMode
import com.hinnka.mycamera.utils.LargeDirectBuffer
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt

class GlesRawStacker(
    private val width: Int,
    private val height: Int,
    private val cfaPattern: Int,
    blackLevel: FloatArray,
    whiteLevel: Int,
    noiseModel: FloatArray,
    private val rawNoiseModel: RawNoiseModel = RawNoiseModel.fromLegacyNoiseModel(noiseModel),
    private val lensShading: FloatArray?,
    private val lensShadingWidth: Int,
    private val lensShadingHeight: Int,
    colorCorrectionMatrix: FloatArray? = null,
    pgtmStatsBounds: Rect? = null,
    private val profileToneMapMode: RawProfileToneMapMode = RawProfileToneMapMode.GooglePixel,
    private val tuning: RawStackTuningProfile = RawStackTuningResolver.resolve(RawStackMode.MFNR),
    debugConfig: RawStackDebugConfig = RawStackDebugConfig.Disabled,
) {
    private data class TextureLevel(val texture: Int, val width: Int, val height: Int)
    private data class ReadOutputTiming(
        val elapsedMs: Long,
        val glReadMs: Long,
        val copyMs: Long,
        val allocMs: Long,
        val mode: String,
    )
    private data class RegistrationAcceptance(
        val accepted: Boolean,
        val transform: RawStackPerspectiveTransform,
        val reason: String,
        val srTransform: RawStackPerspectiveTransform,
        val srTransformSource: String,
        val srWeight: Float,
        val srSeedWeight: Float,
        val srConsistencyWeight: Float,
    )
    private data class RegistrationSrGate(
        val weight: Float,
        val seedWeight: Float,
        val consistencyWeight: Float,
    ) {
        companion object {
            val Zero = RegistrationSrGate(
                weight = 0.0f,
                seedWeight = 0.0f,
                consistencyWeight = 0.0f,
            )
        }
    }
    private data class SuperResolutionOutputDecision(
        val mode: String,
        val fallbackReason: String?,
        val detailFrameCount: Int,
        val detailWeightSum: Float,
    ) {
        val forceBaseUpscale: Boolean
            get() = fallbackReason != null

        companion object {
            val Disabled = SuperResolutionOutputDecision(
                mode = "n/a",
                fallbackReason = null,
                detailFrameCount = 0,
                detailWeightSum = Float.NaN,
            )
        }
    }
    private data class RegistrationTranslation(
        val dx: Float,
        val dy: Float,
    ) {
        val magnitude: Float
            get() = kotlin.math.sqrt(dx * dx + dy * dy)

        fun distanceTo(other: RegistrationTranslation): Float {
            val ddx = dx - other.dx
            val ddy = dy - other.dy
            return kotlin.math.sqrt(ddx * ddx + ddy * ddy)
        }
    }

    data class HdrInputFrame(
        val image: SafeImage,
        val exposureProduct: Double,
    )

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE

    private val textures = ArrayList<Int>()
    private val programs = ArrayList<Int>()
    private val framebuffers = ArrayList<Int>()
    private val buffers = ArrayList<Int>()

    private val normalizedBlackLevel = FloatArray(4) { index ->
        blackLevel.getOrElse(index) { blackLevel.firstOrNull() ?: 0f }
    }
    private val normalizedWhiteLevel = whiteLevel.coerceAtLeast(1).toFloat()
    private val normalizedNoiseAlphaByChannel = rawNoiseModel.normalizedShotNoiseForShader()
    private val normalizedNoiseBetaByChannel = rawNoiseModel.normalizedReadNoiseForShader()
    private val noiseAlpha = rawNoiseModel.greenShotNoise.coerceAtLeast(0f) / 65535.0f
    private val noiseBeta = rawNoiseModel.greenReadNoise.coerceAtLeast(0f) / (65535.0f * 65535.0f)
    private val pgtmColorCorrectionMatrix = FloatArray(9) { index ->
        colorCorrectionMatrix
            ?.getOrNull(index)
            ?.takeIf { it.isFinite() }
            ?: if (index % 4 == 0) 1f else 0f
    }
    private val pgtmStatsBounds = sanitizePgtmStatsBounds(pgtmStatsBounds)
    private val hwmfPrefilter = tuning.prefilter
    private val hwmfBlend = tuning.blend
    private val hwmfPostfilter = tuning.postfilter
    private val hwmfHdr = tuning.hdr
    private val hwmfSr = tuning.superResolution
    private val hwmfDebug = debugConfig.normalized()
    private val registrationSetup = RawStackRegistrationResolver.resolve(width, height)
    private val registrationSummary = registrationSetup.toSummary()
    private val superResolutionEnabled = tuning.mode == RawStackMode.MFSR
    private val superResolutionScale = if (superResolutionEnabled) {
        hwmfSr.internalScale.coerceIn(1.0f, 2.0f)
    } else {
        1.0f
    }
    private val outputWidth = scaledRawDimension(width, superResolutionScale)
    private val outputHeight = scaledRawDimension(height, superResolutionScale)
    private val pyramidLevels = max(
        hwmfPrefilter.pyramidLevels.coerceAtLeast(1),
        registrationSetup.requiredPyramidLevels(),
    )
    private val alignLevel = hwmfPrefilter.alignLevel.coerceAtLeast(0)
    private val flowGridSpacing = hwmfPrefilter.flowGridSpacing.coerceAtLeast(1)
    private val lkRefinePasses = hwmfBlend.lkRefinePasses.coerceAtLeast(0)
    private val flowSmoothPasses = hwmfBlend.flowSmoothPasses.coerceAtLeast(0)

    private val planeWidth = max(1, width / 2)
    private val planeHeight = max(1, height / 2)
    private var gridWidth = 0
    private var gridHeight = 0

    private var proxyProgram = 0
    private var downsampleProgram = 0
    private var alignProgram = 0
    private var hdrShortGlobalAlignProgram = 0
    private var registrationGlobalAlignProgram = 0
    private var lkRefineProgram = 0
    private var smoothFlowProgram = 0
    private var structureProgram = 0
    private var robustnessProgram = 0
    private var tileMaskProgram = 0
    private var registrationSampleProgram = 0
    private var clearAccumulatorProgram = 0
    private var accumulateProgram = 0
    private var normalizeProgram = 0
    private var clearSuperResolutionAccumulatorProgram = 0
    private var accumulateSuperResolutionProgram = 0
    private var normalizeSuperResolutionProgram = 0
    private var hdrRecoveryMaskProgram = 0
    private var hdrRecoveryDilateProgram = 0
    private var hdrRecoveryFeatherProgram = 0
    private var pgtmStatsProgram = 0
    private var diagnosticAlignmentProgram = 0
    private var diagnosticFinalProgram = 0
    private var diagnosticBuffer = 0
    private var diagnosticsFailed = false

    private var renderFbo = 0
    private var readbackFbo = 0

    private var refRaw = 0
    private var curRaw = 0
    private var hdrShortRaw = 0
    private var refProxy = 0
    private var curProxy = 0
    private var flowTexture = 0
    private var flowScratchTexture = 0
    private var hdrShortAlignmentTexture = 0
    private var kernelTexture = 0
    private var robustnessTexture = 0
    private var tileMaskTexture = 0
    private var hdrRecoveryOrigMaskTexture = 0
    private var hdrRecoveryMaskTexture = 0
    private var hdrRecoveryScratchTexture = 0
    private var accumulatorTexture = 0
    private var accumulatorScratchTexture = 0
    private var currentAccumulatorTexture = 0
    private var superResolutionAccumulatorTexture = 0
    private var superResolutionAccumulatorScratchTexture = 0
    private var currentSuperResolutionAccumulatorTexture = 0
    private var superResolutionBaseTexture = 0
    private var outputTexture = 0
    private var lensShadingTexture = 0
    private var currentRegistrationTransform = registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
    private var currentRegistrationSrTransform = registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
    private var currentRegistrationSrWeight = 1.0f
    private var registrationEstimateCount = 0
    private var registrationConfidenceSum = 0f
    private var registrationConfidenceMin = Int.MAX_VALUE
    private var registrationForceIdentityCount = 0
    private var registrationInlierRatioSum = 0f
    private var registrationResidualP90Max = Float.NaN
    private var registrationGlobalScoreSum = 0f
    private var registrationGlobalScoreCount = 0
    private var registrationGlobalMarginMin = Float.POSITIVE_INFINITY
    private var registrationGlobalCoverageSum = 0f
    private val acceptedRegistrationTranslations = ArrayList<RegistrationTranslation>()
    private val acceptedSrRegistrationTranslations = ArrayList<RegistrationTranslation>()
    private var superResolutionDetailFrameCount = 0
    private var superResolutionDetailWeightSum = 0f
    private var superResolutionDetailWeightMax = 0f

    fun process(images: List<SafeImage>): RawStackResult? {
        if (images.isEmpty() || width <= 0 || height <= 0) return null
        if (images.any { it.width != width || it.height != height }) {
            PLog.w(TAG, "GLES RAW stack got mixed frame sizes")
            images.forEach { it.close() }
            return null
        }
        if (images.any { it.format != ImageFormat.RAW_SENSOR }) {
            PLog.w(TAG, "GLES RAW stack only supports RAW_SENSOR input")
            images.forEach { it.close() }
            return null
        }

        val outputByteCount = outputWidth.toLong() * outputHeight.toLong() * 2L
        var outputBuffer: ByteBuffer? = null
        var returned = false
        val startTime = System.currentTimeMillis()
        val originalThreadPriority = GlesGpuScheduler.lowerCurrentThreadPriority(TAG)
        return try {
            outputBuffer = LargeDirectBuffer.allocate(outputByteCount, "GLES RAW fused Bayer")
                ?.order(ByteOrder.nativeOrder()) ?: return null

            initEgl()
            ensureGles31()
            validateOutputTextureLimits()
            initPrograms()
            initResources()
            applyRawRenderState()
            RawStackRuntimeDebug.d(TAG) {
                "HWMF RAW stack mode=${tuning.mode} frames=${images.size} " +
                    "out=${outputWidth}x$outputHeight srScale=${superResolutionScale.formatScale()} " +
                    "${registrationSummary.compactSummary()} regProxy=REG_OUT " +
                    "regCtx=${registrationSetup.referenceContextSummary()} " +
                    "prefilterLevels=$pyramidLevels flowGrid=$flowGridSpacing " +
                    "blendLk=$lkRefinePasses blendSmooth=$flowSmoothPasses"
            }
            images[0].use {
                uploadRawTexture(it, refRaw, "reference")
            }
            buildProxy(refRaw, refProxy, "reference")
            val refPyramid = createPyramid(refProxy)
            val curPyramid = createPyramid(curProxy)
            buildPyramid(refPyramid)
            computeStructureTensor()
            currentRegistrationTransform = registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
            currentRegistrationSrTransform = registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
            currentRegistrationSrWeight = 1.0f
            acceptedRegistrationTranslations.clear()
            acceptedSrRegistrationTranslations.clear()
            resetSuperResolutionDecisionStats()
            clearAccumulator()
            if (superResolutionEnabled) {
                clearSuperResolutionAccumulator()
                accumulateFrame(refRaw, isReference = true)
                accumulateSuperResolutionFrame(refRaw, isReference = true)
            } else {
                accumulateFrame(refRaw, isReference = true)
            }
            GlesGpuScheduler.yieldToUiRenderer()

            var alignedFrameCount = 0
            for (index in 1 until images.size) {
                images[index].use {
                    uploadRawTexture(it, curRaw, "frame $index")
                }
                buildProxy(curRaw, curProxy, "frame $index")
                buildPyramid(curPyramid)
                alignCurrentToReference(refPyramid, curPyramid)
                refineFlow()
                smoothFlow()
                computeRobustness()
                computeTileMask()
                val registrationAccepted = estimateCurrentRegistration(refPyramid, curPyramid, frameIndex = index)
                if (!registrationAccepted) {
                    GlesGpuScheduler.yieldToUiRenderer()
                    continue
                }
                if (superResolutionEnabled) {
                    accumulateFrame(curRaw, isReference = false)
                    if (currentRegistrationSrWeight > 0.0f) {
                        recordSuperResolutionDetailFrame(currentRegistrationSrWeight)
                        accumulateSuperResolutionFrame(curRaw, isReference = false)
                    }
                } else {
                    accumulateFrame(curRaw, isReference = false)
                }
                if (hwmfDebug.collectMetrics) {
                    recordAlignmentDiagnostics()
                }
                alignedFrameCount += 1
                GlesGpuScheduler.yieldToUiRenderer()
            }

            val superResolutionDecision = if (superResolutionEnabled) {
                decideSuperResolutionOutput(alignedFrameCount)
            } else {
                SuperResolutionOutputDecision.Disabled
            }
            if (superResolutionEnabled) {
                normalizeSuperResolutionOutput(superResolutionDecision)
            } else {
                normalizeOutput()
            }
            GlesGpuScheduler.yieldToUiRenderer()
            val readTiming = readOutput(outputBuffer)
            outputBuffer.rewind()
            val diagnostics = if (hwmfDebug.collectMetrics) {
                collectFinalDiagnostics(
                    frameCount = images.size,
                    alignedFrameCount = alignedFrameCount,
                    elapsedMs = System.currentTimeMillis() - startTime,
                    superResolutionDecision = superResolutionDecision,
                )
            } else {
                null
            }
            returned = true
            RawStackRuntimeDebug.i(TAG) {
                "GLES RAW stacking completed in ${System.currentTimeMillis() - startTime}ms " +
                    "readback=${readTiming.elapsedMs}ms glRead=${readTiming.glReadMs}ms " +
                    "copy=${readTiming.copyMs}ms alloc=${readTiming.allocMs}ms mode=${readTiming.mode}"
            }
            RawStackResult(
                fusedBayerBuffer = outputBuffer,
                width = outputWidth,
                height = outputHeight,
                isNormalizedSensorData = true,
                blackLevel = normalizedBlackLevel.copyOf(),
                fusedBayerUsesNativeAllocator = true,
                diagnostics = diagnostics,
            )
        } catch (e: Exception) {
            PLog.e(TAG, "GLES RAW stacking failed", e)
            null
        } finally {
            images.forEach { it.close() }
            release()
            GlesGpuScheduler.restoreCurrentThreadPriority(originalThreadPriority, TAG)
            if (!returned) {
                LargeDirectBuffer.free(outputBuffer)
            }
        }
    }

    fun processHdr(shortFrame: HdrInputFrame, normalFrames: List<HdrInputFrame>): RawStackResult? {
        val allImages = buildList {
            add(shortFrame.image)
            normalFrames.forEach { add(it.image) }
        }
        if (normalFrames.isEmpty() || width <= 0 || height <= 0) {
            allImages.forEach { it.close() }
            return null
        }
        if (allImages.any { it.width != width || it.height != height }) {
            PLog.w(TAG, "GLES RAW HDR stack got mixed frame sizes")
            allImages.forEach { it.close() }
            return null
        }
        if (allImages.any { it.format != ImageFormat.RAW_SENSOR }) {
            PLog.w(TAG, "GLES RAW HDR stack only supports RAW_SENSOR input")
            allImages.forEach { it.close() }
            return null
        }

        val outputByteCount = width.toLong() * height.toLong() * 2L
        var outputBuffer: ByteBuffer? = null
        var returned = false
        val startTime = System.currentTimeMillis()
        val originalThreadPriority = GlesGpuScheduler.lowerCurrentThreadPriority(TAG)
        return try {
            outputBuffer = LargeDirectBuffer.allocate(outputByteCount, "GLES RAW HDR fused Bayer")
                ?.order(ByteOrder.nativeOrder()) ?: return null

            initEgl()
            ensureGles31()
            initPrograms()
            initHdrPrograms()
            initResources()
            applyRawRenderState()
            RawStackRuntimeDebug.d(TAG) {
                "HWMF RAW HDR stack mode=${tuning.mode} normalFrames=${normalFrames.size} " +
                    "${registrationSummary.compactSummary()} regProxy=REG_OUT " +
                    "regCtx=${registrationSetup.referenceContextSummary()} " +
                    "prefilterLevels=$pyramidLevels flowGrid=$flowGridSpacing " +
                    "blendLk=$lkRefinePasses blendSmooth=$flowSmoothPasses"
            }

            val shortExposureProduct = validExposureProduct(shortFrame.exposureProduct)
            val referenceExposureProduct = validExposureProduct(normalFrames.first().exposureProduct)
            val referenceOutputScale = hdrExposureScale(shortExposureProduct, referenceExposureProduct)
            val shortAlignmentScale = hdrExposureScale(referenceExposureProduct, shortExposureProduct)
            val rawHdrBaselineExposureEv = hdrBaselineExposureEv(shortExposureProduct, referenceExposureProduct)
            val alignmentScales = normalFrames.map {
                hdrExposureScale(referenceExposureProduct, validExposureProduct(it.exposureProduct))
            }
            val outputScales = normalFrames.map {
                hdrExposureScale(shortExposureProduct, validExposureProduct(it.exposureProduct))
            }

            shortFrame.image.use {
                uploadRawTexture(it, hdrShortRaw, "short highlight recovery")
            }
            normalFrames.first().image.use {
                uploadRawTexture(it, refRaw, "normal reference")
            }
            buildProxy(refRaw, refProxy, "normal reference", exposureScale = 1.0f)
            val refPyramid = createPyramid(refProxy)
            val curPyramid = createPyramid(curProxy)
            buildPyramid(refPyramid)
            buildProxy(hdrShortRaw, curProxy, "short global alignment", exposureScale = shortAlignmentScale)
            buildPyramid(curPyramid)
            alignHdrShortToReference(refPyramid, curPyramid)
            GlesGpuScheduler.waitForGpuCheckpoint(TAG, "hdr short alignment")
            computeStructureTensor()
            clearAccumulator()
            currentRegistrationTransform = registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
            currentRegistrationSrTransform = registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
            currentRegistrationSrWeight = 1.0f
            acceptedRegistrationTranslations.clear()
            acceptedSrRegistrationTranslations.clear()
            accumulateFrame(
                rawTexture = refRaw,
                isReference = true,
                exposureScale = referenceOutputScale,
                hdrMode = true,
            )
            GlesGpuScheduler.waitForGpuCheckpoint(TAG, "hdr reference accumulation")

            var alignedFrameCount = 0
            normalFrames.drop(1).forEachIndexed { index, frame ->
                val frameIndex = index + 1
                val alignmentScale = alignmentScales[frameIndex]
                val outputScale = outputScales[frameIndex]
                frame.image.use {
                    uploadRawTexture(it, curRaw, "normal frame $frameIndex")
                }
                buildProxy(curRaw, curProxy, "normal frame $frameIndex", exposureScale = alignmentScale)
                buildPyramid(curPyramid)
                alignCurrentToReference(refPyramid, curPyramid)
                GlesGpuScheduler.waitForGpuCheckpoint(TAG, "hdr normal frame $frameIndex alignment")
                refineFlow()
                smoothFlow()
                computeRobustness()
                computeTileMask()
                val registrationAccepted = estimateCurrentRegistration(refPyramid, curPyramid, frameIndex = frameIndex)
                if (!registrationAccepted) {
                    GlesGpuScheduler.waitForGpuCheckpoint(TAG, "hdr normal frame $frameIndex registration rejected")
                    return@forEachIndexed
                }
                accumulateFrame(
                    rawTexture = curRaw,
                    isReference = false,
                    exposureScale = outputScale,
                    hdrMode = true,
                )
                if (hwmfDebug.collectMetrics) {
                    recordAlignmentDiagnostics()
                }
                alignedFrameCount += 1
                GlesGpuScheduler.waitForGpuCheckpoint(TAG, "hdr normal frame $frameIndex accumulation")
            }

            computeHdrRecoveryMask(shortToReferenceExposureScale = shortAlignmentScale)
            GlesGpuScheduler.yieldToUiRenderer()
            normalizeOutput(
                hdrMode = true,
                referenceExposureScale = referenceOutputScale,
                shortExposureScale = 1.0f,
            )
            GlesGpuScheduler.waitForGpuCheckpoint(TAG, "hdr normalize")
            val readTiming = readOutput(outputBuffer)
            outputBuffer.rewind()
            val profileGainTableMap = computeHdrProfileGainTableMap(rawHdrBaselineExposureEv)
            val diagnostics = if (hwmfDebug.collectMetrics) {
                collectFinalDiagnostics(
                    frameCount = normalFrames.size + 1,
                    alignedFrameCount = alignedFrameCount,
                    elapsedMs = System.currentTimeMillis() - startTime,
                )
            } else {
                null
            }
            returned = true
            RawStackRuntimeDebug.i(TAG) {
                "GLES RAW HDR stacking completed in ${System.currentTimeMillis() - startTime}ms " +
                    "readback=${readTiming.elapsedMs}ms glRead=${readTiming.glReadMs}ms " +
                    "copy=${readTiming.copyMs}ms alloc=${readTiming.allocMs}ms mode=${readTiming.mode}"
            }
            RawStackResult(
                fusedBayerBuffer = outputBuffer,
                width = width,
                height = height,
                isNormalizedSensorData = true,
                blackLevel = normalizedBlackLevel.copyOf(),
                fusedBayerUsesNativeAllocator = true,
                profileGainTableMap = profileGainTableMap,
                profileToneMapMode = if (profileGainTableMap != null) {
                    profileToneMapMode
                } else {
                    RawProfileToneMapMode.Default
                },
                diagnostics = diagnostics,
            )
        } catch (e: Exception) {
            PLog.e(TAG, "GLES RAW HDR stacking failed", e)
            null
        } finally {
            allImages.forEach { it.close() }
            release()
            GlesGpuScheduler.restoreCurrentThreadPriority(originalThreadPriority, TAG)
            if (!returned) {
                LargeDirectBuffer.free(outputBuffer)
            }
        }
    }

    private fun initEgl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw IllegalStateException("eglGetDisplay failed")
        }
        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw IllegalStateException("eglInitialize failed: ${EGL14.eglGetError()}")
        }
        val config = chooseConfig(EGL_OPENGL_ES3_BIT_KHR) ?: chooseConfig(EGL14.EGL_OPENGL_ES2_BIT)
            ?: throw IllegalStateException("No EGL config for GLES")
        eglContext = GlesGpuScheduler.createBackgroundContext(eglDisplay, config, TAG)
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            throw IllegalStateException("eglCreateContext failed: ${EGL14.eglGetError()}")
        }
        eglSurface = EGL14.eglCreatePbufferSurface(
            eglDisplay,
            config,
            intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE),
            0,
        )
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw IllegalStateException("eglCreatePbufferSurface failed: ${EGL14.eglGetError()}")
        }
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw IllegalStateException("eglMakeCurrent failed: ${EGL14.eglGetError()}")
        }
    }

    private fun chooseConfig(renderableType: Int): EGLConfig? {
        val attribs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, renderableType,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val count = IntArray(1)
        return if (EGL14.eglChooseConfig(eglDisplay, attribs, 0, configs, 0, configs.size, count, 0) &&
            count[0] > 0
        ) {
            configs[0]
        } else {
            null
        }
    }

    private fun ensureGles31() {
        val version = GLES30.glGetString(GLES30.GL_VERSION).orEmpty()
        if (!version.contains("OpenGL ES 3.1") && !version.contains("OpenGL ES 3.2")) {
            throw IllegalStateException("GLES RAW stack requires OpenGL ES 3.1+, got: $version")
        }
    }

    private fun initPrograms() {
        proxyProgram = linkComputeProgram(RAW_PROXY_COMPUTE_SHADER, "raw_proxy")
        downsampleProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, DOWNSAMPLE_FRAGMENT_SHADER, "raw_downsample")
        alignProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, ALIGN_FRAGMENT_SHADER, "raw_align")
        lkRefineProgram = linkComputeProgram(LK_REFINE_COMPUTE_SHADER, "raw_lk_refine")
        smoothFlowProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, SMOOTH_FLOW_FRAGMENT_SHADER, "raw_smooth_flow")
        structureProgram = linkComputeProgram(STRUCTURE_COMPUTE_SHADER, "raw_structure")
        robustnessProgram = linkComputeProgram(ROBUSTNESS_COMPUTE_SHADER, "raw_robustness")
        tileMaskProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, TILE_MASK_FRAGMENT_SHADER, "raw_tile_mask")
        registrationSampleProgram = linkComputeProgram(
            REGISTRATION_SAMPLE_COMPUTE_SHADER,
            "raw_registration_samples",
        )
        registrationGlobalAlignProgram = linkComputeProgram(
            REGISTRATION_GLOBAL_ALIGN_COMPUTE_SHADER,
            "raw_registration_global_align",
        )
        clearAccumulatorProgram = linkComputeProgram(CLEAR_ACCUMULATOR_COMPUTE_SHADER, "raw_clear_accumulator")
        accumulateProgram = linkComputeProgram(ACCUMULATE_COMPUTE_SHADER, "raw_accumulate")
        normalizeProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, NORMALIZE_FRAGMENT_SHADER, "raw_normalize")
        if (superResolutionEnabled) {
            clearSuperResolutionAccumulatorProgram = linkComputeProgram(
                CLEAR_ACCUMULATOR_COMPUTE_SHADER,
                "raw_sr_clear_accumulator",
            )
            accumulateSuperResolutionProgram = linkComputeProgram(
                SUPER_RESOLUTION_ACCUMULATE_COMPUTE_SHADER,
                "raw_sr_accumulate",
            )
            normalizeSuperResolutionProgram = linkGraphicsProgram(
                FULLSCREEN_VERTEX_SHADER,
                SUPER_RESOLUTION_NORMALIZE_FRAGMENT_SHADER,
                "raw_sr_normalize",
            )
        }
        initDiagnosticPrograms()
    }

    private fun initHdrPrograms() {
        if (hdrShortGlobalAlignProgram != 0) return
        hdrShortGlobalAlignProgram = linkComputeProgram(
            HDR_SHORT_GLOBAL_ALIGN_COMPUTE_SHADER,
            "raw_hdr_short_global_align",
        )
        hdrRecoveryMaskProgram = linkGraphicsProgram(
            FULLSCREEN_VERTEX_SHADER,
            HDR_RECOVERY_MASK_FRAGMENT_SHADER,
            "raw_hdr_recovery_mask",
        )
        hdrRecoveryDilateProgram = linkGraphicsProgram(
            FULLSCREEN_VERTEX_SHADER,
            HDR_RECOVERY_DILATE_FRAGMENT_SHADER,
            "raw_hdr_recovery_dilate",
        )
        hdrRecoveryFeatherProgram = linkGraphicsProgram(
            FULLSCREEN_VERTEX_SHADER,
            HDR_RECOVERY_FEATHER_FRAGMENT_SHADER,
            "raw_hdr_recovery_feather",
        )
        pgtmStatsProgram = linkComputeProgram(PGTM_STATS_COMPUTE_SHADER, "raw_hdr_pgtm_stats")
    }

    private fun validateOutputTextureLimits() {
        val maxTextureSize = IntArray(1)
        GLES30.glGetIntegerv(GLES30.GL_MAX_TEXTURE_SIZE, maxTextureSize, 0)
        val maxSize = maxTextureSize[0].coerceAtLeast(1)
        if (outputWidth > maxSize || outputHeight > maxSize) {
            throw IllegalStateException(
                "RAW ${tuning.mode} output ${outputWidth}x$outputHeight exceeds GL_MAX_TEXTURE_SIZE=$maxSize",
            )
        }
        if (superResolutionEnabled && RawStackRuntimeDebug.enabled) {
            val accumulatorBytes = outputWidth.toLong() * outputHeight.toLong() * 8L
            val accumulatorTotalBytes = accumulatorBytes * 2L
            val baseAccumulatorBytes = width.toLong() * height.toLong() * 8L
            val baseAccumulatorTotalBytes = baseAccumulatorBytes * 2L
            val baseOutputBytes = width.toLong() * height.toLong() * 2L
            val outputBytes = outputWidth.toLong() * outputHeight.toLong() * 2L
            RawStackRuntimeDebug.d(TAG) {
                "HWMF MFSR resources out=${outputWidth}x$outputHeight maxTex=$maxSize " +
                    "srAccumulator=${accumulatorBytes.mibString()}x2 " +
                    "srAccumulatorTotal=${accumulatorTotalBytes.mibString()} " +
                    "baseAccumulator=${baseAccumulatorBytes.mibString()}x2 " +
                    "baseAccumulatorTotal=${baseAccumulatorTotalBytes.mibString()} " +
                    "baseOutput=${baseOutputBytes.mibString()} output=${outputBytes.mibString()}"
            }
        }
    }

    private fun initDiagnosticPrograms() {
        if (!hwmfDebug.collectMetrics || diagnosticsFailed) return
        try {
            diagnosticAlignmentProgram = linkComputeProgram(
                DIAGNOSTIC_ALIGNMENT_COMPUTE_SHADER,
                "raw_stack_diagnostic_alignment",
            )
            diagnosticFinalProgram = linkComputeProgram(
                DIAGNOSTIC_FINAL_COMPUTE_SHADER,
                "raw_stack_diagnostic_final",
            )
        } catch (e: Exception) {
            diagnosticsFailed = true
            diagnosticAlignmentProgram = 0
            diagnosticFinalProgram = 0
            PLog.w(TAG, "RAW stack diagnostics disabled after shader setup failure", e)
        }
    }

    private fun initResources() {
        gridWidth = (planeWidth + flowGridSpacing - 1) / flowGridSpacing
        gridHeight = (planeHeight + flowGridSpacing - 1) / flowGridSpacing

        refRaw = createTexture2D(width, height, GLES30.GL_R16UI, GLES30.GL_NEAREST)
        curRaw = createTexture2D(width, height, GLES30.GL_R16UI, GLES30.GL_NEAREST)
        hdrShortRaw = createTexture2D(width, height, GLES30.GL_R16UI, GLES30.GL_NEAREST)
        refProxy = createTexture2D(planeWidth, planeHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        curProxy = createTexture2D(planeWidth, planeHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        flowTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        flowScratchTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        hdrShortAlignmentTexture = createTexture2D(1, 1, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        kernelTexture = createTexture2D(planeWidth, planeHeight, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        robustnessTexture = createTexture2D(planeWidth, planeHeight, GLES30.GL_R32F, GLES30.GL_NEAREST)
        tileMaskTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_R16F, GLES30.GL_LINEAR)
        hdrRecoveryOrigMaskTexture = createTexture2D(planeWidth, planeHeight, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        hdrRecoveryMaskTexture = createTexture2D(planeWidth, planeHeight, GLES30.GL_R16F, GLES30.GL_NEAREST)
        hdrRecoveryScratchTexture = createTexture2D(planeWidth, planeHeight, GLES30.GL_R16F, GLES30.GL_NEAREST)
        accumulatorTexture = createTexture2D(width, height, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        accumulatorScratchTexture = createTexture2D(width, height, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        currentAccumulatorTexture = accumulatorTexture
        if (superResolutionEnabled) {
            superResolutionAccumulatorTexture = createTexture2D(
                outputWidth,
                outputHeight,
                GLES30.GL_RGBA16F,
                GLES30.GL_NEAREST,
            )
            superResolutionAccumulatorScratchTexture = createTexture2D(
                outputWidth,
                outputHeight,
                GLES30.GL_RGBA16F,
                GLES30.GL_NEAREST,
            )
            currentSuperResolutionAccumulatorTexture = superResolutionAccumulatorTexture
            superResolutionBaseTexture = createTexture2D(width, height, GLES30.GL_RG8, GLES30.GL_NEAREST)
        }
        outputTexture = createTexture2D(outputWidth, outputHeight, GLES30.GL_RG8, GLES30.GL_NEAREST)
        lensShadingTexture = createLensShadingTexture()
        renderFbo = createFramebuffer()
        readbackFbo = createFramebuffer()
        initDiagnosticResources()
    }

    private fun initDiagnosticResources() {
        if (!hwmfDebug.collectMetrics ||
            diagnosticsFailed ||
            diagnosticAlignmentProgram == 0 ||
            diagnosticFinalProgram == 0
        ) {
            return
        }
        val ids = IntArray(1)
        GLES31.glGenBuffers(1, ids, 0)
        diagnosticBuffer = ids[0]
        if (diagnosticBuffer == 0) {
            diagnosticsFailed = true
            PLog.w(TAG, "RAW stack diagnostics disabled: failed to allocate stats buffer")
            return
        }
        buffers += diagnosticBuffer
        val zero = ByteBuffer
            .allocateDirect(DIAGNOSTIC_UINT_COUNT * 4)
            .order(ByteOrder.nativeOrder())
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, diagnosticBuffer)
        GLES31.glBufferData(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            DIAGNOSTIC_UINT_COUNT * 4,
            zero,
            GLES31.GL_DYNAMIC_READ,
        )
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
        checkGlError("initDiagnosticResources")
    }

    private fun createPyramid(baseTexture: Int): List<TextureLevel> {
        val levels = ArrayList<TextureLevel>(pyramidLevels)
        levels += TextureLevel(baseTexture, planeWidth, planeHeight)
        var levelWidth = planeWidth
        var levelHeight = planeHeight
        repeat(pyramidLevels - 1) {
            levelWidth = max(1, (levelWidth + 1) / 2)
            levelHeight = max(1, (levelHeight + 1) / 2)
            levels += TextureLevel(
                createTexture2D(levelWidth, levelHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR),
                levelWidth,
                levelHeight,
            )
        }
        return levels
    }

    private fun uploadRawTexture(image: SafeImage, texture: Int, label: String) {
        val plane = image.planes.firstOrNull() ?: throw IllegalArgumentException("$label has no RAW plane")
        require(plane.rowStride >= width * 2) {
            "$label RAW row stride ${plane.rowStride} is smaller than width $width"
        }
        require(plane.rowStride % 2 == 0) {
            "$label RAW row stride must be 16-bit aligned: ${plane.rowStride}"
        }
        val uploadBuffer = plane.buffer.duplicate().order(ByteOrder.nativeOrder()).apply { position(0) }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, plane.rowStride / 2)
        GLES30.glTexSubImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            GLES30.GL_RED_INTEGER,
            GLES30.GL_UNSIGNED_SHORT,
            uploadBuffer,
        )
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        checkGlError("uploadRawTexture $label")
    }

    private fun validExposureProduct(exposureProduct: Double): Double {
        return exposureProduct.takeIf { it.isFinite() && it > 0.0 } ?: 1.0
    }

    private fun hdrExposureScale(referenceExposureProduct: Double, frameExposureProduct: Double): Float {
        return (referenceExposureProduct / frameExposureProduct)
            .toFloat()
            .coerceIn(0.0001f, 64.0f)
    }

    private fun hdrBaselineExposureEv(shortExposureProduct: Double, referenceExposureProduct: Double): Float {
        if (!shortExposureProduct.isFinite() || shortExposureProduct <= 0.0 ||
            !referenceExposureProduct.isFinite() || referenceExposureProduct <= 0.0
        ) {
            return 0f
        }
        return if (shortExposureProduct < referenceExposureProduct) {
            (ln(referenceExposureProduct / shortExposureProduct) / ln(2.0)).toFloat()
        } else {
            0f
        }.coerceIn(0f, 8f)
    }

    private fun computeHdrProfileGainTableMap(baselineExposureEv: Float): com.hinnka.mycamera.raw.DngProfileGainTableMap? {
        if (!baselineExposureEv.isFinite() || baselineExposureEv <= 0.05f || pgtmStatsProgram == 0) {
            return null
        }
        if (profileToneMapMode != RawProfileToneMapMode.GooglePixel &&
            profileToneMapMode != RawProfileToneMapMode.PhotonPgtm
        ) {
            return null
        }
        val grid = DngHdrProfileGainTableGenerator.gridSizeFor(width, height)
        val pgtmGridWidth = grid.getOrElse(0) { 0 }
        val pgtmGridHeight = grid.getOrElse(1) { 0 }
        if (pgtmGridWidth <= 0 || pgtmGridHeight <= 0) return null
        if (!pgtmStatsBounds.isFullImage()) {
            RawStackRuntimeDebug.d(TAG) {
                "GPU HDR PGTM stats bounds: source=${width}x$height bounds=$pgtmStatsBounds"
            }
        }

        val cellCount = pgtmGridWidth * pgtmGridHeight
        val floatCount = cellCount * DngHdrProfileGainTableGenerator.CELL_STATS_FLOAT_STRIDE
        val byteCount = floatCount * 4
        val buffers = IntArray(1)
        GLES31.glGenBuffers(1, buffers, 0)
        val statsBufferId = buffers[0]
        if (statsBufferId == 0) return null
        return try {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, statsBufferId)
            GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, byteCount, null, GLES31.GL_DYNAMIC_READ)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, PGTM_STATS_BUFFER_BINDING, statsBufferId)
            GLES31.glUseProgram(pgtmStatsProgram)
            bindTexture(pgtmStatsProgram, "uPackedRaw", 0, outputTexture)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(pgtmStatsProgram, "uImageSize"), width, height)
            GLES31.glUniform4i(
                GLES31.glGetUniformLocation(pgtmStatsProgram, "uStatsBounds"),
                pgtmStatsBounds.left,
                pgtmStatsBounds.top,
                pgtmStatsBounds.right,
                pgtmStatsBounds.bottom,
            )
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(pgtmStatsProgram, "uGridSize"),
                pgtmGridWidth,
                pgtmGridHeight,
            )
            GLES31.glUniform1i(GLES31.glGetUniformLocation(pgtmStatsProgram, "uCfaPattern"), cfaPattern)
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(pgtmStatsProgram, "uBaselineExposureGain"),
                2.0f.pow(baselineExposureEv.coerceIn(0f, 8f))
            )
            GLES31.glUniformMatrix3fv(
                GLES31.glGetUniformLocation(pgtmStatsProgram, "uColorCorrectionMatrix"),
                1,
                false,
                transposeMatrix3x3(pgtmColorCorrectionMatrix),
                0
            )
            GLES31.glDispatchCompute(pgtmGridWidth, pgtmGridHeight, 1)
            GLES31.glMemoryBarrier(
                GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT
            )
            checkGlError("computeHdrProfileGainTableMap dispatch")

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, statsBufferId)
            val mapped = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                0,
                byteCount,
                GLES31.GL_MAP_READ_BIT
            ) ?: throw IllegalStateException("PGTM stats buffer map failed")
            val stats = try {
                FloatArray(floatCount).also { values ->
                    val byteBuffer = mapped as? ByteBuffer
                        ?: throw IllegalStateException("PGTM stats buffer is not ByteBuffer")
                    byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer().get(values)
                }
            } finally {
                GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
            }
            val diagnosticBand = DngPgtmDiagnostic.activeBandForSource("$TAG GPU stacker")
            when (profileToneMapMode) {
                RawProfileToneMapMode.PhotonPgtm -> DngPhotonProfileGainTableGenerator.forCellStats(
                    width = width,
                    height = height,
                    baselineExposureEv = baselineExposureEv,
                    packedCellStats = stats,
                    diagnosticBand = diagnosticBand
                )

                else -> DngHdrProfileGainTableGenerator.forCellStats(
                    width = width,
                    height = height,
                    baselineExposureEv = baselineExposureEv,
                    packedCellStats = stats,
                    diagnosticBand = diagnosticBand
                )
            }
        } catch (e: Exception) {
            PLog.w(TAG, "Failed to compute GPU HDR PGTM stats", e)
            null
        } finally {
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, PGTM_STATS_BUFFER_BINDING, 0)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            GLES31.glDeleteBuffers(1, intArrayOf(statsBufferId), 0)
        }
    }

    private fun sanitizePgtmStatsBounds(bounds: Rect?): Rect {
        val imageBounds = Rect(0, 0, width.coerceAtLeast(1), height.coerceAtLeast(1))
        if (bounds == null || bounds.isEmpty) return imageBounds
        return Rect(bounds).takeIf {
            it.intersect(imageBounds) && it.width() >= 2 && it.height() >= 2
        } ?: imageBounds
    }

    private fun Rect.isFullImage(): Boolean {
        return left == 0 && top == 0 && right == width && bottom == height
    }

    private fun buildProxy(
        rawTexture: Int,
        proxyTexture: Int,
        label: String,
        exposureScale: Float = 1.0f,
    ) {
        GLES31.glUseProgram(proxyProgram)
        bindTexture(proxyProgram, "uRaw", 0, rawTexture)
        bindImage(1, proxyTexture, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
        setCommonUniforms(proxyProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(proxyProgram, "uProxySize"), planeWidth, planeHeight)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(proxyProgram, "uExposureScale"), exposureScale)
        GLES31.glDispatchCompute(groupCount(planeWidth), groupCount(planeHeight), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("buildProxy $label")
    }

    private fun buildPyramid(levels: List<TextureLevel>) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, renderFbo)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glUseProgram(downsampleProgram)
        for (index in 1 until levels.size) {
            val input = levels[index - 1]
            val output = levels[index]
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER,
                GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D,
                output.texture,
                0,
            )
            GLES30.glDrawBuffers(1, intArrayOf(GLES30.GL_COLOR_ATTACHMENT0), 0)
            checkFramebuffer("buildPyramid level $index")
            GLES30.glViewport(0, 0, output.width, output.height)
            bindTexture(downsampleProgram, "uInput", 0, input.texture)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(downsampleProgram, "uInputSize"), input.width, input.height)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
            GLES31.glMemoryBarrier(GLES31.GL_FRAMEBUFFER_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
            checkGlError("buildPyramid level $index")
        }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun computeStructureTensor() {
        GLES31.glUseProgram(structureProgram)
        bindTexture(structureProgram, "uProxy", 0, refProxy)
        bindImage(1, kernelTexture, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
        setCommonUniforms(structureProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(structureProgram, "uProxySize"), planeWidth, planeHeight)
        setPrefilterStructureUniforms(structureProgram)
        GLES31.glDispatchCompute(groupCount(planeWidth), groupCount(planeHeight), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("computeStructureTensor")
    }

    private fun alignCurrentToReference(reference: List<TextureLevel>, current: List<TextureLevel>) {
        val levelIndex = alignLevel.coerceAtMost(reference.lastIndex).coerceAtMost(current.lastIndex)
        val ref = reference[levelIndex]
        val cur = current[levelIndex]
        bindFramebufferOutput(flowTexture, "alignCurrentToReference")
        GLES30.glViewport(0, 0, gridWidth, gridHeight)
        GLES30.glUseProgram(alignProgram)
        bindTexture(alignProgram, "uReference", 0, ref.texture)
        bindTexture(alignProgram, "uCurrent", 1, cur.texture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(alignProgram, "uLevelSize"), ref.width, ref.height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(alignProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uTileSize"), flowGridSpacing)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uAlignWindowSize"), hwmfPrefilter.alignWindowSize)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uLevelScale"), 1 shl levelIndex)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uSearchRadius"), hwmfPrefilter.alignSearchRadiusLevel)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uSampleStep"), hwmfPrefilter.alignSampleStep)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(alignProgram, "uCoveragePenalty"), hwmfPrefilter.alignCoveragePenalty)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(alignProgram, "uShiftPenalty"), hwmfPrefilter.alignShiftPenalty)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("alignCurrentToReference")
    }

    private fun alignHdrShortToReference(reference: List<TextureLevel>, shortFrame: List<TextureLevel>) {
        val levelIndex = alignLevel.coerceAtMost(reference.lastIndex).coerceAtMost(shortFrame.lastIndex)
        val ref = reference[levelIndex]
        val cur = shortFrame[levelIndex]
        val minLevelSide = minOf(ref.width, ref.height)
        val searchRadius = minOf(hwmfHdr.shortGlobalSearchRadiusLevel, max(1, minLevelSide / 6))
        val sampleBorder = minOf(hwmfHdr.shortGlobalSampleBorder, max(1, minLevelSide / 5))
        val scoreSide = searchRadius * 2 + 1
        val scoreCount = scoreSide * scoreSide
        val floatCount = scoreCount * HDR_SHORT_ALIGNMENT_SCORE_STRIDE
        val byteCount = floatCount * 4
        val buffers = IntArray(1)
        GLES31.glGenBuffers(1, buffers, 0)
        val scoreBufferId = buffers[0]
        if (scoreBufferId == 0) {
            writeHdrShortAlignmentTexture(0f, 0f)
            return
        }
        try {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, scoreBufferId)
            GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, byteCount, null, GLES31.GL_DYNAMIC_READ)
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                HDR_SHORT_ALIGNMENT_SCORE_BUFFER_BINDING,
                scoreBufferId,
            )
            GLES31.glUseProgram(hdrShortGlobalAlignProgram)
            bindTexture(hdrShortGlobalAlignProgram, "uReference", 0, ref.texture)
            bindTexture(hdrShortGlobalAlignProgram, "uCurrent", 1, cur.texture)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(hdrShortGlobalAlignProgram, "uLevelSize"), ref.width, ref.height)
            GLES31.glUniform1i(GLES31.glGetUniformLocation(hdrShortGlobalAlignProgram, "uLevelScale"), 1 shl levelIndex)
            GLES31.glUniform1i(GLES31.glGetUniformLocation(hdrShortGlobalAlignProgram, "uSearchRadius"), searchRadius)
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(hdrShortGlobalAlignProgram, "uSampleStep"),
                hwmfHdr.shortGlobalSampleStep,
            )
            GLES31.glUniform1i(GLES31.glGetUniformLocation(hdrShortGlobalAlignProgram, "uSampleBorder"), sampleBorder)
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(hdrShortGlobalAlignProgram, "uCoveragePenalty"),
                hwmfHdr.shortGlobalCoveragePenalty,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(hdrShortGlobalAlignProgram, "uShiftPenalty"),
                hwmfHdr.shortGlobalShiftPenalty,
            )
            GLES31.glDispatchCompute(scoreSide, scoreSide, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT)
            checkGlError("alignHdrShortToReference dispatch")

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, scoreBufferId)
            val mapped = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                0,
                byteCount,
                GLES31.GL_MAP_READ_BIT
            ) ?: throw IllegalStateException("HDR short alignment score buffer map failed")
            val scores = try {
                FloatArray(floatCount).also { values ->
                    val byteBuffer = mapped as? ByteBuffer
                        ?: throw IllegalStateException("HDR short alignment score buffer is not ByteBuffer")
                    byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer().get(values)
                }
            } finally {
                GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
            }
            var bestScore = Float.POSITIVE_INFINITY
            var bestDx = 0f
            var bestDy = 0f
            for (index in 0 until scoreCount) {
                val offset = index * HDR_SHORT_ALIGNMENT_SCORE_STRIDE
                val score = scores[offset + 2]
                if (score.isFinite() && score < bestScore) {
                    bestScore = score
                    bestDx = scores[offset]
                    bestDy = scores[offset + 1]
                }
            }
            writeHdrShortAlignmentTexture(bestDx, bestDy)
        } catch (e: Exception) {
            PLog.w(TAG, "Failed to compute HDR short global alignment", e)
            writeHdrShortAlignmentTexture(0f, 0f)
        } finally {
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, HDR_SHORT_ALIGNMENT_SCORE_BUFFER_BINDING, 0)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            GLES31.glDeleteBuffers(1, intArrayOf(scoreBufferId), 0)
        }
    }

    private fun writeHdrShortAlignmentTexture(dx: Float, dy: Float) {
        val buffer = ByteBuffer.allocateDirect(4 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(dx)
                put(dy)
                put(0f)
                put(0f)
                position(0)
            }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, hdrShortAlignmentTexture)
        GLES30.glTexSubImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            0,
            0,
            1,
            1,
            GLES30.GL_RGBA,
            GLES30.GL_FLOAT,
            buffer
        )
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        checkGlError("writeHdrShortAlignmentTexture")
    }

    private fun refineFlow() {
        repeat(lkRefinePasses) { pass ->
            val input = if (pass % 2 == 0) flowTexture else flowScratchTexture
            val output = if (pass % 2 == 0) flowScratchTexture else flowTexture
            GLES31.glUseProgram(lkRefineProgram)
            bindTexture(lkRefineProgram, "uReference", 0, refProxy)
            bindTexture(lkRefineProgram, "uCurrent", 1, curProxy)
            bindTexture(lkRefineProgram, "uInputFlow", 2, input)
            bindImage(3, output, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(lkRefineProgram, "uPlaneSize"), planeWidth, planeHeight)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(lkRefineProgram, "uGridSize"), gridWidth, gridHeight)
            GLES31.glUniform1i(GLES31.glGetUniformLocation(lkRefineProgram, "uTileSize"), flowGridSpacing)
            GLES31.glDispatchCompute(groupCount(gridWidth), groupCount(gridHeight), 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
            checkGlError("refineFlow pass $pass")
        }
        if (lkRefinePasses % 2 != 0) {
            copyFlow(flowScratchTexture, flowTexture, "refineFlow copy")
        }
    }

    private fun smoothFlow() {
        repeat(flowSmoothPasses) { pass ->
            val input = if (pass % 2 == 0) flowTexture else flowScratchTexture
            val output = if (pass % 2 == 0) flowScratchTexture else flowTexture
            bindFramebufferOutput(output, "smoothFlow pass $pass")
            GLES30.glViewport(0, 0, gridWidth, gridHeight)
            GLES30.glUseProgram(smoothFlowProgram)
            bindTexture(smoothFlowProgram, "uInputFlow", 0, input)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(smoothFlowProgram, "uGridSize"), gridWidth, gridHeight)
            GLES31.glUniform1f(GLES31.glGetUniformLocation(smoothFlowProgram, "uOutlierThreshold"), hwmfBlend.flowOutlierThresholdPx)
            GLES31.glUniform1f(GLES31.glGetUniformLocation(smoothFlowProgram, "uOutlierWeight"), hwmfBlend.flowOutlierWeight)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
            finishFramebufferPass("smoothFlow pass $pass")
        }
    }

    private fun copyFlow(input: Int, output: Int, label: String) {
        bindFramebufferOutput(output, label)
        GLES30.glViewport(0, 0, gridWidth, gridHeight)
        GLES30.glUseProgram(smoothFlowProgram)
        bindTexture(smoothFlowProgram, "uInputFlow", 0, input)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(smoothFlowProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(smoothFlowProgram, "uOutlierThreshold"), 100000.0f)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(smoothFlowProgram, "uOutlierWeight"), hwmfBlend.flowOutlierWeight)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass(label)
    }

    private fun computeRobustness() {
        GLES31.glUseProgram(robustnessProgram)
        bindTexture(robustnessProgram, "uReference", 0, refProxy)
        bindTexture(robustnessProgram, "uCurrent", 1, curProxy)
        bindTexture(robustnessProgram, "uFlowGrid", 2, flowTexture)
        bindImage(3, robustnessTexture, GLES31.GL_WRITE_ONLY, GLES30.GL_R32F)
        setCommonUniforms(robustnessProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(robustnessProgram, "uPlaneSize"), planeWidth, planeHeight)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(robustnessProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(robustnessProgram, "uTileSize"), flowGridSpacing)
        setBlendRobustnessUniforms(robustnessProgram)
        GLES31.glDispatchCompute(groupCount(planeWidth), groupCount(planeHeight), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("computeRobustness")
    }

    private fun computeTileMask() {
        bindFramebufferOutput(tileMaskTexture, "computeTileMask")
        GLES30.glViewport(0, 0, gridWidth, gridHeight)
        GLES30.glUseProgram(tileMaskProgram)
        bindTexture(tileMaskProgram, "uReference", 0, refProxy)
        bindTexture(tileMaskProgram, "uRobustness", 1, robustnessTexture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(tileMaskProgram, "uPlaneSize"), planeWidth, planeHeight)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(tileMaskProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(tileMaskProgram, "uTileSize"), flowGridSpacing)
        setTileMaskUniforms(tileMaskProgram)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("computeTileMask")
    }

    private fun estimateCurrentRegistration(
        referencePyramid: List<TextureLevel>,
        currentPyramid: List<TextureLevel>,
        frameIndex: Int,
    ): Boolean {
        val samples = readRegistrationSamples()
        val flowEstimate = if (samples != null) {
            RawStackFrameRegistrationEstimator.estimate(
                setup = registrationSetup,
                samples = samples,
                stage = RawStackRegistrationStage.BLEND,
            )
        } else {
            RawStackFrameRegistrationEstimator.estimate(
                setup = registrationSetup,
                samples = emptyList(),
                stage = RawStackRegistrationStage.BLEND,
            )
        }
        val seedShiftPlane = registrationSeedShiftPlane(samples)
        val globalEstimate = readGlobalRegistrationCandidates(
            referencePyramid = referencePyramid,
            currentPyramid = currentPyramid,
            seedShiftPlane = seedShiftPlane,
        )?.let { candidates ->
            RawStackFrameRegistrationEstimator.estimateGlobalTranslation(
                setup = registrationSetup,
                candidates = candidates,
                stage = RawStackRegistrationStage.BLEND,
            )
        }
        val estimate = selectRegistrationEstimate(globalEstimate, flowEstimate)
        val seedTranslation = registrationSeedTranslation(seedShiftPlane)
        val acceptance = resolveRegistrationAcceptance(estimate, seedTranslation)
        currentRegistrationTransform = acceptance.transform
        currentRegistrationSrTransform = acceptance.srTransform
        currentRegistrationSrWeight = acceptance.srWeight
        if (hwmfDebug.collectMetrics) {
            recordRegistrationGlobalEstimate(globalEstimate)
            recordRegistrationEstimate(estimate, currentRegistrationTransform)
        }
        val accepted = acceptance.accepted
        if (hwmfDebug.logCompactSummary) {
            val candidateTranslation = registrationTranslation(estimate.candidateTransform)
            PLog.d(
                TAG,
                "HWMF registration frame=$frameIndex source=${estimate.source} conf=${estimate.confidence} " +
                    "forceIdentity=${estimate.forceIdentity} accepted=$accepted accept=${acceptance.reason} " +
                    "global=${globalEstimate.globalSummaryOrEmpty()} " +
                    "seed=${seedShiftPlane.vectorSummary()} " +
                    "candidate=${candidateTranslation.translationSummary()} " +
                    "seedDist=${candidateTranslation.seedDistance(seedTranslation).formatPx()} " +
                    "consDist=${registrationConsistencyDistance(candidateTranslation).formatPx()} " +
                    "srWeight=${currentRegistrationSrWeight.formatWeight()} " +
                    "srSource=${acceptance.srTransformSource} " +
                    "srSeed=${acceptance.srSeedWeight.formatWeight()} " +
                    "srCons=${acceptance.srConsistencyWeight.formatWeight()} " +
                    "flowSamples=${flowEstimate.usedSampleCount}/${flowEstimate.sampleCount} " +
                    "inlier=${estimate.inlierRatio.percentString()} resP90=${estimate.residualP90Px.formatPx()} " +
                    "matrix=${currentRegistrationTransform.matrixAt(0).matrixSummary()}"
            )
        }
        if (accepted) {
            recordAcceptedRegistrationTranslation(currentRegistrationTransform)
            if (currentRegistrationSrWeight > 0.0f) {
                recordAcceptedSrRegistrationTranslation(currentRegistrationSrTransform)
            }
        }
        return accepted
    }

    private fun resolveRegistrationAcceptance(
        estimate: RawStackRegistrationEstimate,
        seedTranslation: RegistrationTranslation?,
    ): RegistrationAcceptance {
        if (!estimate.forceIdentity) {
            val candidateSrGate = registrationSrGate(estimate, seedTranslation)
            val seedSrGate = registrationSeedSrGate(estimate, seedTranslation)
            val seedSrTransform = seedTranslation?.let { registrationTransformFromTranslation(it, estimate.transform) }
            val useSeedForSr = seedSrGate != null &&
                seedSrTransform != null &&
                seedSrGate.weight > candidateSrGate.weight + 0.05f
            val srGate = if (useSeedForSr) seedSrGate else candidateSrGate
            return RegistrationAcceptance(
                accepted = true,
                transform = estimate.transform,
                reason = "strict",
                srTransform = if (useSeedForSr) seedSrTransform else estimate.transform,
                srTransformSource = if (useSeedForSr) "SEED_TRANSLATION" else "CANDIDATE",
                srWeight = srGate.weight,
                srSeedWeight = srGate.seedWeight,
                srConsistencyWeight = srGate.consistencyWeight,
            )
        }
        val seedFallbackReason = seedFallbackRegistrationAcceptanceReason(estimate, seedTranslation)
        if (seedFallbackReason != null && seedTranslation != null) {
            val seedTransform = registrationTransformFromTranslation(
                translation = seedTranslation,
                template = estimate.transform,
                rawConfidence = registrationSetup.confidenceConfig.forceIdentityThreshold,
            )
            val srGate = registrationSeedFallbackSrGate(seedTranslation)
            return RegistrationAcceptance(
                accepted = true,
                transform = seedTransform,
                reason = seedFallbackReason,
                srTransform = seedTransform,
                srTransformSource = "SEED_FALLBACK",
                srWeight = srGate.weight,
                srSeedWeight = srGate.seedWeight,
                srConsistencyWeight = srGate.consistencyWeight,
            )
        }
        val nrBaseReason = nrBaseRegistrationAcceptanceReason(estimate, seedTranslation)
        if (nrBaseReason != null) {
            return RegistrationAcceptance(
                accepted = true,
                transform = estimate.candidateTransform.copy(forceIdentity = false),
                reason = nrBaseReason,
                srTransform = estimate.candidateTransform.copy(forceIdentity = false),
                srTransformSource = "NR_BASE",
                srWeight = 0.0f,
                srSeedWeight = 0.0f,
                srConsistencyWeight = 0.0f,
            )
        }
        return RegistrationAcceptance(
            accepted = false,
            transform = estimate.transform,
            reason = "reject",
            srTransform = estimate.transform,
            srTransformSource = "REJECT",
            srWeight = 0.0f,
            srSeedWeight = 0.0f,
            srConsistencyWeight = 0.0f,
        )
    }

    private fun nrBaseRegistrationAcceptanceReason(
        estimate: RawStackRegistrationEstimate,
        seedTranslation: RegistrationTranslation?,
    ): String? {
        if (tuning.mode != RawStackMode.MFNR && tuning.mode != RawStackMode.MFSR) return null
        if (estimate.source != RawStackRegistrationSource.IMAGE_TRANSLATION) return null
        val hasRelaxedConfidence = estimate.confidence >= hwmfBlend.denoiseRelaxedRegistrationConfidenceMin
        val hasSeedConfidence = estimate.confidence >= hwmfBlend.denoiseSeedRegistrationConfidenceMin
        val hasConsistentConfidence = estimate.confidence >= hwmfBlend.denoiseConsistentRegistrationConfidenceMin
        if (!hasRelaxedConfidence && !hasSeedConfidence && !hasConsistentConfidence) return null
        if (!estimate.globalCoverage.isFinite() ||
            estimate.globalCoverage < hwmfBlend.denoiseRelaxedRegistrationCoverageMin
        ) {
            return null
        }
        if (!estimate.globalBestScore.isFinite() ||
            estimate.globalBestScore > hwmfBlend.denoiseRelaxedRegistrationScoreMax
        ) {
            return null
        }
        val translation = registrationTranslation(estimate.candidateTransform) ?: return null
        if (translation.magnitude > hwmfBlend.denoiseRelaxedRegistrationTranslationMaxPx) return null
        if (hasSeedConfidence && isSeedConsistentNrBaseRegistration(estimate, translation, seedTranslation)) {
            return "nr-base-denoise-seed"
        }
        if (hasSeedConfidence && isTightSeedNrBaseRegistration(estimate, translation, seedTranslation)) {
            return "nr-base-denoise-seed-tight"
        }
        if (hasRelaxedConfidence && isRelaxedNrBaseRegistration(estimate, translation, seedTranslation)) {
            return "nr-base-denoise"
        }
        if (hasRelaxedConfidence && isStaticNrBaseRegistration(estimate, translation, seedTranslation)) {
            return "nr-base-denoise-static"
        }
        if (hasConsistentConfidence && isBurstConsistentNrBaseRegistration(estimate, translation)) {
            return "nr-base-denoise-consistent"
        }
        return null
    }

    private fun seedFallbackRegistrationAcceptanceReason(
        estimate: RawStackRegistrationEstimate,
        seedTranslation: RegistrationTranslation?,
    ): String? {
        if (tuning.mode != RawStackMode.MFNR && tuning.mode != RawStackMode.MFSR) return null
        val seed = seedTranslation ?: return null
        if (!estimate.globalBestScore.isFinite() ||
            estimate.globalBestScore > hwmfBlend.denoiseSeedFallbackRegistrationScoreMax
        ) {
            return null
        }
        if (!estimate.globalCoverage.isFinite() ||
            estimate.globalCoverage < hwmfBlend.denoiseSeedFallbackRegistrationCoverageMin
        ) {
            return null
        }
        if (seed.magnitude > hwmfBlend.denoiseSeedFallbackRegistrationTranslationMaxPx) return null
        val median = acceptedRegistrationMedianTranslation()
        if (median != null &&
            seed.distanceTo(median) > hwmfBlend.denoiseSeedFallbackRegistrationDeltaMaxPx
        ) {
            return null
        }
        return "nr-base-denoise-seed-flow"
    }

    private fun registrationSrGate(
        estimate: RawStackRegistrationEstimate,
        seedTranslation: RegistrationTranslation?,
    ): RegistrationSrGate {
        if (!superResolutionEnabled || estimate.forceIdentity) return RegistrationSrGate.Zero
        val translation = registrationTranslation(estimate.candidateTransform) ?: return RegistrationSrGate.Zero
        val confidenceWeight = smoothStep(
            hwmfSr.registrationConfidenceStart.toFloat(),
            hwmfSr.registrationConfidenceFull.toFloat(),
            estimate.confidence.toFloat(),
        )
        val sourceWeight = when (estimate.source) {
            RawStackRegistrationSource.IMAGE_TRANSLATION -> imageTranslationSrWeight(estimate)
            RawStackRegistrationSource.FLOW_AFFINE -> flowAffineSrWeight(estimate)
        }
        val seedWeight = registrationSrSeedWeight(translation, seedTranslation)
        val consistencyWeight = registrationSrConsistencyWeight(translation)
        return RegistrationSrGate(
            weight = (
                confidenceWeight *
                    sourceWeight *
                    seedWeight *
                    consistencyWeight
                ).coerceIn(0.0f, 1.0f),
            seedWeight = seedWeight,
            consistencyWeight = consistencyWeight,
        )
    }

    private fun registrationSeedSrGate(
        estimate: RawStackRegistrationEstimate,
        seedTranslation: RegistrationTranslation?,
    ): RegistrationSrGate? {
        if (!superResolutionEnabled || estimate.forceIdentity) return null
        if (estimate.source != RawStackRegistrationSource.IMAGE_TRANSLATION) return null
        val seed = seedTranslation ?: return null
        val confidenceWeight = smoothStep(
            hwmfSr.registrationConfidenceStart.toFloat(),
            hwmfSr.registrationConfidenceFull.toFloat(),
            estimate.confidence.toFloat(),
        )
        val marginWeight = smoothStep(
            hwmfSr.registrationMarginStart,
            hwmfSr.registrationMarginFull,
            estimate.globalScoreMargin,
        )
        val coverageWeight = smoothStep(
            hwmfSr.registrationCoverageStart,
            hwmfSr.registrationCoverageFull,
            estimate.globalCoverage,
        )
        val consistencyWeight = registrationSrConsistencyWeight(seed)
        return RegistrationSrGate(
            weight = (
                confidenceWeight *
                    marginWeight *
                    coverageWeight *
                    consistencyWeight
                ).coerceIn(0.0f, 1.0f),
            seedWeight = 1.0f,
            consistencyWeight = consistencyWeight,
        )
    }

    private fun imageTranslationSrWeight(estimate: RawStackRegistrationEstimate): Float {
        val marginWeight = smoothStep(
            hwmfSr.registrationMarginStart,
            hwmfSr.registrationMarginFull,
            estimate.globalScoreMargin,
        )
        val coverageWeight = smoothStep(
            hwmfSr.registrationCoverageStart,
            hwmfSr.registrationCoverageFull,
            estimate.globalCoverage,
        )
        return (marginWeight * coverageWeight).coerceIn(0.0f, 1.0f)
    }

    private fun flowAffineSrWeight(estimate: RawStackRegistrationEstimate): Float {
        val inlierWeight = smoothStep(
            hwmfSr.registrationAffineInlierStart,
            hwmfSr.registrationAffineInlierFull,
            estimate.inlierRatio,
        )
        val residualWeight = 1.0f - smoothStep(
            hwmfSr.registrationAffineResidualStartPx,
            hwmfSr.registrationAffineResidualEndPx,
            estimate.residualP90Px,
        )
        return (inlierWeight * residualWeight).coerceIn(0.0f, 1.0f)
    }

    private fun registrationSeedFallbackSrGate(seed: RegistrationTranslation): RegistrationSrGate {
        val consistencyWeight = registrationSrConsistencyWeight(seed)
        val motionWeight = 1.0f - smoothStep(
            hwmfBlend.denoiseSeedFallbackRegistrationTranslationMaxPx * 0.75f,
            hwmfBlend.denoiseSeedFallbackRegistrationTranslationMaxPx,
            seed.magnitude,
        )
        val weight = (
            hwmfSr.registrationSeedFallbackWeight *
                consistencyWeight *
                motionWeight
            ).coerceIn(0.0f, 1.0f)
        return RegistrationSrGate(
            weight = weight,
            seedWeight = 1.0f,
            consistencyWeight = consistencyWeight,
        )
    }

    private fun registrationSrSeedWeight(
        translation: RegistrationTranslation,
        seedTranslation: RegistrationTranslation?,
    ): Float {
        val seed = seedTranslation ?: return 1.0f
        val distance = translation.distanceTo(seed)
        return 1.0f - smoothStep(
            hwmfSr.registrationSeedDistanceStartPx,
            hwmfSr.registrationSeedDistanceEndPx,
            distance,
        )
    }

    private fun registrationSrConsistencyWeight(translation: RegistrationTranslation): Float {
        val distance = registrationSrConsistencyDistance(translation)
        if (!distance.isFinite()) return 1.0f
        return 1.0f - smoothStep(
            hwmfSr.registrationConsistencyDistanceStartPx,
            hwmfSr.registrationConsistencyDistanceEndPx,
            distance,
        )
    }

    private fun smoothStep(edge0: Float, edge1: Float, value: Float): Float {
        if (!edge0.isFinite() || !edge1.isFinite() || !value.isFinite()) return 0.0f
        val lo = minOf(edge0, edge1)
        val hi = max(edge0, edge1)
        if (hi <= lo) return if (value >= hi) 1.0f else 0.0f
        val t = ((value - lo) / (hi - lo)).coerceIn(0.0f, 1.0f)
        return t * t * (3.0f - 2.0f * t)
    }

    private fun isSeedConsistentNrBaseRegistration(
        estimate: RawStackRegistrationEstimate,
        translation: RegistrationTranslation,
        seedTranslation: RegistrationTranslation?,
    ): Boolean {
        if (!estimate.globalScoreMargin.isFinite() ||
            estimate.globalScoreMargin < hwmfBlend.denoiseSeedRegistrationMarginMin
        ) {
            return false
        }
        val seed = seedTranslation ?: return false
        return translation.distanceTo(seed) <= hwmfBlend.denoiseSeedRegistrationDeltaMaxPx
    }

    private fun isTightSeedNrBaseRegistration(
        estimate: RawStackRegistrationEstimate,
        translation: RegistrationTranslation,
        seedTranslation: RegistrationTranslation?,
    ): Boolean {
        if (!estimate.globalScoreMargin.isFinite() ||
            estimate.globalScoreMargin < hwmfBlend.denoiseSeedTightRegistrationMarginMin
        ) {
            return false
        }
        val seed = seedTranslation ?: return false
        return translation.distanceTo(seed) <= hwmfBlend.denoiseSeedTightRegistrationDeltaMaxPx &&
            translation.magnitude <= hwmfBlend.denoiseSeedTightRegistrationTranslationMaxPx
    }

    private fun isRelaxedNrBaseRegistration(
        estimate: RawStackRegistrationEstimate,
        translation: RegistrationTranslation,
        seedTranslation: RegistrationTranslation?,
    ): Boolean {
        if (!estimate.globalScoreMargin.isFinite() ||
            estimate.globalScoreMargin < hwmfBlend.denoiseRelaxedRegistrationMarginMin
        ) {
            return false
        }
        val seed = seedTranslation ?: return true
        return translation.distanceTo(seed) <= hwmfBlend.denoiseSeedRegistrationDeltaMaxPx
    }

    private fun isStaticNrBaseRegistration(
        estimate: RawStackRegistrationEstimate,
        translation: RegistrationTranslation,
        seedTranslation: RegistrationTranslation?,
    ): Boolean {
        if (!estimate.globalScoreMargin.isFinite() ||
            estimate.globalScoreMargin < hwmfBlend.denoiseStaticRegistrationMarginMin
        ) {
            return false
        }
        val seed = seedTranslation
        if (seed != null && translation.distanceTo(seed) > hwmfBlend.denoiseSeedRegistrationDeltaMaxPx) {
            return false
        }
        return translation.magnitude <= hwmfBlend.denoiseStaticRegistrationTranslationMaxPx
    }

    private fun isBurstConsistentNrBaseRegistration(
        estimate: RawStackRegistrationEstimate,
        translation: RegistrationTranslation,
    ): Boolean {
        if (!estimate.globalScoreMargin.isFinite() ||
            estimate.globalScoreMargin < hwmfBlend.denoiseConsistentRegistrationMarginMin
        ) {
            return false
        }
        val median = acceptedRegistrationMedianTranslation() ?: return false
        return translation.distanceTo(median) <= hwmfBlend.denoiseConsistentRegistrationDeltaMaxPx
    }

    private fun recordAcceptedRegistrationTranslation(transform: RawStackPerspectiveTransform) {
        val translation = registrationTranslation(transform) ?: return
        acceptedRegistrationTranslations += translation
    }

    private fun recordAcceptedSrRegistrationTranslation(transform: RawStackPerspectiveTransform) {
        val translation = registrationTranslation(transform) ?: return
        acceptedSrRegistrationTranslations += translation
    }

    private fun acceptedRegistrationMedianTranslation(): RegistrationTranslation? {
        val minAccepted = hwmfBlend.denoiseConsistentRegistrationMinAccepted.coerceAtLeast(1)
        if (acceptedRegistrationTranslations.size < minAccepted) return null
        val xs = acceptedRegistrationTranslations.map { it.dx }.sorted()
        val ys = acceptedRegistrationTranslations.map { it.dy }.sorted()
        return RegistrationTranslation(
            dx = medianOfSorted(xs),
            dy = medianOfSorted(ys),
        )
    }

    private fun acceptedSrRegistrationMedianTranslation(): RegistrationTranslation? {
        val minAccepted = hwmfBlend.denoiseConsistentRegistrationMinAccepted.coerceAtLeast(1)
        if (acceptedSrRegistrationTranslations.size < minAccepted) return null
        val xs = acceptedSrRegistrationTranslations.map { it.dx }.sorted()
        val ys = acceptedSrRegistrationTranslations.map { it.dy }.sorted()
        return RegistrationTranslation(
            dx = medianOfSorted(xs),
            dy = medianOfSorted(ys),
        )
    }

    private fun registrationConsistencyDistance(translation: RegistrationTranslation?): Float {
        if (translation == null) return Float.NaN
        val median = acceptedRegistrationMedianTranslation() ?: return Float.NaN
        return translation.distanceTo(median)
    }

    private fun registrationSrConsistencyDistance(translation: RegistrationTranslation?): Float {
        if (translation == null) return Float.NaN
        val median = acceptedSrRegistrationMedianTranslation() ?: return Float.NaN
        return translation.distanceTo(median)
    }

    private fun registrationSeedTranslation(seedShiftPlane: FloatArray): RegistrationTranslation? {
        if (seedShiftPlane.size < 2) return null
        val dx = seedShiftPlane[0] * 2.0f
        val dy = seedShiftPlane[1] * 2.0f
        if (!dx.isFinite() || !dy.isFinite()) return null
        return RegistrationTranslation(dx, dy)
    }

    private fun registrationTranslation(transform: RawStackPerspectiveTransform): RegistrationTranslation? {
        val matrix = transform.matrixAt(0)
        val tx = matrix.getOrElse(2) { Float.NaN }
        val ty = matrix.getOrElse(5) { Float.NaN }
        if (!tx.isFinite() || !ty.isFinite()) return null
        return RegistrationTranslation(tx, ty)
    }

    private fun registrationTransformFromTranslation(
        translation: RegistrationTranslation,
        template: RawStackPerspectiveTransform,
        rawConfidence: Int = template.rawConfidence,
    ): RawStackPerspectiveTransform {
        return RawStackPerspectiveTransform.fromSingleMatrix(
            stage = template.stage,
            transformDefinedOnWidth = template.transformDefinedOnWidth,
            transformDefinedOnHeight = template.transformDefinedOnHeight,
            geometryColumns = template.geometryColumns,
            geometryRows = template.geometryRows,
            rowMajorMatrix = floatArrayOf(
                1.0f, 0.0f, translation.dx,
                0.0f, 1.0f, translation.dy,
                0.0f, 0.0f, 1.0f,
            ),
            rawConfidence = rawConfidence,
            confidenceConfig = registrationSetup.confidenceConfig,
        )
    }

    private fun medianOfSorted(values: List<Float>): Float {
        if (values.isEmpty()) return Float.NaN
        val middle = values.size / 2
        return if ((values.size and 1) == 0) {
            (values[middle - 1] + values[middle]) * 0.5f
        } else {
            values[middle]
        }
    }

    private fun selectRegistrationEstimate(
        globalEstimate: RawStackRegistrationEstimate?,
        flowEstimate: RawStackRegistrationEstimate,
    ): RawStackRegistrationEstimate {
        if (globalEstimate == null) return flowEstimate
        return when {
            !globalEstimate.forceIdentity -> globalEstimate
            !flowEstimate.forceIdentity -> flowEstimate
            globalEstimate.confidence >= flowEstimate.confidence -> globalEstimate
            else -> flowEstimate
        }
    }

    private fun readGlobalRegistrationCandidates(
        referencePyramid: List<TextureLevel>,
        currentPyramid: List<TextureLevel>,
        seedShiftPlane: FloatArray,
    ): List<RawStackGlobalRegistrationCandidate>? {
        val levelIndex = 0.coerceAtMost(referencePyramid.lastIndex).coerceAtMost(currentPyramid.lastIndex)
        val reference = referencePyramid[levelIndex]
        val current = currentPyramid[levelIndex]
        val minLevelSide = minOf(reference.width, reference.height)
        val searchRadius = minOf(
            max(3, hwmfPrefilter.alignSearchRadiusLevel.coerceAtLeast(1)),
            max(1, minLevelSide / 6),
        )
        val scoreSide = searchRadius * 2 + 1
        val scoreCount = scoreSide * scoreSide
        if (scoreCount <= 0 || registrationGlobalAlignProgram == 0) return null

        val sampleStep = max(2, hwmfPrefilter.alignSampleStep * 4)
        val sampleBorder = minOf(
            max(8, hwmfPrefilter.alignWindowSize),
            max(1, minLevelSide / 5),
        )
        val floatCount = scoreCount * REGISTRATION_GLOBAL_SCORE_STRIDE
        val byteCount = floatCount * 4
        val ids = IntArray(1)
        GLES31.glGenBuffers(1, ids, 0)
        val scoreBuffer = ids[0]
        if (scoreBuffer == 0) {
            PLog.w(TAG, "RAW registration global score buffer allocation failed")
            return null
        }
        return try {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, scoreBuffer)
            GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, byteCount, null, GLES31.GL_DYNAMIC_READ)
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                REGISTRATION_GLOBAL_SCORE_BUFFER_BINDING,
                scoreBuffer,
            )
            GLES31.glUseProgram(registrationGlobalAlignProgram)
            bindTexture(registrationGlobalAlignProgram, "uReference", 0, reference.texture)
            bindTexture(registrationGlobalAlignProgram, "uCurrent", 1, current.texture)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uLevelSize"),
                reference.width,
                reference.height,
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uLevelScale"),
                1 shl levelIndex,
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uSearchRadius"),
                searchRadius,
            )
            GLES31.glUniform2f(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uCenterShift"),
                seedShiftPlane.getOrElse(0) { 0f },
                seedShiftPlane.getOrElse(1) { 0f },
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uSampleStep"),
                sampleStep,
            )
            GLES31.glUniform1i(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uSampleBorder"),
                sampleBorder,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uCoveragePenalty"),
                hwmfPrefilter.alignCoveragePenalty,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(registrationGlobalAlignProgram, "uShiftPenalty"),
                hwmfPrefilter.alignShiftPenalty,
            )
            GLES31.glDispatchCompute(scoreSide, scoreSide, 1)
            GLES31.glMemoryBarrier(
                GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT
            )
            checkGlError("readGlobalRegistrationCandidates dispatch")

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, scoreBuffer)
            val mapped = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                0,
                byteCount,
                GLES31.GL_MAP_READ_BIT,
            ) ?: throw IllegalStateException("registration global score buffer map failed")
            val values = try {
                FloatArray(floatCount).also { out ->
                    val byteBuffer = mapped as? ByteBuffer
                        ?: throw IllegalStateException("registration global score buffer is not ByteBuffer")
                    byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer().get(out)
                }
            } finally {
                GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
            }
            decodeGlobalRegistrationCandidates(values, scoreCount)
        } catch (e: Exception) {
            PLog.w(TAG, "Failed to read RAW global registration scores", e)
            null
        } finally {
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, REGISTRATION_GLOBAL_SCORE_BUFFER_BINDING, 0)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            GLES31.glDeleteBuffers(1, intArrayOf(scoreBuffer), 0)
        }
    }

    private fun registrationSeedShiftPlane(samples: List<RawStackRegistrationSample>?): FloatArray {
        if (samples.isNullOrEmpty()) return floatArrayOf(0f, 0f)
        var weightSum = 0f
        var dxSum = 0f
        var dySum = 0f
        for (sample in samples) {
            val weight = sample.weight
            if (weight <= 0f) continue
            val dx = (sample.targetX - sample.referenceX) * 0.5f
            val dy = (sample.targetY - sample.referenceY) * 0.5f
            if (!dx.isFinite() || !dy.isFinite()) continue
            dxSum += dx * weight
            dySum += dy * weight
            weightSum += weight
        }
        if (weightSum <= 0f) return floatArrayOf(0f, 0f)
        return floatArrayOf(dxSum / weightSum, dySum / weightSum)
    }

    private fun decodeGlobalRegistrationCandidates(
        values: FloatArray,
        scoreCount: Int,
    ): List<RawStackGlobalRegistrationCandidate> {
        val candidates = ArrayList<RawStackGlobalRegistrationCandidate>(scoreCount)
        for (index in 0 until scoreCount) {
            val offset = index * REGISTRATION_GLOBAL_SCORE_STRIDE
            if (offset + REGISTRATION_GLOBAL_SCORE_STRIDE > values.size) continue
            candidates += RawStackGlobalRegistrationCandidate(
                dxRaw = values[offset + 0] * 2.0f,
                dyRaw = values[offset + 1] * 2.0f,
                score = values[offset + 2],
                coverage = values[offset + 3],
            )
        }
        return candidates
    }

    private fun readRegistrationSamples(): List<RawStackRegistrationSample>? {
        val sampleCount = gridWidth * gridHeight
        if (sampleCount <= 0 || registrationSampleProgram == 0) return null
        val floatCount = sampleCount * REGISTRATION_SAMPLE_FLOAT_STRIDE
        val byteCount = floatCount * 4
        val ids = IntArray(1)
        GLES31.glGenBuffers(1, ids, 0)
        val sampleBuffer = ids[0]
        if (sampleBuffer == 0) {
            PLog.w(TAG, "RAW registration sample buffer allocation failed")
            return null
        }
        return try {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, sampleBuffer)
            GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, byteCount, null, GLES31.GL_DYNAMIC_READ)
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                REGISTRATION_SAMPLE_BUFFER_BINDING,
                sampleBuffer,
            )
            GLES31.glUseProgram(registrationSampleProgram)
            bindTexture(registrationSampleProgram, "uFlowGrid", 0, flowTexture)
            bindTexture(registrationSampleProgram, "uRobustness", 1, robustnessTexture)
            bindTexture(registrationSampleProgram, "uTileMask", 2, tileMaskTexture)
            bindTexture(registrationSampleProgram, "uReference", 3, refProxy)
            bindTexture(registrationSampleProgram, "uCurrent", 4, curProxy)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(registrationSampleProgram, "uImageSize"), width, height)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(registrationSampleProgram, "uPlaneSize"), planeWidth, planeHeight)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(registrationSampleProgram, "uGridSize"), gridWidth, gridHeight)
            GLES31.glUniform1i(GLES31.glGetUniformLocation(registrationSampleProgram, "uTileSize"), flowGridSpacing)
            GLES31.glDispatchCompute(groupCount(gridWidth), groupCount(gridHeight), 1)
            GLES31.glMemoryBarrier(
                GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT
            )
            checkGlError("readRegistrationSamples dispatch")

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, sampleBuffer)
            val mapped = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                0,
                byteCount,
                GLES31.GL_MAP_READ_BIT,
            ) ?: throw IllegalStateException("registration sample buffer map failed")
            val values = try {
                FloatArray(floatCount).also { out ->
                    val byteBuffer = mapped as? ByteBuffer
                        ?: throw IllegalStateException("registration sample buffer is not ByteBuffer")
                    byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer().get(out)
                }
            } finally {
                GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
            }
            decodeRegistrationSamples(values)
        } catch (e: Exception) {
            PLog.w(TAG, "Failed to read RAW registration samples", e)
            null
        } finally {
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, REGISTRATION_SAMPLE_BUFFER_BINDING, 0)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
            GLES31.glDeleteBuffers(1, intArrayOf(sampleBuffer), 0)
        }
    }

    private fun decodeRegistrationSamples(values: FloatArray): List<RawStackRegistrationSample> {
        val tileStride = registrationSampleTileStride()
        val samples = ArrayList<RawStackRegistrationSample>((gridWidth / tileStride + 1) * (gridHeight / tileStride + 1))
        for (tileY in 0 until gridHeight step tileStride) {
            for (tileX in 0 until gridWidth step tileStride) {
                val offset = (tileY * gridWidth + tileX) * REGISTRATION_SAMPLE_FLOAT_STRIDE
                if (offset + REGISTRATION_SAMPLE_FLOAT_STRIDE > values.size) continue
                samples += RawStackRegistrationSample(
                    referenceX = values[offset + 0],
                    referenceY = values[offset + 1],
                    targetX = values[offset + 2],
                    targetY = values[offset + 3],
                    robustness = values[offset + 4],
                    tileMask = values[offset + 5],
                    residual = values[offset + 6],
                    detail = values[offset + 7],
                )
            }
        }
        return samples
    }

    private fun registrationSampleTileStride(): Int {
        return max(1, max(gridWidth, gridHeight) / 128)
    }

    private fun recordRegistrationEstimate(
        estimate: RawStackRegistrationEstimate,
        effectiveTransform: RawStackPerspectiveTransform,
    ) {
        registrationEstimateCount += 1
        registrationConfidenceSum += estimate.confidence.toFloat()
        registrationConfidenceMin = minOf(registrationConfidenceMin, estimate.confidence)
        if (effectiveTransform.forceIdentity) {
            registrationForceIdentityCount += 1
        }
        registrationInlierRatioSum += estimate.inlierRatio.takeIf { it.isFinite() } ?: 0f
        if (estimate.residualP90Px.isFinite()) {
            registrationResidualP90Max = if (registrationResidualP90Max.isFinite()) {
                max(registrationResidualP90Max, estimate.residualP90Px)
            } else {
                estimate.residualP90Px
            }
        }
    }

    private fun recordRegistrationGlobalEstimate(estimate: RawStackRegistrationEstimate?) {
        if (estimate == null || !estimate.globalBestScore.isFinite()) return
        registrationGlobalScoreSum += estimate.globalBestScore
        registrationGlobalScoreCount += 1
        if (estimate.globalScoreMargin.isFinite()) {
            registrationGlobalMarginMin = minOf(registrationGlobalMarginMin, estimate.globalScoreMargin)
        }
        if (estimate.globalCoverage.isFinite()) {
            registrationGlobalCoverageSum += estimate.globalCoverage
        }
    }

    private fun registrationQualitySummary(): RawStackRegistrationQualitySummary? {
        if (registrationEstimateCount <= 0) return null
        return RawStackRegistrationQualitySummary(
            estimateCount = registrationEstimateCount,
            meanConfidence = registrationConfidenceSum / registrationEstimateCount.toFloat(),
            minConfidence = registrationConfidenceMin.takeIf { it != Int.MAX_VALUE } ?: 0,
            forceIdentityRatio = registrationForceIdentityCount.toFloat() / registrationEstimateCount.toFloat(),
            meanInlierRatio = registrationInlierRatioSum / registrationEstimateCount.toFloat(),
            residualP90MaxPx = registrationResidualP90Max,
            meanGlobalScore = if (registrationGlobalScoreCount > 0) {
                registrationGlobalScoreSum / registrationGlobalScoreCount.toFloat()
            } else {
                Float.NaN
            },
            minGlobalMargin = registrationGlobalMarginMin.takeIf { it.isFinite() } ?: Float.NaN,
            meanGlobalCoverage = if (registrationGlobalScoreCount > 0) {
                registrationGlobalCoverageSum / registrationGlobalScoreCount.toFloat()
            } else {
                Float.NaN
            },
        )
    }

    private fun clearAccumulator() {
        GLES31.glUseProgram(clearAccumulatorProgram)
        bindImage(0, accumulatorTexture, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(clearAccumulatorProgram, "uImageSize"), width, height)
        GLES31.glDispatchCompute(groupCount(width), groupCount(height), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("clearAccumulator")
        currentAccumulatorTexture = accumulatorTexture
    }

    private fun clearSuperResolutionAccumulator() {
        GLES31.glUseProgram(clearSuperResolutionAccumulatorProgram)
        bindImage(0, superResolutionAccumulatorTexture, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(clearSuperResolutionAccumulatorProgram, "uImageSize"),
            outputWidth,
            outputHeight,
        )
        GLES31.glDispatchCompute(groupCount(outputWidth), groupCount(outputHeight), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("clearSuperResolutionAccumulator")
        currentSuperResolutionAccumulatorTexture = superResolutionAccumulatorTexture
    }

    private fun accumulateFrame(
        rawTexture: Int,
        isReference: Boolean,
        exposureScale: Float = 1.0f,
        hdrMode: Boolean = false,
    ) {
        val outputAccumulator = if (currentAccumulatorTexture == accumulatorTexture) {
            accumulatorScratchTexture
        } else {
            accumulatorTexture
        }
        GLES31.glUseProgram(accumulateProgram)
        bindTexture(accumulateProgram, "uInputRaw", 0, rawTexture)
        bindTexture(accumulateProgram, "uFlowGrid", 1, flowTexture)
        bindTexture(accumulateProgram, "uRobustness", 2, robustnessTexture)
        bindTexture(accumulateProgram, "uTileMask", 3, tileMaskTexture)
        bindTexture(accumulateProgram, "uKernel", 4, kernelTexture)
        bindTexture(accumulateProgram, "uLensShadingMap", 5, lensShadingTexture)
        bindTexture(accumulateProgram, "uAccumulatorInput", 6, currentAccumulatorTexture)
        bindImage(0, outputAccumulator, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
        setCommonUniforms(accumulateProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(accumulateProgram, "uImageSize"), width, height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(accumulateProgram, "uPlaneSize"), planeWidth, planeHeight)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(accumulateProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(accumulateProgram, "uTileSize"), flowGridSpacing)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(accumulateProgram, "uIsReference"), if (isReference) 1 else 0)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(accumulateProgram, "uHdrMode"), if (hdrMode) 1 else 0)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(accumulateProgram, "uExposureScale"), exposureScale)
        setRegistrationUniforms(
            program = accumulateProgram,
            transform = if (isReference) {
                registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
            } else {
                currentRegistrationTransform
            },
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateProgram, "uFrameWeight"),
            if (isReference) 1.0f else hwmfBlend.nonReferenceFrameWeight,
        )
        setBlendAccumulatorUniforms(accumulateProgram)
        GLES31.glDispatchCompute(groupCount(width), groupCount(height), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("accumulateFrame")
        currentAccumulatorTexture = outputAccumulator
    }

    private fun accumulateSuperResolutionFrame(
        rawTexture: Int,
        isReference: Boolean,
    ) {
        val outputAccumulator = if (currentSuperResolutionAccumulatorTexture == superResolutionAccumulatorTexture) {
            superResolutionAccumulatorScratchTexture
        } else {
            superResolutionAccumulatorTexture
        }
        GLES31.glUseProgram(accumulateSuperResolutionProgram)
        bindTexture(accumulateSuperResolutionProgram, "uInputRaw", 0, rawTexture)
        bindTexture(accumulateSuperResolutionProgram, "uFlowGrid", 1, flowTexture)
        bindTexture(accumulateSuperResolutionProgram, "uRobustness", 2, robustnessTexture)
        bindTexture(accumulateSuperResolutionProgram, "uTileMask", 3, tileMaskTexture)
        bindTexture(accumulateSuperResolutionProgram, "uKernel", 4, kernelTexture)
        bindTexture(accumulateSuperResolutionProgram, "uLensShadingMap", 5, lensShadingTexture)
        bindTexture(accumulateSuperResolutionProgram, "uAccumulatorInput", 6, currentSuperResolutionAccumulatorTexture)
        bindImage(0, outputAccumulator, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
        setCommonUniforms(accumulateSuperResolutionProgram)
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uImageSize"),
            width,
            height,
        )
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uOutputSize"),
            outputWidth,
            outputHeight,
        )
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uPlaneSize"),
            planeWidth,
            planeHeight,
        )
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uGridSize"),
            gridWidth,
            gridHeight,
        )
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uTileSize"),
            flowGridSpacing,
        )
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uIsReference"),
            if (isReference) 1 else 0,
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uOutputScale"),
            superResolutionScale,
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uFrameWeight"),
            if (isReference) 1.0f else hwmfBlend.nonReferenceFrameWeight,
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uSrMinEffectiveWeight"),
            hwmfSr.minEffectiveWeight.coerceAtLeast(0f),
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uSrSplatRadius"),
            hwmfSr.splatRadius.coerceAtLeast(0.25f),
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateSuperResolutionProgram, "uRegistrationSrWeight"),
            if (isReference) 1.0f else currentRegistrationSrWeight.coerceIn(0.0f, 1.0f),
        )
        setRegistrationUniforms(
            program = accumulateSuperResolutionProgram,
            transform = if (isReference) {
                registrationSetup.identityTransform(RawStackRegistrationStage.BLEND)
            } else {
                currentRegistrationSrTransform
            },
        )
        setBlendAccumulatorUniforms(accumulateSuperResolutionProgram)
        GLES31.glDispatchCompute(groupCount(outputWidth), groupCount(outputHeight), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("accumulateSuperResolutionFrame")
        currentSuperResolutionAccumulatorTexture = outputAccumulator
    }

    private fun computeHdrRecoveryMask(shortToReferenceExposureScale: Float) {
        computeInitialHdrRecoveryMask(shortToReferenceExposureScale)
        filterHdrRecoveryMask(
            program = hdrRecoveryDilateProgram,
            input = hdrRecoveryOrigMaskTexture,
            output = hdrRecoveryScratchTexture,
            label = "computeHdrRecoveryMask dilate",
        )
        filterHdrRecoveryMask(
            program = hdrRecoveryFeatherProgram,
            input = hdrRecoveryScratchTexture,
            output = hdrRecoveryMaskTexture,
            label = "computeHdrRecoveryMask feather",
        )
    }

    private fun computeInitialHdrRecoveryMask(shortToReferenceExposureScale: Float) {
        bindFramebufferOutput(hdrRecoveryOrigMaskTexture, "computeHdrRecoveryMask initial")
        GLES30.glViewport(0, 0, planeWidth, planeHeight)
        GLES30.glUseProgram(hdrRecoveryMaskProgram)
        bindTexture(hdrRecoveryMaskProgram, "uAccumulator", 0, currentAccumulatorTexture)
        bindTexture(hdrRecoveryMaskProgram, "uReferenceRaw", 1, refRaw)
        bindTexture(hdrRecoveryMaskProgram, "uShortRaw", 2, hdrShortRaw)
        bindTexture(hdrRecoveryMaskProgram, "uShortGlobalAlignment", 3, hdrShortAlignmentTexture)
        setCommonUniforms(hdrRecoveryMaskProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(hdrRecoveryMaskProgram, "uImageSize"), width, height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(hdrRecoveryMaskProgram, "uPlaneSize"), planeWidth, planeHeight)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(hdrRecoveryMaskProgram, "uShortToReferenceExposureScale"),
            shortToReferenceExposureScale,
        )
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("computeHdrRecoveryMask initial")
    }

    private fun filterHdrRecoveryMask(program: Int, input: Int, output: Int, label: String) {
        bindFramebufferOutput(output, label)
        GLES30.glViewport(0, 0, planeWidth, planeHeight)
        GLES30.glUseProgram(program)
        bindTexture(program, "uMask", 0, input)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(program, "uPlaneSize"), planeWidth, planeHeight)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass(label)
    }

    private fun resetSuperResolutionDecisionStats() {
        superResolutionDetailFrameCount = 0
        superResolutionDetailWeightSum = 0f
        superResolutionDetailWeightMax = 0f
    }

    private fun recordSuperResolutionDetailFrame(weight: Float) {
        val clampedWeight = weight.coerceIn(0f, 1f)
        if (clampedWeight <= 0f) return
        superResolutionDetailFrameCount += 1
        superResolutionDetailWeightSum += clampedWeight
        superResolutionDetailWeightMax = max(superResolutionDetailWeightMax, clampedWeight)
    }

    private fun decideSuperResolutionOutput(alignedFrameCount: Int): SuperResolutionOutputDecision {
        val reasons = ArrayList<String>(3)
        if (alignedFrameCount < hwmfSr.fallbackMinAlignedFrames.coerceAtLeast(0)) {
            reasons += "aligned-low"
        }
        if (superResolutionDetailFrameCount < hwmfSr.fallbackMinDetailFrames.coerceAtLeast(0)) {
            reasons += "sr-detail-low"
        }
        if (superResolutionDetailWeightSum < hwmfSr.fallbackMinDetailWeightSum.coerceAtLeast(0f)) {
            reasons += "sr-weight-low"
        }
        val fallbackReason = reasons.takeIf { it.isNotEmpty() }?.joinToString("+")
        val mode = if (fallbackReason == null) "SUPER_RESOLUTION" else "BASE_UPSCALE"
        if (RawStackRuntimeDebug.enabled) {
            val message = "HWMF MFSR output effective=$mode " +
                "reason=${fallbackReason ?: "ok"} aligned=$alignedFrameCount " +
                "srFrames=$superResolutionDetailFrameCount " +
                "srWeightSum=${superResolutionDetailWeightSum.formatWeight()} " +
                "srWeightMax=${superResolutionDetailWeightMax.formatWeight()}"
            if (fallbackReason == null) {
                RawStackRuntimeDebug.d(TAG) { message }
            } else {
                RawStackRuntimeDebug.i(TAG) { message }
            }
        }
        return SuperResolutionOutputDecision(
            mode = mode,
            fallbackReason = fallbackReason,
            detailFrameCount = superResolutionDetailFrameCount,
            detailWeightSum = superResolutionDetailWeightSum,
        )
    }

    private fun normalizeOutput(
        hdrMode: Boolean = false,
        referenceExposureScale: Float = 1.0f,
        shortExposureScale: Float = 1.0f,
        targetTexture: Int = outputTexture,
        targetWidth: Int = width,
        targetHeight: Int = height,
        label: String = "normalizeOutput",
    ) {
        bindFramebufferOutput(targetTexture, label)
        GLES30.glViewport(0, 0, targetWidth, targetHeight)
        GLES30.glUseProgram(normalizeProgram)
        bindTexture(normalizeProgram, "uAccumulator", 0, currentAccumulatorTexture)
        bindTexture(normalizeProgram, "uReferenceRaw", 1, refRaw)
        bindTexture(normalizeProgram, "uShortRaw", 2, hdrShortRaw)
        bindTexture(normalizeProgram, "uLensShadingMap", 3, lensShadingTexture)
        bindTexture(normalizeProgram, "uShortGlobalAlignment", 4, hdrShortAlignmentTexture)
        bindTexture(normalizeProgram, "uHdrRecoveryMask", 5, hdrRecoveryMaskTexture)
        bindTexture(normalizeProgram, "uHdrRecoveryOrigMask", 6, hdrRecoveryOrigMaskTexture)
        setCommonUniforms(normalizeProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(normalizeProgram, "uImageSize"), width, height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(normalizeProgram, "uPlaneSize"), planeWidth, planeHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(normalizeProgram, "uHdrMode"), if (hdrMode) 1 else 0)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(normalizeProgram, "uReferenceExposureScale"),
            referenceExposureScale,
        )
        GLES31.glUniform1f(GLES31.glGetUniformLocation(normalizeProgram, "uShortExposureScale"), shortExposureScale)
        setPostfilterUniforms(normalizeProgram)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass(label)
    }

    private fun normalizeSuperResolutionOutput(decision: SuperResolutionOutputDecision) {
        normalizeOutput(
            targetTexture = superResolutionBaseTexture,
            targetWidth = width,
            targetHeight = height,
            label = "normalizeSuperResolutionBase",
        )
        bindFramebufferOutput(outputTexture, "normalizeSuperResolutionOutput")
        GLES30.glViewport(0, 0, outputWidth, outputHeight)
        GLES30.glUseProgram(normalizeSuperResolutionProgram)
        bindTexture(normalizeSuperResolutionProgram, "uSrAccumulator", 0, currentSuperResolutionAccumulatorTexture)
        bindTexture(normalizeSuperResolutionProgram, "uBasePackedRaw", 1, superResolutionBaseTexture)
        bindTexture(normalizeSuperResolutionProgram, "uReferenceRaw", 2, refRaw)
        bindTexture(normalizeSuperResolutionProgram, "uLensShadingMap", 3, lensShadingTexture)
        setCommonUniforms(normalizeSuperResolutionProgram)
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uImageSize"),
            width,
            height,
        )
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uOutputSize"),
            outputWidth,
            outputHeight,
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uOutputScale"),
            superResolutionScale,
        )
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uForceBaseUpscale"),
            if (decision.forceBaseUpscale) 1 else 0,
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uMinEffectiveWeight"),
            hwmfSr.minEffectiveWeight.coerceAtLeast(0f),
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uDetailRestoreStrength"),
            hwmfSr.detailRestoreStrength.coerceIn(0f, 1f),
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uSrDetailLowLightSuppression"),
            hwmfSr.detailLowLightSuppression.coerceIn(0f, 1f),
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(normalizeSuperResolutionProgram, "uSrDetailLscSuppression"),
            hwmfSr.detailLscSuppression.coerceIn(0f, 1f),
        )
        setPostfilterUniforms(normalizeSuperResolutionProgram)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("normalizeSuperResolutionOutput")
    }

    private fun recordAlignmentDiagnostics() {
        runDiagnosticStep("alignment") {
            GLES31.glUseProgram(diagnosticAlignmentProgram)
            bindTexture(diagnosticAlignmentProgram, "uFlowGrid", 0, flowTexture)
            bindTexture(diagnosticAlignmentProgram, "uRobustness", 1, robustnessTexture)
            bindTexture(diagnosticAlignmentProgram, "uTileMask", 2, tileMaskTexture)
            bindTexture(diagnosticAlignmentProgram, "uReference", 3, refProxy)
            bindTexture(diagnosticAlignmentProgram, "uCurrent", 4, curProxy)
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                DIAGNOSTIC_BUFFER_BINDING,
                diagnosticBuffer,
            )
            setDiagnosticSampleUniforms(diagnosticAlignmentProgram)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uPlaneSize"), planeWidth, planeHeight)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uGridSize"), gridWidth, gridHeight)
            GLES31.glUniform1i(GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uTileSize"), flowGridSpacing)
            setRegistrationUniforms(diagnosticAlignmentProgram, currentRegistrationTransform)
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uFlowHistogramRange"),
                diagnosticFlowHistogramRange(),
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uResidualHistogramRange"),
                diagnosticResidualHistogramRange(),
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uNoiseResidualHistogramRange"),
                diagnosticNoiseResidualHistogramRange(),
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uFlowRangeHistogramRange"),
                diagnosticFlowRangeHistogramRange(),
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uFlowOutlierThreshold"),
                hwmfBlend.flowOutlierThresholdPx.coerceAtLeast(0f),
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uTileRejectThreshold"),
                hwmfDebug.tileRejectThreshold,
            )
            GLES31.glUniform1f(GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uNoiseAlpha"), noiseAlpha)
            GLES31.glUniform1f(GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uNoiseBeta"), noiseBeta)
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uSrReadyResidualThreshold"),
                hwmfDebug.srReadyResidualThreshold,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uSrReadyNoiseResidualThreshold"),
                hwmfDebug.srReadyNoiseResidualThreshold,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uSrReadyFlowRangeThreshold"),
                hwmfDebug.srReadyFlowRangeThresholdPx,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uHighConfidenceRobustnessThreshold"),
                hwmfDebug.highConfidenceRobustnessThreshold,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uHighConfidenceTileThreshold"),
                hwmfDebug.highConfidenceTileThreshold,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticAlignmentProgram, "uSrReadyDetailThreshold"),
                hwmfDebug.srReadyDetailThreshold,
            )
            GLES31.glDispatchCompute(groupCount(diagnosticSampleWidth()), groupCount(diagnosticSampleHeight()), 1)
            GLES31.glMemoryBarrier(
                GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT
            )
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, DIAGNOSTIC_BUFFER_BINDING, 0)
            checkGlError("recordAlignmentDiagnostics")
        }
    }

    private fun collectFinalDiagnostics(
        frameCount: Int,
        alignedFrameCount: Int,
        elapsedMs: Long,
        superResolutionDecision: SuperResolutionOutputDecision = SuperResolutionOutputDecision.Disabled,
    ): RawStackDiagnostics? {
        if (!hwmfDebug.collectMetrics) return null
        val diagnosticAccumulatorTexture = if (superResolutionEnabled) {
            currentSuperResolutionAccumulatorTexture
        } else {
            currentAccumulatorTexture
        }
        val diagnosticAccumulatorWidth = if (superResolutionEnabled) outputWidth else width
        val diagnosticAccumulatorHeight = if (superResolutionEnabled) outputHeight else height
        val diagnosticOutputScale = if (superResolutionEnabled) superResolutionScale else 1.0f
        val finalRecorded = runDiagnosticStep("final") {
            GLES31.glUseProgram(diagnosticFinalProgram)
            bindTexture(diagnosticFinalProgram, "uAccumulator", 0, diagnosticAccumulatorTexture)
            bindTexture(diagnosticFinalProgram, "uLensShadingMap", 1, lensShadingTexture)
            bindTexture(diagnosticFinalProgram, "uPostfilterAccumulator", 2, currentAccumulatorTexture)
            setCommonUniforms(diagnosticFinalProgram)
            setPostfilterUniforms(diagnosticFinalProgram)
            GLES31.glBindBufferBase(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                DIAGNOSTIC_BUFFER_BINDING,
                diagnosticBuffer,
            )
            setDiagnosticSampleUniforms(diagnosticFinalProgram)
            GLES31.glUniform2i(
                GLES31.glGetUniformLocation(diagnosticFinalProgram, "uAccumulatorSize"),
                diagnosticAccumulatorWidth,
                diagnosticAccumulatorHeight,
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticFinalProgram, "uOutputScale"),
                diagnosticOutputScale,
            )
            GLES31.glUniform1i(GLES31.glGetUniformLocation(diagnosticFinalProgram, "uCfaPattern"), cfaPattern)
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticFinalProgram, "uWeightHistogramRange"),
                diagnosticWeightHistogramRange(frameCount),
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticFinalProgram, "uPostResidualHistogramRange"),
                diagnosticPostResidualHistogramRange(),
            )
            GLES31.glUniform1f(
                GLES31.glGetUniformLocation(diagnosticFinalProgram, "uLensShadingEdgeFraction"),
                hwmfDebug.lensShadingEdgeFraction,
            )
            GLES31.glDispatchCompute(groupCount(diagnosticSampleWidth()), groupCount(diagnosticSampleHeight()), 1)
            GLES31.glMemoryBarrier(
                GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT
            )
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, DIAGNOSTIC_BUFFER_BINDING, 0)
            checkGlError("collectFinalDiagnostics")
        }
        if (!finalRecorded) return null
        val stats = readDiagnosticStats() ?: return null
        val diagnostics = buildDiagnostics(
            stats = stats,
            frameCount = frameCount,
            alignedFrameCount = alignedFrameCount,
            elapsedMs = elapsedMs,
            superResolutionDecision = superResolutionDecision,
        )
        if (hwmfDebug.logCompactSummary) {
            PLog.i(TAG, diagnostics.compactSummary())
        }
        return diagnostics
    }

    private fun runDiagnosticStep(label: String, block: () -> Unit): Boolean {
        if (!hwmfDebug.collectMetrics ||
            diagnosticsFailed ||
            diagnosticBuffer == 0 ||
            diagnosticAlignmentProgram == 0 ||
            diagnosticFinalProgram == 0
        ) {
            return false
        }
        return try {
            block()
            true
        } catch (e: Exception) {
            diagnosticsFailed = true
            PLog.w(TAG, "RAW stack diagnostics $label step failed; diagnostics disabled", e)
            false
        }
    }

    private fun setDiagnosticSampleUniforms(program: Int) {
        GLES31.glUniform2i(GLES31.glGetUniformLocation(program, "uImageSize"), width, height)
        GLES31.glUniform2i(
            GLES31.glGetUniformLocation(program, "uSampleGridSize"),
            diagnosticSampleWidth(),
            diagnosticSampleHeight(),
        )
        GLES31.glUniform1i(GLES31.glGetUniformLocation(program, "uSampleStep"), hwmfDebug.sampleStep)
    }

    private fun readDiagnosticStats(): IntArray? {
        if (diagnosticBuffer == 0 || diagnosticsFailed) return null
        return try {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, diagnosticBuffer)
            val mapped = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                0,
                DIAGNOSTIC_UINT_COUNT * 4,
                GLES31.GL_MAP_READ_BIT,
            ) ?: throw IllegalStateException("diagnostics stats buffer map failed")
            try {
                val byteBuffer = mapped as? ByteBuffer
                    ?: throw IllegalStateException("diagnostics stats buffer is not ByteBuffer")
                IntArray(DIAGNOSTIC_UINT_COUNT).also { values ->
                    byteBuffer.order(ByteOrder.nativeOrder()).asIntBuffer().get(values)
                }
            } finally {
                GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
            }
        } catch (e: Exception) {
            diagnosticsFailed = true
            PLog.w(TAG, "Failed to read RAW stack diagnostics", e)
            null
        } finally {
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
        }
    }

    private fun buildDiagnostics(
        stats: IntArray,
        frameCount: Int,
        alignedFrameCount: Int,
        elapsedMs: Long,
        superResolutionDecision: SuperResolutionOutputDecision,
    ): RawStackDiagnostics {
        val flowCount = uintStat(stats, DIAGNOSTIC_FLOW_COUNT_INDEX)
        val robustCount = uintStat(stats, DIAGNOSTIC_ROBUST_COUNT_INDEX)
        val tileCount = uintStat(stats, DIAGNOSTIC_TILE_COUNT_INDEX)
        val weightCount = uintStat(stats, DIAGNOSTIC_WEIGHT_COUNT_INDEX)
        val lscCount = uintStat(stats, DIAGNOSTIC_LSC_COUNT_INDEX)
        val lscEdgeCount = uintStat(stats, DIAGNOSTIC_LSC_EDGE_COUNT_INDEX)
        val residualCount = uintStat(stats, DIAGNOSTIC_RESIDUAL_COUNT_INDEX)
        val highConfidenceCount = uintStat(stats, DIAGNOSTIC_HIGH_CONFIDENCE_COUNT_INDEX)

        return RawStackDiagnostics(
            mode = tuning.mode,
            frameCount = frameCount,
            alignedFrameCount = alignedFrameCount,
            width = outputWidth,
            height = outputHeight,
            sampleStep = hwmfDebug.sampleStep,
            registration = registrationSummary,
            registrationQuality = registrationQualitySummary(),
            superResolutionOutputMode = superResolutionDecision.mode.takeIf { it != "n/a" },
            superResolutionFallbackReason = superResolutionDecision.fallbackReason,
            superResolutionDetailFrameCount = superResolutionDecision.detailFrameCount,
            superResolutionDetailWeightSum = superResolutionDecision.detailWeightSum,
            flowMagnitudePx = metricDistribution(
                stats = stats,
                count = flowCount,
                sumIndex = DIAGNOSTIC_FLOW_SUM_INDEX,
                maxIndex = DIAGNOSTIC_FLOW_MAX_INDEX,
                quantization = DIAGNOSTIC_FLOW_QUANTIZATION,
                histOffset = DIAGNOSTIC_FLOW_HIST_OFFSET,
                histRange = diagnosticFlowHistogramRange(),
            ),
            alignmentResidual = metricDistribution(
                stats = stats,
                count = residualCount,
                sumIndex = DIAGNOSTIC_RESIDUAL_SUM_INDEX,
                maxIndex = DIAGNOSTIC_RESIDUAL_MAX_INDEX,
                quantization = DIAGNOSTIC_RESIDUAL_QUANTIZATION,
                histOffset = DIAGNOSTIC_RESIDUAL_HIST_OFFSET,
                histRange = diagnosticResidualHistogramRange(),
            ),
            noiseNormalizedResidual = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_NOISE_RESIDUAL_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_NOISE_RESIDUAL_SUM_INDEX,
                maxIndex = DIAGNOSTIC_NOISE_RESIDUAL_MAX_INDEX,
                quantization = DIAGNOSTIC_NOISE_RESIDUAL_QUANTIZATION,
                histOffset = DIAGNOSTIC_NOISE_RESIDUAL_HIST_OFFSET,
                histRange = diagnosticNoiseResidualHistogramRange(),
            ),
            flowLocalRangePx = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_FLOW_RANGE_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_FLOW_RANGE_SUM_INDEX,
                maxIndex = DIAGNOSTIC_FLOW_RANGE_MAX_INDEX,
                quantization = DIAGNOSTIC_FLOW_QUANTIZATION,
                histOffset = DIAGNOSTIC_FLOW_RANGE_HIST_OFFSET,
                histRange = diagnosticFlowRangeHistogramRange(),
            ),
            robustness = metricDistribution(
                stats = stats,
                count = robustCount,
                sumIndex = DIAGNOSTIC_ROBUST_SUM_INDEX,
                maxIndex = DIAGNOSTIC_ROBUST_MAX_INDEX,
                quantization = DIAGNOSTIC_UNIT_QUANTIZATION,
                histOffset = DIAGNOSTIC_ROBUST_HIST_OFFSET,
                histRange = 1.0f,
            ),
            tileMask = metricDistribution(
                stats = stats,
                count = tileCount,
                sumIndex = DIAGNOSTIC_TILE_SUM_INDEX,
                maxIndex = DIAGNOSTIC_TILE_MAX_INDEX,
                quantization = DIAGNOSTIC_UNIT_QUANTIZATION,
                histOffset = DIAGNOSTIC_TILE_HIST_OFFSET,
                histRange = 1.0f,
            ),
            accumulatorWeight = metricDistribution(
                stats = stats,
                count = weightCount,
                sumIndex = DIAGNOSTIC_WEIGHT_SUM_INDEX,
                maxIndex = DIAGNOSTIC_WEIGHT_MAX_INDEX,
                quantization = DIAGNOSTIC_WEIGHT_QUANTIZATION,
                histOffset = DIAGNOSTIC_WEIGHT_HIST_OFFSET,
                histRange = diagnosticWeightHistogramRange(frameCount),
            ),
            postfilterResidual = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_POST_RESIDUAL_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_POST_RESIDUAL_SUM_INDEX,
                maxIndex = DIAGNOSTIC_POST_RESIDUAL_MAX_INDEX,
                quantization = DIAGNOSTIC_POST_RESIDUAL_QUANTIZATION,
                histOffset = DIAGNOSTIC_POST_RESIDUAL_HIST_OFFSET,
                histRange = diagnosticPostResidualHistogramRange(),
            ),
            postfilterSmooth = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_POST_SMOOTH_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_POST_SMOOTH_SUM_INDEX,
                maxIndex = DIAGNOSTIC_POST_SMOOTH_MAX_INDEX,
                quantization = DIAGNOSTIC_UNIT_QUANTIZATION,
                histOffset = DIAGNOSTIC_POST_SMOOTH_HIST_OFFSET,
                histRange = 1.0f,
            ),
            postfilterEffectiveSmooth = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_SUM_INDEX,
                maxIndex = DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_MAX_INDEX,
                quantization = DIAGNOSTIC_UNIT_QUANTIZATION,
                histOffset = DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_HIST_OFFSET,
                histRange = 1.0f,
            ),
            postfilterWienerGain = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_POST_WIENER_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_POST_WIENER_SUM_INDEX,
                maxIndex = DIAGNOSTIC_POST_WIENER_MAX_INDEX,
                quantization = DIAGNOSTIC_UNIT_QUANTIZATION,
                histOffset = DIAGNOSTIC_POST_WIENER_HIST_OFFSET,
                histRange = 1.0f,
            ),
            postfilterLscBoost = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_POST_LSC_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_POST_LSC_SUM_INDEX,
                maxIndex = DIAGNOSTIC_POST_LSC_MAX_INDEX,
                quantization = DIAGNOSTIC_UNIT_QUANTIZATION,
                histOffset = DIAGNOSTIC_POST_LSC_HIST_OFFSET,
                histRange = 1.0f,
            ),
            postfilterLowWeightBoost = metricDistribution(
                stats = stats,
                count = uintStat(stats, DIAGNOSTIC_POST_LOW_WEIGHT_COUNT_INDEX),
                sumIndex = DIAGNOSTIC_POST_LOW_WEIGHT_SUM_INDEX,
                maxIndex = DIAGNOSTIC_POST_LOW_WEIGHT_MAX_INDEX,
                quantization = DIAGNOSTIC_UNIT_QUANTIZATION,
                histOffset = DIAGNOSTIC_POST_LOW_WEIGHT_HIST_OFFSET,
                histRange = 1.0f,
            ),
            rejectedTileRatio = ratioStat(
                numerator = uintStat(stats, DIAGNOSTIC_TILE_REJECT_COUNT_INDEX),
                denominator = tileCount,
            ),
            flowOutlierRatio = ratioStat(
                numerator = uintStat(stats, DIAGNOSTIC_FLOW_OUTLIER_COUNT_INDEX),
                denominator = flowCount,
            ),
            highConfidenceTileRatio = ratioStat(
                numerator = highConfidenceCount,
                denominator = residualCount,
            ),
            srAlignmentReadyRatio = ratioStat(
                numerator = uintStat(stats, DIAGNOSTIC_SR_ALIGNMENT_READY_COUNT_INDEX),
                denominator = residualCount,
            ),
            srDetailReadyRatio = ratioStat(
                numerator = uintStat(stats, DIAGNOSTIC_SR_DETAIL_READY_COUNT_INDEX),
                denominator = residualCount,
            ),
            lensShadingMeanGain = meanQuantizedStat(
                sum = uintStat(stats, DIAGNOSTIC_LSC_SUM_INDEX),
                count = lscCount,
                quantization = DIAGNOSTIC_LSC_QUANTIZATION,
            ),
            lensShadingEdgeMeanGain = meanQuantizedStat(
                sum = uintStat(stats, DIAGNOSTIC_LSC_EDGE_SUM_INDEX),
                count = lscEdgeCount,
                quantization = DIAGNOSTIC_LSC_QUANTIZATION,
            ),
            elapsedMs = elapsedMs,
        )
    }

    private fun metricDistribution(
        stats: IntArray,
        count: Long,
        sumIndex: Int,
        maxIndex: Int,
        quantization: Float,
        histOffset: Int,
        histRange: Float,
    ): RawStackMetricDistribution {
        if (count <= 0L || quantization <= 0f) return RawStackMetricDistribution.Empty
        return RawStackMetricDistribution(
            sampleCount = count,
            mean = meanQuantizedStat(uintStat(stats, sumIndex), count, quantization),
            p10 = histogramPercentile(stats, histOffset, count, 0.10f, histRange),
            p50 = histogramPercentile(stats, histOffset, count, 0.50f, histRange),
            p90 = histogramPercentile(stats, histOffset, count, 0.90f, histRange),
            max = uintStat(stats, maxIndex).toFloat() / quantization,
        )
    }

    private fun histogramPercentile(
        stats: IntArray,
        histOffset: Int,
        count: Long,
        percentile: Float,
        range: Float,
    ): Float {
        if (count <= 0L || range <= 0f) return Float.NaN
        val target = max(1L, ceil(count.toDouble() * percentile.toDouble()).toLong())
        var cumulative = 0L
        for (bin in 0 until DIAGNOSTIC_HIST_BINS) {
            cumulative += uintStat(stats, histOffset + bin)
            if (cumulative >= target) {
                return if (DIAGNOSTIC_HIST_BINS > 1) {
                    range * bin.toFloat() / (DIAGNOSTIC_HIST_BINS - 1).toFloat()
                } else {
                    0f
                }
            }
        }
        return range
    }

    private fun meanQuantizedStat(sum: Long, count: Long, quantization: Float): Float {
        return if (count > 0L && quantization > 0f) {
            sum.toFloat() / quantization / count.toFloat()
        } else {
            Float.NaN
        }
    }

    private fun ratioStat(numerator: Long, denominator: Long): Float {
        return if (denominator > 0L) {
            numerator.toFloat() / denominator.toFloat()
        } else {
            Float.NaN
        }
    }

    private fun uintStat(stats: IntArray, index: Int): Long {
        return stats.getOrElse(index) { 0 }.toLong() and 0xFFFF_FFFFL
    }

    private fun diagnosticSampleWidth(): Int {
        return (width + hwmfDebug.sampleStep - 1) / hwmfDebug.sampleStep
    }

    private fun diagnosticSampleHeight(): Int {
        return (height + hwmfDebug.sampleStep - 1) / hwmfDebug.sampleStep
    }

    private fun scaledRawDimension(size: Int, scale: Float): Int {
        val scaled = (size.toFloat() * scale.coerceAtLeast(1.0f)).roundToInt().coerceAtLeast(1)
        return if (scale > 1.0f && scaled > 1 && scaled % 2 != 0) {
            scaled - 1
        } else {
            scaled
        }
    }

    private fun diagnosticFlowHistogramRange(): Float {
        return max(1.0f, hwmfBlend.flowOutlierThresholdPx.coerceAtLeast(0f) * 2.0f)
    }

    private fun diagnosticWeightHistogramRange(frameCount: Int): Float {
        return max(1.0f, frameCount.coerceAtLeast(1).toFloat() * 2.0f)
    }

    private fun diagnosticResidualHistogramRange(): Float {
        return max(0.04f, hwmfDebug.srReadyResidualThreshold * 4.0f)
    }

    private fun diagnosticNoiseResidualHistogramRange(): Float {
        return max(4.0f, hwmfDebug.srReadyNoiseResidualThreshold * 4.0f)
    }

    private fun diagnosticFlowRangeHistogramRange(): Float {
        return max(4.0f, hwmfDebug.srReadyFlowRangeThresholdPx * 4.0f)
    }

    private fun diagnosticPostResidualHistogramRange(): Float {
        return 12.0f
    }

    private fun Float.percentString(): String {
        return if (isFinite()) {
            java.lang.String.format(java.util.Locale.US, "%.1f%%", this * 100f)
        } else {
            "n/a"
        }
    }

    private fun Float.formatPx(): String {
        return if (isFinite()) {
            java.lang.String.format(java.util.Locale.US, "%.2fpx", this)
        } else {
            "n/a"
        }
    }

    private fun Float.formatScale(): String {
        return if (isFinite()) {
            java.lang.String.format(java.util.Locale.US, "%.2f", this)
        } else {
            "n/a"
        }
    }

    private fun Float.formatWeight(): String {
        return if (isFinite()) {
            java.lang.String.format(java.util.Locale.US, "%.3f", this)
        } else {
            "n/a"
        }
    }

    private fun Long.mibString(): String {
        return java.lang.String.format(java.util.Locale.US, "%.1fMiB", this.toDouble() / (1024.0 * 1024.0))
    }

    private fun Float.formatScore(): String {
        return if (isFinite()) {
            java.lang.String.format(java.util.Locale.US, "%.4f", this)
        } else {
            "n/a"
        }
    }

    private fun RawStackRegistrationEstimate.globalSummary(): String {
        return if (globalBestScore.isFinite()) {
            "score=${globalBestScore.formatScore()} margin=${globalScoreMargin.formatScore()} " +
                "coverage=${globalCoverage.percentString()}"
        } else {
            "n/a"
        }
    }

    private fun RawStackRegistrationEstimate?.globalSummaryOrEmpty(): String {
        return this?.globalSummary() ?: "n/a"
    }

    private fun FloatArray.vectorSummary(): String {
        return if (size >= 2 && this[0].isFinite() && this[1].isFinite()) {
            java.lang.String.format(java.util.Locale.US, "[%.2f %.2f]px", this[0], this[1])
        } else {
            "n/a"
        }
    }

    private fun RegistrationTranslation?.translationSummary(): String {
        return if (this != null && dx.isFinite() && dy.isFinite()) {
            java.lang.String.format(java.util.Locale.US, "[%.2f %.2f]px", dx, dy)
        } else {
            "n/a"
        }
    }

    private fun RegistrationTranslation?.seedDistance(seed: RegistrationTranslation?): Float {
        return if (this != null && seed != null) {
            distanceTo(seed)
        } else {
            Float.NaN
        }
    }

    private fun FloatArray.matrixSummary(): String {
        if (size < 9) return "n/a"
        return java.lang.String.format(
            java.util.Locale.US,
            "[%.5f %.5f %.2f; %.5f %.5f %.2f; %.5f %.5f %.5f]",
            this[0], this[1], this[2],
            this[3], this[4], this[5],
            this[6], this[7], this[8],
        )
    }

    private fun readOutput(outputBuffer: ByteBuffer): ReadOutputTiming {
        val startTime = System.currentTimeMillis()
        try {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, readbackFbo)
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER,
                GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D,
                outputTexture,
                0,
            )
            GLES30.glReadBuffer(GLES30.GL_COLOR_ATTACHMENT0)
            checkFramebuffer("readOutput")
            GLES30.glPixelStorei(GLES30.GL_PACK_ALIGNMENT, 1)
            GLES30.glViewport(0, 0, outputWidth, outputHeight)
            outputBuffer.clear()
            val directReadStart = System.currentTimeMillis()
            GLES30.glReadPixels(0, 0, outputWidth, outputHeight, GLES30.GL_RG, GLES30.GL_UNSIGNED_BYTE, outputBuffer)
            val directReadMs = System.currentTimeMillis() - directReadStart
            val directReadError = GLES30.glGetError()
            if (directReadError == GLES30.GL_NO_ERROR) {
                outputBuffer.rewind()
                return ReadOutputTiming(
                    elapsedMs = System.currentTimeMillis() - startTime,
                    glReadMs = directReadMs,
                    copyMs = 0L,
                    allocMs = 0L,
                    mode = "rg8-direct",
                )
            }

            PLog.w(
                TAG,
                "GLES RAW direct RG readback failed: 0x${directReadError.toString(16)}, falling back to RGBA unpack"
            )
            outputBuffer.clear()
            return readOutputViaRgbaFallback(outputBuffer, startTime)
        } finally {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        }
    }

    private fun readOutputViaRgbaFallback(outputBuffer: ByteBuffer, readbackStartTime: Long): ReadOutputTiming {
        val allocStart = System.currentTimeMillis()
        val packed = LargeDirectBuffer.allocate(
            outputWidth.toLong() * outputHeight.toLong() * 4L,
            "GLES RAW packed readback",
        )
            ?: throw IllegalStateException("Failed to allocate GLES RAW packed readback")
        val allocMs = System.currentTimeMillis() - allocStart
        try {
            packed.clear()
            val glReadStart = System.currentTimeMillis()
            GLES30.glReadPixels(0, 0, outputWidth, outputHeight, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, packed)
            val glReadMs = System.currentTimeMillis() - glReadStart
            checkGlError("readOutput fallback")
            packed.rewind()
            outputBuffer.clear()
            val copyStart = System.currentTimeMillis()
            val outShorts = outputBuffer.order(ByteOrder.nativeOrder()).asShortBuffer()
            val pixelCount = outputWidth * outputHeight
            for (index in 0 until pixelCount) {
                val lo = packed.get(index * 4).toInt() and 0xFF
                val hi = packed.get(index * 4 + 1).toInt() and 0xFF
                outShorts.put(index, ((hi shl 8) or lo).toShort())
            }
            outputBuffer.rewind()
            return ReadOutputTiming(
                elapsedMs = System.currentTimeMillis() - readbackStartTime,
                glReadMs = glReadMs,
                copyMs = System.currentTimeMillis() - copyStart,
                allocMs = allocMs,
                mode = "rgba-fallback",
            )
        } finally {
            LargeDirectBuffer.free(packed)
        }
    }

    private fun bindFramebufferOutput(texture: Int, label: String) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, renderFbo)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            texture,
            0,
        )
        GLES30.glDrawBuffers(1, intArrayOf(GLES30.GL_COLOR_ATTACHMENT0), 0)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDisable(GLES30.GL_DITHER)
        checkFramebuffer(label)
    }

    private fun finishFramebufferPass(label: String) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES31.glMemoryBarrier(GLES31.GL_FRAMEBUFFER_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError(label)
    }

    private fun setCommonUniforms(program: Int) {
        GLES31.glUniform1i(GLES31.glGetUniformLocation(program, "uCfaPattern"), cfaPattern)
        GLES31.glUniform1fv(GLES31.glGetUniformLocation(program, "uBlackLevel[0]"), 4, normalizedBlackLevel, 0)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uWhiteLevel"), normalizedWhiteLevel)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uNoiseAlpha"), noiseAlpha)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uNoiseBeta"), noiseBeta)
        GLES31.glUniform1fv(
            GLES31.glGetUniformLocation(program, "uNoiseAlphaByChannel[0]"),
            4,
            normalizedNoiseAlphaByChannel,
            0
        )
        GLES31.glUniform1fv(
            GLES31.glGetUniformLocation(program, "uNoiseBetaByChannel[0]"),
            4,
            normalizedNoiseBetaByChannel,
            0
        )
    }

    private fun setPrefilterStructureUniforms(program: Int) {
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uStructureFlatnessSnrLow"), hwmfPrefilter.structureFlatnessSnrLow)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uStructureFlatnessSnrHigh"), hwmfPrefilter.structureFlatnessSnrHigh)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uStructureKernelDetail"), hwmfPrefilter.structureKernelDetail)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uStructureKernelDenoise"), hwmfPrefilter.structureKernelDenoise)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uStructureKernelShrink"), hwmfPrefilter.structureKernelShrink)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uStructureKernelStretch"), hwmfPrefilter.structureKernelStretch)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uStructureAnisotropyThreshold"),
            hwmfPrefilter.structureAnisotropyThreshold
        )
    }

    private fun setBlendRobustnessUniforms(program: Int) {
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uRobustNoiseFloorSpatialScale"),
            hwmfBlend.robustnessNoiseFloorSpatialScale
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uRobustNoiseFloorEdgeScale"),
            hwmfBlend.robustnessNoiseFloorEdgeScale
        )
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustTauBase"), hwmfBlend.robustnessTauBase)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustTauEdge"), hwmfBlend.robustnessTauEdge)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustResidualPower"), hwmfBlend.robustnessResidualPower)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustFlowPenaltyStartPx"), hwmfBlend.flowPenaltyStartPx)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustFlowPenaltyDecay"), hwmfBlend.flowPenaltyDecay)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uRobustFlowRangePenaltyStartPx"),
            hwmfBlend.flowRangePenaltyStartPx
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uRobustFlowRangePenaltyDecay"),
            hwmfBlend.flowRangePenaltyDecay
        )
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustMinMixFlat"), hwmfBlend.robustMinMixFlat)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustMinMixEdge"), hwmfBlend.robustMinMixEdge)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustCenterMixFlat"), hwmfBlend.robustCenterMixFlat)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustCenterMixEdge"), hwmfBlend.robustCenterMixEdge)
    }

    private fun setTileMaskUniforms(program: Int) {
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileRobustCenter"), hwmfBlend.tileRobustCenter)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileRobustWidth"), hwmfBlend.tileRobustWidth)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileWeakThreshold"), hwmfBlend.tileWeakThreshold)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileWeakStart"), hwmfBlend.tileWeakStart)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileWeakRange"), hwmfBlend.tileWeakRange)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileDetailMid"), hwmfBlend.tileDetailMid)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileDetailHigh"), hwmfBlend.tileDetailHigh)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileDetailBoostLow"), hwmfBlend.tileDetailBoostLow)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileDetailBoostMid"), hwmfBlend.tileDetailBoostMid)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileDetailBoostHigh"), hwmfBlend.tileDetailBoostHigh)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileMaskMinMidDetail"), hwmfBlend.tileMaskMinMidDetail)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uTileMaskMinHighDetail"), hwmfBlend.tileMaskMinHighDetail)
    }

    private fun setBlendAccumulatorUniforms(program: Int) {
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uMinBlendBaseWeight"), hwmfBlend.minBlendBaseWeight)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uRobustnessFloorFactor"), hwmfBlend.robustnessFloorFactor)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uPrecisionReferenceSignal"),
            hwmfBlend.sensorPrecisionReferenceSignal
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLscNoiseGainMax"),
            hwmfBlend.lscNoiseGainMax.coerceAtLeast(1.0f)
        )
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uWienerBaseWeight"), hwmfBlend.wienerBaseWeight)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uDenoiseSignalLow"), hwmfBlend.denoiseSignalLow)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uDenoiseSignalHigh"), hwmfBlend.denoiseSignalHigh)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseLscGainStart"),
            hwmfBlend.denoiseLscGainStart
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseLscGainEnd"),
            hwmfBlend.denoiseLscGainEnd
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseStaticRobustStart"),
            hwmfBlend.denoiseStaticRobustStart
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseStaticRobustEnd"),
            hwmfBlend.denoiseStaticRobustEnd
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseStaticTileStart"),
            hwmfBlend.denoiseStaticTileStart
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseStaticTileEnd"),
            hwmfBlend.denoiseStaticTileEnd
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseNonReferenceWeightBoost"),
            hwmfBlend.denoiseNonReferenceWeightBoost
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseNonReferenceWeightFloor"),
            hwmfBlend.denoiseNonReferenceWeightFloor
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDenoiseReferenceDarkWeightScale"),
            hwmfBlend.denoiseReferenceDarkWeightScale
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uHighlightSuppressionStrength"),
            hwmfBlend.highlightSuppressionStrength
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uHighlightSuppressionStart"),
            hwmfBlend.highlightSuppressionStart
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uHighlightSuppressionEnd"),
            hwmfBlend.highlightSuppressionEnd
        )
    }

    private fun setRegistrationUniforms(program: Int, transform: RawStackPerspectiveTransform) {
        GLES31.glUniformMatrix3fv(
            GLES31.glGetUniformLocation(program, "uRegistrationTransform"),
            1,
            false,
            transposeMatrix3x3(transform.matrixAt(0)),
            0,
        )
        GLES31.glUniform1i(
            GLES31.glGetUniformLocation(program, "uRegistrationForceIdentity"),
            if (transform.forceIdentity) 1 else 0,
        )
    }

    private fun setPostfilterUniforms(program: Int) {
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uFinalSmoothStrength"), hwmfPostfilter.finalSmoothStrength)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uFlatVarianceStart"), hwmfPostfilter.flatVarianceStart)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uFlatVarianceEnd"), hwmfPostfilter.flatVarianceEnd)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDetailKeepNoiseLowScale"),
            hwmfPostfilter.detailKeepNoiseLowScale
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDetailKeepNoiseHighScale"),
            hwmfPostfilter.detailKeepNoiseHighScale
        )
        GLES31.glUniform1f(GLES31.glGetUniformLocation(program, "uDetailKeepOffsetLow"), hwmfPostfilter.detailKeepOffsetLow)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDetailKeepOffsetHigh"),
            hwmfPostfilter.detailKeepOffsetHigh
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uDetailKeepSuppression"),
            hwmfPostfilter.detailKeepSuppression
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLscNoiseGainMax"),
            hwmfPostfilter.lscNoiseGainMax.coerceAtLeast(1.0f)
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uPostfilterNoiseFloor"),
            hwmfPostfilter.noiseFloorVariance.coerceAtLeast(0f)
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLowLightSignalLow"),
            hwmfPostfilter.lowLightSignalLow
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLowLightSignalHigh"),
            hwmfPostfilter.lowLightSignalHigh
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLowLightSmoothBoost"),
            hwmfPostfilter.lowLightSmoothBoost
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLscSmoothGainStart"),
            hwmfPostfilter.lscSmoothGainStart
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLscSmoothGainEnd"),
            hwmfPostfilter.lscSmoothGainEnd
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLscSmoothBoost"),
            hwmfPostfilter.lscSmoothBoost
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLowWeightStart"),
            hwmfPostfilter.lowWeightStart
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLowWeightEnd"),
            hwmfPostfilter.lowWeightEnd
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uLowWeightSmoothBoost"),
            hwmfPostfilter.lowWeightSmoothBoost
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uResidualNoiseWeight"),
            hwmfPostfilter.residualNoiseWeight.coerceAtLeast(0f)
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uResidualNoiseLowScale"),
            hwmfPostfilter.residualNoiseLowScale.coerceAtLeast(0f)
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uResidualNoiseHighScale"),
            hwmfPostfilter.residualNoiseHighScale.coerceAtLeast(hwmfPostfilter.residualNoiseLowScale)
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uResidualSmoothBoost"),
            hwmfPostfilter.residualSmoothBoost.coerceAtLeast(0f)
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uMaxSmoothStrength"),
            hwmfPostfilter.maxSmoothStrength.coerceIn(0f, 1f)
        )
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(program, "uHdrRecoverySmoothSuppression"),
            hwmfPostfilter.hdrRecoverySmoothSuppression
        )
    }

    private fun createTexture2D(textureWidth: Int, textureHeight: Int, internalFormat: Int, filter: Int): Int {
        val ids = IntArray(1)
        GLES30.glGenTextures(1, ids, 0)
        val texture = ids[0]
        textures += texture
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, filter)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, filter)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, internalFormat, textureWidth, textureHeight)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        checkGlError("createTexture2D ${textureWidth}x$textureHeight")
        return texture
    }

    private fun createLensShadingTexture(): Int {
        val hasValidLensShading = lensShading != null &&
            lensShadingWidth > 0 &&
            lensShadingHeight > 0 &&
            lensShading.size >= lensShadingWidth * lensShadingHeight * 4
        val texWidth = if (hasValidLensShading) lensShadingWidth else 1
        val texHeight = if (hasValidLensShading) lensShadingHeight else 1
        val gainCount = texWidth * texHeight * 4
        val gains = FloatArray(gainCount) { 1.0f }
        lensShading?.takeIf { hasValidLensShading }?.let { sourceLensShading ->
            for (index in gains.indices) {
                gains[index] = sourceLensShading[index].takeIf { it.isFinite() && it > 0f } ?: 1.0f
            }
        }
        val buffer = ByteBuffer.allocateDirect(gains.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(gains)
                position(0)
            }
        val ids = IntArray(1)
        GLES30.glGenTextures(1, ids, 0)
        val texture = ids[0]
        textures += texture
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RGBA16F,
            texWidth,
            texHeight,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_FLOAT,
            buffer,
        )
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        checkGlError("createLensShadingTexture")
        return texture
    }

    private fun createFramebuffer(): Int {
        val ids = IntArray(1)
        GLES30.glGenFramebuffers(1, ids, 0)
        framebuffers += ids[0]
        return ids[0]
    }

    private fun bindTexture(program: Int, name: String, unit: Int, texture: Int) {
        GLES31.glActiveTexture(GLES31.GL_TEXTURE0 + unit)
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, texture)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(program, name), unit)
    }

    private fun bindImage(unit: Int, texture: Int, access: Int, format: Int) {
        GLES31.glBindImageTexture(unit, texture, 0, false, 0, access, format)
    }

    private fun linkGraphicsProgram(vertexSource: String, fragmentSource: String, name: String): Int {
        val vertexShader = compileShader(GLES30.GL_VERTEX_SHADER, vertexSource, "$name vertex")
        val fragmentShader = compileShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource, "$name fragment")
        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)
        val linked = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linked, 0)
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)
        if (linked[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(program)
            GLES30.glDeleteProgram(program)
            throw IllegalStateException("Program $name linking failed: $log")
        }
        programs += program
        return program
    }

    private fun linkComputeProgram(source: String, name: String): Int {
        val shader = compileShader(GLES31.GL_COMPUTE_SHADER, source, "$name compute")
        val program = GLES31.glCreateProgram()
        GLES31.glAttachShader(program, shader)
        GLES31.glLinkProgram(program)
        val linked = IntArray(1)
        GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linked, 0)
        GLES31.glDeleteShader(shader)
        if (linked[0] == 0) {
            val log = GLES31.glGetProgramInfoLog(program)
            GLES31.glDeleteProgram(program)
            throw IllegalStateException("Compute program $name linking failed: $log")
        }
        programs += program
        return program
    }

    private fun compileShader(type: Int, source: String, name: String): Int {
        val shader = GLES31.glCreateShader(type)
        GLES31.glShaderSource(shader, source)
        GLES31.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES31.glGetShaderInfoLog(shader)
            GLES31.glDeleteShader(shader)
            throw IllegalStateException("Shader $name compilation failed: $log")
        }
        return shader
    }

    private fun checkFramebuffer(label: String) {
        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            throw IllegalStateException("$label framebuffer incomplete: 0x${status.toString(16)}")
        }
    }

    private fun checkGlError(label: String) {
        var error = GLES30.glGetError()
        if (error != GLES30.GL_NO_ERROR) {
            val first = error
            while (error != GLES30.GL_NO_ERROR) {
                error = GLES30.glGetError()
            }
            throw IllegalStateException("$label GL error: 0x${first.toString(16)}")
        }
    }

    private fun applyRawRenderState() {
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDisable(GLES30.GL_DITHER)
        GLES30.glDisable(GLES30.GL_SCISSOR_TEST)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
        GLES30.glDisable(GLES30.GL_CULL_FACE)
    }

    private fun groupCount(value: Int): Int = (value + LOCAL_SIZE - 1) / LOCAL_SIZE

    private fun transposeMatrix3x3(matrix: FloatArray): FloatArray {
        return floatArrayOf(
            matrix.getOrElse(0) { 1f }, matrix.getOrElse(3) { 0f }, matrix.getOrElse(6) { 0f },
            matrix.getOrElse(1) { 0f }, matrix.getOrElse(4) { 1f }, matrix.getOrElse(7) { 0f },
            matrix.getOrElse(2) { 0f }, matrix.getOrElse(5) { 0f }, matrix.getOrElse(8) { 1f }
        )
    }

    private fun release() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            if (programs.isNotEmpty()) {
                for (program in programs) {
                    GLES30.glDeleteProgram(program)
                }
            }
            if (textures.isNotEmpty()) {
                GLES30.glDeleteTextures(textures.size, textures.toIntArray(), 0)
            }
            if (framebuffers.isNotEmpty()) {
                GLES30.glDeleteFramebuffers(framebuffers.size, framebuffers.toIntArray(), 0)
            }
            if (buffers.isNotEmpty()) {
                GLES31.glDeleteBuffers(buffers.size, buffers.toIntArray(), 0)
            }
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface)
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext)
            }
            EGL14.eglTerminate(eglDisplay)
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY
        eglContext = EGL14.EGL_NO_CONTEXT
        eglSurface = EGL14.EGL_NO_SURFACE
    }

    companion object {
        private const val TAG = "GlesRawStacker"

        private const val EGL_OPENGL_ES3_BIT_KHR = 0x00000040
        private const val LOCAL_SIZE = 16
        private const val PGTM_STATS_BUFFER_BINDING = 7
        private const val HDR_SHORT_ALIGNMENT_SCORE_BUFFER_BINDING = 8
        private const val DIAGNOSTIC_BUFFER_BINDING = 9
        private const val REGISTRATION_SAMPLE_BUFFER_BINDING = 10
        private const val REGISTRATION_GLOBAL_SCORE_BUFFER_BINDING = 11
        private const val HDR_SHORT_ALIGNMENT_SCORE_STRIDE = 4
        private const val REGISTRATION_GLOBAL_SCORE_STRIDE = 4
        private const val REGISTRATION_SAMPLE_FLOAT_STRIDE = 8
        private const val DIAGNOSTIC_HIST_BINS = 64
        private const val DIAGNOSTIC_FLOW_COUNT_INDEX = 0
        private const val DIAGNOSTIC_FLOW_SUM_INDEX = 1
        private const val DIAGNOSTIC_FLOW_MAX_INDEX = 2
        private const val DIAGNOSTIC_FLOW_OUTLIER_COUNT_INDEX = 3
        private const val DIAGNOSTIC_ROBUST_COUNT_INDEX = 4
        private const val DIAGNOSTIC_ROBUST_SUM_INDEX = 5
        private const val DIAGNOSTIC_ROBUST_MAX_INDEX = 6
        private const val DIAGNOSTIC_TILE_COUNT_INDEX = 7
        private const val DIAGNOSTIC_TILE_SUM_INDEX = 8
        private const val DIAGNOSTIC_TILE_MAX_INDEX = 9
        private const val DIAGNOSTIC_TILE_REJECT_COUNT_INDEX = 10
        private const val DIAGNOSTIC_WEIGHT_COUNT_INDEX = 11
        private const val DIAGNOSTIC_WEIGHT_SUM_INDEX = 12
        private const val DIAGNOSTIC_WEIGHT_MAX_INDEX = 13
        private const val DIAGNOSTIC_LSC_COUNT_INDEX = 14
        private const val DIAGNOSTIC_LSC_SUM_INDEX = 15
        private const val DIAGNOSTIC_LSC_MAX_INDEX = 16
        private const val DIAGNOSTIC_LSC_EDGE_COUNT_INDEX = 17
        private const val DIAGNOSTIC_LSC_EDGE_SUM_INDEX = 18
        private const val DIAGNOSTIC_RESIDUAL_COUNT_INDEX = 19
        private const val DIAGNOSTIC_RESIDUAL_SUM_INDEX = 20
        private const val DIAGNOSTIC_RESIDUAL_MAX_INDEX = 21
        private const val DIAGNOSTIC_NOISE_RESIDUAL_COUNT_INDEX = 22
        private const val DIAGNOSTIC_NOISE_RESIDUAL_SUM_INDEX = 23
        private const val DIAGNOSTIC_NOISE_RESIDUAL_MAX_INDEX = 24
        private const val DIAGNOSTIC_FLOW_RANGE_COUNT_INDEX = 25
        private const val DIAGNOSTIC_FLOW_RANGE_SUM_INDEX = 26
        private const val DIAGNOSTIC_FLOW_RANGE_MAX_INDEX = 27
        private const val DIAGNOSTIC_HIGH_CONFIDENCE_COUNT_INDEX = 28
        private const val DIAGNOSTIC_SR_ALIGNMENT_READY_COUNT_INDEX = 29
        private const val DIAGNOSTIC_SR_DETAIL_READY_COUNT_INDEX = 30
        private const val DIAGNOSTIC_POST_RESIDUAL_COUNT_INDEX = 31
        private const val DIAGNOSTIC_POST_RESIDUAL_SUM_INDEX = 32
        private const val DIAGNOSTIC_POST_RESIDUAL_MAX_INDEX = 33
        private const val DIAGNOSTIC_POST_SMOOTH_COUNT_INDEX = 34
        private const val DIAGNOSTIC_POST_SMOOTH_SUM_INDEX = 35
        private const val DIAGNOSTIC_POST_SMOOTH_MAX_INDEX = 36
        private const val DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_COUNT_INDEX = 37
        private const val DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_SUM_INDEX = 38
        private const val DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_MAX_INDEX = 39
        private const val DIAGNOSTIC_POST_WIENER_COUNT_INDEX = 40
        private const val DIAGNOSTIC_POST_WIENER_SUM_INDEX = 41
        private const val DIAGNOSTIC_POST_WIENER_MAX_INDEX = 42
        private const val DIAGNOSTIC_POST_LSC_COUNT_INDEX = 43
        private const val DIAGNOSTIC_POST_LSC_SUM_INDEX = 44
        private const val DIAGNOSTIC_POST_LSC_MAX_INDEX = 45
        private const val DIAGNOSTIC_POST_LOW_WEIGHT_COUNT_INDEX = 46
        private const val DIAGNOSTIC_POST_LOW_WEIGHT_SUM_INDEX = 47
        private const val DIAGNOSTIC_POST_LOW_WEIGHT_MAX_INDEX = 48
        private const val DIAGNOSTIC_HIST_OFFSET = 56
        private const val DIAGNOSTIC_FLOW_HIST_OFFSET = DIAGNOSTIC_HIST_OFFSET
        private const val DIAGNOSTIC_ROBUST_HIST_OFFSET = DIAGNOSTIC_FLOW_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_TILE_HIST_OFFSET = DIAGNOSTIC_ROBUST_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_WEIGHT_HIST_OFFSET = DIAGNOSTIC_TILE_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_RESIDUAL_HIST_OFFSET = DIAGNOSTIC_WEIGHT_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_NOISE_RESIDUAL_HIST_OFFSET = DIAGNOSTIC_RESIDUAL_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_FLOW_RANGE_HIST_OFFSET = DIAGNOSTIC_NOISE_RESIDUAL_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_POST_RESIDUAL_HIST_OFFSET = DIAGNOSTIC_FLOW_RANGE_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_POST_SMOOTH_HIST_OFFSET = DIAGNOSTIC_POST_RESIDUAL_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_HIST_OFFSET = DIAGNOSTIC_POST_SMOOTH_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_POST_WIENER_HIST_OFFSET = DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_POST_LSC_HIST_OFFSET = DIAGNOSTIC_POST_WIENER_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_POST_LOW_WEIGHT_HIST_OFFSET = DIAGNOSTIC_POST_LSC_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_UINT_COUNT = DIAGNOSTIC_POST_LOW_WEIGHT_HIST_OFFSET + DIAGNOSTIC_HIST_BINS
        private const val DIAGNOSTIC_UNIT_QUANTIZATION = 1000f
        private const val DIAGNOSTIC_FLOW_QUANTIZATION = 10f
        private const val DIAGNOSTIC_WEIGHT_QUANTIZATION = 100f
        private const val DIAGNOSTIC_LSC_QUANTIZATION = 1000f
        private const val DIAGNOSTIC_RESIDUAL_QUANTIZATION = 10000f
        private const val DIAGNOSTIC_NOISE_RESIDUAL_QUANTIZATION = 100f
        private const val DIAGNOSTIC_POST_RESIDUAL_QUANTIZATION = 100f

        private val FULLSCREEN_VERTEX_SHADER = """
            #version 300 es
            precision highp float;
            out vec2 vTexCoord;
            void main() {
                vec2 positions[3] = vec2[3](
                    vec2(-1.0, -1.0),
                    vec2( 3.0, -1.0),
                    vec2(-1.0,  3.0)
                );
                vec2 texCoords[3] = vec2[3](
                    vec2(0.0, 0.0),
                    vec2(2.0, 0.0),
                    vec2(0.0, 2.0)
                );
                gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
                vTexCoord = texCoords[gl_VertexID];
            }
        """.trimIndent()

        private val RAW_COMMON = """
            precision highp float;
            precision highp int;
            precision highp usampler2D;

            int baseCfaPattern(int cfaPattern) {
                if (cfaPattern >= 8) return cfaPattern - 8;
                if (cfaPattern >= 4) return cfaPattern - 4;
                return cfaPattern;
            }

            int cfaBlockSize(int cfaPattern) {
                if (cfaPattern >= 8) return 4;
                if (cfaPattern >= 4) return 2;
                return 1;
            }

            int cfaPeriod(int cfaPattern) {
                return cfaBlockSize(cfaPattern) * 2;
            }

            int bayerIndexForPhase(int pattern, int phase) {
                if (pattern == 0) return phase;
                if (pattern == 1) {
                    if (phase == 0) return 1;
                    if (phase == 1) return 0;
                    if (phase == 2) return 3;
                    return 2;
                }
                if (pattern == 2) {
                    if (phase == 0) return 2;
                    if (phase == 1) return 3;
                    if (phase == 2) return 0;
                    return 1;
                }
                if (phase == 0) return 3;
                if (phase == 1) return 2;
                if (phase == 2) return 1;
                return 0;
            }

            int bayerIndexAt(int cfaPattern, ivec2 p) {
                int blockSize = cfaBlockSize(cfaPattern);
                int phase = ((p.y / blockSize) & 1) * 2 + ((p.x / blockSize) & 1);
                return bayerIndexForPhase(baseCfaPattern(cfaPattern), phase);
            }

            int lensShadingChannelAt(int cfaPattern, ivec2 p) {
                int channel = bayerIndexAt(cfaPattern, p);
                if (channel == 0 || channel == 3) return channel;
                return ((p.y & 1) == 0) ? 1 : 2;
            }

        """.trimIndent()

        private val RAW_PROXY_COMPUTE_SHADER = """
            #version 310 es
            $RAW_COMMON
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform highp usampler2D uRaw;
            layout(rgba16f, binding = 1) writeonly uniform highp image2D uProxy;
            uniform ivec2 uProxySize;
            uniform int uCfaPattern;
            uniform float uBlackLevel[4];
            uniform float uWhiteLevel;
            uniform float uExposureScale;

            float rawNormAt(ivec2 p) {
                p = clamp(p, ivec2(0), uProxySize * 2 - ivec2(1));
                int b = bayerIndexAt(uCfaPattern, p);
                float raw = float(texelFetch(uRaw, p, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[b], 1.0);
                return clamp((raw - uBlackLevel[b]) / range, 0.0, 1.0);
            }

            float greenBlock(ivec2 planeCoord) {
                ivec2 s = clamp(planeCoord * 2, ivec2(0), uProxySize * 2 - ivec2(2));
                if (uCfaPattern >= 4) {
                    float p00 = rawNormAt(s);
                    float p10 = rawNormAt(s + ivec2(1, 0));
                    float p01 = rawNormAt(s + ivec2(0, 1));
                    float p11 = rawNormAt(s + ivec2(1, 1));
                    return 0.25 * (p00 + p10 + p01 + p11);
                }
                float g1;
                float g2;
                if (uCfaPattern == 0 || uCfaPattern == 3) {
                    g1 = rawNormAt(s + ivec2(1, 0));
                    g2 = rawNormAt(s + ivec2(0, 1));
                } else {
                    g1 = rawNormAt(s);
                    g2 = rawNormAt(s + ivec2(1, 1));
                }
                return 0.5 * (g1 + g2);
            }

            float tileMaxRaw(ivec2 planeCoord) {
                ivec2 s = clamp(planeCoord * 2, ivec2(0), uProxySize * 2 - ivec2(2));
                float m = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        m = max(m, rawNormAt(s + ivec2(x, y)));
                    }
                }
                return m;
            }

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uProxySize.x || p.y >= uProxySize.y) return;
                float centerRaw = greenBlock(p);
                float center = clamp(centerRaw * uExposureScale, 0.0, 1.0);
                float sum = 0.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        ivec2 q = clamp(p + ivec2(x, y), ivec2(0), uProxySize - ivec2(1));
                        sum += clamp(greenBlock(q) * uExposureScale, 0.0, 1.0);
                    }
                }
                float mean = sum / 9.0;
                float shadowValid = smoothstep(0.004, 0.035, centerRaw);
                float clipValid = 1.0 - smoothstep(0.90, 0.995, tileMaxRaw(p));
                float validity = clamp(shadowValid * clipValid, 0.0, 1.0);
                imageStore(uProxy, p, vec4(clamp(center + 0.35 * (center - mean), 0.0, 1.0), validity, 0.0, 1.0));
            }
        """.trimIndent()

        private val DOWNSAMPLE_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uInput;
            uniform ivec2 uInputSize;
            out vec4 fragColor;

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                ivec2 src = p * 2;
                vec2 sum = vec2(0.0);
                for (int y = 0; y < 2; ++y) {
                    for (int x = 0; x < 2; ++x) {
                        ivec2 q = clamp(src + ivec2(x, y), ivec2(0), uInputSize - ivec2(1));
                        sum += texelFetch(uInput, q, 0).rg;
                    }
                }
                fragColor = vec4(sum * 0.25, 0.0, 1.0);
            }
        """.trimIndent()

        private val ALIGN_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uReference;
            uniform sampler2D uCurrent;
            uniform ivec2 uLevelSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            uniform int uAlignWindowSize;
            uniform int uLevelScale;
            uniform int uSearchRadius;
            uniform int uSampleStep;
            uniform float uCoveragePenalty;
            uniform float uShiftPenalty;
            out vec4 fragColor;

            vec2 readProxy(sampler2D tex, ivec2 p) {
                p = clamp(p, ivec2(0), uLevelSize - ivec2(1));
                return texelFetch(tex, p, 0).rg;
            }

            void main() {
                ivec2 tile = ivec2(gl_FragCoord.xy);
                ivec2 fullCenter = tile * uTileSize;
                ivec2 levelCenter = fullCenter / uLevelScale;
                int levelTile = max(4, uAlignWindowSize / uLevelScale);
                ivec2 levelStart = levelCenter - ivec2(levelTile / 2);
                float bestSad = 1e20;
                ivec2 bestShift = ivec2(0);

                for (int dy = -uSearchRadius; dy <= uSearchRadius; ++dy) {
                    for (int dx = -uSearchRadius; dx <= uSearchRadius; ++dx) {
                        float sad = 0.0;
                        float count = 0.0;
                        float sampleCount = 0.0;
                        for (int sy = 1; sy < levelTile - 1; sy += uSampleStep) {
                            for (int sx = 1; sx < levelTile - 1; sx += uSampleStep) {
                                ivec2 rp = levelStart + ivec2(sx, sy);
                                vec2 rv = readProxy(uReference, rp);
                                vec2 cv = readProxy(uCurrent, rp + ivec2(dx, dy));
                                float w = min(rv.g, cv.g);
                                sad += abs(rv.r - cv.r) * w;
                                count += w;
                                sampleCount += 1.0;
                            }
                        }
                        float coverage = count / max(sampleCount, 1.0);
                        sad = sad / max(count, 1e-4) +
                            uCoveragePenalty * (1.0 - clamp(coverage, 0.0, 1.0)) +
                            uShiftPenalty * float(dx * dx + dy * dy);
                        if (sad < bestSad) {
                            bestSad = sad;
                            bestShift = ivec2(dx, dy);
                        }
                    }
                }
                fragColor = vec4(vec2(bestShift) * float(uLevelScale), bestSad, 1.0);
            }
        """.trimIndent()

        private val HDR_SHORT_GLOBAL_ALIGN_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 8, local_size_y = 8) in;
            uniform sampler2D uReference;
            uniform sampler2D uCurrent;
            uniform ivec2 uLevelSize;
            uniform int uLevelScale;
            uniform int uSearchRadius;
            uniform int uSampleStep;
            uniform int uSampleBorder;
            uniform float uCoveragePenalty;
            uniform float uShiftPenalty;
            layout(std430, binding = $HDR_SHORT_ALIGNMENT_SCORE_BUFFER_BINDING) buffer HdrShortAlignmentScores {
                float scores[];
            };

            const int LOCAL_COUNT = 64;
            const int SCORE_STRIDE = $HDR_SHORT_ALIGNMENT_SCORE_STRIDE;
            shared float sadParts[LOCAL_COUNT];
            shared float weightParts[LOCAL_COUNT];
            shared float sampleParts[LOCAL_COUNT];

            vec2 readProxy(sampler2D tex, ivec2 p) {
                p = clamp(p, ivec2(0), uLevelSize - ivec2(1));
                return texelFetch(tex, p, 0).rg;
            }

            bool insideLevel(ivec2 p) {
                return p.x >= 0 && p.y >= 0 && p.x < uLevelSize.x && p.y < uLevelSize.y;
            }

            float detailAt(ivec2 p) {
                float c = readProxy(uReference, p).r;
                float gx = abs(readProxy(uReference, p + ivec2(1, 0)).r - readProxy(uReference, p - ivec2(1, 0)).r);
                float gy = abs(readProxy(uReference, p + ivec2(0, 1)).r - readProxy(uReference, p - ivec2(0, 1)).r);
                float lap = abs(4.0 * c -
                    readProxy(uReference, p + ivec2(1, 0)).r -
                    readProxy(uReference, p - ivec2(1, 0)).r -
                    readProxy(uReference, p + ivec2(0, 1)).r -
                    readProxy(uReference, p - ivec2(0, 1)).r);
                return gx + gy + 0.5 * lap;
            }

            void main() {
                ivec2 candidate = ivec2(gl_WorkGroupID.xy);
                int scoreSide = uSearchRadius * 2 + 1;
                if (candidate.x >= scoreSide || candidate.y >= scoreSide) return;

                ivec2 localId = ivec2(gl_LocalInvocationID.xy);
                int localIndex = localId.y * 8 + localId.x;
                ivec2 shift = candidate - ivec2(uSearchRadius);
                int sampleWidth = max(0, (uLevelSize.x - 2 * uSampleBorder + uSampleStep - 1) / uSampleStep);
                int sampleHeight = max(0, (uLevelSize.y - 2 * uSampleBorder + uSampleStep - 1) / uSampleStep);
                int totalSamples = sampleWidth * sampleHeight;

                float sad = 0.0;
                float weight = 0.0;
                float sampleCount = 0.0;
                for (int sampleIndex = localIndex; sampleIndex < totalSamples; sampleIndex += LOCAL_COUNT) {
                    int sx = sampleIndex - (sampleIndex / sampleWidth) * sampleWidth;
                    int sy = sampleIndex / sampleWidth;
                    ivec2 rp = ivec2(
                        uSampleBorder + sx * uSampleStep,
                        uSampleBorder + sy * uSampleStep
                    );
                    ivec2 cp = rp + shift;
                    sampleCount += 1.0;
                    if (!insideLevel(cp)) {
                        continue;
                    }

                    vec2 rv = readProxy(uReference, rp);
                    vec2 cv = readProxy(uCurrent, cp);
                    float detail = clamp(detailAt(rp) * 18.0, 0.08, 1.0);
                    float w = min(rv.g, cv.g) * detail;
                    sad += abs(rv.r - cv.r) * w;
                    weight += w;
                }

                sadParts[localIndex] = sad;
                weightParts[localIndex] = weight;
                sampleParts[localIndex] = sampleCount;
                memoryBarrierShared();
                barrier();

                if (localIndex != 0) {
                    return;
                }

                float totalSad = 0.0;
                float totalWeight = 0.0;
                float totalSampleCount = 0.0;
                for (int i = 0; i < LOCAL_COUNT; ++i) {
                    totalSad += sadParts[i];
                    totalWeight += weightParts[i];
                    totalSampleCount += sampleParts[i];
                }
                float coverage = totalWeight / max(totalSampleCount, 1.0);
                float shiftPenalty = uShiftPenalty * float(shift.x * shift.x + shift.y * shift.y);
                float score = totalSad / max(totalWeight, 1e-4) +
                    uCoveragePenalty * (1.0 - clamp(coverage, 0.0, 1.0)) +
                    shiftPenalty;
                int scoreIndex = candidate.y * scoreSide + candidate.x;
                int offset = scoreIndex * SCORE_STRIDE;
                scores[offset + 0] = float(shift.x * uLevelScale);
                scores[offset + 1] = float(shift.y * uLevelScale);
                scores[offset + 2] = score;
                scores[offset + 3] = coverage;
            }
        """.trimIndent()

        private val REGISTRATION_GLOBAL_ALIGN_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 8, local_size_y = 8) in;
            uniform sampler2D uReference;
            uniform sampler2D uCurrent;
            uniform ivec2 uLevelSize;
            uniform int uLevelScale;
            uniform int uSearchRadius;
            uniform vec2 uCenterShift;
            uniform int uSampleStep;
            uniform int uSampleBorder;
            uniform float uCoveragePenalty;
            uniform float uShiftPenalty;
            layout(std430, binding = $REGISTRATION_GLOBAL_SCORE_BUFFER_BINDING) buffer RegistrationGlobalScores {
                float scores[];
            };

            const int LOCAL_COUNT = 64;
            const int SCORE_STRIDE = $REGISTRATION_GLOBAL_SCORE_STRIDE;
            shared float sadParts[LOCAL_COUNT];
            shared float weightParts[LOCAL_COUNT];
            shared float validityParts[LOCAL_COUNT];
            shared float sampleParts[LOCAL_COUNT];

            vec2 readProxy(sampler2D tex, ivec2 p) {
                p = clamp(p, ivec2(0), uLevelSize - ivec2(1));
                return texelFetch(tex, p, 0).rg;
            }

            vec2 sampleProxy(sampler2D tex, vec2 p) {
                vec2 uv = (clamp(p, vec2(0.0), vec2(uLevelSize - ivec2(1))) + vec2(0.5)) / vec2(uLevelSize);
                return texture(tex, uv).rg;
            }

            bool insideLevel(vec2 p) {
                return p.x >= 0.0 && p.y >= 0.0 &&
                    p.x <= float(uLevelSize.x - 1) &&
                    p.y <= float(uLevelSize.y - 1);
            }

            float detailAt(ivec2 p) {
                float c = readProxy(uReference, p).r;
                float gx = abs(readProxy(uReference, p + ivec2(1, 0)).r - readProxy(uReference, p - ivec2(1, 0)).r);
                float gy = abs(readProxy(uReference, p + ivec2(0, 1)).r - readProxy(uReference, p - ivec2(0, 1)).r);
                float lap = abs(4.0 * c -
                    readProxy(uReference, p + ivec2(1, 0)).r -
                    readProxy(uReference, p - ivec2(1, 0)).r -
                    readProxy(uReference, p + ivec2(0, 1)).r -
                    readProxy(uReference, p - ivec2(0, 1)).r);
                return gx + gy + 0.5 * lap;
            }

            void main() {
                ivec2 candidate = ivec2(gl_WorkGroupID.xy);
                int scoreSide = uSearchRadius * 2 + 1;
                if (candidate.x >= scoreSide || candidate.y >= scoreSide) return;

                ivec2 localId = ivec2(gl_LocalInvocationID.xy);
                int localIndex = localId.y * 8 + localId.x;
                vec2 shift = uCenterShift + vec2(candidate - ivec2(uSearchRadius));
                int sampleWidth = max(0, (uLevelSize.x - 2 * uSampleBorder + uSampleStep - 1) / uSampleStep);
                int sampleHeight = max(0, (uLevelSize.y - 2 * uSampleBorder + uSampleStep - 1) / uSampleStep);
                int totalSamples = sampleWidth * sampleHeight;

                float sad = 0.0;
                float weight = 0.0;
                float validity = 0.0;
                float sampleCount = 0.0;
                for (int sampleIndex = localIndex; sampleIndex < totalSamples; sampleIndex += LOCAL_COUNT) {
                    int sx = sampleIndex - (sampleIndex / sampleWidth) * sampleWidth;
                    int sy = sampleIndex / sampleWidth;
                    ivec2 rp = ivec2(
                        uSampleBorder + sx * uSampleStep,
                        uSampleBorder + sy * uSampleStep
                    );
                    vec2 cp = vec2(rp) + shift;
                    sampleCount += 1.0;
                    if (!insideLevel(cp)) {
                        continue;
                    }

                    vec2 rv = readProxy(uReference, rp);
                    vec2 cv = sampleProxy(uCurrent, cp);
                    float valid = min(rv.g, cv.g);
                    float detail = clamp(detailAt(rp) * 18.0, 0.08, 1.0);
                    float w = valid * detail;
                    sad += abs(rv.r - cv.r) * w;
                    weight += w;
                    validity += valid;
                }

                sadParts[localIndex] = sad;
                weightParts[localIndex] = weight;
                validityParts[localIndex] = validity;
                sampleParts[localIndex] = sampleCount;
                memoryBarrierShared();
                barrier();

                if (localIndex != 0) {
                    return;
                }

                float totalSad = 0.0;
                float totalWeight = 0.0;
                float totalValidity = 0.0;
                float totalSampleCount = 0.0;
                for (int i = 0; i < LOCAL_COUNT; ++i) {
                    totalSad += sadParts[i];
                    totalWeight += weightParts[i];
                    totalValidity += validityParts[i];
                    totalSampleCount += sampleParts[i];
                }
                float coverage = totalValidity / max(totalSampleCount, 1.0);
                float shiftPenalty = uShiftPenalty * dot(shift, shift);
                float score = totalSad / max(totalWeight, 1e-4) +
                    uCoveragePenalty * (1.0 - clamp(coverage, 0.0, 1.0)) +
                    shiftPenalty;
                int scoreIndex = candidate.y * scoreSide + candidate.x;
                int offset = scoreIndex * SCORE_STRIDE;
                scores[offset + 0] = shift.x * float(uLevelScale);
                scores[offset + 1] = shift.y * float(uLevelScale);
                scores[offset + 2] = score;
                scores[offset + 3] = coverage;
            }
        """.trimIndent()

        private val LK_REFINE_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform sampler2D uReference;
            uniform sampler2D uCurrent;
            uniform sampler2D uInputFlow;
            layout(rgba16f, binding = 3) writeonly uniform highp image2D uOutputFlow;
            uniform ivec2 uPlaneSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;

            float sampleProxy(sampler2D tex, vec2 p) {
                vec2 uv = (clamp(p, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(tex, uv).r;
            }

            void main() {
                ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
                if (tile.x >= uGridSize.x || tile.y >= uGridSize.y) return;
                vec2 flow = texelFetch(uInputFlow, tile, 0).rg;
                vec2 tileCenter = vec2(tile * uTileSize + ivec2(uTileSize / 2));
                float sIxIx = 0.0;
                float sIyIy = 0.0;
                float sIxIy = 0.0;
                float sIxIt = 0.0;
                float sIyIt = 0.0;
                const int windowRadius = 3;
                const float sigma2 = 10.0;
                for (int oy = -windowRadius; oy <= windowRadius; ++oy) {
                    for (int ox = -windowRadius; ox <= windowRadius; ++ox) {
                        vec2 basePoint = tileCenter + vec2(float(ox), float(oy));
                        if (basePoint.x < 1.0 || basePoint.y < 1.0 ||
                            basePoint.x > float(uPlaneSize.x - 2) ||
                            basePoint.y > float(uPlaneSize.y - 2)) {
                            continue;
                        }
                        float t = sampleProxy(uReference, basePoint);
                        float i = sampleProxy(uCurrent, basePoint + flow);
                        float tx = 0.5 * (sampleProxy(uReference, basePoint + vec2(1.0, 0.0)) -
                                          sampleProxy(uReference, basePoint - vec2(1.0, 0.0)));
                        float ty = 0.5 * (sampleProxy(uReference, basePoint + vec2(0.0, 1.0)) -
                                          sampleProxy(uReference, basePoint - vec2(0.0, 1.0)));
                        float detailGate = clamp(max(abs(tx), abs(ty)) * 18.0, 0.08, 1.0);
                        float spatialW = exp(-0.5 * float(ox * ox + oy * oy) / sigma2);
                        float w = spatialW * detailGate;
                        float it = i - t;
                        sIxIx += w * tx * tx;
                        sIyIy += w * ty * ty;
                        sIxIy += w * tx * ty;
                        sIxIt += w * tx * it;
                        sIyIt += w * ty * it;
                    }
                }
                float trace = sIxIx + sIyIy;
                float lambda = max(1e-4, 0.015 * trace + 5e-4);
                sIxIx += lambda;
                sIyIy += lambda;
                float det = sIxIx * sIyIy - sIxIy * sIxIy;
                vec2 delta = vec2(0.0);
                if (det > 1e-7) {
                    delta.x = (sIyIy * sIxIt - sIxIy * sIyIt) / det;
                    delta.y = (sIxIx * sIyIt - sIxIy * sIxIt) / det;
                    delta = -delta;
                }
                float len = length(delta);
                if (isnan(len) || isinf(len)) {
                    delta = vec2(0.0);
                } else if (len > 1.25) {
                    delta *= 1.25 / len;
                }
                vec2 updated = flow + delta;
                if (isnan(updated.x) || isnan(updated.y) || isinf(updated.x) || isinf(updated.y)) {
                    updated = flow;
                }
                imageStore(uOutputFlow, tile, vec4(updated, 0.0, 1.0));
            }
        """.trimIndent()

        private val SMOOTH_FLOW_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uInputFlow;
            uniform ivec2 uGridSize;
            uniform float uOutlierThreshold;
            uniform float uOutlierWeight;
            out vec4 fragColor;

            vec2 readFlow(ivec2 p) {
                p = clamp(p, ivec2(0), uGridSize - ivec2(1));
                return texelFetch(uInputFlow, p, 0).rg;
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                vec2 center = readFlow(p);
                vec2 sum = center * 4.0;
                float weight = 4.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        if (x == 0 && y == 0) continue;
                        vec2 f = readFlow(p + ivec2(x, y));
                        float d = length(f - center);
                        float w = d > uOutlierThreshold ? uOutlierWeight : 1.0;
                        sum += f * w;
                        weight += w;
                    }
                }
                fragColor = vec4(sum / weight, 0.0, 1.0);
            }
        """.trimIndent()

        private val STRUCTURE_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform sampler2D uProxy;
            layout(rgba16f, binding = 1) writeonly uniform highp image2D uKernel;
            uniform ivec2 uProxySize;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            uniform float uStructureFlatnessSnrLow;
            uniform float uStructureFlatnessSnrHigh;
            uniform float uStructureKernelDetail;
            uniform float uStructureKernelDenoise;
            uniform float uStructureKernelShrink;
            uniform float uStructureKernelStretch;
            uniform float uStructureAnisotropyThreshold;

            float readProxy(ivec2 p) {
                p = clamp(p, ivec2(0), uProxySize - ivec2(1));
                return texelFetch(uProxy, p, 0).r;
            }

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uProxySize.x || p.y >= uProxySize.y) return;
                float sIxIx = 0.0;
                float sIyIy = 0.0;
                float sIxIy = 0.0;
                float sSignal = 0.0;
                for (int y = -2; y <= 2; ++y) {
                    for (int x = -2; x <= 2; ++x) {
                        ivec2 q = p + ivec2(x, y);
                        float center = readProxy(q);
                        float ix = 0.5 * (readProxy(q + ivec2(1, 0)) - readProxy(q - ivec2(1, 0)));
                        float iy = 0.5 * (readProxy(q + ivec2(0, 1)) - readProxy(q - ivec2(0, 1)));
                        sIxIx += ix * ix;
                        sIyIy += iy * iy;
                        sIxIy += ix * iy;
                        sSignal += center;
                    }
                }
                float jxx = sIxIx / 25.0;
                float jyy = sIyIy / 25.0;
                float jxy = sIxIy / 25.0;
                float signalMean = sSignal / 25.0;
                float trace = jxx + jyy;
                float det = jxx * jyy - jxy * jxy;
                float discriminant = sqrt(max(trace * trace * 0.25 - det, 0.0));
                float lambda1 = trace * 0.5 + discriminant;
                float lambda2 = trace * 0.5 - discriminant;
                float noiseSignal = clamp(signalMean, 0.10, 0.75);
                float noiseVar = uNoiseAlpha * noiseSignal + max(uNoiseBeta, 1e-10);
                float snr = lambda1 / max(2.0 * noiseVar * 9.0, 1e-12);
                float flatness = 1.0 - smoothstep(uStructureFlatnessSnrLow, uStructureFlatnessSnrHigh, snr);
                float anisotropy = 1.0 + sqrt(max(lambda1 - lambda2, 0.0) / max(lambda1 + lambda2, 1e-7));
                float kDetail = uStructureKernelDetail;
                float kDenoise = uStructureKernelDenoise;
                float kShrink = uStructureKernelShrink;
                float kStretch = uStructureKernelStretch;
                float k1Base = anisotropy > uStructureAnisotropyThreshold ? 1.0 / kShrink : 1.0;
                float k2Base = anisotropy > uStructureAnisotropyThreshold ? kStretch : 1.0;
                float preK1 = kDetail * mix(k1Base, kDenoise, flatness);
                float preK2 = kDetail * mix(k2Base, kDenoise, flatness);
                float k1 = 1.0 / max(preK1 * preK1, 1e-7);
                float k2 = 1.0 / max(preK2 * preK2, 1e-7);
                float diff = jxx - jyy;
                float hyp = sqrt(diff * diff + 4.0 * jxy * jxy);
                float cos2t = hyp > 1e-9 ? diff / hyp : 1.0;
                float sin2t = hyp > 1e-9 ? 2.0 * jxy / hyp : 0.0;
                vec4 result = vec4(k1, k2, cos2t, sin2t);
                if (any(isnan(result)) || any(isinf(result))) {
                    result = vec4(1.0, 1.0, 1.0, 0.0);
                }
                imageStore(uKernel, p, result);
            }
        """.trimIndent()

        private val ROBUSTNESS_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform sampler2D uReference;
            uniform sampler2D uCurrent;
            uniform sampler2D uFlowGrid;
            layout(r32f, binding = 3) writeonly uniform highp image2D uRobustness;
            uniform ivec2 uPlaneSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            uniform float uRobustNoiseFloorSpatialScale;
            uniform float uRobustNoiseFloorEdgeScale;
            uniform float uRobustTauBase;
            uniform float uRobustTauEdge;
            uniform float uRobustResidualPower;
            uniform float uRobustFlowPenaltyStartPx;
            uniform float uRobustFlowPenaltyDecay;
            uniform float uRobustFlowRangePenaltyStartPx;
            uniform float uRobustFlowRangePenaltyDecay;
            uniform float uRobustMinMixFlat;
            uniform float uRobustMinMixEdge;
            uniform float uRobustCenterMixFlat;
            uniform float uRobustCenterMixEdge;

            float refProxy(ivec2 p) {
                p = clamp(p, ivec2(0), uPlaneSize - ivec2(1));
                return texelFetch(uReference, p, 0).r;
            }

            float curProxy(vec2 p) {
                vec2 uv = (clamp(p, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(uCurrent, uv).r;
            }

            vec2 flowAt(vec2 planePos) {
                vec2 grid = planePos / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uFlowGrid, clamp(uv, vec2(0.0), vec2(1.0))).rg;
            }

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uPlaneSize.x || p.y >= uPlaneSize.y) return;
                vec2 flow = flowAt(vec2(p));
                vec2 curCenter = vec2(p) + flow;
                if (curCenter.x < 1.0 || curCenter.y < 1.0 ||
                    curCenter.x > float(uPlaneSize.x - 2) ||
                    curCenter.y > float(uPlaneSize.y - 2)) {
                    imageStore(uRobustness, p, vec4(0.0));
                    return;
                }

                float greenCenter = refProxy(p);
                float sumG = 0.0;
                float sumG2 = 0.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        float g = refProxy(p + ivec2(x, y));
                        sumG += g;
                        sumG2 += g * g;
                    }
                }
                float meanG = sumG / 9.0;
                float sigma2Spatial = max(sumG2 / 9.0 - meanG * meanG, 0.0);
                float sigma2Noise = max(uNoiseAlpha * greenCenter + uNoiseBeta, 1e-10);
                float sigma2 = max(sigma2Spatial, sigma2Noise);
                float gradX = 0.5 * (refProxy(p + ivec2(1, 0)) - refProxy(p - ivec2(1, 0)));
                float gradY = 0.5 * (refProxy(p + ivec2(0, 1)) - refProxy(p - ivec2(0, 1)));
                float edgeStrength = sqrt(max((gradX * gradX + gradY * gradY) / max(sigma2, 1e-8), 0.0));
                float edgeRelax = smoothstep(1.2, 5.0, edgeStrength);

                vec2 gridCoord = vec2(p) / float(uTileSize) - 0.5;
                ivec2 gCenter = clamp(ivec2(round(gridCoord)), ivec2(0), uGridSize - ivec2(1));
                vec2 fMin = flow;
                vec2 fMax = flow;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        ivec2 gp = clamp(gCenter + ivec2(x, y), ivec2(0), uGridSize - ivec2(1));
                        vec2 f = texelFetch(uFlowGrid, gp, 0).rg;
                        fMin = min(fMin, f);
                        fMax = max(fMax, f);
                    }
                }
                float flowMag = length(flow);
                float flowPenalty = flowMag > uRobustFlowPenaltyStartPx ?
                    exp(-uRobustFlowPenaltyDecay * (flowMag - uRobustFlowPenaltyStartPx)) : 1.0;
                float flowRange = length(fMax - fMin);
                float consistencyPenalty = flowRange > uRobustFlowRangePenaltyStartPx ?
                    exp(-uRobustFlowRangePenaltyDecay * (flowRange - uRobustFlowRangePenaltyStartPx)) : 1.0;
                float globalPenalty = flowPenalty * consistencyPenalty;

                float minR = 1.0;
                float sumR = 0.0;
                float centerR = 1.0;
                float weightSum = 0.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        ivec2 rp = p + ivec2(x, y);
                        float r = refProxy(rp);
                        float c = curProxy(vec2(rp) + flow);
                        float diff = r - c;
                        float d2 = diff * diff;
                        float noiseFloor = mix(
                            uRobustNoiseFloorSpatialScale * sigma2,
                            uRobustNoiseFloorEdgeScale * sigma2Noise,
                            edgeRelax
                        );
                        float den = mix(sigma2, sigma2Noise, edgeRelax);
                        float residual = max(0.0, d2 - noiseFloor) / max(den, 1e-10);
                        float tau = uRobustTauBase + uRobustTauEdge * edgeRelax;
                        float robust = exp(-0.5 * pow(residual / tau, uRobustResidualPower)) * globalPenalty;
                        float w = (x == 0 && y == 0) ? 2.0 : 1.0;
                        sumR += robust * w;
                        weightSum += w;
                        minR = min(minR, robust);
                        if (x == 0 && y == 0) centerR = robust;
                    }
                }
                float avgR = sumR / max(weightSum, 1.0);
                float minMix = mix(uRobustMinMixFlat, uRobustMinMixEdge, edgeRelax);
                float centerMix = mix(uRobustCenterMixFlat, uRobustCenterMixEdge, edgeRelax);
                float outR = clamp(minMix * minR + centerMix * centerR + (1.0 - minMix - centerMix) * avgR, 0.0, 1.0);
                imageStore(uRobustness, p, vec4(outR));
            }
        """.trimIndent()

        private val TILE_MASK_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uReference;
            uniform sampler2D uRobustness;
            uniform ivec2 uPlaneSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            uniform float uTileRobustCenter;
            uniform float uTileRobustWidth;
            uniform float uTileWeakThreshold;
            uniform float uTileWeakStart;
            uniform float uTileWeakRange;
            uniform float uTileDetailMid;
            uniform float uTileDetailHigh;
            uniform float uTileDetailBoostLow;
            uniform float uTileDetailBoostMid;
            uniform float uTileDetailBoostHigh;
            uniform float uTileMaskMinMidDetail;
            uniform float uTileMaskMinHighDetail;
            out float fragColor;

            float readRef(ivec2 p) {
                p = clamp(p, ivec2(0), uPlaneSize - ivec2(1));
                return texelFetch(uReference, p, 0).r;
            }

            void main() {
                ivec2 tile = ivec2(gl_FragCoord.xy);
                ivec2 start = tile * uTileSize;
                float robustSum = 0.0;
                float weakCount = 0.0;
                float detailSum = 0.0;
                float count = 0.0;
                for (int y = 0; y < uTileSize; y += 4) {
                    for (int x = 0; x < uTileSize; x += 4) {
                        ivec2 p = start + ivec2(x, y);
                        if (p.x >= uPlaneSize.x || p.y >= uPlaneSize.y) continue;
                        float r = texelFetch(uRobustness, p, 0).r;
                        float c = readRef(p);
                        float detail = abs(readRef(p + ivec2(1, 0)) - readRef(p - ivec2(1, 0))) +
                            abs(readRef(p + ivec2(0, 1)) - readRef(p - ivec2(0, 1))) +
                            0.5 * abs(4.0 * c - readRef(p + ivec2(1, 0)) - readRef(p - ivec2(1, 0)) -
                            readRef(p + ivec2(0, 1)) - readRef(p - ivec2(0, 1)));
                        robustSum += r;
                        weakCount += r < uTileWeakThreshold ? 1.0 : 0.0;
                        detailSum += detail;
                        count += 1.0;
                    }
                }
                float meanR = robustSum / max(count, 1.0);
                float weak = weakCount / max(count, 1.0);
                float detail = detailSum / max(count, 1.0);
                float robustNorm = clamp((meanR - uTileRobustCenter) / uTileRobustWidth, 0.0, 1.0);
                float weakPenalty = clamp(1.0 - max(0.0, weak - uTileWeakStart) / uTileWeakRange, 0.0, 1.0);
                float detailBoost = detail > uTileDetailHigh ?
                    uTileDetailBoostHigh : (detail > uTileDetailMid ? uTileDetailBoostMid : uTileDetailBoostLow);
                float mask = clamp((0.60 * robustNorm + 0.40 * weakPenalty) * (0.55 + 0.45 * detailBoost), 0.0, 1.0);
                if (detail > uTileDetailHigh) {
                    mask = max(mask, uTileMaskMinHighDetail * robustNorm);
                } else if (detail > uTileDetailMid) {
                    mask = max(mask, uTileMaskMinMidDetail * robustNorm);
                }
                fragColor = mask;
            }
        """.trimIndent()

        private val REGISTRATION_SAMPLE_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform sampler2D uFlowGrid;
            uniform sampler2D uRobustness;
            uniform sampler2D uTileMask;
            uniform sampler2D uReference;
            uniform sampler2D uCurrent;
            uniform ivec2 uImageSize;
            uniform ivec2 uPlaneSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            layout(std430, binding = $REGISTRATION_SAMPLE_BUFFER_BINDING) buffer RegistrationSamples {
                float samples[];
            };

            const int SAMPLE_STRIDE = $REGISTRATION_SAMPLE_FLOAT_STRIDE;

            float refAt(ivec2 p) {
                p = clamp(p, ivec2(0), uPlaneSize - ivec2(1));
                return texelFetch(uReference, p, 0).r;
            }

            float curAt(vec2 p) {
                vec2 uv = (clamp(p, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(uCurrent, uv).r;
            }

            float mapAt(sampler2D tex, vec2 planePos) {
                vec2 uv = (clamp(planePos, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(tex, uv).r;
            }

            float detailAt(ivec2 p) {
                float c = refAt(p);
                float gx = abs(refAt(p + ivec2(1, 0)) - refAt(p - ivec2(1, 0)));
                float gy = abs(refAt(p + ivec2(0, 1)) - refAt(p - ivec2(0, 1)));
                float lap = abs(4.0 * c -
                    refAt(p + ivec2(1, 0)) -
                    refAt(p - ivec2(1, 0)) -
                    refAt(p + ivec2(0, 1)) -
                    refAt(p - ivec2(0, 1)));
                return gx + gy + 0.5 * lap;
            }

            void writeSample(
                ivec2 tile,
                vec2 referenceRaw,
                vec2 targetRaw,
                float robustness,
                float tileMask,
                float residual,
                float detail
            ) {
                int offset = (tile.y * uGridSize.x + tile.x) * SAMPLE_STRIDE;
                samples[offset + 0] = referenceRaw.x;
                samples[offset + 1] = referenceRaw.y;
                samples[offset + 2] = targetRaw.x;
                samples[offset + 3] = targetRaw.y;
                samples[offset + 4] = robustness;
                samples[offset + 5] = tileMask;
                samples[offset + 6] = residual;
                samples[offset + 7] = detail;
            }

            void main() {
                ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
                if (tile.x >= uGridSize.x || tile.y >= uGridSize.y) return;
                vec2 planePos = min(
                    vec2(tile * uTileSize + ivec2(uTileSize / 2)),
                    vec2(uPlaneSize - ivec2(1))
                );
                ivec2 planeCoord = clamp(ivec2(round(planePos)), ivec2(0), uPlaneSize - ivec2(1));
                vec2 flow = texelFetch(uFlowGrid, tile, 0).rg;
                vec2 targetPlane = planePos + flow;
                vec2 referenceRaw = planePos * 2.0;
                vec2 targetRaw = targetPlane * 2.0;
                if (targetRaw.x < 0.0 || targetRaw.y < 0.0 ||
                    targetRaw.x > float(uImageSize.x - 1) ||
                    targetRaw.y > float(uImageSize.y - 1) ||
                    any(isnan(targetRaw)) || any(isinf(targetRaw))) {
                    writeSample(tile, referenceRaw, referenceRaw, 0.0, 0.0, 1.0, 0.0);
                    return;
                }
                float robustness = clamp(mapAt(uRobustness, planePos), 0.0, 1.0);
                float tileMask = clamp(texelFetch(uTileMask, tile, 0).r, 0.0, 1.0);
                float residual = abs(refAt(planeCoord) - curAt(targetPlane));
                float detail = detailAt(planeCoord);
                writeSample(tile, referenceRaw, targetRaw, robustness, tileMask, residual, detail);
            }
        """.trimIndent()

        private val CLEAR_ACCUMULATOR_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 16, local_size_y = 16) in;
            layout(rgba16f, binding = 0) writeonly uniform highp image2D uAccumulator;
            uniform ivec2 uImageSize;

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uImageSize.x || p.y >= uImageSize.y) return;
                imageStore(uAccumulator, p, vec4(0.0));
            }
        """.trimIndent()

        private val ACCUMULATE_COMPUTE_SHADER = """
            #version 310 es
            $RAW_COMMON
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform highp usampler2D uInputRaw;
            uniform sampler2D uFlowGrid;
            uniform sampler2D uRobustness;
            uniform sampler2D uTileMask;
            uniform sampler2D uKernel;
            uniform sampler2D uLensShadingMap;
            uniform sampler2D uAccumulatorInput;
            layout(rgba16f, binding = 0) writeonly uniform highp image2D uAccumulatorOutput;
            uniform ivec2 uImageSize;
            uniform ivec2 uPlaneSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            uniform int uIsReference;
            uniform int uCfaPattern;
            uniform float uBlackLevel[4];
            uniform float uWhiteLevel;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            uniform float uNoiseAlphaByChannel[4];
            uniform float uNoiseBetaByChannel[4];
            uniform float uFrameWeight;
            uniform int uHdrMode;
            uniform float uExposureScale;
            uniform float uMinBlendBaseWeight;
            uniform float uRobustnessFloorFactor;
            uniform float uPrecisionReferenceSignal;
            uniform float uLscNoiseGainMax;
            uniform float uWienerBaseWeight;
            uniform float uDenoiseSignalLow;
            uniform float uDenoiseSignalHigh;
            uniform float uDenoiseLscGainStart;
            uniform float uDenoiseLscGainEnd;
            uniform float uDenoiseStaticRobustStart;
            uniform float uDenoiseStaticRobustEnd;
            uniform float uDenoiseStaticTileStart;
            uniform float uDenoiseStaticTileEnd;
            uniform float uDenoiseNonReferenceWeightBoost;
            uniform float uDenoiseNonReferenceWeightFloor;
            uniform float uDenoiseReferenceDarkWeightScale;
            uniform float uHighlightSuppressionStrength;
            uniform float uHighlightSuppressionStart;
            uniform float uHighlightSuppressionEnd;
            uniform mat3 uRegistrationTransform;
            uniform int uRegistrationForceIdentity;

            vec2 flowAt(vec2 planePos) {
                vec2 grid = planePos / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uFlowGrid, clamp(uv, vec2(0.0), vec2(1.0))).rg;
            }

            vec2 registrationSourceRaw(vec2 rawPos) {
                if (uRegistrationForceIdentity != 0) {
                    return rawPos;
                }
                vec3 mapped = uRegistrationTransform * vec3(rawPos, 1.0);
                float z = abs(mapped.z) > 1e-6 ? mapped.z : 1.0;
                vec2 source = mapped.xy / z;
                if (any(isnan(source)) || any(isinf(source))) {
                    return rawPos;
                }
                return source;
            }

            float mapAt(sampler2D tex, vec2 planePos) {
                vec2 uv = (clamp(planePos, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(tex, uv).r;
            }

            float tileMaskAt(vec2 planePos) {
                vec2 grid = planePos / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uTileMask, clamp(uv, vec2(0.0), vec2(1.0))).r;
            }

            float lscGain(ivec2 samplePos) {
                vec2 uv = (vec2(samplePos) + vec2(0.5)) / vec2(uImageSize);
                vec4 gains = texture(uLensShadingMap, uv);
                int channel = lensShadingChannelAt(uCfaPattern, samplePos);
                if (channel == 0) return gains.r;
                if (channel == 1) return gains.g;
                if (channel == 2) return gains.b;
                return gains.a;
            }

            float lscNoiseGain(ivec2 samplePos) {
                return clamp(lscGain(samplePos), 1e-3, max(uLscNoiseGainMax, 1.0));
            }

            float rawNormAt(ivec2 samplePos, int bayerIndex) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                float raw = float(texelFetch(uInputRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) * lscGain(samplePos) / range, 0.0, 1.0);
            }

            float rawSensorNormAt(ivec2 samplePos) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                int bayerIndex = bayerIndexAt(uCfaPattern, samplePos);
                float raw = float(texelFetch(uInputRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) / range, 0.0, 1.0);
            }

            float tileSensorMax(ivec2 samplePos) {
                ivec2 base = samplePos - ivec2(samplePos.x & 1, samplePos.y & 1);
                float m = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        m = max(m, rawSensorNormAt(base + ivec2(x, y)));
                    }
                }
                return m;
            }

            float tileSensorMean(ivec2 samplePos) {
                ivec2 base = samplePos - ivec2(samplePos.x & 1, samplePos.y & 1);
                float sum = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        sum += rawSensorNormAt(base + ivec2(x, y));
                    }
                }
                return sum * 0.25;
            }

            float highlightClipAlpha(ivec2 samplePos) {
                float pixel = rawSensorNormAt(samplePos);
                float tile = tileSensorMax(samplePos);
                float tileMean = tileSensorMean(samplePos);
                float pixelClip = smoothstep(0.90, 0.985, pixel);
                float tileMeanGate = smoothstep(0.68, 0.88, tileMean);
                float edgeProtect = smoothstep(0.30, 0.10, tile - pixel);
                float tileClip = smoothstep(0.91, 0.995, tile) * tileMeanGate * edgeProtect;
                return clamp(max(pixelClip, 0.62 * tileClip), 0.0, 1.0);
            }

            float nonReferenceOverexposureAlpha(ivec2 samplePos, float sampledValue) {
                float pixel = rawSensorNormAt(samplePos);
                float tile = tileSensorMax(samplePos);
                float tileMean = tileSensorMean(samplePos);
                float pixelClip = smoothstep(0.965, 0.997, pixel);
                float sampledClip = smoothstep(0.985, 0.999, sampledValue);
                float tileClip = smoothstep(0.975, 0.999, tile) * smoothstep(0.78, 0.94, tileMean);
                return clamp(max(max(pixelClip, 0.55 * sampledClip), 0.75 * tileClip), 0.0, 1.0);
            }

            ivec2 cfaPhaseOffset(int cfaPattern, ivec2 p) {
                int period = cfaPeriod(cfaPattern);
                return ivec2(p.x % period, p.y % period);
            }

            ivec2 maxLatticeForPhase(ivec2 phaseOffset, int period) {
                return max((uImageSize - ivec2(1) - phaseOffset) / ivec2(period), ivec2(0));
            }

            float fetchSamePhase(ivec2 lattice, ivec2 phaseOffset, int period, int bayerIndex) {
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return rawNormAt(phaseOffset + lattice * period, bayerIndex);
            }

            float sampleSamePhase(vec2 rawPos, ivec2 phaseOffset, int period, int bayerIndex) {
                ivec2 maxLattice = maxLatticeForPhase(phaseOffset, period);
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                vec2 pos = clamp(
                    (rawPos - phase) / float(period),
                    vec2(0.0),
                    vec2(float(maxLattice.x), float(maxLattice.y))
                );
                ivec2 p0 = ivec2(floor(pos));
                ivec2 p1 = min(p0 + ivec2(1), maxLattice);
                vec2 f = pos - vec2(p0);
                float v00 = fetchSamePhase(p0, phaseOffset, period, bayerIndex);
                float v10 = fetchSamePhase(ivec2(p1.x, p0.y), phaseOffset, period, bayerIndex);
                float v01 = fetchSamePhase(ivec2(p0.x, p1.y), phaseOffset, period, bayerIndex);
                float v11 = fetchSamePhase(p1, phaseOffset, period, bayerIndex);
                return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
            }

            ivec2 nearestSamePhase(vec2 rawPos, ivec2 phaseOffset, int period) {
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                ivec2 lattice = ivec2(round((rawPos - phase) / float(period)));
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return phaseOffset + lattice * period;
            }

            vec3 kernelMatrix(vec4 params) {
                float k1 = params.x;
                float k2 = params.y;
                float c2 = params.z;
                float s2 = params.w;
                float sumK = k1 + k2;
                float diffK = k1 - k2;
                return vec3(0.5 * (sumK + diffK * c2), 0.5 * diffK * s2, 0.5 * (sumK - diffK * c2));
            }

            float kernelWeight(vec2 tap, vec4 params) {
                vec3 k = kernelMatrix(params);
                float mahalanobis = k.x * tap.x * tap.x + 2.0 * k.y * tap.x * tap.y + k.z * tap.y * tap.y;
                return exp(-0.5 * max(mahalanobis, 0.0));
            }

            float noiseVarianceForChannel(float correctedSignalNorm, int bayerIndex, float lscGainForNoise) {
                float channelAlpha = uNoiseAlphaByChannel[bayerIndex];
                float channelBeta = uNoiseBetaByChannel[bayerIndex];
                if (channelAlpha <= 0.0 && channelBeta <= 0.0) return 1e-10;
                float gain = clamp(lscGainForNoise, 1e-3, max(uLscNoiseGainMax, 1.0));
                return max(
                    channelAlpha * clamp(correctedSignalNorm, 0.0, 1.0) * gain +
                    channelBeta * gain * gain,
                    1e-10
                );
            }

            float sensorPrecisionWeight(float correctedSignalNorm, int bayerIndex, float lscGainForNoise) {
                float channelAlpha = uNoiseAlphaByChannel[bayerIndex];
                float channelBeta = uNoiseBetaByChannel[bayerIndex];
                if (channelAlpha <= 0.0 && channelBeta <= 0.0) return 1.0;
                float variance = noiseVarianceForChannel(correctedSignalNorm, bayerIndex, lscGainForNoise);
                float referenceVariance = noiseVarianceForChannel(uPrecisionReferenceSignal, bayerIndex, 1.0);
                return clamp(referenceVariance / variance, 0.05, 4.0);
            }

            float denoiseNeed(float correctedSignalNorm, float lscGainForNoise) {
                float dark = 1.0 - smoothstep(
                    min(uDenoiseSignalLow, uDenoiseSignalHigh),
                    max(uDenoiseSignalLow, uDenoiseSignalHigh),
                    correctedSignalNorm
                );
                float lsc = smoothstep(
                    min(uDenoiseLscGainStart, uDenoiseLscGainEnd),
                    max(uDenoiseLscGainStart, uDenoiseLscGainEnd),
                    lscGainForNoise
                );
                return clamp(max(dark, dark * 0.45 + lsc * 0.55), 0.0, 1.0);
            }

            float denoiseStaticConfidence(float local, float robust) {
                float tileConfidence = smoothstep(
                    min(uDenoiseStaticTileStart, uDenoiseStaticTileEnd),
                    max(uDenoiseStaticTileStart, uDenoiseStaticTileEnd),
                    local
                );
                float robustConfidence = smoothstep(
                    min(uDenoiseStaticRobustStart, uDenoiseStaticRobustEnd),
                    max(uDenoiseStaticRobustStart, uDenoiseStaticRobustEnd),
                    robust
                );
                return clamp(tileConfidence * robustConfidence, 0.0, 1.0);
            }

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uImageSize.x || p.y >= uImageSize.y) return;
                vec4 prev = texelFetch(uAccumulatorInput, p, 0);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                int period = cfaPeriod(uCfaPattern);
                ivec2 phaseOffset = cfaPhaseOffset(uCfaPattern, p);
                vec2 planePos = vec2(float(p.x / 2), float(p.y / 2));
                if (planePos.x < -0.001 || planePos.y < -0.001 ||
                    planePos.x > float(uPlaneSize.x - 1) + 0.001 ||
                    planePos.y > float(uPlaneSize.y - 1) + 0.001) {
                    imageStore(uAccumulatorOutput, p, prev);
                    return;
                }

                float value = 0.0;
                float robust = 1.0;
                float local = 1.0;
                float alignmentRobust = 1.0;
                float clipAlpha = 0.0;
                float sourceLscNoiseGain = 1.0;
                if (uIsReference != 0) {
                    value = rawNormAt(p, bayerIndex);
                    sourceLscNoiseGain = lscNoiseGain(p);
                    if (uHdrMode != 0) {
                        clipAlpha = highlightClipAlpha(p);
                    }
                } else {
                    vec2 sourceRaw = registrationSourceRaw(vec2(p));
                    if (sourceRaw.x < 0.0 || sourceRaw.y < 0.0 ||
                        sourceRaw.x > float(uImageSize.x - 1) ||
                        sourceRaw.y > float(uImageSize.y - 1)) {
                        imageStore(uAccumulatorOutput, p, prev);
                        return;
                    }
                    alignmentRobust = mapAt(uRobustness, planePos);
                    robust = alignmentRobust;
                    local = tileMaskAt(planePos);
                    float base = local * max(robust, uRobustnessFloorFactor * local);
                    if (base <= uMinBlendBaseWeight) {
                        imageStore(uAccumulatorOutput, p, prev);
                        return;
                    }
                    ivec2 kernelCoord = clamp(ivec2(round(planePos)), ivec2(0), uPlaneSize - ivec2(1));
                    vec4 kernel = texelFetch(uKernel, kernelCoord, 0);
                    float sumValue = 0.0;
                    float sumWeight = 0.0;
                    for (int y = -1; y <= 1; ++y) {
                        for (int x = -1; x <= 1; ++x) {
                            vec2 tap = vec2(float(x), float(y));
                            float w = kernelWeight(tap, kernel);
                            sumValue += sampleSamePhase(sourceRaw + tap * float(period), phaseOffset, period, bayerIndex) * w;
                            sumWeight += w;
                        }
                    }
                    value = sumValue / max(sumWeight, 1e-5);
                    ivec2 sourceSample = nearestSamePhase(sourceRaw, phaseOffset, period);
                    sourceLscNoiseGain = lscNoiseGain(sourceSample);
                    if (uHdrMode != 0) {
                        clipAlpha = highlightClipAlpha(sourceSample);
                    } else {
                        clipAlpha = nonReferenceOverexposureAlpha(sourceSample, value);
                    }
                    float highlightSuppression = 1.0 - uHighlightSuppressionStrength *
                        smoothstep(uHighlightSuppressionStart, uHighlightSuppressionEnd, value);
                    highlightSuppression *= 1.0 - clipAlpha;
                    robust = base * highlightSuppression;
                }

                float sourceValue = value;
                float outputValue = uHdrMode != 0 ? clamp(sourceValue * uExposureScale, 0.0, 1.0) : sourceValue;
                float variance = noiseVarianceForChannel(sourceValue, bayerIndex, sourceLscNoiseGain);
                float outputVariance = uHdrMode != 0 ? variance * uExposureScale * uExposureScale : variance;
                float wiener = (outputValue * outputValue) / (outputValue * outputValue + outputVariance + 1e-6);
                float precisionWeight = sensorPrecisionWeight(sourceValue, bayerIndex, sourceLscNoiseGain);
                float weight = precisionWeight * (uWienerBaseWeight + wiener);
                float denoiseAmount = denoiseNeed(sourceValue, sourceLscNoiseGain);
                if (uIsReference == 0) {
                    float staticConfidence = denoiseStaticConfidence(local, alignmentRobust) * (1.0 - clipAlpha);
                    float denoiseBoost = denoiseAmount * staticConfidence;
                    weight *= uFrameWeight * robust;
                    weight *= mix(1.0, max(uDenoiseNonReferenceWeightBoost, 1.0), denoiseBoost);
                    weight = max(
                        weight,
                        precisionWeight * max(uDenoiseNonReferenceWeightFloor, 0.0) * denoiseBoost
                    );
                } else if (uHdrMode == 0) {
                    weight *= mix(1.0, clamp(uDenoiseReferenceDarkWeightScale, 0.05, 1.0), denoiseAmount);
                }
                if (uHdrMode != 0) {
                    weight *= 1.0 - 0.90 * clipAlpha;
                }
                if (weight <= 0.0 && (uHdrMode == 0 || clipAlpha <= 0.0)) {
                    imageStore(uAccumulatorOutput, p, prev);
                    return;
                }

                float clipMass = uHdrMode != 0 ?
                    precisionWeight * uFrameWeight * clipAlpha :
                    weight * clamp(robust, 0.0, 1.0);
                vec4 next = prev + vec4(
                    outputValue * weight,
                    weight,
                    outputValue * outputValue * weight,
                    clipMass
                );
                imageStore(uAccumulatorOutput, p, next);
            }
        """.trimIndent()

        private val SUPER_RESOLUTION_ACCUMULATE_COMPUTE_SHADER = """
            #version 310 es
            $RAW_COMMON
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform highp usampler2D uInputRaw;
            uniform sampler2D uFlowGrid;
            uniform sampler2D uRobustness;
            uniform sampler2D uTileMask;
            uniform sampler2D uKernel;
            uniform sampler2D uLensShadingMap;
            uniform sampler2D uAccumulatorInput;
            layout(rgba16f, binding = 0) writeonly uniform highp image2D uAccumulatorOutput;
            uniform ivec2 uImageSize;
            uniform ivec2 uOutputSize;
            uniform ivec2 uPlaneSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            uniform int uIsReference;
            uniform int uCfaPattern;
            uniform float uBlackLevel[4];
            uniform float uWhiteLevel;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            uniform float uNoiseAlphaByChannel[4];
            uniform float uNoiseBetaByChannel[4];
            uniform float uFrameWeight;
            uniform float uOutputScale;
            uniform float uSrMinEffectiveWeight;
            uniform float uSrSplatRadius;
            uniform float uRegistrationSrWeight;
            uniform float uMinBlendBaseWeight;
            uniform float uRobustnessFloorFactor;
            uniform float uPrecisionReferenceSignal;
            uniform float uLscNoiseGainMax;
            uniform float uWienerBaseWeight;
            uniform float uHighlightSuppressionStrength;
            uniform float uHighlightSuppressionStart;
            uniform float uHighlightSuppressionEnd;
            uniform mat3 uRegistrationTransform;
            uniform int uRegistrationForceIdentity;

            vec2 referenceRawPos(ivec2 outputPos) {
                return (vec2(outputPos) + vec2(0.5)) / max(uOutputScale, 1.0) - vec2(0.5);
            }

            vec2 registrationSourceRaw(vec2 rawPos) {
                if (uRegistrationForceIdentity != 0) {
                    return rawPos;
                }
                vec3 mapped = uRegistrationTransform * vec3(rawPos, 1.0);
                float z = abs(mapped.z) > 1e-6 ? mapped.z : 1.0;
                vec2 source = mapped.xy / z;
                if (any(isnan(source)) || any(isinf(source))) {
                    return rawPos;
                }
                return source;
            }

            float mapAt(sampler2D tex, vec2 planePos) {
                vec2 uv = (clamp(planePos, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(tex, uv).r;
            }

            float tileMaskAt(vec2 planePos) {
                vec2 grid = planePos / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uTileMask, clamp(uv, vec2(0.0), vec2(1.0))).r;
            }

            float lscGain(ivec2 samplePos) {
                vec2 uv = (vec2(samplePos) + vec2(0.5)) / vec2(uImageSize);
                vec4 gains = texture(uLensShadingMap, uv);
                int channel = lensShadingChannelAt(uCfaPattern, samplePos);
                if (channel == 0) return gains.r;
                if (channel == 1) return gains.g;
                if (channel == 2) return gains.b;
                return gains.a;
            }

            float lscNoiseGain(ivec2 samplePos) {
                return clamp(lscGain(samplePos), 1e-3, max(uLscNoiseGainMax, 1.0));
            }

            float rawNormAt(ivec2 samplePos, int bayerIndex) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                float raw = float(texelFetch(uInputRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) * lscGain(samplePos) / range, 0.0, 1.0);
            }

            float rawSensorNormAt(ivec2 samplePos) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                int bayerIndex = bayerIndexAt(uCfaPattern, samplePos);
                float raw = float(texelFetch(uInputRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) / range, 0.0, 1.0);
            }

            float tileSensorMax(ivec2 samplePos) {
                ivec2 base = samplePos - ivec2(samplePos.x & 1, samplePos.y & 1);
                float m = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        m = max(m, rawSensorNormAt(base + ivec2(x, y)));
                    }
                }
                return m;
            }

            float tileSensorMean(ivec2 samplePos) {
                ivec2 base = samplePos - ivec2(samplePos.x & 1, samplePos.y & 1);
                float sum = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        sum += rawSensorNormAt(base + ivec2(x, y));
                    }
                }
                return sum * 0.25;
            }

            float highlightClipAlpha(ivec2 samplePos) {
                float pixel = rawSensorNormAt(samplePos);
                float tile = tileSensorMax(samplePos);
                float tileMean = tileSensorMean(samplePos);
                float pixelClip = smoothstep(0.965, 0.997, pixel);
                float tileClip = smoothstep(0.975, 0.999, tile) * smoothstep(0.78, 0.94, tileMean);
                return clamp(max(pixelClip, 0.75 * tileClip), 0.0, 1.0);
            }

            ivec2 maxLatticeForPhase(ivec2 phaseOffset, int period) {
                return max((uImageSize - ivec2(1) - phaseOffset) / ivec2(period), ivec2(0));
            }

            float fetchSamePhase(ivec2 lattice, ivec2 phaseOffset, int period, int bayerIndex) {
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return rawNormAt(phaseOffset + lattice * period, bayerIndex);
            }

            float sampleSamePhase(vec2 rawPos, ivec2 phaseOffset, int period, int bayerIndex) {
                ivec2 maxLattice = maxLatticeForPhase(phaseOffset, period);
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                vec2 pos = clamp(
                    (rawPos - phase) / float(period),
                    vec2(0.0),
                    vec2(float(maxLattice.x), float(maxLattice.y))
                );
                ivec2 p0 = ivec2(floor(pos));
                ivec2 p1 = min(p0 + ivec2(1), maxLattice);
                vec2 f = pos - vec2(p0);
                float v00 = fetchSamePhase(p0, phaseOffset, period, bayerIndex);
                float v10 = fetchSamePhase(ivec2(p1.x, p0.y), phaseOffset, period, bayerIndex);
                float v01 = fetchSamePhase(ivec2(p0.x, p1.y), phaseOffset, period, bayerIndex);
                float v11 = fetchSamePhase(p1, phaseOffset, period, bayerIndex);
                return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
            }

            ivec2 nearestSamePhase(vec2 rawPos, ivec2 phaseOffset, int period) {
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                ivec2 lattice = ivec2(round((rawPos - phase) / float(period)));
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return phaseOffset + lattice * period;
            }

            vec3 kernelMatrix(vec4 params) {
                float k1 = params.x;
                float k2 = params.y;
                float c2 = params.z;
                float s2 = params.w;
                float sumK = k1 + k2;
                float diffK = k1 - k2;
                return vec3(0.5 * (sumK + diffK * c2), 0.5 * diffK * s2, 0.5 * (sumK - diffK * c2));
            }

            float kernelWeight(vec2 tap, vec4 params) {
                vec3 k = kernelMatrix(params);
                float mahalanobis = k.x * tap.x * tap.x + 2.0 * k.y * tap.x * tap.y + k.z * tap.y * tap.y;
                return exp(-0.5 * max(mahalanobis, 0.0));
            }

            float noiseVarianceForChannel(float correctedSignalNorm, int bayerIndex, float lscGainForNoise) {
                float channelAlpha = uNoiseAlphaByChannel[bayerIndex];
                float channelBeta = uNoiseBetaByChannel[bayerIndex];
                if (channelAlpha <= 0.0 && channelBeta <= 0.0) return 1e-10;
                float gain = clamp(lscGainForNoise, 1e-3, max(uLscNoiseGainMax, 1.0));
                return max(
                    channelAlpha * clamp(correctedSignalNorm, 0.0, 1.0) * gain +
                    channelBeta * gain * gain,
                    1e-10
                );
            }

            float sensorPrecisionWeight(float correctedSignalNorm, int bayerIndex, float lscGainForNoise) {
                float channelAlpha = uNoiseAlphaByChannel[bayerIndex];
                float channelBeta = uNoiseBetaByChannel[bayerIndex];
                if (channelAlpha <= 0.0 && channelBeta <= 0.0) return 1.0;
                float variance = noiseVarianceForChannel(correctedSignalNorm, bayerIndex, lscGainForNoise);
                float referenceVariance = noiseVarianceForChannel(uPrecisionReferenceSignal, bayerIndex, 1.0);
                return clamp(referenceVariance / variance, 0.05, 4.0);
            }

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uOutputSize.x || p.y >= uOutputSize.y) return;

                vec4 prev = texelFetch(uAccumulatorInput, p, 0);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                int period = cfaPeriod(uCfaPattern);
                ivec2 phaseOffset = ivec2(p.x % period, p.y % period);
                vec2 refRaw = referenceRawPos(p);
                vec2 planePos = refRaw * 0.5;

                float robust = 1.0;
                float base = 1.0;
                vec2 sourceRaw = refRaw;
                if (uIsReference == 0) {
                    if (uRegistrationSrWeight <= 0.0) {
                        imageStore(uAccumulatorOutput, p, prev);
                        return;
                    }
                    sourceRaw = registrationSourceRaw(refRaw);
                    if (sourceRaw.x < -0.5 || sourceRaw.y < -0.5 ||
                        sourceRaw.x > float(uImageSize.x) - 0.5 ||
                        sourceRaw.y > float(uImageSize.y) - 0.5) {
                        imageStore(uAccumulatorOutput, p, prev);
                        return;
                    }
                    float local = tileMaskAt(planePos);
                    robust = mapAt(uRobustness, planePos);
                    base = local * max(robust, uRobustnessFloorFactor * local);
                    if (base <= max(uMinBlendBaseWeight, uSrMinEffectiveWeight * 0.25)) {
                        imageStore(uAccumulatorOutput, p, prev);
                        return;
                    }
                }

                ivec2 kernelCoord = clamp(ivec2(round(planePos)), ivec2(0), uPlaneSize - ivec2(1));
                vec4 kernel = texelFetch(uKernel, kernelCoord, 0);
                float sumValue = 0.0;
                float sumWeight = 0.0;
                float tapScale = float(period) * max(uSrSplatRadius, 0.25);
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        vec2 tap = vec2(float(x), float(y));
                        float w = kernelWeight(tap, kernel);
                        sumValue += sampleSamePhase(sourceRaw + tap * tapScale, phaseOffset, period, bayerIndex) * w;
                        sumWeight += w;
                    }
                }
                float value = sumValue / max(sumWeight, 1e-5);
                ivec2 sourceSample = nearestSamePhase(sourceRaw, phaseOffset, period);
                float sourceLscNoiseGain = lscNoiseGain(sourceSample);
                float clipAlpha = highlightClipAlpha(sourceSample);
                float highlightSuppression = 1.0 - uHighlightSuppressionStrength *
                    smoothstep(uHighlightSuppressionStart, uHighlightSuppressionEnd, value);
                if (uIsReference == 0) {
                    robust = base * highlightSuppression * (1.0 - clipAlpha);
                }

                float variance = noiseVarianceForChannel(value, bayerIndex, sourceLscNoiseGain);
                float wiener = (value * value) / (value * value + variance + 1e-6);
                float precisionWeight = sensorPrecisionWeight(value, bayerIndex, sourceLscNoiseGain);
                float weight = precisionWeight * (uWienerBaseWeight + wiener);
                if (uIsReference == 0) {
                    weight *= uFrameWeight * robust * clamp(uRegistrationSrWeight, 0.0, 1.0);
                }
                if (weight <= 0.0) {
                    imageStore(uAccumulatorOutput, p, prev);
                    return;
                }

                vec4 next = prev + vec4(
                    value * weight,
                    weight,
                    value * value * weight,
                    uIsReference != 0 ? 0.0 : weight * clamp(robust, 0.0, 1.0)
                );
                imageStore(uAccumulatorOutput, p, next);
            }
        """.trimIndent()

        private val SUPER_RESOLUTION_NORMALIZE_FRAGMENT_SHADER = """
            #version 300 es
            $RAW_COMMON
            in vec2 vTexCoord;
            out vec4 fragColor;
            uniform sampler2D uSrAccumulator;
            uniform sampler2D uBasePackedRaw;
            uniform highp usampler2D uReferenceRaw;
            uniform sampler2D uLensShadingMap;
            uniform ivec2 uImageSize;
            uniform ivec2 uOutputSize;
            uniform int uCfaPattern;
            uniform float uBlackLevel[4];
            uniform float uWhiteLevel;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            uniform float uNoiseAlphaByChannel[4];
            uniform float uNoiseBetaByChannel[4];
            uniform float uOutputScale;
            uniform int uForceBaseUpscale;
            uniform float uMinEffectiveWeight;
            uniform float uDetailRestoreStrength;
            uniform float uLscNoiseGainMax;
            uniform float uLowLightSignalLow;
            uniform float uLowLightSignalHigh;
            uniform float uLscSmoothGainStart;
            uniform float uLscSmoothGainEnd;
            uniform float uSrDetailLowLightSuppression;
            uniform float uSrDetailLscSuppression;

            vec2 referenceRawPos(ivec2 outputPos) {
                return (vec2(outputPos) + vec2(0.5)) / max(uOutputScale, 1.0) - vec2(0.5);
            }

            float lscGain(ivec2 samplePos) {
                vec2 uv = (vec2(samplePos) + vec2(0.5)) / vec2(uImageSize);
                vec4 gains = texture(uLensShadingMap, uv);
                int channel = lensShadingChannelAt(uCfaPattern, samplePos);
                if (channel == 0) return gains.r;
                if (channel == 1) return gains.g;
                if (channel == 2) return gains.b;
                return gains.a;
            }

            float lscNoiseGain(ivec2 samplePos) {
                return clamp(lscGain(samplePos), 1e-3, max(uLscNoiseGainMax, 1.0));
            }

            float rawNormAt(ivec2 samplePos, int bayerIndex) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                float raw = float(texelFetch(uReferenceRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) * lscGain(samplePos) / range, 0.0, 1.0);
            }

            float basePackedNormAt(ivec2 samplePos) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                vec2 rg = texelFetch(uBasePackedRaw, samplePos, 0).rg;
                float lo = floor(rg.r * 255.0 + 0.5);
                float hi = floor(rg.g * 255.0 + 0.5);
                return clamp((lo + hi * 256.0) / 65535.0, 0.0, 1.0);
            }

            ivec2 maxLatticeForPhase(ivec2 phaseOffset, int period) {
                return max((uImageSize - ivec2(1) - phaseOffset) / ivec2(period), ivec2(0));
            }

            float fetchSamePhase(ivec2 lattice, ivec2 phaseOffset, int period, int bayerIndex) {
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return rawNormAt(phaseOffset + lattice * period, bayerIndex);
            }

            float fetchBaseSamePhase(ivec2 lattice, ivec2 phaseOffset, int period) {
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return basePackedNormAt(phaseOffset + lattice * period);
            }

            float sampleSamePhase(vec2 rawPos, ivec2 phaseOffset, int period, int bayerIndex) {
                ivec2 maxLattice = maxLatticeForPhase(phaseOffset, period);
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                vec2 pos = clamp(
                    (rawPos - phase) / float(period),
                    vec2(0.0),
                    vec2(float(maxLattice.x), float(maxLattice.y))
                );
                ivec2 p0 = ivec2(floor(pos));
                ivec2 p1 = min(p0 + ivec2(1), maxLattice);
                vec2 f = pos - vec2(p0);
                float v00 = fetchSamePhase(p0, phaseOffset, period, bayerIndex);
                float v10 = fetchSamePhase(ivec2(p1.x, p0.y), phaseOffset, period, bayerIndex);
                float v01 = fetchSamePhase(ivec2(p0.x, p1.y), phaseOffset, period, bayerIndex);
                float v11 = fetchSamePhase(p1, phaseOffset, period, bayerIndex);
                return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
            }

            float sampleBaseSamePhase(vec2 rawPos, ivec2 phaseOffset, int period) {
                ivec2 maxLattice = maxLatticeForPhase(phaseOffset, period);
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                vec2 pos = clamp(
                    (rawPos - phase) / float(period),
                    vec2(0.0),
                    vec2(float(maxLattice.x), float(maxLattice.y))
                );
                ivec2 p0 = ivec2(floor(pos));
                ivec2 p1 = min(p0 + ivec2(1), maxLattice);
                vec2 f = pos - vec2(p0);
                float v00 = fetchBaseSamePhase(p0, phaseOffset, period);
                float v10 = fetchBaseSamePhase(ivec2(p1.x, p0.y), phaseOffset, period);
                float v01 = fetchBaseSamePhase(ivec2(p0.x, p1.y), phaseOffset, period);
                float v11 = fetchBaseSamePhase(p1, phaseOffset, period);
                return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
            }

            ivec2 nearestSamePhase(vec2 rawPos, ivec2 phaseOffset, int period) {
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                ivec2 lattice = ivec2(round((rawPos - phase) / float(period)));
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return phaseOffset + lattice * period;
            }

            float referenceOutputNorm(ivec2 outputPos, int bayerIndex, int period, ivec2 phaseOffset) {
                return sampleSamePhase(referenceRawPos(outputPos), phaseOffset, period, bayerIndex);
            }

            float baseOutputNorm(ivec2 outputPos, int period, ivec2 phaseOffset) {
                return sampleBaseSamePhase(referenceRawPos(outputPos), phaseOffset, period);
            }

            float lowLightAmount(float signal) {
                return 1.0 - smoothstep(
                    min(uLowLightSignalLow, uLowLightSignalHigh),
                    max(uLowLightSignalLow, uLowLightSignalHigh),
                    signal
                );
            }

            float lscSmoothAmount(float lscGainForNoise) {
                return smoothstep(
                    min(uLscSmoothGainStart, uLscSmoothGainEnd),
                    max(uLscSmoothGainStart, uLscSmoothGainEnd),
                    lscGainForNoise
                );
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                int period = cfaPeriod(uCfaPattern);
                ivec2 phaseOffset = ivec2(p.x % period, p.y % period);
                vec4 a = texelFetch(uSrAccumulator, p, 0);
                float base = baseOutputNorm(p, period, phaseOffset);
                if (uForceBaseUpscale != 0) {
                    uint raw = uint(floor(clamp(base, 0.0, 1.0) * 65535.0 + 0.5));
                    uint lo = raw & 255u;
                    uint hi = (raw >> 8) & 255u;
                    fragColor = vec4(float(lo) / 255.0, float(hi) / 255.0, 0.0, 1.0);
                    return;
                }
                float reference = referenceOutputNorm(p, bayerIndex, period, phaseOffset);
                float sr = a.g > 1e-5 ? clamp(a.r / max(a.g, 1e-5), 0.0, 1.0) : base;
                float detailSupport = max(a.a, 0.0);
                float confidence = smoothstep(
                    max(uMinEffectiveWeight, 0.02),
                    max(uMinEffectiveWeight * 3.0, 0.35),
                    detailSupport
                );
                vec2 rawPos = referenceRawPos(p);
                ivec2 rawSample = nearestSamePhase(rawPos, phaseOffset, period);
                float lowLight = lowLightAmount(max(base, reference));
                float lsc = lscSmoothAmount(lscNoiseGain(rawSample));
                float detailDenoise = 1.0 - clamp(
                    uSrDetailLowLightSuppression * lowLight +
                    uSrDetailLscSuppression * lsc,
                    0.0,
                    0.95
                );
                float detailWeight = confidence * detailDenoise * clamp(uDetailRestoreStrength, 0.0, 1.0);
                float fused = clamp(base + (sr - base) * detailWeight, 0.0, 1.0);

                uint raw = uint(floor(fused * 65535.0 + 0.5));
                uint lo = raw & 255u;
                uint hi = (raw >> 8) & 255u;
                fragColor = vec4(float(lo) / 255.0, float(hi) / 255.0, 0.0, 1.0);
            }
        """.trimIndent()

        private val HDR_RECOVERY_MASK_FRAGMENT_SHADER = """
            #version 300 es
            $RAW_COMMON
            in vec2 vTexCoord;
            out vec4 fragColor;
            uniform sampler2D uAccumulator;
            uniform highp usampler2D uReferenceRaw;
            uniform highp usampler2D uShortRaw;
            uniform sampler2D uShortGlobalAlignment;
            uniform ivec2 uImageSize;
            uniform ivec2 uPlaneSize;
            uniform int uCfaPattern;
            uniform float uBlackLevel[4];
            uniform float uWhiteLevel;
            uniform float uShortToReferenceExposureScale;

            ivec2 clampRaw(ivec2 p) {
                return clamp(p, ivec2(0), uImageSize - ivec2(1));
            }

            float sensorNorm(usampler2D tex, ivec2 p) {
                p = clampRaw(p);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                float raw = float(texelFetch(tex, p, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) / range, 0.0, 1.0);
            }

            vec2 shortAlignedRawPos(ivec2 p) {
                vec2 rawOffset = texelFetch(uShortGlobalAlignment, ivec2(0), 0).rg * 2.0;
                vec2 rawPos = vec2(float(p.x), float(p.y)) + rawOffset;
                vec2 maxRawPos = vec2(float(uImageSize.x - 1), float(uImageSize.y - 1));
                return clamp(rawPos, vec2(0.0), maxRawPos);
            }

            ivec2 cfaPhaseOffset(int cfaPattern, ivec2 p) {
                int period = cfaPeriod(cfaPattern);
                return ivec2(p.x % period, p.y % period);
            }

            ivec2 maxLatticeForPhase(ivec2 phaseOffset, int period) {
                return max((uImageSize - ivec2(1) - phaseOffset) / ivec2(period), ivec2(0));
            }

            float shortRawNormAt(ivec2 samplePos, int bayerIndex) {
                samplePos = clampRaw(samplePos);
                float raw = float(texelFetch(uShortRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) / range, 0.0, 1.0);
            }

            float fetchShortSamePhase(ivec2 lattice, ivec2 phaseOffset, int period, int bayerIndex) {
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return shortRawNormAt(phaseOffset + lattice * period, bayerIndex);
            }

            float sampleShortSamePhase(vec2 rawPos, ivec2 phaseOffset, int period, int bayerIndex) {
                ivec2 maxLattice = maxLatticeForPhase(phaseOffset, period);
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                vec2 pos = clamp(
                    (rawPos - phase) / float(period),
                    vec2(0.0),
                    vec2(float(maxLattice.x), float(maxLattice.y))
                );
                ivec2 p0 = ivec2(floor(pos));
                ivec2 p1 = min(p0 + ivec2(1), maxLattice);
                vec2 f = pos - vec2(p0);
                float v00 = fetchShortSamePhase(p0, phaseOffset, period, bayerIndex);
                float v10 = fetchShortSamePhase(ivec2(p1.x, p0.y), phaseOffset, period, bayerIndex);
                float v01 = fetchShortSamePhase(ivec2(p0.x, p1.y), phaseOffset, period, bayerIndex);
                float v11 = fetchShortSamePhase(p1, phaseOffset, period, bayerIndex);
                return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
            }

            float shortSensorNorm(ivec2 p) {
                p = clampRaw(p);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                int period = cfaPeriod(uCfaPattern);
                return sampleShortSamePhase(
                    shortAlignedRawPos(p),
                    cfaPhaseOffset(uCfaPattern, p),
                    period,
                    bayerIndex
                );
            }

            float shortReferenceNorm(ivec2 p) {
                return clamp(shortSensorNorm(p) * uShortToReferenceExposureScale, 0.0, 1.0);
            }

            float referenceBlockMax(ivec2 base) {
                float m = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        m = max(m, sensorNorm(uReferenceRaw, base + ivec2(x, y)));
                    }
                }
                return m;
            }

            float shortReferenceBlockMax(ivec2 base) {
                float m = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        m = max(m, shortReferenceNorm(base + ivec2(x, y)));
                    }
                }
                return m;
            }

            float shortReferenceBlockMean(ivec2 base) {
                float sum = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        sum += shortReferenceNorm(base + ivec2(x, y));
                    }
                }
                return sum * 0.25;
            }

            float blockClipRatioMax(ivec2 base) {
                float m = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        ivec2 p = clampRaw(base + ivec2(x, y));
                        vec4 a = texelFetch(uAccumulator, p, 0);
                        m = max(m, a.a / max(a.a + a.g, 1e-5));
                    }
                }
                return m;
            }

            float shortReferenceConsistency(ivec2 base) {
                float residualSum = 0.0;
                float count = 0.0;
                for (int y = -1; y <= 2; ++y) {
                    for (int x = -1; x <= 2; ++x) {
                        ivec2 p = base + ivec2(x, y);
                        float reference = sensorNorm(uReferenceRaw, p);
                        if (reference <= 0.035 || reference >= 0.860) {
                            continue;
                        }
                        float shortReference = shortReferenceNorm(p);
                        float residual = abs(shortReference - reference) / max(max(shortReference, reference), 0.08);
                        residualSum += residual;
                        count += 1.0;
                    }
                }
                float support = smoothstep(2.0, 8.0, count);
                float averageResidual = residualSum / max(count, 1.0);
                float consistency = 1.0 - smoothstep(0.26, 0.58, averageResidual);
                return mix(1.0, consistency, support);
            }

            void main() {
                ivec2 block = clamp(ivec2(gl_FragCoord.xy), ivec2(0), uPlaneSize - ivec2(1));
                ivec2 base = block * 2;
                float refMax = referenceBlockMax(base);
                float clipRatio = blockClipRatioMax(base);
                float shortMax = shortReferenceBlockMax(base);
                float shortMean = shortReferenceBlockMean(base);
                float consistency = shortReferenceConsistency(base);

                float saturatedCore = smoothstep(0.970, 0.995, refMax);
                float accumulatedClip = smoothstep(0.35, 0.65, clipRatio) * smoothstep(0.880, 0.960, refMax);
                float sourceMask = max(saturatedCore, accumulatedClip);

                float shortPeakConfidence = smoothstep(0.46, 0.78, shortMax);
                float shortMeanConfidence = smoothstep(0.16, 0.42, shortMean);
                float shortSignalConfidence = shortPeakConfidence * (0.45 + 0.55 * shortMeanConfidence);
                float recoveryConfidence = clamp(shortSignalConfidence * consistency, 0.0, 1.0);

                sourceMask = clamp(sourceMask * recoveryConfidence, 0.0, 1.0);
                float saturationPush = smoothstep(0.990, 0.999, refMax) * recoveryConfidence;
                fragColor = vec4(sourceMask, saturationPush, 0.0, 1.0);
            }
        """.trimIndent()

        private val HDR_RECOVERY_DILATE_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            precision highp int;
            uniform sampler2D uMask;
            uniform ivec2 uPlaneSize;
            out float fragColor;

            float maskAt(ivec2 p) {
                p = clamp(p, ivec2(0), uPlaneSize - ivec2(1));
                return texelFetch(uMask, p, 0).r;
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                float m = 0.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        m = max(m, maskAt(p + ivec2(x, y)));
                    }
                }
                fragColor = clamp(m, 0.0, 1.0);
            }
        """.trimIndent()

        private val HDR_RECOVERY_FEATHER_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            precision highp int;
            uniform sampler2D uMask;
            uniform ivec2 uPlaneSize;
            out float fragColor;

            float maskAt(ivec2 p) {
                p = clamp(p, ivec2(0), uPlaneSize - ivec2(1));
                return texelFetch(uMask, p, 0).r;
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                float center = maskAt(p);
                float sum = center * 4.0;
                sum += (maskAt(p + ivec2(1, 0)) + maskAt(p + ivec2(-1, 0)) +
                    maskAt(p + ivec2(0, 1)) + maskAt(p + ivec2(0, -1))) * 2.0;
                sum += maskAt(p + ivec2(1, 1)) + maskAt(p + ivec2(-1, 1)) +
                    maskAt(p + ivec2(1, -1)) + maskAt(p + ivec2(-1, -1));
                float feather = sum / 16.0;
                fragColor = clamp(feather, 0.0, 1.0);
            }
        """.trimIndent()

        private val PGTM_STATS_COMPUTE_SHADER = """
            #version 310 es
            $RAW_COMMON
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform sampler2D uPackedRaw;
            uniform ivec2 uImageSize;
            uniform ivec4 uStatsBounds;
            uniform ivec2 uGridSize;
            uniform int uCfaPattern;
            uniform float uBaselineExposureGain;
            uniform mat3 uColorCorrectionMatrix;
            layout(std430, binding = $PGTM_STATS_BUFFER_BINDING) buffer PgtmStats {
                float stats[];
            };

            const int HIST_BINS = 256;
            const int STATS_STRIDE = 8;
            const int SAMPLE_GRID = 16;
            shared uint hist[HIST_BINS];
            shared float inputSamples[HIST_BINS];
            shared uint sampleCount;
            shared uint highlightCount;

            float packedRawAt(ivec2 p) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                vec2 rg = texelFetch(uPackedRaw, p, 0).rg;
                float lo = floor(rg.r * 255.0 + 0.5);
                float hi = floor(rg.g * 255.0 + 0.5);
                return clamp((lo + hi * 256.0) / 65535.0, 0.0, 1.0);
            }

            vec3 blockRgbAt(ivec2 base) {
                float r = 0.0;
                float g = 0.0;
                float b = 0.0;
                float rc = 0.0;
                float gc = 0.0;
                float bc = 0.0;
                for (int y = 0; y <= 1; ++y) {
                    for (int x = 0; x <= 1; ++x) {
                        ivec2 p = clamp(base + ivec2(x, y), ivec2(0), uImageSize - ivec2(1));
                        int channel = bayerIndexAt(uCfaPattern, p);
                        float value = packedRawAt(p);
                        if (channel == 0) {
                            r += value;
                            rc += 1.0;
                        } else if (channel == 3) {
                            b += value;
                            bc += 1.0;
                        } else {
                            g += value;
                            gc += 1.0;
                        }
                    }
                }
                float fallback = (r + g + b) / max(rc + gc + bc, 1.0);
                return vec3(
                    rc > 0.0 ? r / rc : fallback,
                    gc > 0.0 ? g / gc : fallback,
                    bc > 0.0 ? b / bc : fallback
                );
            }

            float pgtmInputAt(ivec2 base) {
                vec3 rgb = blockRgbAt(base);
                vec3 profileRgb = uColorCorrectionMatrix * rgb * uBaselineExposureGain;
                float minChannel = min(profileRgb.r, min(profileRgb.g, profileRgb.b));
                float maxChannel = max(profileRgb.r, max(profileRgb.g, profileRgb.b));
                return max(
                    0.1495 * profileRgb.r +
                    0.2935 * profileRgb.g +
                    0.0570 * profileRgb.b +
                    0.1250 * minChannel +
                    0.3750 * maxChannel,
                    0.0
                );
            }

            void main() {
                ivec2 cell = ivec2(gl_WorkGroupID.xy);
                if (cell.x >= uGridSize.x || cell.y >= uGridSize.y) return;
                ivec2 localId = ivec2(gl_LocalInvocationID.xy);
                int localIndex = localId.y * SAMPLE_GRID + localId.x;

                if (localIndex < HIST_BINS) {
                    hist[localIndex] = 0u;
                }
                if (localIndex == 0) {
                    sampleCount = 0u;
                    highlightCount = 0u;
                }
                barrier();

                ivec2 statsMin = clamp(uStatsBounds.xy, ivec2(0), uImageSize - ivec2(1));
                ivec2 statsMax = clamp(uStatsBounds.zw, statsMin + ivec2(1), uImageSize);
                ivec2 statsSize = max(statsMax - statsMin, ivec2(2));
                int startX = statsMin.x + (cell.x * statsSize.x) / uGridSize.x;
                int endX = statsMin.x + ((cell.x + 1) * statsSize.x + uGridSize.x - 1) / uGridSize.x;
                int startY = statsMin.y + (cell.y * statsSize.y) / uGridSize.y;
                int endY = statsMin.y + ((cell.y + 1) * statsSize.y + uGridSize.y - 1) / uGridSize.y;
                startX += startX & 1;
                startY += startY & 1;
                endX = min(statsMax.x, endX - (endX & 1));
                endY = min(statsMax.y, endY - (endY & 1));

                if (endX - startX < 2 || endY - startY < 2) {
                    if (localIndex == 0) {
                        int cellIndex = cell.y * uGridSize.x + cell.x;
                        int offset = cellIndex * STATS_STRIDE;
                        for (int i = 0; i < STATS_STRIDE; ++i) {
                            stats[offset + i] = 0.0;
                        }
                    }
                    return;
                }

                int cellWidth = max(endX - startX, 2);
                int cellHeight = max(endY - startY, 2);
                int x = startX + ((localId.x * 2 + 1) * cellWidth) / (SAMPLE_GRID * 2);
                int y = startY + ((localId.y * 2 + 1) * cellHeight) / (SAMPLE_GRID * 2);
                x = clamp(x - (x & 1), startX, max(startX, endX - 2));
                y = clamp(y - (y & 1), startY, max(startY, endY - 2));
                float inputValue = pgtmInputAt(ivec2(x, y));
                inputSamples[localIndex] = inputValue;
                float clampedInput = clamp(inputValue, 0.0, 1.0);
                uint bin = uint(clamp(int(floor(clampedInput * float(HIST_BINS - 1) + 0.5)), 0, HIST_BINS - 1));
                atomicAdd(hist[bin], 1u);
                atomicAdd(sampleCount, 1u);
                if (inputValue >= 0.92) {
                    atomicAdd(highlightCount, 1u);
                }
                memoryBarrierShared();
                barrier();

                if (localIndex != 0) {
                    return;
                }

                int samples = int(sampleCount);
                int target10 = max(1, int(ceil(float(samples) * 0.10)));
                int target50 = max(1, int(ceil(float(samples) * 0.50)));
                int target90 = max(1, int(ceil(float(samples) * 0.90)));
                int target98 = max(1, int(ceil(float(samples) * 0.98)));
                uint cumulative = 0u;
                float p10 = 1.0;
                float p50 = 1.0;
                float p90 = 1.0;
                float p98 = 1.0;
                float p995Input = 0.0;
                float p999Input = 0.0;
                bool got10 = false;
                bool got50 = false;
                bool got90 = false;
                bool got98 = false;
                for (int i = 0; i < HIST_BINS; ++i) {
                    cumulative += hist[i];
                    float value = float(i) / float(HIST_BINS - 1);
                    if (!got10 && cumulative >= uint(target10)) {
                        p10 = value;
                        got10 = true;
                    }
                    if (!got50 && cumulative >= uint(target50)) {
                        p50 = value;
                        got50 = true;
                    }
                    if (!got90 && cumulative >= uint(target90)) {
                        p90 = value;
                        got90 = true;
                    }
                    if (!got98 && cumulative >= uint(target98)) {
                        p98 = value;
                        got98 = true;
                    }
                }
                for (int i = 0; i < HIST_BINS; ++i) {
                    float value = inputSamples[i];
                    if (value >= p999Input) {
                        p995Input = p999Input;
                        p999Input = value;
                    } else if (value > p995Input) {
                        p995Input = value;
                    }
                }

                int cellIndex = cell.y * uGridSize.x + cell.x;
                int offset = cellIndex * STATS_STRIDE;
                stats[offset + 0] = samples > 0 ? p10 : 0.0;
                stats[offset + 1] = samples > 0 ? p50 : 0.0;
                stats[offset + 2] = samples > 0 ? p90 : 0.0;
                stats[offset + 3] = samples > 0 ? p98 : 0.0;
                stats[offset + 4] = samples > 0 ? float(highlightCount) / float(samples) : 0.0;
                stats[offset + 5] = float(samples);
                stats[offset + 6] = samples > 0 ? p995Input : 0.0;
                stats[offset + 7] = samples > 0 ? p999Input : 0.0;
            }
        """.trimIndent()

        private val DIAGNOSTIC_ALIGNMENT_COMPUTE_SHADER = """
            #version 310 es
            precision highp float;
            precision highp int;
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform sampler2D uFlowGrid;
            uniform sampler2D uRobustness;
            uniform sampler2D uTileMask;
            uniform sampler2D uReference;
            uniform sampler2D uCurrent;
            uniform ivec2 uImageSize;
            uniform ivec2 uPlaneSize;
            uniform ivec2 uGridSize;
            uniform ivec2 uSampleGridSize;
            uniform int uSampleStep;
            uniform int uTileSize;
            uniform float uFlowHistogramRange;
            uniform float uResidualHistogramRange;
            uniform float uNoiseResidualHistogramRange;
            uniform float uFlowRangeHistogramRange;
            uniform float uFlowOutlierThreshold;
            uniform float uTileRejectThreshold;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            uniform float uSrReadyResidualThreshold;
            uniform float uSrReadyNoiseResidualThreshold;
            uniform float uSrReadyFlowRangeThreshold;
            uniform float uHighConfidenceRobustnessThreshold;
            uniform float uHighConfidenceTileThreshold;
            uniform float uSrReadyDetailThreshold;
            uniform mat3 uRegistrationTransform;
            uniform int uRegistrationForceIdentity;
            layout(std430, binding = $DIAGNOSTIC_BUFFER_BINDING) buffer RawStackDiagnosticStats {
                uint stats[];
            };

            const int HIST_BINS = $DIAGNOSTIC_HIST_BINS;
            const int FLOW_COUNT_INDEX = $DIAGNOSTIC_FLOW_COUNT_INDEX;
            const int FLOW_SUM_INDEX = $DIAGNOSTIC_FLOW_SUM_INDEX;
            const int FLOW_MAX_INDEX = $DIAGNOSTIC_FLOW_MAX_INDEX;
            const int FLOW_OUTLIER_COUNT_INDEX = $DIAGNOSTIC_FLOW_OUTLIER_COUNT_INDEX;
            const int ROBUST_COUNT_INDEX = $DIAGNOSTIC_ROBUST_COUNT_INDEX;
            const int ROBUST_SUM_INDEX = $DIAGNOSTIC_ROBUST_SUM_INDEX;
            const int ROBUST_MAX_INDEX = $DIAGNOSTIC_ROBUST_MAX_INDEX;
            const int TILE_COUNT_INDEX = $DIAGNOSTIC_TILE_COUNT_INDEX;
            const int TILE_SUM_INDEX = $DIAGNOSTIC_TILE_SUM_INDEX;
            const int TILE_MAX_INDEX = $DIAGNOSTIC_TILE_MAX_INDEX;
            const int TILE_REJECT_COUNT_INDEX = $DIAGNOSTIC_TILE_REJECT_COUNT_INDEX;
            const int RESIDUAL_COUNT_INDEX = $DIAGNOSTIC_RESIDUAL_COUNT_INDEX;
            const int RESIDUAL_SUM_INDEX = $DIAGNOSTIC_RESIDUAL_SUM_INDEX;
            const int RESIDUAL_MAX_INDEX = $DIAGNOSTIC_RESIDUAL_MAX_INDEX;
            const int NOISE_RESIDUAL_COUNT_INDEX = $DIAGNOSTIC_NOISE_RESIDUAL_COUNT_INDEX;
            const int NOISE_RESIDUAL_SUM_INDEX = $DIAGNOSTIC_NOISE_RESIDUAL_SUM_INDEX;
            const int NOISE_RESIDUAL_MAX_INDEX = $DIAGNOSTIC_NOISE_RESIDUAL_MAX_INDEX;
            const int FLOW_RANGE_COUNT_INDEX = $DIAGNOSTIC_FLOW_RANGE_COUNT_INDEX;
            const int FLOW_RANGE_SUM_INDEX = $DIAGNOSTIC_FLOW_RANGE_SUM_INDEX;
            const int FLOW_RANGE_MAX_INDEX = $DIAGNOSTIC_FLOW_RANGE_MAX_INDEX;
            const int HIGH_CONFIDENCE_COUNT_INDEX = $DIAGNOSTIC_HIGH_CONFIDENCE_COUNT_INDEX;
            const int SR_ALIGNMENT_READY_COUNT_INDEX = $DIAGNOSTIC_SR_ALIGNMENT_READY_COUNT_INDEX;
            const int SR_DETAIL_READY_COUNT_INDEX = $DIAGNOSTIC_SR_DETAIL_READY_COUNT_INDEX;
            const int FLOW_HIST_OFFSET = $DIAGNOSTIC_FLOW_HIST_OFFSET;
            const int ROBUST_HIST_OFFSET = $DIAGNOSTIC_ROBUST_HIST_OFFSET;
            const int TILE_HIST_OFFSET = $DIAGNOSTIC_TILE_HIST_OFFSET;
            const int RESIDUAL_HIST_OFFSET = $DIAGNOSTIC_RESIDUAL_HIST_OFFSET;
            const int NOISE_RESIDUAL_HIST_OFFSET = $DIAGNOSTIC_NOISE_RESIDUAL_HIST_OFFSET;
            const int FLOW_RANGE_HIST_OFFSET = $DIAGNOSTIC_FLOW_RANGE_HIST_OFFSET;
            const float UNIT_Q = ${DIAGNOSTIC_UNIT_QUANTIZATION};
            const float FLOW_Q = ${DIAGNOSTIC_FLOW_QUANTIZATION};
            const float RESIDUAL_Q = ${DIAGNOSTIC_RESIDUAL_QUANTIZATION};
            const float NOISE_RESIDUAL_Q = ${DIAGNOSTIC_NOISE_RESIDUAL_QUANTIZATION};

            uint quant(float value, float scale) {
                return uint(clamp(value * scale + 0.5, 0.0, 4294967040.0));
            }

            int histBin(float value, float range) {
                float normalized = clamp(value / max(range, 1e-6), 0.0, 1.0);
                return int(clamp(floor(normalized * float(HIST_BINS - 1) + 0.5), 0.0, float(HIST_BINS - 1)));
            }

            vec2 flowAt(vec2 planePos) {
                vec2 grid = planePos / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uFlowGrid, clamp(uv, vec2(0.0), vec2(1.0))).rg;
            }

            vec2 flowAtGrid(ivec2 p) {
                p = clamp(p, ivec2(0), uGridSize - ivec2(1));
                return texelFetch(uFlowGrid, p, 0).rg;
            }

            float flowLocalRange(vec2 planePos, vec2 centerFlow) {
                vec2 gridCoord = planePos / float(uTileSize) - 0.5;
                ivec2 gCenter = clamp(ivec2(round(gridCoord)), ivec2(0), uGridSize - ivec2(1));
                vec2 fMin = centerFlow;
                vec2 fMax = centerFlow;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        vec2 f = flowAtGrid(gCenter + ivec2(x, y));
                        fMin = min(fMin, f);
                        fMax = max(fMax, f);
                    }
                }
                return length(fMax - fMin);
            }

            float planeMapAt(sampler2D tex, vec2 planePos) {
                vec2 uv = (clamp(planePos, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(tex, uv).r;
            }

            float tileMaskAt(vec2 planePos) {
                vec2 grid = planePos / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uTileMask, clamp(uv, vec2(0.0), vec2(1.0))).r;
            }

            float referenceAt(ivec2 p) {
                p = clamp(p, ivec2(0), uPlaneSize - ivec2(1));
                return texelFetch(uReference, p, 0).r;
            }

            float currentAt(vec2 p) {
                vec2 uv = (clamp(p, vec2(0.0), vec2(uPlaneSize - ivec2(1))) + vec2(0.5)) / vec2(uPlaneSize);
                return texture(uCurrent, uv).r;
            }

            vec2 registrationSourcePlane(vec2 planePos) {
                if (uRegistrationForceIdentity != 0) {
                    return planePos;
                }
                vec3 mapped = uRegistrationTransform * vec3(planePos * 2.0, 1.0);
                float z = abs(mapped.z) > 1e-6 ? mapped.z : 1.0;
                vec2 sourceRaw = mapped.xy / z;
                vec2 sourcePlane = sourceRaw * 0.5;
                if (any(isnan(sourcePlane)) || any(isinf(sourcePlane))) {
                    return planePos;
                }
                return sourcePlane;
            }

            float detailAt(ivec2 p) {
                float c = referenceAt(p);
                float gx = abs(referenceAt(p + ivec2(1, 0)) - referenceAt(p - ivec2(1, 0)));
                float gy = abs(referenceAt(p + ivec2(0, 1)) - referenceAt(p - ivec2(0, 1)));
                float lap = abs(4.0 * c -
                    referenceAt(p + ivec2(1, 0)) -
                    referenceAt(p - ivec2(1, 0)) -
                    referenceAt(p + ivec2(0, 1)) -
                    referenceAt(p - ivec2(0, 1)));
                return gx + gy + 0.5 * lap;
            }

            void main() {
                ivec2 sampleIndex = ivec2(gl_GlobalInvocationID.xy);
                if (sampleIndex.x >= uSampleGridSize.x || sampleIndex.y >= uSampleGridSize.y) return;
                ivec2 raw = min(sampleIndex * uSampleStep, uImageSize - ivec2(1));
                vec2 planePos = vec2(raw) * 0.5;
                ivec2 planeCoord = clamp(ivec2(round(planePos)), ivec2(0), uPlaneSize - ivec2(1));

                vec2 flow = flowAt(planePos);
                float flowMagnitude = length(flow);
                float flowRange = flowLocalRange(planePos, flow);
                float robustness = clamp(planeMapAt(uRobustness, planePos), 0.0, 1.0);
                float tile = clamp(tileMaskAt(planePos), 0.0, 1.0);
                float reference = referenceAt(planeCoord);
                vec2 registrationPlane = registrationSourcePlane(planePos);
                bool registrationInside = registrationPlane.x >= 0.0 && registrationPlane.y >= 0.0 &&
                    registrationPlane.x <= float(uPlaneSize.x - 1) &&
                    registrationPlane.y <= float(uPlaneSize.y - 1);
                float alignedCurrent = currentAt(registrationPlane);
                float residual = abs(reference - alignedCurrent);
                float noiseStd = sqrt(max(uNoiseAlpha * clamp(reference, 0.0, 1.0) + uNoiseBeta, 1e-10));
                float noiseResidual = residual / max(noiseStd, 1e-5);
                float detail = detailAt(planeCoord);
                bool highConfidence = robustness >= uHighConfidenceRobustnessThreshold &&
                    tile >= uHighConfidenceTileThreshold;
                bool srAlignmentReady = highConfidence &&
                    uRegistrationForceIdentity == 0 &&
                    registrationInside &&
                    residual <= uSrReadyResidualThreshold;
                bool srDetailReady = srAlignmentReady && detail >= uSrReadyDetailThreshold;

                atomicAdd(stats[FLOW_COUNT_INDEX], 1u);
                atomicAdd(stats[FLOW_SUM_INDEX], quant(flowMagnitude, FLOW_Q));
                atomicMax(stats[FLOW_MAX_INDEX], quant(flowMagnitude, FLOW_Q));
                if (flowMagnitude > uFlowOutlierThreshold) {
                    atomicAdd(stats[FLOW_OUTLIER_COUNT_INDEX], 1u);
                }
                atomicAdd(stats[FLOW_HIST_OFFSET + histBin(flowMagnitude, uFlowHistogramRange)], 1u);

                atomicAdd(stats[ROBUST_COUNT_INDEX], 1u);
                atomicAdd(stats[ROBUST_SUM_INDEX], quant(robustness, UNIT_Q));
                atomicMax(stats[ROBUST_MAX_INDEX], quant(robustness, UNIT_Q));
                atomicAdd(stats[ROBUST_HIST_OFFSET + histBin(robustness, 1.0)], 1u);

                atomicAdd(stats[TILE_COUNT_INDEX], 1u);
                atomicAdd(stats[TILE_SUM_INDEX], quant(tile, UNIT_Q));
                atomicMax(stats[TILE_MAX_INDEX], quant(tile, UNIT_Q));
                if (tile < uTileRejectThreshold) {
                    atomicAdd(stats[TILE_REJECT_COUNT_INDEX], 1u);
                }
                atomicAdd(stats[TILE_HIST_OFFSET + histBin(tile, 1.0)], 1u);

                atomicAdd(stats[RESIDUAL_COUNT_INDEX], 1u);
                atomicAdd(stats[RESIDUAL_SUM_INDEX], quant(residual, RESIDUAL_Q));
                atomicMax(stats[RESIDUAL_MAX_INDEX], quant(residual, RESIDUAL_Q));
                atomicAdd(stats[RESIDUAL_HIST_OFFSET + histBin(residual, uResidualHistogramRange)], 1u);

                atomicAdd(stats[NOISE_RESIDUAL_COUNT_INDEX], 1u);
                atomicAdd(stats[NOISE_RESIDUAL_SUM_INDEX], quant(noiseResidual, NOISE_RESIDUAL_Q));
                atomicMax(stats[NOISE_RESIDUAL_MAX_INDEX], quant(noiseResidual, NOISE_RESIDUAL_Q));
                atomicAdd(
                    stats[NOISE_RESIDUAL_HIST_OFFSET + histBin(noiseResidual, uNoiseResidualHistogramRange)],
                    1u
                );

                atomicAdd(stats[FLOW_RANGE_COUNT_INDEX], 1u);
                atomicAdd(stats[FLOW_RANGE_SUM_INDEX], quant(flowRange, FLOW_Q));
                atomicMax(stats[FLOW_RANGE_MAX_INDEX], quant(flowRange, FLOW_Q));
                atomicAdd(stats[FLOW_RANGE_HIST_OFFSET + histBin(flowRange, uFlowRangeHistogramRange)], 1u);

                if (highConfidence) {
                    atomicAdd(stats[HIGH_CONFIDENCE_COUNT_INDEX], 1u);
                }
                if (srAlignmentReady) {
                    atomicAdd(stats[SR_ALIGNMENT_READY_COUNT_INDEX], 1u);
                }
                if (srDetailReady) {
                    atomicAdd(stats[SR_DETAIL_READY_COUNT_INDEX], 1u);
                }
            }
        """.trimIndent()

        private val DIAGNOSTIC_FINAL_COMPUTE_SHADER = """
            #version 310 es
            $RAW_COMMON
            layout(local_size_x = 16, local_size_y = 16) in;
            uniform sampler2D uAccumulator;
            uniform sampler2D uPostfilterAccumulator;
            uniform sampler2D uLensShadingMap;
            uniform ivec2 uImageSize;
            uniform ivec2 uAccumulatorSize;
            uniform ivec2 uSampleGridSize;
            uniform int uSampleStep;
            uniform int uCfaPattern;
            uniform float uNoiseAlphaByChannel[4];
            uniform float uNoiseBetaByChannel[4];
            uniform float uOutputScale;
            uniform float uWeightHistogramRange;
            uniform float uPostResidualHistogramRange;
            uniform float uLensShadingEdgeFraction;
            uniform float uFinalSmoothStrength;
            uniform float uFlatVarianceStart;
            uniform float uFlatVarianceEnd;
            uniform float uDetailKeepNoiseLowScale;
            uniform float uDetailKeepNoiseHighScale;
            uniform float uDetailKeepOffsetLow;
            uniform float uDetailKeepOffsetHigh;
            uniform float uDetailKeepSuppression;
            uniform float uLscNoiseGainMax;
            uniform float uPostfilterNoiseFloor;
            uniform float uLowLightSignalLow;
            uniform float uLowLightSignalHigh;
            uniform float uLowLightSmoothBoost;
            uniform float uLscSmoothGainStart;
            uniform float uLscSmoothGainEnd;
            uniform float uLscSmoothBoost;
            uniform float uLowWeightStart;
            uniform float uLowWeightEnd;
            uniform float uLowWeightSmoothBoost;
            uniform float uResidualNoiseWeight;
            uniform float uResidualNoiseLowScale;
            uniform float uResidualNoiseHighScale;
            uniform float uResidualSmoothBoost;
            uniform float uMaxSmoothStrength;
            layout(std430, binding = $DIAGNOSTIC_BUFFER_BINDING) buffer RawStackDiagnosticStats {
                uint stats[];
            };

            const int HIST_BINS = $DIAGNOSTIC_HIST_BINS;
            const int WEIGHT_COUNT_INDEX = $DIAGNOSTIC_WEIGHT_COUNT_INDEX;
            const int WEIGHT_SUM_INDEX = $DIAGNOSTIC_WEIGHT_SUM_INDEX;
            const int WEIGHT_MAX_INDEX = $DIAGNOSTIC_WEIGHT_MAX_INDEX;
            const int LSC_COUNT_INDEX = $DIAGNOSTIC_LSC_COUNT_INDEX;
            const int LSC_SUM_INDEX = $DIAGNOSTIC_LSC_SUM_INDEX;
            const int LSC_MAX_INDEX = $DIAGNOSTIC_LSC_MAX_INDEX;
            const int LSC_EDGE_COUNT_INDEX = $DIAGNOSTIC_LSC_EDGE_COUNT_INDEX;
            const int LSC_EDGE_SUM_INDEX = $DIAGNOSTIC_LSC_EDGE_SUM_INDEX;
            const int POST_RESIDUAL_COUNT_INDEX = $DIAGNOSTIC_POST_RESIDUAL_COUNT_INDEX;
            const int POST_RESIDUAL_SUM_INDEX = $DIAGNOSTIC_POST_RESIDUAL_SUM_INDEX;
            const int POST_RESIDUAL_MAX_INDEX = $DIAGNOSTIC_POST_RESIDUAL_MAX_INDEX;
            const int POST_SMOOTH_COUNT_INDEX = $DIAGNOSTIC_POST_SMOOTH_COUNT_INDEX;
            const int POST_SMOOTH_SUM_INDEX = $DIAGNOSTIC_POST_SMOOTH_SUM_INDEX;
            const int POST_SMOOTH_MAX_INDEX = $DIAGNOSTIC_POST_SMOOTH_MAX_INDEX;
            const int POST_EFFECTIVE_SMOOTH_COUNT_INDEX = $DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_COUNT_INDEX;
            const int POST_EFFECTIVE_SMOOTH_SUM_INDEX = $DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_SUM_INDEX;
            const int POST_EFFECTIVE_SMOOTH_MAX_INDEX = $DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_MAX_INDEX;
            const int POST_WIENER_COUNT_INDEX = $DIAGNOSTIC_POST_WIENER_COUNT_INDEX;
            const int POST_WIENER_SUM_INDEX = $DIAGNOSTIC_POST_WIENER_SUM_INDEX;
            const int POST_WIENER_MAX_INDEX = $DIAGNOSTIC_POST_WIENER_MAX_INDEX;
            const int POST_LSC_COUNT_INDEX = $DIAGNOSTIC_POST_LSC_COUNT_INDEX;
            const int POST_LSC_SUM_INDEX = $DIAGNOSTIC_POST_LSC_SUM_INDEX;
            const int POST_LSC_MAX_INDEX = $DIAGNOSTIC_POST_LSC_MAX_INDEX;
            const int POST_LOW_WEIGHT_COUNT_INDEX = $DIAGNOSTIC_POST_LOW_WEIGHT_COUNT_INDEX;
            const int POST_LOW_WEIGHT_SUM_INDEX = $DIAGNOSTIC_POST_LOW_WEIGHT_SUM_INDEX;
            const int POST_LOW_WEIGHT_MAX_INDEX = $DIAGNOSTIC_POST_LOW_WEIGHT_MAX_INDEX;
            const int WEIGHT_HIST_OFFSET = $DIAGNOSTIC_WEIGHT_HIST_OFFSET;
            const int POST_RESIDUAL_HIST_OFFSET = $DIAGNOSTIC_POST_RESIDUAL_HIST_OFFSET;
            const int POST_SMOOTH_HIST_OFFSET = $DIAGNOSTIC_POST_SMOOTH_HIST_OFFSET;
            const int POST_EFFECTIVE_SMOOTH_HIST_OFFSET = $DIAGNOSTIC_POST_EFFECTIVE_SMOOTH_HIST_OFFSET;
            const int POST_WIENER_HIST_OFFSET = $DIAGNOSTIC_POST_WIENER_HIST_OFFSET;
            const int POST_LSC_HIST_OFFSET = $DIAGNOSTIC_POST_LSC_HIST_OFFSET;
            const int POST_LOW_WEIGHT_HIST_OFFSET = $DIAGNOSTIC_POST_LOW_WEIGHT_HIST_OFFSET;
            const float UNIT_Q = ${DIAGNOSTIC_UNIT_QUANTIZATION};
            const float POST_RESIDUAL_Q = ${DIAGNOSTIC_POST_RESIDUAL_QUANTIZATION};
            const float WEIGHT_Q = ${DIAGNOSTIC_WEIGHT_QUANTIZATION};
            const float LSC_Q = ${DIAGNOSTIC_LSC_QUANTIZATION};

            uint quant(float value, float scale) {
                return uint(clamp(value * scale + 0.5, 0.0, 4294967040.0));
            }

            int histBin(float value, float range) {
                float normalized = clamp(value / max(range, 1e-6), 0.0, 1.0);
                return int(clamp(floor(normalized * float(HIST_BINS - 1) + 0.5), 0.0, float(HIST_BINS - 1)));
            }

            float lscGain(ivec2 samplePos) {
                vec2 uv = (vec2(samplePos) + vec2(0.5)) / vec2(uImageSize);
                vec4 gains = texture(uLensShadingMap, uv);
                int channel = lensShadingChannelAt(uCfaPattern, samplePos);
                if (channel == 0) return gains.r;
                if (channel == 1) return gains.g;
                if (channel == 2) return gains.b;
                return gains.a;
            }

            bool isEdgeSample(ivec2 p) {
                int edgeX = max(1, int(float(uImageSize.x) * uLensShadingEdgeFraction + 0.5));
                int edgeY = max(1, int(float(uImageSize.y) * uLensShadingEdgeFraction + 0.5));
                return p.x < edgeX || p.y < edgeY ||
                    p.x >= uImageSize.x - edgeX ||
                    p.y >= uImageSize.y - edgeY;
            }

            ivec2 accumulatorCoord(ivec2 samplePos) {
                vec2 mapped = (vec2(samplePos) + vec2(0.5)) * max(uOutputScale, 1.0) - vec2(0.5);
                return clamp(ivec2(round(mapped)), ivec2(0), uAccumulatorSize - ivec2(1));
            }

            float lscNoiseGain(ivec2 samplePos) {
                return clamp(lscGain(samplePos), 1e-3, max(uLscNoiseGainMax, 1.0));
            }

            float noiseVarianceForChannel(float correctedSignalNorm, int bayerIndex, float lscGainForNoise) {
                float channelAlpha = uNoiseAlphaByChannel[bayerIndex];
                float channelBeta = uNoiseBetaByChannel[bayerIndex];
                float gain = clamp(lscGainForNoise, 1e-3, max(uLscNoiseGainMax, 1.0));
                float floorVariance = max(uPostfilterNoiseFloor, 0.0) * gain * gain;
                if (channelAlpha <= 0.0 && channelBeta <= 0.0) return floorVariance;
                float modelVariance =
                    channelAlpha * clamp(correctedSignalNorm, 0.0, 1.0) * gain +
                    channelBeta * gain * gain;
                return max(modelVariance, floorVariance);
            }

            float postFusedAt(ivec2 p, float fallback) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                vec4 a = texelFetch(uPostfilterAccumulator, p, 0);
                if (a.g <= 1e-5) return fallback;
                return clamp(a.r / max(a.g, 1e-5), 0.0, 1.0);
            }

            float lowLightAmount(float signal) {
                return 1.0 - smoothstep(
                    min(uLowLightSignalLow, uLowLightSignalHigh),
                    max(uLowLightSignalLow, uLowLightSignalHigh),
                    signal
                );
            }

            float lscSmoothAmount(float lscGainForNoise) {
                return smoothstep(
                    min(uLscSmoothGainStart, uLscSmoothGainEnd),
                    max(uLscSmoothGainStart, uLscSmoothGainEnd),
                    lscGainForNoise
                );
            }

            float lowWeightAmount(float accumulatorWeight) {
                return 1.0 - smoothstep(
                    min(uLowWeightStart, uLowWeightEnd),
                    max(uLowWeightStart, uLowWeightEnd),
                    accumulatorWeight
                );
            }

            float accumulatorResidualNoise(vec4 accumulator, float fused) {
                if (accumulator.g <= 1e-5) return 0.0;
                float sampleMean2 = max(accumulator.b / max(accumulator.g, 1e-5), 0.0);
                float temporalVariance = max(sampleMean2 - fused * fused, 0.0);
                return temporalVariance / max(accumulator.g, 1.0);
            }

            float residualNoiseSigmaRatio(float residualNoise, float modelNoise) {
                return sqrt(max(residualNoise, 0.0) / max(modelNoise, 1e-10));
            }

            float residualNoiseAmount(float residualSigmaRatio) {
                float low = max(uResidualNoiseLowScale, 0.0);
                float high = max(uResidualNoiseHighScale, uResidualNoiseLowScale + 1e-4);
                return smoothstep(low, high, residualSigmaRatio);
            }

            float finalSmoothAmount(
                float variance,
                float noise,
                float detailDeviation,
                float signal,
                float lscGainForNoise,
                float accumulatorWeight,
                float residualAmount
            ) {
                float flatness = 1.0 - smoothstep(uFlatVarianceStart, uFlatVarianceEnd, variance);
                float noiseStd = sqrt(max(noise, 1e-10));
                float detailKeep = smoothstep(
                    uDetailKeepNoiseLowScale * noiseStd + uDetailKeepOffsetLow,
                    uDetailKeepNoiseHighScale * noiseStd + uDetailKeepOffsetHigh,
                    detailDeviation
                );
                float base = uFinalSmoothStrength * flatness * (1.0 - uDetailKeepSuppression * detailKeep);
                float lowLight = lowLightAmount(signal);
                float lsc = lscSmoothAmount(lscGainForNoise);
                float lowWeight = lowWeightAmount(accumulatorWeight);
                float residualNeed = residualAmount * max(max(lowLight, lsc), lowWeight);
                float boost =
                    1.0 +
                    uLowLightSmoothBoost * lowLight +
                    uLscSmoothBoost * lsc +
                    uLowWeightSmoothBoost * lowWeight * max(lowLight, lsc) +
                    uResidualSmoothBoost * residualNeed;
                return min(max(uMaxSmoothStrength, 0.0), base * boost);
            }

            void main() {
                ivec2 sampleIndex = ivec2(gl_GlobalInvocationID.xy);
                if (sampleIndex.x >= uSampleGridSize.x || sampleIndex.y >= uSampleGridSize.y) return;
                ivec2 raw = min(sampleIndex * uSampleStep, uImageSize - ivec2(1));

                float weight = max(texelFetch(uAccumulator, accumulatorCoord(raw), 0).g, 0.0);
                atomicAdd(stats[WEIGHT_COUNT_INDEX], 1u);
                atomicAdd(stats[WEIGHT_SUM_INDEX], quant(weight, WEIGHT_Q));
                atomicMax(stats[WEIGHT_MAX_INDEX], quant(weight, WEIGHT_Q));
                atomicAdd(stats[WEIGHT_HIST_OFFSET + histBin(weight, uWeightHistogramRange)], 1u);

                float gain = max(lscGain(raw), 0.0);
                atomicAdd(stats[LSC_COUNT_INDEX], 1u);
                atomicAdd(stats[LSC_SUM_INDEX], quant(gain, LSC_Q));
                atomicMax(stats[LSC_MAX_INDEX], quant(gain, LSC_Q));
                if (isEdgeSample(raw)) {
                    atomicAdd(stats[LSC_EDGE_COUNT_INDEX], 1u);
                    atomicAdd(stats[LSC_EDGE_SUM_INDEX], quant(gain, LSC_Q));
                }

                vec4 post = texelFetch(uPostfilterAccumulator, raw, 0);
                if (post.g <= 1e-5) return;

                float fused = clamp(post.r / max(post.g, 1e-5), 0.0, 1.0);
                float mean = 0.0;
                float mean2 = 0.0;
                float count = 0.0;
                int period = cfaPeriod(uCfaPattern);
                for (int y = -2; y <= 2; ++y) {
                    for (int x = -2; x <= 2; ++x) {
                        ivec2 q = raw + ivec2(x * period, y * period);
                        if (q.x >= 0 && q.y >= 0 && q.x < uImageSize.x && q.y < uImageSize.y) {
                            float v = postFusedAt(q, fused);
                            mean += v;
                            mean2 += v * v;
                            count += 1.0;
                        }
                    }
                }
                mean /= max(count, 1.0);
                mean2 /= max(count, 1.0);

                int bayerIndex = bayerIndexAt(uCfaPattern, raw);
                float variance = max(mean2 - mean * mean, 0.0);
                float sampleLscNoiseGain = lscNoiseGain(raw);
                float modelNoise = noiseVarianceForChannel(fused, bayerIndex, sampleLscNoiseGain) / max(post.g, 1.0);
                float residualNoise = accumulatorResidualNoise(post, fused) * max(uResidualNoiseWeight, 0.0);
                float residualSigmaRatio = residualNoiseSigmaRatio(residualNoise, modelNoise);
                float residualAmount = residualNoiseAmount(residualSigmaRatio);
                float residualNoiseForStructure = min(
                    residualNoise,
                    modelNoise * max(uResidualNoiseHighScale * uResidualNoiseHighScale, 1.0)
                );
                float noise = max(modelNoise, residualNoiseForStructure);
                float structureVariance = max(variance - noise, 0.0);
                float wienerGain = structureVariance / max(variance, 1e-6);
                float detailDeviation = max(abs(fused - mean) - sqrt(max(noise, 1e-10)), 0.0);
                float postSmoothAmount = finalSmoothAmount(
                    structureVariance,
                    noise,
                    detailDeviation,
                    fused,
                    sampleLscNoiseGain,
                    post.g,
                    residualAmount
                );
                float residualMetric = clamp(residualSigmaRatio, 0.0, uPostResidualHistogramRange);
                atomicAdd(stats[POST_RESIDUAL_COUNT_INDEX], 1u);
                atomicAdd(stats[POST_RESIDUAL_SUM_INDEX], quant(residualMetric, POST_RESIDUAL_Q));
                atomicMax(stats[POST_RESIDUAL_MAX_INDEX], quant(residualMetric, POST_RESIDUAL_Q));
                atomicAdd(
                    stats[POST_RESIDUAL_HIST_OFFSET + histBin(residualMetric, uPostResidualHistogramRange)],
                    1u
                );

                float smoothMetric = clamp(postSmoothAmount, 0.0, 1.0);
                atomicAdd(stats[POST_SMOOTH_COUNT_INDEX], 1u);
                atomicAdd(stats[POST_SMOOTH_SUM_INDEX], quant(smoothMetric, UNIT_Q));
                atomicMax(stats[POST_SMOOTH_MAX_INDEX], quant(smoothMetric, UNIT_Q));
                atomicAdd(stats[POST_SMOOTH_HIST_OFFSET + histBin(smoothMetric, 1.0)], 1u);

                float effectiveSmoothMetric = clamp(postSmoothAmount * (1.0 - wienerGain), 0.0, 1.0);
                atomicAdd(stats[POST_EFFECTIVE_SMOOTH_COUNT_INDEX], 1u);
                atomicAdd(stats[POST_EFFECTIVE_SMOOTH_SUM_INDEX], quant(effectiveSmoothMetric, UNIT_Q));
                atomicMax(stats[POST_EFFECTIVE_SMOOTH_MAX_INDEX], quant(effectiveSmoothMetric, UNIT_Q));
                atomicAdd(
                    stats[POST_EFFECTIVE_SMOOTH_HIST_OFFSET + histBin(effectiveSmoothMetric, 1.0)],
                    1u
                );

                float wienerMetric = clamp(wienerGain, 0.0, 1.0);
                atomicAdd(stats[POST_WIENER_COUNT_INDEX], 1u);
                atomicAdd(stats[POST_WIENER_SUM_INDEX], quant(wienerMetric, UNIT_Q));
                atomicMax(stats[POST_WIENER_MAX_INDEX], quant(wienerMetric, UNIT_Q));
                atomicAdd(stats[POST_WIENER_HIST_OFFSET + histBin(wienerMetric, 1.0)], 1u);

                float lscMetric = clamp(lscSmoothAmount(sampleLscNoiseGain), 0.0, 1.0);
                atomicAdd(stats[POST_LSC_COUNT_INDEX], 1u);
                atomicAdd(stats[POST_LSC_SUM_INDEX], quant(lscMetric, UNIT_Q));
                atomicMax(stats[POST_LSC_MAX_INDEX], quant(lscMetric, UNIT_Q));
                atomicAdd(stats[POST_LSC_HIST_OFFSET + histBin(lscMetric, 1.0)], 1u);

                float lowWeightMetric = clamp(lowWeightAmount(post.g), 0.0, 1.0);
                atomicAdd(stats[POST_LOW_WEIGHT_COUNT_INDEX], 1u);
                atomicAdd(stats[POST_LOW_WEIGHT_SUM_INDEX], quant(lowWeightMetric, UNIT_Q));
                atomicMax(stats[POST_LOW_WEIGHT_MAX_INDEX], quant(lowWeightMetric, UNIT_Q));
                atomicAdd(stats[POST_LOW_WEIGHT_HIST_OFFSET + histBin(lowWeightMetric, 1.0)], 1u);
            }
        """.trimIndent()

        private val NORMALIZE_FRAGMENT_SHADER = """
            #version 300 es
            $RAW_COMMON
            in vec2 vTexCoord;
            out vec4 fragColor;
            uniform sampler2D uAccumulator;
            uniform highp usampler2D uReferenceRaw;
            uniform highp usampler2D uShortRaw;
            uniform sampler2D uLensShadingMap;
            uniform sampler2D uShortGlobalAlignment;
            uniform sampler2D uHdrRecoveryMask;
            uniform sampler2D uHdrRecoveryOrigMask;
            uniform ivec2 uImageSize;
            uniform ivec2 uPlaneSize;
            uniform int uCfaPattern;
            uniform float uBlackLevel[4];
            uniform float uWhiteLevel;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            uniform float uNoiseAlphaByChannel[4];
            uniform float uNoiseBetaByChannel[4];
            uniform int uHdrMode;
            uniform float uReferenceExposureScale;
            uniform float uShortExposureScale;
            uniform float uFinalSmoothStrength;
            uniform float uFlatVarianceStart;
            uniform float uFlatVarianceEnd;
            uniform float uDetailKeepNoiseLowScale;
            uniform float uDetailKeepNoiseHighScale;
            uniform float uDetailKeepOffsetLow;
            uniform float uDetailKeepOffsetHigh;
            uniform float uDetailKeepSuppression;
            uniform float uLscNoiseGainMax;
            uniform float uPostfilterNoiseFloor;
            uniform float uLowLightSignalLow;
            uniform float uLowLightSignalHigh;
            uniform float uLowLightSmoothBoost;
            uniform float uLscSmoothGainStart;
            uniform float uLscSmoothGainEnd;
            uniform float uLscSmoothBoost;
            uniform float uLowWeightStart;
            uniform float uLowWeightEnd;
            uniform float uLowWeightSmoothBoost;
            uniform float uResidualNoiseWeight;
            uniform float uResidualNoiseLowScale;
            uniform float uResidualNoiseHighScale;
            uniform float uResidualSmoothBoost;
            uniform float uMaxSmoothStrength;
            uniform float uHdrRecoverySmoothSuppression;

            float lscGain(ivec2 samplePos) {
                vec2 uv = (vec2(samplePos) + vec2(0.5)) / vec2(uImageSize);
                vec4 gains = texture(uLensShadingMap, uv);
                int channel = lensShadingChannelAt(uCfaPattern, samplePos);
                if (channel == 0) return gains.r;
                if (channel == 1) return gains.g;
                if (channel == 2) return gains.b;
                return gains.a;
            }

            float lscNoiseGain(ivec2 samplePos) {
                return clamp(lscGain(samplePos), 1e-3, max(uLscNoiseGainMax, 1.0));
            }

            float noiseVarianceForChannel(float correctedSignalNorm, int bayerIndex, float lscGainForNoise) {
                float channelAlpha = uNoiseAlphaByChannel[bayerIndex];
                float channelBeta = uNoiseBetaByChannel[bayerIndex];
                float gain = clamp(lscGainForNoise, 1e-3, max(uLscNoiseGainMax, 1.0));
                float floorVariance = max(uPostfilterNoiseFloor, 0.0) * gain * gain;
                if (channelAlpha <= 0.0 && channelBeta <= 0.0) return floorVariance;
                float modelVariance =
                    channelAlpha * clamp(correctedSignalNorm, 0.0, 1.0) * gain +
                    channelBeta * gain * gain;
                return max(modelVariance, floorVariance);
            }

            float referenceNorm(ivec2 p, int bayerIndex) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                float raw = float(texelFetch(uReferenceRaw, p, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) * lscGain(p) / range, 0.0, 1.0);
            }

            float referenceOutputNorm(ivec2 p, int bayerIndex) {
                return clamp(referenceNorm(p, bayerIndex) * uReferenceExposureScale, 0.0, 1.0);
            }

            float referenceSensorNorm(ivec2 p) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                float raw = float(texelFetch(uReferenceRaw, p, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) / range, 0.0, 1.0);
            }

            vec2 shortAlignedRawPos(ivec2 p) {
                vec2 rawOffset = texelFetch(uShortGlobalAlignment, ivec2(0), 0).rg * 2.0;
                vec2 rawPos = vec2(float(p.x), float(p.y)) + rawOffset;
                vec2 maxRawPos = vec2(float(uImageSize.x - 1), float(uImageSize.y - 1));
                return clamp(rawPos, vec2(0.0), maxRawPos);
            }

            ivec2 cfaPhaseOffset(int cfaPattern, ivec2 p) {
                int period = cfaPeriod(cfaPattern);
                return ivec2(p.x % period, p.y % period);
            }

            ivec2 maxLatticeForPhase(ivec2 phaseOffset, int period) {
                return max((uImageSize - ivec2(1) - phaseOffset) / ivec2(period), ivec2(0));
            }

            float shortRawNormAt(ivec2 samplePos, int bayerIndex) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                float raw = float(texelFetch(uShortRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) * lscGain(samplePos) / range, 0.0, 1.0);
            }

            float fetchShortSamePhase(ivec2 lattice, ivec2 phaseOffset, int period, int bayerIndex) {
                lattice = clamp(lattice, ivec2(0), maxLatticeForPhase(phaseOffset, period));
                return shortRawNormAt(phaseOffset + lattice * period, bayerIndex);
            }

            float sampleShortSamePhase(vec2 rawPos, ivec2 phaseOffset, int period, int bayerIndex) {
                ivec2 maxLattice = maxLatticeForPhase(phaseOffset, period);
                vec2 phase = vec2(float(phaseOffset.x), float(phaseOffset.y));
                vec2 pos = clamp(
                    (rawPos - phase) / float(period),
                    vec2(0.0),
                    vec2(float(maxLattice.x), float(maxLattice.y))
                );
                ivec2 p0 = ivec2(floor(pos));
                ivec2 p1 = min(p0 + ivec2(1), maxLattice);
                vec2 f = pos - vec2(p0);
                float v00 = fetchShortSamePhase(p0, phaseOffset, period, bayerIndex);
                float v10 = fetchShortSamePhase(ivec2(p1.x, p0.y), phaseOffset, period, bayerIndex);
                float v01 = fetchShortSamePhase(ivec2(p0.x, p1.y), phaseOffset, period, bayerIndex);
                float v11 = fetchShortSamePhase(p1, phaseOffset, period, bayerIndex);
                return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
            }

            float shortRecoveryNorm(ivec2 p, int bayerIndex) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                int period = cfaPeriod(uCfaPattern);
                float norm = sampleShortSamePhase(
                    shortAlignedRawPos(p),
                    cfaPhaseOffset(uCfaPattern, p),
                    period,
                    bayerIndex
                );
                return clamp(norm * uShortExposureScale, 0.0, 1.0);
            }

            float hdrRecoveryMaskAt(ivec2 p) {
                ivec2 block = clamp(p / 2, ivec2(0), uPlaneSize - ivec2(1));
                return clamp(texelFetch(uHdrRecoveryMask, block, 0).r, 0.0, 1.0);
            }

            vec2 hdrRecoveryOrigMaskAt(ivec2 p) {
                ivec2 block = clamp(p / 2, ivec2(0), uPlaneSize - ivec2(1));
                return clamp(texelFetch(uHdrRecoveryOrigMask, block, 0).rg, vec2(0.0), vec2(1.0));
            }

            vec4 readAccum(ivec2 p) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                return texelFetch(uAccumulator, p, 0);
            }

            float meanAt(ivec2 p) {
                int b = bayerIndexAt(uCfaPattern, p);
                vec4 a = readAccum(p);
                if (a.g <= (uHdrMode != 0 ? 0.02 : 1e-5)) return referenceOutputNorm(p, b);
                return clamp(a.r / max(a.g, 1e-5), 0.0, 1.0);
            }

            float lowLightAmount(float signal) {
                return 1.0 - smoothstep(
                    min(uLowLightSignalLow, uLowLightSignalHigh),
                    max(uLowLightSignalLow, uLowLightSignalHigh),
                    signal
                );
            }

            float lscSmoothAmount(float lscGainForNoise) {
                return smoothstep(
                    min(uLscSmoothGainStart, uLscSmoothGainEnd),
                    max(uLscSmoothGainStart, uLscSmoothGainEnd),
                    lscGainForNoise
                );
            }

            float lowWeightAmount(float accumulatorWeight) {
                return 1.0 - smoothstep(
                    min(uLowWeightStart, uLowWeightEnd),
                    max(uLowWeightStart, uLowWeightEnd),
                    accumulatorWeight
                );
            }

            float accumulatorResidualNoise(vec4 accumulator, float fused) {
                if (accumulator.g <= 1e-5) return 0.0;
                float sampleMean2 = max(accumulator.b / max(accumulator.g, 1e-5), 0.0);
                float temporalVariance = max(sampleMean2 - fused * fused, 0.0);
                return temporalVariance / max(accumulator.g, 1.0);
            }

            float residualNoiseSigmaRatio(float residualNoise, float modelNoise) {
                return sqrt(max(residualNoise, 0.0) / max(modelNoise, 1e-10));
            }

            float residualNoiseAmount(float residualSigmaRatio) {
                float low = max(uResidualNoiseLowScale, 0.0);
                float high = max(uResidualNoiseHighScale, uResidualNoiseLowScale + 1e-4);
                return smoothstep(low, high, residualSigmaRatio);
            }

            float finalSmoothAmount(
                float variance,
                float noise,
                float detailDeviation,
                float signal,
                float lscGainForNoise,
                float accumulatorWeight,
                float residualAmount
            ) {
                float flatness = 1.0 - smoothstep(uFlatVarianceStart, uFlatVarianceEnd, variance);
                float noiseStd = sqrt(max(noise, 1e-10));
                float detailKeep = smoothstep(
                    uDetailKeepNoiseLowScale * noiseStd + uDetailKeepOffsetLow,
                    uDetailKeepNoiseHighScale * noiseStd + uDetailKeepOffsetHigh,
                    detailDeviation
                );
                float base = uFinalSmoothStrength * flatness * (1.0 - uDetailKeepSuppression * detailKeep);
                if (uHdrMode != 0) return base;

                float lowLight = lowLightAmount(signal);
                float lsc = lscSmoothAmount(lscGainForNoise);
                float lowWeight = lowWeightAmount(accumulatorWeight);
                float residualNeed = residualAmount * max(max(lowLight, lsc), lowWeight);
                float boost =
                    1.0 +
                    uLowLightSmoothBoost * lowLight +
                    uLscSmoothBoost * lsc +
                    uLowWeightSmoothBoost * lowWeight * max(lowLight, lsc) +
                    uResidualSmoothBoost * residualNeed;
                return min(max(uMaxSmoothStrength, 0.0), base * boost);
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                vec4 a = readAccum(p);
                float reference = referenceOutputNorm(p, bayerIndex);
                float fused = a.g > 1e-5 ? clamp(a.r / max(a.g, 1e-5), 0.0, 1.0) : reference;
                float recoveryMix = 0.0;
                if (uHdrMode != 0) {
                    float normalConfidence = smoothstep(0.04, 0.35, a.g);
                    recoveryMix = hdrRecoveryMaskAt(p);
                    vec2 origRecovery = hdrRecoveryOrigMaskAt(p);
                    if (origRecovery.r > 0.5) {
                        recoveryMix += (1.0 - recoveryMix) * origRecovery.g;
                    }
                    recoveryMix = clamp(recoveryMix, 0.0, 1.0);
                    fused = mix(reference, fused, normalConfidence);
                    if (recoveryMix > 0.0001) {
                        float shortValue = shortRecoveryNorm(p, bayerIndex);
                        float shortPresenceFloor = max(0.012, reference * 0.25);
                        float shortPresenceWidth = max(0.035, reference * 0.45);
                        float shortPresence = smoothstep(
                            shortPresenceFloor,
                            shortPresenceFloor + shortPresenceWidth,
                            shortValue
                        );
                        recoveryMix *= shortPresence;
                        fused = mix(fused, shortValue, recoveryMix);
                    }
                }

                float mean = 0.0;
                float mean2 = 0.0;
                float count = 0.0;
                int period = cfaPeriod(uCfaPattern);
                if (uHdrMode != 0) {
                    for (int y = -1; y <= 1; ++y) {
                        for (int x = -1; x <= 1; ++x) {
                            ivec2 q = p + ivec2(x * period, y * period);
                            if (q.x >= 0 && q.y >= 0 && q.x < uImageSize.x && q.y < uImageSize.y) {
                                float v = meanAt(q);
                                mean += v;
                                mean2 += v * v;
                                count += 1.0;
                            }
                        }
                    }
                } else {
                    for (int y = -2; y <= 2; ++y) {
                        for (int x = -2; x <= 2; ++x) {
                            ivec2 q = p + ivec2(x * period, y * period);
                            if (q.x >= 0 && q.y >= 0 && q.x < uImageSize.x && q.y < uImageSize.y) {
                                float v = meanAt(q);
                                mean += v;
                                mean2 += v * v;
                                count += 1.0;
                            }
                        }
                    }
                }
                mean /= max(count, 1.0);
                mean2 /= max(count, 1.0);
                float variance = max(mean2 - mean * mean, 0.0);
                float sampleLscNoiseGain = lscNoiseGain(p);
                float modelNoise = noiseVarianceForChannel(fused, bayerIndex, sampleLscNoiseGain) / max(a.g, 1.0);
                float residualNoise = accumulatorResidualNoise(a, fused) * max(uResidualNoiseWeight, 0.0);
                float residualSigmaRatio = residualNoiseSigmaRatio(residualNoise, modelNoise);
                float residualAmount = residualNoiseAmount(residualSigmaRatio);
                float residualNoiseForStructure = min(
                    residualNoise,
                    modelNoise * max(uResidualNoiseHighScale * uResidualNoiseHighScale, 1.0)
                );
                float noise = max(modelNoise, residualNoiseForStructure);
                float structureVariance = max(variance - noise, 0.0);
                float wienerGain = structureVariance / max(variance, 1e-6);
                float detailDeviation = max(abs(fused - mean) - sqrt(max(noise, 1e-10)), 0.0);
                float smoothAmount = finalSmoothAmount(
                    structureVariance,
                    noise,
                    detailDeviation,
                    max(fused, reference),
                    sampleLscNoiseGain,
                    a.g,
                    residualAmount
                );
                if (uHdrMode != 0) {
                    smoothAmount *= 1.0 - uHdrRecoverySmoothSuppression * recoveryMix;
                }
                fused = mix(fused, mean + wienerGain * (fused - mean), smoothAmount);
                fused = clamp(fused, 0.0, 1.0);

                uint raw = uint(floor(fused * 65535.0 + 0.5));
                uint lo = raw & 255u;
                uint hi = (raw >> 8) & 255u;
                fragColor = vec4(float(lo) / 255.0, float(hi) / 255.0, 0.0, 1.0);
            }
        """.trimIndent()
    }
}
