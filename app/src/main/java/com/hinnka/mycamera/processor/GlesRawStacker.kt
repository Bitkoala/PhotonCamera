package com.hinnka.mycamera.processor

import android.graphics.ImageFormat
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLES31
import com.hinnka.mycamera.model.SafeImage
import com.hinnka.mycamera.utils.LargeDirectBuffer
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

class GlesRawStacker(
    private val width: Int,
    private val height: Int,
    private val cfaPattern: Int,
    blackLevel: FloatArray,
    whiteLevel: Int,
    noiseModel: FloatArray,
    private val lensShading: FloatArray?,
    private val lensShadingWidth: Int,
    private val lensShadingHeight: Int,
) {
    private data class TextureLevel(val texture: Int, val width: Int, val height: Int)

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE

    private val textures = ArrayList<Int>()
    private val programs = ArrayList<Int>()
    private val framebuffers = ArrayList<Int>()

    private val normalizedBlackLevel = FloatArray(4) { index ->
        blackLevel.getOrElse(index) { blackLevel.firstOrNull() ?: 0f }
    }
    private val normalizedWhiteLevel = whiteLevel.coerceAtLeast(1).toFloat()
    private val noiseAlpha = noiseModel.getOrElse(0) { 0f }.coerceAtLeast(0f) / 65535.0f
    private val noiseBeta = noiseModel.getOrElse(1) { 0f }.coerceAtLeast(0f) / (65535.0f * 65535.0f)

    private val planeWidth = max(1, width / 2)
    private val planeHeight = max(1, height / 2)
    private var gridWidth = 0
    private var gridHeight = 0

    private var proxyProgram = 0
    private var downsampleProgram = 0
    private var alignProgram = 0
    private var lkRefineProgram = 0
    private var smoothFlowProgram = 0
    private var structureProgram = 0
    private var robustnessProgram = 0
    private var tileMaskProgram = 0
    private var clearAccumulatorProgram = 0
    private var accumulateProgram = 0
    private var normalizeProgram = 0

    private var renderFbo = 0
    private var readbackFbo = 0

    private var refRaw = 0
    private var curRaw = 0
    private var refProxy = 0
    private var curProxy = 0
    private var flowTexture = 0
    private var flowScratchTexture = 0
    private var kernelTexture = 0
    private var robustnessTexture = 0
    private var tileMaskTexture = 0
    private var accumulatorTexture = 0
    private var accumulatorScratchTexture = 0
    private var currentAccumulatorTexture = 0
    private var outputTexture = 0
    private var lensShadingTexture = 0

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

        val outputByteCount = width.toLong() * height.toLong() * 2L
        var outputBuffer: ByteBuffer? = null
        var returned = false
        val startTime = System.currentTimeMillis()
        return try {
            outputBuffer = LargeDirectBuffer.allocate(outputByteCount, "GLES RAW fused Bayer")
                ?.order(ByteOrder.nativeOrder()) ?: return null

            initEgl()
            ensureGles31()
            initPrograms()
            initResources()
            PLog.d(
                TAG,
                "GLES RAW stack frames=${images.size} size=${width}x$height plane=${planeWidth}x$planeHeight grid=${gridWidth}x$gridHeight"
            )

            images[0].use {
                uploadRawTexture(it, refRaw, "reference")
            }
            buildProxy(refRaw, refProxy, "reference")
            val refPyramid = createPyramid(refProxy)
            val curPyramid = createPyramid(curProxy)
            buildPyramid(refPyramid)
            computeStructureTensor()
            clearAccumulator()
            accumulateFrame(refRaw, isReference = true)

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
                accumulateFrame(curRaw, isReference = false)
            }

            normalizeOutput()
            readOutput(outputBuffer)
            outputBuffer.rewind()
            returned = true
            PLog.i(TAG, "GLES RAW stacking completed in ${System.currentTimeMillis() - startTime}ms")
            RawStackResult(
                fusedBayerBuffer = outputBuffer,
                width = width,
                height = height,
                isNormalizedSensorData = true,
                blackLevel = normalizedBlackLevel.copyOf(),
                fusedBayerUsesNativeAllocator = true,
            )
        } catch (e: Exception) {
            PLog.e(TAG, "GLES RAW stacking failed", e)
            null
        } finally {
            images.forEach { it.close() }
            release()
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
        eglContext = EGL14.eglCreateContext(
            eglDisplay,
            config,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION,
                3,
                EGL_CONTEXT_MINOR_VERSION_KHR,
                1,
                EGL14.EGL_NONE,
            ),
            0,
        )
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            eglContext = EGL14.eglCreateContext(
                eglDisplay,
                config,
                EGL14.EGL_NO_CONTEXT,
                intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE),
                0,
            )
        }
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
        clearAccumulatorProgram = linkComputeProgram(CLEAR_ACCUMULATOR_COMPUTE_SHADER, "raw_clear_accumulator")
        accumulateProgram = linkComputeProgram(ACCUMULATE_COMPUTE_SHADER, "raw_accumulate")
        normalizeProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, NORMALIZE_FRAGMENT_SHADER, "raw_normalize")
    }

    private fun initResources() {
        gridWidth = (planeWidth + FLOW_GRID_SPACING - 1) / FLOW_GRID_SPACING
        gridHeight = (planeHeight + FLOW_GRID_SPACING - 1) / FLOW_GRID_SPACING

        refRaw = createTexture2D(width, height, GLES30.GL_R16UI, GLES30.GL_NEAREST)
        curRaw = createTexture2D(width, height, GLES30.GL_R16UI, GLES30.GL_NEAREST)
        refProxy = createTexture2D(planeWidth, planeHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        curProxy = createTexture2D(planeWidth, planeHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        flowTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        flowScratchTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        kernelTexture = createTexture2D(planeWidth, planeHeight, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        robustnessTexture = createTexture2D(planeWidth, planeHeight, GLES30.GL_R32F, GLES30.GL_NEAREST)
        tileMaskTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_R16F, GLES30.GL_LINEAR)
        accumulatorTexture = createTexture2D(width, height, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        accumulatorScratchTexture = createTexture2D(width, height, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        currentAccumulatorTexture = accumulatorTexture
        outputTexture = createTexture2D(width, height, GLES30.GL_RGBA8, GLES30.GL_NEAREST)
        lensShadingTexture = createLensShadingTexture()
        renderFbo = createFramebuffer()
        readbackFbo = createFramebuffer()
    }

    private fun createPyramid(baseTexture: Int): List<TextureLevel> {
        val levels = ArrayList<TextureLevel>(PYRAMID_LEVELS)
        levels += TextureLevel(baseTexture, planeWidth, planeHeight)
        var levelWidth = planeWidth
        var levelHeight = planeHeight
        repeat(PYRAMID_LEVELS - 1) {
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

    private fun buildProxy(rawTexture: Int, proxyTexture: Int, label: String) {
        GLES31.glUseProgram(proxyProgram)
        bindTexture(proxyProgram, "uRaw", 0, rawTexture)
        bindImage(1, proxyTexture, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
        setCommonUniforms(proxyProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(proxyProgram, "uProxySize"), planeWidth, planeHeight)
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
        GLES31.glDispatchCompute(groupCount(planeWidth), groupCount(planeHeight), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("computeStructureTensor")
    }

    private fun alignCurrentToReference(reference: List<TextureLevel>, current: List<TextureLevel>) {
        val levelIndex = ALIGN_LEVEL.coerceAtMost(reference.lastIndex).coerceAtMost(current.lastIndex)
        val ref = reference[levelIndex]
        val cur = current[levelIndex]
        bindFramebufferOutput(flowTexture, "alignCurrentToReference")
        GLES30.glViewport(0, 0, gridWidth, gridHeight)
        GLES30.glUseProgram(alignProgram)
        bindTexture(alignProgram, "uReference", 0, ref.texture)
        bindTexture(alignProgram, "uCurrent", 1, cur.texture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(alignProgram, "uLevelSize"), ref.width, ref.height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(alignProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uTileSize"), FLOW_GRID_SPACING)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uAlignWindowSize"), ALIGN_WINDOW_SIZE)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uLevelScale"), 1 shl levelIndex)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uSearchRadius"), SEARCH_RADIUS_LEVEL)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(alignProgram, "uSampleStep"), ALIGN_SAMPLE_STEP)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("alignCurrentToReference")
    }

    private fun refineFlow() {
        repeat(LK_REFINE_PASSES) { pass ->
            val input = if (pass % 2 == 0) flowTexture else flowScratchTexture
            val output = if (pass % 2 == 0) flowScratchTexture else flowTexture
            GLES31.glUseProgram(lkRefineProgram)
            bindTexture(lkRefineProgram, "uReference", 0, refProxy)
            bindTexture(lkRefineProgram, "uCurrent", 1, curProxy)
            bindTexture(lkRefineProgram, "uInputFlow", 2, input)
            bindImage(3, output, GLES31.GL_WRITE_ONLY, GLES30.GL_RGBA16F)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(lkRefineProgram, "uPlaneSize"), planeWidth, planeHeight)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(lkRefineProgram, "uGridSize"), gridWidth, gridHeight)
            GLES31.glUniform1i(GLES31.glGetUniformLocation(lkRefineProgram, "uTileSize"), FLOW_GRID_SPACING)
            GLES31.glDispatchCompute(groupCount(gridWidth), groupCount(gridHeight), 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
            checkGlError("refineFlow pass $pass")
        }
        if (LK_REFINE_PASSES % 2 != 0) {
            copyFlow(flowScratchTexture, flowTexture, "refineFlow copy")
        }
    }

    private fun smoothFlow() {
        repeat(FLOW_SMOOTH_PASSES) { pass ->
            val input = if (pass % 2 == 0) flowTexture else flowScratchTexture
            val output = if (pass % 2 == 0) flowScratchTexture else flowTexture
            bindFramebufferOutput(output, "smoothFlow pass $pass")
            GLES30.glViewport(0, 0, gridWidth, gridHeight)
            GLES30.glUseProgram(smoothFlowProgram)
            bindTexture(smoothFlowProgram, "uInputFlow", 0, input)
            GLES31.glUniform2i(GLES31.glGetUniformLocation(smoothFlowProgram, "uGridSize"), gridWidth, gridHeight)
            GLES31.glUniform1f(GLES31.glGetUniformLocation(smoothFlowProgram, "uOutlierThreshold"), FLOW_OUTLIER_THRESHOLD_PX)
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
        GLES31.glUniform1i(GLES31.glGetUniformLocation(robustnessProgram, "uTileSize"), FLOW_GRID_SPACING)
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
        GLES31.glUniform1i(GLES31.glGetUniformLocation(tileMaskProgram, "uTileSize"), FLOW_GRID_SPACING)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("computeTileMask")
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

    private fun accumulateFrame(rawTexture: Int, isReference: Boolean) {
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
        GLES31.glUniform1i(GLES31.glGetUniformLocation(accumulateProgram, "uTileSize"), FLOW_GRID_SPACING)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(accumulateProgram, "uIsReference"), if (isReference) 1 else 0)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateProgram, "uFrameWeight"),
            if (isReference) 1.0f else NON_REFERENCE_FRAME_WEIGHT,
        )
        GLES31.glDispatchCompute(groupCount(width), groupCount(height), 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError("accumulateFrame")
        currentAccumulatorTexture = outputAccumulator
    }

    private fun normalizeOutput() {
        bindFramebufferOutput(outputTexture, "normalizeOutput")
        GLES30.glViewport(0, 0, width, height)
        GLES30.glUseProgram(normalizeProgram)
        bindTexture(normalizeProgram, "uAccumulator", 0, currentAccumulatorTexture)
        bindTexture(normalizeProgram, "uReferenceRaw", 1, refRaw)
        bindTexture(normalizeProgram, "uLensShadingMap", 2, lensShadingTexture)
        setCommonUniforms(normalizeProgram)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(normalizeProgram, "uImageSize"), width, height)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("normalizeOutput")
    }

    private fun readOutput(outputBuffer: ByteBuffer) {
        val packed = LargeDirectBuffer.allocate(width.toLong() * height.toLong() * 4L, "GLES RAW packed readback")
            ?: throw IllegalStateException("Failed to allocate GLES RAW packed readback")
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
            GLES30.glViewport(0, 0, width, height)
            packed.clear()
            GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, packed)
            checkGlError("readOutput")
            packed.rewind()
            outputBuffer.clear()
            val outShorts = outputBuffer.order(ByteOrder.nativeOrder()).asShortBuffer()
            val pixelCount = width * height
            for (index in 0 until pixelCount) {
                val lo = packed.get(index * 4).toInt() and 0xFF
                val hi = packed.get(index * 4 + 1).toInt() and 0xFF
                outShorts.put(index, ((hi shl 8) or lo).toShort())
            }
            outputBuffer.rewind()
        } finally {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
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
        return texture
    }

    private fun createLensShadingTexture(): Int {
        val texWidth = if (lensShading != null && lensShadingWidth > 0) lensShadingWidth else 1
        val texHeight = if (lensShading != null && lensShadingHeight > 0) lensShadingHeight else 1
        val gainCount = texWidth * texHeight * 4
        val gains = FloatArray(gainCount) { 1.0f }
        if (lensShading != null) {
            val copyCount = minOf(lensShading.size, gains.size)
            System.arraycopy(lensShading, 0, gains, 0, copyCount)
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

    private fun groupCount(value: Int): Int = (value + LOCAL_SIZE - 1) / LOCAL_SIZE

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
        private const val EGL_CONTEXT_MINOR_VERSION_KHR = 0x30FB
        private const val LOCAL_SIZE = 16
        private const val PYRAMID_LEVELS = 4
        private const val ALIGN_LEVEL = 2
        private const val FLOW_GRID_SPACING = 8
        private const val ALIGN_WINDOW_SIZE = 32
        private const val SEARCH_RADIUS_LEVEL = 6
        private const val ALIGN_SAMPLE_STEP = 2
        private const val LK_REFINE_PASSES = 2
        private const val FLOW_SMOOTH_PASSES = 2
        private const val FLOW_OUTLIER_THRESHOLD_PX = 24.0f
        private const val NON_REFERENCE_FRAME_WEIGHT = 0.92f

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

            int bayerIndexAt(int cfaPattern, ivec2 p) {
                int phase = (p.y & 1) * 2 + (p.x & 1);
                if (cfaPattern == 0) return phase;
                if (cfaPattern == 1) {
                    if (phase == 0) return 1;
                    if (phase == 1) return 0;
                    if (phase == 2) return 3;
                    return 2;
                }
                if (cfaPattern == 2) {
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

            ivec2 bayerOffset(int cfaPattern, int planeIndex) {
                if (cfaPattern == 0) {
                    if (planeIndex == 0) return ivec2(0, 0);
                    if (planeIndex == 1) return ivec2(1, 0);
                    if (planeIndex == 2) return ivec2(0, 1);
                    return ivec2(1, 1);
                }
                if (cfaPattern == 1) {
                    if (planeIndex == 0) return ivec2(1, 0);
                    if (planeIndex == 1) return ivec2(0, 0);
                    if (planeIndex == 2) return ivec2(1, 1);
                    return ivec2(0, 1);
                }
                if (cfaPattern == 2) {
                    if (planeIndex == 0) return ivec2(0, 1);
                    if (planeIndex == 1) return ivec2(1, 1);
                    if (planeIndex == 2) return ivec2(0, 0);
                    return ivec2(1, 0);
                }
                if (planeIndex == 0) return ivec2(1, 1);
                if (planeIndex == 1) return ivec2(0, 1);
                if (planeIndex == 2) return ivec2(1, 0);
                return ivec2(0, 0);
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

            float rawNormAt(ivec2 p) {
                p = clamp(p, ivec2(0), uProxySize * 2 - ivec2(1));
                int b = bayerIndexAt(uCfaPattern, p);
                float raw = float(texelFetch(uRaw, p, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[b], 1.0);
                return clamp((raw - uBlackLevel[b]) / range, 0.0, 1.0);
            }

            float greenBlock(ivec2 planeCoord) {
                ivec2 s = clamp(planeCoord * 2, ivec2(0), uProxySize * 2 - ivec2(2));
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

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uProxySize.x || p.y >= uProxySize.y) return;
                float center = greenBlock(p);
                float sum = 0.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        ivec2 q = clamp(p + ivec2(x, y), ivec2(0), uProxySize - ivec2(1));
                        sum += greenBlock(q);
                    }
                }
                float mean = sum / 9.0;
                imageStore(uProxy, p, vec4(clamp(center + 0.35 * (center - mean), 0.0, 1.0), 0.0, 0.0, 1.0));
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
                float sum = 0.0;
                for (int y = 0; y < 2; ++y) {
                    for (int x = 0; x < 2; ++x) {
                        ivec2 q = clamp(src + ivec2(x, y), ivec2(0), uInputSize - ivec2(1));
                        sum += texelFetch(uInput, q, 0).r;
                    }
                }
                fragColor = vec4(sum * 0.25, 0.0, 0.0, 1.0);
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
            out vec4 fragColor;

            float readTex(sampler2D tex, ivec2 p) {
                p = clamp(p, ivec2(0), uLevelSize - ivec2(1));
                return texelFetch(tex, p, 0).r;
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
                        for (int sy = 1; sy < levelTile - 1; sy += uSampleStep) {
                            for (int sx = 1; sx < levelTile - 1; sx += uSampleStep) {
                                ivec2 rp = levelStart + ivec2(sx, sy);
                                float rv = readTex(uReference, rp);
                                float cv = readTex(uCurrent, rp + ivec2(dx, dy));
                                sad += abs(rv - cv);
                                count += 1.0;
                            }
                        }
                        sad = sad / max(count, 1.0) + 0.0006 * float(dx * dx + dy * dy);
                        if (sad < bestSad) {
                            bestSad = sad;
                            bestShift = ivec2(dx, dy);
                        }
                    }
                }
                fragColor = vec4(vec2(bestShift) * float(uLevelScale), bestSad, 1.0);
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
                        float w = d > uOutlierThreshold ? 0.15 : 1.0;
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
                float flatness = 1.0 - smoothstep(0.35, 4.0, snr);
                float anisotropy = 1.0 + sqrt(max(lambda1 - lambda2, 0.0) / max(lambda1 + lambda2, 1e-7));
                float kDetail = 0.30;
                float kDenoise = 1.0;
                float kShrink = 3.0;
                float kStretch = 5.0;
                float k1Base = anisotropy > 1.6 ? 1.0 / kShrink : 1.0;
                float k2Base = anisotropy > 1.6 ? kStretch : 1.0;
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
                float flowPenalty = flowMag > 15.0 ? exp(-0.1 * (flowMag - 15.0)) : 1.0;
                float flowRange = length(fMax - fMin);
                float consistencyPenalty = flowRange > 8.0 ? exp(-0.1 * (flowRange - 8.0)) : 1.0;
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
                        float noiseFloor = mix(4.5 * sigma2, sigma2Noise, edgeRelax);
                        float den = mix(sigma2, sigma2Noise, edgeRelax);
                        float residual = max(0.0, d2 - noiseFloor) / max(den, 1e-10);
                        float tau = 1.0 + 1.2 * edgeRelax;
                        float robust = exp(-0.5 * pow(residual / tau, 8.0)) * globalPenalty;
                        float w = (x == 0 && y == 0) ? 2.0 : 1.0;
                        sumR += robust * w;
                        weightSum += w;
                        minR = min(minR, robust);
                        if (x == 0 && y == 0) centerR = robust;
                    }
                }
                float avgR = sumR / max(weightSum, 1.0);
                float minMix = mix(0.35, 0.15, edgeRelax);
                float centerMix = mix(0.35, 0.55, edgeRelax);
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
                        weakCount += r < 0.5 ? 1.0 : 0.0;
                        detailSum += detail;
                        count += 1.0;
                    }
                }
                float meanR = robustSum / max(count, 1.0);
                float weak = weakCount / max(count, 1.0);
                float detail = detailSum / max(count, 1.0);
                float robustNorm = clamp((meanR - 0.58) / 0.24, 0.0, 1.0);
                float weakPenalty = clamp(1.0 - max(0.0, weak - 0.10) / 0.30, 0.0, 1.0);
                float detailBoost = detail > 0.055 ? 1.0 : (detail > 0.025 ? 0.70 : 0.35);
                float mask = clamp((0.55 * robustNorm + 0.45 * weakPenalty) * (0.55 + 0.45 * detailBoost), 0.0, 1.0);
                if (detail > 0.055) {
                    mask = max(mask, 0.35);
                } else if (detail > 0.025) {
                    mask = max(mask, 0.20);
                }
                fragColor = mask;
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
            uniform float uFrameWeight;

            vec2 flowAt(vec2 planePos) {
                vec2 grid = planePos / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uFlowGrid, clamp(uv, vec2(0.0), vec2(1.0))).rg;
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

            float lscGain(int bayerIndex, ivec2 samplePos) {
                vec2 uv = (vec2(samplePos) + vec2(0.5)) / vec2(uImageSize);
                vec4 gains = texture(uLensShadingMap, uv);
                if (bayerIndex == 0) return gains.r;
                if (bayerIndex == 1) return gains.g;
                if (bayerIndex == 2) return gains.b;
                return gains.a;
            }

            float rawNormAt(ivec2 samplePos, int bayerIndex) {
                samplePos = clamp(samplePos, ivec2(0), uImageSize - ivec2(1));
                float raw = float(texelFetch(uInputRaw, samplePos, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) * lscGain(bayerIndex, samplePos) / range, 0.0, 1.0);
            }

            float fetchSameColor(ivec2 lattice, ivec2 offset, int bayerIndex) {
                lattice = clamp(lattice, ivec2(0), uPlaneSize - ivec2(1));
                return rawNormAt(offset + lattice * 2, bayerIndex);
            }

            float sampleSameColor(vec2 planePos, ivec2 offset, int bayerIndex) {
                vec2 pos = clamp(planePos, vec2(0.0), vec2(uPlaneSize - ivec2(1)));
                ivec2 p0 = ivec2(floor(pos));
                ivec2 p1 = min(p0 + ivec2(1), uPlaneSize - ivec2(1));
                vec2 f = pos - vec2(p0);
                float v00 = fetchSameColor(p0, offset, bayerIndex);
                float v10 = fetchSameColor(ivec2(p1.x, p0.y), offset, bayerIndex);
                float v01 = fetchSameColor(ivec2(p0.x, p1.y), offset, bayerIndex);
                float v11 = fetchSameColor(p1, offset, bayerIndex);
                return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
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

            float sensorPrecisionWeight(float signalNorm) {
                if (uNoiseAlpha <= 0.0 && uNoiseBeta <= 0.0) return 1.0;
                float variance = max(uNoiseAlpha * clamp(signalNorm, 0.0, 1.0) + uNoiseBeta, 1e-10);
                float referenceVariance = max(uNoiseAlpha * 0.18 + uNoiseBeta, 1e-10);
                return clamp(referenceVariance / variance, 0.05, 4.0);
            }

            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uImageSize.x || p.y >= uImageSize.y) return;
                vec4 prev = texelFetch(uAccumulatorInput, p, 0);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                ivec2 offset = bayerOffset(uCfaPattern, bayerIndex);
                vec2 planePos = 0.5 * (vec2(p) - vec2(offset));
                if (planePos.x < -0.001 || planePos.y < -0.001 ||
                    planePos.x > float(uPlaneSize.x - 1) + 0.001 ||
                    planePos.y > float(uPlaneSize.y - 1) + 0.001) {
                    imageStore(uAccumulatorOutput, p, prev);
                    return;
                }

                float value = 0.0;
                float robust = 1.0;
                float local = 1.0;
                if (uIsReference != 0) {
                    value = rawNormAt(p, bayerIndex);
                } else {
                    vec2 flow = flowAt(planePos);
                    vec2 sourcePlane = planePos + flow;
                    if (sourcePlane.x < 0.0 || sourcePlane.y < 0.0 ||
                        sourcePlane.x > float(uPlaneSize.x - 1) ||
                        sourcePlane.y > float(uPlaneSize.y - 1)) {
                        imageStore(uAccumulatorOutput, p, prev);
                        return;
                    }
                    robust = mapAt(uRobustness, planePos);
                    local = tileMaskAt(planePos);
                    float base = local * max(robust, 0.01 * local);
                    if (base <= 0.001) {
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
                            sumValue += sampleSameColor(sourcePlane + tap, offset, bayerIndex) * w;
                            sumWeight += w;
                        }
                    }
                    value = sumValue / max(sumWeight, 1e-5);
                    float highlightSuppression = 1.0 - 0.62 * smoothstep(0.78, 0.98, value);
                    robust = base * highlightSuppression;
                }

                float variance = max(uNoiseAlpha * value + uNoiseBeta, 1e-10);
                float wiener = (value * value) / (value * value + variance + 1e-6);
                float weight = sensorPrecisionWeight(value) * (0.25 + wiener);
                if (uIsReference == 0) {
                    weight *= uFrameWeight * robust;
                }
                if (weight <= 0.0) {
                    imageStore(uAccumulatorOutput, p, prev);
                    return;
                }

                vec4 next = prev + vec4(value * weight, weight, value * value * weight, weight * clamp(robust, 0.0, 1.0));
                imageStore(uAccumulatorOutput, p, next);
            }
        """.trimIndent()

        private val NORMALIZE_FRAGMENT_SHADER = """
            #version 300 es
            $RAW_COMMON
            in vec2 vTexCoord;
            out vec4 fragColor;
            uniform sampler2D uAccumulator;
            uniform highp usampler2D uReferenceRaw;
            uniform sampler2D uLensShadingMap;
            uniform ivec2 uImageSize;
            uniform int uCfaPattern;
            uniform float uBlackLevel[4];
            uniform float uWhiteLevel;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;

            float lscGain(int bayerIndex, ivec2 samplePos) {
                vec2 uv = (vec2(samplePos) + vec2(0.5)) / vec2(uImageSize);
                vec4 gains = texture(uLensShadingMap, uv);
                if (bayerIndex == 0) return gains.r;
                if (bayerIndex == 1) return gains.g;
                if (bayerIndex == 2) return gains.b;
                return gains.a;
            }

            float referenceNorm(ivec2 p, int bayerIndex) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                float raw = float(texelFetch(uReferenceRaw, p, 0).r);
                float range = max(uWhiteLevel - uBlackLevel[bayerIndex], 1.0);
                return clamp(max(raw - uBlackLevel[bayerIndex], 0.0) * lscGain(bayerIndex, p) / range, 0.0, 1.0);
            }

            vec4 readAccum(ivec2 p) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                return texelFetch(uAccumulator, p, 0);
            }

            float meanAt(ivec2 p) {
                int b = bayerIndexAt(uCfaPattern, p);
                vec4 a = readAccum(p);
                if (a.g <= 1e-5) return referenceNorm(p, b);
                return clamp(a.r / max(a.g, 1e-5), 0.0, 1.0);
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                int bayerIndex = bayerIndexAt(uCfaPattern, p);
                vec4 a = readAccum(p);
                float fused = a.g > 1e-5 ? clamp(a.r / max(a.g, 1e-5), 0.0, 1.0) : referenceNorm(p, bayerIndex);

                float mean = 0.0;
                float mean2 = 0.0;
                float count = 0.0;
                for (int y = -2; y <= 2; ++y) {
                    for (int x = -2; x <= 2; ++x) {
                        ivec2 q = p + ivec2(x * 2, y * 2);
                        if (q.x < 0 || q.y < 0 || q.x >= uImageSize.x || q.y >= uImageSize.y) continue;
                        float v = meanAt(q);
                        mean += v;
                        mean2 += v * v;
                        count += 1.0;
                    }
                }
                mean /= max(count, 1.0);
                mean2 /= max(count, 1.0);
                float variance = max(mean2 - mean * mean, 0.0);
                float noise = max(uNoiseAlpha * fused + uNoiseBeta, uNoiseBeta) / max(a.g, 1.0);
                float wienerGain = max(variance - noise, 0.0) / max(variance, 1e-6);
                float flatness = 1.0 - smoothstep(0.0004, 0.006, variance);
                fused = mix(fused, mean + wienerGain * (fused - mean), 0.55 * flatness);
                fused = clamp(fused, 0.0, 1.0);

                uint raw = uint(floor(fused * 65535.0 + 0.5));
                uint lo = raw & 255u;
                uint hi = (raw >> 8) & 255u;
                fragColor = vec4(float(lo) / 255.0, float(hi) / 255.0, 0.0, 1.0);
            }
        """.trimIndent()
    }
}
