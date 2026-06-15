package com.hinnka.mycamera.processor

import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.ImageFormat
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLES31
import androidx.core.graphics.createBitmap
import com.hinnka.mycamera.model.SafeImage
import com.hinnka.mycamera.utils.LargeDirectBuffer
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import kotlin.math.max

class GlesYuvStacker(
    private val width: Int,
    private val height: Int,
    private val outputWidth: Int,
    private val outputHeight: Int,
    private val rotation: Int,
    private val colorSpace: ColorSpace,
    private val inputFormat: Int,
) {
    private data class TextureLevel(val texture: Int, val width: Int, val height: Int)

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE

    private val textures = ArrayList<Int>()
    private val programs = ArrayList<Int>()
    private val framebuffers = ArrayList<Int>()

    private var downsampleProgram = 0
    private var structureProgram = 0
    private var alignProgram = 0
    private var smoothFlowProgram = 0
    private var robustnessProgram = 0
    private var tileMaskProgram = 0
    private var accumulateProgram = 0
    private var normalizeProgram = 0
    private var p010LumaProgram = 0
    private var p010ChromaProgram = 0
    private var planarChroma8Program = 0
    private var planarChroma16Program = 0

    private var renderFbo = 0
    private var readbackFbo = 0

    private var refY = 0
    private var refCbCr = 0
    private var curY = 0
    private var curCbCr = 0
    private var refYStaging = 0
    private var refCbCrStaging = 0
    private var curYStaging = 0
    private var curCbCrStaging = 0
    private var planarUStaging = 0
    private var planarVStaging = 0
    private var kernelTexture = 0
    private var flowTexture = 0
    private var flowScratchTexture = 0
    private var robustnessTexture = 0
    private var tileMaskTexture = 0
    private var accumulatorTexture = 0
    private var accumulatorScratchTexture = 0
    private var currentAccumulatorTexture = 0
    private var outputTexture = 0

    private var gridWidth = 0
    private var gridHeight = 0
    private val renderOutputWidth = outputWidth
    private val renderOutputHeight = outputHeight
    private val normalizedRotation = normalizeRotation(rotation)
    private val cpuRotateReadback = normalizedRotation == 90 || normalizedRotation == 270
    private val gpuOutputWidth = if (cpuRotateReadback) renderOutputHeight else renderOutputWidth
    private val gpuOutputHeight = if (cpuRotateReadback) renderOutputWidth else renderOutputHeight
    private val highPrecisionInput = inputFormat == ImageFormat.YCBCR_P010
    private val lumaInternalFormat = if (highPrecisionInput) GLES30.GL_R16F else GLES30.GL_R8
    private val chromaInternalFormat = if (highPrecisionInput) GLES30.GL_RG16F else GLES30.GL_RG8
    private val chromaWidth = (width + 1) / 2
    private val chromaHeight = (height + 1) / 2

    fun process(images: List<SafeImage>): Bitmap? {
        if (images.isEmpty() || width <= 0 || height <= 0 || outputWidth <= 0 || outputHeight <= 0) {
            return null
        }
        if (!supportsImageFormat(inputFormat)) {
            PLog.w(TAG, "Unsupported GLES YUV stack format: $inputFormat")
            return null
        }
        if (images.any { it.format != inputFormat }) {
            PLog.w(TAG, "Mixed YUV formats in one stack are not supported")
            return null
        }

        val startTime = System.currentTimeMillis()
        val originalThreadPriority = GlesGpuScheduler.lowerCurrentThreadPriority(TAG)
        try {
            initEgl()
            ensureGles31()
            initPrograms()
            initResources()
            PLog.d(
                TAG,
                "GLES stack format=${formatName(inputFormat)} internal=${if (highPrecisionInput) "R16F/RG16F" else "R8/RG8"} " +
                    "flowGrid=${FLOW_GRID_SPACING}px grid=${gridWidth}x${gridHeight}"
            )

            if (!uploadImagePlanes(images.first(), refY, refCbCr, refYStaging, refCbCrStaging, "reference")) {
                return null
            }

            val refPyramid = createPyramid(refY)
            val curPyramid = createPyramid(curY)
            buildPyramid(refPyramid)
            computeStructureTensor(refY)
            clearAccumulator()
            accumulateFrame(refY, refCbCr, isReference = true)
            GlesGpuScheduler.yieldToUiRenderer()

            for (index in 1 until images.size) {
                if (!uploadImagePlanes(images[index], curY, curCbCr, curYStaging, curCbCrStaging, "frame $index")) {
                    PLog.w(TAG, "Failed to upload frame $index YUV planes")
                    return null
                }
                buildPyramid(curPyramid)
                alignCurrentToReference(refPyramid, curPyramid)
                smoothFlow()
                computeRobustness()
                computeTileMask()
                accumulateFrame(curY, curCbCr, isReference = false)
                GlesGpuScheduler.yieldToUiRenderer()
            }

            normalizeOutput()
            GlesGpuScheduler.yieldToUiRenderer()
            val bitmap = readOutputBitmap() ?: return null
            PLog.i(TAG, "GLES YUV stacking completed in ${System.currentTimeMillis() - startTime}ms")
            return bitmap
        } catch (e: Exception) {
            PLog.e(TAG, "GLES YUV stacking failed", e)
            return null
        } finally {
            release()
            GlesGpuScheduler.restoreCurrentThreadPriority(originalThreadPriority, TAG)
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

        val surfaceAttribs = intArrayOf(
            EGL14.EGL_WIDTH, max(1, gpuOutputWidth),
            EGL14.EGL_HEIGHT, max(1, gpuOutputHeight),
            EGL14.EGL_NONE,
        )
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, config, surfaceAttribs, 0)
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
            throw IllegalStateException("GLES compute requires OpenGL ES 3.1+, got: $version")
        }
    }

    private fun initPrograms() {
        downsampleProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, DOWNSAMPLE_FRAGMENT_SHADER, "downsample")
        structureProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, STRUCTURE_FRAGMENT_SHADER, "structure")
        alignProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, ALIGN_FRAGMENT_SHADER, "align_tiles")
        smoothFlowProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, SMOOTH_FLOW_FRAGMENT_SHADER, "smooth_flow")
        robustnessProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, ROBUSTNESS_FRAGMENT_SHADER, "robustness")
        tileMaskProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, TILE_MASK_FRAGMENT_SHADER, "tile_mask")
        accumulateProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, ACCUMULATE_FRAGMENT_SHADER, "accumulate")
        normalizeProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, NORMALIZE_FRAGMENT_SHADER, "normalize")
        p010LumaProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, P010_LUMA_FRAGMENT_SHADER, "p010_luma")
        p010ChromaProgram = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, P010_CHROMA_FRAGMENT_SHADER, "p010_chroma")
        planarChroma8Program = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, PLANAR_CHROMA_8_FRAGMENT_SHADER, "planar_chroma_8")
        planarChroma16Program = linkGraphicsProgram(FULLSCREEN_VERTEX_SHADER, PLANAR_CHROMA_16_FRAGMENT_SHADER, "planar_chroma_16")
    }

    private fun initResources() {
        gridWidth = (width + FLOW_GRID_SPACING - 1) / FLOW_GRID_SPACING
        gridHeight = (height + FLOW_GRID_SPACING - 1) / FLOW_GRID_SPACING

        refY = createTexture2D(width, height, lumaInternalFormat, GLES30.GL_LINEAR)
        refCbCr = createTexture2D(chromaWidth, chromaHeight, chromaInternalFormat, GLES30.GL_LINEAR)
        curY = createTexture2D(width, height, lumaInternalFormat, GLES30.GL_LINEAR)
        curCbCr = createTexture2D(chromaWidth, chromaHeight, chromaInternalFormat, GLES30.GL_LINEAR)
        if (highPrecisionInput) {
            refYStaging = createTexture2D(width, height, GLES30.GL_R16UI, GLES30.GL_NEAREST)
            refCbCrStaging = createTexture2D(chromaWidth, chromaHeight, GLES30.GL_RG16UI, GLES30.GL_NEAREST)
            curYStaging = createTexture2D(width, height, GLES30.GL_R16UI, GLES30.GL_NEAREST)
            curCbCrStaging = createTexture2D(chromaWidth, chromaHeight, GLES30.GL_RG16UI, GLES30.GL_NEAREST)
        }

        kernelTexture = createTexture2D(width, height, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        flowTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        flowScratchTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_RGBA16F, GLES30.GL_LINEAR)
        robustnessTexture = createTexture2D(width, height, GLES30.GL_RGBA8, GLES30.GL_LINEAR)
        tileMaskTexture = createTexture2D(gridWidth, gridHeight, GLES30.GL_RGBA8, GLES30.GL_LINEAR)
        accumulatorTexture = createTexture2D(width, height, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        accumulatorScratchTexture = createTexture2D(width, height, GLES30.GL_RGBA16F, GLES30.GL_NEAREST)
        currentAccumulatorTexture = accumulatorTexture
        outputTexture = createTexture2D(gpuOutputWidth, gpuOutputHeight, GLES30.GL_RGBA8, GLES30.GL_NEAREST)

        renderFbo = createFramebuffer()
        readbackFbo = createFramebuffer()
    }

    private fun createPyramid(baseTexture: Int): List<TextureLevel> {
        val levels = ArrayList<TextureLevel>(PYRAMID_LEVELS)
        levels += TextureLevel(baseTexture, width, height)
        var levelWidth = width
        var levelHeight = height
        repeat(PYRAMID_LEVELS - 1) {
            levelWidth = max(1, (levelWidth + 1) / 2)
            levelHeight = max(1, (levelHeight + 1) / 2)
            val texture = createTexture2D(
                levelWidth,
                levelHeight,
                GLES30.GL_RGBA8,
                GLES30.GL_LINEAR,
            )
            levels += TextureLevel(texture, levelWidth, levelHeight)
        }
        return levels
    }

    private fun uploadImagePlanes(
        image: SafeImage,
        yTexture: Int,
        cbCrTexture: Int,
        yStagingTexture: Int,
        cbCrStagingTexture: Int,
        label: String,
    ): Boolean {
        val planes = image.planes
        if (planes.size < 3) {
            PLog.w(TAG, "$label has ${planes.size} planes")
            return false
        }
        if (label == "reference") {
            PLog.d(
                TAG,
                "GLES plane upload yRow=${planes[0].rowStride} cbRow=${planes[1].rowStride} cbPixel=${planes[1].pixelStride} chroma=${chromaWidth}x${chromaHeight}"
            )
        }

        return if (highPrecisionInput) {
            uploadP010Planes(image, yTexture, cbCrTexture, yStagingTexture, cbCrStagingTexture, label)
        } else {
            uploadYuv420Planes(image, yTexture, cbCrTexture, label)
        }
    }

    private fun uploadYuv420Planes(
        image: SafeImage,
        yTexture: Int,
        cbCrTexture: Int,
        label: String,
    ): Boolean {
        val planes = image.planes
        val yPlane = planes[0]
        val cbPlane = planes[1]
        val crPlane = planes[2]

        uploadTextureData(
            texture = yTexture,
            width = width,
            height = height,
            format = GLES30.GL_RED,
            type = GLES30.GL_UNSIGNED_BYTE,
            rowLength = yPlane.rowStride,
            buffer = yPlane.buffer,
            label = "$label Y",
        )

        return when (cbPlane.pixelStride) {
            2 -> {
                uploadTextureData(
                    texture = cbCrTexture,
                    width = chromaWidth,
                    height = chromaHeight,
                    format = GLES30.GL_RG,
                    type = GLES30.GL_UNSIGNED_BYTE,
                    rowLength = max(1, cbPlane.rowStride / 2),
                    buffer = cbPlane.buffer,
                    label = "$label CbCr",
                )
                true
            }
            1 -> {
                ensurePlanarStaging(GLES30.GL_R8)
                uploadTextureData(
                    texture = planarUStaging,
                    width = chromaWidth,
                    height = chromaHeight,
                    format = GLES30.GL_RED,
                    type = GLES30.GL_UNSIGNED_BYTE,
                    rowLength = cbPlane.rowStride,
                    buffer = cbPlane.buffer,
                    label = "$label Cb",
                )
                uploadTextureData(
                    texture = planarVStaging,
                    width = chromaWidth,
                    height = chromaHeight,
                    format = GLES30.GL_RED,
                    type = GLES30.GL_UNSIGNED_BYTE,
                    rowLength = crPlane.rowStride,
                    buffer = crPlane.buffer,
                    label = "$label Cr",
                )
                convertPlanarChroma8(cbCrTexture, label)
                true
            }
            else -> {
                PLog.w(TAG, "$label unsupported YUV_420_888 chroma pixelStride=${cbPlane.pixelStride}")
                false
            }
        }
    }

    private fun uploadP010Planes(
        image: SafeImage,
        yTexture: Int,
        cbCrTexture: Int,
        yStagingTexture: Int,
        cbCrStagingTexture: Int,
        label: String,
    ): Boolean {
        val planes = image.planes
        val yPlane = planes[0]
        val cbPlane = planes[1]
        val crPlane = planes[2]

        if (yStagingTexture == 0 || cbCrStagingTexture == 0) {
            PLog.w(TAG, "$label missing P010 staging textures")
            return false
        }

        uploadTextureData(
            texture = yStagingTexture,
            width = width,
            height = height,
            format = GLES30.GL_RED_INTEGER,
            type = GLES30.GL_UNSIGNED_SHORT,
            rowLength = max(1, yPlane.rowStride / 2),
            buffer = yPlane.buffer,
            label = "$label P010 Y",
        )
        convertP010Luma(yStagingTexture, yTexture, label)

        return when (cbPlane.pixelStride) {
            4 -> {
                uploadTextureData(
                    texture = cbCrStagingTexture,
                    width = chromaWidth,
                    height = chromaHeight,
                    format = GLES30.GL_RG_INTEGER,
                    type = GLES30.GL_UNSIGNED_SHORT,
                    rowLength = max(1, cbPlane.rowStride / 4),
                    buffer = cbPlane.buffer,
                    label = "$label P010 CbCr",
                )
                convertP010Chroma(cbCrStagingTexture, cbCrTexture, label)
                true
            }
            2 -> {
                ensurePlanarStaging(GLES30.GL_R16UI)
                uploadTextureData(
                    texture = planarUStaging,
                    width = chromaWidth,
                    height = chromaHeight,
                    format = GLES30.GL_RED_INTEGER,
                    type = GLES30.GL_UNSIGNED_SHORT,
                    rowLength = max(1, cbPlane.rowStride / 2),
                    buffer = cbPlane.buffer,
                    label = "$label P010 Cb",
                )
                uploadTextureData(
                    texture = planarVStaging,
                    width = chromaWidth,
                    height = chromaHeight,
                    format = GLES30.GL_RED_INTEGER,
                    type = GLES30.GL_UNSIGNED_SHORT,
                    rowLength = max(1, crPlane.rowStride / 2),
                    buffer = crPlane.buffer,
                    label = "$label P010 Cr",
                )
                convertPlanarChroma16(cbCrTexture, label)
                true
            }
            else -> {
                PLog.w(TAG, "$label unsupported P010 chroma pixelStride=${cbPlane.pixelStride}")
                false
            }
        }
    }

    private fun uploadTextureData(
        texture: Int,
        width: Int,
        height: Int,
        format: Int,
        type: Int,
        rowLength: Int,
        buffer: ByteBuffer,
        label: String,
    ) {
        val uploadBuffer = buffer.duplicate()
        uploadBuffer.position(0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, rowLength)
        GLES30.glTexSubImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            format,
            type,
            uploadBuffer,
        )
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        checkGlError("uploadTextureData $label")
    }

    private fun ensurePlanarStaging(internalFormat: Int) {
        if (planarUStaging != 0 && planarVStaging != 0) {
            return
        }
        planarUStaging = createTexture2D(chromaWidth, chromaHeight, internalFormat, GLES30.GL_NEAREST)
        planarVStaging = createTexture2D(chromaWidth, chromaHeight, internalFormat, GLES30.GL_NEAREST)
    }

    private fun convertP010Luma(inputTexture: Int, outputTexture: Int, label: String) {
        bindFramebufferOutput(outputTexture, "convertP010Luma $label")
        GLES30.glViewport(0, 0, width, height)
        GLES30.glUseProgram(p010LumaProgram)
        bindTexture(p010LumaProgram, "uInput", 0, inputTexture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(p010LumaProgram, "uSize"), width, height)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("convertP010Luma $label")
    }

    private fun convertP010Chroma(inputTexture: Int, outputTexture: Int, label: String) {
        bindFramebufferOutput(outputTexture, "convertP010Chroma $label")
        GLES30.glViewport(0, 0, chromaWidth, chromaHeight)
        GLES30.glUseProgram(p010ChromaProgram)
        bindTexture(p010ChromaProgram, "uInput", 0, inputTexture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(p010ChromaProgram, "uSize"), chromaWidth, chromaHeight)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("convertP010Chroma $label")
    }

    private fun convertPlanarChroma8(outputTexture: Int, label: String) {
        bindFramebufferOutput(outputTexture, "convertPlanarChroma8 $label")
        GLES30.glViewport(0, 0, chromaWidth, chromaHeight)
        GLES30.glUseProgram(planarChroma8Program)
        bindTexture(planarChroma8Program, "uCb", 0, planarUStaging)
        bindTexture(planarChroma8Program, "uCr", 1, planarVStaging)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(planarChroma8Program, "uSize"), chromaWidth, chromaHeight)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("convertPlanarChroma8 $label")
    }

    private fun convertPlanarChroma16(outputTexture: Int, label: String) {
        bindFramebufferOutput(outputTexture, "convertPlanarChroma16 $label")
        GLES30.glViewport(0, 0, chromaWidth, chromaHeight)
        GLES30.glUseProgram(planarChroma16Program)
        bindTexture(planarChroma16Program, "uCb", 0, planarUStaging)
        bindTexture(planarChroma16Program, "uCr", 1, planarVStaging)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(planarChroma16Program, "uSize"), chromaWidth, chromaHeight)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("convertPlanarChroma16 $label")
    }

    private fun buildPyramid(levels: List<TextureLevel>) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, renderFbo)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT1, GLES30.GL_TEXTURE_2D, 0, 0)
        val drawBuffers = intArrayOf(GLES30.GL_COLOR_ATTACHMENT0)
        GLES30.glDrawBuffers(drawBuffers.size, drawBuffers, 0)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glUseProgram(downsampleProgram)
        for (index in 1 until levels.size) {
            val input = levels[index - 1]
            val output = levels[index]
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, output.texture, 0)
            checkFramebuffer("buildPyramid level $index")
            GLES30.glViewport(0, 0, output.width, output.height)
            bindTexture(downsampleProgram, "uInput", 0, input.texture)
            GLES30.glUniform2i(GLES30.glGetUniformLocation(downsampleProgram, "uInputSize"), input.width, input.height)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
            GLES31.glMemoryBarrier(GLES31.GL_FRAMEBUFFER_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
            checkGlError("buildPyramid level $index")
        }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
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
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT1,
            GLES30.GL_TEXTURE_2D,
            0,
            0,
        )
        val drawBuffers = intArrayOf(GLES30.GL_COLOR_ATTACHMENT0)
        GLES30.glDrawBuffers(drawBuffers.size, drawBuffers, 0)
        GLES30.glDisable(GLES30.GL_BLEND)
        checkFramebuffer(label)
    }

    private fun finishFramebufferPass(label: String) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES31.glMemoryBarrier(GLES31.GL_FRAMEBUFFER_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
        checkGlError(label)
    }

    private fun computeStructureTensor(referenceY: Int) {
        bindFramebufferOutput(kernelTexture, "computeStructureTensor")
        GLES30.glViewport(0, 0, width, height)
        GLES30.glUseProgram(structureProgram)
        bindTexture(structureProgram, "uLuma", 0, referenceY)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(structureProgram, "uImageSize"), width, height)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(structureProgram, "uNoiseAlpha"), NOISE_ALPHA)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(structureProgram, "uNoiseBeta"), NOISE_BETA)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("computeStructureTensor")
    }

    private fun clearAccumulator() {
        bindFramebufferOutput(accumulatorTexture, "clearAccumulator")
        GLES30.glViewport(0, 0, width, height)
        GLES30.glClearColor(0f, 0f, 0f, 0f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        finishFramebufferPass("clearAccumulator")
        currentAccumulatorTexture = accumulatorTexture
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

    private fun computeRobustness() {
        bindFramebufferOutput(robustnessTexture, "computeRobustness")
        GLES30.glViewport(0, 0, width, height)
        GLES30.glUseProgram(robustnessProgram)
        bindTexture(robustnessProgram, "uReferenceY", 0, refY)
        bindTexture(robustnessProgram, "uReferenceCbCr", 1, refCbCr)
        bindTexture(robustnessProgram, "uCurrentY", 2, curY)
        bindTexture(robustnessProgram, "uCurrentCbCr", 3, curCbCr)
        bindTexture(robustnessProgram, "uFlowGrid", 4, flowTexture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(robustnessProgram, "uImageSize"), width, height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(robustnessProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(robustnessProgram, "uTileSize"), FLOW_GRID_SPACING)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(robustnessProgram, "uNoiseAlpha"), NOISE_ALPHA)
        GLES31.glUniform1f(GLES31.glGetUniformLocation(robustnessProgram, "uNoiseBeta"), NOISE_BETA)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("computeRobustness")
    }

    private fun computeTileMask() {
        bindFramebufferOutput(tileMaskTexture, "computeTileMask")
        GLES30.glViewport(0, 0, gridWidth, gridHeight)
        GLES30.glUseProgram(tileMaskProgram)
        bindTexture(tileMaskProgram, "uReferenceY", 0, refY)
        bindTexture(tileMaskProgram, "uRobustness", 1, robustnessTexture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(tileMaskProgram, "uImageSize"), width, height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(tileMaskProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(tileMaskProgram, "uTileSize"), FLOW_GRID_SPACING)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("computeTileMask")
    }

    private fun accumulateFrame(
        yTexture: Int,
        cbCrTexture: Int,
        isReference: Boolean,
    ) {
        val outputAccumulator = if (currentAccumulatorTexture == accumulatorTexture) {
            accumulatorScratchTexture
        } else {
            accumulatorTexture
        }
        bindFramebufferOutput(outputAccumulator, "accumulateFrame")
        GLES30.glViewport(0, 0, width, height)
        GLES30.glUseProgram(accumulateProgram)
        bindTexture(accumulateProgram, "uCurrentY", 0, yTexture)
        bindTexture(accumulateProgram, "uCurrentCbCr", 1, cbCrTexture)
        bindTexture(accumulateProgram, "uFlowGrid", 2, flowTexture)
        bindTexture(accumulateProgram, "uRobustness", 3, robustnessTexture)
        bindTexture(accumulateProgram, "uTileMask", 4, tileMaskTexture)
        bindTexture(accumulateProgram, "uKernel", 5, kernelTexture)
        bindTexture(accumulateProgram, "uAccumulatorInput", 6, currentAccumulatorTexture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(accumulateProgram, "uImageSize"), width, height)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(accumulateProgram, "uGridSize"), gridWidth, gridHeight)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(accumulateProgram, "uTileSize"), FLOW_GRID_SPACING)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(accumulateProgram, "uIsReference"), if (isReference) 1 else 0)
        GLES31.glUniform1f(
            GLES31.glGetUniformLocation(accumulateProgram, "uFrameWeight"),
            if (isReference) 1.0f else NON_REFERENCE_FRAME_WEIGHT,
        )
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("accumulateFrame")
        currentAccumulatorTexture = outputAccumulator
    }

    private fun normalizeOutput() {
        bindFramebufferOutput(outputTexture, "normalizeOutput")
        GLES30.glViewport(0, 0, gpuOutputWidth, gpuOutputHeight)
        GLES30.glUseProgram(normalizeProgram)
        bindTexture(normalizeProgram, "uAccumulator", 0, currentAccumulatorTexture)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(normalizeProgram, "uInputSize"), width, height)
        val transform = computeRenderTransform()
        GLES31.glUniform3f(
            GLES31.glGetUniformLocation(normalizeProgram, "uTransformX"),
            transform[0],
            transform[1],
            transform[2],
        )
        GLES31.glUniform3f(
            GLES31.glGetUniformLocation(normalizeProgram, "uTransformY"),
            transform[3],
            transform[4],
            transform[5],
        )
        GLES31.glUniform1f(GLES31.glGetUniformLocation(normalizeProgram, "uNoiseBeta"), NOISE_BETA)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(normalizeProgram, "uIsP010"), if (highPrecisionInput) 1 else 0)
        val directOffset = computeDirectSourceOffset()
        GLES31.glUniform1i(GLES31.glGetUniformLocation(normalizeProgram, "uDirectSource"), if (cpuRotateReadback) 1 else 0)
        GLES31.glUniform2i(GLES31.glGetUniformLocation(normalizeProgram, "uDirectOffset"), directOffset[0], directOffset[1])
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
        finishFramebufferPass("normalizeOutput")
    }

    private fun readOutputBitmap(): Bitmap? {
        val bitmap = try {
            createBitmap(renderOutputWidth, renderOutputHeight, colorSpace = colorSpace)
        } catch (e: OutOfMemoryError) {
            PLog.e(TAG, "OOM creating GLES stack bitmap ($renderOutputWidth x $renderOutputHeight)", e)
            return null
        }

        val bufferByteCount = gpuOutputWidth.toLong() * gpuOutputHeight.toLong() * 4L
        val buffer = LargeDirectBuffer.allocate(bufferByteCount, "GLES YUV stack readback") ?: return null
        try {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, readbackFbo)
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, outputTexture, 0)
            GLES30.glReadBuffer(GLES30.GL_COLOR_ATTACHMENT0)
            checkFramebuffer("readOutputBitmap")
            GLES30.glPixelStorei(GLES30.GL_PACK_ALIGNMENT, 1)
            GLES30.glViewport(0, 0, gpuOutputWidth, gpuOutputHeight)
            GLES30.glReadPixels(0, 0, gpuOutputWidth, gpuOutputHeight, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            buffer.position(0)
            if (cpuRotateReadback) {
                copyRotatedReadbackToBitmap(buffer, bitmap)
            } else {
                bitmap.copyPixelsFromBuffer(buffer)
            }
            checkGlError("readOutputBitmap")
        } finally {
            LargeDirectBuffer.free(buffer)
        }
        return bitmap
    }

    private fun copyRotatedReadbackToBitmap(buffer: ByteBuffer, bitmap: Bitmap) {
        val rotatedByteCount = renderOutputWidth.toLong() * renderOutputHeight.toLong() * 4L
        val rotated = LargeDirectBuffer.allocate(rotatedByteCount, "GLES YUV rotated readback") ?: return
        try {
            for (y in 0 until renderOutputHeight) {
                for (x in 0 until renderOutputWidth) {
                    val sourceX: Int
                    val sourceY: Int
                    if (normalizedRotation == 90) {
                        sourceX = y
                        sourceY = gpuOutputHeight - 1 - x
                    } else {
                        sourceX = gpuOutputWidth - 1 - y
                        sourceY = x
                    }
                    val sourceOffset = (sourceY * gpuOutputWidth + sourceX) * 4
                    val destOffset = (y * renderOutputWidth + x) * 4
                    rotated.put(destOffset, buffer.get(sourceOffset))
                    rotated.put(destOffset + 1, buffer.get(sourceOffset + 1))
                    rotated.put(destOffset + 2, buffer.get(sourceOffset + 2))
                    rotated.put(destOffset + 3, buffer.get(sourceOffset + 3))
                }
            }
            rotated.position(0)
            bitmap.copyPixelsFromBuffer(rotated)
        } finally {
            LargeDirectBuffer.free(rotated)
        }
    }

    private fun createTexture2D(
        textureWidth: Int,
        textureHeight: Int,
        internalFormat: Int,
        filter: Int,
    ): Int {
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

    private fun compileShader(type: Int, source: String, name: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(shader)
            GLES30.glDeleteShader(shader)
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

    private fun normalizeRotation(degrees: Int): Int {
        val normalized = ((degrees % 360) + 360) % 360
        return when (normalized) {
            90, 180, 270 -> normalized
            else -> 0
        }
    }

    private fun computeNormalizeTransform(): FloatArray {
        val sensorWidth = width.toFloat()
        val sensorHeight = height.toFloat()
        val rotatedWidth = if (normalizedRotation == 90 || normalizedRotation == 270) height else width
        val rotatedHeight = if (normalizedRotation == 90 || normalizedRotation == 270) width else height
        val cropX = (((rotatedWidth - outputWidth).coerceAtLeast(0)) / 4 * 2).toFloat()
        val cropY = (((rotatedHeight - outputHeight).coerceAtLeast(0)) / 4 * 2).toFloat()
        return when (normalizedRotation) {
            90 -> floatArrayOf(
                0.0f, 1.0f, cropY,
                -1.0f, 0.0f, sensorHeight - 1.0f - cropX,
            )
            180 -> floatArrayOf(
                -1.0f, 0.0f, sensorWidth - 1.0f - cropX,
                0.0f, -1.0f, sensorHeight - 1.0f - cropY,
            )
            270 -> floatArrayOf(
                0.0f, -1.0f, sensorWidth - 1.0f - cropY,
                1.0f, 0.0f, cropX,
            )
            else -> floatArrayOf(
                1.0f, 0.0f, cropX,
                0.0f, 1.0f, cropY,
            )
        }
    }

    private fun computeRenderTransform(): FloatArray {
        if (!cpuRotateReadback) {
            return computeNormalizeTransform()
        }
        val offset = computeDirectSourceOffset()
        val offsetX = offset[0].toFloat()
        val offsetY = offset[1].toFloat()
        return floatArrayOf(
            1.0f, 0.0f, offsetX,
            0.0f, 1.0f, offsetY,
        )
    }

    private fun computeDirectSourceOffset(): IntArray {
        if (!cpuRotateReadback) {
            return intArrayOf(0, 0)
        }
        val rotatedWidth = if (normalizedRotation == 90 || normalizedRotation == 270) height else width
        val rotatedHeight = if (normalizedRotation == 90 || normalizedRotation == 270) width else height
        val cropX = ((rotatedWidth - outputWidth).coerceAtLeast(0) / 4) * 2
        val cropY = ((rotatedHeight - outputHeight).coerceAtLeast(0) / 4) * 2
        return when (normalizedRotation) {
            90 -> intArrayOf(
                cropY,
                height - cropX - gpuOutputHeight,
            )
            270 -> intArrayOf(
                width - cropY - gpuOutputWidth,
                cropX,
            )
            else -> intArrayOf(0, 0)
        }
    }

    companion object {
        private const val TAG = "GlesYuvStacker"

        private const val EGL_OPENGL_ES3_BIT_KHR = 0x00000040
        private const val PYRAMID_LEVELS = 4
        private const val ALIGN_LEVEL = 2
        private const val FLOW_GRID_SPACING = 8
        private const val ALIGN_WINDOW_SIZE = 32
        private const val SEARCH_RADIUS_LEVEL = 6
        private const val ALIGN_SAMPLE_STEP = 2
        private const val FLOW_SMOOTH_PASSES = 2
        private const val FLOW_OUTLIER_THRESHOLD_PX = 24.0f
        private const val NON_REFERENCE_FRAME_WEIGHT = 0.92f
        private const val NOISE_ALPHA = 0.005f
        private const val NOISE_BETA = 0.001f

        fun supportsImageFormat(format: Int): Boolean {
            return format == ImageFormat.YUV_420_888 || format == ImageFormat.YCBCR_P010
        }

        private fun formatName(format: Int): String {
            return when (format) {
                ImageFormat.YUV_420_888 -> "YUV_420_888"
                ImageFormat.YCBCR_P010 -> "YCBCR_P010"
                else -> format.toString()
            }
        }

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

        private val P010_LUMA_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            precision highp usampler2D;
            uniform usampler2D uInput;
            uniform ivec2 uSize;
            out float outY;

            void main() {
                ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), uSize - ivec2(1));
                outY = float(texelFetch(uInput, p, 0).r) * (1.0 / 65535.0);
            }
        """.trimIndent()

        private val P010_CHROMA_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            precision highp usampler2D;
            uniform usampler2D uInput;
            uniform ivec2 uSize;
            out vec2 outCbCr;

            void main() {
                ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), uSize - ivec2(1));
                uvec2 cbcr = texelFetch(uInput, p, 0).rg;
                outCbCr = vec2(float(cbcr.r), float(cbcr.g)) * (1.0 / 65535.0);
            }
        """.trimIndent()

        private val PLANAR_CHROMA_8_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uCb;
            uniform sampler2D uCr;
            uniform ivec2 uSize;
            out vec2 outCbCr;

            void main() {
                ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), uSize - ivec2(1));
                outCbCr = vec2(texelFetch(uCb, p, 0).r, texelFetch(uCr, p, 0).r);
            }
        """.trimIndent()

        private val PLANAR_CHROMA_16_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            precision highp usampler2D;
            uniform usampler2D uCb;
            uniform usampler2D uCr;
            uniform ivec2 uSize;
            out vec2 outCbCr;

            void main() {
                ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), uSize - ivec2(1));
                outCbCr = vec2(
                    float(texelFetch(uCb, p, 0).r),
                    float(texelFetch(uCr, p, 0).r)
                ) * (1.0 / 65535.0);
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
                float v = sum * 0.25;
                fragColor = vec4(v, 0.0, 0.0, 1.0);
            }
        """.trimIndent()

        private val STRUCTURE_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uLuma;
            uniform ivec2 uImageSize;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            out vec4 fragColor;

            float readY(ivec2 p) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                return texelFetch(uLuma, p, 0).r;
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                float sIxIx = 0.0;
                float sIyIy = 0.0;
                float sIxIy = 0.0;
                for (int y = -2; y <= 2; ++y) {
                    for (int x = -2; x <= 2; ++x) {
                        ivec2 q = p + ivec2(x, y);
                        float ix = 0.5 * (readY(q + ivec2(1, 0)) - readY(q - ivec2(1, 0)));
                        float iy = 0.5 * (readY(q + ivec2(0, 1)) - readY(q - ivec2(0, 1)));
                        sIxIx += ix * ix;
                        sIyIy += iy * iy;
                        sIxIy += ix * iy;
                    }
                }

                float jxx = sIxIx / 25.0;
                float jyy = sIyIy / 25.0;
                float jxy = sIxIy / 25.0;
                float trace = jxx + jyy;
                float diff = jxx - jyy;
                float root = sqrt(max(diff * diff + 4.0 * jxy * jxy, 0.0));
                float lambda1 = 0.5 * (trace + root);
                float lambda2 = 0.5 * (trace - root);

                float noiseVar = uNoiseAlpha * 0.5 + max(uNoiseBeta, 1e-10);
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

                float len = sqrt(max(diff * diff + 4.0 * jxy * jxy, 0.0));
                float cos2t = len < 1e-9 ? 1.0 : diff / len;
                float sin2t = len < 1e-9 ? 0.0 : 2.0 * jxy / len;
                fragColor = vec4(k1, k2, cos2t, sin2t);
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
                                ivec2 cp = rp + ivec2(dx, dy);
                                float rv = readTex(uReference, rp);
                                float cv = readTex(uCurrent, cp);
                                sad += abs(rv - cv);
                                count += 1.0;
                            }
                        }
                        sad /= max(count, 1.0);
                        float shiftPenalty = 0.0006 * float(dx * dx + dy * dy);
                        sad += shiftPenalty;
                        if (sad < bestSad) {
                            bestSad = sad;
                            bestShift = ivec2(dx, dy);
                        }
                    }
                }

                vec2 flow = vec2(bestShift) * float(uLevelScale);
                fragColor = vec4(flow, 0.0, 1.0);
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
                        if (x == 0 && y == 0) {
                            continue;
                        }
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

        private val ROBUSTNESS_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uReferenceY;
            uniform sampler2D uReferenceCbCr;
            uniform sampler2D uCurrentY;
            uniform sampler2D uCurrentCbCr;
            uniform sampler2D uFlowGrid;
            uniform ivec2 uImageSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            uniform float uNoiseAlpha;
            uniform float uNoiseBeta;
            out vec4 fragColor;

            vec2 flowAt(vec2 pixel) {
                vec2 grid = pixel / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uFlowGrid, clamp(uv, vec2(0.0), vec2(1.0))).rg;
            }

            float refY(ivec2 p) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                return texelFetch(uReferenceY, p, 0).r;
            }

            float curYRaw(vec2 pixel) {
                vec2 uv = (pixel + vec2(0.5)) / vec2(uImageSize);
                return texture(uCurrentY, clamp(uv, vec2(0.0), vec2(1.0))).r;
            }

            float curYNorm(vec2 pixel) {
                return curYRaw(pixel);
            }

            vec2 chromaUv(vec2 pixel) {
                ivec2 chromaSize = (uImageSize + ivec2(1)) / 2;
                vec2 chromaPixel = floor(pixel * 0.5);
                return clamp((chromaPixel + vec2(0.5)) / vec2(chromaSize), vec2(0.0), vec2(1.0));
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                vec2 flow = flowAt(vec2(p));
                vec2 curPixel = vec2(p) + flow;
                if (curPixel.x < 1.0 || curPixel.y < 1.0 ||
                    curPixel.x > float(uImageSize.x - 2) || curPixel.y > float(uImageSize.y - 2)) {
                    fragColor = vec4(0.0);
                    return;
                }

                float center = refY(p);
                float gx = refY(p + ivec2(1, 0)) - refY(p - ivec2(1, 0));
                float gy = refY(p + ivec2(0, 1)) - refY(p - ivec2(0, 1));
                float localVar = 0.0;
                float minR = 1.0;
                float sumR = 0.0;

                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        ivec2 rp = p + ivec2(x, y);
                        float ry = refY(rp);
                        vec2 cp = curPixel + vec2(x, y);
                        float cy = curYNorm(cp);
                        float d = ry - cy;
                        float sigmaNoise = max(uNoiseAlpha * max(ry, 0.05) + uNoiseBeta, 1e-10);
                        float sigma = max(sigmaNoise, 0.0004);
                        float residual = max(0.0, d * d - 2.0 * sigmaNoise) / sigma;
                        float r = exp(-0.5 * pow(residual, 4.0));
                        minR = min(minR, r);
                        sumR += r;
                        float dc = ry - center;
                        localVar += dc * dc;
                    }
                }

                localVar /= 9.0;
                float edgeStrength = sqrt((gx * gx + gy * gy) / max(localVar + uNoiseBeta, 1e-6));
                float edgeRelax = smoothstep(1.2, 5.0, edgeStrength);

                vec2 refC = texture(uReferenceCbCr, chromaUv(vec2(p))).rg;
                vec2 curC = texture(uCurrentCbCr, chromaUv(curPixel)).rg;
                float chromaResidual = dot(refC - curC, refC - curC);
                float chromaPenalty = exp(-chromaResidual / 0.018);

                float flowPenalty = length(flow) > 15.0 ? exp(-0.1 * (length(flow) - 15.0)) : 1.0;
                float avgR = sumR / 9.0;
                float centerMix = mix(0.35, 0.55, edgeRelax);
                float minMix = mix(0.35, 0.15, edgeRelax);
                float robust = (minR * minMix + avgR * (1.0 - minMix)) * chromaPenalty * flowPenalty;
                robust = mix(robust, avgR * chromaPenalty * flowPenalty, centerMix * 0.25);
                float outR = clamp(robust, 0.0, 1.0);
                fragColor = vec4(outR, outR, outR, 1.0);
            }
        """.trimIndent()

        private val TILE_MASK_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uReferenceY;
            uniform sampler2D uRobustness;
            uniform ivec2 uImageSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            out vec4 fragColor;

            float readY(ivec2 p) {
                p = clamp(p, ivec2(0), uImageSize - ivec2(1));
                return texelFetch(uReferenceY, p, 0).r;
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
                        if (p.x >= uImageSize.x || p.y >= uImageSize.y) {
                            continue;
                        }
                        float r = texelFetch(uRobustness, p, 0).r;
                        float c = readY(p);
                        float detail = abs(readY(p + ivec2(1, 0)) - readY(p - ivec2(1, 0))) +
                            abs(readY(p + ivec2(0, 1)) - readY(p - ivec2(0, 1))) +
                            0.5 * abs(4.0 * c - readY(p + ivec2(1, 0)) - readY(p - ivec2(1, 0)) -
                                readY(p + ivec2(0, 1)) - readY(p - ivec2(0, 1)));
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
                fragColor = vec4(mask, mask, mask, 1.0);
            }
        """.trimIndent()

        private val ACCUMULATE_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uCurrentY;
            uniform sampler2D uCurrentCbCr;
            uniform sampler2D uFlowGrid;
            uniform sampler2D uRobustness;
            uniform sampler2D uTileMask;
            uniform sampler2D uKernel;
            uniform sampler2D uAccumulatorInput;
            uniform ivec2 uImageSize;
            uniform ivec2 uGridSize;
            uniform int uTileSize;
            uniform int uIsReference;
            uniform float uFrameWeight;
            out vec4 fragColor;

            vec2 flowAt(vec2 pixel) {
                vec2 grid = pixel / float(uTileSize);
                vec2 uv = (grid + vec2(0.5)) / vec2(uGridSize);
                return texture(uFlowGrid, clamp(uv, vec2(0.0), vec2(1.0))).rg;
            }

            vec2 gridUv(vec2 pixel) {
                return clamp((pixel / float(uTileSize) + vec2(0.5)) / vec2(uGridSize), vec2(0.0), vec2(1.0));
            }

            float kernelWeight(vec2 tap, vec4 kp) {
                float cosT = sqrt(max(0.0, 0.5 * (1.0 + kp.z)));
                float sinT = sign(kp.w) * sqrt(max(0.0, 0.5 * (1.0 - kp.z)));
                float u = cosT * tap.x + sinT * tap.y;
                float v = -sinT * tap.x + cosT * tap.y;
                float e = 0.25 * (kp.x * u * u + kp.y * v * v);
                return exp(-0.5 * e);
            }

            vec3 sampleYcc(vec2 pixel) {
                vec2 uv = (pixel + vec2(0.5)) / vec2(uImageSize);
                uv = clamp(uv, vec2(0.0), vec2(1.0));
                float y = texture(uCurrentY, uv).r;
                ivec2 chromaSize = (uImageSize + ivec2(1)) / 2;
                vec2 chromaPixel = floor(pixel * 0.5);
                vec2 chromaUv = clamp((chromaPixel + vec2(0.5)) / vec2(chromaSize), vec2(0.0), vec2(1.0));
                vec2 cbcr = texture(uCurrentCbCr, chromaUv).rg;
                return vec3(y, cbcr);
            }

            void main() {
                ivec2 p = ivec2(gl_FragCoord.xy);
                vec4 prev = texelFetch(uAccumulatorInput, p, 0);
                if (uIsReference != 0) {
                    vec3 ycc = sampleYcc(vec2(p));
                    float weight = max(uFrameWeight, 1e-6);
                    fragColor = prev + vec4(ycc * weight, weight);
                    return;
                }

                vec2 flow = flowAt(vec2(p));
                vec2 source = vec2(p) + flow;
                if (source.x < 1.0 || source.y < 1.0 ||
                    source.x > float(uImageSize.x - 2) || source.y > float(uImageSize.y - 2)) {
                    fragColor = prev;
                    return;
                }

                vec2 uv = (vec2(p) + vec2(0.5)) / vec2(uImageSize);
                float robust = texture(uRobustness, uv).r;
                float local = texture(uTileMask, gridUv(vec2(p))).r;
                float baseWeight = uFrameWeight * local * max(robust, 0.01 * local);
                if (baseWeight <= 0.001) {
                    fragColor = prev;
                    return;
                }

                vec4 kp = texelFetch(uKernel, p, 0);
                vec3 sum = vec3(0.0);
                float weight = 0.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        vec2 tap = vec2(x, y);
                        float kw = kernelWeight(tap, kp);
                        vec3 ycc = sampleYcc(source + tap);
                        float w = baseWeight * kw;
                        sum += ycc * w;
                        weight += w;
                    }
                }

                if (weight > 1e-5) {
                    fragColor = prev + vec4(sum, weight);
                } else {
                    fragColor = prev;
                }
            }
        """.trimIndent()

        private val NORMALIZE_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            uniform sampler2D uAccumulator;
            uniform ivec2 uInputSize;
            uniform vec3 uTransformX;
            uniform vec3 uTransformY;
            uniform float uNoiseBeta;
            uniform int uIsP010;
            uniform int uDirectSource;
            uniform ivec2 uDirectOffset;
            out vec4 fragColor;

            bool valid(vec2 p) {
                return p.x >= 0.0 && p.y >= 0.0 &&
                    p.x <= float(uInputSize.x - 1) &&
                    p.y <= float(uInputSize.y - 1);
            }

            vec4 readAccumulator(ivec2 p) {
                p = clamp(p, ivec2(0), uInputSize - ivec2(1));
                return texelFetch(uAccumulator, p, 0);
            }

            ivec2 sourceTexel(vec2 pixel) {
                return clamp(ivec2(floor(pixel + vec2(0.5))), ivec2(0), uInputSize - ivec2(1));
            }

            bool validTexel(ivec2 p) {
                return p.x >= 0 && p.y >= 0 && p.x < uInputSize.x && p.y < uInputSize.y;
            }

            vec3 readYcc(ivec2 p) {
                vec4 a = readAccumulator(p);
                if (a.a <= 1e-4) {
                    return vec3(0.0, 0.5, 0.5);
                }
                return clamp(a.rgb / a.a, vec3(0.0), vec3(1.0));
            }

            float readWeight(ivec2 p) {
                return readAccumulator(p).a;
            }

            vec3 yccToRgb(vec3 ycc) {
                float y = ycc.x;
                float cb = ycc.y - 0.5;
                float cr = ycc.z - 0.5;
                if (uIsP010 != 0) {
                    return vec3(
                        y + 1.4746 * cr,
                        y - 0.16455 * cb - 0.57135 * cr,
                        y + 1.8814 * cb
                    );
                }
                return vec3(
                    y + 1.402 * cr,
                    y - 0.344136 * cb - 0.714136 * cr,
                    y + 1.772 * cb
                );
            }

            void main() {
                ivec2 outP = ivec2(gl_FragCoord.xy);
                ivec2 srcP;
                if (uDirectSource != 0) {
                    srcP = clamp(outP + uDirectOffset, ivec2(0), uInputSize - ivec2(1));
                } else {
                    vec2 src = vec2(
                        float(outP.x) * uTransformX.x + float(outP.y) * uTransformX.y + uTransformX.z,
                        float(outP.x) * uTransformY.x + float(outP.y) * uTransformY.y + uTransformY.z
                    );
                    srcP = sourceTexel(src);
                }

                vec3 ycc = readYcc(srcP);
                float accumWeight = readWeight(srcP);

                float mean = 0.0;
                float mean2 = 0.0;
                float count = 0.0;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        ivec2 q = srcP + ivec2(x, y);
                        if (!validTexel(q)) {
                            continue;
                        }
                        float yy = readYcc(q).x;
                        mean += yy;
                        mean2 += yy * yy;
                        count += 1.0;
                    }
                }
                mean /= max(count, 1.0);
                mean2 /= max(count, 1.0);
                float variance = max(mean2 - mean * mean, 0.0);
                float noise = uNoiseBeta / max(accumWeight, 1.0);
                float wienerGain = max(variance - noise, 0.0) / max(variance, 1e-6);
                float flatness = 1.0 - smoothstep(0.0004, 0.006, variance);
                ycc.x = mix(ycc.x, mean + wienerGain * (ycc.x - mean), 0.55 * flatness);

                vec3 rgb = clamp(yccToRgb(ycc), vec3(0.0), vec3(1.0));
                fragColor = vec4(rgb, 1.0);
            }
        """.trimIndent()

    }
}
