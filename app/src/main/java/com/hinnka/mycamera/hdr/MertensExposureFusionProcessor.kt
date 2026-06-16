package com.hinnka.mycamera.hdr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLUtils
import com.hinnka.mycamera.utils.LargeDirectBuffer
import com.hinnka.mycamera.utils.PLog
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.Executors
import kotlin.math.ln
import kotlin.math.min
import kotlin.system.measureTimeMillis

object MertensExposureFusionProcessor {
    private const val TAG = "MertensExposureFusion"
    private const val HDR_FRAME_COUNT = 3
    private const val REFERENCE_FRAME_INDEX = 0
    private const val DEFAULT_CONTRAST_WEIGHT = 1.0f
    private const val DEFAULT_SATURATION_WEIGHT = 1.0f
    private const val DEFAULT_EXPOSURE_WEIGHT = 1.0f

    private val dispatcher = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "MertensExposureFusion-GL").apply { isDaemon = true }
    }.asCoroutineDispatcher()

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null
    private var indexBuffer: ShortBuffer? = null

    private var weightProgram = 0
    private var normalizeProgram = 0
    private var pyrDownProgram = 0
    private var laplacianProgram = 0
    private var combineProgram = 0
    private var reconstructProgram = 0
    private var copyProgram = 0
    private var isInitialized = false

    suspend fun fuse(
        frames: List<Bitmap>,
        contrastWeight: Float = DEFAULT_CONTRAST_WEIGHT,
        saturationWeight: Float = DEFAULT_SATURATION_WEIGHT,
        exposureWeight: Float = DEFAULT_EXPOSURE_WEIGHT,
        exposureProducts: FloatArray? = null,
        enableDeghostMask: Boolean = false,
    ): Bitmap? = withContext(dispatcher) {
        if (frames.size != HDR_FRAME_COUNT) {
            PLog.w(TAG, "Expected $HDR_FRAME_COUNT frames, got ${frames.size}")
            return@withContext null
        }
        val width = frames.first().width
        val height = frames.first().height
        if (width <= 0 || height <= 0 || frames.any { it.width != width || it.height != height || it.isRecycled }) {
            PLog.w(TAG, "Invalid HDR bracket frame dimensions")
            return@withContext null
        }
        if (!ensureInitialized()) return@withContext null

        var result: Bitmap? = null
        val elapsed = measureTimeMillis {
            result = runCatching {
                renderFusion(frames, contrastWeight, saturationWeight, exposureWeight, exposureProducts, enableDeghostMask)
            }.onFailure {
                PLog.e(TAG, "Mertens exposure fusion failed", it)
            }.getOrNull()
        }
        PLog.d(
            TAG,
            "Mertens exposure fusion took ${elapsed}ms, size=${width}x${height}, deghostMask=$enableDeghostMask, success=${result != null}"
        )
        result
    }

    private fun renderFusion(
        frames: List<Bitmap>,
        contrastWeight: Float,
        saturationWeight: Float,
        exposureWeight: Float,
        exposureProducts: FloatArray?,
        enableDeghostMask: Boolean,
    ): Bitmap? {
        val width = frames.first().width
        val height = frames.first().height
        val maxLevel = (ln(min(width, height).toFloat()) / ln(2f)).toInt().coerceAtLeast(0)
        val uploads = frames.map { prepareUploadBitmap(it) }
        val inputTextures = uploads.map { uploadBitmapTexture(it.bitmap) }
        val exposureScales = normalizeExposureProducts(exposureProducts)
        val resultPyramid = ArrayList<RenderTarget>(maxLevel + 1)
        var currentImages = inputTextures.map { TextureLevel(width, height, it, owned = false) }
        var currentWeights: RenderTarget? = null
        var reconstructed: RenderTarget? = null

        try {
            val referenceTexture = inputTextures[REFERENCE_FRAME_INDEX]
            val rawWeights = inputTextures.mapIndexed { index, texture ->
                createRenderTarget(width, height, halfFloat = true).also {
                    renderWeight(
                        imageTexture = texture,
                        referenceTexture = referenceTexture,
                        target = it,
                        contrastWeight = contrastWeight,
                        saturationWeight = saturationWeight,
                        exposureWeight = exposureWeight,
                        exposureScale = exposureScales[index],
                        useDeghostMask = enableDeghostMask && index != REFERENCE_FRAME_INDEX,
                    )
                }
            }
            currentWeights = createRenderTarget(width, height, halfFloat = true)
            renderNormalizeWeights(rawWeights, currentWeights)
            rawWeights.forEach { it.release() }

            for (level in 0 until maxLevel) {
                val weights = currentWeights ?: throw IllegalStateException("Missing Mertens weight pyramid level $level")
                val currentWidth = currentImages.first().width
                val currentHeight = currentImages.first().height
                val nextWidth = ((currentWidth + 1) / 2).coerceAtLeast(1)
                val nextHeight = ((currentHeight + 1) / 2).coerceAtLeast(1)

                val nextImages = currentImages.map { source ->
                    createRenderTarget(nextWidth, nextHeight, halfFloat = true).also {
                        renderPyrDown(source.textureId, source.width, source.height, it)
                    }
                }
                val nextWeights = createRenderTarget(nextWidth, nextHeight, halfFloat = true).also {
                    renderPyrDown(weights.textureId, weights.width, weights.height, it)
                }

                val laplacians = currentImages.mapIndexed { index, source ->
                    createRenderTarget(currentWidth, currentHeight, halfFloat = true).also {
                        renderLaplacian(source.textureId, nextImages[index].textureId, nextImages[index].width, nextImages[index].height, it)
                    }
                }
                val resultLevel = createRenderTarget(currentWidth, currentHeight, halfFloat = true)
                renderWeightedSum(laplacians, weights, resultLevel)
                resultPyramid.add(resultLevel)

                laplacians.forEach { it.release() }
                currentImages.forEach { it.releaseIfOwned() }
                weights.release()

                currentImages = nextImages.map { TextureLevel(it.width, it.height, it.textureId, owned = true, owner = it) }
                currentWeights = nextWeights
            }

            val weights = currentWeights
            val topLevel = createRenderTarget(currentImages.first().width, currentImages.first().height, halfFloat = true)
            renderWeightedSum(
                currentImages.map { TextureOnlyTarget(it.width, it.height, it.textureId) },
                weights,
                topLevel
            )
            resultPyramid.add(topLevel)
            currentImages.forEach { it.releaseIfOwned() }
            weights.release()
            currentWeights = null

            reconstructed = resultPyramid.removeAt(resultPyramid.lastIndex)
            for (level in resultPyramid.lastIndex downTo 0) {
                val base = resultPyramid[level]
                val currentReconstruction = requireNotNull(reconstructed) {
                    "Missing Mertens reconstruction level ${level + 1}"
                }
                val isFinalLevel = level == 0
                val target = createRenderTarget(base.width, base.height, halfFloat = !isFinalLevel)
                renderReconstruct(
                    base.textureId,
                    currentReconstruction.textureId,
                    currentReconstruction.width,
                    currentReconstruction.height,
                    target
                )
                currentReconstruction.release()
                base.release()
                reconstructed = target
            }

            val finalReconstruction = requireNotNull(reconstructed) {
                "Missing Mertens final reconstruction"
            }
            if (maxLevel == 0) {
                val output = createRenderTarget(width, height, halfFloat = false)
                renderCopy(finalReconstruction.textureId, output)
                finalReconstruction.release()
                reconstructed = output
            }

            return readBitmap(requireNotNull(reconstructed) { "Missing Mertens output target" })
        } finally {
            inputTextures.forEach { GLES30.glDeleteTextures(1, intArrayOf(it), 0) }
            uploads.forEach { it.recycleIfTemporary() }
            currentImages.forEach { it.releaseIfOwned() }
            currentWeights?.release()
            reconstructed?.release()
            resultPyramid.forEach { it.release() }
        }
    }

    private fun renderWeight(
        imageTexture: Int,
        referenceTexture: Int,
        target: RenderTarget,
        contrastWeight: Float,
        saturationWeight: Float,
        exposureWeight: Float,
        exposureScale: Float,
        useDeghostMask: Boolean,
    ) {
        GLES30.glUseProgram(weightProgram)
        bindTarget(target)
        bindTexture(weightProgram, "uImage", 0, imageTexture)
        bindTexture(weightProgram, "uReferenceImage", 1, referenceTexture)
        GLES30.glUniform2i(GLES30.glGetUniformLocation(weightProgram, "uImageSize"), target.width, target.height)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(weightProgram, "uContrastWeight"), contrastWeight)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(weightProgram, "uSaturationWeight"), saturationWeight)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(weightProgram, "uExposureWeight"), exposureWeight)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(weightProgram, "uExposureScale"), exposureScale)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(weightProgram, "uUseDeghostMask"), if (useDeghostMask) 1 else 0)
        drawQuad(weightProgram)
        checkGlError("renderWeight")
    }

    private fun normalizeExposureProducts(exposureProducts: FloatArray?): FloatArray {
        val reference = exposureProducts
            ?.getOrNull(REFERENCE_FRAME_INDEX)
            ?.takeIf { it.isFinite() && it > 0f }
            ?: 1f
        return FloatArray(HDR_FRAME_COUNT) { index ->
            val product = exposureProducts
                ?.getOrNull(index)
                ?.takeIf { it.isFinite() && it > 0f }
                ?: reference
            (product / reference).coerceIn(1f / 32f, 32f)
        }
    }

    private fun renderNormalizeWeights(rawWeights: List<RenderTarget>, target: RenderTarget) {
        GLES30.glUseProgram(normalizeProgram)
        bindTarget(target)
        rawWeights.forEachIndexed { index, weight ->
            bindTexture(normalizeProgram, "uWeight$index", index, weight.textureId)
        }
        drawQuad(normalizeProgram)
        checkGlError("renderNormalizeWeights")
    }

    private fun renderPyrDown(sourceTexture: Int, sourceWidth: Int, sourceHeight: Int, target: RenderTarget) {
        GLES30.glUseProgram(pyrDownProgram)
        bindTarget(target)
        bindTexture(pyrDownProgram, "uInputTexture", 0, sourceTexture)
        GLES30.glUniform2i(GLES30.glGetUniformLocation(pyrDownProgram, "uSourceSize"), sourceWidth, sourceHeight)
        drawQuad(pyrDownProgram)
        checkGlError("renderPyrDown")
    }

    private fun renderLaplacian(
        baseTexture: Int,
        nextTexture: Int,
        nextWidth: Int,
        nextHeight: Int,
        target: RenderTarget,
    ) {
        GLES30.glUseProgram(laplacianProgram)
        bindTarget(target)
        bindTexture(laplacianProgram, "uBaseTexture", 0, baseTexture)
        bindTexture(laplacianProgram, "uNextTexture", 1, nextTexture)
        GLES30.glUniform2i(GLES30.glGetUniformLocation(laplacianProgram, "uNextSize"), nextWidth, nextHeight)
        drawQuad(laplacianProgram)
        checkGlError("renderLaplacian")
    }

    private fun renderWeightedSum(inputs: List<FramebufferSource>, weights: RenderTarget, target: RenderTarget) {
        GLES30.glUseProgram(combineProgram)
        bindTarget(target)
        inputs.forEachIndexed { index, input ->
            bindTexture(combineProgram, "uImage$index", index, input.textureId)
        }
        bindTexture(combineProgram, "uWeights", 3, weights.textureId)
        drawQuad(combineProgram)
        checkGlError("renderWeightedSum")
    }

    private fun renderReconstruct(baseTexture: Int, nextTexture: Int, nextWidth: Int, nextHeight: Int, target: RenderTarget) {
        GLES30.glUseProgram(reconstructProgram)
        bindTarget(target)
        bindTexture(reconstructProgram, "uBaseTexture", 0, baseTexture)
        bindTexture(reconstructProgram, "uNextTexture", 1, nextTexture)
        GLES30.glUniform2i(GLES30.glGetUniformLocation(reconstructProgram, "uNextSize"), nextWidth, nextHeight)
        drawQuad(reconstructProgram)
        checkGlError("renderReconstruct")
    }

    private fun renderCopy(sourceTexture: Int, target: RenderTarget) {
        GLES30.glUseProgram(copyProgram)
        bindTarget(target)
        bindTexture(copyProgram, "uInputTexture", 0, sourceTexture)
        drawQuad(copyProgram)
        checkGlError("renderCopy")
    }

    private fun bindTarget(target: RenderTarget) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, target.framebufferId)
        GLES30.glViewport(0, 0, target.width, target.height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
    }

    private fun bindTexture(program: Int, uniformName: String, unit: Int, textureId: Int) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, uniformName), unit)
    }

    private fun readBitmap(target: RenderTarget): Bitmap? {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, target.framebufferId)
        GLES30.glViewport(0, 0, target.width, target.height)
        val buffer = LargeDirectBuffer.allocate(target.width.toLong() * target.height.toLong() * 4L, "Mertens HDR readback")
            ?: return null
        return try {
            GLES30.glReadPixels(0, 0, target.width, target.height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer)
            checkGlError("readBitmap")
            Bitmap.createBitmap(target.width, target.height, Bitmap.Config.ARGB_8888).also {
                buffer.position(0)
                it.copyPixelsFromBuffer(buffer)
            }
        } finally {
            LargeDirectBuffer.free(buffer)
        }
    }

    private fun uploadBitmapTexture(bitmap: Bitmap): Int {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
        checkGlError("uploadBitmapTexture")
        return textures[0]
    }

    private fun prepareUploadBitmap(bitmap: Bitmap): UploadBitmap {
        if (bitmap.config == Bitmap.Config.ARGB_8888 && !bitmap.isRecycled) {
            return UploadBitmap(bitmap, isTemporary = false)
        }
        bitmap.copy(Bitmap.Config.ARGB_8888, false)?.let {
            return UploadBitmap(it, isTemporary = true)
        }
        val converted = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Canvas(converted).drawBitmap(bitmap, 0f, 0f, null)
        return UploadBitmap(converted, isTemporary = true)
    }

    private fun createRenderTarget(width: Int, height: Int, halfFloat: Boolean): RenderTarget {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        val internalFormat = if (halfFloat) GLES30.GL_RGBA16F else GLES30.GL_RGBA
        val type = if (halfFloat) GLES30.GL_HALF_FLOAT else GLES30.GL_UNSIGNED_BYTE
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            internalFormat,
            width,
            height,
            0,
            GLES30.GL_RGBA,
            type,
            null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        val framebuffers = IntArray(1)
        GLES30.glGenFramebuffers(1, framebuffers, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, framebuffers[0])
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            textures[0],
            0
        )
        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            GLES30.glDeleteTextures(1, textures, 0)
            GLES30.glDeleteFramebuffers(1, framebuffers, 0)
            throw IllegalStateException("Incomplete Mertens render target: 0x${Integer.toHexString(status)}, halfFloat=$halfFloat")
        }
        checkGlError("createRenderTarget")
        return RenderTarget(width, height, textures[0], framebuffers[0])
    }

    private fun ensureInitialized(): Boolean {
        if (isInitialized) return true
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) return false
        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) return false

        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        val configAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE
        )
        if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)) return false
        val eglConfig = configs[0] ?: return false
        eglContext = EGL14.eglCreateContext(
            eglDisplay,
            eglConfig,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE),
            0
        )
        if (eglContext == EGL14.EGL_NO_CONTEXT) return false
        eglSurface = EGL14.eglCreatePbufferSurface(
            eglDisplay,
            eglConfig,
            intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE),
            0
        )
        if (eglSurface == EGL14.EGL_NO_SURFACE) return false
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) return false

        initBuffers()
        val vertexShader = compileShader(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER)
        weightProgram = linkProgram(vertexShader, WEIGHT_FRAGMENT_SHADER, "mertensWeight")
        normalizeProgram = linkProgram(vertexShader, NORMALIZE_FRAGMENT_SHADER, "mertensNormalize")
        pyrDownProgram = linkProgram(vertexShader, PYR_DOWN_FRAGMENT_SHADER, "mertensPyrDown")
        laplacianProgram = linkProgram(vertexShader, LAPLACIAN_FRAGMENT_SHADER, "mertensLaplacian")
        combineProgram = linkProgram(vertexShader, COMBINE_FRAGMENT_SHADER, "mertensCombine")
        reconstructProgram = linkProgram(vertexShader, RECONSTRUCT_FRAGMENT_SHADER, "mertensReconstruct")
        copyProgram = linkProgram(vertexShader, COPY_FRAGMENT_SHADER, "mertensCopy")
        GLES30.glDeleteShader(vertexShader)
        isInitialized = weightProgram != 0 &&
            normalizeProgram != 0 &&
            pyrDownProgram != 0 &&
            laplacianProgram != 0 &&
            combineProgram != 0 &&
            reconstructProgram != 0 &&
            copyProgram != 0
        return isInitialized
    }

    private fun initBuffers() {
        vertexBuffer = ByteBuffer.allocateDirect(VERTICES.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(VERTICES)
            position(0)
        }
        texCoordBuffer = ByteBuffer.allocateDirect(TEX_COORDS.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(TEX_COORDS)
            position(0)
        }
        indexBuffer = ByteBuffer.allocateDirect(DRAW_ORDER.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer().apply {
            put(DRAW_ORDER)
            position(0)
        }
    }

    private fun drawQuad(program: Int) {
        val positionHandle = GLES30.glGetAttribLocation(program, "aPosition")
        val texCoordHandle = GLES30.glGetAttribLocation(program, "aTexCoord")
        if (positionHandle >= 0) vertexBuffer?.let {
            it.position(0)
            GLES30.glEnableVertexAttribArray(positionHandle)
            GLES30.glVertexAttribPointer(positionHandle, 2, GLES30.GL_FLOAT, false, 0, it)
        }
        if (texCoordHandle >= 0) texCoordBuffer?.let {
            it.position(0)
            GLES30.glEnableVertexAttribArray(texCoordHandle)
            GLES30.glVertexAttribPointer(texCoordHandle, 2, GLES30.GL_FLOAT, false, 0, it)
        }
        indexBuffer?.let {
            it.position(0)
            GLES30.glDrawElements(GLES30.GL_TRIANGLES, DRAW_ORDER.size, GLES30.GL_UNSIGNED_SHORT, it)
        }
        if (positionHandle >= 0) GLES30.glDisableVertexAttribArray(positionHandle)
        if (texCoordHandle >= 0) GLES30.glDisableVertexAttribArray(texCoordHandle)
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            PLog.e(TAG, "Shader compile failed: ${GLES30.glGetShaderInfoLog(shader)}")
            GLES30.glDeleteShader(shader)
            return 0
        }
        return shader
    }

    private fun linkProgram(vertexShader: Int, fragmentSource: String, name: String): Int {
        val fragmentShader = compileShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource)
        if (vertexShader == 0 || fragmentShader == 0) return 0
        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)
        GLES30.glDeleteShader(fragmentShader)
        val linked = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            PLog.e(TAG, "$name link failed: ${GLES30.glGetProgramInfoLog(program)}")
            GLES30.glDeleteProgram(program)
            return 0
        }
        return program
    }

    private fun checkGlError(label: String) {
        val error = GLES30.glGetError()
        if (error != GLES30.GL_NO_ERROR) {
            throw IllegalStateException("$label GL error: 0x${Integer.toHexString(error)}")
        }
    }

    private interface FramebufferSource {
        val width: Int
        val height: Int
        val textureId: Int
    }

    private data class RenderTarget(
        override val width: Int,
        override val height: Int,
        override val textureId: Int,
        val framebufferId: Int,
    ) : FramebufferSource {
        private var released = false

        fun release() {
            if (released) return
            GLES30.glDeleteTextures(1, intArrayOf(textureId), 0)
            GLES30.glDeleteFramebuffers(1, intArrayOf(framebufferId), 0)
            released = true
        }
    }

    private data class TextureOnlyTarget(
        override val width: Int,
        override val height: Int,
        override val textureId: Int,
    ) : FramebufferSource

    private data class TextureLevel(
        override val width: Int,
        override val height: Int,
        override val textureId: Int,
        val owned: Boolean,
        val owner: RenderTarget? = null,
    ) : FramebufferSource {
        fun releaseIfOwned() {
            if (owned) {
                owner?.release() ?: GLES30.glDeleteTextures(1, intArrayOf(textureId), 0)
            }
        }
    }

    private data class UploadBitmap(
        val bitmap: Bitmap,
        val isTemporary: Boolean,
    ) {
        fun recycleIfTemporary() {
            if (isTemporary && !bitmap.isRecycled) {
                bitmap.recycle()
            }
        }
    }

    private val VERTICES = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
    private val TEX_COORDS = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)
    private val DRAW_ORDER = shortArrayOf(0, 1, 2, 1, 3, 2)

    private val VERTEX_SHADER = """
        #version 300 es
        in vec2 aPosition;
        in vec2 aTexCoord;
        out vec2 vTexCoord;
        void main() {
            gl_Position = vec4(aPosition, 0.0, 1.0);
            vTexCoord = aTexCoord;
        }
    """.trimIndent()

    private val COMMON_PYRAMID_GLSL = """
        int reflect101(int p, int size) {
            if (size <= 1) return 0;
            int period = size * 2 - 2;
            int m = p;
            if (m < 0) m = -m;
            m = m - (m / period) * period;
            return m >= size ? period - m : m;
        }

        float kernel5(int offset) {
            int a = offset < 0 ? -offset : offset;
            if (a == 0) return 6.0;
            if (a == 1) return 4.0;
            return 1.0;
        }

        vec4 pyrUpSample(sampler2D inputTexture, ivec2 dstCoord, ivec2 sourceSize) {
            vec4 sum = vec4(0.0);
            for (int y = -2; y <= 2; y++) {
                int syNumerator = dstCoord.y - y;
                if ((syNumerator < 0 ? -syNumerator : syNumerator) - ((syNumerator < 0 ? -syNumerator : syNumerator) / 2) * 2 != 0) {
                    continue;
                }
                int sy = reflect101(syNumerator / 2, sourceSize.y);
                float wy = kernel5(y);
                for (int x = -2; x <= 2; x++) {
                    int sxNumerator = dstCoord.x - x;
                    if ((sxNumerator < 0 ? -sxNumerator : sxNumerator) - ((sxNumerator < 0 ? -sxNumerator : sxNumerator) / 2) * 2 != 0) {
                        continue;
                    }
                    int sx = reflect101(sxNumerator / 2, sourceSize.x);
                    float wx = kernel5(x);
                    sum += texelFetch(inputTexture, ivec2(sx, sy), 0) * (wx * wy);
                }
            }
            return sum / 64.0;
        }
    """.trimIndent()

    private val WEIGHT_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp int;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uImage;
        uniform sampler2D uReferenceImage;
        uniform ivec2 uImageSize;
        uniform float uContrastWeight;
        uniform float uSaturationWeight;
        uniform float uExposureWeight;
        uniform float uExposureScale;
        uniform int uUseDeghostMask;

        int reflect101(int p, int size) {
            if (size <= 1) return 0;
            int period = size * 2 - 2;
            int m = p;
            if (m < 0) m = -m;
            m = m - (m / period) * period;
            return m >= size ? period - m : m;
        }

        float grayAt(ivec2 coord) {
            int x = reflect101(coord.x, uImageSize.x);
            int y = reflect101(coord.y, uImageSize.y);
            vec3 rgb = texelFetch(uImage, ivec2(x, y), 0).rgb;
            return dot(rgb, vec3(0.299, 0.587, 0.114));
        }

        float luma(vec3 rgb) {
            return dot(rgb, vec3(0.299, 0.587, 0.114));
        }

        float max3(vec3 value) {
            return max(max(value.r, value.g), value.b);
        }

        float validity(vec3 rgb) {
            float y = luma(rgb);
            float blackGate = smoothstep(0.055, 0.145, y);
            float whiteGate = 1.0 - smoothstep(0.865, 0.975, max3(rgb));
            return clamp(blackGate * whiteGate, 0.0, 1.0);
        }

        float referenceStructure(ivec2 coord) {
            float c = luma(texelFetch(uReferenceImage, coord, 0).rgb);
            float l = luma(texelFetch(uReferenceImage, ivec2(reflect101(coord.x - 1, uImageSize.x), coord.y), 0).rgb);
            float r = luma(texelFetch(uReferenceImage, ivec2(reflect101(coord.x + 1, uImageSize.x), coord.y), 0).rgb);
            float u = luma(texelFetch(uReferenceImage, ivec2(coord.x, reflect101(coord.y - 1, uImageSize.y)), 0).rgb);
            float d = luma(texelFetch(uReferenceImage, ivec2(coord.x, reflect101(coord.y + 1, uImageSize.y)), 0).rgb);
            float gradient = abs(r - l) + abs(d - u);
            float laplacian = abs(l + r + u + d - 4.0 * c);
            return smoothstep(0.030, 0.095, gradient + 0.5 * laplacian);
        }

        float censusMismatch(ivec2 coord, float refCenter, float sideCenter) {
            float mismatch = 0.0;
            float count = 0.0;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    if (x == 0 && y == 0) {
                        continue;
                    }
                    ivec2 p = ivec2(reflect101(coord.x + x, uImageSize.x), reflect101(coord.y + y, uImageSize.y));
                    float refNeighbor = luma(texelFetch(uReferenceImage, p, 0).rgb);
                    float sideNeighbor = luma(texelFetch(uImage, p, 0).rgb);
                    float refRank = refNeighbor > refCenter ? 1.0 : 0.0;
                    float sideRank = sideNeighbor > sideCenter ? 1.0 : 0.0;
                    mismatch += abs(refRank - sideRank);
                    count += 1.0;
                }
            }
            return mismatch / max(count, 1.0);
        }

        float deghostScoreAt(ivec2 coord) {
            int x = reflect101(coord.x, uImageSize.x);
            int y = reflect101(coord.y, uImageSize.y);
            vec3 side = texelFetch(uImage, ivec2(x, y), 0).rgb;
            vec3 reference = texelFetch(uReferenceImage, ivec2(x, y), 0).rgb;
            float scale = clamp(uExposureScale, 0.03125, 32.0);

            float sideLuma = luma(side);
            float referenceLuma = luma(reference);
            float comparable = validity(side) * validity(reference) * referenceStructure(ivec2(x, y));
            if (comparable <= 0.001) {
                return 0.0;
            }

            float exposureGap = abs(log2(scale));
            float logResidual = abs(log(max(sideLuma, 1e-4)) - log(max(referenceLuma, 1e-4)) - log(scale));
            float logMotion = smoothstep(0.28 + 0.04 * min(exposureGap, 3.0), 0.72, logResidual);
            float expectedSide = referenceLuma * scale;
            float linearResidual = abs(sideLuma - expectedSide) / max(max(sideLuma, expectedSide), 1e-4);
            float linearMotion = smoothstep(0.12, 0.42, linearResidual);
            float rankMotion = smoothstep(0.32, 0.68, censusMismatch(ivec2(x, y), referenceLuma, sideLuma));
            float score = max(max(linearMotion, logMotion), rankMotion);
            return comparable * score;
        }

        float computeDeghostAlpha(ivec2 coord) {
            if (uUseDeghostMask == 0) {
                return 1.0;
            }
            float score = 0.0;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    score = max(score, deghostScoreAt(coord + ivec2(x, y)));
                }
            }
            float reject = smoothstep(0.36, 0.76, score);
            return mix(1.0, 0.0, reject);
        }

        void main() {
            ivec2 coord = ivec2(gl_FragCoord.xy);
            vec3 rgb = texelFetch(uImage, coord, 0).rgb;
            float center = grayAt(coord);
            float contrast = abs(
                grayAt(coord + ivec2(-1, 0)) +
                grayAt(coord + ivec2(1, 0)) +
                grayAt(coord + ivec2(0, -1)) +
                grayAt(coord + ivec2(0, 1)) -
                center * 4.0
            );

            float mean = (rgb.r + rgb.g + rgb.b) / 3.0;
            vec3 deviation = rgb - vec3(mean);
            float saturation = sqrt(dot(deviation, deviation));

            vec3 expoDelta = rgb - vec3(0.5);
            vec3 expo = exp(-(expoDelta * expoDelta) / 0.08);
            float wellExposedness = expo.r * expo.g * expo.b;

            float weight =
                pow(max(contrast, 0.0), uContrastWeight) *
                pow(max(saturation, 0.0), uSaturationWeight) *
                pow(max(wellExposedness, 0.0), uExposureWeight);
            weight = weight * computeDeghostAlpha(coord) + 1e-12;
            fragColor = vec4(weight, weight, weight, 1.0);
        }
    """.trimIndent()

    private val NORMALIZE_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uWeight0;
        uniform sampler2D uWeight1;
        uniform sampler2D uWeight2;
        void main() {
            float w0 = texture(uWeight0, vTexCoord).r;
            float w1 = texture(uWeight1, vTexCoord).r;
            float w2 = texture(uWeight2, vTexCoord).r;
            float sum = max(w0 + w1 + w2, 1e-12);
            fragColor = vec4(w0 / sum, w1 / sum, w2 / sum, 1.0);
        }
    """.trimIndent()

    private val PYR_DOWN_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp int;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform ivec2 uSourceSize;

        int reflect101(int p, int size) {
            if (size <= 1) return 0;
            int period = size * 2 - 2;
            int m = p;
            if (m < 0) m = -m;
            m = m - (m / period) * period;
            return m >= size ? period - m : m;
        }

        float kernel5(int offset) {
            int a = offset < 0 ? -offset : offset;
            if (a == 0) return 6.0;
            if (a == 1) return 4.0;
            return 1.0;
        }

        void main() {
            ivec2 dst = ivec2(gl_FragCoord.xy);
            ivec2 center = dst * 2;
            vec4 sum = vec4(0.0);
            for (int y = -2; y <= 2; y++) {
                float wy = kernel5(y);
                int sy = reflect101(center.y + y, uSourceSize.y);
                for (int x = -2; x <= 2; x++) {
                    float wx = kernel5(x);
                    int sx = reflect101(center.x + x, uSourceSize.x);
                    sum += texelFetch(uInputTexture, ivec2(sx, sy), 0) * (wx * wy);
                }
            }
            fragColor = sum / 256.0;
        }
    """.trimIndent()

    private val LAPLACIAN_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp int;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uBaseTexture;
        uniform sampler2D uNextTexture;
        uniform ivec2 uNextSize;
        $COMMON_PYRAMID_GLSL
        void main() {
            ivec2 coord = ivec2(gl_FragCoord.xy);
            vec4 base = texelFetch(uBaseTexture, coord, 0);
            vec4 up = pyrUpSample(uNextTexture, coord, uNextSize);
            fragColor = vec4(base.rgb - up.rgb, 1.0);
        }
    """.trimIndent()

    private val COMBINE_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uImage0;
        uniform sampler2D uImage1;
        uniform sampler2D uImage2;
        uniform sampler2D uWeights;
        void main() {
            vec3 weights = texture(uWeights, vTexCoord).rgb;
            vec3 color =
                texture(uImage0, vTexCoord).rgb * weights.r +
                texture(uImage1, vTexCoord).rgb * weights.g +
                texture(uImage2, vTexCoord).rgb * weights.b;
            fragColor = vec4(color, 1.0);
        }
    """.trimIndent()

    private val RECONSTRUCT_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp int;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uBaseTexture;
        uniform sampler2D uNextTexture;
        uniform ivec2 uNextSize;
        $COMMON_PYRAMID_GLSL
        void main() {
            ivec2 coord = ivec2(gl_FragCoord.xy);
            vec4 base = texelFetch(uBaseTexture, coord, 0);
            vec4 up = pyrUpSample(uNextTexture, coord, uNextSize);
            fragColor = vec4(base.rgb + up.rgb, 1.0);
        }
    """.trimIndent()

    private val COPY_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        void main() {
            fragColor = vec4(clamp(texture(uInputTexture, vTexCoord).rgb, 0.0, 1.0), 1.0);
        }
    """.trimIndent()
}
