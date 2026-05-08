package com.hinnka.mycamera.hdr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Gainmap
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLUtils
import android.os.Build
import androidx.annotation.RequiresApi
import com.hinnka.mycamera.utils.PLog
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.Executors
import kotlin.system.measureTimeMillis

class GpuReferenceGainmapProducer : GainmapProducer {
    private val dispatcher = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "GpuReferenceGainmapProducer-GL").apply { isDaemon = true }
    }.asCoroutineDispatcher()

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var computeProgram = 0
    private var blurHProgram = 0
    private var blurVProgram = 0
    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null
    private var indexBuffer: ShortBuffer? = null
    private var isInitialized = false

    override suspend fun build(source: GainmapSourceSet, strength: Float): GainmapResult? = withContext(dispatcher) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) return@withContext null
        val config = configFor(source.sourceKind) ?: return@withContext null
        val hdrReference = source.hdrReference?.bitmap
        val sdrBase = source.sdrBase
        if (sdrBase.width <= 0 || sdrBase.height <= 0) {
            return@withContext null
        }
        if (config.requiresHdrReference && (hdrReference == null || hdrReference.width <= 0 || hdrReference.height <= 0)) {
            return@withContext null
        }
        if (!ensureInitialized()) return@withContext null

        var result: GainmapResult? = null
        val elapsed = measureTimeMillis {
            result = runCatching {
                renderGainmap(source, sdrBase, hdrReference, config, strength)
            }.onFailure {
                PLog.e(TAG, "GPU gainmap failed for ${source.sourceKind}", it)
            }.getOrNull()
        }
        PLog.d(TAG, "GPU gainmap build took ${elapsed}ms, source=${source.sourceKind}, success=${result != null}")
        result
    }

    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
    private fun renderGainmap(
        source: GainmapSourceSet,
        sdrBase: Bitmap,
        hdrReference: Bitmap?,
        config: Config,
        strength: Float,
    ): GainmapResult? {
        val width = (sdrBase.width / DOWNSAMPLE).coerceAtLeast(1)
        val height = (sdrBase.height / DOWNSAMPLE).coerceAtLeast(1)
        val fullHdrRatio = source.displayHdrSdrRatio.takeIf { it > 1f } ?: config.defaultFullHdrRatio

        val sdrUpload = prepareUploadBitmap(sdrBase)
        val hdrUpload = hdrReference?.let { prepareUploadBitmap(it) }
        val sdrTexture = uploadBitmapTexture(sdrUpload.bitmap)
        val hdrTexture = hdrUpload?.let { uploadBitmapTexture(it.bitmap) } ?: sdrTexture
        val computeTarget = createRenderTarget(width, height)
        val blurTarget = createRenderTarget(width, height)
        val finalTarget = createRenderTarget(width, height)

        try {
            renderComputePass(
                target = computeTarget,
                sdrTexture = sdrTexture,
                hdrTexture = hdrTexture,
                config = config,
                fullHdrRatio = fullHdrRatio,
                strength = strength,
            )
            renderBlurPass(
                program = blurHProgram,
                target = blurTarget,
                inputTexture = computeTarget.textureId,
                width = width,
                height = height,
            )
            renderBlurPass(
                program = blurVProgram,
                target = finalTarget,
                inputTexture = blurTarget.textureId,
                width = width,
                height = height,
            )

            val gainmapBitmap = readAlphaBitmap(width, height) ?: return null
            val gainmap = Gainmap(gainmapBitmap).apply {
                setRatioMin(config.minGainRatio, config.minGainRatio, config.minGainRatio)
                setRatioMax(config.maxGainRatio, config.maxGainRatio, config.maxGainRatio)
                setGamma(1.0f, 1.0f, 1.0f)
                setEpsilonSdr(EPSILON, EPSILON, EPSILON)
                setEpsilonHdr(EPSILON, EPSILON, EPSILON)
                setMinDisplayRatioForHdrTransition(config.minDisplayRatioForHdrTransition)
                setDisplayRatioForFullHdr(fullHdrRatio)
            }
            return GainmapResult(gainmap, source.sourceKind, source.confidence)
        } finally {
            GLES30.glDeleteTextures(1, intArrayOf(sdrTexture), 0)
            if (hdrTexture != sdrTexture) {
                GLES30.glDeleteTextures(1, intArrayOf(hdrTexture), 0)
            }
            computeTarget.release()
            blurTarget.release()
            finalTarget.release()
            sdrUpload.recycleIfTemporary()
            hdrUpload?.recycleIfTemporary()
        }
    }

    private fun renderComputePass(
        target: RenderTarget,
        sdrTexture: Int,
        hdrTexture: Int,
        config: Config,
        fullHdrRatio: Float,
        strength: Float,
    ) {
        GLES30.glUseProgram(computeProgram)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, target.framebufferId)
        GLES30.glViewport(0, 0, target.width, target.height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sdrTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(computeProgram, "uSdrTexture"), 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, hdrTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(computeProgram, "uHdrTexture"), 1)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(computeProgram, "uFullHdrRatio"), fullHdrRatio)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(computeProgram, "uMinGainRatio"), config.minGainRatio)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(computeProgram, "uMaxGainRatio"), config.maxGainRatio)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(computeProgram, "uStrength"), strength)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(computeProgram, "uHasHdrReference"), if (config.requiresHdrReference) 1 else 0)
        drawQuad(computeProgram)
        checkGlError("renderComputePass")
    }

    private fun renderBlurPass(program: Int, target: RenderTarget, inputTexture: Int, width: Int, height: Int) {
        GLES30.glUseProgram(program)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, target.framebufferId)
        GLES30.glViewport(0, 0, target.width, target.height)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uInputTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(program, "uTexelSize"), 1f / width, 1f / height)
        drawQuad(program)
        checkGlError("renderBlurPass")
    }

    private fun readAlphaBitmap(width: Int, height: Int): Bitmap? {
        val rgba = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
        GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, rgba)
        checkGlError("readAlphaBitmap")
        rgba.position(0)

        val alpha = ByteArray(width * height)
        for (y in 0 until height) {
            val srcRow = height - 1 - y
            val srcOffset = srcRow * width * 4
            val dstOffset = y * width
            for (x in 0 until width) {
                alpha[dstOffset + x] = rgba.get(srcOffset + x * 4)
            }
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8).also {
            it.copyPixelsFromBuffer(ByteBuffer.wrap(alpha))
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

    private fun createRenderTarget(width: Int, height: Int): RenderTarget {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RGBA,
            width,
            height,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_UNSIGNED_BYTE,
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
        checkGlError("createRenderTarget")
        return RenderTarget(width, height, textures[0], framebuffers[0])
    }

    private fun ensureInitialized(): Boolean {
        if (isInitialized) return true
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) return false
        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) return false
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
        computeProgram = linkProgram(vertexShader, GAINMAP_FRAGMENT_SHADER, "gainmapCompute")
        blurHProgram = linkProgram(vertexShader, BLUR_H_FRAGMENT_SHADER, "gainmapBlurH")
        blurVProgram = linkProgram(vertexShader, BLUR_V_FRAGMENT_SHADER, "gainmapBlurV")
        GLES30.glDeleteShader(vertexShader)
        isInitialized = computeProgram != 0 && blurHProgram != 0 && blurVProgram != 0
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
        vertexBuffer?.let {
            it.position(0)
            GLES30.glEnableVertexAttribArray(positionHandle)
            GLES30.glVertexAttribPointer(positionHandle, 2, GLES30.GL_FLOAT, false, 0, it)
        }
        texCoordBuffer?.let {
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

    private fun configFor(sourceKind: SourceKind): Config? {
        return when (sourceKind) {
            SourceKind.RAW -> Config(maxGainRatio = 4.5f, defaultFullHdrRatio = 1.35f, requiresHdrReference = true)
            SourceKind.HLG_CAPTURE -> Config(maxGainRatio = 3.5f, defaultFullHdrRatio = 1.45f, requiresHdrReference = true)
            SourceKind.SDR_BITMAP -> Config(
                maxGainRatio = 4.0f,
                defaultFullHdrRatio = 1.8f,
                minDisplayRatioForHdrTransition = 1.02f,
                requiresHdrReference = false,
            )
        }
    }

    private data class Config(
        val minGainRatio: Float = 1.0f,
        val maxGainRatio: Float,
        val defaultFullHdrRatio: Float,
        val minDisplayRatioForHdrTransition: Float = 1.0f,
        val requiresHdrReference: Boolean,
    )

    private data class RenderTarget(
        val width: Int,
        val height: Int,
        val textureId: Int,
        val framebufferId: Int,
    ) {
        fun release() {
            GLES30.glDeleteTextures(1, intArrayOf(textureId), 0)
            GLES30.glDeleteFramebuffers(1, intArrayOf(framebufferId), 0)
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

    companion object {
        private const val TAG = "GpuReferenceGainmapProducer"
        private const val DOWNSAMPLE = 4
        private const val EPSILON = 1e-4f
        private val VERTICES = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val TEX_COORDS = floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f)
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

        private val GAINMAP_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            in vec2 vTexCoord;
            out vec4 fragColor;
            uniform sampler2D uSdrTexture;
            uniform sampler2D uHdrTexture;
            uniform float uFullHdrRatio;
            uniform float uMinGainRatio;
            uniform float uMaxGainRatio;
            uniform float uStrength;
            uniform int uHasHdrReference;

            float srgbToLinear(float value) {
                return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4);
            }

            float gainmapSmoothstep(float edge0, float edge1, float x) {
                float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
                return t * t * (3.0 - 2.0 * t);
            }

            float displayLuma(float sceneLuma, float fullHdrRatio) {
                float effectiveHeadroom = clamp(fullHdrRatio, 1.15, 3.2);
                if (sceneLuma <= 1.0) return max(sceneLuma, 0.0);
                float systemGamma = clamp(1.03 + (effectiveHeadroom - 1.0) * 0.16, 1.0, 1.25);
                float normalized = pow(max(sceneLuma, 0.0), systemGamma);
                float aboveWhite = max(normalized - 1.0, 0.0);
                float compressedBoost = 1.0 - exp(-aboveWhite * 0.85);
                return 1.0 + (effectiveHeadroom - 1.0) * compressedBoost;
            }

            void main() {
                vec3 sdrEncoded = texture(uSdrTexture, vTexCoord).rgb;
                vec3 sdr = vec3(
                    srgbToLinear(sdrEncoded.r),
                    srgbToLinear(sdrEncoded.g),
                    srgbToLinear(sdrEncoded.b)
                );
                float sdrLuma = max(dot(sdr, vec3(0.2126, 0.7152, 0.0722)), 0.0);
                float maxChannel = max(sdr.r, max(sdr.g, sdr.b));
                float minChannel = min(sdr.r, min(sdr.g, sdr.b));
                float saturation = maxChannel <= 0.0001 ? 0.0 : clamp((maxChannel - minChannel) / maxChannel, 0.0, 1.0);

                float displayHeadroom = clamp(uFullHdrRatio, 1.1, uMaxGainRatio);
                float tonalPosition = max(sdrLuma * 0.72 + maxChannel * 0.28, 0.0);
                float globalRamp = pow(gainmapSmoothstep(0.02, 0.96, tonalPosition), 0.82);
                float shoulderRamp = pow(gainmapSmoothstep(0.34, 1.0, maxChannel), 1.35);
                float chromaPenalty = 1.0 - saturation * 0.10;
                float lift = (0.035 + 0.4 * globalRamp + 0.34 * shoulderRamp) * chromaPenalty;
                float toneRatio = clamp(1.0 + (displayHeadroom - 1.0) * lift, 1.0, uMaxGainRatio);

                float targetRatio = toneRatio;
                if (uHasHdrReference == 1) {
                    vec3 hdr = max(texture(uHdrTexture, vTexCoord).rgb, vec3(0.0));
                    float hdrSceneLuma = max(dot(hdr, vec3(0.2627, 0.6780, 0.0593)), 0.0);
                    float hdrDisplayLuma = displayLuma(hdrSceneLuma, uFullHdrRatio);
                    float referenceRatio = clamp(hdrDisplayLuma / max(sdrLuma, 0.0001), uMinGainRatio, uMaxGainRatio);
                    float referenceWeight = clamp(
                        gainmapSmoothstep(0.05, 0.98, sdrLuma * 0.70 + maxChannel * 0.30) *
                        (0.35 * gainmapSmoothstep(0.65, 2.0, hdrSceneLuma) + 0.65 * gainmapSmoothstep(0.01, 0.28, hdrDisplayLuma - sdrLuma)),
                        0.0,
                        1.0
                    );
                    targetRatio = toneRatio + max(referenceRatio - toneRatio, 0.0) * referenceWeight * 0.82;
                }
                targetRatio = clamp(targetRatio, uMinGainRatio, uMaxGainRatio);
                float normalizedStrength = clamp(uStrength, 0.25, 2.0);
                float strengthRatio = clamp((targetRatio - 1.0) * normalizedStrength + 1.0, uMinGainRatio, uMaxGainRatio);
                float encoded = log(strengthRatio / uMinGainRatio) / log(uMaxGainRatio / uMinGainRatio);
                fragColor = vec4(vec3(clamp(encoded, 0.0, 1.0)), 1.0);
            }
        """.trimIndent()

        private val BLUR_H_FRAGMENT_SHADER = blurShader(horizontal = true)
        private val BLUR_V_FRAGMENT_SHADER = blurShader(horizontal = false)

        private fun blurShader(horizontal: Boolean): String {
            val offset = if (horizontal) "vec2(float(i) * uTexelSize.x, 0.0)" else "vec2(0.0, float(i) * uTexelSize.y)"
            return """
                #version 300 es
                precision highp float;
                in vec2 vTexCoord;
                out vec4 fragColor;
                uniform sampler2D uInputTexture;
                uniform vec2 uTexelSize;
                void main() {
                    float sum = 0.0;
                    for (int i = -3; i <= 3; i++) {
                        sum += texture(uInputTexture, vTexCoord + $offset).r;
                    }
                    float value = sum / 7.0;
                    fragColor = vec4(vec3(value), 1.0);
                }
            """.trimIndent()
        }
    }
}
