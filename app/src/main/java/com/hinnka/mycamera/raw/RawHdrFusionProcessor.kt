package com.hinnka.mycamera.raw

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import com.hinnka.mycamera.utils.LargeDirectBuffer
import com.hinnka.mycamera.utils.PLog
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.Executors
import kotlin.system.measureTimeMillis

data class RawHdrFusionResult(
    var fusedBayerBuffer: ByteBuffer?,
    val width: Int,
    val height: Int,
    val blackLevel: FloatArray,
    val fusedBayerUsesNativeAllocator: Boolean = true,
)

object RawHdrFusionProcessor {
    private const val TAG = "RawHdrFusion"
    private const val FRAME_COUNT = 3
    private const val REFERENCE_FRAME_INDEX = 0
    private const val VALUE_DOMAIN_SENSOR = 0
    private const val VALUE_DOMAIN_NORMALIZED_SENSOR_RANGE = 1

    data class InputFrame(
        val rawBuffer: ByteBuffer,
        val width: Int,
        val height: Int,
        val rowStrideBytes: Int,
        val exposureProduct: Double,
        val valueDomain: Int,
        val onUploaded: (() -> Unit)? = null,
    )

    private val dispatcher = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "RawHdrFusion-GL").apply { isDaemon = true }
    }.asCoroutineDispatcher()

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null
    private var indexBuffer: ShortBuffer? = null
    private var fusionProgram = 0
    private var isInitialized = false

    suspend fun fuse(
        frames: List<InputFrame>,
        cfaPattern: Int,
        blackLevel: FloatArray,
        whiteLevel: Int,
        noiseModel: FloatArray,
        enableDeghostMask: Boolean = true,
    ): RawHdrFusionResult? = withContext(dispatcher) {
        if (frames.size != FRAME_COUNT) {
            PLog.w(TAG, "Expected $FRAME_COUNT RAW HDR frames, got ${frames.size}")
            return@withContext null
        }
        val width = frames.first().width
        val height = frames.first().height
        if (width <= 0 || height <= 0 || frames.any { it.width != width || it.height != height }) {
            PLog.w(TAG, "Invalid RAW HDR frame dimensions")
            return@withContext null
        }
        if (!ensureInitialized()) return@withContext null

        val exposureProducts = frames.map { it.exposureProduct.coerceAtLeast(1.0) }
        val baseExposure = exposureProducts.minOrNull() ?: 1.0
        val exposureScales = FloatArray(FRAME_COUNT) { index ->
            (baseExposure / exposureProducts[index]).toFloat().coerceIn(0.0001f, 1.0f)
        }
        val normalizedBlackLevel = FloatArray(4) { index ->
            blackLevel.getOrElse(index) { blackLevel.firstOrNull() ?: 0f }
        }
        val normalizedNoiseModel = floatArrayOf(
            noiseModel.getOrElse(0) { 0f }.coerceAtLeast(0f),
            noiseModel.getOrElse(1) { 0f }.coerceAtLeast(0f),
        )
        var outputBuffer: ByteBuffer? = null
        var returned = false
        try {
            val byteCount = width.toLong() * height.toLong() * 2L
            outputBuffer = LargeDirectBuffer.allocate(byteCount, "RAW HDR fused Bayer")?.order(ByteOrder.nativeOrder())
            if (outputBuffer == null) return@withContext null

            val elapsed = measureTimeMillis {
                renderFusion(
                    frames = frames,
                    outputBuffer = outputBuffer,
                    cfaPattern = cfaPattern,
                    blackLevel = normalizedBlackLevel,
                    whiteLevel = whiteLevel,
                    exposureScales = exposureScales,
                    noiseModel = normalizedNoiseModel,
                    enableDeghostMask = enableDeghostMask,
                )
            }
            outputBuffer.rewind()
            returned = true
            PLog.d(
                TAG,
                "RAW HDR fusion took ${elapsed}ms, size=${width}x$height, deghostMask=$enableDeghostMask, " +
                        "exposureScales=${exposureScales.joinToString()}, valueDomains=${frames.joinToString { it.valueDomain.toString() }}"
            )
            RawHdrFusionResult(
                fusedBayerBuffer = outputBuffer,
                width = width,
                height = height,
                blackLevel = normalizedBlackLevel,
            )
        } catch (e: Exception) {
            PLog.e(TAG, "RAW HDR fusion failed", e)
            null
        } finally {
            if (!returned) {
                LargeDirectBuffer.free(outputBuffer)
            }
        }
    }

    fun sensorValueDomain(): Int = VALUE_DOMAIN_SENSOR

    fun normalizedSensorRangeValueDomain(): Int = VALUE_DOMAIN_NORMALIZED_SENSOR_RANGE

    private fun renderFusion(
        frames: List<InputFrame>,
        outputBuffer: ByteBuffer,
        cfaPattern: Int,
        blackLevel: FloatArray,
        whiteLevel: Int,
        exposureScales: FloatArray,
        noiseModel: FloatArray,
        enableDeghostMask: Boolean,
    ) {
        val inputTextures = frames.map { uploadRawTexture(it) }
        var target: RenderTarget? = null
        val previousState = captureRenderState()
        try {
            applyRawRenderState()
            target = createRawRenderTarget(frames.first().width, frames.first().height)
            GLES30.glUseProgram(fusionProgram)
            bindTarget(target)
            inputTextures.forEachIndexed { index, textureId ->
                bindTexture(fusionProgram, "uFrame$index", index, textureId)
            }
            GLES30.glUniform2i(
                GLES30.glGetUniformLocation(fusionProgram, "uSize"),
                target.width,
                target.height
            )
            GLES30.glUniform1i(GLES30.glGetUniformLocation(fusionProgram, "uCfaPattern"), cfaPattern)
            GLES30.glUniform1fv(GLES30.glGetUniformLocation(fusionProgram, "uBlackLevel[0]"), 4, blackLevel, 0)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(fusionProgram, "uWhiteLevel"), whiteLevel.toFloat())
            GLES30.glUniform1fv(
                GLES30.glGetUniformLocation(fusionProgram, "uExposureScale[0]"),
                FRAME_COUNT,
                exposureScales,
                0
            )
            val valueDomains = frames.map { it.valueDomain }.toIntArray()
            GLES30.glUniform1iv(
                GLES30.glGetUniformLocation(fusionProgram, "uValueDomain[0]"),
                FRAME_COUNT,
                valueDomains,
                0
            )
            GLES30.glUniform2f(
                GLES30.glGetUniformLocation(fusionProgram, "uNoiseModel"),
                noiseModel[0],
                noiseModel[1]
            )
            GLES30.glUniform1i(
                GLES30.glGetUniformLocation(fusionProgram, "uUseDeghostMask"),
                if (enableDeghostMask) 1 else 0
            )
            drawQuad(fusionProgram)
            checkGlError("renderFusion")
            readRawTarget(target, outputBuffer)
        } finally {
            inputTextures.forEach { GLES30.glDeleteTextures(1, intArrayOf(it), 0) }
            target?.release()
            previousState.restore()
        }
    }

    private fun uploadRawTexture(frame: InputFrame): Int {
        require(frame.rowStrideBytes >= frame.width * 2) {
            "RAW row stride ${frame.rowStrideBytes} is smaller than width ${frame.width}"
        }
        require(frame.rowStrideBytes % 2 == 0) {
            "RAW row stride must be 16-bit aligned: ${frame.rowStrideBytes}"
        }
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, frame.rowStrideBytes / 2)
        val uploadBuffer = frame.rawBuffer.duplicate().order(ByteOrder.nativeOrder()).apply {
            position(0)
        }
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_R16UI,
            frame.width,
            frame.height,
            0,
            GLES30.GL_RED_INTEGER,
            GLES30.GL_UNSIGNED_SHORT,
            uploadBuffer
        )
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)
        checkGlError("uploadRawTexture")
        frame.onUploaded?.invoke()
        return textures[0]
    }

    private fun createRawRenderTarget(width: Int, height: Int): RenderTarget {
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textures[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
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
            throw IllegalStateException("Incomplete RAW HDR render target: 0x${Integer.toHexString(status)}")
        }
        checkGlError("createRawRenderTarget")
        return RenderTarget(width, height, textures[0], framebuffers[0])
    }

    private fun readRawTarget(target: RenderTarget, outputBuffer: ByteBuffer) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, target.framebufferId)
        GLES30.glViewport(0, 0, target.width, target.height)
        GLES30.glPixelStorei(GLES30.GL_PACK_ALIGNMENT, 1)
        val packedByteCount = target.width.toLong() * target.height.toLong() * 4L
        val packed = LargeDirectBuffer.allocate(packedByteCount, "RAW HDR packed RGBA readback")
            ?: throw IllegalStateException("Failed to allocate RAW HDR packed readback buffer")
        try {
            packed.clear()
            GLES30.glReadPixels(
                0,
                0,
                target.width,
                target.height,
                GLES30.GL_RGBA,
                GLES30.GL_UNSIGNED_BYTE,
                packed
            )
            checkGlError("readRawTarget")
            packed.rewind()
            outputBuffer.clear()
            val output = outputBuffer.order(ByteOrder.nativeOrder()).asShortBuffer()
            val pixelCount = target.width * target.height
            for (index in 0 until pixelCount) {
                val lo = packed.get(index * 4).toInt() and 0xFF
                val hi = packed.get(index * 4 + 1).toInt() and 0xFF
                output.put(index, ((hi shl 8) or lo).toShort())
            }
            outputBuffer.rewind()
        } finally {
            LargeDirectBuffer.free(packed)
        }
    }

    private fun bindTarget(target: RenderTarget) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, target.framebufferId)
        GLES30.glDrawBuffers(1, intArrayOf(GLES30.GL_COLOR_ATTACHMENT0), 0)
        GLES30.glViewport(0, 0, target.width, target.height)
        GLES30.glClearBufferuiv(GLES30.GL_COLOR, 0, intArrayOf(0, 0, 0, 0), 0)
    }

    private fun captureRenderState(): RenderState {
        return RenderState(
            blend = GLES30.glIsEnabled(GLES30.GL_BLEND),
            dither = GLES30.glIsEnabled(GLES30.GL_DITHER),
            scissor = GLES30.glIsEnabled(GLES30.GL_SCISSOR_TEST),
            depth = GLES30.glIsEnabled(GLES30.GL_DEPTH_TEST),
            stencil = GLES30.glIsEnabled(GLES30.GL_STENCIL_TEST),
            cullFace = GLES30.glIsEnabled(GLES30.GL_CULL_FACE),
        )
    }

    private fun applyRawRenderState() {
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDisable(GLES30.GL_DITHER)
        GLES30.glDisable(GLES30.GL_SCISSOR_TEST)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_STENCIL_TEST)
        GLES30.glDisable(GLES30.GL_CULL_FACE)
    }

    private fun bindTexture(program: Int, uniformName: String, unit: Int, textureId: Int) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, uniformName), unit)
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
        fusionProgram = linkProgram(vertexShader, FUSION_FRAGMENT_SHADER, "rawHdrFusion")
        GLES30.glDeleteShader(vertexShader)
        isInitialized = fusionProgram != 0
        return isInitialized
    }

    private fun initBuffers() {
        vertexBuffer = ByteBuffer.allocateDirect(VERTICES.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(VERTICES)
                position(0)
            }
        texCoordBuffer = ByteBuffer.allocateDirect(TEX_COORDS.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(TEX_COORDS)
                position(0)
            }
        indexBuffer = ByteBuffer.allocateDirect(DRAW_ORDER.size * 2)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
            .apply {
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

    private data class RenderTarget(
        val width: Int,
        val height: Int,
        val textureId: Int,
        val framebufferId: Int,
    ) {
        private var released = false

        fun release() {
            if (released) return
            GLES30.glDeleteTextures(1, intArrayOf(textureId), 0)
            GLES30.glDeleteFramebuffers(1, intArrayOf(framebufferId), 0)
            released = true
        }
    }

    private data class RenderState(
        val blend: Boolean,
        val dither: Boolean,
        val scissor: Boolean,
        val depth: Boolean,
        val stencil: Boolean,
        val cullFace: Boolean,
    ) {
        fun restore() {
            set(GLES30.GL_BLEND, blend)
            set(GLES30.GL_DITHER, dither)
            set(GLES30.GL_SCISSOR_TEST, scissor)
            set(GLES30.GL_DEPTH_TEST, depth)
            set(GLES30.GL_STENCIL_TEST, stencil)
            set(GLES30.GL_CULL_FACE, cullFace)
        }

        private fun set(capability: Int, enabled: Boolean) {
            if (enabled) {
                GLES30.glEnable(capability)
            } else {
                GLES30.glDisable(capability)
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

    private val FUSION_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp int;
        precision highp usampler2D;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform highp usampler2D uFrame0;
        uniform highp usampler2D uFrame1;
        uniform highp usampler2D uFrame2;
        uniform ivec2 uSize;
        uniform int uCfaPattern;
        uniform float uBlackLevel[4];
        uniform float uWhiteLevel;
        uniform float uExposureScale[3];
        uniform int uValueDomain[3];
        uniform vec2 uNoiseModel;
        uniform int uUseDeghostMask;

        int channelIndex(ivec2 p) {
            int x = p.x - (p.x / 2) * 2;
            int y = p.y - (p.y / 2) * 2;
            if (uCfaPattern == 0) {
                if (y == 0 && x == 0) return 0;
                if (y == 0 && x == 1) return 1;
                if (y == 1 && x == 0) return 2;
                return 3;
            }
            if (uCfaPattern == 1) {
                if (y == 0 && x == 0) return 1;
                if (y == 0 && x == 1) return 0;
                if (y == 1 && x == 0) return 3;
                return 2;
            }
            if (uCfaPattern == 2) {
                if (y == 0 && x == 0) return 2;
                if (y == 0 && x == 1) return 3;
                if (y == 1 && x == 0) return 0;
                return 1;
            }
            if (y == 0 && x == 0) return 3;
            if (y == 0 && x == 1) return 2;
            if (y == 1 && x == 0) return 1;
            return 0;
        }

        uint rawSample(int frame, ivec2 p) {
            if (frame == 0) return texelFetch(uFrame0, p, 0).r;
            if (frame == 1) return texelFetch(uFrame1, p, 0).r;
            return texelFetch(uFrame2, p, 0).r;
        }

        float rawToNormalized(uint rawValue, int channel, int domain) {
            float value = float(rawValue);
            if (domain == 1) {
                return clamp(value / 65535.0, 0.0, 1.0);
            }
            float black = uBlackLevel[channel];
            float range = max(1.0, uWhiteLevel - black);
            return clamp((value - black) / range, 0.0, 1.0);
        }

        float frameNorm(int frame, ivec2 p, int channel) {
            return rawToNormalized(rawSample(frame, p), channel, uValueDomain[frame]);
        }

        ivec2 bayerTileBase(ivec2 p) {
            int x = p.x - (p.x / 2) * 2;
            int y = p.y - (p.y / 2) * 2;
            return p - ivec2(x, y);
        }

        float frameTileMaxNorm(int frame, ivec2 p) {
            ivec2 base = bayerTileBase(p);
            float maxNorm = 0.0;
            for (int y = 0; y <= 1; y++) {
                for (int x = 0; x <= 1; x++) {
                    ivec2 q = clamp(base + ivec2(x, y), ivec2(0), uSize - ivec2(1));
                    maxNorm = max(maxNorm, frameNorm(frame, q, channelIndex(q)));
                }
            }
            return maxNorm;
        }

        float frameHighlightGate(int frame, ivec2 p) {
            return 1.0 - smoothstep(0.90, 0.995, frameTileMaxNorm(frame, p));
        }

        int clampSameParity(int coord, int offset, int maxCoord) {
            int q = coord + offset;
            if (q < 0) q = coord - offset;
            if (q > maxCoord) q = coord - offset;
            return clamp(q, 0, maxCoord);
        }

        ivec2 sameColorCoord(ivec2 p, ivec2 offset) {
            int x = offset.x == 0 ? p.x : clampSameParity(p.x, offset.x, uSize.x - 1);
            int y = offset.y == 0 ? p.y : clampSameParity(p.y, offset.y, uSize.y - 1);
            return ivec2(x, y);
        }

        float contrastWeight(int frame, ivec2 p, int channel) {
            float c = frameNorm(frame, p, channel) * uExposureScale[frame];
            float left = frameNorm(frame, sameColorCoord(p, ivec2(-2, 0)), channel) * uExposureScale[frame];
            float right = frameNorm(frame, sameColorCoord(p, ivec2(2, 0)), channel) * uExposureScale[frame];
            float up = frameNorm(frame, sameColorCoord(p, ivec2(0, -2)), channel) * uExposureScale[frame];
            float down = frameNorm(frame, sameColorCoord(p, ivec2(0, 2)), channel) * uExposureScale[frame];
            float laplacian = abs(c - 0.25 * (left + right + up + down));
            return pow(laplacian + 0.0005, 0.25);
        }

        float noiseVariance(int frame, float norm, int channel) {
            float sensorRange = max(1.0, uWhiteLevel - uBlackLevel[channel]);
            float sensorSignal = norm * sensorRange;
            float variance = max(0.0000001, (uNoiseModel.x * sensorSignal + uNoiseModel.y) / (sensorRange * sensorRange));
            float scale = uExposureScale[frame];
            return variance * scale * scale;
        }

        float scaledLinearAt(int frame, ivec2 p, int channel) {
            return frameNorm(frame, p, channel) * uExposureScale[frame];
        }

        float captureTrust(int frame, ivec2 p, int channel) {
            float norm = frameNorm(frame, p, channel);
            float scaledSignal = norm * uExposureScale[frame];
            float sigma = sqrt(noiseVariance(frame, norm, channel));
            float snr = scaledSignal / max(sigma, 0.000001);
            float shadowGate = smoothstep(2.0, 6.0, snr);
            float highlightGate = frameHighlightGate(frame, p);
            return clamp(shadowGate * highlightGate, 0.0, 1.0);
        }

        float tileGreenScaledAt(int frame, ivec2 p) {
            ivec2 base = bayerTileBase(p);
            float sum = 0.0;
            float count = 0.0;
            for (int y = 0; y <= 1; y++) {
                for (int x = 0; x <= 1; x++) {
                    ivec2 q = clamp(base + ivec2(x, y), ivec2(0), uSize - ivec2(1));
                    int channel = channelIndex(q);
                    if (channel == 1 || channel == 2) {
                        sum += scaledLinearAt(frame, q, channel);
                        count += 1.0;
                    }
                }
            }
            return sum / max(count, 1.0);
        }

        float tileMaxRelativeResidual(int frame, ivec2 p) {
            ivec2 base = bayerTileBase(p);
            float residual = 0.0;
            for (int y = 0; y <= 1; y++) {
                for (int x = 0; x <= 1; x++) {
                    ivec2 q = clamp(base + ivec2(x, y), ivec2(0), uSize - ivec2(1));
                    int channel = channelIndex(q);
                    float side = scaledLinearAt(frame, q, channel);
                    float reference = scaledLinearAt(${REFERENCE_FRAME_INDEX}, q, channel);
                    residual = max(residual, abs(side - reference) / max(max(side, reference), 1e-5));
                }
            }
            return residual;
        }

        float tileTrust(int frame, ivec2 p) {
            float signal = tileGreenScaledAt(frame, p);
            float shadowGate = smoothstep(0.003, 0.025, signal);
            float highlightGate = frameHighlightGate(frame, p);
            return clamp(shadowGate * highlightGate, 0.0, 1.0);
        }

        float referenceTileStructure(ivec2 p) {
            float c = tileGreenScaledAt(${REFERENCE_FRAME_INDEX}, p);
            float left = tileGreenScaledAt(${REFERENCE_FRAME_INDEX}, p + ivec2(-2, 0));
            float right = tileGreenScaledAt(${REFERENCE_FRAME_INDEX}, p + ivec2(2, 0));
            float up = tileGreenScaledAt(${REFERENCE_FRAME_INDEX}, p + ivec2(0, -2));
            float down = tileGreenScaledAt(${REFERENCE_FRAME_INDEX}, p + ivec2(0, 2));
            float gradient = abs(right - left) + abs(down - up);
            float laplacian = abs(left + right + up + down - 4.0 * c);
            return smoothstep(0.002, 0.018, gradient + 0.5 * laplacian);
        }

        float tileCensusMismatch(int frame, ivec2 p, float refCenter, float sideCenter) {
            float mismatch = 0.0;
            float count = 0.0;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    if (x == 0 && y == 0) {
                        continue;
                    }
                    ivec2 q = p + ivec2(x * 2, y * 2);
                    float refNeighbor = tileGreenScaledAt(${REFERENCE_FRAME_INDEX}, q);
                    float sideNeighbor = tileGreenScaledAt(frame, q);
                    float refRank = refNeighbor > refCenter ? 1.0 : 0.0;
                    float sideRank = sideNeighbor > sideCenter ? 1.0 : 0.0;
                    mismatch += abs(refRank - sideRank);
                    count += 1.0;
                }
            }
            return mismatch / max(count, 1.0);
        }

        float deghostTileScoreAt(int frame, ivec2 p) {
            float sideSignal = tileGreenScaledAt(frame, p);
            float referenceSignal = tileGreenScaledAt(${REFERENCE_FRAME_INDEX}, p);
            float comparable = tileTrust(frame, p) * tileTrust(${REFERENCE_FRAME_INDEX}, p) * referenceTileStructure(p);
            if (comparable <= 0.001) {
                return 0.0;
            }

            float relativeResidual = abs(sideSignal - referenceSignal) / max(max(sideSignal, referenceSignal), 1e-5);
            float linearMotion = smoothstep(0.10, 0.34, max(relativeResidual, tileMaxRelativeResidual(frame, p)));
            float logResidual = abs(log(max(sideSignal, 1e-5)) - log(max(referenceSignal, 1e-5)));
            float logMotion = smoothstep(0.22, 0.62, logResidual);
            float rankMotion = smoothstep(0.25, 0.62, tileCensusMismatch(frame, p, referenceSignal, sideSignal));
            return comparable * max(max(linearMotion, logMotion), rankMotion);
        }

        float referenceStructure(ivec2 p, int channel) {
            float c = scaledLinearAt(${REFERENCE_FRAME_INDEX}, p, channel);
            float left = scaledLinearAt(${REFERENCE_FRAME_INDEX}, sameColorCoord(p, ivec2(-2, 0)), channel);
            float right = scaledLinearAt(${REFERENCE_FRAME_INDEX}, sameColorCoord(p, ivec2(2, 0)), channel);
            float up = scaledLinearAt(${REFERENCE_FRAME_INDEX}, sameColorCoord(p, ivec2(0, -2)), channel);
            float down = scaledLinearAt(${REFERENCE_FRAME_INDEX}, sameColorCoord(p, ivec2(0, 2)), channel);
            float gradient = abs(right - left) + abs(down - up);
            float laplacian = abs(left + right + up + down - 4.0 * c);
            float referenceNorm = frameNorm(${REFERENCE_FRAME_INDEX}, p, channel);
            float sigma = sqrt(noiseVariance(${REFERENCE_FRAME_INDEX}, referenceNorm, channel));
            float structureFloor = max(0.0015, sigma * 3.0);
            return smoothstep(structureFloor * 2.0, structureFloor * 8.0, gradient + 0.5 * laplacian);
        }

        float censusMismatch(int frame, ivec2 p, int channel, float refCenter, float sideCenter) {
            float mismatch = 0.0;
            float count = 0.0;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    if (x == 0 && y == 0) {
                        continue;
                    }
                    ivec2 coord = sameColorCoord(p, ivec2(x * 2, y * 2));
                    float refNeighbor = scaledLinearAt(${REFERENCE_FRAME_INDEX}, coord, channel);
                    float sideNeighbor = scaledLinearAt(frame, coord, channel);
                    float refRank = refNeighbor > refCenter ? 1.0 : 0.0;
                    float sideRank = sideNeighbor > sideCenter ? 1.0 : 0.0;
                    mismatch += abs(refRank - sideRank);
                    count += 1.0;
                }
            }
            return mismatch / max(count, 1.0);
        }

        float deghostScoreAt(int frame, ivec2 p, int channel) {
            float sideNorm = frameNorm(frame, p, channel);
            float referenceNorm = frameNorm(${REFERENCE_FRAME_INDEX}, p, channel);
            float sideSignal = sideNorm * uExposureScale[frame];
            float referenceSignal = referenceNorm * uExposureScale[${REFERENCE_FRAME_INDEX}];

            float comparable = captureTrust(frame, p, channel) * captureTrust(${REFERENCE_FRAME_INDEX}, p, channel) * referenceStructure(p, channel);
            if (comparable <= 0.001) {
                return 0.0;
            }

            float sideVariance = noiseVariance(frame, sideNorm, channel);
            float referenceVariance = noiseVariance(${REFERENCE_FRAME_INDEX}, referenceNorm, channel);
            float residualSigma = sqrt(sideVariance + referenceVariance);
            float residual = abs(sideSignal - referenceSignal);
            float residualLow = max(0.004, residualSigma * 3.0);
            float residualHigh = max(0.020, residualSigma * 8.0);
            float linearMotion = smoothstep(residualLow, residualHigh, residual);

            float exposureRatio = clamp(uExposureScale[${REFERENCE_FRAME_INDEX}] / max(uExposureScale[frame], 1e-6), 0.03125, 32.0);
            float exposureGap = abs(log2(exposureRatio));
            float logResidual = abs(log(max(sideSignal, 1e-5)) - log(max(referenceSignal, 1e-5)));
            float logMotion = smoothstep(0.28 + 0.04 * min(exposureGap, 3.0), 0.72, logResidual);
            float rankMotion = smoothstep(0.32, 0.68, censusMismatch(frame, p, channel, referenceSignal, sideSignal));
            float score = max(max(linearMotion, logMotion), rankMotion);
            return comparable * score;
        }

        float computeDeghostAlpha(int frame, ivec2 p, int channel) {
            if (uUseDeghostMask == 0 || frame == ${REFERENCE_FRAME_INDEX}) {
                return 1.0;
            }
            float score = 0.0;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    score = max(score, deghostTileScoreAt(frame, p + ivec2(x * 2, y * 2)));
                }
            }
            float reject = smoothstep(0.30, 0.70, score);
            return mix(1.0, 0.0, reject);
        }

        float frameWeight(int frame, ivec2 p, int channel, out float scaledSignal) {
            float norm = frameNorm(frame, p, channel);
            float scale = uExposureScale[frame];
            scaledSignal = norm * scale;

            float shadowGate = smoothstep(0.002, 0.035, norm);
            float highlightGate = frameHighlightGate(frame, p);
            float clipWeight = max(0.0, shadowGate * highlightGate);

            float centered = exp(-((norm - 0.5) * (norm - 0.5)) / (2.0 * 0.22 * 0.22));
            float contrast = contrastWeight(frame, p, channel);

            float sensorRange = max(1.0, uWhiteLevel - uBlackLevel[channel]);
            float sensorSignal = norm * sensorRange;
            float variance = max(0.0000001, (uNoiseModel.x * sensorSignal + uNoiseModel.y) / (sensorRange * sensorRange));
            float scaledVariance = variance * scale * scale;
            float wiener = (scaledSignal * scaledSignal) / (scaledSignal * scaledSignal + scaledVariance + 0.000001);

            return clipWeight * (0.25 + centered) * (0.2 + contrast) * (0.25 + wiener) *
                computeDeghostAlpha(frame, p, channel);
        }

        void main() {
            ivec2 p = ivec2(gl_FragCoord.xy);
            int channel = channelIndex(p);

            float s0;
            float s1;
            float s2;
            float w0 = frameWeight(0, p, channel, s0);
            float w1 = frameWeight(1, p, channel, s1);
            float w2 = frameWeight(2, p, channel, s2);
            float weightSum = w0 + w1 + w2;

            float fused;
            if (weightSum > 0.000001) {
                fused = (s0 * w0 + s1 * w1 + s2 * w2) / weightSum;
            } else {
                fused = s0;
            }

            fused = clamp(fused, 0.0, 1.0);
            uint raw = uint(floor(fused * 65535.0 + 0.5));
            uint lo = raw & 255u;
            uint hi = (raw >> 8) & 255u;
            fragColor = vec4(float(lo) / 255.0, float(hi) / 255.0, 0.0, 1.0);
        }
    """.trimIndent()
}
