package com.hinnka.mycamera.lut

/**
 * GLSL 着色器源代码
 */
object Shaders {

    /**
     * 顶点着色器
     *
     * 处理顶点位置和纹理坐标变换
     */
    /**
     * 顶点着色器
     *
     * 处理顶点位置和纹理坐标变换
     */
    val VERTEX_SHADER = """
        #version 300 es

        // 顶点属性
        in vec4 aPosition;
        in vec2 aTexCoord;

        // 输出到片元着色器
        out vec2 vTexCoord;
        out vec2 vRawCoord; // 原始坐标用于色散计算

        // MVP 变换矩阵（用于 center crop 缩放）
        uniform mat4 uMVPMatrix;

        // SurfaceTexture 变换矩阵
        uniform mat4 uSTMatrix;
        uniform vec4 uCropRect;

        void main() {
            // 应用 MVP 矩阵进行顶点变换（center crop）
            gl_Position = uMVPMatrix * aPosition;
            vec2 croppedCoord = vec2(
                mix(uCropRect.x, uCropRect.z, aTexCoord.x),
                mix(uCropRect.y, uCropRect.w, aTexCoord.y)
            );
            // 应用 SurfaceTexture 变换矩阵
            vTexCoord = (uSTMatrix * vec4(croppedCoord, 0.0, 1.0)).xy;
            vRawCoord = croppedCoord;
        }
    """.trimIndent()

    /**
     * 简单的直通片元着色器（无 LUT）
     *
     * 用于调试或禁用 LUT 时
     */
    val FRAGMENT_SHADER_PASSTHROUGH = """
        #version 300 es
        #extension GL_OES_EGL_image_external_essl3 : require

        precision mediump float;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform samplerExternalOES uCameraTexture;

        void main() {
            fragColor = texture(uCameraTexture, vTexCoord);
        }
    """.trimIndent()

    /**
     * 片元着色器 - 2D 纹理复制 (支持 sampler2D)
     * 用于从 FBO 纹理复制到屏幕或视频编码器
     */
    val FRAGMENT_SHADER_COPY_2D = """
        #version 300 es
        precision mediump float;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uCameraTexture;

        void main() {
            fragColor = texture(uCameraTexture, vTexCoord);
        }
    """.trimIndent()

    /** 简单顶点着色器（HDF 后处理 Pass 专用，无 MVP/ST 矩阵） */
    val SIMPLE_VERTEX_SHADER = """
        #version 300 es
        in vec4 aPosition;
        in vec2 aTexCoord;
        out vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = aTexCoord;
        }
    """.trimIndent()

    /** HDF Pass 1: 高光提取 + 水平高斯模糊 (实时预览) */
    val HDF_PREVIEW_EXTRACT_BLUR_H = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        uniform float uThreshold;
        uniform float uStrength;
        void main() {
            vec3 color = texture(uInputTexture, vTexCoord).rgb;
            float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
            float extractionVal = mix(luma, max(color.r, max(color.g, color.b)), 0.6);
            float highlightMask = smoothstep(uThreshold - 0.1, uThreshold + 0.25, extractionVal);
            float midMask = smoothstep(uThreshold - 0.5, uThreshold, extractionVal) * 0.4;
            float mask = (highlightMask + midMask * uStrength);
            vec3 sum = color * mask * 0.204164;
            float blurOffsets[4] = float[](1.407333, 3.294215, 5.176470, 7.058823);
            float blurWeights[4] = float[](0.304005, 0.093910, 0.010416, 0.000005);
            for (int i = 0; i < 4; i++) {
                float off = blurOffsets[i] * uTexelSize.x * 2.0;
                sum += texture(uInputTexture, vTexCoord + vec2(off, 0.0)).rgb * blurWeights[i];
                sum += texture(uInputTexture, vTexCoord - vec2(off, 0.0)).rgb * blurWeights[i];
            }
            fragColor = vec4(sum, 1.0);
        }
    """.trimIndent()

    /** HDF Pass 2: 垂直高斯模糊 */
    val HDF_PREVIEW_BLUR_V = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        void main() {
            vec3 sum = texture(uInputTexture, vTexCoord).rgb * 0.204164;
            float blurOffsets[4] = float[](1.407333, 3.294215, 5.176470, 7.058823);
            float blurWeights[4] = float[](0.304005, 0.093910, 0.010416, 0.000005);
            for (int i = 0; i < 4; i++) {
                float off = blurOffsets[i] * uTexelSize.y * 2.0;
                sum += texture(uInputTexture, vTexCoord + vec2(0.0, off)).rgb * blurWeights[i];
                sum += texture(uInputTexture, vTexCoord - vec2(0.0, off)).rgb * blurWeights[i];
            }
            fragColor = vec4(sum, 1.0);
        }
    """.trimIndent()

    /** Soft Light Pass 1: 整图柔焦水平模糊，用于镜头柔光扩散 */
    val SOFT_LIGHT_PREVIEW_BLUR_H = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        void main() {
            vec3 sum = texture(uInputTexture, vTexCoord).rgb * 0.204164;
            float blurOffsets[4] = float[](1.407333, 3.294215, 5.176470, 7.058823);
            float blurWeights[4] = float[](0.304005, 0.093910, 0.010416, 0.000005);
            for (int i = 0; i < 4; i++) {
                float off = blurOffsets[i] * uTexelSize.x * 2.8;
                sum += texture(uInputTexture, vTexCoord + vec2(off, 0.0)).rgb * blurWeights[i];
                sum += texture(uInputTexture, vTexCoord - vec2(off, 0.0)).rgb * blurWeights[i];
            }
            fragColor = vec4(sum, 1.0);
        }
    """.trimIndent()

    /** HDF 合成：原图 + HDF 扩散 + spektrafilm 风格红色 halation */
    val HDF_PREVIEW_COMPOSITE = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uOriginalTexture;
        uniform sampler2D uBloomTexture;
        uniform float uHalation;
        uniform sampler2D uRedHalationTexture;
        uniform float uRedHalation;
        uniform sampler2D uSoftLightTexture;
        uniform float uSoftLight;
        
        void main() {
            vec4 color = texture(uOriginalTexture, vTexCoord);
            
            if (uSoftLight > 0.0) {
                vec3 softBlur = texture(uSoftLightTexture, vTexCoord).rgb;
                vec3 screen = vec3(1.0) - (vec3(1.0) - color.rgb) * (vec3(1.0) - softBlur);
                vec3 softGlow = mix(color.rgb, screen, 0.42);
                color.rgb = mix(color.rgb, softGlow, uSoftLight * 0.75);
                float softLuma = dot(softBlur, vec3(0.2126, 0.7152, 0.0722));
                color.rgb += vec3(softLuma) * (uSoftLight * 0.025);
                color.rgb = (color.rgb - 0.5) * (1.0 - uSoftLight * 0.05) + 0.5;
            }
            
            if (uHalation > 0.0) {
                vec3 bloom = texture(uBloomTexture, vTexCoord).rgb;
                float bLuma = dot(bloom, vec3(0.2126, 0.7152, 0.0722));
                bloom = mix(vec3(bLuma), bloom, 1.6);
                vec3 bloomEffect = bloom * uHalation * 1.4;
                color.rgb = vec3(1.0) - (vec3(1.0) - color.rgb) * (vec3(1.0) - bloomEffect);
                float mist = bLuma * uHalation * 0.15;
                color.rgb += mist;
                color.rgb = (color.rgb - 0.5) * (1.0 - uHalation * 0.08) + 0.5;
            }
            
            if (uRedHalation > 0.0) {
                vec3 halationBlur = texture(uRedHalationTexture, vTexCoord).rgb;
                float halationMask = smoothstep(0.001, 0.06, dot(halationBlur, vec3(0.2126, 0.7152, 0.0722)));
                vec3 halationStrength = vec3(0.42, 0.14, 0.02) * uRedHalation;
                color.rgb += halationBlur * halationStrength * halationMask;
            }
            
            fragColor = clamp(color, 0.0, 1.0);
        }
    """.trimIndent()

    /** Bevy Bloom: first downsample pass with Karis firefly reduction and soft threshold. */
    val BEVY_BLOOM_DOWNSAMPLE_FIRST = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uInputTexelSize;
        uniform vec4 uThreshold;

        float tonemappingLuminance(vec3 v) {
            return dot(v, vec3(0.2126, 0.7152, 0.0722));
        }

        float karisAverage(vec3 color) {
            float luma = tonemappingLuminance(pow(max(color, vec3(0.0)), vec3(1.0 / 2.2))) / 4.0;
            return 1.0 / (1.0 + luma);
        }

        vec3 thresholdHighlight(vec3 color) {
            float luma = tonemappingLuminance(color);
            float mask = 0.0;
            if (uThreshold.z > 0.0) {
                mask = smoothstep(uThreshold.y, uThreshold.y + uThreshold.z, luma);
            } else {
                mask = step(uThreshold.x, luma);
            }
            return color * mask;
        }

        vec3 sampleInput(vec2 uv) {
            vec3 color = texture(uInputTexture, uv).rgb;
            if (uThreshold.x > 0.0 || uThreshold.z > 0.0) {
                color = thresholdHighlight(color);
            }
            return color;
        }

        vec3 sample13Tap(vec2 uv) {
            vec2 ps = uInputTexelSize;
            vec2 pl = 2.0 * ps;
            vec2 ns = -ps;
            vec2 nl = -pl;
            vec3 a = sampleInput(uv + vec2(nl.x, pl.y));
            vec3 b = sampleInput(uv + vec2(0.0, pl.y));
            vec3 c = sampleInput(uv + vec2(pl.x, pl.y));
            vec3 d = sampleInput(uv + vec2(nl.x, 0.0));
            vec3 e = sampleInput(uv);
            vec3 f = sampleInput(uv + vec2(pl.x, 0.0));
            vec3 g = sampleInput(uv + vec2(nl.x, nl.y));
            vec3 h = sampleInput(uv + vec2(0.0, nl.y));
            vec3 i = sampleInput(uv + vec2(pl.x, nl.y));
            vec3 j = sampleInput(uv + vec2(ns.x, ps.y));
            vec3 k = sampleInput(uv + vec2(ps.x, ps.y));
            vec3 l = sampleInput(uv + vec2(ns.x, ns.y));
            vec3 m = sampleInput(uv + vec2(ps.x, ns.y));

            vec3 group0 = (a + b + d + e) * (0.125 / 4.0);
            vec3 group1 = (b + c + e + f) * (0.125 / 4.0);
            vec3 group2 = (d + e + g + h) * (0.125 / 4.0);
            vec3 group3 = (e + f + h + i) * (0.125 / 4.0);
            vec3 group4 = (j + k + l + m) * (0.5 / 4.0);
            group0 *= karisAverage(group0);
            group1 *= karisAverage(group1);
            group2 *= karisAverage(group2);
            group3 *= karisAverage(group3);
            group4 *= karisAverage(group4);
            return group0 + group1 + group2 + group3 + group4;
        }

        void main() {
            vec3 sampleColor = sample13Tap(vTexCoord);
            fragColor = vec4(clamp(sampleColor, vec3(0.0), vec3(1.0)), 1.0);
        }
    """.trimIndent()

    /** Bevy Bloom: subsequent 13-tap downsample passes. */
    val BEVY_BLOOM_DOWNSAMPLE = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uInputTexelSize;

        vec3 sample13Tap(vec2 uv) {
            vec2 ps = uInputTexelSize;
            vec2 pl = 2.0 * ps;
            vec2 ns = -ps;
            vec2 nl = -pl;
            vec3 a = texture(uInputTexture, uv + vec2(nl.x, pl.y)).rgb;
            vec3 b = texture(uInputTexture, uv + vec2(0.0, pl.y)).rgb;
            vec3 c = texture(uInputTexture, uv + vec2(pl.x, pl.y)).rgb;
            vec3 d = texture(uInputTexture, uv + vec2(nl.x, 0.0)).rgb;
            vec3 e = texture(uInputTexture, uv).rgb;
            vec3 f = texture(uInputTexture, uv + vec2(pl.x, 0.0)).rgb;
            vec3 g = texture(uInputTexture, uv + vec2(nl.x, nl.y)).rgb;
            vec3 h = texture(uInputTexture, uv + vec2(0.0, nl.y)).rgb;
            vec3 i = texture(uInputTexture, uv + vec2(pl.x, nl.y)).rgb;
            vec3 j = texture(uInputTexture, uv + vec2(ns.x, ps.y)).rgb;
            vec3 k = texture(uInputTexture, uv + vec2(ps.x, ps.y)).rgb;
            vec3 l = texture(uInputTexture, uv + vec2(ns.x, ns.y)).rgb;
            vec3 m = texture(uInputTexture, uv + vec2(ps.x, ns.y)).rgb;
            vec3 sampleColor = (a + c + g + i) * 0.03125;
            sampleColor += (b + d + f + h) * 0.0625;
            sampleColor += (e + j + k + l + m) * 0.125;
            return sampleColor;
        }

        void main() {
            fragColor = vec4(sample13Tap(vTexCoord), 1.0);
        }
    """.trimIndent()

    /** Bevy Bloom: 3x3 tent upsample pass. */
    val BEVY_BLOOM_UPSAMPLE = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uInputTexelSize;

        void main() {
            float x = uInputTexelSize.x;
            float y = uInputTexelSize.y;
            vec2 uv = vTexCoord;
            vec3 a = texture(uInputTexture, vec2(uv.x - x, uv.y + y)).rgb;
            vec3 b = texture(uInputTexture, vec2(uv.x, uv.y + y)).rgb;
            vec3 c = texture(uInputTexture, vec2(uv.x + x, uv.y + y)).rgb;
            vec3 d = texture(uInputTexture, vec2(uv.x - x, uv.y)).rgb;
            vec3 e = texture(uInputTexture, vec2(uv.x, uv.y)).rgb;
            vec3 f = texture(uInputTexture, vec2(uv.x + x, uv.y)).rgb;
            vec3 g = texture(uInputTexture, vec2(uv.x - x, uv.y - y)).rgb;
            vec3 h = texture(uInputTexture, vec2(uv.x, uv.y - y)).rgb;
            vec3 i = texture(uInputTexture, vec2(uv.x + x, uv.y - y)).rgb;
            vec3 sampleColor = e * 0.25;
            sampleColor += (b + d + f + h) * 0.125;
            sampleColor += (a + c + g + i) * 0.0625;
            fragColor = vec4(sampleColor, 1.0);
        }
    """.trimIndent()

    /** LDR Bloom: final blurred highlight contribution. */
    val BEVY_BLOOM_COMPOSITE = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uBloomTexture;
        uniform sampler2D uBloomTextureNext;
        uniform vec2 uBloomTexelSize;
        uniform vec2 uBloomTexelSizeNext;
        uniform float uBlend;
        uniform float uMipBlend;

        vec3 sampleTentLower(vec2 uv) {
            float x = uBloomTexelSize.x;
            float y = uBloomTexelSize.y;
            vec3 a = texture(uBloomTexture, vec2(uv.x - x, uv.y + y)).rgb;
            vec3 b = texture(uBloomTexture, vec2(uv.x, uv.y + y)).rgb;
            vec3 c = texture(uBloomTexture, vec2(uv.x + x, uv.y + y)).rgb;
            vec3 d = texture(uBloomTexture, vec2(uv.x - x, uv.y)).rgb;
            vec3 e = texture(uBloomTexture, vec2(uv.x, uv.y)).rgb;
            vec3 f = texture(uBloomTexture, vec2(uv.x + x, uv.y)).rgb;
            vec3 g = texture(uBloomTexture, vec2(uv.x - x, uv.y - y)).rgb;
            vec3 h = texture(uBloomTexture, vec2(uv.x, uv.y - y)).rgb;
            vec3 i = texture(uBloomTexture, vec2(uv.x + x, uv.y - y)).rgb;
            vec3 sampleColor = e * 0.25;
            sampleColor += (b + d + f + h) * 0.125;
            sampleColor += (a + c + g + i) * 0.0625;
            return sampleColor;
        }

        vec3 sampleTentUpper(vec2 uv) {
            float x = uBloomTexelSizeNext.x;
            float y = uBloomTexelSizeNext.y;
            vec3 a = texture(uBloomTextureNext, vec2(uv.x - x, uv.y + y)).rgb;
            vec3 b = texture(uBloomTextureNext, vec2(uv.x, uv.y + y)).rgb;
            vec3 c = texture(uBloomTextureNext, vec2(uv.x + x, uv.y + y)).rgb;
            vec3 d = texture(uBloomTextureNext, vec2(uv.x - x, uv.y)).rgb;
            vec3 e = texture(uBloomTextureNext, vec2(uv.x, uv.y)).rgb;
            vec3 f = texture(uBloomTextureNext, vec2(uv.x + x, uv.y)).rgb;
            vec3 g = texture(uBloomTextureNext, vec2(uv.x - x, uv.y - y)).rgb;
            vec3 h = texture(uBloomTextureNext, vec2(uv.x, uv.y - y)).rgb;
            vec3 i = texture(uBloomTextureNext, vec2(uv.x + x, uv.y - y)).rgb;
            vec3 sampleColor = e * 0.25;
            sampleColor += (b + d + f + h) * 0.125;
            sampleColor += (a + c + g + i) * 0.0625;
            return sampleColor;
        }

        void main() {
            vec3 lowerBloom = sampleTentLower(vTexCoord);
            vec3 upperBloom = sampleTentUpper(vTexCoord);
            vec3 bloom = mix(lowerBloom, upperBloom, uMipBlend) * uBlend;
            fragColor = vec4(clamp(bloom, 0.0, 1.0), 1.0);
        }
    """.trimIndent()

    /** Halation Pass 1: 高光重建 + 暖红背反射种子 + 水平高斯模糊 */
    val HALATION_PREVIEW_EXTRACT_BLUR_H = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        uniform float uThreshold;
        uniform float uStrength;
        void main() {
            vec3 tint = vec3(1.0, 0.28, 0.04);
            
            #define EXTRACT(sampleColor) \
                (max(sampleColor - vec3(uThreshold), vec3(0.0)) * tint * (1.5 + uStrength * 3.0) * smoothstep(uThreshold - 0.24, uThreshold + 0.36, max(sampleColor.r, max(sampleColor.g, sampleColor.b))))

            vec3 color = texture(uInputTexture, vTexCoord).rgb;
            vec3 sum = EXTRACT(color) * 0.204164;
            
            float blurOffsets[4] = float[](1.407333, 3.294215, 5.176470, 7.058823);
            float blurWeights[4] = float[](0.304005, 0.093910, 0.010416, 0.000005);
            for (int i = 0; i < 4; i++) {
                float off = blurOffsets[i] * uTexelSize.x * 2.0;
                sum += EXTRACT(texture(uInputTexture, vTexCoord + vec2(off, 0.0)).rgb) * blurWeights[i];
                sum += EXTRACT(texture(uInputTexture, vTexCoord - vec2(off, 0.0)).rgb) * blurWeights[i];
            }
            fragColor = vec4(sum, 1.0);
        }
    """.trimIndent()

    /** Halation Pass 2: 垂直高斯模糊 */
    val HALATION_PREVIEW_BLUR_V = HDF_PREVIEW_BLUR_V

    /**
     * Focus Peaking Shader
     */
    val FRAGMENT_SHADER_FOCUS_PEAKING = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        uniform float uThreshold;
        uniform vec3 uPeakColor;

        void main() {
            vec4 color = texture(uInputTexture, vTexCoord);

            // Sobel edge detection
            float l00 = dot(texture(uInputTexture, vTexCoord + vec2(-uTexelSize.x, -uTexelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
            float l10 = dot(texture(uInputTexture, vTexCoord + vec2(0.0, -uTexelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
            float l20 = dot(texture(uInputTexture, vTexCoord + vec2(uTexelSize.x, -uTexelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
            float l01 = dot(texture(uInputTexture, vTexCoord + vec2(-uTexelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
            float l21 = dot(texture(uInputTexture, vTexCoord + vec2(uTexelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
            float l02 = dot(texture(uInputTexture, vTexCoord + vec2(-uTexelSize.x, uTexelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
            float l12 = dot(texture(uInputTexture, vTexCoord + vec2(0.0, uTexelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
            float l22 = dot(texture(uInputTexture, vTexCoord + vec2(uTexelSize.x, uTexelSize.y)).rgb, vec3(0.299, 0.587, 0.114));

            float gx = l00 + 2.0 * l01 + l02 - l20 - 2.0 * l21 - l22;
            float gy = l00 + 2.0 * l10 + l20 - l02 - 2.0 * l12 - l22;
            float edge = sqrt(gx * gx + gy * gy);
            float peakFactor = smoothstep(uThreshold, uThreshold * 1.5, edge);
            fragColor = vec4(mix(color.rgb, uPeakColor, peakFactor * 0.9), color.a);
        }
    """.trimIndent()

    /**
     * 片元着色器 - 带色彩配方和 3D LUT 支持
     *
     * 处理流程：相机采样 → 色彩配方调整 → LUT处理（可选） → 输出
     *
     * 性能优化：
     * - 使用 highp float 提高色彩精度
     * - Early exit 优化（无调整时直接返回）
     * - 避免 HSL 转换，使用基于 Luma 的快速饱和度算法
     */
    val FRAGMENT_SHADER_COLOR_RECIPE = """
    #version 300 es
    #extension GL_OES_EGL_image_external_essl3 : require

    precision highp float;

    // 从顶点着色器接收的纹理坐标
    in vec2 vTexCoord;
    in vec2 vRawCoord; // 原始坐标（用于色散等空间相关效果）

    // 输出颜色
    out vec4 fragColor;

    // 相机 OES 纹理
    uniform samplerExternalOES uCameraTexture;

    // 3D LUT 纹理
    uniform mediump sampler3D uLutTexture;

    // LUT 控制
    uniform float uLutSize;
    uniform float uLutIntensity;
    uniform bool uLutEnabled;
    uniform int uLutMaskType;
    uniform int uLutCurve; // 0=sRGB, 1=Linear, 2=V-Log, 3=S-Log3, 4=F-Log2, 5=LogC4, 6=AppleLog, 7=HLG, 8=ACEScct, 9=Log3G10
    uniform int uLutColorSpace; // 0=sRGB, 1=DCI-P3, 2=BT2020, 3=ARRI4, 4=AppleLog2, 5=S-Gamut3.Cine, 6=ACES_AP1, 7=V-Gamut, 9=REDWideGamutRGB
    uniform bool uVideoLogEnabled;
    uniform int uVideoLogCurve;
    uniform int uVideoColorSpace;
    uniform bool uIsHlgInput;  // true：相机以 HLG10 采集，LUT 查找前用 HLG EOTF 解码至线性

    // 色彩配方控制
    uniform bool uColorRecipeEnabled;
    uniform mat4 uSTMatrix; // 传递矩阵到片元用于 CA 偏移采样

    // 色彩配方参数（阶段1：核心3参数）
    uniform float uExposure;      // -2.0 ~ +2.0 (EV)
    uniform float uContrast;      // 0.5 ~ 1.5
    uniform float uSaturation;    // 0.0 ~ 2.0

    // 色彩配方参数（阶段1：额外3参数）
    uniform float uTemperature;   // -1.0 ~ +1.0 (暖/冷色调)
    uniform float uTint;          // -1.0 ~ +1.0 (绿/品红偏移)
    uniform float uFade;          // 0.0 ~ 1.0 (褪色效果)
    uniform float uVibrance;      // 0.0 ~ 2.0 (蓝色增强 - vibrance)

    // 色彩配方参数（阶段2：高级参数）
    uniform float uHighlights;    // -1.0 ~ +1.0 (高光调整)
    uniform float uShadows;       // -1.0 ~ +1.0 (阴影调整)
    uniform float uToneToe;       // -1.0 ~ +1.0 (暗部曲线塑形)
    uniform float uToneShoulder;  // -1.0 ~ +1.0 (亮部曲线塑形)
    uniform float uTonePivot;     // -1.0 ~ +1.0 (曲线中点偏移)

    // 色彩配方参数（阶段3：质感效果）
    uniform float uFilmGrain;     // 0.0 ~ 1.0 (颗粒强度)
    uniform float uVignette;      // -1.0 ~ +1.0 (晕影，负值暗角，正值亮角)
    uniform float uBleachBypass;  // 0.0 ~ 1.0 (留银冲洗强度)
    uniform float uChromaticAberration; // 0.0 ~ 1.0 (色散强度)
    uniform float uNoise;         // 0.0 ~ 1.0 (噪点强度，包含亮度和彩色噪点)
    uniform float uNoiseSeed;     // 用于每帧刷新噪点的随机种子
    uniform float uLowRes;        // 0.0 ~ 1.0 (低分辨率强度，0为无效果)
    uniform float uAspectRatio;   // 图像长宽比 (Width/Height)，用于确保低分像素块是正方形
    uniform float uLchHueAdjustments[9];
    uniform float uLchChromaAdjustments[9];
    uniform float uLchLightnessAdjustments[9];
    uniform mat3 uPrimaryCalibrationMatrix;
    uniform float uAperture;      // 计算光圈 (1.4 ~ 16.0)
    uniform vec2 uFocusPoint;     // 对焦点 (0.0 ~ 1.0)
    uniform vec2 uTexelSize;      // 当前渲染目标像素尺寸，用于局部影调 base 采样

    // 曲线调整纹理 (256×1 RGBA8)
    // R = master_curve(red_curve(x)), G = master_curve(green_curve(x)), B = master_curve(blue_curve(x))
    uniform sampler2D uCurveTexture;
    uniform bool uCurveEnabled;

    const vec3 W = vec3(0.2126, 0.7152, 0.0722);
    const float PI = 3.14159265359;

    float log10(float x) { return log(x) * 0.4342944819; }
    vec3 log10(vec3 x) { return log(x) * 0.4342944819; }

    // Preserve the sign so wide-gamut highlight channels below 0 do not produce NaN in pow().
    vec3 srgbToLinear(vec3 c) {
        vec3 absC = abs(c);
        vec3 result = mix(absC / 12.92, pow((absC + 0.055) / 1.055, vec3(2.4)), step(0.04045, absC));
        return sign(c) * result;
    }

    vec3 linearToSrgb(vec3 l) {
        vec3 absL = abs(l);
        vec3 result = mix(absL * 12.92, 1.055 * pow(absL, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, absL));
        return sign(l) * result;
    }

    // BT.2100 HLG EOTF：将 HLG 编码信号解码为场景线性光（可能 > 1.0，保留 HDR 高光）
    vec3 hlgToLinear(vec3 e) {
        float ha = 0.17883277;
        float hb = 1.0 - 4.0 * ha;
        float hc = 0.5 - ha * log(4.0 * ha);
        vec3 low = e * e / 3.0;
        vec3 high = (exp((e - hc) / ha) + hb) / 12.0;
        return mix(low, high, step(vec3(0.5), e));
    }

    // BT.2020 原色的线性光转换到线性 sRGB，供现有 SDR/LUT 管线继续处理。
    vec3 bt2020ToLinearSrgb(vec3 rgb) {
        return mat3(
            1.660491, -0.124550, -0.018151,
            -0.587641, 1.132900, -0.100579,
            -0.072850, -0.008350, 1.118730
        ) * rgb;
    }

    vec3 applyExposureInLinearSpace(vec3 srgbColor, float exposureEv) {
        vec3 linearColor = srgbToLinear(max(srgbColor, vec3(0.0)));
        linearColor *= exp2(exposureEv);
        return linearToSrgb(linearColor);
    }

    // NaN 保护
    float sanitizeFloat(float value) {
        if (value != value) return 0.0;
//        if (value > 1.0) return 1.0;
//        if (value < 0.0) return 0.0;
        return value;
    }

    vec3 sanitizeColor(vec3 color) {
        return vec3(
            sanitizeFloat(color.r),
            sanitizeFloat(color.g),
            sanitizeFloat(color.b)
        );
    }

    float getLuma(vec3 color) {
        return dot(color, W);
    }

    vec3 prepareToneSample(vec3 sampleColor) {
        vec3 prepared = sampleColor;
        if (uIsHlgInput) {
            prepared = hlgToLinear(prepared);
            prepared = linearToSrgb(prepared);
        }
        if (abs(uExposure) > 0.001) {
            prepared = applyExposureInLinearSpace(prepared, uExposure);
        }
        return sanitizeColor(prepared);
    }

    vec3 sampleToneSource(vec2 uv) {
        return prepareToneSample(texture(uCameraTexture, clamp(uv, vec2(0.0), vec2(1.0))).rgb);
    }

    vec3 shRgbToXyz(vec3 rgb) {
        vec3 linearRgb = srgbToLinear(rgb);
        return mat3(
            0.4360747, 0.2225045, 0.0139322,
            0.3850649, 0.7168786, 0.0971045,
            0.1430804, 0.0606169, 0.7141733
        ) * linearRgb;
    }

    vec3 shXyzToRgb(vec3 xyz) {
        vec3 linearRgb = mat3(
             3.1338561, -0.9787684,  0.0719453,
            -1.6168667,  1.9161415, -0.2289914,
            -0.4906146,  0.0334540,  1.4052427
        ) * xyz;
        return linearToSrgb(linearRgb);
    }

    ${ShadowsHighlightsShader.GLSL}

    float hash12(vec2 p) {
        vec3 p3 = fract(vec3(p.xyx) * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    }

    float gaussianNoise(vec2 p) {
        return (hash12(p) + hash12(p + 17.17) + hash12(p + 43.31) + hash12(p + 91.73) - 2.0) * 0.5;
    }

    vec3 applyDensityParticleGrain(vec3 srgbColor, vec2 uv, float amount) {
        float grainAmount = pow(clamp(amount, 0.0, 1.0), 0.58);
        vec3 linearColor = max(srgbToLinear(max(srgbColor, vec3(0.0))), vec3(1e-4));
        vec3 density = -log10(linearColor);
        vec3 densityMin = vec3(0.03);
        vec3 densityMax = vec3(2.2) + densityMin;
        vec3 development = clamp((density + densityMin) / densityMax, vec3(0.02), vec3(0.98));
        float effectivePixelSizeUm = mix(5.2, 1.8, grainAmount);
        float agxParticleAreaUm2 = 0.2;
        vec3 particleScale = vec3(1.6, 1.6, 3.2);
        vec3 particles = (effectivePixelSizeUm * effectivePixelSizeUm) / (agxParticleAreaUm2 * particleScale);
        vec3 uniformity = vec3(0.97, 0.99, 0.97);
        vec2 grainCoord = uv * mix(820.0, 1380.0, grainAmount);
        float lumaGrain = gaussianNoise(grainCoord + vec2(11.0, 7.0));
        vec2 dyeCoord = uv * mix(260.0, 420.0, grainAmount);
        vec3 dyeCloud = vec3(
            gaussianNoise(dyeCoord + vec2(31.0, 53.0)),
            gaussianNoise(dyeCoord + vec2(71.0, 23.0)),
            gaussianNoise(dyeCoord + vec2(19.0, 97.0))
        );
        float clump = gaussianNoise(floor(uv * 180.0) + vec2(5.0, 19.0));
        vec3 saturation = 1.0 - development * uniformity;
        vec3 densityStd = densityMax * sqrt(max(development * (1.0 - development) * saturation, vec3(0.001)) / max(particles, vec3(1.0)));
        float lumaDensityStd = dot(densityStd, vec3(0.333333));
        float positiveLuma = dot(linearColor, vec3(0.2126, 0.7152, 0.0722));
        float positiveHighlightMask = smoothstep(0.55, 0.92, positiveLuma);
        float highlightGrainVisibility = mix(1.0, 0.32, positiveHighlightMask);
        vec3 densityNoise = vec3(lumaGrain * lumaDensityStd * 2.7);
        densityNoise += dyeCloud * densityStd * 0.28;
        densityNoise += clump * grainAmount * vec3(0.013, 0.012, 0.016);
        densityNoise *= highlightGrainVisibility;
        density = max(density + densityNoise * grainAmount * 1.8, vec3(0.0));
        return linearToSrgb(pow(vec3(10.0), -density));
    }

    float applyToneCurveToLuma(float luma, float toe, float shoulder, float pivot) {
        float safeLuma = clamp(luma, 0.0, 1.0);
        float pivotPoint = clamp(0.5 + pivot * 0.12, 0.2, 0.8);
        float toeAmount = clamp(abs(toe), 0.0, 1.0);
        float shoulderAmount = clamp(abs(shoulder), 0.0, 1.0);
        float toeGamma = (toe >= 0.0)
            ? mix(1.0, 0.68, toeAmount)
            : mix(1.0, 1.85, toeAmount);
        float shoulderGamma = (shoulder >= 0.0)
            ? mix(1.0, 0.72, shoulderAmount)
            : mix(1.0, 1.85, shoulderAmount);

        if (safeLuma <= pivotPoint) {
            float segment = clamp(safeLuma / max(pivotPoint, 0.0001), 0.0, 1.0);
            return clamp(pow(segment, toeGamma) * pivotPoint, 0.0, 1.0);
        }

        float segment = clamp((safeLuma - pivotPoint) / max(1.0 - pivotPoint, 0.0001), 0.0, 1.0);
        float result = 1.0 - pow(max(0.0, 1.0 - segment), shoulderGamma) * (1.0 - pivotPoint);
        return clamp(result, 0.0, 1.0);
    }

    vec3 applyToneCurve(vec3 color, float toe, float shoulder, float pivot) {
        if (abs(toe) < 0.001 && abs(shoulder) < 0.001 && abs(pivot) < 0.001) {
            return color;
        }
        vec3 safeColor = clamp(color, 0.0, 1.0); // 曲线数学需要 [0,1] 输入
        float luma = dot(safeColor, W);
        float curvedLuma = applyToneCurveToLuma(luma, toe, shoulder, pivot);
        if (luma < 0.0001) {
            return safeColor;
        }
        vec3 scaled = safeColor * (curvedLuma / luma);
        return mix(vec3(curvedLuma), scaled, 0.92);
    }

    vec3 linearRgbToOklab(vec3 c) {
        vec3 lms = mat3(
            0.4122214708, 0.2119034982, 0.0883024619,
            0.5363325363, 0.6806995451, 0.2817188376,
            0.0514459929, 0.1073969566, 0.6299787005
        ) * c;
        vec3 lmsCbrt = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
        return mat3(
            0.2104542553, 1.9779984951, 0.0259040371,
            0.7936177850, -2.4285922050, 0.7827717662,
            -0.0040720468, 0.4505937099, -0.8086757660
        ) * lmsCbrt;
    }

    vec3 oklabToLinearRgb(vec3 lab) {
        vec3 lms = mat3(
            1.0, 1.0, 1.0,
            0.3963377774, -0.1055613458, -0.0894841775,
            0.2158037573, -0.0638541728, -1.2914855480
        ) * lab;
        vec3 lms3 = lms * lms * lms;
        return mat3(
            4.0767416621, -1.2684380046, -0.0041960863,
            -3.3077115913, 2.6097574011, -0.7034186147,
            0.2309699292, -0.3413193965, 1.7076147010
        ) * lms3;
    }

    vec3 linearRgbToCieLab(vec3 linearRgb) {
        vec3 xyz = mat3(
            0.4124564, 0.2126729, 0.0193339,
            0.3575761, 0.7151522, 0.1191920,
            0.1804375, 0.0721750, 0.9503041
        ) * clamp(linearRgb, 0.0, 1.0);
        xyz /= vec3(0.95047, 1.0, 1.08883);
        vec3 f = mix(7.787037 * xyz + vec3(16.0 / 116.0), pow(max(xyz, vec3(0.0)), vec3(1.0 / 3.0)), step(vec3(0.008856), xyz));
        return vec3(116.0 * f.y - 16.0, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z));
    }

    float wrapAngle(float angle) {
        return mod(angle + PI, 2.0 * PI) - PI;
    }

    float colorBandWeight(float hue, float center, float chroma) {
        float dist = abs(wrapAngle(hue - center));
        float hueWeight = 1.0 - smoothstep(radians(18.0), radians(42.0), dist);
        float chromaWeight = smoothstep(0.02, 0.08, chroma);
        return hueWeight * chromaWeight;
    }

    float fullCoverageBandWeight(float hue, float center, float chroma) {
        float dist = abs(wrapAngle(hue - center));
        // Use smoothstep for a more natural bell-curve falloff. 
        // 85 degrees ensures coverage for wider gaps in Oklab hue space (e.g. Blue-Cyan).
        float hueWeight = smoothstep(radians(85.0), 0.0, dist);
        float chromaWeight = smoothstep(0.005, 0.03, chroma);
        return hueWeight * chromaWeight;
    }

    float rtRange(float value, float minValue, float maxValue) {
        return step(minValue, value) * (1.0 - step(maxValue, value));
    }

    float rtSkinCase(float l, float h, float c, float lMin, float lMax, float hMin, float hMax, float cMin, float cMax, float weight) {
        return rtRange(l, lMin, lMax) * rtRange(h, hMin, hMax) * rtRange(c, cMin, cMax) * weight;
    }

    float skinBandWeight(vec3 linearColor) {
        // Direct port of RawTherapee Color::SkinSat ranges in CIELAB LCh.
        // Weights only rank RT's real / extended / transition categories.
        vec3 lab = linearRgbToCieLab(linearColor);
        float l = lab.x;
        float h = atan(lab.z, lab.y);
        float c = length(lab.yz);
        float core = 1.0;
        float extended = 0.67;
        float transition = 0.33;
        float w = 0.0;

        w = max(w, rtSkinCase(l, h, c, 85.0, 100.0, 0.73, 1.23, 8.0, 22.0, core));
        w = max(w, rtSkinCase(l, h, c, 92.0, 100.0, 0.80, 1.65, 7.0, 15.0, extended));
        w = max(w, rtSkinCase(l, h, c, 92.0, 100.0, -0.10, 1.65, 7.0, 18.0, transition));
        w = max(w, rtSkinCase(l, h, c, 85.0, 92.0, 0.70, 1.40, 7.0, 34.0, extended));
        w = max(w, rtSkinCase(l, h, c, 85.0, 92.0, 0.00, 1.65, 7.0, 43.0, transition));

        w = max(w, rtSkinCase(l, h, c, 70.0, 85.0, 0.40, 1.29, 8.0, 50.0, core));
        w = max(w, rtSkinCase(l, h, c, 70.0, 85.0, -0.18, 1.50, 7.0, 56.0, extended));
        w = max(w, rtSkinCase(l, h, c, 70.0, 85.0, -0.18, 1.65, 7.0, 63.0, transition));

        w = max(w, rtSkinCase(l, h, c, 52.0, 70.0, 0.30, 1.37, 11.0, 47.0, core));
        w = max(w, rtSkinCase(l, h, c, 52.0, 70.0, -0.18, 1.50, 7.0, 56.0, extended));
        w = max(w, rtSkinCase(l, h, c, 52.0, 70.0, -0.18, 1.65, 7.0, 63.0, transition));

        w = max(w, rtSkinCase(l, h, c, 35.0, 52.0, 0.30, 1.27, 13.0, 44.0, core));
        w = max(w, rtSkinCase(l, h, c, 35.0, 52.0, -0.18, 1.50, 7.0, 56.0, extended));
        w = max(w, rtSkinCase(l, h, c, 35.0, 52.0, -0.18, 1.65, 7.0, 63.0, transition));

        w = max(w, rtSkinCase(l, h, c, 20.0, 35.0, 0.30, 1.22, 7.0, 40.0, core));
        w = max(w, rtSkinCase(l, h, c, 20.0, 35.0, -0.18, 1.50, 7.0, 56.0, extended));
        w = max(w, rtSkinCase(l, h, c, 20.0, 35.0, -0.18, 1.65, 7.0, 63.0, transition));

        w = max(w, rtSkinCase(l, h, c, 10.0, 20.0, -0.20, 1.05, 8.0, 28.0, core));
        w = max(w, rtSkinCase(l, h, c, 10.0, 20.0, -0.18, 1.00, 7.0, 40.0, extended));
        w = max(w, rtSkinCase(l, h, c, 10.0, 20.0, -0.18, 1.60, 7.0, 50.0, transition));

        w = max(w, rtSkinCase(l, h, c, 0.0, 10.0, -0.18, 1.00, 8.0, 28.0, core));
        w = max(w, rtSkinCase(l, h, c, 0.0, 10.0, -0.18, 1.00, 7.0, 40.0, extended));
        w = max(w, rtSkinCase(l, h, c, 0.0, 10.0, -0.18, 1.60, 7.0, 50.0, transition));
        return w;
    }

    float skyBandWeight(vec3 linearColor) {
        vec3 lab = linearRgbToCieLab(linearColor);
        float l = lab.x;
        float h = atan(lab.z, lab.y);
        float c = length(lab.yz);
        float rtWaveletSkyHue = rtRange(h, -2.60, -1.30);
        float chromaGate = smoothstep(7.0, 18.0, c);
        float lightnessGate = smoothstep(18.0, 45.0, l);
        return rtWaveletSkyHue * chromaGate * lightnessGate;
    }

    float lutMaskWeight(int maskType, vec3 linearColor) {
        if (maskType == 1) {
            return skinBandWeight(linearColor);
        }
        if (maskType == 2) {
            return skyBandWeight(linearColor);
        }
        return 1.0;
    }

    vec3 applyOklchDensity(vec3 srgbColor, float density) {
        if (abs(density) < 0.0001) {
            return srgbColor;
        }

        vec3 linearColor = srgbToLinear(max(srgbColor, vec3(0.0)));
        vec3 lab = linearRgbToOklab(linearColor);
        float chroma = length(lab.yz);
        float hue = atan(lab.z, lab.y);
        const float CHROMA_BIAS = 0.35;
        float densityScale = max(0.0, 1.0 + density * CHROMA_BIAS);
        float newChroma = chroma * densityScale;
        const float DENSITY_K = 1.85;
        float newLightness = clamp(lab.x * exp(-DENSITY_K * density * chroma), 0.0, 1.0); // OkLab L ∈ [0,1]
        vec3 denseLab = vec3(newLightness, cos(hue) * newChroma, sin(hue) * newChroma);
        vec3 denseLinear = max(oklabToLinearRgb(denseLab), vec3(0.0));
        return linearToSrgb(denseLinear);
    }

    vec3 applyLchColorMixer(vec3 srgbColor) {
        vec3 linearColor = srgbToLinear(max(srgbColor, vec3(0.0)));
        vec3 lab = linearRgbToOklab(linearColor);
        float chroma = length(lab.yz);
        float hue = atan(lab.z, lab.y);
        if (hue < 0.0) hue += 2.0 * PI;

        // Optimized centers mapping LR colors to Oklab hue angles
        float centers[8] = float[](
            radians(29.0),  // Red
            radians(52.0),  // Orange
            radians(86.0),  // Yellow
            radians(144.0), // Green
            radians(196.0), // Aqua/Cyan
            radians(263.0), // Blue
            radians(304.0), // Purple
            radians(341.0)  // Magenta
        );

        float hueShift = 0.0;
        float chromaScale = 1.0;
        float lightnessShift = 0.0;
        float bandWeights[8];
        float totalBandWeight = 0.0;

        for (int i = 0; i < 8; i++) {
            float weight = fullCoverageBandWeight(hue, centers[i], chroma);
            bandWeights[i] = weight;
            totalBandWeight += weight;
        }

        if (totalBandWeight > 0.0001) {
            for (int i = 0; i < 8; i++) {
                float weight = bandWeights[i] / totalBandWeight;
                // Reduce hue range to +/- 20 degrees for professional results
                hueShift += uLchHueAdjustments[i + 1] * weight * radians(20.0);
                chromaScale += uLchChromaAdjustments[i + 1] * weight;
                lightnessShift += uLchLightnessAdjustments[i + 1] * weight * 0.15;
            }

            // Restore the chroma fade-out that was lost during normalization
            // This prevents harsh banding (断层) when colors transition to neutral
            float commonChromaWeight = smoothstep(0.005, 0.03, chroma);
            hueShift *= commonChromaWeight;
            chromaScale = mix(1.0, chromaScale, commonChromaWeight);
            lightnessShift *= commonChromaWeight;
        }

        float skinWeight = skinBandWeight(linearColor);
        if (skinWeight > 0.0001) {
            hueShift += uLchHueAdjustments[0] * skinWeight * radians(10.0);
            chromaScale += uLchChromaAdjustments[0] * skinWeight;
            lightnessShift += uLchLightnessAdjustments[0] * skinWeight * 0.08;
        }

        if (abs(hueShift) < 0.0001 && abs(chromaScale - 1.0) < 0.0001 && abs(lightnessShift) < 0.0001) {
            return srgbColor;
        }

        float newHue = hue + hueShift;
        float newChroma = max(0.0, chroma * max(0.0, chromaScale));
        float newLightness = clamp(lab.x + lightnessShift, 0.0, 1.0); // OkLab L ∈ [0,1]
        vec3 mixedLab = vec3(newLightness, cos(newHue) * newChroma, sin(newHue) * newChroma);
        vec3 mixedLinear = max(oklabToLinearRgb(mixedLab), vec3(0.0));
        return linearToSrgb(mixedLinear);
    }

    vec3 applyPrimaryCalibration(vec3 color) {
        vec3 linearColor = srgbToLinear(max(color, vec3(0.0)));
        vec3 calibratedLinear = max(uPrimaryCalibrationMatrix * linearColor, vec3(0.0));
        return linearToSrgb(calibratedLinear);
    }

    vec3 applyLutCurve(vec3 l, int curveType) {
        if (curveType == 0) { // sRGB
            return linearToSrgb(l);
        }
        if (curveType == 1) return l; // LINEAR
        if (curveType == 2) { // V-Log
            return mix(5.6 * l + 0.125, 0.241514 * log10(l + 0.00873) + 0.598206, step(0.01, l));
        }
        if (curveType == 3) { // S-Log3
            return mix((l * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0, (420.0 + log10((l + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0, step(0.01125, l));
        }
        if (curveType == 4) { // F-Log2
            return mix(8.799461 * l + 0.092864, 0.245281 * log10(5.555556 * l + 0.064829) + 0.384316, step(0.00089, l));
        }
        if (curveType == 5) { // LogC4
            return mix(8.80302 * l + 0.158957, 0.21524584 * log10(2231.8263 * l + 64.0) - 0.29590839, step(-0.018057, l));
        }
        if (curveType == 6) { // AppleLog
            return mix(mix(vec3(0.0), 47.28711236 * pow(l + 0.05641088, vec3(2.0)), step(-0.05641088, l)), 0.08550479 * (log(l + 0.00964052) / log(2.0)) + 0.69336945, step(0.01, l));
        }
        if (curveType == 7) { // HLG
            float ha = 0.17883277;
            float hb = 1.0 - 4.0 * ha;
            float hc = 0.5 - ha * log(4.0 * ha);
            return mix(sqrt(3.0 * l), ha * log(12.0 * l - hb) + hc, step(1.0 / 12.0, l));
        }
        if (curveType == 8) { // ACEScct
            return mix(10.540237 * l + 0.072905536, 0.18955931 * log10(max(l, vec3(1e-6))) + 0.5547945, step(0.0078125, l));
        }
        if (curveType == 9) { // Log3G10
            return mix(15.1927 * l + 0.151927, 0.224282 * log10(155.975327 * l + 2.55975327), step(-0.01, l));
        }
        return l;
    }

    vec3 applyLutColorSpace(vec3 rgb, int colorSpace) {
        if (colorSpace == 0) return rgb; // sRGB

        // Matrices for Linear sRGB to target color space (aligned with ColorSpace.kt)
        if (colorSpace == 1) { // DCI-P3 (Bradford adapted)
            return mat3(0.875905, 0.035332, 0.016382, 0.122070, 0.964542, 0.063767, 0.002025, 0.000126, 0.919851) * rgb;
        }
        if (colorSpace == 2) { // BT2020
            return mat3(0.627404, 0.069097, 0.016391, 0.329283, 0.919540, 0.088013, 0.043313, 0.011362, 0.895595) * rgb;
        }
        if (colorSpace == 3) { // ARRI4
            return mat3(0.565837, 0.088626, 0.017750, 0.340331, 0.809347, 0.109448, 0.093832, 0.102028, 0.872802) * rgb;
        }
        if (colorSpace == 4) { // AppleLog2
            return mat3(0.608104, 0.062316, 0.031133, 0.259353, 0.804609, 0.133756, 0.132543, 0.133076, 0.835112) * rgb;
        }
        if (colorSpace == 5) { // S-Gamut3.Cine
            return mat3(0.645679, 0.087530, 0.036957, 0.259115, 0.759700, 0.129281, 0.095206, 0.152770, 0.833762) * rgb;
        }
        if (colorSpace == 6) { // ACES_AP1
            return mat3(0.613083, 0.070004, 0.020491, 0.341167, 0.918063, 0.106764, 0.045750, 0.011934, 0.872745) * rgb;
        }
        if (colorSpace == 7) { // V-Gamut
            return mat3(0.585196, 0.078589, 0.022794, 0.322642, 0.819627, 0.114217, 0.092162, 0.101784, 0.862989) * rgb;
        }
        if (colorSpace == 9) { // REDWideGamutRGB
            return mat3(0.541973, 0.076993, 0.058875, 0.360148, 0.767969, 0.273495, 0.097891, 0.155019, 0.667533) * rgb;
        }
        return rgb;
    }

    void main()
    {
        // === 预处理：模拟真实低分辨率效果（模糊+锯齿，而非纯碎的马赛克） ===
        vec2 uvCoord = vTexCoord;
        vec2 rcCoord = vRawCoord;
        if (uLowRes > 0.005) {
            // 算法改良：不再使用 floor() 产生的硬边缘，而是通过降采样+双线性插值模拟“糊”感
            // 同时保留边缘的低分锯齿感
            float blocksX = mix(512.0, 32.0, uLowRes);
            vec2 gridSize = vec2(1.0 / blocksX, 1.0 / (blocksX / uAspectRatio));

            // 1. 计算当前像素所在的网格坐标
            vec2 gridUV = floor(vTexCoord / gridSize) * gridSize + gridSize * 0.5;
            vec2 gridRC = floor(vRawCoord / gridSize) * gridSize + gridSize * 0.5;

            // 2. 混合原始坐标和网格坐标，实现“糊”且有锯齿的效果
            // 边缘会有锯齿，但平整区域通过插值变得平滑，避免彩色碎块
            uvCoord = mix(vTexCoord, gridUV, 0.95);
            rcCoord = mix(vRawCoord, gridRC, 0.95);
        }

        // 从相机纹理采样原始颜色 (应用色散效果)
        vec4 color;
        if (uChromaticAberration > 0.001) {
            // 色散：基于 vRawCoord（0.0~1.0 原始平面）计算偏移
            vec2 center = vec2(0.5);
            vec2 dir = rcCoord - center;
            float dist = length(dir);
            float offset = pow(dist, 1.5) * uChromaticAberration * 0.08;

            // 关键：基于原始坐标偏移后，必须通过 uSTMatrix 转换回实际采样坐标
            vec2 rCoord = (uSTMatrix * vec4(rcCoord + dir * offset, 0.0, 1.0)).xy;
            vec2 bCoord = (uSTMatrix * vec4(rcCoord - dir * offset, 0.0, 1.0)).xy;

            float r = texture(uCameraTexture, rCoord).r;
            float g = texture(uCameraTexture, uvCoord).g;
            float b = texture(uCameraTexture, bCoord).b;
            float a = texture(uCameraTexture, uvCoord).a;
            color = vec4(r, g, b, a);
        } else {
            color = texture(uCameraTexture, uvCoord);
        }
        
        if (uIsHlgInput) {
            color.rgb = hlgToLinear(color.rgb);
            color.rgb = linearToSrgb(color.rgb);
        }

        // === 色彩配方处理（按专业后期流程顺序） ===
        if (uColorRecipeEnabled) {
            // 1. 曝光调整（在线性空间执行 EV 增益，再回到显示空间）
            if (abs(uExposure) > 0.001) {
                color.rgb = applyExposureInLinearSpace(color.rgb, uExposure);
                color.rgb = sanitizeColor(color.rgb);
            }

            // 2. 高光/阴影调整：RT 风格 tonal width + 局部 base mask，避免线性乘法洗白全图
            color.rgb = applyShadowsHighlights(color.rgb, uvCoord);
            color.rgb = sanitizeColor(color.rgb);

            // 计算基础亮度，后续复用
            float luma = getLuma(color.rgb);

            // 3. 对比度（围绕中灰点调整）
            if (abs(uContrast - 1.0) > 0.001) {
                color.rgb = (color.rgb - 0.5) * uContrast + 0.5;
                color.rgb = sanitizeColor(color.rgb);
            }

            // 3.5. 影调曲线（独立于简单对比度，塑造高调/低调 profile）
            color.rgb = applyToneCurve(color.rgb, uToneToe, uToneShoulder, uTonePivot);
            color.rgb = sanitizeColor(color.rgb);

            // 4. 白平衡调整（色温 + 色调）
            color.r += uTemperature * 0.1;
            color.b -= uTemperature * 0.1;
            color.g += uTint * 0.05;
            color.rgb = sanitizeColor(color.rgb);

            // 5. 饱和度
            if (abs(uSaturation - 1.0) > 0.001) {
                luma = dot(color.rgb, W);
                color.rgb = mix(vec3(luma), color.rgb, uSaturation);
                color.rgb = sanitizeColor(color.rgb);
            }

            // 6. 色彩密度（OkLCh density）
            if (abs(uVibrance) > 0.001) {
                color.rgb = applyOklchDensity(color.rgb, uVibrance);
                color.rgb = sanitizeColor(color.rgb);
            }

            // 6.5. 颜色校准 (Camera Calibration)
            color.rgb = applyPrimaryCalibration(color.rgb);
            color.rgb = sanitizeColor(color.rgb);

            color.rgb = applyLchColorMixer(color.rgb);
            color.rgb = sanitizeColor(color.rgb);

            // 7. 褪色效果
            if (uFade > 0.001) {
                float fadeAmount = uFade * 0.3;
                color.rgb = mix(color.rgb, vec3(0.5), fadeAmount) + fadeAmount * 0.1;
                color.rgb = sanitizeColor(color.rgb);
            }

            // 8. 留银冲洗
            if (uBleachBypass > 0.001) {
                luma = dot(color.rgb, W);
                vec3 desaturated = mix(color.rgb, vec3(luma), 0.6);
                desaturated = (desaturated - 0.5) * 1.3 + 0.5;
                desaturated.r *= 0.95;
                desaturated.g *= 1.02;
                desaturated.b *= 1.05;
                color.rgb = mix(color.rgb, desaturated, uBleachBypass);
                color.rgb = sanitizeColor(color.rgb);
            }

            // 9. 晕影
            if (abs(uVignette) > 0.001) {
                float dist = distance(vTexCoord, vec2(0.5));
                float vignetteMask = smoothstep(0.8, 0.3, dist);
                if (uVignette < 0.0) {
                    color.rgb *= mix(0.01, 1.0, vignetteMask) * abs(uVignette) + (1.0 + uVignette);
                } else {
                    color.rgb = mix(color.rgb, vec3(1.0), (1.0 - vignetteMask) * uVignette);
                }
                color.rgb = sanitizeColor(color.rgb);
            }

            // 10. 颗粒 (静态底片颗粒)
            if (uFilmGrain > 0.001) {
                color.rgb = applyDensityParticleGrain(color.rgb, uvCoord, uFilmGrain);
                color.rgb = sanitizeColor(color.rgb);
            }

            // 11. 随机噪点 (增强的亮度和彩色噪点，动态刷新)
            if (uNoise > 0.001) {
                // 将浮点精度控制在适合 fract 的范围，避免伪影
                vec2 seedOffset = vec2(fract(uNoiseSeed * 1.234), fract(uNoiseSeed * 3.456));
                vec2 noiseCoord = uvCoord * 800.0 + seedOffset * 100.0;

                // 亮度噪点
                float lumNoise = fract(sin(dot(noiseCoord, vec2(12.9898, 78.233))) * 43758.5453);
                lumNoise = (lumNoise - 0.5) * 2.0;

                // 彩色噪点 (R, G, B 分别生成)
                float colorNoiseR = fract(sin(dot(noiseCoord + vec2(1.1, 2.2), vec2(39.346, 11.135))) * 43758.5453);
                float colorNoiseG = fract(sin(dot(noiseCoord + vec2(3.3, 4.4), vec2(73.156, 52.235))) * 43758.5453);
                float colorNoiseB = fract(sin(dot(noiseCoord + vec2(5.5, 6.6), vec2(27.423, 83.136))) * 43758.5453);
                vec3 colorNoise = (vec3(colorNoiseR, colorNoiseG, colorNoiseB) - 0.5) * 2.0;

                luma = dot(color.rgb, W);
                // 蒙版强度：中灰区域稍强，极端高光/阴影保留
                float noiseMask = mix(0.5, 1.0, 1.0 - abs(luma - 0.5) * 1.5);

                // 混合：结合亮度和彩色噪点，使彩色噪点不那么突兀但又可见
                vec3 finalNoise = mix(vec3(lumNoise), mix(vec3(lumNoise), colorNoise, 0.7), 0.8);

                // 基准强度为 0.3，效果较明显
                color.rgb += finalNoise * uNoise * max(0.0, noiseMask);
                color.rgb = sanitizeColor(color.rgb);
            }

            color.rgb = sanitizeColor(color.rgb);
        }

        // === 曲线调整（色彩配方之后、LUT 之前） ===
        if (uCurveEnabled) {
            vec3 clamped = clamp(color.rgb, 0.0, 1.0); // 纹理坐标需要 [0,1]
            float r = texture(uCurveTexture, vec2(clamped.r, 0.5)).r;
            float g = texture(uCurveTexture, vec2(clamped.g, 0.5)).g;
            float b = texture(uCurveTexture, vec2(clamped.b, 0.5)).b;
            color.rgb = sanitizeColor(vec3(r, g, b));
        }
        
        if (uVideoLogEnabled) {
            vec3 linearColor = srgbToLinear(max(color.rgb, vec3(0.0)));
            vec3 outputColorSpace = applyLutColorSpace(linearColor, uVideoColorSpace);
            color.rgb = sanitizeColor(applyLutCurve(outputColorSpace, uVideoLogCurve));
        }

        // === LUT 处理（在色彩配方之后） ===
        if (uLutEnabled && uLutIntensity > 0.0) {
            vec3 lutInColor;
            float effectiveLutIntensity = uLutIntensity;
            if (uVideoLogEnabled) {
                // Video Log 模式：color.rgb 已经是 Log 编码，直接作为 LUT 坐标
                lutInColor = color.rgb;
            } else {
                vec3 linearRGB = srgbToLinear(max(color.rgb, vec3(0.0)));
                effectiveLutIntensity *= lutMaskWeight(uLutMaskType, linearRGB);
                vec3 colorSpaceRGB = applyLutColorSpace(linearRGB, uLutColorSpace);
                lutInColor = applyLutCurve(colorSpaceRGB, uLutCurve);
            }
            float scale = (uLutSize - 1.0) / uLutSize;
            float offset = 1.0 / (2.0 * uLutSize);
            vec3 lutCoord = lutInColor * scale + offset;
            vec4 lutColor = texture(uLutTexture, lutCoord);
            color.rgb = mix(color.rgb, lutColor.rgb, effectiveLutIntensity);
            color.rgb = sanitizeColor(color.rgb);
        }

        fragColor = vec4(clamp(sanitizeColor(color.rgb), 0.0, 1.0), color.a);
    }
    """.trimIndent()

    val FRAGMENT_SHADER_COLOR_RECIPE_2D = FRAGMENT_SHADER_COLOR_RECIPE
        .replace("#extension GL_OES_EGL_image_external_essl3 : require\n", "")
        .replace("uniform samplerExternalOES uCameraTexture;", "uniform sampler2D uCameraTexture;")

    /**
     * 全屏四边形的顶点坐标
     * 覆盖整个屏幕 (-1, -1) 到 (1, 1)
     */
    val FULL_QUAD_VERTICES = floatArrayOf(
        // X, Y
        -1.0f, -1.0f,  // 左下
        1.0f, -1.0f,  // 右下
        -1.0f, 1.0f,  // 左上
        1.0f, 1.0f   // 右上
    )

    /**
     * 纹理坐标
     * OpenGL 纹理坐标系：左下角为 (0, 0)
     */
    val TEXTURE_COORDS = floatArrayOf(
        // U, V
        0.0f, 0.0f,  // 左下
        1.0f, 0.0f,  // 右下
        0.0f, 1.0f,  // 左上
        1.0f, 1.0f   // 右上
    )

    /**
     * 后处理专用纹理坐标（垂直翻转）
     * 用于让 glReadPixels 直接读取到正向的图片
     */
    val POST_PROCESS_TEXTURE_COORDS = floatArrayOf(
        0.0f, 1.0f, // Top-left -> GL Bottom-left
        1.0f, 1.0f, // Top-right -> GL Bottom-right
        0.0f, 0.0f, // Bottom-left -> GL Top-left
        1.0f, 0.0f  // Bottom-right -> GL Top-right
    )

    /**
     * 绘制顺序索引
     * 使用两个三角形绘制四边形
     */
    val DRAW_ORDER = shortArrayOf(
        0, 1, 2,  // 第一个三角形
        1, 3, 2   // 第二个三角形
    )

    /**
     * 高质量 Bokeh 片元着色器 (OpenGL ES 3.0)
     * 采用 Golden-Angle 螺旋采样实现圆盘虚化
     */
    val BOKEH_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uInputTexture;
        uniform sampler2D uDepthTexture;

        uniform mat4 uDepthMatrix; // 用于对齐深度图坐标（处理 Y 翻转和 FOV 缩放）
        uniform float uMaxBlurRadius;
        uniform float uAperture;
        uniform float uFocusDepth;
        uniform vec2 uTexelSize;

        const float PI = 3.14159265359;
        const float GOLDEN_ANGLE = 2.39996323;
        const int SAMPLES = 64; // 实时预览：64 采样配合 Jitter 已足够顺滑且性能均衡

        // Interleaved Gradient Noise (IGN) - 用于低采样下消除环状伪影
        float random(vec2 fragCoord) {
            vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
            return fract(magic.z * fract(dot(fragCoord, magic.xy)));
        }

        void main() {
            vec2 depthUV = (uDepthMatrix * vec4(vTexCoord, 0.0, 1.0)).xy;
            vec4 centerColor = texture(uInputTexture, vTexCoord);
            float centerDepth = texture(uDepthTexture, depthUV).r;

            float coc = abs(centerDepth - uFocusDepth) * uMaxBlurRadius * (1.0 / uAperture);
            coc = clamp(coc, 0.0, uMaxBlurRadius);

            if (coc < 0.5) {
                fragColor = centerColor;
                return;
            }

            vec3 accColor = vec3(0.0);
            float accWeight = 0.0;

            // 重要优化：利用 IGN 随机扰动旋转角度，消除固定采样模式带来的色带感
            // 相对于 brute-force 160 采样，这样能以更低功耗达到相同平滑度
            float jitter = random(gl_FragCoord.xy) * GOLDEN_ANGLE;

            for (int i = 0; i < SAMPLES; i++) {
                // 面积均匀分布
                float r = sqrt(float(i) / float(SAMPLES)) * coc;
                float theta = float(i) * GOLDEN_ANGLE + jitter;

                vec2 offset = vec2(cos(theta), sin(theta)) * r * uTexelSize;
                vec2 sampleUV = clamp(vTexCoord + offset, 0.0, 1.0);

                // Mipmap LOD 计算，融合相邻片元，在不加入噪点的情况下自然抹平色带
                float sampleRadiusPixels = r * 1.5 / sqrt(float(SAMPLES));
                float lod = max(0.0, log2(sampleRadiusPixels));

                vec3 sampleColor = texture(uInputTexture, sampleUV, lod).rgb;
                vec2 sDepthUV = clamp((uDepthMatrix * vec4(sampleUV, 0.0, 1.0)).xy, 0.0, 1.0);
                float sampleDepth = texture(uDepthTexture, sDepthUV).r;

                float sampleCoc = abs(sampleDepth - uFocusDepth) * uMaxBlurRadius * (1.0 / uAperture);
                sampleCoc = clamp(sampleCoc, 0.0, uMaxBlurRadius);

                // 软权重逻辑
                float w = smoothstep(r - 0.5, r + 1.0, sampleCoc);

                float depthDiff = sampleDepth - centerDepth;
                float bgOcclusion = smoothstep(-0.10, -0.01, depthDiff);
                float fgHalo = smoothstep(0.06, 0.01, depthDiff);
                w *= bgOcclusion * fgHalo;

                accColor += sampleColor * w;
                accWeight += w;
            }
            vec3 finalColor = accWeight > 0.001 ? (accColor / accWeight) : centerColor.rgb;
            fragColor = vec4(finalColor, centerColor.a);
        }
    """.trimIndent()

    /**
     * 无缝联合双边上采样 (Seamless JBU)
     * 采用标准的 2x2 邻域双线性混合，配合颜色权重，彻底消除网格感。
     */
    val JBU_UPSAMPLE_FRAGMENT_SHADER = """#version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uLowResDepth;  
        uniform sampler2D uHighResGuide; 
        uniform vec2 uLowResTexelSize;   

        const float SIGMA_R = 0.12; 

        void main() {
            vec3 guideColor = texture(uHighResGuide, vTexCoord).rgb;
            
            // 基础线性混合深度，作为极端情况下的保底
            float baseDepth = texture(uLowResDepth, vTexCoord).r;
            
            // 计算在低分辨率纹理空间下的坐标
            vec2 pos = vTexCoord / uLowResTexelSize - 0.5;
            vec2 p0 = floor(pos);
            vec2 f = fract(pos);
            
            float totalWeight = 0.0;
            float totalDepth = 0.0;

            // 采样相邻的 2x2 个低分中心点 (标准双线性权重范围)
            for(int y = 0; y <= 1; y++) {
                for(int x = 0; x <= 1; x++) {
                    vec2 offset = vec2(float(x), float(y));
                    vec2 sampleCoord = (p0 + offset + 0.5) * uLowResTexelSize;
                    
                    float d = texture(uLowResDepth, sampleCoord).r;
                    vec3 c = texture(uHighResGuide, sampleCoord).rgb;

                    // 1. 标准双线性空间权重 (线性连续，无边界跳变)
                    float wS = (x == 0 ? (1.0 - f.x) : f.x) * (y == 0 ? (1.0 - f.y) : f.y);

                    // 2. 颜色相似度权重
                    float dC = distance(guideColor, c);
                    float wC = exp(-(dC * dC) / (2.0 * SIGMA_R * SIGMA_R));

                    float w = wS * wC;
                    totalDepth += d * w;
                    totalWeight += w;
                }
            }

            float finalDepth = totalWeight > 0.001 ? totalDepth / totalWeight : baseDepth;
            fragColor = vec4(vec3(finalDepth), 1.0);
        }
    """.trimIndent()

    /**
     * 软细节增强 (Soft Detail Refiner)
     * 取代暴力的锐化，只做温和的边缘收缩
     */
    val DEPTH_SHARPEN_FRAGMENT_SHADER = """#version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uDepthTexture;
        uniform vec2 uTexelSize;

        void main() {
            float center = texture(uDepthTexture, vTexCoord).r;
            
            // 采用极小半径平滑
            float n = texture(uDepthTexture, vTexCoord + vec2(0, uTexelSize.y)).r;
            float s = texture(uDepthTexture, vTexCoord - vec2(0, uTexelSize.y)).r;
            float e = texture(uDepthTexture, vTexCoord + vec2(uTexelSize.x, 0)).r;
            float w = texture(uDepthTexture, vTexCoord - vec2(uTexelSize.x, 0)).r;

            float avg = (n + s + e + w + center) / 5.0;
            
            // 温和的对比度拉伸，不产生硬边缘
            float refined = mix(center, smoothstep(0.05, 0.95, center), 0.3);
            
            fragColor = vec4(vec3(clamp(refined, 0.0, 1.0)), 1.0);
        }
    """.trimIndent()

    /**
     * 后期处理专用：物理级的 PSF Splatting 模拟算法。
     * 采用大半径 Gather、逆向 HDR 提亮（点光源扩张）、能量守恒、球面像差（肥皂泡边缘）和口径蚀来高度还原真实的单反镜头虚化质感。
     */
    val PSF_SPLAT_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uInputTexture;
        uniform sampler2D uDepthTexture;

        uniform mat4 uDepthMatrix;
        uniform float uMaxBlurRadius;
        uniform float uAperture;
        uniform float uFocusDepth;
        uniform vec2 uTexelSize;

        const float PI = 3.14159265359;
        const float GOLDEN_ANGLE = 2.39996323;
        const int SAMPLES = 640;
        const float LENS_GAMMA = 2.2;

        // 简单的哈希函数，用于产生像素级的抖动
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
        }

        float backgroundGap(float depth) {
            return max(uFocusDepth - depth - 0.025, 0.0);
        }

        float computeCoc(float depth) {
            // Align with BokehMe: Symmetric defocus for both foreground and background
            float gap = max(abs(uFocusDepth - depth) - 0.015, 0.0);
            float defocus = pow(gap, 1.1);
            return clamp(defocus * uMaxBlurRadius * (1.0 / max(uAperture, 0.45)), 0.0, uMaxBlurRadius);
        }

        float apertureWeight(vec2 offsetPixels, float coc) {
            vec2 p = offsetPixels / max(coc, 0.001);
            float lenP = length(p);
            
            // Perfect circular "creamy" bokeh with a soap bubble rim.
            float inside = smoothstep(1.0, 0.88, lenP);
            float rim = smoothstep(0.7, 0.98, lenP);
            return inside * (1.0 + rim * 0.4); 
        }

        vec3 toLinear(vec3 color) {
            return pow(clamp(color, 0.0, 1.0), vec3(LENS_GAMMA));
        }

        vec3 toDisplay(vec3 color) {
            return pow(max(color, vec3(0.0)), vec3(1.0 / LENS_GAMMA));
        }

        void main() {
            vec2 depthUV = clamp((uDepthMatrix * vec4(vTexCoord, 0.0, 1.0)).xy, 0.0, 1.0);
            vec4 centerColor = texture(uInputTexture, vTexCoord);
            float centerDepth = texture(uDepthTexture, depthUV).r;

            float centerCoc = computeCoc(centerDepth);

            if (centerCoc < 0.2) {
                fragColor = centerColor;
                return;
            }

            // 引入随机旋转抖动，打破 Vogel Spiral 的环状条纹
            float noise = hash(vTexCoord + 0.5);
            float rotation = noise * PI * 2.0;

            float centerWeight = 4.0 / (centerCoc * 0.3 + 1.0);
            vec3 accColor = toLinear(centerColor.rgb) * centerWeight;
            float accWeight = centerWeight;

            float softBase = max(2.5, uMaxBlurRadius * 0.08);

            for (int i = 0; i < SAMPLES; i++) {
                float f = float(i + 1);
                float r = sqrt(f / float(SAMPLES)) * uMaxBlurRadius;
                float theta = f * GOLDEN_ANGLE + rotation;

                vec2 offset = vec2(cos(theta), sin(theta)) * r * uTexelSize;
                vec2 sampleUV = clamp(vTexCoord + offset, 0.0, 1.0);
                vec2 offsetPixels = offset / uTexelSize;

                // 第一遍采样获取亮度，用于决定 LOD
                vec3 sColorBase = textureLod(uInputTexture, sampleUV, 2.0).rgb;
                float baseLuma = dot(sColorBase, vec3(0.299, 0.587, 0.114));
                
                // 动态 LOD：高光处使用更高的 LOD 使其融合，消除条纹
                float lod = log2(r * 0.3 + 1.8) + smoothstep(0.4, 0.9, baseLuma) * 1.5;
                vec3 sColor = textureLod(uInputTexture, sampleUV, lod).rgb;
                
                vec2 sDepthUV = clamp((uDepthMatrix * vec4(sampleUV, 0.0, 1.0)).xy, 0.0, 1.0);
                float sDepth = texture(uDepthTexture, sDepthUV).r;

                float sCoc = computeCoc(sDepth);

                float fW = smoothstep(r - softBase, r + softBase * 0.5, sCoc);
                float bW = smoothstep(r - softBase, r + softBase * 0.5, centerCoc);

                float depthDiff = sDepth - centerDepth;
                float isNearer = smoothstep(0.01, 0.04, depthDiff);

                float weight = mix(bW, fW, isNearer);
                weight *= apertureWeight(offsetPixels, max(sCoc, centerCoc));

                float isSharpForeground = isNearer * (1.0 - smoothstep(0.3, 2.0, sCoc));
                weight *= (1.0 - isSharpForeground);

                if (weight > 0.0001) {
                    vec3 sLinear = toLinear(sColor);
                    float luma = dot(sLinear, vec3(0.2126, 0.7152, 0.0722));
                    
                    // 更加平滑的高光增强曲线
                    float highlight = max(0.0, luma - 0.55);
                    float hdrBoost = 1.0 + pow(highlight, 1.8) * 18.0 * smoothstep(1.5, 5.0, sCoc);

                    float edge = smoothstep(0.75, 1.03, r / max(sCoc, 0.1));
                    float ring = 1.0 + edge * 0.3 * smoothstep(1.5, 5.0, sCoc);

                    float fw = weight * ring;
                    accColor += sLinear * hdrBoost * fw;
                    accWeight += fw;
                }
            }

            vec3 finalColor = accWeight > 0.001 ? toDisplay(accColor / accWeight) : centerColor.rgb;
            finalColor = clamp(finalColor, 0.0, 1.0);

            fragColor = vec4(finalColor, centerColor.a);
        }
    """.trimIndent()
}
