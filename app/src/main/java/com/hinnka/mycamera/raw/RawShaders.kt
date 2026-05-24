package com.hinnka.mycamera.raw

/**
 * RAW 图像处理的 GLSL 着色器
 *
 * 实现完整的 RAW 处理管线：
 * 1. 黑电平校正和归一化
 * 2. Malvar-He-Cutler (MHC) 解马赛克算法
 * 3. 白平衡增益
 * 4. 色彩校正矩阵 (CCM)
 * 5. Gamma 校正 (sRGB)
 */
object RawShaders {

    /**
     * 顶点着色器 - 简单的全屏四边形渲染
     */
    val VERTEX_SHADER = """
        #version 300 es
        
        in vec4 aPosition;
        in vec2 aTexCoord;
        
        out vec2 vTexCoord;
        
        uniform mat4 uTexMatrix;
        
        void main() {
            gl_Position = aPosition;
            vTexCoord = (uTexMatrix * vec4(aTexCoord, 0.0, 1.0)).xy;
        }
    """.trimIndent()

    /**
     * 全屏四边形顶点坐标
     */
    val FULL_QUAD_VERTICES = floatArrayOf(
        -1.0f, -1.0f,  // 左下
        1.0f, -1.0f,  // 右下
        -1.0f, 1.0f,  // 左上
        1.0f, 1.0f   // 右上
    )

    /**
     * 纹理坐标（Y 轴翻转，适配 Android Bitmap）
     */
    val TEXTURE_COORDS = floatArrayOf(
        0.0f, 0.0f,  // LB viewport -> Tex (0,0) [Sensor Row 0/Bottom of Tex] -> glReadPixels reads to Bitmap Top
        1.0f, 0.0f,  // RB viewport -> Tex (1,0)
        0.0f, 1.0f,  // LT viewport -> Tex (0,1)
        1.0f, 1.0f   // RT viewport -> Tex (1,1)
    )

    /**
     * 片元着色器 - 第二步：将解马赛克后的 RGB 纹理渲染到最终尺寸
     * 应用旋转、裁切和缩放
     */
    val PASSTHROUGH_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        
        in vec2 vTexCoord;
        out vec4 fragColor;
        
        uniform sampler2D uTexture;
        
        void main() {
            fragColor = texture(uTexture, vTexCoord);
        }
    """.trimIndent()

    /**
     * Combined Processing Shader
     *
     * Process chain:
     * 1. Linear RGB input in working space
     * 2. DCP hue/sat map and look table
     * 3. Local SDR tone mapping on working-linear data
     * 4. DCP tone curve or ACR3 curve (mutually exclusive)
     * 5. Working space -> Linear sRGB
     * 6. Linear sRGB -> sRGB
     */
    val COMBINED_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp sampler3D;
        
        in vec2 vTexCoord;
        out vec4 fragColor;
        
        uniform sampler2D uInputTexture;
        uniform sampler2D uCurveTexture;
        uniform sampler2D uHighlightBaseTexture;
        uniform sampler2D uLensShadingMap;
        uniform sampler3D uDcpHueSatTexture;
        uniform sampler3D uDcpLookTableTexture;
        uniform mat3 uOutputTransform;
        uniform float uCurveSize;
        uniform bool uCurveEnabled;
        uniform float uHighlightWhitePoint;
        uniform bool uHighlightBaseEnabled;
        uniform bool uLensShadingEnabled;
        uniform float uLensShadingPower;
        uniform bool uDcpHueSatEnabled;
        uniform bool uDcpLookTableEnabled;
        uniform ivec3 uDcpHueSatDivisions;
        uniform ivec3 uDcpLookTableDivisions;
        uniform int uDcpHueSatEncoding;
        uniform int uDcpLookTableEncoding;
        
        float luminance(vec3 color) {
            return max(dot(color, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
        }

        vec3 applyLensShading(vec3 color) {
            if (!uLensShadingEnabled) {
                return color;
            }
            vec4 gain = texture(uLensShadingMap, vTexCoord);
            vec4 centerGain = texture(uLensShadingMap, vec2(0.5));
            float lumaGain = dot(vec3(gain.r, 0.5 * (gain.g + gain.b), gain.a), vec3(0.2126, 0.7152, 0.0722));
            float centerLumaGain = dot(vec3(centerGain.r, 0.5 * (centerGain.g + centerGain.b), centerGain.a), vec3(0.2126, 0.7152, 0.0722));
            lumaGain /= max(centerLumaGain, 1e-4);
            lumaGain = pow(max(lumaGain, 1e-4), clamp(uLensShadingPower, 0.0, 1.0));
            return color * max(lumaGain, 0.0);
        }

        float sampleCurve(float value) {
            if (!uCurveEnabled || uCurveSize <= 1.0) {
                return value;
            }
            float clampedValue = clamp(value, 0.0, 1.0);
            float coordX = clampedValue * ((uCurveSize - 1.0) / uCurveSize) + (0.5 / uCurveSize);
            return texture(uCurveTexture, vec2(coordX, 0.5)).r;
        }

        vec3 applyCurve(vec3 color) {
            float luma = luminance(color);
            float scale = sampleCurve(luma) / max(luma, 0.00001);
            return clamp(color * scale, 0.0, 1.0);
        }

        void adobeRgbTone(inout float maxValue, inout float midValue, inout float minValue) {
            float oldMax = maxValue;
            float oldMid = midValue;
            float oldMin = minValue;
            maxValue = sampleCurve(oldMax);
            minValue = sampleCurve(oldMin);
            if (abs(oldMax - oldMin) < 1e-6) {
                midValue = minValue;
            } else {
                midValue = minValue + ((maxValue - minValue) * (oldMid - oldMin) / (oldMax - oldMin));
            }
        }

        vec3 applyAdobeCurve(vec3 color) {
            vec3 clipped = clamp(color, 0.0, 1.0);
            float r = clipped.r;
            float g = clipped.g;
            float b = clipped.b;

            if (r >= g) {
                if (g > b) {
                    adobeRgbTone(r, g, b);
                } else if (b > r) {
                    adobeRgbTone(b, r, g);
                } else if (b > g) {
                    adobeRgbTone(r, b, g);
                } else {
                    r = sampleCurve(r);
                    g = sampleCurve(g);
                    b = g;
                }
            } else {
                if (r >= b) {
                    adobeRgbTone(g, r, b);
                } else if (b > g) {
                    adobeRgbTone(b, g, r);
                } else {
                    adobeRgbTone(g, b, r);
                }
            }

            return clamp(vec3(r, g, b), 0.0, 1.0);
        }

        vec3 linearToSrgb(vec3 color) {
            vec3 clampedColor = max(color, vec3(0.0));
            vec3 low = clampedColor * 12.92;
            vec3 high = 1.055 * pow(clampedColor, vec3(1.0 / 2.4)) - 0.055;
            bvec3 useHigh = greaterThan(clampedColor, vec3(0.0031308));
            return vec3(
                useHigh.r ? high.r : low.r,
                useHigh.g ? high.g : low.g,
                useHigh.b ? high.b : low.b
            );
        }

        float encodeValue(float value, int encoding) {
            value = clamp(value, 0.0, 1.0);
            if (encoding == 1) {
                return linearToSrgb(vec3(value)).r;
            }
            return value;
        }

        float decodeValue(float value, int encoding) {
            value = clamp(value, 0.0, 1.0);
            if (encoding == 1) {
                vec3 srgb = max(vec3(value), vec3(0.0));
                bvec3 useHigh = greaterThan(srgb, vec3(0.04045));
                vec3 low = srgb / 12.92;
                vec3 high = pow((srgb + 0.055) / 1.055, vec3(2.4));
                return useHigh.r ? high.r : low.r;
            }
            return value;
        }

        vec3 rgbToDcpHsv(vec3 rgb) {
            float maxValue = max(rgb.r, max(rgb.g, rgb.b));
            float minValue = min(rgb.r, min(rgb.g, rgb.b));
            float delta = maxValue - minValue;

            float hue = 0.0;
            if (delta > 1e-6) {
                if (maxValue == rgb.r) {
                    hue = mod((rgb.g - rgb.b) / delta, 6.0);
                } else if (maxValue == rgb.g) {
                    hue = ((rgb.b - rgb.r) / delta) + 2.0;
                } else {
                    hue = ((rgb.r - rgb.g) / delta) + 4.0;
                }
            }
            float sat = maxValue > 1e-6 ? delta / maxValue : 0.0;
            return vec3(hue, sat, maxValue);
        }

        vec3 dcpHsvToRgb(vec3 hsv) {
            float hue = mod(hsv.x, 6.0);
            float sat = max(hsv.y, 0.0);
            float value = max(hsv.z, 0.0);
            float chroma = value * sat;
            float x = chroma * (1.0 - abs(mod(hue, 2.0) - 1.0));
            vec3 rgb;
            if (hue < 1.0) rgb = vec3(chroma, x, 0.0);
            else if (hue < 2.0) rgb = vec3(x, chroma, 0.0);
            else if (hue < 3.0) rgb = vec3(0.0, chroma, x);
            else if (hue < 4.0) rgb = vec3(0.0, x, chroma);
            else if (hue < 5.0) rgb = vec3(x, 0.0, chroma);
            else rgb = vec3(chroma, 0.0, x);
            float matchValue = value - chroma;
            return rgb + vec3(matchValue);
        }

        vec3 sampleDcpMap(sampler3D tableTexture, ivec3 divisions, vec3 hsv) {
            int hueDivisions = divisions.x;
            int satDivisions = divisions.y;
            int valueDivisions = divisions.z;
            if (hueDivisions <= 0 || satDivisions <= 0 || valueDivisions <= 0) {
                return vec3(0.0, 1.0, 1.0);
            }

            float hScale = float(hueDivisions) / 6.0;
            float sScale = float(max(satDivisions - 1, 0));
            float vScale = float(max(valueDivisions - 1, 0));

            float hScaled = hsv.x * hScale;
            float sScaled = hsv.y * sScale;
            float vScaled = hsv.z * vScale;

            int maxHueIndex0 = hueDivisions - 1;
            int maxSatIndex0 = max(satDivisions - 2, 0);
            int maxValIndex0 = max(valueDivisions - 2, 0);

            int hIndex0 = int(floor(hScaled));
            int sIndex0 = min(int(floor(sScaled)), maxSatIndex0);
            int vIndex0 = min(int(floor(vScaled)), maxValIndex0);
            int hIndex1 = hIndex0 + 1;
            if (hIndex0 >= maxHueIndex0) {
                hIndex0 = maxHueIndex0;
                hIndex1 = 0;
            }

            float hFract1 = hScaled - float(hIndex0);
            float sFract1 = sScaled - float(sIndex0);
            float vFract1 = vScaled - float(vIndex0);
            float hFract0 = 1.0 - hFract1;
            float sFract0 = 1.0 - sFract1;
            float vFract0 = 1.0 - vFract1;

            vec3 p000 = texelFetch(tableTexture, ivec3(sIndex0, hIndex0, vIndex0), 0).rgb;
            vec3 p001 = texelFetch(tableTexture, ivec3(sIndex0, hIndex1, vIndex0), 0).rgb;
            vec3 p010 = texelFetch(tableTexture, ivec3(min(sIndex0 + 1, satDivisions - 1), hIndex0, vIndex0), 0).rgb;
            vec3 p011 = texelFetch(tableTexture, ivec3(min(sIndex0 + 1, satDivisions - 1), hIndex1, vIndex0), 0).rgb;

            vec3 v000 = p000;
            vec3 v001 = p001;
            vec3 v010 = p010;
            vec3 v011 = p011;

            if (valueDivisions > 1) {
                vec3 p100 = texelFetch(tableTexture, ivec3(sIndex0, hIndex0, min(vIndex0 + 1, valueDivisions - 1)), 0).rgb;
                vec3 p101 = texelFetch(tableTexture, ivec3(sIndex0, hIndex1, min(vIndex0 + 1, valueDivisions - 1)), 0).rgb;
                vec3 p110 = texelFetch(tableTexture, ivec3(min(sIndex0 + 1, satDivisions - 1), hIndex0, min(vIndex0 + 1, valueDivisions - 1)), 0).rgb;
                vec3 p111 = texelFetch(tableTexture, ivec3(min(sIndex0 + 1, satDivisions - 1), hIndex1, min(vIndex0 + 1, valueDivisions - 1)), 0).rgb;
                v000 = v000 * vFract0 + p100 * vFract1;
                v001 = v001 * vFract0 + p101 * vFract1;
                v010 = v010 * vFract0 + p110 * vFract1;
                v011 = v011 * vFract0 + p111 * vFract1;
            }

            vec3 edge0 = v000 * hFract0 + v001 * hFract1;
            vec3 edge1 = v010 * hFract0 + v011 * hFract1;
            return edge0 * sFract0 + edge1 * sFract1;
        }

        vec3 srgbToLinear(vec3 srgb) {
            vec3 color = max(srgb, vec3(0.0));
            bvec3 useHigh = greaterThan(color, vec3(0.04045));
            vec3 low = color / 12.92;
            vec3 high = pow((color + 0.055) / 1.055, vec3(2.4));
            return vec3(
                useHigh.r ? high.r : low.r,
                useHigh.g ? high.g : low.g,
                useHigh.b ? high.b : low.b
            );
        }

        vec3 applyDcpHsvMap(vec3 color, sampler3D tableTexture, ivec3 divisions, int encoding) {
            if (min(color.r, min(color.g, color.b)) < 0.0) {
                return color;
            }

            vec3 hsv = rgbToDcpHsv(color);
            vec3 tableHsv = hsv;
            if (encoding == 1 && divisions.z > 1) {
                tableHsv.z = linearToSrgb(vec3(hsv.z)).r;
            }

            vec3 modify = sampleDcpMap(tableTexture, divisions, tableHsv);
            hsv.x = mod(hsv.x + (modify.x * 6.0 / 360.0), 6.0);
            hsv.y = hsv.y * modify.y;
            if (encoding == 1) {
                float encodedValue = max(tableHsv.z * modify.z, 0.0);
                hsv.z = srgbToLinear(vec3(encodedValue)).r;
            } else {
                hsv.z = hsv.z * modify.z;
            }
            hsv.y = max(hsv.y, 0.0);
            hsv.z = max(hsv.z, 0.0);

            return dcpHsvToRgb(hsv);
        }

        float sampleLuma(vec2 uv) {
            return luminance(texture(uInputTexture, uv).rgb);
        }

        float localAverageLuma(vec2 uv, vec2 texelSize, float radius) {
            vec2 dx = vec2(texelSize.x * radius, 0.0);
            vec2 dy = vec2(0.0, texelSize.y * radius);

            float center = sampleLuma(uv) * 0.28;
            float axial =
                sampleLuma(clamp(uv + dx, vec2(0.0), vec2(1.0))) +
                sampleLuma(clamp(uv - dx, vec2(0.0), vec2(1.0))) +
                sampleLuma(clamp(uv + dy, vec2(0.0), vec2(1.0))) +
                sampleLuma(clamp(uv - dy, vec2(0.0), vec2(1.0)));
            float diagonal =
                sampleLuma(clamp(uv + dx + dy, vec2(0.0), vec2(1.0))) +
                sampleLuma(clamp(uv + dx - dy, vec2(0.0), vec2(1.0))) +
                sampleLuma(clamp(uv - dx + dy, vec2(0.0), vec2(1.0))) +
                sampleLuma(clamp(uv - dx - dy, vec2(0.0), vec2(1.0)));

            return center + axial * 0.12 + diagonal * 0.06;
        }

        const float HALO_THRESHOLD_MIN = 0.1;
        const float HALO_THRESHOLD_MAX = 0.4;
        const float RADIUS_SMALL = 1.5;
        const float RADIUS_LARGE = 8.0;

        float highlightRolloffLuma(float luma) {
            if (uHighlightWhitePoint <= 1.0) return luma;

            float start = 0.4;
            if (luma <= start) return luma;

            float x = (luma - start) / max(uHighlightWhitePoint - start, 1e-6);

            float y = x *
                (1.0 + x / uHighlightWhitePoint) /
                (1.0 + x);

            return mix(luma, start + y * (uHighlightWhitePoint - start),
                       smoothstep(start, uHighlightWhitePoint, luma));
        }
        
        float localAdaptationLuma() {
            vec2 texelSize = 1.0 / vec2(textureSize(uInputTexture, 0));

            // 1. 获取两个尺度的局部亮度
            float localSmall = localAverageLuma(vTexCoord, texelSize, RADIUS_SMALL);
            float localLarge = localAverageLuma(vTexCoord, texelSize, RADIUS_LARGE);
            
            // 2. 计算边缘对比度，决定混合权重
            float scaleContrast = abs(log2(localSmall + 1e-4) - log2(localLarge + 1e-4));
            return mix(localLarge, localSmall, smoothstep(HALO_THRESHOLD_MIN, HALO_THRESHOLD_MAX, scaleContrast));
        }
        
        
        vec3 highlightRolloff(vec3 color) {
            if (uHighlightWhitePoint <= 1.0) return color;
        
            float luma = luminance(color);
            float newLuma = highlightRolloffLuma(luma);
        
            return color * (newLuma / max(luma, 1e-6));
        }

        vec3 combinedLocalToneMapping(vec3 sceneLinear, float originalLuma) {
            float toneInputLuma = luminance(sceneLinear);
            float localAdaptation = localAdaptationLuma();

            // Local Reinhard-equivalent floor: this preserves the current acceptable baseline.
            float compressionFloor = toneInputLuma / (1.0 + localAdaptation);

            float baseLuma = uHighlightBaseEnabled
                ? max(texture(uHighlightBaseTexture, vTexCoord).r, 1e-4)
                : max(localAdaptation, 1e-4);
            float localContrast = originalLuma / baseLuma;
            float highlightMask = smoothstep(0.4, 0.95, max(originalLuma, toneInputLuma));

            // 独立 tone map: 局部适应量负责压缩，高光里亮于低频 base 的结构减少压缩。
            // 暗于 base 的像素不参与负向恢复，避免窗景整体发灰。
            float brightContrast = max(localContrast - 1.0, 0.0);
            float compressionRelease = min(brightContrast, 0.85) * 0.8 * highlightMask;
            float mappedLuma = compressionFloor * (1.0 + compressionRelease);

            // 保证不会比原始局部 Reinhard 更暗，也不会超过 tone 输入亮度或 SDR 白。
            mappedLuma = max(mappedLuma, compressionFloor);
            mappedLuma = min(mappedLuma, min(toneInputLuma, 1.0));

            return clamp(sceneLinear * (mappedLuma / max(toneInputLuma, 1e-5)), 0.0, 1.0);
        }

        void main() {
            vec3 color = texture(uInputTexture, vTexCoord).rgb;
            color = applyLensShading(color);

            if (uDcpHueSatEnabled) {
                color = applyDcpHsvMap(color, uDcpHueSatTexture, uDcpHueSatDivisions, uDcpHueSatEncoding);
            }
            if (uDcpLookTableEnabled) {
                color = applyDcpHsvMap(color, uDcpLookTableTexture, uDcpLookTableDivisions, uDcpLookTableEncoding);
            }

            float originalLuma = luminance(color);
            color = highlightRolloff(color);
            color = combinedLocalToneMapping(color, originalLuma);
            
            color = applyAdobeCurve(color);

            color = uOutputTransform * color;
            color = linearToSrgb(color);

            fragColor = vec4(color, 1.0);
        }
    """.trimIndent()

    /**
     * Low-resolution highlight base shader.
     *
     * R stores a broad luma base used to restore local contrast after SDR tone mapping.
     */
    val HIGHLIGHT_BASE_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uInputTexture;
        uniform sampler2D uLensShadingMap;
        uniform vec2 uSourceTexelSize;
        uniform float uBlurRadius;
        uniform bool uLensShadingEnabled;
        uniform float uLensShadingPower;

        float luminance(vec3 color) {
            return max(dot(color, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
        }

        vec3 applyLensShadingAt(vec3 color, vec2 uv) {
            if (!uLensShadingEnabled) {
                return color;
            }
            vec4 gain = texture(uLensShadingMap, uv);
            vec4 centerGain = texture(uLensShadingMap, vec2(0.5));
            float lumaGain = dot(vec3(gain.r, 0.5 * (gain.g + gain.b), gain.a), vec3(0.2126, 0.7152, 0.0722));
            float centerLumaGain = dot(vec3(centerGain.r, 0.5 * (centerGain.g + centerGain.b), centerGain.a), vec3(0.2126, 0.7152, 0.0722));
            lumaGain /= max(centerLumaGain, 1e-4);
            lumaGain = pow(max(lumaGain, 1e-4), clamp(uLensShadingPower, 0.0, 1.0));
            return color * max(lumaGain, 0.0);
        }

        float sampleLuma(vec2 uv) {
            uv = clamp(uv, vec2(0.0), vec2(1.0));
            return luminance(applyLensShadingAt(texture(uInputTexture, uv).rgb, uv));
        }

        void main() {
            vec2 stepSize = uSourceTexelSize * uBlurRadius;
            float sum = 0.0;
            float weightSum = 0.0;

            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    vec2 offset = vec2(float(x), float(y));
                    float weight = exp(-dot(offset, offset) * 0.5);
                    sum += sampleLuma(vTexCoord + offset * stepSize) * weight;
                    weightSum += weight;
                }
            }

            float baseLuma = sum / max(weightSum, 1e-5);
            fragColor = vec4(baseLuma, baseLuma, baseLuma, 1.0);
        }
    """.trimIndent()


    /**
     * Dedicated Sharpening Shader
     * Lightweight UnSharp Mask inspired by darktable's default sharpen preset.
     */
    val SHARPEN_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        
        in vec2 vTexCoord;
        out vec4 fragColor;
        
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        uniform float uSharpening;
        uniform float uRadius;
        uniform float uThreshold;

        float luminance(vec3 color) {
            return dot(color, vec3(0.2126, 0.7152, 0.0722));
        }
        
        void main() {
            vec3 center = texture(uInputTexture, vTexCoord).rgb;
            if (uSharpening <= 0.0) {
                fragColor = vec4(center, 1.0);
                return;
            }

            float r = max(uRadius, 0.001);
            float sigma = max(r * 0.5, 0.001);
            float twoSigma2 = 2.0 * sigma * sigma;
            vec3 blur = vec3(0.0);
            float weightSum = 0.0;

            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    vec2 offset = vec2(float(x), float(y));
                    float dist2 = dot(offset, offset);
                    float weight = exp(-dist2 / twoSigma2);
                    blur += texture(uInputTexture, vTexCoord + offset * uTexelSize * r).rgb * weight;
                    weightSum += weight;
                }
            }
            blur /= max(weightSum, 1e-5);

            float centerLuma = luminance(center);
            float blurLuma = luminance(blur);
            float delta = centerLuma - blurLuma;
            float detail = sign(delta) * max(abs(delta) - uThreshold, 0.0);
            vec3 result = center + center * (detail / max(centerLuma, 1e-5)) * uSharpening;

            fragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
        }
    """.trimIndent()

    /**
     * HDR Reference Shader
     *
     * RAW 线性输入已经按传感器白点归一化，直接输出会让拍到白点的灯光仍然只有
     * SDR reference white (= 1.0)，gainmap 没有高光余量可写。这里只扩展接近白点
     * 的 RAW 高光到 scene-linear HDR headroom，不做 SDR tone mapping。
     */
    val HDR_REFERENCE_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uInputTexture;
        uniform float uHighlightStart;
        uniform float uWhitePointSceneLuma;

        float luminance(vec3 color) {
            return max(dot(color, vec3(0.2126, 0.7152, 0.0722)), 1e-5);
        }

        void main() {
            vec3 color = max(texture(uInputTexture, vTexCoord).rgb, vec3(0.0));
            float luma = luminance(color);
            float highlight = smoothstep(uHighlightStart, 1.0, luma);
            float targetLuma = mix(luma, max(luma, uWhitePointSceneLuma), highlight);
            color *= targetLuma / luma;
            fragColor = vec4(max(color, vec3(0.0)), 1.0);
        }
    """.trimIndent()

    /**
     * 绘制顺序索引
     */
    val DRAW_ORDER = shortArrayOf(
        0, 1, 2,  // 第一个三角形
        1, 3, 2   // 第二个三角形
    )
}
