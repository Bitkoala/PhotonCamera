package com.hinnka.mycamera.raw

import com.hinnka.mycamera.lut.ShadowsHighlightsShader

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
     * 3. Shared local shadows/highlights in working RGB, before curve clipping
     * 4. Highlight rolloff + DCP tone curve or ACR3 curve
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
        uniform sampler3D uDcpHueSatTexture;
        uniform sampler3D uDcpLookTableTexture;
        uniform sampler3D uSpectralFilmTexture;
        uniform sampler3D uAgxBaseSrgbTexture;
        uniform mat3 uOutputTransform;
        uniform float uCurveSize;
        uniform bool uCurveEnabled;
        uniform vec2 uTexelSize;
        uniform float uHighlights;
        uniform float uShadows;
        uniform bool uDcpHueSatEnabled;
        uniform bool uDcpLookTableEnabled;
        uniform ivec3 uDcpHueSatDivisions;
        uniform ivec3 uDcpLookTableDivisions;
        uniform int uDcpHueSatEncoding;
        uniform int uDcpLookTableEncoding;
        uniform int uSpectralFilmSize;
        uniform int uAgxLutSize;
        uniform int uRawColorEngine;

        const int RAW_COLOR_ENGINE_ADOBE_CURVE = 0;
        const int RAW_COLOR_ENGINE_AGX = 1;
        const int RAW_COLOR_ENGINE_SPECTRAL_FILM = 2;
        const float AGX_LOG_MIN = -12.47393;
        const float AGX_LOG_MAX = 12.5260688117;
        
        float luminance(vec3 color) {
            return max(dot(color, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
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

            vec3 lookupHsv = vec3(tableHsv.x, tableHsv.y, clamp(tableHsv.z, 0.0, 1.0));
            vec3 modify = sampleDcpMap(tableTexture, divisions, lookupHsv);
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

        vec3 linearToProPhoto(vec3 color) {
            vec3 clamped = max(color, vec3(0.0));
            vec3 isHigh = step(vec3(0.001953125), clamped);
            vec3 lowPart = 16.0 * clamped;
            vec3 highPart = pow(clamped, vec3(1.0 / 1.8));
            return mix(lowPart, highPart, isHigh);
        }

        vec3 proPhotoToLinear(vec3 color) {
            vec3 clamped = clamp(color, 0.0, 1.0);
            vec3 isHigh = step(vec3(0.03125), clamped);
            vec3 lowPart = clamped / 16.0;
            vec3 highPart = pow(clamped, vec3(1.8));
            return mix(lowPart, highPart, isHigh);
        }

        vec3 applySpectralFilm(vec3 color) {
            if (uSpectralFilmSize <= 1) {
                return color;
            }
            vec3 normalizedColor = color / 2.88;
            vec3 encodedColor = linearToProPhoto(normalizedColor);
            vec3 lutCoord = clamp(encodedColor, 0.0, 1.0);
            vec3 lutResult = texture(uSpectralFilmTexture, lutCoord).rgb;
            return proPhotoToLinear(lutResult);
        }

        vec3 fetchCubeLut(sampler3D lutTexture, int lutSize, ivec3 coord) {
            ivec3 maxCoord = ivec3(max(lutSize - 1, 0));
            return texelFetch(lutTexture, clamp(coord, ivec3(0), maxCoord), 0).rgb;
        }

        vec3 sampleCubeLut(sampler3D lutTexture, int lutSize, vec3 coord) {
            if (lutSize <= 1) {
                return coord;
            }

            float maxIndex = float(lutSize - 1);
            vec3 scaled = clamp(coord, vec3(0.0), vec3(1.0)) * maxIndex;
            ivec3 p0 = ivec3(floor(scaled));
            ivec3 p1 = min(p0 + ivec3(1), ivec3(lutSize - 1));
            vec3 f = scaled - vec3(p0);

            vec3 c000 = fetchCubeLut(lutTexture, lutSize, p0);
            vec3 c100 = fetchCubeLut(lutTexture, lutSize, ivec3(p1.x, p0.y, p0.z));
            vec3 c010 = fetchCubeLut(lutTexture, lutSize, ivec3(p0.x, p1.y, p0.z));
            vec3 c001 = fetchCubeLut(lutTexture, lutSize, ivec3(p0.x, p0.y, p1.z));
            vec3 c110 = fetchCubeLut(lutTexture, lutSize, ivec3(p1.x, p1.y, p0.z));
            vec3 c101 = fetchCubeLut(lutTexture, lutSize, ivec3(p1.x, p0.y, p1.z));
            vec3 c011 = fetchCubeLut(lutTexture, lutSize, ivec3(p0.x, p1.y, p1.z));
            vec3 c111 = fetchCubeLut(lutTexture, lutSize, p1);

            if (f.x >= f.y) {
                if (f.y >= f.z) {
                    return c000 + f.x * (c100 - c000) + f.y * (c110 - c100) + f.z * (c111 - c110);
                } else if (f.x >= f.z) {
                    return c000 + f.x * (c100 - c000) + f.z * (c101 - c100) + f.y * (c111 - c101);
                } else {
                    return c000 + f.z * (c001 - c000) + f.x * (c101 - c001) + f.y * (c111 - c101);
                }
            } else {
                if (f.x >= f.z) {
                    return c000 + f.y * (c010 - c000) + f.x * (c110 - c010) + f.z * (c111 - c110);
                } else if (f.y >= f.z) {
                    return c000 + f.y * (c010 - c000) + f.z * (c011 - c010) + f.x * (c111 - c011);
                } else {
                    return c000 + f.z * (c001 - c000) + f.y * (c011 - c001) + f.x * (c111 - c011);
                }
            }
        }

        vec3 sampleAgxBaseSrgb(vec3 coord) {
            return sampleCubeLut(uAgxBaseSrgbTexture, uAgxLutSize, coord);
        }

        vec3 agxLogEncode(vec3 color) {
            vec3 safeColor = max(color, vec3(exp2(AGX_LOG_MIN)));
            return clamp(
                (log2(safeColor) - vec3(AGX_LOG_MIN)) / (AGX_LOG_MAX - AGX_LOG_MIN),
                vec3(0.0),
                vec3(1.0)
            );
        }

        vec3 applyAgX(vec3 color) {
            if (uAgxLutSize <= 1) {
                return uOutputTransform * applyAdobeCurve(color);
            }
            vec3 egamut = max(color, vec3(0.0));
            vec3 agxLog = agxLogEncode(egamut);
            vec3 rec1886Encoded = sampleAgxBaseSrgb(agxLog);
            return pow(max(rec1886Encoded, vec3(0.0)), vec3(2.4));
        }

        float proPhotoLuminance(vec3 color) {
            return max(dot(color, vec3(0.2880402, 0.7118741, 0.0000857)), 1e-5);
        }

        float highlightRolloffLuma(float luma) {
            const float rolloffStart = 0.2;
            const float rolloffRange = 1.0 - rolloffStart;
            if (luma <= rolloffStart) return luma;

            float x = (luma - rolloffStart) / rolloffRange;
            return 1.0 - rolloffRange / (1.0 + x);
        }

        vec3 highlightRolloff(vec3 color) {
            float luma = proPhotoLuminance(color);
            float newLuma = highlightRolloffLuma(luma);
            return color * (newLuma / max(luma, 1e-6));
        }
        
        vec3 applyDcpMaps(vec3 color) {
            if (uDcpHueSatEnabled) {
                color = applyDcpHsvMap(color, uDcpHueSatTexture, uDcpHueSatDivisions, uDcpHueSatEncoding);
            }
            if (uDcpLookTableEnabled) {
                color = applyDcpHsvMap(color, uDcpLookTableTexture, uDcpLookTableDivisions, uDcpLookTableEncoding);
            }
            return color;
        }

        vec3 applyDisplayTone(vec3 color) {
            if (uRawColorEngine == RAW_COLOR_ENGINE_AGX) {
                return applyAgX(color);
            }
            if (uRawColorEngine == RAW_COLOR_ENGINE_SPECTRAL_FILM && uSpectralFilmSize > 1) {
                return applySpectralFilm(color);
            }
            color = applyDcpMaps(color);
            color = highlightRolloff(color);
            return applyAdobeCurve(color);
        }

        vec3 applyOutputTransformAfterDisplayTone(vec3 color) {
            if (uRawColorEngine == RAW_COLOR_ENGINE_AGX) {
                return color;
            }
            return uOutputTransform * color;
        }
        
        vec3 sampleToneSource(vec2 uv) {
            vec3 sampleColor = texture(uInputTexture, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
            vec3 color = applyDisplayTone(sampleColor);
            color = applyOutputTransformAfterDisplayTone(color);
            return color;
        }

        vec3 shRgbToXyz(vec3 rgb) {
            return mat3(
                0.7976749, 0.2880402, 0.0000000,
                0.1351917, 0.7118741, 0.0000000,
                0.0313534, 0.0000857, 0.8252100
            ) * rgb;
        }

        vec3 shXyzToRgb(vec3 xyz) {
            return mat3(
                1.3459433, -0.5445989, 0.0000000,
               -0.2556075,  1.5081673, 0.0000000,
               -0.0511118,  0.0205351, 1.2118128
            ) * xyz;
        }

        ${ShadowsHighlightsShader.GLSL}

        void main() {
            vec3 color = texture(uInputTexture, vTexCoord).rgb;
            color = applyDisplayTone(color);
            color = applyOutputTransformAfterDisplayTone(color);
            color = applyShadowsHighlights(color, vTexCoord);
            color = linearToSrgb(color);
            fragColor = vec4(color, 1.0);
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
