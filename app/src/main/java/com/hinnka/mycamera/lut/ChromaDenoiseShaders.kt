package com.hinnka.mycamera.lut

/**
 * Shared shader snippets used by bitmap/RAW chroma noise reduction.
 *
 * The pipeline is chroma-only multi-pass a trous wavelet denoise:
 * RGB -> YCbCr, repeated B-spline decompose, reverse soft-threshold synthesize,
 * then original luma + filtered chroma back to RGB.
 */
object ChromaDenoiseShaders {
    const val SIGMA_STRENGTH_AT_SLIDER_ONE = 6.0f
    const val WAVELET_BANDS = 4

    private val COMMON = """
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;

        float safeInvSq(float v) {
            return 1.0 / max(v * v, 1e-6);
        }
        vec3 rgb2ycbcr(vec3 rgb) {
            float y = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
            return vec3(y,
                        dot(rgb, vec3(-0.169, -0.331,  0.5  )),
                        dot(rgb, vec3( 0.5,  -0.419, -0.081 )));
        }
        vec3 ycbcr2rgb(vec3 ycc) {
            return vec3(ycc.x + 1.402  * ycc.z,
                        ycc.x - 0.3441 * ycc.y - 0.7141 * ycc.z,
                        ycc.x + 1.772  * ycc.y);
        }
        float noiseSigmaForLuma(vec2 noiseModel, float luma) {
            return sqrt(max(noiseModel.x * max(luma, 0.0) + noiseModel.y, 1e-10));
        }
        float bSplineTap(int offset) {
            int a = offset < 0 ? -offset : offset;
            if (a == 0) return 0.375;
            if (a == 1) return 0.25;
            return 0.0625;
        }
        vec2 softThreshold(vec2 value, float threshold) {
            return sign(value) * max(abs(value) - vec2(threshold), vec2(0.0));
        }
        float chromaProtection(float chromaLen, float noiseSigma) {
            return smoothstep(noiseSigma * 3.5 + 0.008, noiseSigma * 10.0 + 0.035, chromaLen);
        }
    """

    val PREPARE_YCBCR = """
        #version 300 es
        $COMMON
        uniform sampler2D uInputTexture;

        void main() {
            vec4 color = texture(uInputTexture, vTexCoord);
            fragColor = vec4(rgb2ycbcr(color.rgb), color.a);
        }
    """.trimIndent()

    val WAVELET_DECOMPOSE = """
        #version 300 es
        $COMMON
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        uniform vec2 uNoiseModel;
        uniform float uH;
        uniform int uScale;
        uniform int uOutputMode;

        vec4 sampleYcc(vec2 uv) {
            return texture(uInputTexture, uv);
        }

        vec4 coarseForCenter(vec4 centerYcc) {
            vec4 sum = vec4(0.0);
            float sumW = 0.0;
            float strength01 = clamp(uH / 6.0, 0.0, 1.0);
            float noiseSigma = noiseSigmaForLuma(uNoiseModel, centerYcc.x);
            float localH = max(uH * noiseSigma, 1e-5);
            float scaleF = float(uScale);
            float lumaSigma = max(
                noiseSigma * mix(1.55, 4.2, strength01) + 0.003 + log2(scaleF + 1.0) * 0.0015,
                localH * mix(0.42, 0.72, smoothstep(1.0, 8.0, scaleF))
            );
            float invLumaSigma2 = safeInvSq(lumaSigma);

            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    vec2 ofs = vec2(float(x), float(y)) * uTexelSize * scaleF;
                    vec4 sYcc = sampleYcc(vTexCoord + ofs);
                    float kernel = bSplineTap(x) * bSplineTap(y);
                    float dL = sYcc.x - centerYcc.x;
                    float guide = exp(-(dL * dL) * invLumaSigma2);
                    float w = kernel * guide;
                    sum += sYcc * w;
                    sumW += w;
                }
            }

            vec4 coarse = sum / max(sumW, 1e-6);
            coarse.a = centerYcc.a;
            return coarse;
        }

        void main() {
            vec4 centerYcc = sampleYcc(vTexCoord);
            vec4 coarse = coarseForCenter(centerYcc);
            fragColor = (uOutputMode == 0) ? coarse : vec4(centerYcc.rgb - coarse.rgb, centerYcc.a);
        }
    """.trimIndent()

    val WAVELET_SYNTHESIZE = """
        #version 300 es
        $COMMON
        uniform sampler2D uCoarseTexture;
        uniform sampler2D uDetailTexture;
        uniform vec2 uTexelSize;
        uniform vec2 uNoiseModel;
        uniform float uH;
        uniform float uBandScale;

        float flatLumaWeight(float centerLuma, float noiseSigma) {
            float left = texture(uCoarseTexture, vTexCoord + vec2(-uTexelSize.x, 0.0)).x;
            float right = texture(uCoarseTexture, vTexCoord + vec2(uTexelSize.x, 0.0)).x;
            float up = texture(uCoarseTexture, vTexCoord + vec2(0.0, -uTexelSize.y)).x;
            float down = texture(uCoarseTexture, vTexCoord + vec2(0.0, uTexelSize.y)).x;
            float localGradient = max(max(abs(centerLuma - left), abs(centerLuma - right)),
                                      max(abs(centerLuma - up), abs(centerLuma - down)));
            float edgeThreshold = noiseSigma * 2.2 + 0.006;
            return 1.0 - smoothstep(edgeThreshold * 0.65, edgeThreshold * 2.2, localGradient);
        }

        float bandThreshold(float luma, vec2 coarseChroma) {
            float noiseSigma = noiseSigmaForLuma(uNoiseModel, luma);
            float strength01 = clamp(uH / 6.0, 0.0, 1.0);
            float localH = max(uH * noiseSigma, 1e-5);
            float shadowWeight = (1.0 - smoothstep(0.035, 0.24, luma)) *
                                 smoothstep(0.004, 0.055, localH);
            float flatWeight = flatLumaWeight(luma, noiseSigma);
            float stableChroma = chromaProtection(length(coarseChroma), noiseSigma);
            float base = noiseSigma * mix(0.22, 1.28, strength01) + strength01 * 0.0025;
            float areaBoost = 1.0 + 1.2 * flatWeight + 0.75 * shadowWeight;
            float protectionScale = mix(1.0, 0.45, stableChroma * (1.0 - shadowWeight * 0.45));
            return base * uBandScale * areaBoost * protectionScale;
        }

        void main() {
            vec4 coarse = texture(uCoarseTexture, vTexCoord);
            vec4 detail = texture(uDetailTexture, vTexCoord);
            vec2 keptChroma = softThreshold(detail.yz, bandThreshold(coarse.x, coarse.yz));
            fragColor = vec4(coarse.x, coarse.yz + keptChroma, coarse.a);
        }
    """.trimIndent()

    val COMPOSE_RGB = """
        #version 300 es
        $COMMON
        uniform sampler2D uOriginalTexture;
        uniform sampler2D uChromaTexture;
        uniform vec2 uNoiseModel;
        uniform float uH;

        void main() {
            vec4 original = texture(uOriginalTexture, vTexCoord);
            vec3 originalYcc = rgb2ycbcr(original.rgb);
            vec2 originalChroma = originalYcc.yz;
            vec2 filteredChroma = texture(uChromaTexture, vTexCoord).yz;

            float noiseSigma = noiseSigmaForLuma(uNoiseModel, originalYcc.x);
            float strength01 = clamp(uH / 6.0, 0.0, 1.0);
            float localH = max(uH * noiseSigma, 1e-5);
            float shadowWeight = (1.0 - smoothstep(0.035, 0.24, originalYcc.x)) *
                                 smoothstep(0.004, 0.055, localH);
            float chromaLen = length(originalChroma);
            float stableChroma = chromaProtection(chromaLen, noiseSigma);
            float maxDelta = noiseSigma * mix(2.4, 7.5, strength01) * mix(1.0, 1.65, shadowWeight) +
                             chromaLen * mix(0.10, 0.30, strength01);
            vec2 delta = filteredChroma - originalChroma;
            filteredChroma = originalChroma + delta * min(1.0, maxDelta / max(length(delta), 1e-6));

            float filteredLen = length(filteredChroma);
            float minProtectedLen = chromaLen * (1.0 - mix(0.08, 0.22, strength01) * stableChroma);
            if (stableChroma > 0.001 && filteredLen < minProtectedLen) {
                filteredChroma *= minProtectedLen / max(filteredLen, 1e-6);
            }

            vec3 rgb = ycbcr2rgb(vec3(originalYcc.x, filteredChroma));
            fragColor = vec4(rgb, original.a);
        }
    """.trimIndent()
}
