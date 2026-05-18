package com.hinnka.mycamera.raw

/**
 * Mobile-optimised BM3D (Block-Matching 3D Filtering) denoising shaders.
 *
 * Replaces [NLMShaders] with a two-step BM3D approximation that trades slightly
 * higher ALU/texture cost for significantly better detail retention and noise
 * reduction, particularly in high-ISO shots.
 *
 * Pipeline (same FBO bindings as NLM):
 *   PASS0_CHROMA_DENOISE  →  gfChromaFboId
 *   PASS1_BASIC_ESTIMATE  →  gfFboId[0]   (RGB = basic estimate, A = match fraction)
 *   PASS2_WIENER          →  gfFboId[1]   (final Wiener-filtered result)
 *
 * Uniform interface (caller side):
 *   Pass 0  uInputTexture=source,              uH=chromaH, uNoiseModel=(S,O)
 *   Pass 1  uInputTexture=chromaTex,           uH=h,       uNoiseModel=(S,O)
 *   Pass 2  uInputTexture=chromaTex (noisy),
 *           uBasicTexture=gfTexId[0] (basic),  uH=h,       uNoiseModel=(S,O)
 *
 * Performance budget (5×5 search, 3×3 luma patch):
 *   Pass 1 ≈ 234 tex reads/px   Pass 2 ≈ 260 tex reads/px
 *   Total   ≈ 520 reads/px  (vs ≈ 122 for NLM — 4× cost, substantially higher quality)
 */
object BM3DShaders {

    // ─── Pass 0 : Chroma Denoise ──────────────────────────────────────────────
    /**
     * Cross-bilateral filter on the chroma channels guided by luma.
     * Runs before luma denoising to supply a cleaner guide image.
     * Wide chroma tolerance + tight luma tolerance prevents halo/bleed.
     */
    val PASS0_CHROMA_DENOISE = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        uniform mat4 uTexMatrix;
        uniform float uH;
        uniform vec2 uNoiseModel;

        vec3 rgb2ycbcr(vec3 rgb) {
            float y = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
            return vec3(y,
                        dot(rgb, vec3(-0.169, -0.331,  0.5  )) + 0.5,
                        dot(rgb, vec3( 0.5,  -0.419, -0.081 )) + 0.5);
        }
        vec3 ycbcr2rgb(vec3 yuv) {
            float cb = yuv.y - 0.5, cr = yuv.z - 0.5;
            return vec3(yuv.x + 1.402  * cr,
                        yuv.x - 0.3441 * cb - 0.7141 * cr,
                        yuv.x + 1.772  * cb);
        }
        float noiseSigmaForLuma(float luma) {
            return sqrt(max(uNoiseModel.x * max(luma, 0.0) + uNoiseModel.y, 1e-10));
        }

        void main() {
            vec3 oriRgb = texture(uInputTexture, vTexCoord).rgb;
            if (uH <= 0.00001) { fragColor = vec4(oriRgb, 1.0); return; }

            vec3  yuv  = rgb2ycbcr(oriRgb);
            vec2  sumUV = vec2(0.0);
            float sumW  = 0.0;

            // Chroma noise is low-frequency → large spatial step scale
            float stepScale   = 6.5;
            float localH      = uH * noiseSigmaForLuma(yuv.x);
            float invChromaH2 = 1.0 / max((localH * 1.5) * (localH * 1.5), 1e-6);
            float invLumaH2   = 1.0 / max((localH * 0.5) * (localH * 0.5), 1e-6);

            for (int x = -2; x <= 2; x++) {
                for (int y = -2; y <= 2; y++) {
                    vec2 ofs  = vec2(float(x), float(y)) * uTexelSize * stepScale;
                    vec3 sYuv = rgb2ycbcr(texture(uInputTexture, vTexCoord + ofs).rgb);

                    float wS = exp(-float(x*x + y*y) * 0.25);
                    float dL = sYuv.x - yuv.x;
                    vec2  dC = sYuv.yz - yuv.yz;
                    float wR = exp(-(dL * dL) * invLumaH2 - dot(dC, dC) * invChromaH2);
                    float w  = wS * wR;
                    if (x == 0 && y == 0) w = max(w, 0.1);
                    sumUV += sYuv.yz * w;
                    sumW  += w;
                }
            }
            yuv.yz = sumUV / sumW;
            fragColor = vec4(ycbcr2rgb(yuv), 1.0);
        }
    """.trimIndent()

    // ─── Pass 1 : BM3D Basic Estimate ────────────────────────────────────────
    /**
     * Hard-threshold block matching in a 5×5 search window with 3×3 luma patches.
     *
     * Matches the BM3D Step-1 philosophy: a candidate block is either in the group
     * (ssd ≤ τ) or out (weight = 0), producing a collaborative average that acts as
     * a spatial hard-threshold in the transform domain.
     *
     * Output alpha = match fraction (0–1), forwarded to Pass 2 for noise estimation.
     */
    val PASS1_BASIC_ESTIMATE = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;
        uniform sampler2D uInputTexture;
        uniform vec2 uTexelSize;
        uniform mat4 uTexMatrix;
        uniform float uH;
        uniform vec2 uNoiseModel;

        // 5×5 search window, 3×3 luma patch
        #define SEARCH_RADIUS 2
        #define PATCH_RADIUS  1

        const vec3 LW = vec3(0.2126, 0.7152, 0.0722);

        vec3 rgb2ycbcr(vec3 rgb) {
            return vec3(dot(rgb, LW),
                        dot(rgb, vec3(-0.169, -0.331,  0.5  )) + 0.5,
                        dot(rgb, vec3( 0.5,  -0.419, -0.081 )) + 0.5);
        }
        vec3 ycbcr2rgb(vec3 yuv) {
            float cb = yuv.y - 0.5, cr = yuv.z - 0.5;
            return vec3(yuv.x + 1.402  * cr,
                        yuv.x - 0.3441 * cb - 0.7141 * cr,
                        yuv.x + 1.772  * cb);
        }
        float softMatchWeight(float ssd, float threshold) {
            float t = max(threshold, 1e-6);
            float soft = exp(-ssd / t);
            float gate = 1.0 - smoothstep(t * 0.35, t * 1.55, ssd);
            return soft * gate;
        }
        float noiseSigmaForLuma(float luma) {
            return sqrt(max(uNoiseModel.x * max(luma, 0.0) + uNoiseModel.y, 1e-10));
        }

        void main() {
            vec3 centerVal = texture(uInputTexture, vTexCoord).rgb;
            if (uH <= 0.00001) { fragColor = vec4(centerVal, 1.0); return; }

            vec2  T = uTexelSize;
            vec3  centerYuv = rgb2ycbcr(centerVal);

            // Pre-fetch the 3×3 center-patch luma values (unrolled to avoid
            // dynamic-array indexing, which can be slow on some mobile GPUs)
            float c00 = dot(texture(uInputTexture, vTexCoord + vec2(-1.0,-1.0)*T).rgb, LW);
            float c01 = dot(texture(uInputTexture, vTexCoord + vec2(-1.0, 0.0)*T).rgb, LW);
            float c02 = dot(texture(uInputTexture, vTexCoord + vec2(-1.0, 1.0)*T).rgb, LW);
            float c10 = dot(texture(uInputTexture, vTexCoord + vec2( 0.0,-1.0)*T).rgb, LW);
            float c11 = dot(centerVal, LW);
            float c12 = dot(texture(uInputTexture, vTexCoord + vec2( 0.0, 1.0)*T).rgb, LW);
            float c20 = dot(texture(uInputTexture, vTexCoord + vec2( 1.0,-1.0)*T).rgb, LW);
            float c21 = dot(texture(uInputTexture, vTexCoord + vec2( 1.0, 0.0)*T).rgb, LW);
            float c22 = dot(texture(uInputTexture, vTexCoord + vec2( 1.0, 1.0)*T).rgb, LW);
            float localH = uH * noiseSigmaForLuma(c11);

            // BM3D hard threshold: τ = factor × σ² × patch_size
            // factor 2.7 is the standard BM3D constant, tuned here for 3×3 patches
            float threshold = localH * localH * 9.0 * 2.7;

            float sumLuma   = 0.0;
            vec2  sumChroma = vec2(0.0);
            float sumWeight = 0.0;

            for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
                for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++) {
                    vec2 nUV  = vTexCoord + vec2(float(dx), float(dy)) * T;
                    // Cache the neighbor center pixel for reuse in SSD + accumulation
                    vec3  nC  = texture(uInputTexture, nUV).rgb;
                    float nL  = dot(nC, LW);

                    // 3×3 luma SSD – center pixel reuses nL from the cached fetch
                    float ssd = 0.0;
                    float d;
                    d = c00 - dot(texture(uInputTexture, nUV + vec2(-1.0,-1.0)*T).rgb, LW); ssd += d*d;
                    d = c01 - dot(texture(uInputTexture, nUV + vec2(-1.0, 0.0)*T).rgb, LW); ssd += d*d;
                    d = c02 - dot(texture(uInputTexture, nUV + vec2(-1.0, 1.0)*T).rgb, LW); ssd += d*d;
                    d = c10 - dot(texture(uInputTexture, nUV + vec2( 0.0,-1.0)*T).rgb, LW); ssd += d*d;
                    d = c11 - nL;                                                             ssd += d*d;
                    d = c12 - dot(texture(uInputTexture, nUV + vec2( 0.0, 1.0)*T).rgb, LW); ssd += d*d;
                    d = c20 - dot(texture(uInputTexture, nUV + vec2( 1.0,-1.0)*T).rgb, LW); ssd += d*d;
                    d = c21 - dot(texture(uInputTexture, nUV + vec2( 1.0, 0.0)*T).rgb, LW); ssd += d*d;
                    d = c22 - dot(texture(uInputTexture, nUV + vec2( 1.0, 1.0)*T).rgb, LW); ssd += d*d;

                    float w = softMatchWeight(ssd, threshold);
                    if (dx == 0 && dy == 0) w = max(w, 1.0);

                    vec3 nYuv = rgb2ycbcr(nC);
                    sumLuma   += nYuv.x * w;
                    sumChroma += nYuv.yz * w;
                    sumWeight += w;
                }
            }

            // Store normalised match count in alpha → Pass 2 uses it to refine noise estimate
            float maxCandidates = float((2*SEARCH_RADIUS+1) * (2*SEARCH_RADIUS+1)); // 25
            float invWeight = 1.0 / max(sumWeight, 1.0);
            float basicLuma = sumLuma * invWeight;

            // Fallback for low-match cases (edges/isolated noise) to break up clusters.
            // Uses a tiny 3x3 guided luma average since we already have the patch data.
            float lwSum = 1.0; float lvalSum = c11;
            float invH2 = 1.0 / max(localH * localH * 2.0, 1e-5);
            float w;
            w = exp(-(c00-c11)*(c00-c11)*invH2); lwSum += w; lvalSum += c00*w;
            w = exp(-(c01-c11)*(c01-c11)*invH2); lwSum += w; lvalSum += c01*w;
            w = exp(-(c02-c11)*(c02-c11)*invH2); lwSum += w; lvalSum += c02*w;
            w = exp(-(c10-c11)*(c10-c11)*invH2); lwSum += w; lvalSum += c10*w;
            w = exp(-(c12-c11)*(c12-c11)*invH2); lwSum += w; lvalSum += c12*w;
            w = exp(-(c20-c11)*(c20-c11)*invH2); lwSum += w; lvalSum += c20*w;
            w = exp(-(c21-c11)*(c21-c11)*invH2); lwSum += w; lvalSum += c21*w;
            w = exp(-(c22-c11)*(c22-c11)*invH2); lwSum += w; lvalSum += c22*w;

            float localGuided = lvalSum / lwSum;
            float edgeFallback = clamp(3.0 - sumWeight, 0.0, 1.0);
            basicLuma = mix(basicLuma, localGuided, edgeFallback * 0.35);

            vec3 outYuv = vec3(basicLuma, mix(centerYuv.yz, sumChroma * invWeight, 0.35));
            fragColor = vec4(clamp(ycbcr2rgb(outYuv), 0.0, 1.0), clamp(sumWeight / maxCandidates, 0.0, 1.0));
        }
    """.trimIndent()

    // ─── Pass 2 : BM3D Wiener Refinement ─────────────────────────────────────
    /**
     * Uses the basic estimate (Pass 1) as a noise-reduced guide for block matching.
     * For each matched group, estimates the local signal variance from the basic
     * estimate and computes the Wiener-optimal blend weight:
     *
     *   w = σ_s² / (σ_s² + σ_n²/N)
     *
     * where σ_s² = signal variance in the group, σ_n² = noise variance (uH²),
     * N = number of matched patches.
     *
     * The final pixel is mixed between the basic estimate (near-noise-free prior)
     * and the average of matched noisy patches (structure-preserving signal estimate).
     *
     * Uniforms:
     *   uInputTexture  – original (chroma-denoised) noisy image
     *   uBasicTexture  – Pass-1 output (RGB = basic estimate, A = match fraction)
     */
    val PASS2_WIENER = """
        #version 300 es
        precision highp float;
        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uInputTexture; // chroma-denoised noisy image
        uniform sampler2D uBasicTexture; // Pass-1 basic estimate  (A = match fraction)
        uniform vec2 uTexelSize;
        uniform mat4 uTexMatrix;
        uniform float uH;
        uniform vec2 uNoiseModel;

        // Same search / patch geometry as Pass 1
        #define SEARCH_RADIUS 2
        #define PATCH_RADIUS  1

        const vec3 LW = vec3(0.2126, 0.7152, 0.0722);

        vec3 rgb2ycbcr(vec3 rgb) {
            return vec3(dot(rgb, LW),
                        dot(rgb, vec3(-0.169, -0.331,  0.5  )) + 0.5,
                        dot(rgb, vec3( 0.5,  -0.419, -0.081 )) + 0.5);
        }
        vec3 ycbcr2rgb(vec3 yuv) {
            float cb = yuv.y - 0.5, cr = yuv.z - 0.5;
            return vec3(yuv.x + 1.402  * cr,
                        yuv.x - 0.3441 * cb - 0.7141 * cr,
                        yuv.x + 1.772  * cb);
        }
        float softMatchWeight(float ssd, float threshold) {
            float t = max(threshold, 1e-6);
            float soft = exp(-ssd / t);
            float gate = 1.0 - smoothstep(t * 0.35, t * 1.55, ssd);
            return soft * gate;
        }
        float noiseSigmaForLuma(float luma) {
            return sqrt(max(uNoiseModel.x * max(luma, 0.0) + uNoiseModel.y, 1e-10));
        }
        float wideGuidedLumaBase(vec2 uv, vec2 texelSize, float centerGuideLuma, float sigma_n2) {
            float sumY = 0.0;
            float sumW = 0.0;
            float stepScale = 3.5;
            float guideSigma2 = max(sigma_n2 * 48.0, 8e-5);

            for (int dx = -2; dx <= 2; dx++) {
                for (int dy = -2; dy <= 2; dy++) {
                    vec2 ofs = vec2(float(dx), float(dy)) * texelSize * stepScale;
                    vec2 suv = uv + ofs;
                    float guideY = dot(texture(uBasicTexture, suv).rgb, LW);
                    float noisyY = dot(texture(uInputTexture, suv).rgb, LW);
                    float dGuide = guideY - centerGuideLuma;
                    float wSpatial = exp(-float(dx * dx + dy * dy) * 0.18);
                    float wRange = exp(-(dGuide * dGuide) / guideSigma2);
                    float w = wSpatial * wRange;
                    if (dx == 0 && dy == 0) w = max(w, 0.25);
                    sumY += noisyY * w;
                    sumW += w;
                }
            }

            return sumY / max(sumW, 1e-5);
        }

        void main() {
            vec3 noisyCenter = texture(uInputTexture, vTexCoord).rgb;
            vec4 basicSample = texture(uBasicTexture,  vTexCoord);
            vec3 basicCenter = basicSample.rgb;
            // alpha from Pass 1: fraction of the 25 candidates that were merged
            // → approximates N_basic, the effective patch count in the basic estimate
            float N_basic    = basicSample.a * 25.0 + 1.0;

            if (uH <= 0.00001) { fragColor = vec4(noisyCenter, 1.0); return; }

            vec2  T = uTexelSize;

            // Pre-fetch 3×3 center-patch luma from BASIC estimate (less noisy guide)
            float c00 = dot(texture(uBasicTexture, vTexCoord + vec2(-1.0,-1.0)*T).rgb, LW);
            float c01 = dot(texture(uBasicTexture, vTexCoord + vec2(-1.0, 0.0)*T).rgb, LW);
            float c02 = dot(texture(uBasicTexture, vTexCoord + vec2(-1.0, 1.0)*T).rgb, LW);
            float c10 = dot(texture(uBasicTexture, vTexCoord + vec2( 0.0,-1.0)*T).rgb, LW);
            float c11 = dot(basicCenter, LW);
            float c12 = dot(texture(uBasicTexture, vTexCoord + vec2( 0.0, 1.0)*T).rgb, LW);
            float c20 = dot(texture(uBasicTexture, vTexCoord + vec2( 1.0,-1.0)*T).rgb, LW);
            float c21 = dot(texture(uBasicTexture, vTexCoord + vec2( 1.0, 0.0)*T).rgb, LW);
            float c22 = dot(texture(uBasicTexture, vTexCoord + vec2( 1.0, 1.0)*T).rgb, LW);

            // Sensor noise model in this pass' linear-light domain:
            // variance = S * luma + O. uH is a dimensionless strength multiplier.
            float localH = uH * noiseSigmaForLuma(c11);
            float sigma_n2 = localH * localH;
            float threshold = sigma_n2 * 9.0 * 2.7; // same τ as Pass 1, adapted by luma

            vec3  basicYuv  = rgb2ycbcr(basicCenter);
            float sumNoisyY = 0.0;
            vec2  sumNoisyC = vec2(0.0);
            float sumWeight = 0.0;
            float sumL      = 0.0;  // luma sum  for variance estimation
            float sumL2     = 0.0;  // luma² sum for variance estimation

            for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++) {
                for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++) {
                    vec2  nUV    = vTexCoord + vec2(float(dx), float(dy)) * T;
                    // Cache basic estimate center for SSD + variance stats
                    vec3  nBasic = texture(uBasicTexture, nUV).rgb;
                    float nL     = dot(nBasic, LW);

                    // Patch SSD over BASIC estimate – cleaner matching than noisy image
                    float ssd = 0.0;
                    float d;
                    d = c00 - dot(texture(uBasicTexture, nUV + vec2(-1.0,-1.0)*T).rgb, LW); ssd += d*d;
                    d = c01 - dot(texture(uBasicTexture, nUV + vec2(-1.0, 0.0)*T).rgb, LW); ssd += d*d;
                    d = c02 - dot(texture(uBasicTexture, nUV + vec2(-1.0, 1.0)*T).rgb, LW); ssd += d*d;
                    d = c10 - dot(texture(uBasicTexture, nUV + vec2( 0.0,-1.0)*T).rgb, LW); ssd += d*d;
                    d = c11 - nL;                                                             ssd += d*d;
                    d = c12 - dot(texture(uBasicTexture, nUV + vec2( 0.0, 1.0)*T).rgb, LW); ssd += d*d;
                    d = c20 - dot(texture(uBasicTexture, nUV + vec2( 1.0,-1.0)*T).rgb, LW); ssd += d*d;
                    d = c21 - dot(texture(uBasicTexture, nUV + vec2( 1.0, 0.0)*T).rgb, LW); ssd += d*d;
                    d = c22 - dot(texture(uBasicTexture, nUV + vec2( 1.0, 1.0)*T).rgb, LW); ssd += d*d;

                    float w = softMatchWeight(ssd, threshold);
                    if (dx == 0 && dy == 0) w = max(w, 1.0);

                    // Accumulate signal-variance statistics from basic estimate
                    sumL      += nL * w;
                    sumL2     += nL * nL * w;

                    // Filter the NOISY image with weights derived from basic-estimate matching
                    vec3 nNoisyYuv = rgb2ycbcr(texture(uInputTexture, nUV).rgb);
                    sumNoisyY += nNoisyYuv.x * w;
                    sumNoisyC += nNoisyYuv.yz * w;
                    sumWeight += w;
                }
            }

            // ── Wiener weight computation ────────────────────────────────────
            // Estimate signal variance from the matched group in the basic estimate.
            // var_total(basic group) ≈ var_signal + σ_n²/N_basic (residual noise)
            float N        = max(sumWeight, 1.0);
            float meanL    = sumL / N;
            float varTotal = sumL2 / N - meanL * meanL;
            // Subtract residual noise in basic estimate (σ_n² / N_basic)
            float varSignal = max(0.0, varTotal - sigma_n2 / N_basic);

            // Wiener weight: w = σ_s² / (σ_s² + σ_n²/N)
            float noiseOfAvg = sigma_n2 / N;
            float wienerW    = varSignal / max(varSignal + noiseOfAvg, 1e-6);

            // Enhanced edge/detail protection:
            // Trust the noisy image more at edges, but only if we have enough matches (N)
            // to actually reduce noise. If N is small, we favor the basic estimate.
            float relN = smoothstep(1.0, 5.0, N);
            float edgeSq = (c21 - c01) * (c21 - c01) + (c12 - c10) * (c12 - c10);
            float detailProtect = smoothstep(sigma_n2 * 20.0, sigma_n2 * 100.0, edgeSq + varSignal);

            float maxTrust = mix(0.65, 0.92, relN);
            wienerW = mix(wienerW * 0.70, min(wienerW, maxTrust), detailProtect);

            vec3 avgNoisyYuv = vec3(sumNoisyY / N, sumNoisyC / N);
            float fullResY = mix(basicYuv.x, avgNoisyYuv.x, clamp(wienerW, 0.0, 1.0));

            // Low-frequency luma residual suppression. This does not replace the
            // signal with a blurred image: it only removes a bounded residual, and
            // only where the basic-estimate guide says the area is flat.
            float lowFreqY = wideGuidedLumaBase(vTexCoord, T, c11, sigma_n2);
            float flatness = 1.0 - smoothstep(
                max(sigma_n2 * 18.0, 2e-5),
                max(sigma_n2 * 120.0, 1.2e-4),
                edgeSq + varSignal
            );
            float matchConfidence = smoothstep(2.5, 8.0, N) * smoothstep(0.08, 0.36, basicSample.a);
            float residualLimit = max(localH * 0.65, 0.003);
            float residualPull = clamp(fullResY - lowFreqY, -residualLimit, residualLimit);
            float lowFreqStrength = 0.35 * flatness * matchConfidence * (1.0 - detailProtect * 0.65);
            float finalY = fullResY - residualPull * clamp(lowFreqStrength, 0.0, 0.35);

            vec3 outYuv = vec3(
                finalY,
                mix(basicYuv.yz, avgNoisyYuv.yz, clamp(wienerW * 0.25, 0.0, 0.45))
            );
            fragColor = vec4(clamp(ycbcr2rgb(outYuv), 0.0, 1.0), 1.0);
        }
    """.trimIndent()
}
