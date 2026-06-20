package com.hinnka.mycamera.raw

object RawHdrLocalToneShaders {
    val BASE_GRID_COMPUTE = """
        #version 310 es
        precision highp float;
        precision highp int;

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

        uniform sampler2D uInputTexture;
        uniform ivec2 uImageSize;
        uniform ivec2 uGridSize;
        uniform float uExposureScale;

        layout(rgba16f, binding = 0) writeonly uniform highp image2D uBaseGrid;

        const float LUMA_FLOOR = 0.0000152588;
        const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

        float safeLog2(float value) {
            return log(max(value, LUMA_FLOOR)) / log(2.0);
        }

        float gridLogGuide(vec2 gridPosition) {
            vec2 uv = (gridPosition + vec2(0.5)) / vec2(uGridSize);
            uv = clamp(uv, vec2(0.0), vec2(1.0));
            vec3 rgb = max(texture(uInputTexture, uv).rgb * uExposureScale, vec3(0.0));
            float luma = dot(rgb, LUMA_WEIGHTS);
            float peak = max(rgb.r, max(rgb.g, rgb.b));
            return safeLog2(max(0.70 * luma + 0.30 * peak, LUMA_FLOOR));
        }

        void main() {
            ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
            if (tile.x >= uGridSize.x || tile.y >= uGridSize.y) {
                return;
            }

            vec2 tilePos = vec2(tile);
            float center = gridLogGuide(tilePos);
            float weightedSum = 0.0;
            float weightSum = 0.0;

            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    vec2 offset = vec2(float(x), float(y));
                    vec2 samplePos = clamp(
                        tilePos + offset,
                        vec2(0.0),
                        vec2(uGridSize) - vec2(1.0)
                    );
                    float sampleLog = gridLogGuide(samplePos);
                    float spatialWeight = exp(-dot(offset, offset) * 0.18);
                    float rangeDelta = min(abs(sampleLog - center), 6.0);
                    float rangeWeight = exp(-rangeDelta * rangeDelta * 0.18);
                    float weight = spatialWeight * rangeWeight;
                    weightedSum += sampleLog * weight;
                    weightSum += weight;
                }
            }

            float localBase = weightSum > 0.0 ? weightedSum / weightSum : center;
            imageStore(uBaseGrid, tile, vec4(localBase, center, weightSum, 1.0));
        }
    """.trimIndent()

    val CURVE_TABLE_COMPUTE = """
        #version 310 es
        precision highp float;
        precision highp int;

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

        uniform sampler2D uBaseGrid;
        uniform ivec2 uGridSize;
        uniform int uBinCount;
        uniform float uLogMin;
        uniform float uLogMax;
        uniform float uTargetMidLog;
        uniform float uTargetLowLog;
        uniform float uTargetHighLog;
        uniform float uBaseCompression;
        uniform float uShadowCompressionBias;
        uniform float uDetailScale;
        uniform float uMaxLiftEv;
        uniform float uMaxPullEv;

        layout(rgba16f, binding = 0) writeonly uniform highp image2D uGainTable;

        float smoothStepRaw(float edge0, float edge1, float x) {
            float width = edge1 - edge0;
            if (abs(width) < 0.000001) {
                return x >= edge1 ? 1.0 : 0.0;
            }
            float t = clamp((x - edge0) / width, 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        }

        float compressBaseLog(float baseLog) {
            float delta = baseLog - uTargetMidLog;
            float highScale = uBaseCompression;
            float lowScale = mix(1.0, uBaseCompression, uShadowCompressionBias);
            float compressed = uTargetMidLog + delta * (delta >= 0.0 ? highScale : lowScale);

            float highExcess = max(compressed - uTargetHighLog, 0.0);
            compressed = compressed - highExcess * (1.0 - highScale);

            float lowExcess = max(uTargetLowLog - compressed, 0.0);
            compressed = compressed + lowExcess * (1.0 - lowScale);
            return compressed;
        }

        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            int tableWidth = uGridSize.x * uBinCount;
            if (coord.x >= tableWidth || coord.y >= uGridSize.y) {
                return;
            }

            int tileX = coord.x / uBinCount;
            int bin = coord.x - tileX * uBinCount;
            ivec2 tile = ivec2(tileX, coord.y);
            float localBase = texelFetch(uBaseGrid, tile, 0).r;
            float binT = (float(bin) + 0.5) / float(uBinCount);
            float inputLog = mix(uLogMin, uLogMax, binT);

            float detail = clamp(inputLog - localBase, -5.0, 5.0);
            float detailWeight = smoothStepRaw(0.0, 3.5, abs(detail));
            float effectiveDetailScale = mix(1.0, uDetailScale, detailWeight);
            float targetLog = compressBaseLog(localBase) + detail * effectiveDetailScale;
            float gainEv = clamp(targetLog - inputLog, -uMaxPullEv, uMaxLiftEv);

            float shadowGuard = smoothStepRaw(uLogMin + 0.75, uTargetLowLog + 0.75, inputLog);
            if (gainEv > 0.0) {
                gainEv *= shadowGuard;
            }

            imageStore(uGainTable, coord, vec4(gainEv, localBase, inputLog + gainEv, 1.0));
        }
    """.trimIndent()

    val APPLY_FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp int;

        in vec2 vTexCoord;
        out vec4 fragColor;

        uniform sampler2D uInputTexture;
        uniform sampler2D uGainTable;
        uniform ivec2 uGridSize;
        uniform int uBinCount;
        uniform float uExposureScale;
        uniform float uLogMin;
        uniform float uLogMax;
        uniform float uStrength;

        const float LUMA_FLOOR = 0.0000152588;
        const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

        float safeLog2(float value) {
            return log(max(value, LUMA_FLOOR)) / log(2.0);
        }

        float fetchGain(int tileX, int tileY, int bin) {
            tileX = clamp(tileX, 0, uGridSize.x - 1);
            tileY = clamp(tileY, 0, uGridSize.y - 1);
            bin = clamp(bin, 0, uBinCount - 1);
            return texelFetch(uGainTable, ivec2(tileX * uBinCount + bin, tileY), 0).r;
        }

        float interpolateGain(vec2 uv, float guideLog) {
            vec2 gridPos = clamp(
                uv * vec2(uGridSize) - vec2(0.5),
                vec2(0.0),
                vec2(uGridSize) - vec2(1.0)
            );
            ivec2 tile0 = ivec2(floor(gridPos));
            ivec2 tile1 = min(tile0 + ivec2(1), uGridSize - ivec2(1));
            vec2 gridF = gridPos - vec2(tile0);

            float binPos = clamp(
                (guideLog - uLogMin) / max(uLogMax - uLogMin, 0.000001) * float(uBinCount) - 0.5,
                0.0,
                float(uBinCount - 1)
            );
            int bin0 = int(floor(binPos));
            int bin1 = min(bin0 + 1, uBinCount - 1);
            float binF = binPos - float(bin0);

            float g00b0 = fetchGain(tile0.x, tile0.y, bin0);
            float g10b0 = fetchGain(tile1.x, tile0.y, bin0);
            float g01b0 = fetchGain(tile0.x, tile1.y, bin0);
            float g11b0 = fetchGain(tile1.x, tile1.y, bin0);
            float gainB0 = mix(mix(g00b0, g10b0, gridF.x), mix(g01b0, g11b0, gridF.x), gridF.y);

            float g00b1 = fetchGain(tile0.x, tile0.y, bin1);
            float g10b1 = fetchGain(tile1.x, tile0.y, bin1);
            float g01b1 = fetchGain(tile0.x, tile1.y, bin1);
            float g11b1 = fetchGain(tile1.x, tile1.y, bin1);
            float gainB1 = mix(mix(g00b1, g10b1, gridF.x), mix(g01b1, g11b1, gridF.x), gridF.y);

            return mix(gainB0, gainB1, binF);
        }

        void main() {
            vec3 sourceRgb = texture(uInputTexture, vTexCoord).rgb;
            vec3 displayRgb = max(sourceRgb * uExposureScale, vec3(0.0));
            float luma = dot(displayRgb, LUMA_WEIGHTS);
            float peak = max(displayRgb.r, max(displayRgb.g, displayRgb.b));
            float guideLog = safeLog2(max(0.70 * luma + 0.30 * peak, LUMA_FLOOR));

            float gainEv = interpolateGain(vTexCoord, guideLog) * clamp(uStrength, 0.0, 1.0);
            vec3 result = max(sourceRgb * exp2(gainEv), vec3(0.0));
            fragColor = vec4(result, 1.0);
        }
    """.trimIndent()
}
