package com.hinnka.mycamera.raw

/**
 * GPU RCD 解马赛克计算着色器库 (1:1 高保真直接移植版)
 *
 * 逐行平移自 darktable 的 demosaic_rcd.cl 核心算子
 * 使用 SSBO (Shader Storage Buffer Object) 在全局显存中共享像素差分与梯度状态，
 * 彻底避免手写算法简化引起的图像拉链与边缘杂色！
 */
object RcdShaders {

    /**
     * 1. 初始化与归一化片元导入 (rcd_populate.comp)
     */
    val POPULATE = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout (binding = 10) uniform highp usampler2D uRawTexture; // 单通道 R16UI 原始图像，移至 binding = 10 避开 SSBO

        layout(std430, binding = 0) buffer CFA_Buf    { float cfa[]; };
        layout(std430, binding = 1) buffer RGB0_Buf   { float rgb0[]; }; // R
        layout(std430, binding = 2) buffer RGB1_Buf   { float rgb1[]; }; // G
        layout(std430, binding = 3) buffer RGB2_Buf   { float rgb2[]; }; // B

        uniform ivec2 uImageSize;
        uniform int uCfaPattern;
        uniform vec4 uBlackLevel; // R, Gr, Gb, B 或 [0,1,2,3] 四通道黑电平
        uniform float uWhiteLevel;
        uniform vec4 uWhiteBalanceGains; // R, Gr, Gb, B 或 [0,1,2,3] 四通道白平衡增益

        #define RED 0
        #define GREEN 1
        #define BLUE 2

        int getBayerColor(int cfaPattern, int col, int row) {
            int r = row % 2;
            int c = col % 2;
            if (cfaPattern == 0) { // RGGB
                if (r == 0) return (c == 0) ? 0 : 1;
                else return (c == 0) ? 1 : 2;
            } else if (cfaPattern == 1) { // GRBG
                if (r == 0) return (c == 0) ? 1 : 0;
                else return (c == 0) ? 2 : 1;
            } else if (cfaPattern == 2) { // GBRG
                if (r == 0) return (c == 0) ? 1 : 2;
                else return (c == 0) ? 0 : 1;
            } else { // BGGR (3)
                if (r == 0) return (c == 0) ? 2 : 1;
                else return (c == 0) ? 1 : 0;
            }
        }

        int getBlackLevelIndex(int cfaPattern, int col, int row) {
            int r = row % 2;
            int c = col % 2;
            if (cfaPattern == 0) { // RGGB
                if (r == 0) return (c == 0) ? 0 : 1; // R, Gr
                else return (c == 0) ? 2 : 3;        // Gb, B
            } else if (cfaPattern == 1) { // GRBG
                if (r == 0) return (c == 0) ? 1 : 0; // Gr, R
                else return (c == 0) ? 3 : 2;        // B, Gb
            } else if (cfaPattern == 2) { // GBRG
                if (r == 0) return (c == 0) ? 2 : 3; // Gb, B
                else return (c == 0) ? 0 : 1;        // R, Gr
            } else { // BGGR (3)
                if (r == 0) return (c == 0) ? 3 : 2; // B, Gb
                else return (c == 0) ? 1 : 0;        // Gr, R
            }
        }

        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            if (coord.x >= uImageSize.x || coord.y >= uImageSize.y) return;

            int idx = coord.y * uImageSize.x + coord.x;

            uint rawVal = texelFetch(uRawTexture, coord, 0).r;
            float val = float(rawVal);

            int blIdx = getBlackLevelIndex(uCfaPattern, coord.x, coord.y);
            float bl = uBlackLevel[blIdx];
            float wl = max(uWhiteLevel, bl + 1.0);
            val = min(clamp((val - bl) / (wl - bl), 0.0, 1.0) * uWhiteBalanceGains[blIdx], 1.0);

            cfa[idx] = val;

            int color = getBayerColor(uCfaPattern, coord.x, coord.y);
            if (color == RED) {
                rgb0[idx] = val;
                rgb1[idx] = 0.0;
                rgb2[idx] = 0.0;
            } else if (color == GREEN) {
                rgb0[idx] = 0.0;
                rgb1[idx] = val;
                rgb2[idx] = 0.0;
            } else { // BLUE
                rgb0[idx] = 0.0;
                rgb1[idx] = 0.0;
                rgb2[idx] = val;
            }
        }
    """.trimIndent()

    /**
     * 2. 共享内存梯度估计与水平/垂直选择 (rcd_step_1.comp)
     */
    val STEP_1 = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 0) buffer CFA_Buf    { float cfa[]; };
        layout(std430, binding = 4) buffer VH_Dir_Buf { float VH_dir[]; };

        uniform ivec2 uImageSize;

        #define epssq 1e-10f

        shared float sh_buffer[576]; // 24 * 24

        float fsquare(float x) {
            return x * x;
        }

        float rcd_vdiff_local(int offset, int stride) {
            float val = sh_buffer[offset - 3 * stride] - sh_buffer[offset - stride] - sh_buffer[offset + stride] + sh_buffer[offset + 3 * stride] 
                        - 3.0f * (sh_buffer[offset - 2 * stride] + sh_buffer[offset + 2 * stride]) + 6.0f * sh_buffer[offset];
            return fsquare(val);
        }

        float rcd_hdiff_local(int offset) {
            float val = sh_buffer[offset - 3] - sh_buffer[offset - 1] - sh_buffer[offset + 1] + sh_buffer[offset + 3]
                        - 3.0f * (sh_buffer[offset - 2] + sh_buffer[offset + 2]) + 6.0f * sh_buffer[offset];
            return fsquare(val);
        }

        void main() {
            int xlsz = 16;
            int ylsz = 16;
            int xlid = int(gl_LocalInvocationID.x);
            int ylid = int(gl_LocalInvocationID.y);
            int xgid = int(gl_WorkGroupID.x);
            int ygid = int(gl_WorkGroupID.y);
            int l = ylid * xlsz + xlid;
            int lsz = xlsz * ylsz; // 256
            int stride = 24;
            int maxbuf = 576;
            int xul = xgid * xlsz - 2;
            int yul = ygid * ylsz - 2;

            for (int n = 0; n <= maxbuf / lsz; n++) {
                int bufidx = n * lsz + l;
                if (bufidx >= maxbuf) continue;
                int xx = clamp(xul + bufidx % stride, 0, uImageSize.x - 1);
                int yy = clamp(yul + bufidx / stride, 0, uImageSize.y - 1);
                sh_buffer[bufidx] = cfa[yy * uImageSize.x + xx];
            }

            memoryBarrierShared();
            barrier();

            int col = 2 + int(gl_GlobalInvocationID.x);
            int row = 2 + int(gl_GlobalInvocationID.y);
            if (row >= uImageSize.y - 2 || col >= uImageSize.x - 2) return;

            int idx = row * uImageSize.x + col;
            int buf_offset = (ylid + 4) * stride + (xlid + 4);

            float V_Stat = max(epssq, rcd_vdiff_local(buf_offset - stride, stride)
                                      + rcd_vdiff_local(buf_offset, stride)
                                      + rcd_vdiff_local(buf_offset + stride, stride));
            float H_Stat = max(epssq, rcd_hdiff_local(buf_offset - 1)
                                      + rcd_hdiff_local(buf_offset)
                                      + rcd_hdiff_local(buf_offset + 1));
            VH_dir[idx] = V_Stat / (V_Stat + H_Stat);
        }
    """.trimIndent()

    /**
     * 3. 邻域低通滤波 LPF (rcd_step_2.comp)
     */
    val STEP_2 = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 0) buffer CFA_Buf { float cfa[]; };
        layout(std430, binding = 5) buffer LPF_Buf { float lpf[]; };

        uniform ivec2 uImageSize;
        uniform int uCfaPattern;

        int getBayerColor(int cfaPattern, int col, int row) {
            int r = row % 2;
            int c = col % 2;
            if (cfaPattern == 0) { // RGGB
                if (r == 0) return (c == 0) ? 0 : 1;
                else return (c == 0) ? 1 : 2;
            } else if (cfaPattern == 1) { // GRBG
                if (r == 0) return (c == 0) ? 1 : 0;
                else return (c == 0) ? 2 : 1;
            } else if (cfaPattern == 2) { // GBRG
                if (r == 0) return (c == 0) ? 1 : 2;
                else return (c == 0) ? 0 : 1;
            } else { // BGGR (3)
                if (r == 0) return (c == 0) ? 2 : 1;
                else return (c == 0) ? 1 : 0;
            }
        }

        void main() {
            int row = 2 + int(gl_GlobalInvocationID.y);
            int col = 2 + (getBayerColor(uCfaPattern, 0, row) & 1) + 2 * int(gl_GlobalInvocationID.x);
            if (col >= uImageSize.x - 2 || row >= uImageSize.y - 2) return;

            int idx = row * uImageSize.x + col;
            int w = uImageSize.x;

            lpf[idx / 2] = cfa[idx]
               + 0.5f * (cfa[idx - w] + cfa[idx + w] + cfa[idx - 1] + cfa[idx + 1])
               + 0.25f * (cfa[idx - w - 1] + cfa[idx - w + 1] + cfa[idx + w - 1] + cfa[idx + w + 1]);
        }
    """.trimIndent()

    /**
     * 4. 绿色通道在红蓝 CFA 位置插值 (rcd_step_3.comp)
     */
    val STEP_3 = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 0) buffer CFA_Buf    { float cfa[]; };
        layout(std430, binding = 2) buffer RGB1_Buf   { float rgb1[]; };
        layout(std430, binding = 4) buffer VH_Dir_Buf { float VH_dir[]; };
        layout(std430, binding = 5) buffer LPF_Buf    { float lpf[]; };

        uniform ivec2 uImageSize;
        uniform int uCfaPattern;

        #define eps 1e-5f

        int getBayerColor(int cfaPattern, int col, int row) {
            int r = row % 2;
            int c = col % 2;
            if (cfaPattern == 0) { // RGGB
                if (r == 0) return (c == 0) ? 0 : 1;
                else return (c == 0) ? 1 : 2;
            } else if (cfaPattern == 1) { // GRBG
                if (r == 0) return (c == 0) ? 1 : 0;
                else return (c == 0) ? 2 : 1;
            } else if (cfaPattern == 2) { // GBRG
                if (r == 0) return (c == 0) ? 1 : 2;
                else return (c == 0) ? 0 : 1;
            } else { // BGGR (3)
                if (r == 0) return (c == 0) ? 2 : 1;
                else return (c == 0) ? 1 : 0;
            }
        }

        void main() {
            int row = 4 + int(gl_GlobalInvocationID.y);
            int col = 4 + (getBayerColor(uCfaPattern, 0, row) & 1) + 2 * int(gl_GlobalInvocationID.x);
            if (col >= uImageSize.x - 5 || row >= uImageSize.y - 5) return;

            int w = uImageSize.x;
            int idx = row * w + col;
            int lidx = idx / 2;
            int w2 = 2 * w;
            int w3 = 3 * w;
            int w4 = 4 * w;

            float VH_Central_Value   = VH_dir[idx];
            float VH_Neighbourhood_Value = 0.25f * (VH_dir[idx - w - 1] + VH_dir[idx - w + 1] + VH_dir[idx + w - 1] + VH_dir[idx + w + 1]);
            float VH_Disc = (abs(0.5f - VH_Central_Value) < abs(0.5f - VH_Neighbourhood_Value)) ? VH_Neighbourhood_Value : VH_Central_Value;

            float cfai = cfa[idx];
            float N_Grad = eps + abs(cfa[idx - w] - cfa[idx + w]) + abs(cfai - cfa[idx - w2]) + abs(cfa[idx - w] - cfa[idx - w3]) + abs(cfa[idx - w2] - cfa[idx - w4]);
            float S_Grad = eps + abs(cfa[idx + w] - cfa[idx - w]) + abs(cfai - cfa[idx + w2]) + abs(cfa[idx + w] - cfa[idx + w3]) + abs(cfa[idx + w2] - cfa[idx + w4]);
            float W_Grad = eps + abs(cfa[idx - 1] - cfa[idx + 1]) + abs(cfai - cfa[idx - 2]) + abs(cfa[idx - 1] - cfa[idx - 3]) + abs(cfa[idx - 2] - cfa[idx - 4]);
            float E_Grad = eps + abs(cfa[idx + 1] - cfa[idx - 1]) + abs(cfai - cfa[idx + 2]) + abs(cfa[idx + 1] - cfa[idx + 3]) + abs(cfa[idx + 2] - cfa[idx + 4]);

            float lfpi = lpf[lidx];
            float N_Est = cfa[idx - w] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx - w]);
            float S_Est = cfa[idx + w] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx + w]);
            float W_Est = cfa[idx - 1] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx - 1]);
            float E_Est = cfa[idx + 1] * (lfpi + lfpi) / (eps + lfpi + lpf[lidx + 1]);

            float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
            float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);

            rgb1[idx] = mix(V_Est, H_Est, clamp(VH_Disc, 0.0, 1.0));
        }
    """.trimIndent()

    /**
     * 5. 对角线差分计算 (rcd_step_4_0.comp)
     */
    val STEP_4_0 = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 0) buffer CFA_Buf    { float cfa[]; };
        layout(std430, binding = 6) buffer P_Diff_Buf { float p_diff[]; };
        layout(std430, binding = 7) buffer Q_Diff_Buf { float q_diff[]; };

        uniform ivec2 uImageSize;

        float fsquare(float x) {
            return x * x;
        }

        void main() {
            int row = 3 + int(gl_GlobalInvocationID.y);
            int col = 3 + 2 * int(gl_GlobalInvocationID.x);
            if (col >= uImageSize.x - 4 || row >= uImageSize.y - 4) return;

            int w = uImageSize.x;
            int idx = row * w + col;
            int idx2 = idx / 2;
            int w2 = 2 * w;
            int w3 = 3 * w;

            p_diff[idx2] = fsquare((cfa[idx - w3 - 3] - cfa[idx - w - 1] - cfa[idx + w + 1] + cfa[idx + w3 + 3]) - 3.0f * (cfa[idx - w2 - 2] + cfa[idx + w2 + 2]) + 6.0f * cfa[idx]);
            q_diff[idx2] = fsquare((cfa[idx - w3 + 3] - cfa[idx - w + 1] - cfa[idx + w - 1] + cfa[idx + w3 - 3]) - 3.0f * (cfa[idx - w2 + 2] + cfa[idx + w2 - 2]) + 6.0f * cfa[idx]);
        }
    """.trimIndent()

    /**
     * 6. 对角线方向选择强弱度 (rcd_step_4_1.comp)
     */
    val STEP_4_1 = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 6) buffer P_Diff_Buf { float p_diff[]; };
        layout(std430, binding = 7) buffer Q_Diff_Buf { float q_diff[]; };
        layout(std430, binding = 8) buffer PQ_Dir_Buf { float PQ_dir[]; };

        uniform ivec2 uImageSize;
        uniform int uCfaPattern;

        #define epssq 1e-10f

        int getBayerColor(int cfaPattern, int col, int row) {
            int r = row % 2;
            int c = col % 2;
            if (cfaPattern == 0) { // RGGB
                if (r == 0) return (c == 0) ? 0 : 1;
                else return (c == 0) ? 1 : 2;
            } else if (cfaPattern == 1) { // GRBG
                if (r == 0) return (c == 0) ? 1 : 0;
                else return (c == 0) ? 2 : 1;
            } else if (cfaPattern == 2) { // GBRG
                if (r == 0) return (c == 0) ? 1 : 2;
                else return (c == 0) ? 0 : 1;
            } else { // BGGR (3)
                if (r == 0) return (c == 0) ? 2 : 1;
                else return (c == 0) ? 1 : 0;
            }
        }

        void main() {
            int row = 2 + int(gl_GlobalInvocationID.y);
            int col = 2 + (getBayerColor(uCfaPattern, 0, row) & 1) + 2 * int(gl_GlobalInvocationID.x);
            if (col >= uImageSize.x - 3 || row >= uImageSize.y - 3) return;

            int w = uImageSize.x;
            int idx = row * w + col;
            int idx2 = idx / 2;
            int idx3 = (idx - w - 1) / 2;
            int idx4 = (idx + w - 1) / 2;

            float P_Stat = max(epssq, p_diff[idx3]     + p_diff[idx2] + p_diff[idx4 + 1]);
            float Q_Stat = max(epssq, q_diff[idx3 + 1] + q_diff[idx2] + q_diff[idx4]);
            PQ_dir[idx2] = P_Stat / (P_Stat + Q_Stat);
        }
    """.trimIndent()

    /**
     * 7. 红蓝通道在红蓝 CFA 位置根据对角线插值 (rcd_step_4_2.comp)
     */
    val STEP_4_2 = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 0) buffer CFA_Buf    { float cfa[]; };
        layout(std430, binding = 1) buffer RGB0_Buf   { float rgb0[]; }; // R
        layout(std430, binding = 2) buffer RGB1_Buf   { float rgb1[]; }; // G
        layout(std430, binding = 3) buffer RGB2_Buf   { float rgb2[]; }; // B
        layout(std430, binding = 8) buffer PQ_Dir_Buf { float PQ_dir[]; };

        uniform ivec2 uImageSize;
        uniform int uCfaPattern;

        #define eps 1e-5f
        #define RED 0
        #define GREEN 1
        #define BLUE 2

        int getBayerColor(int cfaPattern, int col, int row) {
            int r = row % 2;
            int c = col % 2;
            if (cfaPattern == 0) { // RGGB
                if (r == 0) return (c == 0) ? 0 : 1;
                else return (c == 0) ? 1 : 2;
            } else if (cfaPattern == 1) { // GRBG
                if (r == 0) return (c == 0) ? 1 : 0;
                else return (c == 0) ? 2 : 1;
            } else if (cfaPattern == 2) { // GBRG
                if (r == 0) return (c == 0) ? 1 : 2;
                else return (c == 0) ? 0 : 1;
            } else { // BGGR (3)
                if (r == 0) return (c == 0) ? 2 : 1;
                else return (c == 0) ? 1 : 0;
            }
        }

        void main() {
            int row = 4 + int(gl_GlobalInvocationID.y);
            int col = 4 + (getBayerColor(uCfaPattern, 0, row) & 1) + 2 * int(gl_GlobalInvocationID.x);
            if (col >= uImageSize.x - 4 || row >= uImageSize.y - 4) return;

            int w = uImageSize.x;
            int idx = row * w + col;
            int pqidx = idx / 2;
            int pqidx2 = (idx - w - 1) / 2;
            int pqidx3 = (idx + w - 1) / 2;
            int w2 = 2 * w;
            int w3 = 3 * w;

            int targetColor = 2 - getBayerColor(uCfaPattern, col, row);

            float PQ_Central_Value   = PQ_dir[pqidx];
            float PQ_Neighbourhood_Value = 0.25f * (PQ_dir[pqidx2] + PQ_dir[pqidx2 + 1] + PQ_dir[pqidx3] + PQ_dir[pqidx3 + 1]);
            float PQ_Disc = (abs(0.5f - PQ_Central_Value) < abs(0.5f - PQ_Neighbourhood_Value)) ? PQ_Neighbourhood_Value : PQ_Central_Value;

            float PQ_Disc_Clamped = clamp(PQ_Disc, 0.0f, 1.0f);

            if (targetColor == RED) {
                float NW_Grad = eps + abs(rgb0[idx - w - 1] - rgb0[idx + w + 1]) + abs(rgb0[idx - w - 1] - rgb0[idx - w3 - 3]) + abs(rgb1[idx] - rgb1[idx - w2 - 2]);
                float NE_Grad = eps + abs(rgb0[idx - w + 1] - rgb0[idx + w - 1]) + abs(rgb0[idx - w + 1] - rgb0[idx - w3 + 3]) + abs(rgb1[idx] - rgb1[idx - w2 + 2]);
                float SW_Grad = eps + abs(rgb0[idx - w + 1] - rgb0[idx + w - 1]) + abs(rgb0[idx + w - 1] - rgb0[idx + w3 - 3]) + abs(rgb1[idx] - rgb1[idx + w2 - 2]);
                float SE_Grad = eps + abs(rgb0[idx - w - 1] - rgb0[idx + w + 1]) + abs(rgb0[idx + w + 1] - rgb0[idx + w3 + 3]) + abs(rgb1[idx] - rgb1[idx + w2 + 2]);

                float NW_Est = rgb0[idx - w - 1] - rgb1[idx - w - 1];
                float NE_Est = rgb0[idx - w + 1] - rgb1[idx - w + 1];
                float SW_Est = rgb0[idx + w - 1] - rgb1[idx + w - 1];
                float SE_Est = rgb0[idx + w + 1] - rgb1[idx + w + 1];

                float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
                float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);

                rgb0[idx] = rgb1[idx] + mix(P_Est, Q_Est, PQ_Disc_Clamped);
            } else if (targetColor == BLUE) {
                float NW_Grad = eps + abs(rgb2[idx - w - 1] - rgb2[idx + w + 1]) + abs(rgb2[idx - w - 1] - rgb2[idx - w3 - 3]) + abs(rgb1[idx] - rgb1[idx - w2 - 2]);
                float NE_Grad = eps + abs(rgb2[idx - w + 1] - rgb2[idx + w - 1]) + abs(rgb2[idx - w + 1] - rgb2[idx - w3 + 3]) + abs(rgb1[idx] - rgb1[idx - w2 + 2]);
                float SW_Grad = eps + abs(rgb2[idx - w + 1] - rgb2[idx + w - 1]) + abs(rgb2[idx + w - 1] - rgb2[idx + w3 - 3]) + abs(rgb1[idx] - rgb1[idx + w2 - 2]);
                float SE_Grad = eps + abs(rgb2[idx - w - 1] - rgb2[idx + w + 1]) + abs(rgb2[idx + w + 1] - rgb2[idx + w3 + 3]) + abs(rgb1[idx] - rgb1[idx + w2 + 2]);

                float NW_Est = rgb2[idx - w - 1] - rgb1[idx - w - 1];
                float NE_Est = rgb2[idx - w + 1] - rgb1[idx - w + 1];
                float SW_Est = rgb2[idx + w - 1] - rgb1[idx + w - 1];
                float SE_Est = rgb2[idx + w + 1] - rgb1[idx + w + 1];

                float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
                float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);

                rgb2[idx] = rgb1[idx] + mix(P_Est, Q_Est, PQ_Disc_Clamped);
            }
        }
    """.trimIndent()

    /**
     * 8. 红蓝通道在 G CFA 位置插值 (rcd_step_4_3.comp)
     */
    val STEP_4_3 = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 0) buffer CFA_Buf    { float cfa[]; };
        layout(std430, binding = 1) buffer RGB0_Buf   { float rgb0[]; }; // R
        layout(std430, binding = 2) buffer RGB1_Buf   { float rgb1[]; }; // G
        layout(std430, binding = 3) buffer RGB2_Buf   { float rgb2[]; }; // B
        layout(std430, binding = 4) buffer VH_Dir_Buf { float VH_dir[]; };

        uniform ivec2 uImageSize;
        uniform int uCfaPattern;

        #define eps 1e-5f

        int getBayerColor(int cfaPattern, int col, int row) {
            int r = row % 2;
            int c = col % 2;
            if (cfaPattern == 0) { // RGGB
                if (r == 0) return (c == 0) ? 0 : 1;
                else return (c == 0) ? 1 : 2;
            } else if (cfaPattern == 1) { // GRBG
                if (r == 0) return (c == 0) ? 1 : 0;
                else return (c == 0) ? 2 : 1;
            } else if (cfaPattern == 2) { // GBRG
                if (r == 0) return (c == 0) ? 1 : 2;
                else return (c == 0) ? 0 : 1;
            } else { // BGGR (3)
                if (r == 0) return (c == 0) ? 2 : 1;
                else return (c == 0) ? 1 : 0;
            }
        }

        void main() {
            int row = 4 + int(gl_GlobalInvocationID.y);
            int col = 4 + (getBayerColor(uCfaPattern, 1, row) & 1) + 2 * int(gl_GlobalInvocationID.x);
            if (col >= uImageSize.x - 4 || row >= uImageSize.y - 4) return;

            int w = uImageSize.x;
            int idx = row * w + col;
            int w2 = 2 * w;
            int w3 = 3 * w;

            float VH_Central_Value   = VH_dir[idx];
            float VH_Neighbourhood_Value = 0.25f * (VH_dir[idx - w - 1] + VH_dir[idx - w + 1] + VH_dir[idx + w - 1] + VH_dir[idx + w + 1]);
            float VH_Disc = (abs(0.5f - VH_Central_Value) < abs(0.5f - VH_Neighbourhood_Value)) ? VH_Neighbourhood_Value : VH_Central_Value;

            float VH_Disc_Clamped = clamp(VH_Disc, 0.0f, 1.0f);

            float rgbi1 = rgb1[idx];
            float N1 = eps + abs(rgbi1 - rgb1[idx - w2]);
            float S1 = eps + abs(rgbi1 - rgb1[idx + w2]);
            float W1 = eps + abs(rgbi1 - rgb1[idx - 2]);
            float E1 = eps + abs(rgbi1 - rgb1[idx + 2]);

            float rgb1mw1 = rgb1[idx - w];
            float rgb1pw1 = rgb1[idx + w];
            float rgb1m1 =  rgb1[idx - 1];
            float rgb1p1 =  rgb1[idx + 1];

            // 1. 红色通道插值
            {
                float SNabs = abs(rgb0[idx - w] - rgb0[idx + w]);
                float EWabs = abs(rgb0[idx - 1] - rgb0[idx + 1]);

                float N_Grad = N1 + SNabs + abs(rgb0[idx - w] - rgb0[idx - w3]);
                float S_Grad = S1 + SNabs + abs(rgb0[idx + w] - rgb0[idx + w3]);
                float W_Grad = W1 + EWabs + abs(rgb0[idx - 1] - rgb0[idx - 3]);
                float E_Grad = E1 + EWabs + abs(rgb0[idx + 1] - rgb0[idx + 3]);

                float N_Est = rgb0[idx - w] - rgb1mw1;
                float S_Est = rgb0[idx + w] - rgb1pw1;
                float W_Est = rgb0[idx - 1] - rgb1m1;
                float E_Est = rgb0[idx + 1] - rgb1p1;

                float V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad);
                float H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad);

                rgb0[idx] = rgb1[idx] + mix(V_Est, H_Est, VH_Disc_Clamped);
            }

            // 2. 蓝色通道插值
            {
                float SNabs = abs(rgb2[idx - w] - rgb2[idx + w]);
                float EWabs = abs(rgb2[idx - 1] - rgb2[idx + 1]);

                float N_Grad = N1 + SNabs + abs(rgb2[idx - w] - rgb2[idx - w3]);
                float S_Grad = S1 + SNabs + abs(rgb2[idx + w] - rgb2[idx + w3]);
                float W_Grad = W1 + EWabs + abs(rgb2[idx - 1] - rgb2[idx - 3]);
                float E_Grad = E1 + EWabs + abs(rgb2[idx + 1] - rgb2[idx + 3]);

                float N_Est = rgb2[idx - w] - rgb1mw1;
                float S_Est = rgb2[idx + w] - rgb1pw1;
                float W_Est = rgb2[idx - 1] - rgb1m1;
                float E_Est = rgb2[idx + 1] - rgb1p1;

                float V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad);
                float H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad);

                rgb2[idx] = rgb1[idx] + mix(V_Est, H_Est, VH_Disc_Clamped);
            }
        }
    """.trimIndent()

    /**
     * 9. 合并 RGB 重建结果写出到 RGBA16F 纹理 (rcd_write_output.comp)
     */
    val WRITE_OUTPUT = """
        #version 310 es
        precision highp float;
        precision highp int;
        layout (local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 1) buffer RGB0_Buf   { float rgb0[]; }; // R
        layout(std430, binding = 2) buffer RGB1_Buf   { float rgb1[]; }; // G
        layout(std430, binding = 3) buffer RGB2_Buf   { float rgb2[]; }; // B

        layout (rgba16f, binding = 11) writeonly uniform highp image2D uOutputImage;

        uniform ivec2 uImageSize;
        uniform int uBorder;

        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            if (coord.x >= uImageSize.x || coord.y >= uImageSize.y) return;

            vec4 color = vec4(0.0, 0.0, 0.0, 1.0);

            if (coord.x >= uBorder && coord.x < uImageSize.x - uBorder &&
                coord.y >= uBorder && coord.y < uImageSize.y - uBorder) {
                int idx = coord.y * uImageSize.x + coord.x;
                color.r = max(0.0f, rgb0[idx]);
                color.g = max(0.0f, rgb1[idx]);
                color.b = max(0.0f, rgb2[idx]);
            }

            imageStore(uOutputImage, coord, color);
        }
    """.trimIndent()
}
