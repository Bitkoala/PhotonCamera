package com.hinnka.mycamera.model

import androidx.annotation.Keep
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.hinnka.mycamera.camera.AspectRatio

/**
 * 拍摄预设组合（用户自定义档位配置）
 */
@Keep
data class CameraPreset(
    val id: String,
    val name: String,
    val lutId: String?,
    val colorRecipe: ColorRecipeParams, // 仅保留可烘焙色彩微调
    val effects: EffectParams,          // 独立物理效果
    val aspectRatio: String = AspectRatio.RATIO_4_3.name,
    val useRaw: Boolean = false,
    val useMFNR: Boolean = false,
    val useMFSR: Boolean = false,
    val frameId: String? = null,
    // Quick RAW 功能
    val rawDcpId: String? = null,
    val rawSpectralFilmEnabled: Boolean = false,
    val rawSpectralFilmStock: String? = null,
    val rawSpectralFilmPrint: String? = null,
    val rawDROMode: String = "OFF",
    // 基准色彩校正
    val jpgBaselineLutId: String? = null,
    val rawBaselineLutId: String? = null,
    val phantomBaselineLutId: String? = null,
    // 是否为内置预设
    val isBuiltIn: Boolean = false
) {
    fun toJson(): String = gson.toJson(this)

    companion object {
        private val gson = Gson()

        // 场景默认预设
        val BUILT_IN_PRESETS = listOf(
            CameraPreset(
                id = "builtin_default",
                name = "builtin_default",
                lutId = null,
                colorRecipe = ColorRecipeParams.DEFAULT,
                effects = EffectParams.DEFAULT,
                frameId = null,
                useRaw = false,
                useMFNR = false,
                rawDcpId = null,
                rawSpectralFilmEnabled = false,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_portrait",
                name = "builtin_portrait",
                lutId = "standard",
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    exposure = 0.2f,
                    contrast = 0.95f,
                    fade = 0.05f
                ),
                effects = EffectParams.DEFAULT,
                frameId = "polaroid",
                useRaw = false,
                useMFNR = true,
                rawDcpId = null,
                rawSpectralFilmEnabled = false,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_leica_m9_moment",
                name = "builtin_leica_m9_moment",
                lutId = "leica_m9",
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    exposure = -0.3f,
                    contrast = 1.05f,
                    saturation = 0.88f,
                    color = 0.12f,
                    redChroma = 0.08f,
                    redLightness = -0.03f,
                    orangeChroma = 0.06f,
                    yellowHue = -0.03f,
                    yellowChroma = 0.05f,
                    greenChroma = -0.04f,
                    cyanHue = -0.04f,
                    blueChroma = 0.06f,
                    primaryRedSaturation = 0.06f,
                    primaryBlueSaturation = 0.04f,
                ),
                effects = EffectParams.DEFAULT.copy(
                    vignette = -0.2f,
                ),
                frameId = "leica",
                useRaw = true,
                useMFNR = false,
                rawDcpId = null,
                rawSpectralFilmEnabled = false,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_cinematic",
                name = "builtin_cinematic",
                lutId = null,
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    shadows = -0.08f,
                    highlights = 0.05f
                ),
                effects = EffectParams.DEFAULT,
                frameId = "xpan",
                aspectRatio = AspectRatio.XPAN.name,
                useRaw = true,
                rawDcpId = null,
                rawSpectralFilmEnabled = true,
                rawSpectralFilmStock = "kodak_vision3_250d",
                rawSpectralFilmPrint = "kodak_2383",
                rawDROMode = "OFF",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_classic_film",
                name = "builtin_classic_film",
                lutId = null,
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    temperature = 0.08f,
                    contrast = 1.05f
                ),
                effects = EffectParams.DEFAULT.copy(
                    vignette = -0.25f,
                    filmGrain = 0.25f,
                    hdf = 0.25f,
                    halation = 0.25f,
                ),
                frameId = "time",
                useRaw = true,
                rawDcpId = null,
                rawSpectralFilmEnabled = true,
                rawSpectralFilmStock = "kodak_gold_200",
                rawSpectralFilmPrint = "kodak_2383",
                rawDROMode = "OFF",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_monochrome",
                name = "builtin_monochrome",
                lutId = "monochrome",
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    contrast = 1.2f
                ),
                effects = EffectParams.DEFAULT,
                frameId = "black_border",
                useRaw = false,
                rawDcpId = null,
                rawSpectralFilmEnabled = false,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
        )

        fun fromJson(json: String): CameraPreset? {
            return try {
                gson.fromJson(json, CameraPreset::class.java)
            } catch (e: Exception) {
                null
            }
        }

        fun listFromJson(json: String): List<CameraPreset> {
            if (json.isEmpty()) return emptyList()
            return try {
                val type = object : TypeToken<List<CameraPreset>>() {}.type
                gson.fromJson(json, type) ?: emptyList()
            } catch (e: Exception) {
                emptyList()
            }
        }

        fun listToJson(list: List<CameraPreset>): String = gson.toJson(list)
    }
}
