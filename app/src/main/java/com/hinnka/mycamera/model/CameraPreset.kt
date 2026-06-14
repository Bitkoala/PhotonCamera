package com.hinnka.mycamera.model

import androidx.annotation.Keep
import com.google.gson.Gson
import com.google.gson.JsonElement
import com.google.gson.JsonParser
import com.google.gson.reflect.TypeToken
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.raw.RawColorEngine

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
    val useHdrComposition: Boolean = true,
    val useMFSR: Boolean = false,
    val frameId: String? = null,
    // Quick RAW 功能
    val rawDcpId: String? = null,
    val rawColorEngine: String = RawColorEngine.AgX.name,
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
    fun toJson(): String = gson.toJson(withoutLegacyHdf())

    fun withoutLegacyHdf(): CameraPreset {
        return copy(
            colorRecipe = colorRecipe.copy(halation = 0f),
            effects = effects.copy(hdf = 0f)
        )
    }

    companion object {
        private val gson = Gson()

        private fun normalizePresetJson(element: JsonElement): JsonElement {
            if (element.isJsonObject) {
                val obj = element.asJsonObject
                if (!obj.has("useHdrComposition")) {
                    obj.addProperty("useHdrComposition", false)
                }
            }
            return element
        }

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
                rawColorEngine = RawColorEngine.AgX.name,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_portrait",
                name = "builtin_portrait",
                lutId = "standard",
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    exposure = 0.2f,
                ),
                effects = EffectParams.DEFAULT,
                frameId = "polaroid",
                useRaw = false,
                useMFNR = true,
                rawDcpId = null,
                rawColorEngine = RawColorEngine.AgX.name,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_leica_m9_moment",
                name = "builtin_leica_m9_moment",
                lutId = "leica_m9",
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    exposure = -0.3f,
                    saturation = 0.9f,
                    color = 0.12f,
                    primaryRedSaturation = 0.06f,
                    primaryBlueSaturation = 0.04f,
                    masterCurvePoints = floatArrayOf(
                        0.0f, 0.0f, 0.25f, 0.24f, 0.66f, 0.74f, 1.0f, 1.0f
                    )
                ),
                effects = EffectParams.DEFAULT.copy(
                    vignette = -0.2f,
                ),
                frameId = "leica",
                useRaw = true,
                useMFNR = false,
                rawDcpId = null,
                rawColorEngine = RawColorEngine.AgX.name,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
            CameraPreset(
                id = "builtin_cinematic",
                name = "builtin_cinematic",
                lutId = "ricoh_yellow",
                colorRecipe = ColorRecipeParams.DEFAULT.copy(
                    shadows = -0.08f,
                    highlights = 0.05f
                ),
                effects = EffectParams.DEFAULT,
                frameId = "xpan",
                aspectRatio = AspectRatio.XPAN.name,
                useRaw = true,
                rawDcpId = null,
                rawColorEngine = RawColorEngine.AgX.name,
                rawDROMode = "DR100",
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
                    halation = 0.25f,
                ),
                frameId = "time",
                useRaw = true,
                rawDcpId = null,
                rawColorEngine = RawColorEngine.SpectralFilm.name,
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
                rawColorEngine = RawColorEngine.AgX.name,
                rawDROMode = "DR100",
                isBuiltIn = true
            ),
        )

        fun fromJson(json: String): CameraPreset? {
            return try {
                val element = normalizePresetJson(JsonParser.parseString(json))
                gson.fromJson(element, CameraPreset::class.java)?.withoutLegacyHdf()
            } catch (e: Exception) {
                null
            }
        }

        fun listFromJson(json: String): List<CameraPreset> {
            if (json.isEmpty()) return emptyList()
            return try {
                val type = object : TypeToken<List<CameraPreset>>() {}.type
                val element = JsonParser.parseString(json)
                if (element.isJsonArray) {
                    element.asJsonArray.forEach { normalizePresetJson(it) }
                }
                (gson.fromJson(element, type) ?: emptyList<CameraPreset>()).map { it.withoutLegacyHdf() }
            } catch (e: Exception) {
                emptyList()
            }
        }

        fun listToJson(list: List<CameraPreset>): String = gson.toJson(list.map { it.withoutLegacyHdf() })
    }
}
