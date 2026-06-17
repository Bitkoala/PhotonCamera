package com.hinnka.mycamera.camera

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToInt

data class IszLensConfig(
    val baseCameraId: String,
    val iszZoomRatio: Float,
    val isMacro: Boolean = false
) {
    val virtualCameraId: String
        get() = createVirtualCameraId(baseCameraId, iszZoomRatio)

    fun toJsonObject(): JSONObject {
        return JSONObject().apply {
            put(KEY_BASE_CAMERA_ID, baseCameraId)
            put(KEY_ISZ_ZOOM_RATIO, iszZoomRatio)
            put(KEY_IS_MACRO, isMacro)
        }
    }

    companion object {
        private const val KEY_BASE_CAMERA_ID = "base_camera_id"
        private const val KEY_ISZ_ZOOM_RATIO = "isz_zoom_ratio"
        private const val KEY_IS_MACRO = "is_macro"
        private const val VIRTUAL_CAMERA_ID_PREFIX = "isz"

        fun createVirtualCameraId(baseCameraId: String, iszZoomRatio: Float): String {
            return "$VIRTUAL_CAMERA_ID_PREFIX:$baseCameraId:${formatRatioForId(iszZoomRatio)}"
        }

        fun deserializeList(value: String?): List<IszLensConfig> {
            if (value.isNullOrBlank()) return emptyList()
            val array = runCatching { JSONArray(value) }.getOrNull() ?: return emptyList()
            return buildList {
                for (index in 0 until array.length()) {
                    val obj = array.optJSONObject(index) ?: continue
                    val baseCameraId = obj.optString(KEY_BASE_CAMERA_ID).trim()
                    val iszZoomRatio = obj.optDouble(KEY_ISZ_ZOOM_RATIO, 0.0).toFloat()
                    val isMacro = obj.optBoolean(KEY_IS_MACRO, false)
                    if (baseCameraId.isNotEmpty() && iszZoomRatio >= 1f) {
                        add(IszLensConfig(baseCameraId, iszZoomRatio, isMacro))
                    }
                }
            }.distinctBy { it.virtualCameraId }
        }

        fun serializeList(configs: List<IszLensConfig>): String {
            return JSONArray().apply {
                configs
                    .filter { it.baseCameraId.isNotBlank() && it.iszZoomRatio >= 1f }
                    .distinctBy { it.virtualCameraId }
                    .forEach { put(it.toJsonObject()) }
            }.toString()
        }

        fun displayRatioLabel(ratio: Float): String {
            val rounded = ratio.roundToInt()
            return if (abs(ratio - rounded) < 0.05f) {
                "${rounded}x"
            } else {
                String.format(Locale.US, "%.1fx", ratio)
            }
        }

        private fun formatRatioForId(ratio: Float): String {
            return displayRatioLabel(ratio).dropLast(1)
        }
    }
}
