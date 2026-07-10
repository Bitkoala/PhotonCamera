package com.hinnka.mycamera.camera

import com.hinnka.mycamera.raw.RawCfaCorrection
import com.hinnka.mycamera.raw.RawWhiteLevelCorrection
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToInt

data class IszLensConfig(
    val baseCameraId: String,
    val iszZoomRatio: Float,
    val isMacro: Boolean = false,
    val rawBlackBorderCrop: RawBlackBorderCrop = RawBlackBorderCrop(),
    val vendorCaptureProfileId: String? = null
) {
    val virtualCameraId: String
        get() = createVirtualCameraId(baseCameraId, iszZoomRatio, vendorCaptureProfileId)

    fun toJsonObject(): JSONObject {
        return JSONObject().apply {
            put(KEY_BASE_CAMERA_ID, baseCameraId)
            put(KEY_ISZ_ZOOM_RATIO, iszZoomRatio)
            put(KEY_IS_MACRO, isMacro)
            sanitizeVendorCaptureProfileId(vendorCaptureProfileId)?.let {
                put(KEY_VENDOR_CAPTURE_PROFILE_ID, it)
            }
            val sanitizedCrop = sanitizeRawBlackBorderCrop(rawBlackBorderCrop)
            put(KEY_RAW_BLACK_BORDER_CROP_LEFT_PX, sanitizedCrop.leftPx)
            put(KEY_RAW_BLACK_BORDER_CROP_TOP_PX, sanitizedCrop.topPx)
            put(KEY_RAW_BLACK_BORDER_CROP_RIGHT_PX, sanitizedCrop.rightPx)
            put(KEY_RAW_BLACK_BORDER_CROP_BOTTOM_PX, sanitizedCrop.bottomPx)
        }
    }

    companion object {
        private const val KEY_BASE_CAMERA_ID = "base_camera_id"
        private const val KEY_ISZ_ZOOM_RATIO = "isz_zoom_ratio"
        private const val KEY_IS_MACRO = "is_macro"
        private const val KEY_VENDOR_CAPTURE_PROFILE_ID = "vendor_capture_profile_id"
        private const val KEY_RAW_BLACK_BORDER_CROP_PX = "raw_black_border_crop_px"
        private const val KEY_RAW_BLACK_BORDER_CROP_LEFT_PX = "raw_black_border_crop_left_px"
        private const val KEY_RAW_BLACK_BORDER_CROP_TOP_PX = "raw_black_border_crop_top_px"
        private const val KEY_RAW_BLACK_BORDER_CROP_RIGHT_PX = "raw_black_border_crop_right_px"
        private const val KEY_RAW_BLACK_BORDER_CROP_BOTTOM_PX = "raw_black_border_crop_bottom_px"
        private const val MAX_RAW_BLACK_BORDER_CROP_PX = 4096
        private const val VIRTUAL_CAMERA_ID_PREFIX = "isz"

        fun createVirtualCameraId(
            baseCameraId: String,
            iszZoomRatio: Float,
            vendorCaptureProfileId: String? = null
        ): String {
            val baseId = "$VIRTUAL_CAMERA_ID_PREFIX:$baseCameraId:${formatRatioForId(iszZoomRatio)}"
            val profileId = sanitizeVendorCaptureProfileId(vendorCaptureProfileId) ?: return baseId
            return "$baseId:$profileId"
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
                    val vendorCaptureProfileId = sanitizeVendorCaptureProfileId(
                        obj.optString(KEY_VENDOR_CAPTURE_PROFILE_ID, "")
                    )
                    val legacyLeftCropPx = obj.optInt(KEY_RAW_BLACK_BORDER_CROP_PX, 0)
                    val rawBlackBorderCrop = sanitizeRawBlackBorderCrop(
                        RawBlackBorderCrop(
                            leftPx = obj.optInt(KEY_RAW_BLACK_BORDER_CROP_LEFT_PX, legacyLeftCropPx),
                            topPx = obj.optInt(KEY_RAW_BLACK_BORDER_CROP_TOP_PX, 0),
                            rightPx = obj.optInt(KEY_RAW_BLACK_BORDER_CROP_RIGHT_PX, 0),
                            bottomPx = obj.optInt(KEY_RAW_BLACK_BORDER_CROP_BOTTOM_PX, 0)
                        )
                    )
                    if (baseCameraId.isNotEmpty() && iszZoomRatio >= 1f) {
                        add(
                            IszLensConfig(
                                baseCameraId = baseCameraId,
                                iszZoomRatio = iszZoomRatio,
                                isMacro = isMacro,
                                rawBlackBorderCrop = rawBlackBorderCrop,
                                vendorCaptureProfileId = vendorCaptureProfileId
                            )
                        )
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

        fun sanitizeVendorCaptureProfileId(value: String?): String? {
            val trimmed = value?.trim()?.takeIf { it.isNotEmpty() } ?: return null
            return trimmed
                .map { char ->
                    if (char.isLetterOrDigit() || char == '_' || char == '-' || char == '.') {
                        char
                    } else {
                        '_'
                    }
                }
                .joinToString(separator = "")
                .take(160)
                .takeIf { it.isNotBlank() }
        }

        fun sanitizeRawBlackBorderCropPx(value: Int): Int {
            return value.coerceIn(0, MAX_RAW_BLACK_BORDER_CROP_PX)
        }

        fun sanitizeRawBlackBorderCrop(crop: RawBlackBorderCrop): RawBlackBorderCrop {
            return RawBlackBorderCrop(
                leftPx = sanitizeRawBlackBorderCropPx(crop.leftPx),
                topPx = sanitizeRawBlackBorderCropPx(crop.topPx),
                rightPx = sanitizeRawBlackBorderCropPx(crop.rightPx),
                bottomPx = sanitizeRawBlackBorderCropPx(crop.bottomPx)
            )
        }
    }
}

data class RawBlackBorderCrop(
    val leftPx: Int = 0,
    val topPx: Int = 0,
    val rightPx: Int = 0,
    val bottomPx: Int = 0
) {
    val hasCrop: Boolean
        get() = leftPx > 0 || topPx > 0 || rightPx > 0 || bottomPx > 0
}

/**
 * RAW DNG 元数据校正参数，会以 ISZ 虚拟镜头 ID 为键持久化。
 *
 * ISZ 镜头和其基础物理镜头共用同一枚传感器，但其输出的 RAW 元数据可能不同，
 * 因此不能复用基础镜头的校正值。
 */
data class IszRawDngMetadataCorrections(
    val blackLevelMode: String = RawCfaCorrection.MODE_DEFAULT,
    val customBlackLevel: Float = 0f,
    val whiteLevelMode: String = RawWhiteLevelCorrection.MODE_DEFAULT,
    val customWhiteLevel: Float = 0f,
    val cfaCorrectionMode: String = RawCfaCorrection.MODE_DEFAULT
)
