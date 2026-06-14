package com.hinnka.mycamera.raw

const val RAW_COLOR_ENGINE_METERING_COMPENSATION_EV = 0.7f

enum class RawExposureCompensationDomain {
    Curve,
    Linear
}

enum class RawColorEngine(
    val shaderId: Int,
    val workingColorSpace: ColorSpace,
    val defaultExposureCompensationEv: Float,
    val meteringCompensationEv: Float,
    val exposureCompensationDomain: RawExposureCompensationDomain
) {
    AdobeCurve(
        shaderId = 0,
        workingColorSpace = ColorSpace.ProPhoto,
        defaultExposureCompensationEv = 0f,
        meteringCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        exposureCompensationDomain = RawExposureCompensationDomain.Curve
    ),
    AgX(
        shaderId = 1,
        workingColorSpace = ColorSpace.FilmLightEGamut,
        defaultExposureCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        meteringCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        exposureCompensationDomain = RawExposureCompensationDomain.Linear
    ),
    SpectralFilm(
        shaderId = 2,
        workingColorSpace = ColorSpace.ProPhoto,
        defaultExposureCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        meteringCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        exposureCompensationDomain = RawExposureCompensationDomain.Linear
    ),
    ;

    companion object {
        fun fromPersistedName(value: String?): RawColorEngine {
            return entries.firstOrNull { it.name.equals(value, ignoreCase = true) } ?: AgX
        }
    }
}
