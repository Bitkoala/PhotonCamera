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
    AgX(
        shaderId = 1,
        workingColorSpace = ColorSpace.FilmLightEGamut,
        defaultExposureCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        meteringCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        exposureCompensationDomain = RawExposureCompensationDomain.Linear
    ),
    DarktableSigmoid(
        shaderId = 3,
        workingColorSpace = ColorSpace.BT2020,
        defaultExposureCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        meteringCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        exposureCompensationDomain = RawExposureCompensationDomain.Linear
    ),
    DarktableFilmic(
        shaderId = 4,
        workingColorSpace = ColorSpace.BT2020,
        defaultExposureCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        meteringCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        exposureCompensationDomain = RawExposureCompensationDomain.Linear
    ),
    AdobeCurve(
        shaderId = 0,
        workingColorSpace = ColorSpace.ProPhoto,
        defaultExposureCompensationEv = 0f,
        meteringCompensationEv = RAW_COLOR_ENGINE_METERING_COMPENSATION_EV,
        exposureCompensationDomain = RawExposureCompensationDomain.Curve
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
        fun fromPersistedName(
            value: String?,
            fallback: RawColorEngine = AgX
        ): RawColorEngine {
            return entries.firstOrNull { it.name.equals(value, ignoreCase = true) } ?: fallback
        }
    }
}
