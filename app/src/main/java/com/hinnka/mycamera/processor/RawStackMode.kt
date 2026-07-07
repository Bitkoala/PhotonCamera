package com.hinnka.mycamera.processor

enum class RawStackMode {
    MFNR,
    MFSR,
    HDR_MFNR,
    HDR_MFSR;

    val isHdr: Boolean
        get() = this == HDR_MFNR || this == HDR_MFSR

    val isSuperResolution: Boolean
        get() = this == MFSR || this == HDR_MFSR
}
