package com.hinnka.mycamera.processor

object RawStackTuningResolver {
    fun resolve(
        mode: RawStackMode,
        frameCount: Int = 0,
        superResolutionScale: Float = 1.5f,
    ): RawStackTuningProfile {
        val base = RawStackTuningProfile(
            mode = mode,
            requestedFrameCount = frameCount.coerceAtLeast(0),
        )
        return if (mode.isSuperResolution) {
            base.copy(
                superResolution = base.superResolution.copy(
                    outputScale = superResolutionScale.coerceIn(1.0f, base.superResolution.internalScale),
                )
            )
        } else {
            base
        }
    }
}
