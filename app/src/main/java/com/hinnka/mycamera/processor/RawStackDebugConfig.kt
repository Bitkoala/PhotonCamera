package com.hinnka.mycamera.processor

import com.hinnka.mycamera.BuildConfig

data class RawStackDebugConfig(
    val collectMetrics: Boolean = false,
    val logCompactSummary: Boolean = false,
    val sampleStep: Int = 16,
    val tileRejectThreshold: Float = 0.10f,
    val lensShadingEdgeFraction: Float = 0.125f,
    val srReadyResidualThreshold: Float = 0.018f,
    val srReadyNoiseResidualThreshold: Float = 2.5f,
    val srReadyFlowRangeThresholdPx: Float = 1.25f,
    val highConfidenceRobustnessThreshold: Float = 0.90f,
    val highConfidenceTileThreshold: Float = 0.60f,
    val srReadyDetailThreshold: Float = 0.004f,
) {
    val needsMetrics: Boolean
        get() = collectMetrics || logCompactSummary

    fun normalized(): RawStackDebugConfig {
        if (!BuildConfig.DEBUG) return Disabled
        return copy(
            collectMetrics = needsMetrics,
            sampleStep = sampleStep.coerceAtLeast(1),
            tileRejectThreshold = tileRejectThreshold.coerceIn(0f, 1f),
            lensShadingEdgeFraction = lensShadingEdgeFraction.coerceIn(0.02f, 0.45f),
            srReadyResidualThreshold = srReadyResidualThreshold.coerceIn(0.001f, 0.20f),
            srReadyNoiseResidualThreshold = srReadyNoiseResidualThreshold.coerceIn(0.25f, 16.0f),
            srReadyFlowRangeThresholdPx = srReadyFlowRangeThresholdPx.coerceIn(0.10f, 16.0f),
            highConfidenceRobustnessThreshold = highConfidenceRobustnessThreshold.coerceIn(0f, 1f),
            highConfidenceTileThreshold = highConfidenceTileThreshold.coerceIn(0f, 1f),
            srReadyDetailThreshold = srReadyDetailThreshold.coerceIn(0f, 0.20f),
        )
    }

    companion object {
        val Disabled = RawStackDebugConfig()
        val CompactSummary = RawStackDebugConfig(
            collectMetrics = true,
            logCompactSummary = true,
        )

        fun forCurrentBuild(): RawStackDebugConfig {
            // Diagnostic shaders and SSBO readback are opt-in. A normal debug photo must exercise
            // the same production pass graph instead of paying for metrics that do not affect output.
            return Disabled
        }
    }
}
