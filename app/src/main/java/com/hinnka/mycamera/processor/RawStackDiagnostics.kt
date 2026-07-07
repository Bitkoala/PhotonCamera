package com.hinnka.mycamera.processor

import java.util.Locale

data class RawStackMetricDistribution(
    val sampleCount: Long,
    val mean: Float,
    val p10: Float,
    val p50: Float,
    val p90: Float,
    val max: Float,
) {
    val isAvailable: Boolean
        get() = sampleCount > 0L

    companion object {
        val Empty = RawStackMetricDistribution(
            sampleCount = 0L,
            mean = Float.NaN,
            p10 = Float.NaN,
            p50 = Float.NaN,
            p90 = Float.NaN,
            max = Float.NaN,
        )
    }
}

data class RawStackDiagnostics(
    val mode: RawStackMode,
    val frameCount: Int,
    val alignedFrameCount: Int,
    val width: Int,
    val height: Int,
    val sampleStep: Int,
    val registration: RawStackRegistrationSummary? = null,
    val registrationQuality: RawStackRegistrationQualitySummary? = null,
    val flowMagnitudePx: RawStackMetricDistribution = RawStackMetricDistribution.Empty,
    val alignmentResidual: RawStackMetricDistribution = RawStackMetricDistribution.Empty,
    val noiseNormalizedResidual: RawStackMetricDistribution = RawStackMetricDistribution.Empty,
    val flowLocalRangePx: RawStackMetricDistribution = RawStackMetricDistribution.Empty,
    val robustness: RawStackMetricDistribution = RawStackMetricDistribution.Empty,
    val tileMask: RawStackMetricDistribution = RawStackMetricDistribution.Empty,
    val accumulatorWeight: RawStackMetricDistribution = RawStackMetricDistribution.Empty,
    val rejectedTileRatio: Float = Float.NaN,
    val flowOutlierRatio: Float = Float.NaN,
    val highConfidenceTileRatio: Float = Float.NaN,
    val srAlignmentReadyRatio: Float = Float.NaN,
    val srDetailReadyRatio: Float = Float.NaN,
    val lensShadingMeanGain: Float = Float.NaN,
    val lensShadingEdgeMeanGain: Float = Float.NaN,
    val elapsedMs: Long = 0L,
) {
    fun compactSummary(): String {
        val registrationSummary = registration?.compactSummary()?.let { "$it " }.orEmpty()
        val registrationQualitySummary = registrationQuality?.compactSummary()?.let { "$it " }.orEmpty()
        return "HWMF diag mode=$mode frames=$frameCount aligned=$alignedFrameCount " +
            "size=${width}x$height step=$sampleStep " +
            registrationSummary +
            registrationQualitySummary +
            "flowMean=${flowMagnitudePx.mean.fmt()} flowP90=${flowMagnitudePx.p90.fmt()} " +
            "flowMax=${flowMagnitudePx.max.fmt()} flowOut=${flowOutlierRatio.percent()} " +
            "resP90=${alignmentResidual.p90.fmt()} resN90=${noiseNormalizedResidual.p90.fmt()} " +
            "flowRangeP90=${flowLocalRangePx.p90.fmt()} " +
            "robP50=${robustness.p50.fmt()} robP10=${robustness.p10.fmt()} " +
            "tileReject=${rejectedTileRatio.percent()} tileP50=${tileMask.p50.fmt()} " +
            "hiConf=${highConfidenceTileRatio.percent()} srAlign=${srAlignmentReadyRatio.percent()} " +
            "srDetail=${srDetailReadyRatio.percent()} " +
            "weightMean=${accumulatorWeight.mean.fmt()} weightP10=${accumulatorWeight.p10.fmt()} " +
            "lscMean=${lensShadingMeanGain.fmt()} lscEdge=${lensShadingEdgeMeanGain.fmt()} " +
            "elapsed=${elapsedMs}ms"
    }
}

private fun Float.fmt(): String {
    return if (isFinite()) {
        String.format(Locale.US, "%.3f", this)
    } else {
        "n/a"
    }
}

private fun Float.percent(): String {
    return if (isFinite()) {
        String.format(Locale.US, "%.1f%%", this * 100f)
    } else {
        "n/a"
    }
}
