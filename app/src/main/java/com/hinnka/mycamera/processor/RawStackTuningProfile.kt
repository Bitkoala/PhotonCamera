package com.hinnka.mycamera.processor

/**
 * Photon-side HWMF tuning profile.
 *
 * The stage grouping deliberately follows the CamX/CHI HWMF flow:
 * Prefilter -> BlendInit/BlendLoop -> Postfilter. Values here are Photon
 * defaults derived from the current GLES RAW stacker behavior.
 */
data class RawStackTuningProfile(
    val mode: RawStackMode = RawStackMode.MFNR,
    val requestedFrameCount: Int = 0,
    val prefilter: RawStackPrefilterTuning = RawStackPrefilterTuning(),
    val blend: RawStackBlendTuning = RawStackBlendTuning(),
    val postfilter: RawStackPostfilterTuning = RawStackPostfilterTuning(),
    val hdr: RawStackHdrTuning = RawStackHdrTuning(),
    val superResolution: RawStackSuperResolutionTuning = RawStackSuperResolutionTuning(),
)

data class RawStackPrefilterTuning(
    val pyramidLevels: Int = 4,
    val alignLevel: Int = 2,
    val flowGridSpacing: Int = 8,
    val alignWindowSize: Int = 32,
    val alignSearchRadiusLevel: Int = 6,
    val alignSampleStep: Int = 2,
    val alignCoveragePenalty: Float = 0.08f,
    val alignShiftPenalty: Float = 0.0006f,
    val structureFlatnessSnrLow: Float = 0.35f,
    val structureFlatnessSnrHigh: Float = 4.0f,
    val structureKernelDetail: Float = 0.26f,
    val structureKernelDenoise: Float = 1.0f,
    val structureKernelShrink: Float = 3.0f,
    val structureKernelStretch: Float = 3.6f,
    val structureAnisotropyThreshold: Float = 1.6f,
)

data class RawStackBlendTuning(
    val lkRefinePasses: Int = 2,
    val flowSmoothPasses: Int = 2,
    val flowOutlierThresholdPx: Float = 12.0f,
    val flowOutlierWeight: Float = 0.15f,
    val nonReferenceFrameWeight: Float = 0.92f,
    val robustnessNoiseFloorSpatialScale: Float = 4.5f,
    val robustnessNoiseFloorEdgeScale: Float = 0.75f,
    val robustnessTauBase: Float = 1.0f,
    val robustnessTauEdge: Float = 0.75f,
    val robustnessResidualPower: Float = 8.0f,
    val flowPenaltyStartPx: Float = 15.0f,
    val flowPenaltyDecay: Float = 0.1f,
    val flowRangePenaltyStartPx: Float = 8.0f,
    val flowRangePenaltyDecay: Float = 0.1f,
    val robustMinMixFlat: Float = 0.35f,
    val robustMinMixEdge: Float = 0.15f,
    val robustCenterMixFlat: Float = 0.35f,
    val robustCenterMixEdge: Float = 0.65f,
    val tileRobustCenter: Float = 0.62f,
    val tileRobustWidth: Float = 0.22f,
    val tileWeakThreshold: Float = 0.5f,
    val tileWeakStart: Float = 0.08f,
    val tileWeakRange: Float = 0.26f,
    val tileDetailMid: Float = 0.025f,
    val tileDetailHigh: Float = 0.055f,
    val tileDetailBoostLow: Float = 0.30f,
    val tileDetailBoostMid: Float = 0.55f,
    val tileDetailBoostHigh: Float = 0.85f,
    val tileMaskMinMidDetail: Float = 0.14f,
    val tileMaskMinHighDetail: Float = 0.28f,
    val minBlendBaseWeight: Float = 0.001f,
    val robustnessFloorFactor: Float = 0.01f,
    val sensorPrecisionReferenceSignal: Float = 0.18f,
    val lscNoiseGainMax: Float = 4.0f,
    val wienerBaseWeight: Float = 0.25f,
    val highlightSuppressionStrength: Float = 0.62f,
    val highlightSuppressionStart: Float = 0.78f,
    val highlightSuppressionEnd: Float = 0.98f,
)

data class RawStackPostfilterTuning(
    val finalSmoothStrength: Float = 0.38f,
    val flatVarianceStart: Float = 0.00025f,
    val flatVarianceEnd: Float = 0.0035f,
    val detailKeepNoiseLowScale: Float = 1.4f,
    val detailKeepNoiseHighScale: Float = 3.2f,
    val detailKeepOffsetLow: Float = 0.0025f,
    val detailKeepOffsetHigh: Float = 0.010f,
    val detailKeepSuppression: Float = 0.85f,
    val lscNoiseGainMax: Float = 4.0f,
    val hdrRecoverySmoothSuppression: Float = 0.65f,
)

data class RawStackHdrTuning(
    val shortGlobalSearchRadiusLevel: Int = 8,
    val shortGlobalSampleStep: Int = 6,
    val shortGlobalSampleBorder: Int = 8,
    val shortGlobalCoveragePenalty: Float = 0.12f,
    val shortGlobalShiftPenalty: Float = 0.0008f,
)

data class RawStackSuperResolutionTuning(
    val internalScale: Float = 2.0f,
    val outputScale: Float = 1.5f,
    val splatRadius: Float = 1.0f,
    val minEffectiveWeight: Float = 0.35f,
    val holeFillPasses: Int = 2,
    val detailRestoreStrength: Float = 0.35f,
)
