package com.hinnka.mycamera.gallery

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorSpace
import android.graphics.Gainmap
import android.os.Build
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.data.UserPreferencesRepository
import com.hinnka.mycamera.frame.FrameManager
import com.hinnka.mycamera.frame.FrameRenderer
import com.hinnka.mycamera.frame.FrameTemplate
import com.hinnka.mycamera.hdr.GainmapResult
import com.hinnka.mycamera.hdr.GainmapSourceSet
import com.hinnka.mycamera.hdr.HlgImageProcessor
import com.hinnka.mycamera.hdr.HdrBuffer
import com.hinnka.mycamera.hdr.SourceKind
import com.hinnka.mycamera.lut.BaselineColorCorrectionTarget
import com.hinnka.mycamera.lut.ColorCorrectionPipelineResolver
import com.hinnka.mycamera.lut.LutImageProcessor
import com.hinnka.mycamera.lut.LutManager
import com.hinnka.mycamera.processor.DepthBokehProcessor
import com.hinnka.mycamera.raw.RawDemosaicProcessor
import com.hinnka.mycamera.raw.RawHdrRenderResult
import com.hinnka.mycamera.raw.SpectralFilmTuning
import com.hinnka.mycamera.utils.BitmapUtils
import com.hinnka.mycamera.utils.LargeDirectBuffer
import com.hinnka.mycamera.utils.PLog
import com.hinnka.mycamera.utils.YuvProcessor
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.system.measureTimeMillis

/**
 * 照片处理器
 *
 * 集中管理照片的 LUT、旋转、亮度和边框应用逻辑
 */
class PhotoProcessor(
    private val lutManager: LutManager,
    private val lutImageProcessor: LutImageProcessor,
    private val frameManager: FrameManager,
    private val frameRenderer: FrameRenderer,
    private val depthBokehProcessor: DepthBokehProcessor,
    private val userPreferencesRepository: UserPreferencesRepository,
) {
    private val hlgImageProcessor = HlgImageProcessor()
    private val colorCorrectionPipelineResolver = ColorCorrectionPipelineResolver(lutManager)

    data class HdrFrameOutput(
        val bitmap: Bitmap,
        val gainmapResult: GainmapResult?,
    )

    private data class ResolvedFrame(
        val template: FrameTemplate,
        val metadata: MediaMetadata,
    )

    private suspend fun shouldDecodeHlgInput(metadata: MediaMetadata): Boolean {
        val isHlg = metadata.dynamicRangeProfile == "HLG10"
        if (!isHlg) return false
        return userPreferencesRepository.userPreferences.firstOrNull()?.hlgHardwareCompatibilityEnabled ?: false
    }

    private suspend fun resolveRawAutoWhiteBalanceEstimate(metadata: MediaMetadata): Boolean {
        return metadata.rawAutoWhiteBalanceEstimate
            ?: (userPreferencesRepository.userPreferences.firstOrNull()?.rawAutoWhiteBalanceEstimate ?: false)
    }

    private suspend fun resolveRawAutoExposure(metadata: MediaMetadata): Boolean {
        return metadata.rawAutoExposure
            ?: (userPreferencesRepository.userPreferences.firstOrNull()?.rawAutoExposure ?: true)
    }

    private fun resolveNoiseReduction(metadata: MediaMetadata, fallback: Float): Float {
        return metadata.noiseReduction ?: (if (metadata.isImported) 0f else fallback)
    }

    private fun resolveChromaNoiseReduction(metadata: MediaMetadata, fallback: Float): Float {
        return metadata.chromaNoiseReduction ?: (if (metadata.isImported) 0f else fallback)
    }

    suspend fun prepareUltraHdrSource(
        context: Context,
        photoId: String,
        metadata: MediaMetadata,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f
    ): GainmapSourceSet? {
        if (!metadata.manualHdrEffectEnabled) {
            return null
        }

        var source: GainmapSourceSet? = null

        val dngFile = GalleryManager.getDngFile(context, photoId)
        if (dngFile.exists()) {
            source = processDngForUltraHdr(
                context = context,
                photoId = photoId,
                dngPath = dngFile.absolutePath,
                metadata = metadata,
                sharpening = sharpening,
                noiseReduction = noiseReduction,
                chromaNoiseReduction = chromaNoiseReduction
            )
        } else {
            val yuvFile = GalleryManager.getYuvFile(context, photoId)
            if (hlgImageProcessor.isHlgCapture(metadata)) {
                val prepareStart = System.currentTimeMillis()
                val hdrData = GalleryManager.loadHdrData(
                    context = context,
                    photoId = photoId,
                    fallbackWidth = metadata.width,
                    fallbackHeight = metadata.height
                )
                if (hdrData != null) {
                    try {
                        val photoFile = GalleryManager.getPhotoFile(context, photoId)
                        val photoBitmap = BitmapFactory.decodeFile(photoFile.absolutePath) ?: return null
                        val finalSharpening = metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening)
                        val finalNoiseReduction = metadata.noiseReduction ?: (if (metadata.isImported) 0f else noiseReduction)
                        val finalChromaNoiseReduction =
                            metadata.chromaNoiseReduction ?: (if (metadata.isImported) 0f else chromaNoiseReduction)

                        val sdrPostElapsedStart = System.currentTimeMillis()
                        val sdrBitmap = processBitmap(
                            context = context,
                            photoId = photoId,
                            input = photoBitmap,
                            metadata = metadata,
                            sharpening = finalSharpening,
                            noiseReduction = finalNoiseReduction,
                            chromaNoiseReduction = finalChromaNoiseReduction,
                            useComputationalAperture = true,
                            applyFrameWatermark = false
                        )
                        val hdrReferenceBitmap = hlgImageProcessor.createHdrReferenceFromRawSidecar(
                            buffer = hdrData.buffer,
                            width = hdrData.width,
                            height = hdrData.height
                        ).let {
                            applyCrop(it, metadata, "hlg_sidecar_hdr")
                        }
                        PLog.d(
                            "PhotoProcessor",
                            "prepareUltraHdrSource(HLG sidecar) took ${System.currentTimeMillis() - prepareStart}ms " +
                                    "(sdrPost=${System.currentTimeMillis() - sdrPostElapsedStart}ms, " +
                                    "hasHdr=true, sidecar=${hdrData.width}x${hdrData.height}, compressed=${hdrData.compressed})"
                        )
                        source = GainmapSourceSet(
                            sdrBase = sdrBitmap,
                            hdrReference = HdrBuffer(hdrReferenceBitmap, "hlg_sidecar_rgba16"),
                            sourceKind = SourceKind.HLG_CAPTURE,
                            confidence = 0.75f,
                            displayHdrSdrRatio = readDisplayHdrSdrRatio()
                        )
                    } finally {
                        if (hdrData.usesLargeDirectAllocator) {
                            LargeDirectBuffer.free(hdrData.buffer)
                        }
                    }
                } else if (yuvFile.exists()) {
                    val data = GalleryManager.loadYuvData(context, photoId)
                    if (data != null) {
                        try {
                            val finalSharpening = metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening)
                            val finalNoiseReduction = metadata.noiseReduction ?: (if (metadata.isImported) 0f else noiseReduction)
                            val finalChromaNoiseReduction =
                                metadata.chromaNoiseReduction ?: (if (metadata.isImported) 0f else chromaNoiseReduction)
                            val colorCorrection = resolveColorCorrection(
                                metadata = metadata,
                                fallbackTarget = BaselineColorCorrectionTarget.JPG
                            )
                            val sourceElapsed = measureTimeMillis {
                                source = hlgImageProcessor.createSourceFromCompressedArgb(
                                    buffer = data,
                                    width = metadata.width,
                                    height = metadata.height
                                )
                            }
                            val hlgSource = source ?: return null
                            var sdrBitmap = hlgSource.sdrBase
                            val sdrPostElapsed = measureTimeMillis {
                                sdrBitmap = lutImageProcessor.applyLutStack(
                                    sdrBitmap,
                                    isHlgInput = false,
                                    colorCorrection.baselineLayer,
                                    colorCorrection.creativeLayer,
                                    finalSharpening,
                                    finalNoiseReduction,
                                    finalChromaNoiseReduction
                                )
                                sdrBitmap = applyCrop(sdrBitmap, metadata, "hlg_sdr")
                            }

                            val hdrReferenceBitmap = hlgSource.hdrReference?.bitmap?.let {
                                applyCrop(it, metadata, "hlg_hdr")
                            }
                            PLog.d(
                                "PhotoProcessor",
                                "prepareUltraHdrSource(HLG) took ${System.currentTimeMillis() - prepareStart}ms " +
                                        "(source=${sourceElapsed}ms, sdrPost=${sdrPostElapsed}ms, hasHdr=${hdrReferenceBitmap != null})"
                            )

                            source = GainmapSourceSet(
                                sdrBase = sdrBitmap,
                                hdrReference = hdrReferenceBitmap?.let { HdrBuffer(it, "hlg_bt2020_linear") },
                                sourceKind = SourceKind.HLG_CAPTURE,
                                confidence = hlgSource.confidence,
                                displayHdrSdrRatio = readDisplayHdrSdrRatio()
                            )
                        } finally {
                            YuvProcessor.free(data)
                        }
                    }
                }
            }
        }

        if (source == null) {
            val photoFile = GalleryManager.getPhotoFile(context, photoId)
            if (photoFile.exists()) {
                if (EmbeddedGainmapReusePolicy.canReuse(metadata)) {
                    val bitmap = GalleryManager.loadBitmap(context, photoId, preserveHdr = true)
                    if (bitmap != null) {
                        source = GainmapSourceSet(
                            sdrBase = bitmap,
                            sourceKind = SourceKind.SDR_BITMAP,
                            confidence = 1.0f,
                            displayHdrSdrRatio = readDisplayHdrSdrRatio()
                        )
                    }
                } else if (metadata.hasEmbeddedGainmap) {
                    PLog.d("PhotoProcessor", "Skip embedded gainmap reuse after edits: $photoId")
                }
            }
        }

        // If source was generated from DNG/YUV, and we have an AI denoised base, replace the sdrBase.
        // The AI base is persisted in ai_denoise.jpg so exports and HDR gainmaps never rerun the slow model.
        if (source != null && metadata.hasAiDenoisedBase) {
            val aiFile = GalleryManager.getAiDenoiseFile(context, photoId)
            if (aiFile.exists()) {
                val bitmap = BitmapFactory.decodeFile(aiFile.absolutePath)
                if (bitmap != null) {
                    val finalSharpening = metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening)
                    val finalNoiseReduction = metadata.noiseReduction ?: (if (metadata.isImported) 0f else noiseReduction)
                    val finalChromaNoiseReduction =
                        metadata.chromaNoiseReduction ?: (if (metadata.isImported) 0f else chromaNoiseReduction)
                        
                    val sdrBitmap = processBitmap(
                        context = context,
                        photoId = photoId,
                        input = bitmap,
                        metadata = metadata,
                        sharpening = finalSharpening,
                        noiseReduction = finalNoiseReduction,
                        chromaNoiseReduction = finalChromaNoiseReduction,
                        useComputationalAperture = true,
                        applyFrameWatermark = false
                    )
                    source.sdrBase.recycle()
                    source = source.copy(sdrBase = sdrBitmap)
                }
            }
        }

        if (source != null) {
            return source
        }


        val fallbackFile = if (metadata.hasAiDenoisedBase) {
            GalleryManager.getAiDenoiseFile(context, photoId)
        } else {
            GalleryManager.getPhotoFile(context, photoId)
        }
        if (fallbackFile.exists()) {
            val bitmap = BitmapFactory.decodeFile(fallbackFile.absolutePath) ?: return null
            val sdrBitmap = processBitmap(
                context = context,
                photoId = photoId,
                input = bitmap,
                metadata = metadata,
                sharpening = sharpening,
                noiseReduction = noiseReduction,
                chromaNoiseReduction = chromaNoiseReduction,
                useComputationalAperture = true,
                applyFrameWatermark = false
            )
            return GainmapSourceSet(
                sdrBase = sdrBitmap,
                hdrReference = null,
                sourceKind = SourceKind.SDR_BITMAP,
                confidence = 0.35f,
                displayHdrSdrRatio = readDisplayHdrSdrRatio()
            )
        }

        return null
    }

    fun hasDeferredHlgSource(metadata: MediaMetadata): Boolean {
        return hlgImageProcessor.isHlgCapture(metadata)
    }

    private fun readDisplayHdrSdrRatio(): Float = GalleryManager.hdrSdrRatio

    suspend fun prepareUltraHdrSourceFromRawResult(
        context: Context,
        photoId: String?,
        rawResult: RawHdrRenderResult,
        metadata: MediaMetadata,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f,
        applyMirror: Boolean = false,
    ): GainmapSourceSet? = withContext(Dispatchers.IO) {
        val displayHdrSdrRatio = readDisplayHdrSdrRatio()
        val finalSharpening = metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening)

        val colorCorrection = resolveColorCorrection(
            metadata = metadata,
            fallbackTarget = BaselineColorCorrectionTarget.RAW
        )

        var sdrBitmap = rawResult.sdrBitmap
        var hdrReferenceBitmap = rawResult.hdrReferenceBitmap

        if (applyMirror && metadata.isMirrored) {
            sdrBitmap = BitmapUtils.flipHorizontal(sdrBitmap)
            hdrReferenceBitmap = hdrReferenceBitmap?.let { BitmapUtils.flipHorizontal(it) }
        }

        metadata.computationalAperture?.let { aperture ->
            sdrBitmap = depthBokehProcessor.applyHighQualityBokeh(
                context,
                photoId,
                sdrBitmap,
                metadata.focusPointX,
                metadata.focusPointY,
                aperture
            )
            hdrReferenceBitmap = hdrReferenceBitmap?.let {
                depthBokehProcessor.applyHighQualityBokeh(
                    context,
                    photoId,
                    it,
                    metadata.focusPointX,
                    metadata.focusPointY,
                    aperture
                )
            }
            photoId?.let { id -> GalleryManager.saveBokehPhoto(context, id, sdrBitmap) }
        }

        sdrBitmap = lutImageProcessor.applyLutStack(
            sdrBitmap,
            isHlgInput = false,
            colorCorrection.baselineLayer,
            colorCorrection.creativeLayer,
            finalSharpening,
            noiseReductionValue = 0f,
            chromaNoiseReductionValue = 0f
        )

        sdrBitmap = applyCrop(sdrBitmap, metadata, "raw_sdr")
        hdrReferenceBitmap = hdrReferenceBitmap?.let { applyCrop(it, metadata, "raw_hdr") }

        GainmapSourceSet(
            sdrBase = sdrBitmap,
            hdrReference = hdrReferenceBitmap?.let {
                HdrBuffer(
                    bitmap = it,
                    description = "raw_scene_normalized"
                )
            },
            sourceKind = SourceKind.RAW,
            confidence = 0.8f,
            displayHdrSdrRatio = displayHdrSdrRatio
        )
    }

    suspend fun process(
        context: Context, photoId: String, metadata: MediaMetadata,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f
    ): Bitmap? {
        val dngFile = GalleryManager.getDngFile(context, photoId)
        val yuvFile = GalleryManager.getYuvFile(context, photoId)
        val photoFile = GalleryManager.getPhotoFile(context, photoId)

        if (metadata.hasAiDenoisedBase) {
            val aiFile = GalleryManager.getAiDenoiseFile(context, photoId)
            if (aiFile.exists()) {
                val bitmap = GalleryManager.loadBitmap(context, android.net.Uri.fromFile(aiFile)) ?: return null
                return processBitmap(
                    context,
                    photoId,
                    bitmap,
                    metadata,
                    sharpening,
                    noiseReduction,
                    chromaNoiseReduction,
                    true
                )
            }
            return null
        }

        if (dngFile.exists()) {
            return processDng(
                context,
                photoId,
                dngFile.absolutePath,
                metadata,
                sharpening,
                noiseReduction,
                chromaNoiseReduction
            )
        } else if (yuvFile.exists()) {
            val data = GalleryManager.loadYuvData(context, photoId) ?: return null
            return processYuv(
                context,
                photoId,
                data,
                metadata,
                sharpening,
                noiseReduction,
                chromaNoiseReduction
            )
        } else if (photoFile.exists()) {
            val bitmap = GalleryManager.loadOriginalBitmap(context, photoId) ?: return null
            return processBitmap(
                context,
                photoId,
                bitmap,
                metadata,
                sharpening,
                noiseReduction,
                chromaNoiseReduction,
                true
            )
        }
        return null
    }

    private suspend fun processDngForUltraHdr(
        context: Context,
        photoId: String?,
        dngPath: String,
        metadata: MediaMetadata,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f
    ): GainmapSourceSet? = withContext(Dispatchers.IO) {
        val rawNoiseReduction = resolveNoiseReduction(metadata, noiseReduction)
        val rawChromaNoiseReduction = resolveChromaNoiseReduction(metadata, chromaNoiseReduction)
        val rawResult = RawDemosaicProcessor.getInstance().processForHdrSources(
            context = context,
            dngFilePath = dngPath,
            aspectRatio = resolveRawAspectRatio(metadata),
            cropRegion = metadata.cropRegion,
            rotation = metadata.rotation,
            exposureBias = metadata.exposureBias ?: 0f,
            rawExposureCompensation = metadata.rawExposureCompensation ?: 0f,
            rawAutoExposure = resolveRawAutoExposure(metadata),
            rawHighlightsAdjustment = metadata.rawHighlightsAdjustment ?: 0f,
            rawShadowsAdjustment = metadata.rawShadowsAdjustment ?: 0f,
            rawBlackPointCorrection = metadata.rawBlackPointCorrection ?: 0f,
            rawWhitePointCorrection = metadata.rawWhitePointCorrection ?: 0f,
            rawAutoWhiteBalanceEstimate = resolveRawAutoWhiteBalanceEstimate(metadata),
            rawBlackLevelMode = metadata.rawBlackLevelMode,
            rawCustomBlackLevel = metadata.rawCustomBlackLevel,
            rawWhiteLevelMode = metadata.rawWhiteLevelMode,
            sharpeningValue = 0.4f,
            denoiseValue = rawNoiseReduction,
            chromaDenoiseValue = rawChromaNoiseReduction,
            rawDcpId = metadata.rawDcpId,
            rawRenderingEngine = metadata.rawRenderingEngine,
            rawToneMappingParameters = metadata.rawToneMappingParameters,
            rawCfaCorrectionMode = metadata.rawCfaCorrectionMode,
            spectralFilmStock = metadata.spectralFilmStock,
            spectralFilmPrint = metadata.spectralFilmPrint,
            spectralFilmTuning = SpectralFilmTuning(
                cDensityGain = metadata.spectralFilmCDensityGain,
                mDensityGain = metadata.spectralFilmMDensityGain,
                yDensityGain = metadata.spectralFilmYDensityGain
            )
        ) ?: return@withContext null
        prepareUltraHdrSourceFromRawResult(
            context = context,
            photoId = photoId,
            rawResult = rawResult,
            metadata = metadata,
            sharpening = sharpening,
            noiseReduction = noiseReduction,
            chromaNoiseReduction = chromaNoiseReduction,
            applyMirror = true
        )
    }

    /**
     * @param dngPath dng 文件路径
     * @param metadata 照片元数据（包含编辑配置和拍摄信息）
     * @param sharpening 锐化强度
     * @param noiseReduction 降噪强度
     * @param chromaNoiseReduction 减少杂色强度
     * @return 处理后的 Bitmap
     */
    suspend fun processDng(
        context: Context,
        photoId: String?,
        dngPath: String,
        metadata: MediaMetadata,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f
    ): Bitmap? = withContext(Dispatchers.IO) {
        var result: Bitmap?

        // 优先从元数据中获取软件处理参数
        // 智能回退：如果是导入的照片且元数据中没存过，则默认值为 0，不应用额外处理
        val finalSharpening = metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening)
        val finalNoiseReduction = resolveNoiseReduction(metadata, noiseReduction)
        val finalChromaNoiseReduction = resolveChromaNoiseReduction(metadata, chromaNoiseReduction)

        // 1. 应用 LUT
        val colorCorrection = resolveColorCorrection(
            metadata = metadata,
            fallbackTarget = BaselineColorCorrectionTarget.RAW
        )
        val cropRegion = metadata.cropRegion

        val bitmap = RawDemosaicProcessor.getInstance().process(
            context,
            dngPath,
            resolveRawAspectRatio(metadata),
            cropRegion,
            metadata.rotation,
            metadata.exposureBias ?: 0f,
            rawExposureCompensation = metadata.rawExposureCompensation ?: 0f,
            rawAutoExposure = resolveRawAutoExposure(metadata),
            rawHighlightsAdjustment = metadata.rawHighlightsAdjustment ?: 0f,
            rawShadowsAdjustment = metadata.rawShadowsAdjustment ?: 0f,
            rawBlackPointCorrection = metadata.rawBlackPointCorrection ?: 0f,
            rawWhitePointCorrection = metadata.rawWhitePointCorrection ?: 0f,
            rawAutoWhiteBalanceEstimate = resolveRawAutoWhiteBalanceEstimate(metadata),
            rawBlackLevelMode = metadata.rawBlackLevelMode,
            rawCustomBlackLevel = metadata.rawCustomBlackLevel,
            rawWhiteLevelMode = metadata.rawWhiteLevelMode,
            denoiseValue = finalNoiseReduction,
            chromaDenoiseValue = finalChromaNoiseReduction,
            rawDcpId = metadata.rawDcpId,
            rawRenderingEngine = metadata.rawRenderingEngine,
            rawToneMappingParameters = metadata.rawToneMappingParameters,
            rawCfaCorrectionMode = metadata.rawCfaCorrectionMode,
            spectralFilmStock = metadata.spectralFilmStock,
            spectralFilmPrint = metadata.spectralFilmPrint,
            spectralFilmTuning = SpectralFilmTuning(
                cDensityGain = metadata.spectralFilmCDensityGain,
                mDensityGain = metadata.spectralFilmMDensityGain,
                yDensityGain = metadata.spectralFilmYDensityGain
            )
        )

        result = bitmap?.let {
            var b = it

            metadata.computationalAperture?.let { aperture ->
                b = depthBokehProcessor.applyHighQualityBokeh(
                    context, photoId, b,
                    metadata.focusPointX, metadata.focusPointY, aperture
                )
                photoId?.let { photoId -> GalleryManager.saveBokehPhoto(context, photoId, b) }
            }

            lutImageProcessor.applyLutStack(
                b,
                isHlgInput = false,
                colorCorrection.baselineLayer,
                colorCorrection.creativeLayer,
                finalSharpening,
                noiseReductionValue = 0f,
                chromaNoiseReductionValue = 0f
            )
        }

        result ?: return@withContext null

        result = applyCrop(result, metadata, "dng")
        result = applyFrame(result, metadata)

        result
    }

    /**
     * @param input 输入 ARGB的像素数组
     * @param metadata 照片元数据（包含编辑配置和拍摄信息）
     * @param sharpening 锐化强度
     * @param noiseReduction 降噪强度
     * @param chromaNoiseReduction 减少杂色强度
     * @return 处理后的 Bitmap
     */
    suspend fun processYuv(
        context: Context,
        photoId: String?,
        input: ByteBuffer,
        metadata: MediaMetadata,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f
    ): Bitmap = withContext(Dispatchers.IO) {

        // 优先从元数据中获取软件处理参数
        // 智能回退：如果是导入的照片且元数据中没存过，则默认值为 0，不应用额外处理
        val finalSharpening = metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening)
        val finalNoiseReduction = metadata.noiseReduction ?: (if (metadata.isImported) 0f else noiseReduction)
        val finalChromaNoiseReduction =
            metadata.chromaNoiseReduction ?: (if (metadata.isImported) 0f else chromaNoiseReduction)

        val colorCorrection = resolveColorCorrection(
            metadata = metadata,
            fallbackTarget = BaselineColorCorrectionTarget.JPG
        )

        // 1. 应用 LUT
        var result = lutImageProcessor.applyLutStack(
            input.asShortBuffer(),
            metadata.width,
            metadata.height,
            ColorSpace.get(metadata.colorSpace),
            isHlgInput = shouldDecodeHlgInput(metadata),
            colorCorrection.baselineLayer,
            colorCorrection.creativeLayer,
            finalSharpening,
            finalNoiseReduction,
            finalChromaNoiseReduction,
            linearInputToneMap = metadata.usesLinearPipelineToneMap(),
            linearInputExposureEv = metadata.rawExposureCompensation ?: 0f,
            naturalLightDefaultChromaDenoise = metadata.shouldApplyNaturalLightDefaultChromaDenoise(),
            rawRenderingEngine = metadata.rawRenderingEngine,
            rawToneMappingParameters = metadata.rawToneMappingParameters
        )
        YuvProcessor.free(input)

        metadata.computationalAperture?.let { aperture ->
            result = depthBokehProcessor.applyHighQualityBokeh(
                context, photoId, result,
                metadata.focusPointX, metadata.focusPointY, aperture
            )
        }

        result = applyCrop(result, metadata, "yuv")
        result = applyFrame(result, metadata)

        result
    }

    suspend fun processCapturePreviewToneMap(
        input: Bitmap,
        metadata: MediaMetadata,
    ): Bitmap = withContext(Dispatchers.IO) {
        if (!metadata.usesLinearPipelineToneMap()) return@withContext input
        if (shouldDecodeHlgInput(metadata)) return@withContext input
        lutImageProcessor.applyLut(
            bitmap = input,
            isHlgInput = false,
            lutConfig = null,
            colorRecipeParams = null,
            linearInputToneMap = true,
            linearInputExposureEv = metadata.rawExposureCompensation ?: 0f,
            rawRenderingEngine = metadata.rawRenderingEngine,
            rawToneMappingParameters = metadata.rawToneMappingParameters
        )
    }

    /**
     * @param input 输入 Bitmap
     * @param metadata 照片元数据（包含编辑配置和拍摄信息）
     * @param sharpening 锐化强度
     * @param noiseReduction 降噪强度
     * @param chromaNoiseReduction 减少杂色强度
     * @return 处理后的 Bitmap
     */
    suspend fun processBitmap(
        context: Context,
        photoId: String?,
        input: Bitmap,
        metadata: MediaMetadata,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f,
        useComputationalAperture: Boolean = false,
        applyFrameWatermark: Boolean = true,
    ): Bitmap = withContext(Dispatchers.IO) {
        var result = input
        val finalSharpening = metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening)
        val finalNoiseReduction = metadata.noiseReduction ?: (if (metadata.isImported) 0f else noiseReduction)
        val finalChromaNoiseReduction =
            metadata.chromaNoiseReduction ?: (if (metadata.isImported) 0f else chromaNoiseReduction)

        val colorCorrection = resolveColorCorrection(
            metadata = metadata,
            fallbackTarget = BaselineColorCorrectionTarget.JPG
        )

        if (useComputationalAperture) {
            metadata.computationalAperture?.let { aperture ->
                result = depthBokehProcessor.applyHighQualityBokeh(
                    context, photoId, result,
                    metadata.focusPointX, metadata.focusPointY, aperture
                )
                photoId?.let { photoId -> GalleryManager.saveBokehPhoto(context, photoId, result) }
            }
        }

        // 1. 应用 LUT
        result = lutImageProcessor.applyLutStack(
            result,
            isHlgInput = shouldDecodeHlgInput(metadata),
            colorCorrection.baselineLayer,
            colorCorrection.creativeLayer,
            finalSharpening,
            finalNoiseReduction,
            finalChromaNoiseReduction
        )

        result = applyCrop(result, metadata, "bitmap")
        if (applyFrameWatermark) {
            result = applyFrame(result, metadata)
        }

        result
    }

    suspend fun applyFrameForHdrOutput(
        input: Bitmap,
        metadata: MediaMetadata,
        gainmapResult: GainmapResult?,
    ): HdrFrameOutput = withContext(Dispatchers.IO) {
        val resolvedFrame = resolveFrame(input, metadata) ?: return@withContext HdrFrameOutput(input, gainmapResult)
        val framedBitmap = frameRenderer.render(input, resolvedFrame.template, resolvedFrame.metadata)
        if (framedBitmap === input) {
            return@withContext HdrFrameOutput(input, gainmapResult)
        }

        val framedGainmapResult = frameGainmapResult(
            input = input,
            template = resolvedFrame.template,
            gainmapResult = gainmapResult
        )
        HdrFrameOutput(framedBitmap, framedGainmapResult)
    }


    private fun applyCrop(input: Bitmap, metadata: MediaMetadata, label: String = "bitmap"): Bitmap {
        val cropRegion = metadata.postCropRegion ?: return input
        if (cropRegion.width() <= 0 || cropRegion.height() <= 0) return input

        val sourceWidth = metadata.width.takeIf { it > 0 } ?: input.width
        val sourceHeight = metadata.height.takeIf { it > 0 } ?: input.height
        val mappedCropRegion = mapPostCropRegionToInput(cropRegion, sourceWidth, sourceHeight, input.width, input.height)
        if (mappedCropRegion.isEmpty) return input

        val safeLeft = mappedCropRegion.left
        val safeTop = mappedCropRegion.top
        val safeRight = mappedCropRegion.right
        val safeBottom = mappedCropRegion.bottom

        val safeWidth = safeRight - safeLeft
        val safeHeight = safeBottom - safeTop

        if (safeWidth <= 0 || safeHeight <= 0 || (safeWidth == input.width && safeHeight == input.height)) {
            return input
        }

        val cropped = Bitmap.createBitmap(input, safeLeft, safeTop, safeWidth, safeHeight)
        if (input != cropped && !input.isRecycled) {
            // NOTE: processYuv and processBitmap assign 'result', so we can recycle the old one if it is not the original source
            // Wait, input might be the original bitmap passed to processBitmap?
            // If it is the original, we should NOT recycle it because it may still be needed/managed outside.
        }
        return cropped
    }

    private fun mapPostCropRegionToInput(
        cropRegion: android.graphics.Rect,
        sourceWidth: Int,
        sourceHeight: Int,
        inputWidth: Int,
        inputHeight: Int
    ): android.graphics.Rect {
        val scaleX = inputWidth.toFloat() / sourceWidth.toFloat()
        val scaleY = inputHeight.toFloat() / sourceHeight.toFloat()

        return android.graphics.Rect(
            (cropRegion.left * scaleX).roundToInt().coerceIn(0, inputWidth),
            (cropRegion.top * scaleY).roundToInt().coerceIn(0, inputHeight),
            (cropRegion.right * scaleX).roundToInt().coerceIn(0, inputWidth),
            (cropRegion.bottom * scaleY).roundToInt().coerceIn(0, inputHeight)
        )
    }

    private fun resolveRawAspectRatio(metadata: MediaMetadata): AspectRatio? {
        val storedRatio = metadata.ratio ?: return null
        metadata.postCropRegion ?: return null
        val metadataAspect = metadata.width.takeIf { it > 0 }?.let { width ->
            metadata.height.takeIf { it > 0 }?.let { height ->
                width.toFloat() / height.toFloat()
            }
        } ?: return storedRatio
        val metadataIsLandscape = metadata.width >= metadata.height
        val storedAspect = storedRatio.getValue(metadataIsLandscape)

        if (abs(storedAspect - metadataAspect) <= 0.01f) {
            return storedRatio
        }

        return AspectRatio.entries.minBy { ratio ->
            abs(ratio.getValue(metadataIsLandscape) - metadataAspect)
        }
    }

    private suspend fun resolveColorCorrection(
        metadata: MediaMetadata,
        fallbackTarget: BaselineColorCorrectionTarget
    ) = colorCorrectionPipelineResolver.resolveFromMetadata(
        fallbackTarget = fallbackTarget,
        metadata = metadata
    )

    private suspend fun applyFrame(
        input: Bitmap,
        metadata: MediaMetadata,
    ): Bitmap {
        val resolvedFrame = resolveFrame(input, metadata) ?: return input
        return frameRenderer.render(input, resolvedFrame.template, resolvedFrame.metadata)
    }

    private suspend fun resolveFrame(
        input: Bitmap,
        metadata: MediaMetadata,
    ): ResolvedFrame? {
        val frameId = metadata.frameId ?: return null
        val template = frameManager.loadTemplate(frameId) ?: return null
        val customProperties = frameManager.loadCustomProperties(frameId)
        val finalMetadata = metadata.copy(
            deviceModel = metadata.deviceModel ?: Build.MODEL,
            brand = metadata.brand ?: Build.MANUFACTURER.replaceFirstChar { it.uppercase() },
            dateTaken = metadata.dateTaken ?: System.currentTimeMillis(),
            width = if (metadata.width > 0) metadata.width else input.width,
            height = if (metadata.height > 0) metadata.height else input.height,
            customProperties = metadata.customProperties.ifEmpty { customProperties }
        )
        return ResolvedFrame(template, finalMetadata)
    }

    private fun frameGainmapResult(
        input: Bitmap,
        template: FrameTemplate,
        gainmapResult: GainmapResult?,
    ): GainmapResult? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE || gainmapResult == null) {
            return gainmapResult
        }

        val sourceGainmap = gainmapResult.gainmap
        val sourceContents = sourceGainmap.getGainmapContents()
        val framedContents = frameRenderer.renderGainmapContents(input, sourceContents, template)
        if (framedContents === sourceContents) {
            return gainmapResult
        }

        return gainmapResult.copy(
            gainmap = copyGainmapWithContents(sourceGainmap, framedContents)
        )
    }

    private fun copyGainmapWithContents(source: Gainmap, contents: Bitmap): Gainmap {
        val copy = Gainmap(contents)
        source.getRatioMin().also { copy.setRatioMin(it[0], it[1], it[2]) }
        source.getRatioMax().also { copy.setRatioMax(it[0], it[1], it[2]) }
        source.getGamma().also { copy.setGamma(it[0], it[1], it[2]) }
        source.getEpsilonSdr().also { copy.setEpsilonSdr(it[0], it[1], it[2]) }
        source.getEpsilonHdr().also { copy.setEpsilonHdr(it[0], it[1], it[2]) }
        copy.setMinDisplayRatioForHdrTransition(source.getMinDisplayRatioForHdrTransition())
        copy.setDisplayRatioForFullHdr(source.getDisplayRatioForFullHdr())
        return copy
    }
}
