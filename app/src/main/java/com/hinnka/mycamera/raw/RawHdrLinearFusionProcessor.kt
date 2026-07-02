package com.hinnka.mycamera.raw

import android.graphics.Bitmap
import android.hardware.camera2.CaptureResult
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.model.SafeImage
import com.hinnka.mycamera.processor.GlesRawStacker
import com.hinnka.mycamera.utils.LargeDirectBuffer
import com.hinnka.mycamera.utils.PLog
import java.nio.ByteBuffer
import java.nio.ByteOrder

object RawHdrLinearFusionProcessor {
    private const val TAG = "RawHdrLinearFusion"
    private const val MIN_EXPOSURE_PRODUCT = 1.0e-9

    data class InputFrame(
        val image: SafeImage,
        val captureResult: CaptureResult,
        val exposureProduct: Double,
        val originalIndex: Int,
    )

    data class Result(
        val linearRgbBuffer: ByteBuffer,
        val width: Int,
        val height: Int,
        val rowStepSamples: Int,
        val colStepSamples: Int,
        val baselineExposureEv: Float,
        val shortExposureProduct: Double,
        val normalExposureProduct: Double,
        val longExposureProduct: Double,
        val previewBitmap: Bitmap?,
    )

    suspend fun process(
        frames: List<InputFrame>,
        metadata: RawMetadata,
        applyLensShadingCorrection: Boolean,
        aspectRatio: AspectRatio,
        rotation: Int,
    ): Result? {
        val bracket = selectBracketFrames(frames) ?: return null
        val short = bracket.first()
        val normalFrames = bracket.drop(1).dropLast(1)
        val long = bracket.last()
        var stackedNormalBuffer: ByteBuffer? = null
        return try {
            val preparedFrames = buildList {
                add(short.toLinearInputFrame())
                add(prepareNormalInputFrame(normalFrames, metadata, applyLensShadingCorrection).also { prepared ->
                    if (normalFrames.size > 1) stackedNormalBuffer = prepared.rawData
                })
                add(long.toLinearInputFrame())
            }
            val processorResult = RawDemosaicProcessor.getInstance().fuseLinearRcdHdrFrames(
                frames = preparedFrames,
                metadata = metadata,
                applyLensShadingCorrection = applyLensShadingCorrection,
                aspectRatio = aspectRatio,
                rotation = rotation,
                renderPreview = false,
            ) ?: return null
            Result(
                linearRgbBuffer = processorResult.linearRgbBuffer,
                width = processorResult.width,
                height = processorResult.height,
                rowStepSamples = processorResult.rowStepSamples,
                colStepSamples = processorResult.colStepSamples,
                baselineExposureEv = processorResult.baselineExposureEv,
                shortExposureProduct = processorResult.shortExposureProduct,
                normalExposureProduct = processorResult.normalExposureProduct,
                longExposureProduct = processorResult.longExposureProduct,
                previewBitmap = processorResult.previewBitmap,
            )
        } finally {
            LargeDirectBuffer.free(stackedNormalBuffer)
        }
    }

    private fun InputFrame.toLinearInputFrame(): LinearHdrFusionInputFrame {
        val plane = image.planes[0]
        val rawBuffer = plane.buffer.duplicate().order(ByteOrder.nativeOrder())
        rawBuffer.position(0)
        return LinearHdrFusionInputFrame(
            rawData = rawBuffer,
            width = image.width,
            height = image.height,
            rowStride = plane.rowStride,
            exposureProduct = exposureProduct,
            originalIndex = originalIndex,
        )
    }

    private fun prepareNormalInputFrame(
        normalFrames: List<InputFrame>,
        metadata: RawMetadata,
        applyLensShadingCorrection: Boolean,
    ): LinearHdrFusionInputFrame {
        if (normalFrames.size == 1) {
            return normalFrames.first().toLinearInputFrame()
        }
        val first = normalFrames.first()
        val stacker = GlesRawStacker(
            width = first.image.width,
            height = first.image.height,
            cfaPattern = metadata.cfaPattern,
            blackLevel = metadata.blackLevel,
            whiteLevel = metadata.whiteLevel.toInt(),
            noiseModel = metadata.noiseProfile,
            lensShading = metadata.lensShadingMap.takeIf { applyLensShadingCorrection },
            lensShadingWidth = if (applyLensShadingCorrection) metadata.lensShadingMapWidth else 0,
            lensShadingHeight = if (applyLensShadingCorrection) metadata.lensShadingMapHeight else 0,
        )
        val stackResult = stacker.process(normalFrames.map { it.image })
            ?: throw IllegalStateException("Failed to stack RAW HDR normal frames")
        val fusedNormalBuffer = stackResult.fusedBayerBuffer
            ?: throw IllegalStateException("RAW HDR normal stack did not return a Bayer buffer")
        fusedNormalBuffer.position(0)
        PLog.d(
            TAG,
            "RAW HDR normal 0EV stack prepared before RCD: frames=${normalFrames.size} " +
                "reference=${first.originalIndex}:${first.exposureProduct}"
        )
        return LinearHdrFusionInputFrame(
            rawData = fusedNormalBuffer,
            width = stackResult.width,
            height = stackResult.height,
            rowStride = stackResult.width * 2,
            exposureProduct = first.exposureProduct,
            originalIndex = first.originalIndex,
            blackLevel = floatArrayOf(0f, 0f, 0f, 0f),
            whiteLevel = 65535f,
            lensShadingAlreadyApplied = applyLensShadingCorrection && metadata.lensShadingMap != null,
            effectiveFrameCount = normalFrames.size,
        )
    }

    private fun selectBracketFrames(frames: List<InputFrame>): List<InputFrame>? {
        if (frames.size < 3) {
            PLog.w(TAG, "RAW HDR linear fusion requires 3 frames, got ${frames.size}")
            return null
        }
        val sorted = frames.sortedWith(
            compareBy<InputFrame> { validExposureProduct(it.exposureProduct) }
                .thenBy { it.originalIndex }
        )
        val short = sorted.first()
        val long = sorted.last()
        val normalFrames = sorted.drop(1).dropLast(1).ifEmpty {
            listOf(sorted[sorted.size / 2])
        }
        PLog.d(
            TAG,
            "RAW HDR bracket roles: short=${short.originalIndex}:${short.exposureProduct} " +
                "normal=${normalFrames.joinToString { "${it.originalIndex}:${it.exposureProduct}" }} " +
                "long=${long.originalIndex}:${long.exposureProduct}"
        )
        return buildList {
            add(short)
            addAll(normalFrames)
            add(long)
        }
    }

    private fun validExposureProduct(product: Double): Double =
        product.takeIf { it.isFinite() && it > MIN_EXPOSURE_PRODUCT } ?: MIN_EXPOSURE_PRODUCT
}
