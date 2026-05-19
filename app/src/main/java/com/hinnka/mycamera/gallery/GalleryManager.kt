package com.hinnka.mycamera.gallery

import android.app.PendingIntent
import android.content.ContentValues
import android.content.Context
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.graphics.createBitmap
import androidx.exifinterface.media.ExifInterface
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.data.ContentRepository
import com.hinnka.mycamera.gallery.db.GalleryMediaStore
import com.hinnka.mycamera.hdr.GainmapResult
import com.hinnka.mycamera.hdr.GainmapSourceSet
import com.hinnka.mycamera.hdr.HdrGainmapStrength
import com.hinnka.mycamera.hdr.SourceKind
import com.hinnka.mycamera.hdr.UltraHdrWriter
import com.hinnka.mycamera.hdr.UnifiedGainmapProducer
import com.hinnka.mycamera.livephoto.MotionPhotoWriter
import com.hinnka.mycamera.model.SafeImage
import com.hinnka.mycamera.processor.MultiFrameStacker
import com.hinnka.mycamera.raw.RawDemosaicProcessor
import com.hinnka.mycamera.raw.RawMetadata
import com.hinnka.mycamera.raw.RawProcessingPreferences
import com.hinnka.mycamera.utils.BitmapUtils
import com.hinnka.mycamera.utils.DngBlackLevelPatcher
import com.hinnka.mycamera.utils.PLog
import com.hinnka.mycamera.utils.RawProcessor
import com.hinnka.mycamera.utils.YuvProcessor
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import kotlin.io.copyTo
import kotlin.io.deleteRecursively
import kotlin.io.extension
import kotlin.io.inputStream
import kotlin.io.readBytes
import kotlin.io.walkBottomUp
import kotlin.math.roundToInt
import kotlin.system.measureTimeMillis
import kotlin.use

/**
 * 照片管理器
 *
 * 统一管理照片文件、元数据、缩略图等
 * 存储路径: context.filesDir/photos/<photoId>/
 */
object GalleryManager {
    private const val TAG = "PhotoManager"
    private val metadataMutex = Mutex()
    private const val THUMBNAIL_MAX_EDGE = 512
    private const val PHOTOS_DIR = "photos"
    private const val BURST_DIR = "burst"
    private const val PHOTO_FILE = "original.jpg"
    private const val YUV_FILE = "original.jxl"
    private const val HDR_FILE = "original_hdr.bin"
    private const val VIDEO_FILE = "video.mp4"
    private const val DNG_FILE = "original.dng"
    private const val AI_DENOISE_FILE = "ai_denoise.jpg"
    private const val THUMBNAIL_FILE = "thumbnail.jpg"
    private const val BOKEH_FILE = "bokeh.jpg"
    private const val DETAIL_HDR_FILE = "detail_hdr.jpg"
    private const val MULTIPLE_EXPOSURE_DIR = "multiple_exposure_sessions"
    private const val MULTIPLE_EXPOSURE_PREVIEW_FILE = "preview.jpg"

    data class VideoRecordInfo(
        val uri: Uri,
        val displayName: String,
        val dateTaken: Long,
        val size: Long,
        val width: Int,
        val height: Int,
        val durationMs: Long,
        val mimeType: String?,
        val frameRate: Int?,
        val bitrate: Long?,
        val rotationDegrees: Int?,
        val hasAudio: Boolean?
    )

    data class HdrSidecarData(
        val buffer: ByteBuffer,
        val width: Int,
        val height: Int,
        val compressed: Boolean
    )

    val processingScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val gainmapProducer = UnifiedGainmapProducer()

    @Volatile
    var hdrSdrRatio: Float = 0f
    private val detailHdrBuildJobs = ConcurrentHashMap<String, Job>()
    private val _detailHdrReadyEvents = MutableSharedFlow<String>(extraBufferCapacity = 16)
    val detailHdrReadyEvents: SharedFlow<String> = _detailHdrReadyEvents.asSharedFlow()

    private suspend fun resolveRawAutoWhiteBalanceEstimate(
        context: Context,
        metadata: MediaMetadata?
    ): Boolean {
        return metadata?.rawAutoWhiteBalanceEstimate
            ?: (ContentRepository.getInstance(context).userPreferencesRepository.userPreferences.firstOrNull()
                ?.rawAutoWhiteBalanceEstimate ?: false)
    }

    private suspend fun resolveRawAutoExposure(
        context: Context,
        metadata: MediaMetadata?
    ): Boolean {
        return metadata?.rawAutoExposure
            ?: (ContentRepository.getInstance(context).userPreferencesRepository.userPreferences.firstOrNull()
                ?.rawAutoExposure ?: true)
    }
    private val _photoLibraryChangedEvents = MutableSharedFlow<Unit>(extraBufferCapacity = 16)
    val photoLibraryChangedEvents: SharedFlow<Unit> = _photoLibraryChangedEvents.asSharedFlow()
    private val hdrWorkLock = Any()
    private val hdrWorkCounts = ConcurrentHashMap<String, Int>()

    fun notifyPhotoLibraryChanged() {
        _photoLibraryChangedEvents.tryEmit(Unit)
    }

    private fun getPhotosBaseDir(context: Context): File {
        return File(context.getExternalFilesDir(Environment.DIRECTORY_PICTURES), PHOTOS_DIR)
    }

    private fun getPhotoDir(context: Context, photoId: String, create: Boolean = false): File {
        val dir = File(getPhotosBaseDir(context), photoId)
        if (create && !dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    private fun getMultipleExposureBaseDir(context: Context): File {
        return File(context.cacheDir, MULTIPLE_EXPOSURE_DIR)
    }

    private fun getMultipleExposureSessionDir(context: Context, sessionId: String, create: Boolean = false): File {
        val dir = File(getMultipleExposureBaseDir(context), sessionId)
        if (create && !dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    private fun getMultipleExposurePreviewFile(context: Context, sessionId: String): File {
        return File(getMultipleExposureSessionDir(context, sessionId, true), MULTIPLE_EXPOSURE_PREVIEW_FILE)
    }

    fun getPhotoFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), PHOTO_FILE)
    }

    fun getYuvFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), YUV_FILE)
    }

    fun getHdrFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), HDR_FILE)
    }

    fun getDngFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), DNG_FILE)
    }

    fun getAiDenoiseFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), AI_DENOISE_FILE)
    }

    fun getDepthFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), "depthmap.png")
    }

    fun getThumbnailFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), THUMBNAIL_FILE)
    }

    fun getBokehFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), BOKEH_FILE)
    }

    fun getDetailHdrFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), DETAIL_HDR_FILE)
    }

    private fun beginHdrWork(photoId: String) {
        synchronized(hdrWorkLock) {
            hdrWorkCounts[photoId] = (hdrWorkCounts[photoId] ?: 0) + 1
        }
    }

    private fun endHdrWork(photoId: String) {
        synchronized(hdrWorkLock) {
            val nextCount = (hdrWorkCounts[photoId] ?: 1) - 1
            if (nextCount > 0) {
                hdrWorkCounts[photoId] = nextCount
            } else {
                hdrWorkCounts.remove(photoId)
            }
        }
    }

    fun isHdrWorkInFlight(photoId: String): Boolean {
        return (hdrWorkCounts[photoId] ?: 0) > 0
    }

    private suspend fun awaitDetailHdrBuildIdle(photoId: String) {
        val existingJob = detailHdrBuildJobs[photoId] ?: return
        if (!existingJob.isActive) return
        PLog.d(TAG, "Waiting for detail HDR build before RAW refresh: $photoId")
        existingJob.join()
    }

    fun deleteDetailHdrFile(context: Context, photoId: String) {
        val detailFile = getDetailHdrFile(context, photoId)
        val photoFile = getPhotoFile(context, photoId)
        val photoDir = getPhotoDir(context, photoId)
        val stableTimestamp = photoFile.takeIf { it.exists() }?.lastModified()
        if (detailFile.exists()) {
            detailFile.delete()
        }
        stableTimestamp?.let { photoDir.setLastModified(it) }
    }

    fun getVideoFile(context: Context, photoId: String): File {
        return File(getPhotoDir(context, photoId), VIDEO_FILE)
    }

    private fun getExistingMediaFile(context: Context, photoId: String): File? {
        val photoFile = getPhotoFile(context, photoId)
        if (photoFile.exists()) return photoFile
        val videoFile = getVideoFile(context, photoId)
        if (videoFile.exists()) return videoFile
        return null
    }

    private fun readVideoRecordInfo(context: Context, uri: Uri): VideoRecordInfo? {
        return try {
            val projection = arrayOf(
                MediaStore.Video.Media.DISPLAY_NAME,
                MediaStore.Video.Media.DATE_ADDED,
                MediaStore.Video.Media.SIZE,
                MediaStore.Video.Media.WIDTH,
                MediaStore.Video.Media.HEIGHT,
                MediaStore.Video.Media.DURATION,
                MediaStore.Video.Media.MIME_TYPE
            )
            context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
                if (!cursor.moveToFirst()) return@use null
                val displayName = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DISPLAY_NAME))
                val dateTaken = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATE_ADDED)) * 1000L
                val size = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.SIZE))
                val width = cursor.getInt(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.WIDTH))
                val height = cursor.getInt(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.HEIGHT))
                val durationMs = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DURATION))
                val mimeType = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Video.Media.MIME_TYPE))

                var frameRate: Int? = null
                var bitrate: Long? = null
                var rotationDegrees: Int? = null
                var hasAudio: Boolean? = null
                var resolvedWidth = width
                var resolvedHeight = height
                var resolvedDurationMs = durationMs

                val retriever = MediaMetadataRetriever()
                try {
                    retriever.setDataSource(context, uri)
                    resolvedWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                        ?.toIntOrNull()
                        ?.takeIf { it > 0 } ?: resolvedWidth
                    resolvedHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                        ?.toIntOrNull()
                        ?.takeIf { it > 0 } ?: resolvedHeight
                    resolvedDurationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull()
                        ?.takeIf { it > 0L } ?: resolvedDurationMs
                    frameRate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                        ?.toFloatOrNull()?.roundToInt()
                    if (frameRate == null) {
                        val frameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                            ?.toLongOrNull()
                        if (frameCount != null && resolvedDurationMs > 0L) {
                            frameRate = ((frameCount * 1000f) / resolvedDurationMs).roundToInt().takeIf { it > 0 }
                        }
                    }
                    bitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toLongOrNull()
                    rotationDegrees = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull()
                    hasAudio = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_AUDIO)?.let {
                        it == "yes" || it == "true" || it == "1"
                    }
                } catch (e: Exception) {
                    PLog.w(TAG, "Video retriever metadata unavailable for $uri: ${e.message}")
                } finally {
                    retriever.release()
                }

                VideoRecordInfo(
                    uri = uri,
                    displayName = displayName,
                    dateTaken = dateTaken,
                    size = size,
                    width = resolvedWidth,
                    height = resolvedHeight,
                    durationMs = resolvedDurationMs,
                    mimeType = mimeType,
                    frameRate = frameRate,
                    bitrate = bitrate,
                    rotationDegrees = rotationDegrees,
                    hasAudio = hasAudio
                )
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to read video record info: $uri", e)
            null
        }
    }

    private fun saveVideoThumbnail(context: Context, uri: Uri, outputFile: File): Boolean {
        return try {
            val retriever = MediaMetadataRetriever()
            val bitmap = try {
                retriever.setDataSource(context, uri)
                retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            } finally {
                retriever.release()
            } ?: return false

            val thumbnail = createScaledThumbnail(bitmap, THUMBNAIL_MAX_EDGE)

            outputFile.parentFile?.mkdirs()
            FileOutputStream(outputFile).use { out ->
                thumbnail.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
            if (thumbnail != bitmap) {
                bitmap.recycle()
            }
            thumbnail.recycle()
            true
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to save video thumbnail for $uri", e)
            false
        }
    }

    private fun createScaledThumbnail(bitmap: Bitmap, maxEdge: Int): Bitmap {
        val largestEdge = maxOf(bitmap.width, bitmap.height)
        if (largestEdge <= maxEdge) {
            return bitmap
        }

        val scale = maxEdge.toFloat() / largestEdge.toFloat()
        val targetWidth = (bitmap.width * scale).roundToInt().coerceAtLeast(1)
        val targetHeight = (bitmap.height * scale).roundToInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }

    private fun hasBitmapGainmap(bitmap: Bitmap?): Boolean {
        if (bitmap == null || Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) return false
        return try {
            val hasGainmap = bitmap.javaClass.getMethod("hasGainmap")
            hasGainmap.invoke(bitmap) as? Boolean ?: false
        } catch (_: Throwable) {
            false
        }
    }

    private fun canReuseEmbeddedGainmap(metadata: MediaMetadata): Boolean {
        return metadata.manualHdrEffectEnabled &&
                !metadata.hasAiDenoisedBase &&
                metadata.hasEmbeddedGainmap &&
                HdrGainmapStrength.coerce(metadata.hdrEffectStrength) == HdrGainmapStrength.DEFAULT &&
                metadata.lutId == null &&
                metadata.colorRecipeParams == null &&
                metadata.sharpening == null &&
                metadata.noiseReduction == null &&
                metadata.chromaNoiseReduction == null &&
                metadata.frameId == null &&
                metadata.cropRegion == null &&
                metadata.postCropRegion == null &&
                metadata.computationalAperture == null
    }

    private fun writeFinalJpeg(
        bitmap: Bitmap,
        outputStream: FileOutputStream,
        quality: Int,
        gainmapResult: GainmapResult? = null,
    ): Boolean {
        var success = false
        val elapsed = measureTimeMillis {
            success = UltraHdrWriter.writeJpeg(
                UltraHdrWriter.Request(
                    bitmap = bitmap,
                    outputStream = outputStream,
                    quality = quality,
                    gainmap = gainmapResult?.gainmap,
                )
            )
        }
        PLog.d(
            TAG,
            "writeFinalJpeg took ${elapsed}ms, size=${bitmap.width}x${bitmap.height}, gainmap=${gainmapResult != null}, success=$success"
        )
        return success
    }

    suspend fun buildDetailHdrCache(
        context: Context,
        photoId: String,
        metadata: MediaMetadata? = null,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f,
        quality: Int = 92,
        preparedUltraHdrSource: GainmapSourceSet? = null,
        preparedGainmapResult: GainmapResult? = null,
    ): Boolean = withContext(Dispatchers.IO) {
        beginHdrWork(photoId)
        try {
            val resolvedMetadata = metadata ?: loadMetadata(context, photoId) ?: return@withContext false
            if (!resolvedMetadata.manualHdrEffectEnabled) {
                deleteDetailHdrFile(context, photoId)
                return@withContext false
            }
            val photoFile = getPhotoFile(context, photoId)
            val photoDir = getPhotoDir(context, photoId)
            val stableTimestamp = photoFile.takeIf { it.exists() }?.lastModified()
            val detailFile = getDetailHdrFile(context, photoId)
            val tempFile = File(detailFile.parentFile, "detail_hdr_temp.jpg")

            val photoProcessor = ContentRepository.getInstance(context).photoProcessor
            val ultraHdrSource = preparedUltraHdrSource ?: photoProcessor.prepareUltraHdrSource(
                context = context,
                photoId = photoId,
                metadata = resolvedMetadata,
                sharpening = sharpening,
                noiseReduction = noiseReduction,
                chromaNoiseReduction = chromaNoiseReduction
            )
            if (preparedUltraHdrSource != null) {
                PLog.d(TAG, "buildDetailHdrCache reusing prepared HDR source for $photoId")
            }

            if (ultraHdrSource == null) {
                if (detailFile.exists()) {
                    detailFile.delete()
                }
                stableTimestamp?.let { photoDir.setLastModified(it) }
                return@withContext false
            }

            val gainmapResult = preparedGainmapResult ?: gainmapProducer.build(
                ultraHdrSource,
                HdrGainmapStrength.coerce(resolvedMetadata.hdrEffectStrength)
            )
            if (preparedGainmapResult != null) {
                PLog.d(TAG, "buildDetailHdrCache reused prepared gainmap for $photoId")
            }
            FileOutputStream(tempFile).use { outputStream ->
                if (!writeFinalJpeg(ultraHdrSource.sdrBase, outputStream, quality, gainmapResult)) {
                    tempFile.delete()
                    return@withContext false
                }
            }

            if (detailFile.exists()) {
                detailFile.delete()
            }
            tempFile.renameTo(detailFile)
            stableTimestamp?.let { photoDir.setLastModified(it) }
            _detailHdrReadyEvents.tryEmit(photoId)
            PLog.d(
                TAG,
                "buildDetailHdrCache success: $photoId, source=${ultraHdrSource.sourceKind}, gainmap=${gainmapResult != null}"
            )
            true
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to build detail HDR cache for $photoId", e)
            false
        } finally {
            endHdrWork(photoId)
        }
    }

    fun queueDetailHdrCacheBuild(
        context: Context,
        photoId: String,
        metadata: MediaMetadata? = null,
        sharpening: Float = 0f,
        noiseReduction: Float = 0f,
        chromaNoiseReduction: Float = 0f
    ) {
        val existingJob = detailHdrBuildJobs[photoId]
        val job = processingScope.launch {
            try {
                existingJob?.join()
                buildDetailHdrCache(
                    context = context,
                    photoId = photoId,
                    metadata = metadata,
                    sharpening = sharpening,
                    noiseReduction = noiseReduction,
                    chromaNoiseReduction = chromaNoiseReduction
                )
            } finally {
                detailHdrBuildJobs.remove(photoId, coroutineContext[Job])
            }
        }
        detailHdrBuildJobs[photoId] = job
    }

    suspend fun exportPhoto(
        context: Context,
        id: String,
        bitmap: Bitmap? = null,
        photoProcessor: PhotoProcessor,
        metadata: MediaMetadata,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95,
        suffix: String? = null,
        preparedUltraHdrSource: GainmapSourceSet? = null,
        preparedGainmapResult: GainmapResult? = null,
    ): Boolean {
        return withContext(Dispatchers.IO) {
            val tempExportFile = File(context.cacheDir, "temp_export_${System.nanoTime()}.jpg")
            try {
                if (bitmap == null && canReuseEmbeddedGainmap(metadata)) {
                    val embeddedBitmap = loadOriginalBitmap(context, id)
                    if (embeddedBitmap != null && hasBitmapGainmap(embeddedBitmap)) {
                        PLog.d(TAG, "Reusing embedded gainmap for export: $id")
                        return@withContext exportBitmapToMediaStore(
                            context = context,
                            id = id,
                            bitmap = embeddedBitmap,
                            metadata = metadata,
                            photoQuality = photoQuality,
                            suffix = suffix
                        )
                    }
                }

                var ultraHdrSource = preparedUltraHdrSource
                if (ultraHdrSource == null) {
                    val ultraHdrPrepareElapsed = measureTimeMillis {
                        ultraHdrSource = photoProcessor.prepareUltraHdrSource(
                            context = context,
                            photoId = id,
                            metadata = metadata,
                            sharpening = sharpeningValue,
                            noiseReduction = noiseReductionValue,
                            chromaNoiseReduction = chromaNoiseReductionValue
                        )
                    }
                    PLog.d(TAG, "prepareUltraHdrSource took ${ultraHdrPrepareElapsed}ms")
                } else {
                    PLog.d(TAG, "prepareUltraHdrSource reused in-memory source for export: $id")
                }
                var gainmapResult: GainmapResult? = preparedGainmapResult
                if (preparedGainmapResult == null) {
                    val gainmapElapsed = measureTimeMillis {
                        gainmapResult = ultraHdrSource?.let {
                            gainmapProducer.build(it, HdrGainmapStrength.coerce(metadata.hdrEffectStrength))
                        }
                    }
                    PLog.d(TAG, "gainmapProducer.build took ${gainmapElapsed}ms, enabled=${gainmapResult != null}")
                } else {
                    PLog.d(TAG, "gainmapProducer.build reused prepared result, enabled=true")
                }

                // 读取照片
                val sourceBitmap = ultraHdrSource
                    ?.takeUnless { it.sourceKind == SourceKind.SDR_BITMAP && metadata.hasEmbeddedGainmap }
                    ?.sdrBase
                val processedBitmap = (sourceBitmap ?: if (metadata.hasAiDenoisedBase) {
                    photoProcessor.process(
                        context, id, metadata,
                        sharpeningValue, noiseReductionValue, chromaNoiseReductionValue
                    )
                } else bitmap?.let {
                    photoProcessor.processBitmap(
                        context, id, bitmap, metadata,
                        sharpeningValue, noiseReductionValue, chromaNoiseReductionValue,
                        true
                    )
                } ?: photoProcessor.process(
                    context, id, metadata,
                    sharpeningValue, noiseReductionValue, chromaNoiseReductionValue
                )) ?: return@withContext false

                PLog.d(
                    TAG,
                    "processedBitmap = ${processedBitmap.colorSpace?.name}, ultraHdrSource=${ultraHdrSource?.sourceKind}, gainmap=${gainmapResult != null}"
                )

                // 保存到指定目录
                val date = metadata.dateTaken ?: System.currentTimeMillis()

                val lutName =
                    metadata.lutId?.let { ContentRepository.getInstance(context).lutManager.getLutInfo(it)?.getName() }
                var withSuffix = suffix?.let { "_$it" } ?: ""
                lutName?.let {
                    withSuffix += ".$lutName"
                }

                val filename =
                    "PhotonCamera_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(date))}$withSuffix.jpg"
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DCIM + "/PhotonCamera")
                }

                val uri = context.contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    contentValues
                )

                uri?.let {
                    val videoFile = File(getPhotoDir(context, id), VIDEO_FILE)
                    val isLivePhoto = videoFile.exists()

                    val exportWriteElapsed = measureTimeMillis {
                        FileOutputStream(tempExportFile).use { outputStream ->
                            writeFinalJpeg(
                                bitmap = processedBitmap,
                                outputStream = outputStream,
                                quality = photoQuality,
                                gainmapResult = if (isLivePhoto) null else gainmapResult
                            )
                        }
                    }
                    PLog.d(TAG, "exportPhoto writeFinalJpeg wrapper took ${exportWriteElapsed}ms")

                    ExifWriter.writeExif(
                        tempExportFile, metadata.toCaptureInfo().copy(
                            imageWidth = processedBitmap.width,
                            imageHeight = processedBitmap.height
                        )
                    )

                    if (isLivePhoto) {
                        val tempMotionPhotoFile = File(context.cacheDir, "temp_motion_${System.nanoTime()}.jpg")
                        try {
                            PLog.d(
                                TAG,
                                "Attempting to create Motion Photo for export: JPEG=${tempExportFile.length()}, Video=${videoFile.length()}"
                            )

                            // 重新从磁盘加载最新元数据，以获取可能刚写回的 presentationTimestampUs
                            val latestMetadata = loadMetadata(context, id) ?: metadata
                            val success = MotionPhotoWriter.write(
                                tempExportFile.absolutePath,
                                videoFile.absolutePath,
                                tempMotionPhotoFile.absolutePath,
                                latestMetadata.presentationTimestampUs ?: 0L,
                                context
                            )

                            PLog.d(TAG, "MotionPhotoWriter result: $success")

                            context.contentResolver.openOutputStream(uri)?.use { outputStream ->
                                if (success) {
                                    tempMotionPhotoFile.inputStream().use { input -> input.copyTo(outputStream) }
                                    PLog.d(
                                        TAG,
                                        "Exported Live Photo successfully: ${tempMotionPhotoFile.length()} bytes"
                                    )
                                } else {
                                    // Fallback to normal JPEG (with EXIF)
                                    PLog.w(TAG, "Motion Photo synthesis failed, falling back to JPEG")
                                    tempExportFile.inputStream().use { input -> input.copyTo(outputStream) }
                                }
                            }

                            if (Build.MANUFACTURER.lowercase().contains("vivo")) {
                                val filename = filename.replace(".jpg", ".mp4")
                                val contentValues = ContentValues().apply {
                                    put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                                    put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                                    put(
                                        MediaStore.MediaColumns.RELATIVE_PATH,
                                        Environment.DIRECTORY_DCIM + "/PhotonCamera"
                                    )
                                }

                                val uri = context.contentResolver.insert(
                                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                                    contentValues
                                )
                                uri?.let { uri -> context.contentResolver.openOutputStream(uri) }?.use { outputStream ->
                                    val tempMotionVideoFile =
                                        File(tempMotionPhotoFile.absolutePath.replace(".jpg", ".mp4"))
                                    if (tempMotionVideoFile.exists()) {
                                        tempMotionVideoFile.inputStream().use { input -> input.copyTo(outputStream) }
                                    }
                                    tempMotionVideoFile.delete()

                                    updateMetadata(context, id) { current ->
                                        current.copy(
                                            exportedUris = current.exportedUris + uri.toString()
                                        )
                                    }
                                }
                            }
                        } finally {
                            tempMotionPhotoFile.delete()
                        }
                    } else {
                        // 3b. Normal Export: Copy Temp File (with EXIF) to MediaStore
                        context.contentResolver.openOutputStream(uri)?.use { outputStream ->
                            tempExportFile.inputStream().use { input -> input.copyTo(outputStream) }
                        }
                    }

                    // Save exported URI to metadata
                    updateMetadata(context, id) { current ->
                        current.copy(
                            exportedUris = current.exportedUris + uri.toString()
                        )
                    }
                    PLog.d(TAG, "Exported URI saved: $uri for photo $id")

                    return@withContext true
                }

                processedBitmap.recycle()
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to export photo", e)
            } finally {
                tempExportFile.delete()
            }

            false
        }
    }

    private suspend fun exportBitmapToMediaStore(
        context: Context,
        id: String,
        bitmap: Bitmap,
        metadata: MediaMetadata,
        photoQuality: Int,
        suffix: String?,
    ): Boolean {
        val tempExportFile = File(context.cacheDir, "temp_export_${System.nanoTime()}.jpg")
        try {
            val date = metadata.dateTaken ?: System.currentTimeMillis()
            val lutName =
                metadata.lutId?.let { ContentRepository.getInstance(context).lutManager.getLutInfo(it)?.getName() }
            var withSuffix = suffix?.let { "_$it" } ?: ""
            lutName?.let { withSuffix += ".$it" }
            val filename =
                "PhotonCamera_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(date))}$withSuffix.jpg"
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DCIM + "/PhotonCamera")
            }
            val uri = context.contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            ) ?: return false

            FileOutputStream(tempExportFile).use { outputStream ->
                writeFinalJpeg(bitmap, outputStream, photoQuality)
            }
            ExifWriter.writeExif(
                tempExportFile, metadata.toCaptureInfo().copy(
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height
                )
            )
            context.contentResolver.openOutputStream(uri)?.use { output ->
                tempExportFile.inputStream().use { input -> input.copyTo(output) }
            }

            updateMetadata(context, id) { current ->
                current.copy(
                    exportedUris = current.exportedUris + uri.toString()
                )
            }
            PLog.d(TAG, "Exported embedded-gainmap URI saved: $uri for photo $id")
            return true
        } finally {
            tempExportFile.delete()
        }
    }

    suspend fun exportDng(context: Context, photoId: String, data: ByteArray, metadata: MediaMetadata) =
        withContext(Dispatchers.IO) {
            try {
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                    .format(Date())
                val dngFilename = "PhotonCamera_${timestamp}.dng"
                val dngContentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, dngFilename)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng")
                    put(
                        MediaStore.MediaColumns.RELATIVE_PATH,
                        Environment.DIRECTORY_DCIM + "/PhotonCamera"
                    )
                }

                val dngUri = context.contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    dngContentValues
                )

                dngUri?.let { uri ->
                    context.contentResolver.openOutputStream(uri)?.use { outputStream ->
                        outputStream.write(data)
                    }
                    PLog.d(TAG, "DNG exported: $uri")

                    updateMetadata(context, photoId) { current ->
                        current.copy(
                            exportedUris = current.exportedUris + uri.toString()
                        )
                    }
                    PLog.d(TAG, "Exported URI saved: $uri")
                }
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to export DNG", e)
            }
        }

    suspend fun exportDng(context: Context, photoId: String, sourceFile: File, metadata: MediaMetadata) =
        withContext(Dispatchers.IO) {
            try {
                if (!sourceFile.exists() || sourceFile.length() <= 0L) {
                    PLog.w(TAG, "Skipping DNG export because source file is missing or empty: ${sourceFile.absolutePath}")
                    return@withContext
                }

                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                    .format(Date())
                val dngFilename = "PhotonCamera_${timestamp}.dng"
                val dngContentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, dngFilename)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng")
                    put(
                        MediaStore.MediaColumns.RELATIVE_PATH,
                        Environment.DIRECTORY_DCIM + "/PhotonCamera"
                    )
                }

                val dngUri = context.contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    dngContentValues
                )

                dngUri?.let { uri ->
                    context.contentResolver.openOutputStream(uri)?.use { outputStream ->
                        sourceFile.inputStream().use { inputStream ->
                            inputStream.copyTo(outputStream)
                        }
                    }
                    PLog.d(TAG, "DNG exported from file: $uri")

                    updateMetadata(context, photoId) { current ->
                        current.copy(
                            exportedUris = current.exportedUris + uri.toString()
                        )
                    }
                    PLog.d(TAG, "Exported URI saved: $uri")
                }
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to export DNG", e)
            }
        }

    suspend fun preparePhoto(
        context: Context,
        metadata: MediaMetadata,
        captureResult: CaptureResult?,
        thumbnail: Bitmap?,
        useLivePhoto: Boolean,
        superResolutionScale: Float,
        includeCropRegionInOutputSize: Boolean = true,
        photoId: String? = null,
    ) = withContext(Dispatchers.IO) {
        try {
            val photoId = photoId ?: UUID.randomUUID().toString()
            val photoDir = getPhotoDir(context, photoId, true)
            val photoFile = File(photoDir, PHOTO_FILE)
            val videoFile = File(photoDir, VIDEO_FILE)
            val thumbnailFile = File(photoDir, THUMBNAIL_FILE)

            var cropRegion = captureResult?.get(CaptureResult.SCALER_CROP_REGION)
            if (superResolutionScale > 1.0f && cropRegion != null) {
                cropRegion = Rect(
                    (cropRegion.left * superResolutionScale).roundToInt(),
                    (cropRegion.top * superResolutionScale).roundToInt(),
                    (cropRegion.right * superResolutionScale).roundToInt(),
                    (cropRegion.bottom * superResolutionScale).roundToInt()
                )
            }
            if (cropRegion != null && !includeCropRegionInOutputSize) {
                PLog.d(TAG, "Ignoring SCALER_CROP_REGION for output sizing: $cropRegion")
            }
            val effectiveCropRegion = cropRegion?.takeIf { includeCropRegionInOutputSize }

            val dimensions =
                BitmapUtils.calculateProcessedRect(
                    metadata.width,
                    metadata.height,
                    metadata.ratio,
                    effectiveCropRegion,
                    metadata.rotation
                )
            val finalWidth = dimensions.width()
            val finalHeight = dimensions.height()
            // 保存元数据
            val metadataWithInfo = metadata.copy(
                width = finalWidth,
                height = finalHeight,
                cropRegion = effectiveCropRegion,
            )
            saveMetadata(context, photoId, metadataWithInfo)

            if (thumbnail != null && !thumbnail.isRecycled) {
                generateThumbnail(thumbnail, thumbnailFile)
            } else {
                PLog.d(TAG, "Thumbnail unavailable: $thumbnail")
            }
            photoFile.createNewFile()
            if (useLivePhoto) {
                videoFile.createNewFile()
            }
            photoId
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to prepare photo", e)
            null
        }
    }

    suspend fun saveVideo(
        context: Context,
        photoId: String,
        livePhotoVideoDeferred: Deferred<Pair<File, Long>?>? = null
    ) {
        val photoDir = getPhotoDir(context, photoId, true)
        val videoFile = File(photoDir, VIDEO_FILE)
        val livePhotoResult = livePhotoVideoDeferred?.await()
        livePhotoResult?.first?.let { cacheVideoFile ->
            if (cacheVideoFile.exists()) {
                try {
                    cacheVideoFile.copyTo(videoFile, overwrite = true)
                    cacheVideoFile.delete()

                    // 更新元数据以包含时间戳
                    val currentMeta = loadMetadata(context, photoId) ?: return
                    saveMetadata(context, photoId, currentMeta.copy(presentationTimestampUs = livePhotoResult.second))
                } catch (e: Exception) {
                    PLog.e(TAG, "Failed to move video file", e)
                }
                PLog.d(TAG, "Motion Photo synthesized for $photoId with TS: ${livePhotoResult.second}")
            }
        }
    }

    suspend fun saveBokehPhoto(context: Context, photoId: String, bitmap: Bitmap) = withContext(Dispatchers.IO) {
        val photoDir = getPhotoDir(context, photoId, true)
        val bokehFile = File(photoDir, BOKEH_FILE)
        FileOutputStream(bokehFile).use { outputStream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
        }
    }

    suspend fun generateBokehPhoto(context: Context, photoId: String, metadata: MediaMetadata, bitmap: Bitmap) {
        val aperture = metadata.computationalAperture ?: 0f
        if (aperture <= 0f) {
            getBokehFile(context, photoId).takeIf { it.exists() }?.delete()
            return
        }
        val focusPointX = metadata.focusPointX
        val focusPointY = metadata.focusPointY
        val bokeh = ContentRepository.getInstance(context).depthBokehProcessor.applyHighQualityBokeh(
            context,
            photoId,
            bitmap,
            focusPointX,
            focusPointY,
            aperture
        )
        saveBokehPhoto(context, photoId, bokeh)
    }

    suspend fun saveYuvPhoto(
        context: Context,
        photoId: String,
        image: SafeImage,
        rotation: Int,
        aspectRatio: AspectRatio,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95
    ) = withContext(Dispatchers.IO) {
        beginHdrWork(photoId)
        try {
            val photoDir = getPhotoDir(context, photoId, true)

            // 预先准备所有文件路径
            val photoFile = File(photoDir, PHOTO_FILE)
            val tempFile = File(photoDir, "temp.jpg")
            val yuvFile = File(photoDir, YUV_FILE)
            val hdrFile = File(photoDir, HDR_FILE)

            val metadata = loadMetadata(context, photoId) ?: return@withContext

            // 创建预览用的 Bitmap
            var previewBitmap =
                createBitmap(metadata.width, metadata.height, colorSpace = ColorSpace.get(metadata.colorSpace))

            PLog.d(TAG, "saveYuvPhoto: ${metadata.width} ${metadata.height} ${metadata.colorSpace.name}")

            // YUV 格式：使用 native 处理（包含旋转和裁切）并直接保存为 FP16 JXL
            var success = false
            val nativeSaveElapsed = measureTimeMillis {
                success = image.use {
                    YuvProcessor.processAndSave16(
                        image, aspectRatio, rotation,
                        yuvFile.absolutePath,
                        hdrSidecarPath = if (metadata.dynamicRangeProfile == "HLG10") hdrFile.absolutePath else null,
                        previewBitmap = previewBitmap
                    )
                }
            }
            PLog.d(TAG, "saveYuvPhoto processAndSave16 took ${nativeSaveElapsed}ms, success=$success")

            if (metadata.isMirrored) {
                previewBitmap = BitmapUtils.flipHorizontal(previewBitmap)
            }

            if (success) {
                FileOutputStream(tempFile).use { outputStream ->
                    writeFinalJpeg(previewBitmap, outputStream, photoQuality)
                }
                tempFile.renameTo(photoFile)
                generateBokehPhoto(context, photoId, metadata, previewBitmap)
                queueDetailHdrCacheBuild(
                    context = context,
                    photoId = photoId,
                    metadata = metadata,
                    sharpening = sharpeningValue,
                    noiseReduction = noiseReductionValue,
                    chromaNoiseReduction = chromaNoiseReductionValue
                )
//                updateThumbnail(context, photoId, photoProcessor, metadata)
                if (shouldAutoSave) {
                    exportPhoto(
                        context,
                        photoId,
                        null,
                        photoProcessor,
                        metadata,
                        sharpeningValue,
                        noiseReductionValue,
                        chromaNoiseReductionValue,
                        photoQuality
                    )
                }
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to savePhoto", e)
        } finally {
            endHdrWork(photoId)
        }
    }

    suspend fun saveRawPhoto(
        context: Context,
        photoId: String,
        image: SafeImage,
        thumbnail: Bitmap?,
        rotation: Int,
        aspectRatio: AspectRatio,
        characteristics: CameraCharacteristics,
        captureResult: CaptureResult?,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95,
        exposureBias: Float? = null,
        droMode: RawProcessingPreferences.DROMode = RawProcessingPreferences.DROMode.OFF,
        exportDngWithRawExport: Boolean = false
    ) = withContext(Dispatchers.IO) {
        try {
            val photoDir = getPhotoDir(context, photoId, true)

            // 预先准备所有文件路径
            val photoFile = File(photoDir, PHOTO_FILE)
            val dngFile = File(photoDir, DNG_FILE)
            val tempDngFile = File(photoDir, "temp.dng")
            val tempFile = File(photoDir, "temp.jpg")

            val metadata = loadMetadata(context, photoId)
            if (metadata == null) {
                PLog.e(TAG, "saveRawPhoto aborted: metadata unavailable for $photoId")
                image.close()
                return@withContext
            }

            val resolvedCaptureResult = captureResult
            if (resolvedCaptureResult == null) {
                PLog.e(TAG, "saveRawPhoto aborted: captureResult unavailable for $photoId")
                image.close()
                return@withContext
            }

            var dngSaveAttempted = false
            FileOutputStream(tempDngFile).use { outputStream ->
                image.use {
                    try {
                        RawProcessor.saveToDng(
                            image,
                            characteristics,
                            resolvedCaptureResult,
                            outputStream,
                            rotation,
                            thumbnail
                        )
                        dngSaveAttempted = true
                    } catch (e: Throwable) {
                        PLog.e(TAG, "DNG save failed", e)
                    }
                }
            }
            val dngWritten = dngSaveAttempted && tempDngFile.exists() && tempDngFile.length() > 0L
            if (!dngWritten) {
                tempDngFile.delete()
                return@withContext
            }
            patchSavedDngBlackLevel(tempDngFile, metadata)
            if (dngFile.exists()) {
                dngFile.delete()
            }
            if (!tempDngFile.renameTo(dngFile)) {
                tempDngFile.copyTo(dngFile, overwrite = true)
                tempDngFile.delete()
            }
            if (shouldAutoSave && exportDngWithRawExport) {
                exportDng(context, photoId, dngFile, metadata)
            }

            val rawResult = RawDemosaicProcessor.getInstance().processForHdrSources(
                context,
                dngFile.absolutePath,
                aspectRatio = aspectRatio,
                cropRegion = metadata.cropRegion,
                rotation = rotation,
                exposureBias = exposureBias ?: 0f,
                rawExposureCompensation = metadata.rawExposureCompensation ?: 0f,
                rawAutoExposure = resolveRawAutoExposure(context, metadata),
                rawBlackPointCorrection = metadata.rawBlackPointCorrection ?: 0f,
                rawWhitePointCorrection = metadata.rawWhitePointCorrection ?: 0f,
                rawAutoWhiteBalanceEstimate = resolveRawAutoWhiteBalanceEstimate(context, metadata),
                rawDROMode = droMode,
                sharpeningValue = 0.4f,
                denoiseValue = metadata.rawDenoiseValue,
                rawDcpId = metadata.rawDcpId
            ) ?: return@withContext
            var bitmap = rawResult.sdrBitmap

            if (metadata.isMirrored) {
                bitmap = BitmapUtils.flipHorizontal(bitmap)
            }

            FileOutputStream(tempFile).use { outputStream ->
                writeFinalJpeg(bitmap, outputStream, photoQuality)
            }
            tempFile.renameTo(photoFile)
            generateBokehPhoto(context, photoId, metadata, bitmap)
            val preparedUltraHdrSource = if (metadata.manualHdrEffectEnabled) {
                photoProcessor.prepareUltraHdrSourceFromRawResult(
                    context = context,
                    photoId = photoId,
                    rawResult = rawResult,
                    metadata = metadata,
                    sharpening = sharpeningValue,
                    noiseReduction = noiseReductionValue,
                    chromaNoiseReduction = chromaNoiseReductionValue,
                    applyMirror = true
                )
            } else {
                null
            }
            val preparedGainmapResult = preparedUltraHdrSource?.let { source ->
                var result: GainmapResult? = null
                val gainmapElapsed = measureTimeMillis {
                    result = gainmapProducer.build(source, HdrGainmapStrength.coerce(metadata.hdrEffectStrength))
                }
                PLog.d(TAG, "saveRawPhoto prepared gainmap for reuse, took=${gainmapElapsed}ms")
                result
            }
            preparedUltraHdrSource?.let {
                PLog.d(TAG, "saveRawPhoto building detail HDR from in-memory RAW result: $photoId")
                buildDetailHdrCache(
                    context = context,
                    photoId = photoId,
                    metadata = metadata,
                    sharpening = sharpeningValue,
                    noiseReduction = noiseReductionValue,
                    chromaNoiseReduction = chromaNoiseReductionValue,
                    preparedUltraHdrSource = it,
                    preparedGainmapResult = preparedGainmapResult
                )
            }
            updateThumbnail(context, photoId, photoProcessor, metadata, bitmap)
            if (shouldAutoSave) {
                exportPhoto(
                    context,
                    photoId,
                    bitmap,
                    photoProcessor,
                    metadata,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality,
                    preparedUltraHdrSource = preparedUltraHdrSource,
                    preparedGainmapResult = preparedGainmapResult
                )
            }
            preparedUltraHdrSource?.hdrReference?.bitmap?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }
            preparedUltraHdrSource?.sdrBase?.let {
                if (it !== bitmap && !it.isRecycled) {
                    it.recycle()
                }
            }
            if (!bitmap.isRecycled) {
                bitmap.recycle()
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to savePhoto", e)
        }
    }

    /**
     * 保存新拍摄的照片
     *
     * @param context Context
     * @param image 原始 Image (YUV420 或 RAW_SENSOR)
     * @param metadata 编辑元数据（LUT、边框等）
     */
    suspend fun savePhoto(
        context: Context,
        photoId: String,
        image: SafeImage,
        thumbnail: Bitmap? = null,
        rotation: Int,
        aspectRatio: AspectRatio,
        characteristics: CameraCharacteristics,
        captureResult: CaptureResult?,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95,
        exposureBias: Float? = null,
        droMode: RawProcessingPreferences.DROMode = RawProcessingPreferences.DROMode.OFF,
        exportDngWithRawExport: Boolean = false
    ) {
        // 根据图像格式处理
        when (val format = image.format) {
            ImageFormat.YUV_420_888, ImageFormat.YCBCR_P010, ImageFormat.NV21 -> {
                saveYuvPhoto(
                    context,
                    photoId,
                    image,
                    rotation,
                    aspectRatio,
                    shouldAutoSave,
                    photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality
                )
            }

            ImageFormat.RAW_SENSOR, ImageFormat.RAW10, ImageFormat.RAW12 -> {
                saveRawPhoto(
                    context,
                    photoId,
                    image,
                    thumbnail,
                    rotation,
                    aspectRatio,
                    characteristics,
                    captureResult,
                    shouldAutoSave,
                    photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality,
                    exposureBias,
                    droMode,
                    exportDngWithRawExport
                )
            }

            else -> {
                PLog.e(TAG, "Unsupported image format: $format")
            }
        }
    }

    suspend fun saveBitmapPhoto(
        context: Context,
        photoId: String,
        bitmap: Bitmap,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95
    ) = withContext(Dispatchers.IO) {
        try {
            val photoDir = getPhotoDir(context, photoId, true)
            val photoFile = File(photoDir, PHOTO_FILE)
            val tempFile = File(photoDir, "temp.jpg")
            val metadata = loadMetadata(context, photoId) ?: return@withContext

            FileOutputStream(tempFile).use { outputStream ->
                writeFinalJpeg(bitmap, outputStream, photoQuality)
            }
            tempFile.renameTo(photoFile)
//            generateBokehPhoto(context, photoId, metadata, bitmap)
            queueDetailHdrCacheBuild(
                context = context,
                photoId = photoId,
                metadata = metadata,
                sharpening = sharpeningValue,
                noiseReduction = noiseReductionValue,
                chromaNoiseReduction = chromaNoiseReductionValue
            )
//            updateThumbnail(context, photoId, photoProcessor, metadata)

            if (shouldAutoSave) {
                exportPhoto(
                    context,
                    photoId,
                    bitmap,
                    photoProcessor,
                    metadata,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality
                )
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to save bitmap photo", e)
        }
    }

    suspend fun saveMultipleExposureFrame(
        context: Context,
        sessionId: String,
        frameIndex: Int,
        image: SafeImage,
        rotation: Int,
        aspectRatio: AspectRatio,
        shouldMirror: Boolean,
        photoQuality: Int = 95
    ): File? = withContext(Dispatchers.IO) {
        try {
            val sessionDir = getMultipleExposureSessionDir(context, sessionId, true)
            val frameFile = File(sessionDir, String.format(Locale.US, "frame_%02d.jpg", frameIndex))
            val previewBitmap = image.use {
                YuvProcessor.processAndToBitmap(it.image, aspectRatio, rotation)
            }
            val finalBitmap = if (shouldMirror) {
                BitmapUtils.flipHorizontal(previewBitmap)
            } else {
                previewBitmap
            }
            FileOutputStream(frameFile).use { outputStream ->
                finalBitmap.compress(Bitmap.CompressFormat.JPEG, photoQuality, outputStream)
            }
            if (finalBitmap !== previewBitmap) {
                previewBitmap.recycle()
            }
            finalBitmap.recycle()
            frameFile
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to save multiple exposure frame", e)
            null
        }
    }

    fun getMultipleExposureFrameFiles(context: Context, sessionId: String): List<File> {
        val dir = getMultipleExposureSessionDir(context, sessionId)
        return dir.listFiles { file ->
            file.isFile && file.name.startsWith("frame_") && file.extension.equals(
                "jpg",
                ignoreCase = true
            )
        }
            ?.sortedBy { it.name }
            ?: emptyList()
    }

    suspend fun composeMultipleExposurePreview(
        context: Context,
        sessionId: String,
        maxEdge: Int = 1440,
        photoQuality: Int = 85
    ): Bitmap? = withContext(Dispatchers.IO) {
        val frameFiles = getMultipleExposureFrameFiles(context, sessionId)
        val result = composeAverageBitmap(frameFiles, maxEdge) ?: return@withContext null
        try {
            FileOutputStream(getMultipleExposurePreviewFile(context, sessionId)).use { outputStream ->
                result.compress(Bitmap.CompressFormat.JPEG, photoQuality, outputStream)
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to save multiple exposure preview", e)
        }
        result
    }

    suspend fun composeMultipleExposurePhoto(
        context: Context,
        sessionId: String
    ): Bitmap? = withContext(Dispatchers.IO) {
        composeAverageBitmap(getMultipleExposureFrameFiles(context, sessionId), null)
    }

    fun removeLastMultipleExposureFrame(context: Context, sessionId: String): Boolean {
        val lastFrame = getMultipleExposureFrameFiles(context, sessionId).lastOrNull() ?: return false
        return runCatching { lastFrame.delete() }.getOrDefault(false)
    }

    fun clearMultipleExposureSession(context: Context, sessionId: String) {
        runCatching {
            deleteEmptyDirs(getMultipleExposureSessionDir(context, sessionId))
            getMultipleExposureSessionDir(context, sessionId).deleteRecursively()
        }.onFailure {
            PLog.e(TAG, "Failed to clear multiple exposure session", it)
        }
    }

    private fun composeAverageBitmap(frameFiles: List<File>, maxEdge: Int?): Bitmap? {
        if (frameFiles.isEmpty()) return null

        val options = BitmapFactory.Options().apply {
            if (maxEdge != null) {
                inJustDecodeBounds = true
                BitmapFactory.decodeFile(frameFiles.first().absolutePath, this)
                val largestEdge = maxOf(outWidth, outHeight).coerceAtLeast(1)
                inSampleSize = if (largestEdge > maxEdge) {
                    Integer.highestOneBit((largestEdge / maxEdge).coerceAtLeast(1))
                } else {
                    1
                }
                inJustDecodeBounds = false
            }
        }

        val firstBitmap = BitmapFactory.decodeFile(frameFiles.first().absolutePath, options) ?: return null
        val width = firstBitmap.width
        val height = firstBitmap.height
        val bufferSize = firstBitmap.byteCount
        val outputBuffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())
        val inputBuffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())
        val outputInts = outputBuffer.asIntBuffer()
        val inputInts = inputBuffer.asIntBuffer()
        firstBitmap.copyPixelsToBuffer(outputBuffer)
        firstBitmap.recycle()
        var blendedCount = 1

        frameFiles.drop(1).forEach { file ->
            val bitmap = BitmapFactory.decodeFile(file.absolutePath, options) ?: return@forEach
            if (bitmap.width == width && bitmap.height == height) {
                inputBuffer.clear()
                bitmap.copyPixelsToBuffer(inputBuffer)
                blendAverageInto(outputInts, inputInts, width * height, blendedCount + 1)
                blendedCount++
            } else {
                PLog.w(TAG, "Skipping mismatched multiple exposure frame: ${file.name}")
            }
            bitmap.recycle()
        }

        return createBitmap(width, height, Bitmap.Config.ARGB_8888).also { output ->
            outputBuffer.rewind()
            output.copyPixelsFromBuffer(outputBuffer)
        }
    }

    private fun blendAverageInto(
        outputInts: IntBuffer,
        inputInts: IntBuffer,
        pixelCount: Int,
        nextFrameIndex: Int
    ) {
        outputInts.rewind()
        inputInts.rewind()
        for (i in 0 until pixelCount) {
            val basePixel = outputInts.get(i)
            val newPixel = inputInts.get(i)

            val baseA = basePixel ushr 24 and 0xFF
            val baseR = basePixel ushr 16 and 0xFF
            val baseG = basePixel ushr 8 and 0xFF
            val baseB = basePixel and 0xFF

            val newA = newPixel ushr 24 and 0xFF
            val newR = newPixel ushr 16 and 0xFF
            val newG = newPixel ushr 8 and 0xFF
            val newB = newPixel and 0xFF

            outputInts.put(
                i,
                Color.argb(
                    baseA + (newA - baseA) / nextFrameIndex,
                    baseR + (newR - baseR) / nextFrameIndex,
                    baseG + (newG - baseG) / nextFrameIndex,
                    baseB + (newB - baseB) / nextFrameIndex
                )
            )
        }
        outputInts.rewind()
        inputInts.rewind()
    }

    suspend fun saveYuvStackedPhoto(
        context: Context,
        photoId: String,
        images: List<SafeImage>,
        rotation: Int,
        aspectRatio: AspectRatio,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95,
        useSuperResolution: Boolean = false,
        superResolutionScale: Float = 1.0f,
        useGpuAcceleration: Boolean = true,
    ) = withContext(Dispatchers.IO) {
        try {
            val photoDir = getPhotoDir(context, photoId, true)
            val metadata = loadMetadata(context, photoId) ?: return@withContext

            // 预先准备所有文件路径
            val photoFile = File(photoDir, PHOTO_FILE)
            val tempFile = File(photoDir, "temp.jpg")
            val yuvFile = File(photoDir, YUV_FILE)

            var currentUseSuperResolution = useSuperResolution
            var result = MultiFrameStacker.processBurst(
                images,
                rotation,
                aspectRatio,
                yuvFile.absolutePath,
                currentUseSuperResolution,
                useGpuAcceleration,
                ColorSpace.get(metadata.colorSpace)
            )

            if (result == null && currentUseSuperResolution) {
                PLog.w(TAG, "processBurst failed with SR, retrying without SR")
                currentUseSuperResolution = false
                result = MultiFrameStacker.processBurst(
                    images,
                    rotation,
                    aspectRatio,
                    yuvFile.absolutePath,
                    false,
                    useGpuAcceleration,
                    ColorSpace.get(metadata.colorSpace)
                )
            }

            if (result == null) return@withContext

            if (metadata.isMirrored) {
                result = BitmapUtils.flipHorizontal(result)
            }

            // Save Original (Stacked Result)
            FileOutputStream(tempFile).use { outputStream ->
                writeFinalJpeg(result, outputStream, photoQuality)
            }
            tempFile.renameTo(photoFile)
            generateBokehPhoto(context, photoId, metadata, result)
            // Auto Save
            if (shouldAutoSave) {
                val metadata = loadMetadata(context, photoId) ?: return@withContext
                exportPhoto(
                    context,
                    photoId,
                    result,
                    photoProcessor,
                    metadata,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality
                )
            }
            result.recycle()
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to savePhoto", e)
        }
    }

    suspend fun saveRawStackedPhoto(
        context: Context,
        photoId: String,
        images: List<SafeImage>,
        rotation: Int,
        aspectRatio: AspectRatio,
        characteristics: CameraCharacteristics?,
        captureResult: CaptureResult?,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95,
        useSuperResolution: Boolean = false,
        superResolutionScale: Float = 1.0f,
        useGpuAcceleration: Boolean = true,
        exposureBias: Float? = null,
        exportDngWithRawExport: Boolean = false
    ) = withContext(Dispatchers.IO) {
        try {
            val photoDir = getPhotoDir(context, photoId, true)

            // 预先准备所有文件路径
            val photoFile = File(photoDir, PHOTO_FILE)
            val dngFile = File(photoDir, DNG_FILE)
            val tempFile = File(photoDir, "temp.jpg")
            val yuvFile = File(photoDir, YUV_FILE)

            val metadata = loadMetadata(context, photoId) ?: return@withContext

            characteristics ?: return@withContext
            captureResult ?: return@withContext

            val firstImageWidth = images[0].width
            val firstImageHeight = images[0].height

            val rawMetadata = RawMetadata.create(
                firstImageWidth,
                firstImageHeight,
                characteristics,
                captureResult,
                exposureBias,
                RawDemosaicProcessor.getInstance().getRawColorSpace()
            )

            var currentUseSuperResolution = useSuperResolution
            var rawStackResult = MultiFrameStacker.processBurstRaw(
                images, rawMetadata.cfaPattern,
                currentUseSuperResolution,
                superResolutionScale,
                useGpuAcceleration,
                masterBlackLevel = rawMetadata.blackLevel,
                whiteLevel = rawMetadata.whiteLevel.toInt(),
                whiteBalanceGains = rawMetadata.whiteBalanceGains,
                noiseModel = rawMetadata.noiseProfile,
                lensShading = null,
                lensShadingWidth = 0,
                lensShadingHeight = 0,
            )

            val finalStackResult = rawStackResult ?: return@withContext

            val dngWritten = trySaveStackedRawDng(
                context = context,
                photoId = photoId,
                dngFile = dngFile,
                fusedBayerBuffer = finalStackResult.fusedBayerBuffer ?: return@withContext,
                width = finalStackResult.width,
                height = finalStackResult.height,
                rawMetadata = rawMetadata,
                isNormalizedSensorData = finalStackResult.isNormalizedSensorData,
                characteristics = characteristics,
                captureResult = captureResult,
                rotation = rotation,
                thumbnail = null,
                metadata = metadata,
                shouldAutoSave = shouldAutoSave,
                exportDngWithRawExport = exportDngWithRawExport
            )
            if (!dngWritten) {
                PLog.e(TAG, "Failed to persist stacked RAW DNG before rendering preview")
                return@withContext
            }
            finalStackResult.fusedBayerBuffer = null
            @Suppress("ExplicitGarbageCollectionCall")
            System.gc()

            val rawResult = RawDemosaicProcessor.getInstance().processForHdrSources(
                context,
                dngFile.absolutePath,
                aspectRatio = aspectRatio,
                cropRegion = metadata.cropRegion,
                rotation = rotation,
                exposureBias = exposureBias ?: 0f,
                rawExposureCompensation = metadata.rawExposureCompensation ?: 0f,
                rawAutoExposure = resolveRawAutoExposure(context, metadata),
                rawBlackPointCorrection = metadata.rawBlackPointCorrection ?: 0f,
                rawWhitePointCorrection = metadata.rawWhitePointCorrection ?: 0f,
                rawAutoWhiteBalanceEstimate = resolveRawAutoWhiteBalanceEstimate(context, metadata),
                rawDROMode = RawProcessingPreferences.DROMode.fromPersistedName(
                    metadata.droMode
                ),
                sharpeningValue = 0.4f,
                denoiseValue = metadata.rawDenoiseValue,
                rawDcpId = metadata.rawDcpId
            ) ?: return@withContext
            var bitmap = rawResult.sdrBitmap

            if (metadata.isMirrored) {
                bitmap = BitmapUtils.flipHorizontal(bitmap)
            }

            // Save Original (Stacked Result)
            FileOutputStream(tempFile).use { outputStream ->
                writeFinalJpeg(bitmap, outputStream, photoQuality)
            }
            tempFile.renameTo(photoFile)
            generateBokehPhoto(context, photoId, metadata, bitmap)

            val preparedUltraHdrSource = if (metadata.manualHdrEffectEnabled) {
                photoProcessor.prepareUltraHdrSourceFromRawResult(
                    context = context,
                    photoId = photoId,
                    rawResult = rawResult,
                    metadata = metadata,
                    sharpening = sharpeningValue,
                    noiseReduction = noiseReductionValue,
                    chromaNoiseReduction = chromaNoiseReductionValue,
                    applyMirror = true
                )
            } else {
                null
            }
            val preparedGainmapResult = preparedUltraHdrSource?.let { source ->
                var result: GainmapResult? = null
                val gainmapElapsed = measureTimeMillis {
                    result = gainmapProducer.build(source, HdrGainmapStrength.coerce(metadata.hdrEffectStrength))
                }
                PLog.d(TAG, "saveRawStackedPhoto prepared gainmap for reuse, took=${gainmapElapsed}ms")
                result
            }
            preparedUltraHdrSource?.let {
                PLog.d(TAG, "saveRawStackedPhoto building detail HDR from in-memory RAW result: $photoId")
                buildDetailHdrCache(
                    context = context,
                    photoId = photoId,
                    metadata = metadata,
                    sharpening = sharpeningValue,
                    noiseReduction = noiseReductionValue,
                    chromaNoiseReduction = chromaNoiseReductionValue,
                    preparedUltraHdrSource = it,
                    preparedGainmapResult = preparedGainmapResult
                )
            }

            updateThumbnail(context, photoId, photoProcessor, metadata, bitmap)
            // Auto Save
            if (shouldAutoSave) {
                exportPhoto(
                    context,
                    photoId,
                    bitmap,
                    photoProcessor,
                    metadata,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality,
                    preparedUltraHdrSource = preparedUltraHdrSource,
                    preparedGainmapResult = preparedGainmapResult
                )
            }
            preparedUltraHdrSource?.hdrReference?.bitmap?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }
            preparedUltraHdrSource?.sdrBase?.let {
                if (it !== bitmap && !it.isRecycled) {
                    it.recycle()
                }
            }
            if (!bitmap.isRecycled) {
                bitmap.recycle()
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to savePhoto", e)
        }
    }

    private suspend fun trySaveStackedRawDng(
        context: Context,
        photoId: String,
        dngFile: File,
        fusedBayerBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rawMetadata: RawMetadata,
        isNormalizedSensorData: Boolean,
        characteristics: CameraCharacteristics,
        captureResult: CaptureResult,
        rotation: Int,
        thumbnail: Bitmap?,
        metadata: MediaMetadata,
        shouldAutoSave: Boolean,
        exportDngWithRawExport: Boolean,
    ): Boolean {
        val tempDngFile = File(dngFile.parentFile, "temp_stacked.dng")
        val dngWritten = try {
            FileOutputStream(tempDngFile).use { outputStream ->
                RawProcessor.saveRawBufferToDng(
                    rawBuffer = fusedBayerBuffer,
                    width = width,
                    height = height,
                    characteristics = characteristics,
                    captureResult = captureResult,
                    outputStream = outputStream,
                    rotation = rotation,
                    thumbnail = thumbnail,
                    cfaPattern = rawMetadata.cfaPattern,
                    blackLevel = rawMetadata.blackLevel,
                    whiteLevel = rawMetadata.whiteLevel.toInt(),
                    valueDomain = if (isNormalizedSensorData) {
                        RawProcessor.RawBufferValueDomain.NORMALIZED_SENSOR_RANGE
                    } else {
                        RawProcessor.RawBufferValueDomain.SENSOR
                    }
                )
            }
        } catch (e: Throwable) {
            PLog.w(TAG, "Failed to build stacked RAW DNG, ignoring", e)
            false
        }

        if (!dngWritten || !tempDngFile.exists() || tempDngFile.length() <= 0L) {
            tempDngFile.delete()
            return false
        }
        patchSavedDngBlackLevel(tempDngFile, metadata)

        try {
            if (dngFile.exists()) {
                dngFile.delete()
            }
            if (!tempDngFile.renameTo(dngFile)) {
                tempDngFile.copyTo(dngFile, overwrite = true)
                tempDngFile.delete()
            }
        } catch (e: Exception) {
            tempDngFile.delete()
            if (dngFile.exists()) {
                dngFile.delete()
            }
            PLog.w(TAG, "Failed to persist stacked RAW DNG, ignoring", e)
            return false
        }

        if (shouldAutoSave && exportDngWithRawExport) {
            exportDng(context, photoId, dngFile, metadata)
        }

        return true
    }

    private fun patchSavedDngBlackLevel(dngFile: File, metadata: MediaMetadata) {
        val patched = DngBlackLevelPatcher.patchFromMode(
            file = dngFile,
            mode = metadata.rawBlackLevelMode,
            customBlackLevel = metadata.rawCustomBlackLevel
        )
        if (patched) {
            PLog.d(TAG, "Applied DNG BlackLevel correction (${metadata.rawBlackLevelMode}) to ${dngFile.name}")
        }
    }


    /**
     * 保存堆栈合成后的照片
     */
    suspend fun saveStackedPhoto(
        context: Context,
        photoId: String,
        images: List<SafeImage>,
        rotation: Int,
        aspectRatio: AspectRatio,
        characteristics: CameraCharacteristics?,
        captureResult: CaptureResult?,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        sharpeningValue: Float,
        noiseReductionValue: Float,
        chromaNoiseReductionValue: Float,
        photoQuality: Int = 95,
        useSuperResolution: Boolean = false,
        superResolutionScale: Float = 1.0f,
        useGpuAcceleration: Boolean = true,
        exposureBias: Float? = null,
        exportDngWithRawExport: Boolean = false
    ) = withContext(Dispatchers.IO) {
        when (val format = images[0].format) {
            ImageFormat.YUV_420_888, ImageFormat.YCBCR_P010, ImageFormat.NV21 -> {
                saveYuvStackedPhoto(
                    context,
                    photoId,
                    images,
                    rotation,
                    aspectRatio,
                    shouldAutoSave,
                    photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality,
                    useSuperResolution,
                    superResolutionScale,
                    useGpuAcceleration
                )
            }

            ImageFormat.RAW_SENSOR, ImageFormat.RAW10, ImageFormat.RAW12 -> {
                saveRawStackedPhoto(
                    context,
                    photoId,
                    images,
                    rotation,
                    aspectRatio,
                    characteristics,
                    captureResult,
                    shouldAutoSave,
                    photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQuality,
                    useSuperResolution,
                    superResolutionScale,
                    useGpuAcceleration,
                    exposureBias,
                    exportDngWithRawExport
                )
            }

            else -> {
                PLog.e(TAG, "Unsupported image format: $format")
                return@withContext null
            }
        }
    }

    /**
     * 保存连拍照片
     */
    suspend fun saveBurstPhoto(
        context: Context,
        photoId: String,
        image: SafeImage,
        shouldAutoSave: Boolean = true,
        photoProcessor: PhotoProcessor,
        photoQuality: Int = 95
    ) {
        val photoDir = getPhotoDir(context, photoId, true)
        val mainPhotoFile = File(photoDir, PHOTO_FILE)
        val burstDir = File(photoDir, BURST_DIR)
        if (!burstDir.exists()) {
            burstDir.mkdirs()
        }
        try {
            val photoFile = File(burstDir, System.currentTimeMillis().toString() + ".jpg")

            val metadata = loadMetadata(context, photoId) ?: return
            val sharpeningValue = metadata.sharpening ?: 0f
            val noiseReductionValue = metadata.noiseReduction ?: 0f
            val chromaNoiseReductionValue = metadata.chromaNoiseReduction ?: 0f

            image.use {
                YuvProcessor.processAndSave(
                    image, metadata.rotation, photoFile.absolutePath
                )
            }
            if (!mainPhotoFile.exists() || mainPhotoFile.length() == 0L) {
                processingScope.launch {
                    photoFile.copyTo(mainPhotoFile, overwrite = true)
                    if (shouldAutoSave) {
                        exportPhoto(
                            context,
                            photoId,
                            null,
                            photoProcessor,
                            metadata,
                            sharpeningValue,
                            noiseReductionValue,
                            chromaNoiseReductionValue,
                            photoQuality
                        )
                    }
                }
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to savePhoto", e)
        }
    }

    /**
     * 获取指定照片的连拍照片文件列表
     */
    fun getBurstPhotos(context: Context, photoId: String): List<File> {
        val burstDir = File(getPhotoDir(context, photoId), BURST_DIR)
        return if (burstDir.exists()) {
            burstDir.listFiles()?.toList()?.sortedBy { it.lastModified() } ?: emptyList()
        } else {
            emptyList()
        }
    }

    /**
     * 将连拍照片设为主图并重新生成缩略图
     */
    suspend fun setMainBurstPhoto(context: Context, photoId: String, burstFile: File): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val photoDir = getPhotoDir(context, photoId, true)
                val mainPhotoFile = File(photoDir, PHOTO_FILE)
                val thumbnailFile = File(photoDir, THUMBNAIL_FILE)

                if (burstFile.exists()) {
                    burstFile.copyTo(mainPhotoFile, overwrite = true)
                    generateThumbnail(burstFile, thumbnailFile)
                    true
                } else {
                    false
                }
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to set main burst photo", e)
                false
            }
        }
    }

    /**
     * 检查是否有连拍照片
     */
    fun hasBurstPhotos(context: Context, photoId: String): Boolean {
        val burstDir = File(getPhotoDir(context, photoId), BURST_DIR)
        return burstDir.exists() && (burstDir.listFiles()?.isNotEmpty() == true)
    }

    /**
     * 生成 512 缩略图
     */
    private suspend fun generateThumbnail(bitmap: Bitmap, targetFile: File) {
        withContext(Dispatchers.IO) {
            try {
                // 生成适合预览和 widget 的小尺寸缩略图
                val thumbnail = createScaledThumbnail(bitmap, THUMBNAIL_MAX_EDGE)
                FileOutputStream(targetFile).use { out ->
                    thumbnail.compress(Bitmap.CompressFormat.JPEG, 80, out)
                }
                if (thumbnail != bitmap) {
                    thumbnail.recycle()
                }
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to generate thumbnail", e)
            }
        }
    }

    /**
     * 生成缩略图
     */
    private fun generateThumbnail(sourceFile: File, targetFile: File) {
        try {
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFile(sourceFile.absolutePath, options)

            // 计算缩放比例，缩略图大小不超过 THUMBNAIL_MAX_EDGE
            val targetSize = THUMBNAIL_MAX_EDGE
            options.inSampleSize = calculateInSampleSize(options, targetSize, targetSize)
            options.inJustDecodeBounds = false

            val bitmap = BitmapFactory.decodeFile(sourceFile.absolutePath, options)

            if (bitmap != null) {
                FileOutputStream(targetFile).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out)
                }
                bitmap.recycle()
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to generate thumbnail", e)
        }
    }

    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        val (height: Int, width: Int) = options.outHeight to options.outWidth
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }

    /**
     * 获取所有照片 ID 列表（按时间降序）
     */
    fun getPhotoIds(context: Context): List<String> {
        return runBlocking {
            GalleryMediaStore.getPhotoIds(context)
        }
    }

    /**
     * 为直接传入的系统 URI 创建删除请求（用于 Android 11+）
     */
    fun createSystemDeleteRequest(context: Context, uri: Uri): PendingIntent? {
        if (uri.scheme != "content") {
            PLog.w(TAG, "createSystemDeleteRequest: URI scheme must be content, but was ${uri.scheme}")
            return null
        }
        return try {
            MediaStore.createDeleteRequest(context.contentResolver, listOf(uri))
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to create system delete request for $uri", e)
            null
        }
    }

    /**
     * 创建删除系统相册照片的请求（弹出确认对话框）
     *
     * 仅适用于 Android 11+ (API 30+)
     * 返回 PendingIntent，需要在 Activity 中通过 startIntentSenderForResult 启动
     */
    fun createDeleteRequest(context: Context, photoId: String): PendingIntent? {
        return try {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
                PLog.w(TAG, "createDeleteRequest requires Android 11+")
                return null
            }

            // 加载元数据，获取导出的 URI 列表（使用 runBlocking 同步调用）
            val metadata = runBlocking {
                loadMetadata(context, photoId)
            }
            val exportedUris = buildList {
                addAll(metadata?.exportedUris ?: emptyList())
                val sourceUri = metadata?.sourceUri
                if (metadata?.mediaType == MediaType.VIDEO && !sourceUri.isNullOrBlank()) {
                    add(sourceUri)
                }
            }

            if (exportedUris.isEmpty()) {
                PLog.d(TAG, "No exported URIs to delete for photo: $photoId")
                return null
            }

            // 将字符串 URI 转换为 Uri 对象列表，并过滤非法 URI
            val uriList = exportedUris.mapNotNull { uriString ->
                try {
                    val uri = Uri.parse(uriString)
                    if (uri.scheme == "content") uri else {
                        PLog.w(TAG, "Ignoring non-content URI: $uriString")
                        null
                    }
                } catch (e: Exception) {
                    PLog.e(TAG, "Invalid URI: $uriString", e)
                    null
                }
            }

            if (uriList.isEmpty()) {
                return null
            }

            // 创建删除请求（会弹出系统确认对话框）
            MediaStore.createDeleteRequest(context.contentResolver, uriList)
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to create delete request for photo: $photoId", e)
            null
        }
    }

    /**
     * 删除照片及其所有相关文件
     */
    suspend fun deletePhoto(
        context: Context,
        photoId: String
    ): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val photoDir = getPhotoDir(context, photoId)
                if (photoDir.exists()) {
                    photoDir.deleteRecursively()
                }
                GalleryMediaStore.deleteMedia(context, photoId)
                PLog.d(TAG, "Photo deleted: $photoId")
                deleteEmptyDirs(getPhotosBaseDir(context))
                true
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to delete photo: $photoId", e)
                false
            }
        }
    }

    fun deleteEmptyDirs(root: File) {
        if (!root.exists() || !root.isDirectory) return
        root.walkBottomUp().forEach { file ->
            if (file.isDirectory && file != root) {
                val contents = file.listFiles()
                if (contents != null && contents.isEmpty()) {
                    val deleted = file.delete()
                    if (deleted) {
                        // PLog.d(TAG, "已清理空文件夹: ${file.absolutePath}")
                    } else {
                        PLog.e(TAG, "无法删除文件夹 (可能权限不足): ${file.absolutePath}")
                    }
                }
            }
        }
    }

    /**
     * 仅删除应用内部的照片，不删除系统相册中的导出照片
     */
    suspend fun deletePhotoOnly(context: Context, photoId: String): Boolean {
        return deletePhoto(context, photoId)
    }

    /**
     * 加载元数据
     */
    suspend fun loadMetadata(context: Context, photoId: String): MediaMetadata? {
        return metadataMutex.withLock {
            loadMetadataInternal(context, photoId)
        }
    }

    private suspend fun loadMetadataInternal(context: Context, photoId: String): MediaMetadata? {
        return GalleryMediaStore.loadMetadata(context, photoId)
    }

    /**
     * 保存元数据
     */
    suspend fun saveMetadata(context: Context, photoId: String, metadata: MediaMetadata): Boolean {
        return metadataMutex.withLock {
            saveMetadataInternal(context, photoId, metadata)
        }
    }

    private suspend fun saveMetadataInternal(context: Context, photoId: String, metadata: MediaMetadata): Boolean {
        getPhotoDir(context, photoId, true)
        return GalleryMediaStore.saveMetadata(context, photoId, metadata)
    }

    /**
     * 原子地更新元数据
     */
    suspend fun updateMetadata(
        context: Context,
        photoId: String,
        update: (MediaMetadata) -> MediaMetadata
    ): MediaMetadata? {
        return metadataMutex.withLock {
            GalleryMediaStore.updateMetadata(context, photoId, update)
        }
    }

    suspend fun recordVideoCapture(
        context: Context,
        uri: Uri,
        photoId: String = UUID.randomUUID().toString()
    ): String? = withContext(Dispatchers.IO) {
        val info = readVideoRecordInfo(context, uri) ?: return@withContext null
        val metadata = MediaMetadata(
            mediaType = MediaType.VIDEO,
            dateTaken = info.dateTaken,
            width = info.width,
            height = info.height,
            sourceUri = uri.toString(),
            mimeType = info.mimeType,
            durationMs = info.durationMs,
            frameRate = info.frameRate,
            bitrate = info.bitrate,
            rotationDegrees = info.rotationDegrees,
            hasAudio = info.hasAudio,
            videoWidth = info.width,
            videoHeight = info.height,
            captureMode = "video"
        )
        val dir = getPhotoDir(context, photoId, true)
        val thumbnailSaved = saveVideoThumbnail(context, uri, getThumbnailFile(context, photoId))
        if (!thumbnailSaved) {
            PLog.w(TAG, "Video thumbnail not generated for $uri")
        }
        val metadataSaved = saveMetadata(context, photoId, metadata)
        if (!metadataSaved) {
            dir.deleteRecursively()
            return@withContext null
        }
        dir.setLastModified(info.dateTaken)
        notifyPhotoLibraryChanged()
        photoId
    }

    fun loadYuvData(context: Context, photoId: String): ByteBuffer? {
        val yuvFile = getYuvFile(context, photoId)
        if (!yuvFile.exists()) {
            return null
        }
        val start = System.currentTimeMillis()
        return YuvProcessor.loadCompressedArgb(yuvFile.absolutePath).also {
            PLog.d(TAG, "loadYuvData took ${System.currentTimeMillis() - start}ms, success=${it != null}")
        }
    }

    fun loadHdrData(
        context: Context,
        photoId: String,
        fallbackWidth: Int,
        fallbackHeight: Int
    ): HdrSidecarData? {
        val hdrFile = getHdrFile(context, photoId)
        if (!hdrFile.exists()) {
            return null
        }

        val start = System.currentTimeMillis()
        val dims = YuvProcessor.getCompressedArgbDimensions(hdrFile.absolutePath)
        if (dims != null && dims.size >= 2) {
            val buffer = YuvProcessor.loadCompressedArgb(hdrFile.absolutePath)
            if (buffer != null) {
                PLog.d(
                    TAG,
                    "loadHdrData loaded compressed sidecar in ${System.currentTimeMillis() - start}ms, " +
                        "size=${dims[0]}x${dims[1]}, bytes=${buffer.capacity()}"
                )
                return HdrSidecarData(
                    buffer = buffer,
                    width = dims[0],
                    height = dims[1],
                    compressed = true
                )
            }
        }

        return try {
            val bytes = hdrFile.readBytes()
            val expectedBytes = fallbackWidth * fallbackHeight * 4 * 2
            if (bytes.size != expectedBytes) {
                PLog.e(
                    TAG,
                    "Failed to load HDR sidecar: unexpected legacy size ${bytes.size}, expected=$expectedBytes"
                )
                null
            } else {
                PLog.d(
                    TAG,
                    "loadHdrData loaded legacy raw sidecar in ${System.currentTimeMillis() - start}ms, " +
                        "size=${fallbackWidth}x${fallbackHeight}, bytes=${bytes.size}"
                )
                HdrSidecarData(
                    buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                        put(bytes)
                        rewind()
                    },
                    width = fallbackWidth,
                    height = fallbackHeight,
                    compressed = false
                )
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to load HDR sidecar", e)
            null
        }
    }

    fun loadBitmap(context: Context, photoId: String, maxEdge: Int? = null, preserveHdr: Boolean = false): Bitmap? {
        val bokehFile = getBokehFile(context, photoId)
        if (bokehFile.exists()) {
            return loadBitmap(context, Uri.fromFile(bokehFile), maxEdge, preserveHdr)
        }
        val photoFile = getPhotoFile(context, photoId)
        if (photoFile.exists()) {
            return loadBitmap(context, Uri.fromFile(photoFile), maxEdge, preserveHdr)
        }
        val thumbnailFile = getThumbnailFile(context, photoId)
        if (thumbnailFile.exists()) {
            return loadBitmap(context, Uri.fromFile(thumbnailFile), maxEdge, preserveHdr)
        }
        return null
    }

    fun loadOriginalBitmap(
        context: Context,
        photoId: String,
        maxEdge: Int? = null,
        preserveHdr: Boolean = false
    ): Bitmap? {
        val photoFile = getPhotoFile(context, photoId)
        if (!photoFile.exists()) {
            return null
        }
        return loadBitmap(context, Uri.fromFile(photoFile), maxEdge, preserveHdr)
    }

    suspend fun updateThumbnail(
        context: Context,
        photoId: String,
        photoProcessor: PhotoProcessor,
        metadata: MediaMetadata? = null,
        inputBitmap: Bitmap? = null
    ) {
        withContext(Dispatchers.IO) {
            try {
                val resolvedMetadata = metadata ?: loadMetadata(context, photoId) ?: return@withContext
                // Use provided bitmap or load from disk if unavailable
                val originalBitmap = inputBitmap?.let { 
                    createScaledThumbnail(it, THUMBNAIL_MAX_EDGE) 
                } ?: loadOriginalBitmap(context, photoId, maxEdge = THUMBNAIL_MAX_EDGE)
                  ?: loadBitmap(context, photoId, maxEdge = THUMBNAIL_MAX_EDGE)
                  ?: return@withContext

                // 应用所有效果（LUT、虚化、裁切、边框等）到缩略图尺寸的位图
                val processedBitmap = photoProcessor.processBitmap(
                    context = context,
                    photoId = photoId,
                    input = originalBitmap,
                    metadata = resolvedMetadata,
                    sharpening = resolvedMetadata.sharpening ?: 0f,
                    noiseReduction = resolvedMetadata.noiseReduction ?: 0f,
                    chromaNoiseReduction = resolvedMetadata.chromaNoiseReduction ?: 0f,
                    useComputationalAperture = true
                )

                val thumbnailFile = getThumbnailFile(context, photoId)
                generateThumbnail(processedBitmap, thumbnailFile)

                if (processedBitmap !== originalBitmap) {
                    processedBitmap.recycle()
                }
                // Only recycle originalBitmap if it was newly loaded or scaled
                if (originalBitmap !== inputBitmap) {
                    originalBitmap.recycle()
                }
                PLog.d(TAG, "Thumbnail updated for photo: $photoId")
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to update thumbnail for photo: $photoId", e)
            }
        }
    }

    fun loadBitmap(context: Context, uri: Uri, maxEdge: Int? = null, preserveHdr: Boolean = false): Bitmap? {
        var infoSize: android.util.Size? = null
        var infoMimeType: String? = null
        val source = ImageDecoder.createSource(context.contentResolver, uri)

        val bitmap = runCatching {
            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                if (preserveHdr) {
                    decoder.allocator = ImageDecoder.ALLOCATOR_HARDWARE
                } else {
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
                // 记录原始信息
                infoSize = info.size
                infoMimeType = info.mimeType

                if (maxEdge != null) {
                    val width = info.size.width
                    val height = info.size.height

                    if (width > maxEdge || height > maxEdge) {
                        val scale = maxEdge.toFloat() / maxOf(width, height)
                        decoder.setTargetSize((width * scale).toInt(), (height * scale).toInt())
                    }
                }
            }
        }.getOrNull() ?: return null
        val isDng = infoMimeType?.contains("dng", ignoreCase = true) == true

        if (!isDng) return bitmap

        return try {
            val orientation = context.contentResolver.openInputStream(uri)?.use { input ->
                ExifInterface(input).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } ?: ExifInterface.ORIENTATION_NORMAL

            // 如果方向正常，直接返回
            if (orientation == ExifInterface.ORIENTATION_NORMAL || orientation == ExifInterface.ORIENTATION_UNDEFINED) {
                return bitmap
            }

            val (infoW, infoH) = infoSize?.let { it.width to it.height } ?: (0 to 0)

            // 准确判断方向是否已被处理：
            // 1. 检查当前方向是否涉及宽高交换
            val rotationSwapsSize = orientation == ExifInterface.ORIENTATION_ROTATE_90 ||
                    orientation == ExifInterface.ORIENTATION_ROTATE_270 ||
                    orientation == ExifInterface.ORIENTATION_TRANSPOSE ||
                    orientation == ExifInterface.ORIENTATION_TRANSVERSE

            val alreadyHandled = if (rotationSwapsSize && infoW != infoH && infoW > 0) {
                // 如果是 90/270 度旋转且非正方形，检查 Bitmap 宽高比是否相对于原图已反转
                // (bitmapW > bitmapH) 不等于 (infoW > infoH) 说明发生了交换，即已被处理
                (bitmap.width > bitmap.height) != (infoW > infoH)
            } else true

            if (alreadyHandled) {
                bitmap
            } else {
                rotateImageIfRequired(bitmap, orientation)
            }
        } catch (e: Exception) {
            bitmap
        }
    }

    private fun detectEmbeddedGainmap(context: Context, photoFile: File): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE || !photoFile.exists()) return false
        // Use a small maxEdge for performance; gainmap detection works on downsampled bitmaps.
        return hasBitmapGainmap(loadBitmap(context, Uri.fromFile(photoFile), maxEdge = 512, preserveHdr = true))
    }


    /**
     * 从 URI 获取文件名
     */
    private fun getFileName(context: Context, uri: Uri): String? {
        return try {
            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                cursor.moveToFirst()
                cursor.getString(nameIndex)
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * 从系统相册导入照片
     */
    suspend fun importPhoto(
        context: Context,
        uri: Uri,
        lutId: String?,
        computationalAperture: Float? = null,
        photoId: String? = null,
    ): String? {
        return withContext(Dispatchers.IO) {
            try {
                val photoId = photoId ?: UUID.randomUUID().toString()
                val photoDir = getPhotoDir(context, photoId, true)
                val photoFile = File(photoDir, PHOTO_FILE)
                val dngFile = File(photoDir, DNG_FILE)
                val thumbnailFile = File(photoDir, THUMBNAIL_FILE)

                if (photoFile.exists()) {
                    val bakPhotoFile = File(photoDir, "original.${photoFile.lastModified()}.jpg")
                    photoFile.renameTo(bakPhotoFile)
                }

                if (dngFile.exists()) {
                    val bakPhotoFile = File(photoDir, "original.${photoFile.lastModified()}.dng")
                    dngFile.renameTo(bakPhotoFile)
                }

                // 2. 读取元数据以获取旋转信息
                val metadata = MediaMetadata.fromUri(context, uri).copy(
                    lutId = lutId,
                    computationalAperture = computationalAperture,
                    sourceUri = uri.toString()
                )

                // 1. 检测是否为 RAW 文件
                val mimeType = context.contentResolver.getType(uri)
                val fileName = getFileName(context, uri) ?: ""
                val userPrefs =
                    ContentRepository.getInstance(context).userPreferencesRepository.userPreferences.firstOrNull()
                val isRaw = mimeType?.contains("raw", ignoreCase = true) == true ||
                        mimeType?.contains("dng", ignoreCase = true) == true ||
                        fileName.endsWith(".dng", ignoreCase = true) ||
                        fileName.endsWith(".rw2", ignoreCase = true) ||
                        fileName.endsWith(".arw", ignoreCase = true) ||
                        fileName.endsWith(".cr3", ignoreCase = true)

                if (isRaw) {
                    // --- RAW 处理逻辑 ---
                    // 1. 复制 RAW 文件
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(dngFile).use { output ->
                            input.copyTo(output)
                        }
                    }

                    // 3. 处理 RAW 以生成 JPEG 预览
                    var updatedMetadata = metadata
                    val processedBitmap = RawDemosaicProcessor.getInstance().process(
                        context,
                        dngFile.absolutePath, null, null, 0,
                        rawExposureCompensation = updatedMetadata.rawExposureCompensation ?: 0f,
                        rawAutoExposure = resolveRawAutoExposure(context, updatedMetadata),
                        rawBlackPointCorrection = updatedMetadata.rawBlackPointCorrection ?: 0f,
                        rawWhitePointCorrection = updatedMetadata.rawWhitePointCorrection ?: 0f,
                        rawAutoWhiteBalanceEstimate = resolveRawAutoWhiteBalanceEstimate(context, updatedMetadata),
                        rawDROMode = RawProcessingPreferences.DROMode.fromPersistedName(
                            updatedMetadata.droMode
                        ),
                        sharpeningValue = 0.4f,
                        denoiseValue = updatedMetadata.rawDenoiseValue,
                        rawDcpId = updatedMetadata.rawDcpId,
                        onMetadata = { raw ->
                            updatedMetadata = updatedMetadata.merge(raw)
                        }
                    )

                    if (processedBitmap != null) {
                        // 保存为 original.jpg
                        FileOutputStream(photoFile).use { out ->
                            processedBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                        }

                        // 生成缩略图
                        generateThumbnail(processedBitmap, thumbnailFile)

                        // 更新元数据
                        updatedMetadata = updatedMetadata.copy(
                            width = processedBitmap.width,
                            height = processedBitmap.height,
                            rotation = 0,
                            manualHdrEffectEnabled = false,
                        )
                        saveMetadata(context, photoId, updatedMetadata)
//                        if (updatedMetadata.computationalAperture != null) {
//                            generateBokehPhoto(context, photoId, updatedMetadata, processedBitmap)
//                        }

                        processedBitmap.recycle()
                    } else {
                        // 降级：如果 RAW 处理失败，尝试直接解码（某些 DNG 包含内置预览图）
                        // 传递元数据确保旋转信息被正确处理
                        tempImportJpeg(uri, context, metadata, photoFile, thumbnailFile)
                    }
                } else {
                    // --- 常规 JPEG 处理逻辑 ---
                    // 传递元数据确保旋转信息被正确处理
                    tempImportJpeg(uri, context, metadata, photoFile, thumbnailFile)
                    val hasEmbeddedGainmap = detectEmbeddedGainmap(context, photoFile)
                    updateMetadata(context, photoId) { current ->
                        current.copy(
                            hasEmbeddedGainmap = hasEmbeddedGainmap,
                            manualHdrEffectEnabled = hasEmbeddedGainmap
                        )
                    }
//                    if (metadata.computationalAperture != null) {
//                        val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
//                        if (bitmap != null) {
//                            generateBokehPhoto(context, photoId, metadata, bitmap)
//                            bitmap.recycle()
//                        }
//                    }
                }

                // Check for Motion Photo after import
                if (photoFile.exists() && MotionPhotoWriter.isMotionPhoto(photoFile.absolutePath)) {
                    val videoFile = File(photoDir, VIDEO_FILE)
                    if (MotionPhotoWriter.extractVideo(photoFile.absolutePath, videoFile.absolutePath)) {
                        PLog.d(TAG, "Extracted video from imported Motion Photo: $photoId")
                        val timestampUs = MotionPhotoWriter.getPresentationTimestampUs(photoFile.absolutePath)
                        updateMetadata(context, photoId) { current ->
                            current.copy(presentationTimestampUs = timestampUs)
                        }
                    }
                }

                val importedMetadata = loadMetadata(context, photoId) ?: metadata
                queueDetailHdrCacheBuild(
                    context = context,
                    photoId = photoId,
                    metadata = importedMetadata,
                    sharpening = importedMetadata.sharpening ?: 0f,
                    noiseReduction = importedMetadata.noiseReduction ?: 0f,
                    chromaNoiseReduction = importedMetadata.chromaNoiseReduction ?: 0f
                )

                PLog.d(TAG, "Photo imported: $photoId (isRaw: $isRaw)")
                photoId
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to import photo", e)
                null
            }
        }
    }

    suspend fun refreshRawPreview(
        context: Context,
        photoId: String,
    ): Bitmap? {
        return withContext(Dispatchers.IO) {
            try {
                awaitDetailHdrBuildIdle(photoId)

                val photoDir = getPhotoDir(context, photoId, true)
                val photoFile = File(photoDir, PHOTO_FILE)
                val tempPhotoFile = File(photoDir, "raw_refresh_temp.jpg")
                val dngFile = File(photoDir, DNG_FILE)
                val thumbnailFile = File(photoDir, THUMBNAIL_FILE)

                if (!dngFile.exists()) return@withContext null

                // 2. 读取元数据以获取旋转信息
                val metadata = loadMetadata(context, photoId)

                // 3. 处理 RAW 以生成 JPEG 预览
                var updatedMetadata = metadata
                val processedBitmap = RawDemosaicProcessor.getInstance().process(
                    context,
                    dngFile.absolutePath, metadata?.ratio, metadata?.cropRegion, 0,
                    rawExposureCompensation = updatedMetadata?.rawExposureCompensation ?: 0f,
                    rawAutoExposure = resolveRawAutoExposure(context, updatedMetadata),
                    rawBlackPointCorrection = updatedMetadata?.rawBlackPointCorrection ?: 0f,
                    rawWhitePointCorrection = updatedMetadata?.rawWhitePointCorrection ?: 0f,
                    rawAutoWhiteBalanceEstimate = resolveRawAutoWhiteBalanceEstimate(context, updatedMetadata),
                    rawDROMode = updatedMetadata?.droMode?.let { RawProcessingPreferences.DROMode.fromPersistedName(it) }
                        ?: RawProcessingPreferences.DROMode.OFF,
                    sharpeningValue = updatedMetadata?.sharpening ?: 0.4f,
                    denoiseValue = (updatedMetadata ?: MediaMetadata()).rawDenoiseValue,
                    rawDcpId = updatedMetadata?.rawDcpId,
                    onMetadata = { raw ->
                        updatedMetadata = updatedMetadata?.merge(raw) ?: MediaMetadata().merge(raw)
                    }
                )

                if (processedBitmap != null) {
                    // 先写临时文件再替换，避免详情页或 HDR 任务读到半写入的 original.jpg。
                    FileOutputStream(tempPhotoFile).use { out ->
                        processedBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                    }
                    if (photoFile.exists() && !photoFile.delete()) {
                        tempPhotoFile.delete()
                        return@withContext null
                    }
                    if (!tempPhotoFile.renameTo(photoFile)) {
                        tempPhotoFile.delete()
                        return@withContext null
                    }
                    // 刷新 RAW 后，AI 降噪结果已失效，清理之
                    getAiDenoiseFile(context, photoId).takeIf { it.exists() }?.delete()

                    updatedMetadata?.let {
                        val finalMetadata = it.copy(hasAiDenoisedBase = false)
                        generateBokehPhoto(context, photoId, finalMetadata, processedBitmap.copy(Bitmap.Config.ARGB_8888, true))
                        saveMetadata(context, photoId, finalMetadata)
                        if (finalMetadata.manualHdrEffectEnabled) {
                            deleteDetailHdrFile(context, photoId)
                            queueDetailHdrCacheBuild(
                                context = context,
                                photoId = photoId,
                                metadata = finalMetadata,
                                sharpening = finalMetadata.sharpening ?: 0f,
                                noiseReduction = finalMetadata.noiseReduction ?: 0f,
                                chromaNoiseReduction = finalMetadata.chromaNoiseReduction ?: 0f
                            )
                        }
                    }
                    // 生成缩略图
                    generateThumbnail(processedBitmap, thumbnailFile)
                }
                processedBitmap
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to refresh RAW preview", e)
                File(getPhotoDir(context, photoId, true), "raw_refresh_temp.jpg").takeIf { it.exists() }?.delete()
                null
            }
        }
    }

    private suspend fun tempImportJpeg(
        uri: Uri,
        context: Context,
        metadata: MediaMetadata,
        photoFile: File,
        thumbnailFile: File
    ) {
        val photoDir = photoFile.parentFile ?: return
        val tempFile = File(photoDir, "temp.jpg")
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        }

        var updatedMetadata = metadata

        val exif = ExifInterface(tempFile)
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        if (orientation != ExifInterface.ORIENTATION_NORMAL &&
            orientation != ExifInterface.ORIENTATION_UNDEFINED
        ) {
            val bitmap = BitmapFactory.decodeFile(tempFile.absolutePath)
            if (bitmap != null) {
                val rotatedBitmap = rotateImageIfRequired(bitmap, orientation)
                FileOutputStream(photoFile).use { out ->
                    rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
                }
                val newExif = ExifInterface(photoFile)
                newExif.setAttribute(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL.toString()
                )
                newExif.saveAttributes()

                updatedMetadata = metadata.copy(
                    width = rotatedBitmap.width,
                    height = rotatedBitmap.height,
                    rotation = 0,
                )

                if (rotatedBitmap != bitmap) {
                    rotatedBitmap.recycle()
                }
                bitmap.recycle()
            } else {
                tempFile.copyTo(photoFile, overwrite = true)
            }
        } else {
            tempFile.renameTo(photoFile)
        }

        if (tempFile.exists()) {
            tempFile.delete()
        }

        saveMetadata(context, photoDir.name, updatedMetadata)
        generateThumbnail(photoFile, thumbnailFile)
    }

    /**
     * 根据 EXIF 方向信息旋转图片
     */
    internal fun rotateImageIfRequired(img: Bitmap, orientation: Int): Bitmap {
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> {
                rotateImage(img, 90f)
            }

            ExifInterface.ORIENTATION_ROTATE_180 -> {
                rotateImage(img, 180f)
            }

            ExifInterface.ORIENTATION_ROTATE_270 -> {
                rotateImage(img, 270f)
            }

            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
                flipImage(img, horizontal = true, vertical = false)
            }

            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                flipImage(img, horizontal = false, vertical = true)
            }

            ExifInterface.ORIENTATION_TRANSPOSE -> {
                // 先水平翻转，再旋转 270 度
                val flipped = flipImage(img, horizontal = true, vertical = false)
                val rotated = rotateImage(flipped, 270f)
                if (flipped != img) flipped.recycle()
                rotated
            }

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                // 先垂直翻转，再旋转 270 度
                val flipped = flipImage(img, horizontal = false, vertical = true)
                val rotated = rotateImage(flipped, 270f)
                if (flipped != img) flipped.recycle()
                rotated
            }

            else -> img
        }
    }

    /**
     * 旋转图片
     */
    internal fun rotateImage(img: Bitmap, degree: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degree)
        return Bitmap.createBitmap(img, 0, 0, img.width, img.height, matrix, false)
    }

    /**
     * 翻转图片
     */
    internal fun flipImage(img: Bitmap, horizontal: Boolean, vertical: Boolean): Bitmap {
        val matrix = Matrix()
        matrix.postScale(
            if (horizontal) -1f else 1f,
            if (vertical) -1f else 1f
        )
        return Bitmap.createBitmap(img, 0, 0, img.width, img.height, matrix, false)
    }
}
