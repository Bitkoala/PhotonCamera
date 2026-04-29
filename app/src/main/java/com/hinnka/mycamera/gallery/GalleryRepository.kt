package com.hinnka.mycamera.gallery

import android.content.Context
import android.content.ContentUris
import android.media.MediaMetadataRetriever
import android.provider.MediaStore
import com.hinnka.mycamera.utils.StartupTrace
import com.hinnka.mycamera.utils.PLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

/**
 * 相册数据仓库
 * 负责从私有存储读取和管理 PhotonCamera 拍摄的照片
 */
class GalleryRepository(private val context: Context) {

    companion object {
        private const val TAG = "GalleryRepository"
    }

    /**
     * 获取所有照片（按时间倒序）
     */
    fun getPhotos(): Flow<List<MediaData>> = flow {
        emit(queryPhotos())
    }.flowOn(Dispatchers.IO)

    /**
     * 同步获取所有照片
     */
    suspend fun getPhotosSync(): List<MediaData> = withContext(Dispatchers.IO) {
        queryPhotos()
    }

    /**
     * 分页获取 Photon 相册照片
     */
    suspend fun getPhotosPage(offset: Int, limit: Int): List<MediaData> = withContext(Dispatchers.IO) {
        queryPhotos(offset = offset, limit = limit)
    }

    /**
     * 获取最新的一张照片
     */
    suspend fun getLatestPhoto(): MediaData? = withContext(Dispatchers.IO) {
        queryPhotos(limit = 1).firstOrNull()
    }

    /**
     * 删除单张照片
     */
    suspend fun deletePhoto(photo: MediaData): Boolean = withContext(Dispatchers.IO) {
        GalleryManager.deletePhoto(context, photo.id)
    }

    /**
     * 批量删除照片
     */
    suspend fun deletePhotos(photos: List<MediaData>): Int = withContext(Dispatchers.IO) {
        var deletedCount = 0
        photos.forEach { photo ->
            if (deletePhoto(photo)) {
                deletedCount++
            }
        }
        deletedCount
    }

    /**
     * 查询系统相册照片
     */
    suspend fun getSystemPhotos(offset: Int, limit: Int): List<MediaData> = withContext(Dispatchers.IO) {
        val imageItems = querySystemImages(offset = 0, limit = offset + limit)
        val videoItems = querySystemVideos(offset = 0, limit = offset + limit)
        (imageItems + videoItems)
            .sortedByDescending { it.dateAdded }
            .drop(offset)
            .take(limit)
    }

    /**
     * 查询全部系统相册照片
     */
    suspend fun getAllSystemPhotos(): List<MediaData> = withContext(Dispatchers.IO) {
        (querySystemImages() + querySystemVideos())
            .sortedByDescending { it.dateAdded }
    }

    /**
     * 查询私有存储中的照片
     */
    private suspend fun queryPhotos(offset: Int = 0, limit: Int = Int.MAX_VALUE): List<MediaData> {
        val ids = GalleryManager.getPhotoIds(context)
        val photos = mutableListOf<MediaData>()
        var skippedValidCount = 0
        var invalidCount = 0

        for (id in ids) {
            val photo = GalleryManager.buildPhotoData(context, id)
            if (photo == null) {
                invalidCount++
                continue
            }

            if (skippedValidCount < offset) {
                skippedValidCount++
                continue
            }

            photos.add(photo)
            if (photos.size >= limit) {
                break
            }
        }

        if (invalidCount > 0) {
            PLog.w(
                TAG,
                "queryPhotos skipped $invalidCount invalid entries, offset=$offset, limit=$limit, loaded=${photos.size}"
            )
        }
        return photos
    }

    private fun querySystemImages(offset: Int = 0, limit: Int? = null): List<MediaData> {
        val items = mutableListOf<MediaData>()
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATE_ADDED,
            MediaStore.Images.Media.SIZE,
            MediaStore.Images.Media.WIDTH,
            MediaStore.Images.Media.HEIGHT,
            MediaStore.Images.Media.MIME_TYPE
        )
        val queryArgs = android.os.Bundle().apply {
            putInt(android.content.ContentResolver.QUERY_ARG_OFFSET, offset)
            limit?.let { putInt(android.content.ContentResolver.QUERY_ARG_LIMIT, it) }
            putString(
                android.content.ContentResolver.QUERY_ARG_SQL_SORT_ORDER,
                "${MediaStore.Images.Media.DATE_ADDED} DESC"
            )
        }
        context.contentResolver.query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, projection, queryArgs, null)
            ?.use { cursor ->
                val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
                val dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)
                val sizeColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.SIZE)
                val widthColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.WIDTH)
                val heightColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.HEIGHT)
                val mimeColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.MIME_TYPE)

                while (cursor.moveToNext()) {
                    val id = cursor.getLong(idColumn)
                    val contentUri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
                    items.add(
                        MediaData(
                            id = "image_$id",
                            uri = contentUri,
                            thumbnailUri = contentUri,
                            displayName = cursor.getString(nameColumn),
                            dateAdded = cursor.getLong(dateColumn) * 1000L,
                            size = cursor.getLong(sizeColumn),
                            width = cursor.getInt(widthColumn),
                            height = cursor.getInt(heightColumn),
                            mediaType = MediaType.IMAGE,
                            mimeType = cursor.getString(mimeColumn),
                            sourceUri = contentUri
                        )
                    )
                }
            }
        return items
    }

    private fun querySystemVideos(offset: Int = 0, limit: Int? = null): List<MediaData> {
        val items = mutableListOf<MediaData>()
        val projection = arrayOf(
            MediaStore.Video.Media._ID,
            MediaStore.Video.Media.DISPLAY_NAME,
            MediaStore.Video.Media.DATE_ADDED,
            MediaStore.Video.Media.SIZE,
            MediaStore.Video.Media.WIDTH,
            MediaStore.Video.Media.HEIGHT,
            MediaStore.Video.Media.MIME_TYPE,
            MediaStore.Video.Media.DURATION
        )
        val queryArgs = android.os.Bundle().apply {
            putInt(android.content.ContentResolver.QUERY_ARG_OFFSET, offset)
            limit?.let { putInt(android.content.ContentResolver.QUERY_ARG_LIMIT, it) }
            putString(
                android.content.ContentResolver.QUERY_ARG_SQL_SORT_ORDER,
                "${MediaStore.Video.Media.DATE_ADDED} DESC"
            )
        }
        context.contentResolver.query(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, projection, queryArgs, null)
            ?.use { cursor ->
                val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media._ID)
                val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DISPLAY_NAME)
                val dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATE_ADDED)
                val sizeColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.SIZE)
                val widthColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.WIDTH)
                val heightColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.HEIGHT)
                val mimeColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.MIME_TYPE)
                val durationColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DURATION)

                while (cursor.moveToNext()) {
                    val id = cursor.getLong(idColumn)
                    val contentUri = ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, id)
                    val mimeType = cursor.getString(mimeColumn)
                    val durationMs = cursor.getLong(durationColumn)
                    var frameRate: Int? = null
                    var bitrate: Long? = null
                    var rotationDegrees: Int? = null
                    var hasAudio: Boolean? = null
                    val retriever = MediaMetadataRetriever()
                    try {
                        retriever.setDataSource(context, contentUri)
                        frameRate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                            ?.toFloatOrNull()?.toInt()
                        bitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toLongOrNull()
                        rotationDegrees = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull()
                        hasAudio = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_AUDIO)?.let {
                            it == "yes" || it == "true" || it == "1"
                        }
                    } finally {
                        retriever.release()
                    }
                    items.add(
                        MediaData(
                            id = "video_$id",
                            uri = contentUri,
                            thumbnailUri = contentUri,
                            displayName = cursor.getString(nameColumn),
                            dateAdded = cursor.getLong(dateColumn) * 1000L,
                            size = cursor.getLong(sizeColumn),
                            width = cursor.getInt(widthColumn),
                            height = cursor.getInt(heightColumn),
                            mediaType = MediaType.VIDEO,
                            mimeType = mimeType,
                            durationMs = durationMs,
                            sourceUri = contentUri,
                            metadata = MediaMetadata(
                                mediaType = MediaType.VIDEO,
                                dateTaken = cursor.getLong(dateColumn) * 1000L,
                                width = cursor.getInt(widthColumn),
                                height = cursor.getInt(heightColumn),
                                sourceUri = contentUri.toString(),
                                mimeType = mimeType,
                                durationMs = durationMs,
                                frameRate = frameRate,
                                bitrate = bitrate,
                                rotationDegrees = rotationDegrees,
                                hasAudio = hasAudio,
                                videoWidth = cursor.getInt(widthColumn),
                                videoHeight = cursor.getInt(heightColumn)
                            )
                        )
                    )
                }
            }
        return items
    }
}
