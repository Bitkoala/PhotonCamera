package com.hinnka.mycamera.viewmodel

import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.LruCache
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateMap
import androidx.core.content.FileProvider
import androidx.core.graphics.scale
import androidx.core.net.toUri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.hinnka.mycamera.data.ContentRepository
import com.hinnka.mycamera.frame.FrameInfo
import com.hinnka.mycamera.gallery.GalleryManager
import com.hinnka.mycamera.gallery.MediaData
import com.hinnka.mycamera.gallery.MediaMetadata
import com.hinnka.mycamera.hdr.HdrGainmapStrength
import com.hinnka.mycamera.hdr.UnifiedGainmapProducer
import com.hinnka.mycamera.lut.creator.AiPhotoEvaluation
import com.hinnka.mycamera.lut.LutConfig
import com.hinnka.mycamera.lut.LutInfo
import com.hinnka.mycamera.lut.PhotoTransformation
import com.hinnka.mycamera.lut.creator.OpenAIApiClient
import com.hinnka.mycamera.model.ColorRecipeParams
import com.hinnka.mycamera.raw.DcpInfo
import com.hinnka.mycamera.raw.RawProcessingPreferences
import com.hinnka.mycamera.ui.gallery.CropAspectOption
import com.hinnka.mycamera.ui.gallery.calculateInitialCropRect
import com.hinnka.mycamera.utils.PLog
import com.hinnka.mycamera.utils.StartupTrace
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlin.math.roundToInt

/**
 * 相册 Tab
 */
enum class GalleryTab {
    PHOTON, SYSTEM
}

/**
 * 相册 ViewModel
 * 管理照片列表、选择状态和各种操作
 */
class GalleryViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "GalleryViewModel"
    }


    // 内容仓库（单例，与 CameraViewModel 共享）
    private val contentRepository = ContentRepository.getInstance(application)

    private val repository = contentRepository.galleryRepository
    private val detailGainmapProducer = UnifiedGainmapProducer()

    // 用户偏好设置仓库
    private val userPreferencesRepository = contentRepository.userPreferencesRepository

    // 软件处理参数
    val sharpening: StateFlow<Float> = userPreferencesRepository.userPreferences
        .map { it.sharpening }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)

    val noiseReduction: StateFlow<Float> = userPreferencesRepository.userPreferences
        .map { it.noiseReduction }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)

    val chromaNoiseReduction: StateFlow<Float> = userPreferencesRepository.userPreferences
        .map { it.chromaNoiseReduction }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)

    val categoryOrder: StateFlow<List<String>> = userPreferencesRepository.userPreferences
        .map { it.categoryOrder }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    val photoQuality: Flow<Int> = userPreferencesRepository.userPreferences.map { it.photoQuality }

    val defaultLutId: Flow<String?> = userPreferencesRepository.userPreferences.map { it.lutId }
    val defaultVirtualAperture: StateFlow<Float> = userPreferencesRepository.userPreferences.map { it.defaultVirtualAperture }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)

    val droMode: StateFlow<String> = userPreferencesRepository.userPreferences
        .map { it.droMode }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "OFF")

    // 计费管理器
    private val billingManager = com.hinnka.mycamera.billing.BillingManagerImpl(application)
    val isPurchased = billingManager.isPurchased

    // 付费弹窗状态
    var showPaymentDialog by mutableStateOf(false)

    // 水印编辑弹窗状态
    var showWatermarkSheet by mutableStateOf(false)

    // 当前选中的 Tab
    var selectedTab by mutableStateOf(GalleryTab.PHOTON)
        private set

    // 照片列表
    private val _photos = MutableStateFlow<List<MediaData>>(emptyList())
    val photos = _photos.asStateFlow()
    private val _systemPhotos = MutableStateFlow<List<MediaData>>(emptyList())
    val systemPhotos = _systemPhotos.asStateFlow()
    val currentPhotos = combine(photos, systemPhotos, snapshotFlow { selectedTab }) { p, s, tab ->
        if (tab == GalleryTab.PHOTON) {
            p
        } else {
            // Optimize lookup by creating a map of sourceUri to PhotoData
            val photonMap = mutableMapOf<String, MediaData>()
            p.forEach { photo ->
                photo.sourceUri?.toString()?.let { photonMap[it] = photo }
                photo.metadata?.sourceUri?.let { photonMap[it] = photo }
            }

            s.map { systemPhoto ->
                photonMap[systemPhoto.uri.toString()]?.let { photonPhoto ->
                    systemPhoto.copy(
                        relatedPhoto = photonPhoto
                    )
                } ?: systemPhoto
            }
        }
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private var systemOffset = 0
    private var photonOffset = 0
    var hasMoreSystemPhotos by mutableStateOf(true)
        private set
    var hasMorePhotonPhotos by mutableStateOf(true)
        private set

    private val _isLoading = MutableStateFlow(false)
    val isLoading = _isLoading.asStateFlow()

    private val _isSystemLoadingMore = MutableStateFlow(false)
    val isSystemLoadingMore: StateFlow<Boolean> = _isSystemLoadingMore.asStateFlow()
    private val _isPhotonLoadingMore = MutableStateFlow(false)
    val isPhotonLoadingMore: StateFlow<Boolean> = _isPhotonLoadingMore.asStateFlow()

    // 权限状态
    var hasGalleryPermission by mutableStateOf(false)
        private set

    // 分享状态
    private val _isSharing = MutableStateFlow(false)
    val isSharing: StateFlow<Boolean> = _isSharing.asStateFlow()

    // 导出状态
    private val _isExporting = MutableStateFlow(false)
    val isExporting: StateFlow<Boolean> = _isExporting.asStateFlow()
    var exportProgress by mutableStateOf(0 to 0)
        private set

    // 多选模式
    var isSelectionMode by mutableStateOf(false)
        private set

    // 选中的照片
    val selectedPhotos = mutableStateListOf<MediaData>()

    // 当前查看的照片索引
    var currentPhotoIndex by mutableStateOf(0)
        private set

    // 编辑状态
    var isEditing by mutableStateOf(false)
        private set

    // LUT 编辑状态
    var editLutId = MutableStateFlow<String?>(null)
        private set

    @OptIn(ExperimentalCoroutinesApi::class)
    var editLutRecipeParams = editLutId.flatMapLatest { id ->
        if (id == null) {
            flowOf(ColorRecipeParams.DEFAULT)
        } else {
            contentRepository.lutManager.getColorRecipeParams(id)
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.Eagerly,
        initialValue = ColorRecipeParams.DEFAULT
    )

    /** 仅此照片的色彩配方覆盖，null 表示跟随 LUT 默认配方 */
    var editPhotoRecipeParams = MutableStateFlow<ColorRecipeParams?>(null)
        private set

    var editLutConfig: LutConfig? by mutableStateOf(null)
        private set

    // 系统相册删除
    var systemDeletePendingIntent by mutableStateOf<android.app.PendingIntent?>(null)
        private set
    private var pendingDeleteSystemPhoto: MediaData? = null

    // 当前照片的元数据
    var currentMediaMetadata: MediaMetadata? by mutableStateOf(null)
        private set

    private var currentPhotoMetadataId: String? = null

    // 当前照片的平均亮度（调试用）
    val currentBrightness = SnapshotStateMap<String, Float>()

    // 照片刷新密钥，用于强制 UI 重新加载图片
    val photoRefreshKeys = SnapshotStateMap<String, Long>()

    // 正在刷新的照片 ID 集合
    val refreshingPhotos = mutableStateListOf<String>()

    // 预览图 LRU 缓存，按 Bitmap 字节数计算大小
    private val previewBitmapCache = object : LruCache<String, Bitmap>(
        // 限制缓存大小为可用内存的 1/8
        (Runtime.getRuntime().maxMemory() / 8).toInt()
    ) {
        override fun sizeOf(key: String, value: Bitmap): Int {
            return value.allocationByteCount
        }
    }

    private val detailBitmapCache = object : LruCache<String, Bitmap>(
        (Runtime.getRuntime().maxMemory() / 12).toInt()
    ) {
        override fun sizeOf(key: String, value: Bitmap): Int {
            return value.allocationByteCount
        }
    }

    private val aiEvaluationCache = mutableMapOf<String, AiPhotoEvaluation>()

    // 可用的 LUT 列表
    var availableLuts: List<LutInfo> by mutableStateOf(emptyList())
        private set

    var availableDcps: List<DcpInfo> by mutableStateOf(emptyList())
        private set

    // 边框编辑状态
    var editFrameId = MutableStateFlow<String?>(null)
        private set

    // 可用的边框列表
    var availableFrames: List<FrameInfo> by mutableStateOf(emptyList())
        private set

    // 细节处理编辑状态 (Sharpening, Noise Reduction, Chroma Noise Reduction)
    var editSharpening = MutableStateFlow(0f)
        private set
    var editNoiseReduction = MutableStateFlow(0f)
        private set
    var editChromaNoiseReduction = MutableStateFlow(0f)
        private set
    var editRawDenoise = MutableStateFlow(0f)
        private set
    var editRawExposureCompensation = MutableStateFlow(0f)
        private set
    var editRawBlackPointCorrection = MutableStateFlow(0f)
        private set
    var editRawWhitePointCorrection = MutableStateFlow(0f)
        private set
    var editRawDcpId = MutableStateFlow<String?>(null)
        private set

    // Computational Bokeh editing state
    var editComputationalAperture = MutableStateFlow<Float?>(null)
        private set
    var editFocusPointX = MutableStateFlow<Float?>(null)
        private set
    var editFocusPointY = MutableStateFlow<Float?>(null)
        private set

    // 裁剪编辑状态
    var editCropRect = MutableStateFlow<RectF?>(null)
        private set
    var editCropAspectOption = MutableStateFlow<CropAspectOption>(CropAspectOption.Free)
        private set

    private var bokehJob: Job? = null

    fun setComputationalAperture(value: Float?) {
        editComputationalAperture.value = value
        updateBokehPhoto()
    }

    fun setFocusPoint(x: Float, y: Float) {
        editFocusPointX.value = x
        editFocusPointY.value = y
        updateBokehPhoto()
    }

    private fun updateBokehPhoto() {
        bokehJob?.cancel()
        bokehJob = viewModelScope.launch(Dispatchers.IO) {
            val context = getApplication<Application>()
            val photoData = getCurrentPhoto() ?: return@launch
            val aperture = editComputationalAperture.value
            if (aperture == null || aperture <= 0) {
                GalleryManager.getBokehFile(context, photoData.id).takeIf { it.exists() }?.delete()
                return@launch
            }
            val focusPointX = editFocusPointX.value
            val focusPointY = editFocusPointY.value
            val bitmap = GalleryManager.loadOriginalBitmap(context, photoData.id)
                ?: GalleryManager.loadBitmap(context, photoData.uri) ?: return@launch
            if (!isActive) return@launch
            val bokeh = contentRepository.depthBokehProcessor.applyHighQualityBokeh(
                context,
                photoData.id,
                bitmap,
                focusPointX,
                focusPointY,
                aperture
            )
            if (!isActive) return@launch
            GalleryManager.saveBokehPhoto(context, photoData.id, bokeh)
            photoRefreshKeys[photoData.id] = System.currentTimeMillis()
        }
    }

    fun hasDepthInfo(photo: MediaData?): Boolean {
        if (photo == null) return false
        if (selectedTab == GalleryTab.SYSTEM) return false
        val context = getApplication<Application>()
        return GalleryManager.getDepthFile(context, photo.id).exists()
    }

    // 最新照片（用于相机界面显示入口）
    private val _latestPhoto = MutableStateFlow<MediaData?>(null)
    val latestPhoto: StateFlow<MediaData?> = _latestPhoto.asStateFlow()

    // 待删除的照片（用于 Activity Result 回调）
    var pendingDeletePhoto: MediaData? by mutableStateOf(null)
        private set

    // 删除请求的 PendingIntent（用于启动系统删除对话框）
    var deletePendingIntent: android.app.PendingIntent? by mutableStateOf(null)
        private set

    // 批量删除相关状态
    private var pendingDeletePhotos: List<MediaData> = emptyList()
    var batchDeletePendingIntent: android.app.PendingIntent? by mutableStateOf(null)
        private set

    init {
        // 检查系统相册权限
        StartupTrace.measure("GalleryViewModel.checkGalleryPermission") {
            checkGalleryPermission()
        }

        viewModelScope.launch {
            loadPhotos()
        }

        viewModelScope.launch {
            GalleryManager.detailHdrReadyEvents.collect { photoId ->
                invalidatePreviewCache(photoId)
                photoRefreshKeys[photoId] = System.currentTimeMillis()
                PLog.d(TAG, "detail HDR ready, refreshed photo: $photoId")
            }
        }

        viewModelScope.launch {
            GalleryManager.photoLibraryChangedEvents.collect {
                PLog.d(TAG, "photo library changed, reload photon photos")
                loadPhotos()
            }
        }

        // 订阅 ContentRepository 的 StateFlow，结合用户自定义排序
        viewModelScope.launch {
            contentRepository.availableLuts.combine(
                userPreferencesRepository.userPreferences.map { it.filterOrder }
            ) { luts, order ->
                if (order.isEmpty()) {
                    luts
                } else {
                    val orderMap = order.withIndex().associate { it.value to it.index }
                    luts.sortedBy { orderMap[it.id] ?: Int.MAX_VALUE }
                }
            }.collect { sortedLuts ->
                availableLuts = sortedLuts
            }
        }

        viewModelScope.launch {
            contentRepository.availableFrames.combine(
                userPreferencesRepository.userPreferences.map { it.frameOrder }
            ) { frames, order ->
                if (order.isEmpty()) {
                    frames
                } else {
                    val orderMap = order.withIndex().associate { it.value to it.index }
                    frames.sortedBy { orderMap[it.id] ?: Int.MAX_VALUE }
                }
            }.collect { sortedFrames ->
                availableFrames = sortedFrames
            }
        }

        viewModelScope.launch {
            contentRepository.availableDcps.collect { dcps ->
                availableDcps = dcps.sortedBy { it.getName() }
            }
        }
        StartupTrace.mark("GalleryViewModel.init end")
    }

    /**
     * 加载当前选中的 Tab 内容
     */
    suspend fun loadCurrentTabData() {
        when (selectedTab) {
            GalleryTab.SYSTEM -> loadSystemPhotos(reset = true)
            GalleryTab.PHOTON -> loadPhotos(reset = true)
        }
    }

    suspend fun loadCurrentTabMore() {
        // 当前相册列表使用全量加载，底部触发不再追加分页数据。
        hasMoreSystemPhotos = false
        hasMorePhotonPhotos = false
    }

    /**
     * 切换 Tab
     */
    suspend fun selectTab(tab: GalleryTab) {
        if (selectedTab != tab) {
            selectedTab = tab
            loadCurrentTabData()
        }
    }

    /**
     * 检查并更新权限状态
     */
    fun checkGalleryPermission() {
        val context = getApplication<Application>()
        hasGalleryPermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.checkSelfPermission(android.Manifest.permission.READ_MEDIA_IMAGES) == android.content.pm.PackageManager.PERMISSION_GRANTED &&
                context.checkSelfPermission(android.Manifest.permission.READ_MEDIA_VIDEO) == android.content.pm.PackageManager.PERMISSION_GRANTED
        } else {
            context.checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE) == android.content.pm.PackageManager.PERMISSION_GRANTED
        }
        if (hasGalleryPermission && selectedTab == GalleryTab.SYSTEM && _systemPhotos.value.isEmpty()) {
            viewModelScope.launch {
                loadSystemPhotos(reset = true)
            }
        }
    }

    /**
     * 加载系统照片
     */
    suspend fun loadSystemPhotos(reset: Boolean = false) {
        if (!hasGalleryPermission) return
        if (!reset) return

        if (reset) {
            if (_systemPhotos.value.isEmpty()) {
                _isLoading.value = true
            }
            systemOffset = 0
            hasMoreSystemPhotos = false
        }

        try {
            val newPhotos = repository.getAllSystemPhotos()
            _systemPhotos.value = newPhotos
            systemOffset = newPhotos.size
            hasMoreSystemPhotos = false
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to load system photos", e)
        } finally {
            _isLoading.value = false
            _isSystemLoadingMore.value = false
        }
    }

    /**
     * 加载照片列表
     */
    suspend fun loadPhotos(reset: Boolean = true) {
        if (!reset) return

        if (reset) {
            if (_photos.value.isEmpty()) _isLoading.value = true
            photonOffset = 0
            hasMorePhotonPhotos = false
        }

        try {
            val newList = repository.getPhotosSync()
            val currentMap = _photos.value.associateBy { it.id }

            // 保留运行时状态，但使用最新加载到的 metadata/sourceUri 等持久化信息
            val mergedList = newList.map { fresh ->
                currentMap[fresh.id]?.let { existing ->
                    existing.copy(
                        uri = fresh.uri,
                        thumbnailUri = fresh.thumbnailUri,
                        displayName = fresh.displayName,
                        dateAdded = fresh.dateAdded,
                        size = fresh.size,
                        width = fresh.width,
                        height = fresh.height,
                        mediaType = fresh.mediaType,
                        mimeType = fresh.mimeType,
                        durationMs = fresh.durationMs,
                        sourceUri = fresh.sourceUri,
                        isMotionPhoto = fresh.isMotionPhoto,
                        isBurstPhoto = fresh.isBurstPhoto,
                        metadata = fresh.metadata,
                        relatedPhoto = fresh.relatedPhoto
                    )
                } ?: fresh
            }
            _photos.value = mergedList
            photonOffset = mergedList.size
            hasMorePhotonPhotos = false
            _latestPhoto.value = _photos.value.firstOrNull()
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to load photos", e)
        } finally {
            _isLoading.value = false
            _isPhotonLoadingMore.value = false
        }
    }

    /**
     * 刷新最新照片
     */
    fun refreshLatestPhoto() {
        viewModelScope.launch {
            val photo = repository.getLatestPhoto()
            photo?.let {
                if (it.isVideo) {
                    _latestPhoto.value = it
                    if (_photos.value.none { existing -> existing.id == it.id }) {
                        _photos.value = listOf(it) + _photos.value
                    }
                    return@let
                }
                val context = getApplication<Application>()
                val metadata = GalleryManager.loadMetadata(context, it.id)
                var updatedPhoto = if (metadata != null) {
                    it.copy(
                        metadata = metadata,
                        width = if (metadata.width > 0) metadata.width else it.width,
                        height = if (metadata.height > 0) metadata.height else it.height
                    )
                } else {
                    it
                }

                // If dimensions are missing, load from file (same as loadPhotos)
                if (updatedPhoto.width == 0 || updatedPhoto.height == 0) {
                    try {
                        val photoFile = GalleryManager.getPhotoFile(context, it.id)
                        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                        BitmapFactory.decodeFile(photoFile.absolutePath, options)
                        updatedPhoto =
                            updatedPhoto.copy(width = options.outWidth, height = options.outHeight)

                        metadata?.let { m ->
                            val newMetadata =
                                m.copy(width = options.outWidth, height = options.outHeight)
                            GalleryManager.saveMetadata(context, it.id, newMetadata)
                        }
                    } catch (e: Exception) {
                        PLog.e(TAG, "Failed to load dimensions for latest photo", e)
                    }
                }
                _latestPhoto.value = updatedPhoto

                // Also update the photos list if it's not already in it
                val currentPhotos = _photos.value
                if (currentPhotos.none { it.id == updatedPhoto.id }) {
                    _photos.value = listOf(updatedPhoto) + currentPhotos
                }
            }
        }
    }

    /**
     * 导入从系统相册分享的照片
     */
    fun importSharedImage(uri: Uri, onSuccess: (String) -> Unit = {}) {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val photoId = GalleryManager.importPhoto(context, uri, null)
            if (photoId != null) {
                loadPhotos()
                selectedTab = GalleryTab.PHOTON
                setCurrentPhotoById(photoId)
                onSuccess(photoId)
            }
        }
    }

    /**
     * 批量导入从系统相册分享的照片
     */
    fun importSharedImages(uris: List<Uri>) {
        viewModelScope.launch {
            val context = getApplication<Application>()
            var lastPhotoId: String? = null
            uris.forEach { uri ->
                val id = GalleryManager.importPhoto(context, uri, null)
                if (id != null) lastPhotoId = id
            }
            if (lastPhotoId != null) {
                loadPhotos()
                selectedTab = GalleryTab.PHOTON
                setCurrentPhotoById(lastPhotoId)
            }
        }
    }

    fun registerRecordedVideo(uri: Uri, onComplete: (String?) -> Unit = {}) {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val mediaId = GalleryManager.recordVideoCapture(context, uri)
            if (mediaId != null) {
                loadPhotos()
                refreshLatestPhoto()
            }
            onComplete(mediaId)
        }
    }

    /**
     * 加载当前照片的元数据
     */
    fun loadCurrentPhotoMetadata() {
        val photo = getCurrentPhoto() ?: return

        // 如果已经加载了该照片的元数据，且不是正在编辑模式（编辑模式下可能需要刷新），则跳过
        if (!isEditing && currentPhotoMetadataId == photo.id) {
            return
        }

        // 优先使用 PhotoData 中已有的元数据（由 loadPhotos 预加载）
        photo.metadata?.let { m ->
            currentPhotoMetadataId = photo.id
            applyMetadataToEditState(m)
            return
        }

        photo.relatedPhoto?.metadata?.let { m ->
            currentPhotoMetadataId = photo.id
            applyMetadataToEditState(m)
            return
        }

        viewModelScope.launch {
            val context = getApplication<Application>()

            // 在 IO 线程加载元数据
            val metadata = withContext(Dispatchers.IO) {
                if (selectedTab == GalleryTab.SYSTEM) {
                    MediaMetadata.fromUri(context, photo.uri)
                } else {
                    GalleryManager.loadMetadata(context, photo.id)
                }
            }

            // 在主线程更新状态
            applyMetadataToEditState(metadata)

            // 缓存到列表，避免下次滑动到这张图时重复加载
            if (metadata != null) {
                if (selectedTab == GalleryTab.SYSTEM) {
                    _systemPhotos.update { current ->
                        current.map { if (it.id == photo.id) it.copy(metadata = metadata) else it }
                    }
                } else {
                    _photos.update { current ->
                        current.map { if (it.id == photo.id) it.copy(metadata = metadata) else it }
                    }
                }
            }
        }
    }

    private fun restoreCropEditState(photo: MediaData?, metadata: MediaMetadata?) {
        if (metadata == null) {
            editCropRect.value = null
            editCropAspectOption.value = CropAspectOption.Free
            return
        }

        val cw = photo?.width ?: metadata.width
        val ch = photo?.height ?: metadata.height
        if (metadata.postCropRegion != null && cw > 0 && ch > 0) {
            editCropRect.value = RectF(
                metadata.postCropRegion.left.toFloat() / cw,
                metadata.postCropRegion.top.toFloat() / ch,
                metadata.postCropRegion.right.toFloat() / cw,
                metadata.postCropRegion.bottom.toFloat() / ch
            )

            editCropAspectOption.value = CropAspectOption.Custom(
                metadata.postCropRegion.width().toFloat(),
                metadata.postCropRegion.height().toFloat()
            )
        } else {
            editCropRect.value = null
            editCropAspectOption.value = CropAspectOption.Free
        }
    }

    private fun applyMetadataToEditState(metadata: MediaMetadata?) {
        val photo = getCurrentPhoto()
        currentPhotoMetadataId = photo?.id
        currentMediaMetadata = metadata
        metadata?.let { m ->
            editLutId.value = m.lutId
            editFrameId.value = m.frameId
            editSharpening.value = m.sharpening ?: 0f
            editNoiseReduction.value = m.noiseReduction ?: 0f
            editChromaNoiseReduction.value = m.chromaNoiseReduction ?: 0f
            editRawDenoise.value = m.rawDenoiseValue ?: 0f
            editRawExposureCompensation.value = m.rawExposureCompensation ?: 0f
            editRawBlackPointCorrection.value = m.rawBlackPointCorrection ?: 0f
            editRawWhitePointCorrection.value = m.rawWhitePointCorrection ?: 0f
            editRawDcpId.value = m.rawDcpId
            restoreCropEditState(photo, m)

            // 加载 LUT 配置
            m.lutId?.let { id ->
                viewModelScope.launch {
                    editLutConfig = withContext(Dispatchers.IO) {
                        contentRepository.lutManager.loadLut(id)
                    }
                }
            }
        } ?: restoreCropEditState(photo, null)
    }

    fun loadThumbnail(photo: MediaData): Bitmap? {
        val context = getApplication<Application>()
        return try {
            if (photo.thumbnailUri.scheme == "content") {
                context.contentResolver.loadThumbnail(photo.thumbnailUri, android.util.Size(512, 512), null)
            } else {
                val inputStream = context.contentResolver.openInputStream(photo.thumbnailUri)
                val options = BitmapFactory.Options().apply {
                    // PhotonCamera thumbnails are 512x512, no need to downsample much, but safe is better
                    inJustDecodeBounds = true
                    BitmapFactory.decodeStream(inputStream, null, this)
                    inputStream?.close()

                    inSampleSize = 1
                    if (outWidth > 1024 || outHeight > 1024) {
                        inSampleSize = 2
                    }
                    inJustDecodeBounds = false
                }
                val inputStream2 = context.contentResolver.openInputStream(photo.thumbnailUri)
                val bitmap = BitmapFactory.decodeStream(inputStream2, null, options)
                inputStream2?.close()
                bitmap
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to load thumbnail for ${photo.id}", e)
            null
        }
    }

    /**
     * 进入多选模式
     */
    fun enterSelectionMode() {
        isSelectionMode = true
        selectedPhotos.clear()
    }

    /**
     * 退出多选模式
     */
    fun exitSelectionMode() {
        isSelectionMode = false
        selectedPhotos.clear()
    }

    /**
     * 切换照片选中状态
     */
    fun togglePhotoSelection(photo: MediaData) {
        if (selectedPhotos.contains(photo)) {
            selectedPhotos.remove(photo)
        } else {
            selectedPhotos.add(photo)
        }

        // 如果没有选中的照片，退出多选模式
        if (selectedPhotos.isEmpty()) {
            exitSelectionMode()
        }
    }

    /**
     * 全选/取消全选
     */
    fun toggleSelectAll() {
        if (selectedPhotos.size == _photos.value.size) {
            selectedPhotos.clear()
        } else {
            selectedPhotos.clear()
            selectedPhotos.addAll(_photos.value)
        }
    }

    /**
     * 设置当前查看的照片索引
     */
    fun setCurrentPhoto(index: Int) {
        currentPhotoIndex = index.coerceIn(0, (currentPhotos.value.size - 1).coerceAtLeast(0))
        loadCurrentPhotoMetadata()
    }

    /**
     * 根据 ID 设置当前查看的照片
     */
    fun setCurrentPhotoById(id: String) {
        val index = currentPhotos.value.indexOfFirst { it.id == id }
        if (index != -1) {
            setCurrentPhoto(index)
        }
    }

    /**
     * 获取当前照片
     */
    fun getCurrentPhoto(): MediaData? {
        return currentPhotos.value.getOrNull(currentPhotoIndex)
    }

    /**
     * 获取删除系统相册照片的请求 PendingIntent（用于 Android 11+）
     *
     * @return PendingIntent 如果有导出的照片需要删除，返回 PendingIntent；否则返回 null
     */
    private fun getDeleteRequest(photo: MediaData): android.app.PendingIntent? {
        val context = getApplication<Application>()
        return GalleryManager.createDeleteRequest(context, photo.id)
    }

    /**
     * 请求删除照片
     *
     * 这个方法会：
     * 1. 如果 deleteExported == false，直接删除应用内部照片，不触发系统删除对话框
     * 2. 如果 deleteExported == true 且在 Android 11+ 上，检查是否有导出的照片需要删除
     * 3. 如果有导出的照片，设置 deletePendingIntent 和 pendingDeletePhoto，UI 层需要监听这些状态并启动删除确认对话框
     * 4. 如果没有导出的照片，或者在 Android 10 及以下，直接删除照片
     */
    fun requestDeletePhoto(photo: MediaData, deleteExported: Boolean = true) {
        if (selectedTab == GalleryTab.SYSTEM) {
            // 系统相册删除
            val pendingIntent = GalleryManager.createSystemDeleteRequest(getApplication(), photo.uri)
            if (pendingIntent != null) {
                pendingDeleteSystemPhoto = photo
                systemDeletePendingIntent = pendingIntent
                PLog.d(TAG, "Set system delete pending intent for photo ${photo.id}")
            }
            return
        }

        if (!deleteExported) {
            // 不删除系统相册照片，直接删除应用内部照片
            deletePhotoOnlyInternal(photo)
            return
        }

        // Android 11+: 检查是否有导出的照片
        val pendingIntent = getDeleteRequest(photo)

        if (pendingIntent != null) {
            // 有导出的照片，需要用户确认
            pendingDeletePhoto = photo
            deletePendingIntent = pendingIntent
            PLog.d(TAG, "Set delete pending intent for photo ${photo.id}")
        } else {
            // 没有导出的照片，直接删除应用内照片
            deletePhotoOnlyInternal(photo)
        }
    }

    /**
     * 仅删除应用内部照片（不删除系统相册）
     */
    private fun deletePhotoOnlyInternal(photo: MediaData) {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val success = GalleryManager.deletePhotoOnly(context, photo.id)
            if (success) {
                loadPhotos()
                // 如果删除的是当前照片，调整索引
                if (photo == getCurrentPhoto() && currentPhotoIndex >= _photos.value.size) {
                    currentPhotoIndex = (_photos.value.size - 1).coerceAtLeast(0)
                }
                PLog.d(TAG, "Photo deleted (app only): ${photo.id}")
            } else {
                PLog.e(TAG, "Failed to delete photo: ${photo.id}")
            }
        }
    }

    /**
     * 清除删除请求状态
     */
    fun clearDeleteRequest() {
        pendingDeletePhoto = null
        deletePendingIntent = null
        pendingDeleteSystemPhoto = null
        systemDeletePendingIntent = null
    }

    /**
     * 清除批量删除请求状态
     */
    fun clearBatchDeleteRequest() {
        pendingDeletePhotos = emptyList()
        batchDeletePendingIntent = null
    }

    /**
     * 仅删除应用内部的照片（在用户确认删除系统相册照片后调用）
     */
    fun deletePhotoAfterConfirmation(onComplete: (Boolean) -> Unit = {}) {
        val photo = pendingDeletePhoto ?: return
        viewModelScope.launch {
            val context = getApplication<Application>()
            val success = GalleryManager.deletePhotoOnly(context, photo.id)
            if (success) {
                loadPhotos()

                // 如果删除的是当前照片，调整索引
                if (photo == getCurrentPhoto() && currentPhotoIndex >= _photos.value.size) {
                    currentPhotoIndex = (_photos.value.size - 1).coerceAtLeast(0)
                }

                PLog.d(TAG, "Photo deleted after confirmation: ${photo.id}")
            }
            clearDeleteRequest()
            onComplete(success)
        }
    }

    /**
     * 系统相册照片删除确认后的回调
     */
    fun deleteSystemPhotoAfterConfirmation(onComplete: (Boolean) -> Unit = {}) {
        viewModelScope.launch {
            pendingDeleteSystemPhoto?.relatedPhoto?.takeIf { it.isVideo }?.let { relatedVideo ->
                GalleryManager.deletePhotoOnly(getApplication(), relatedVideo.id)
                loadPhotos()
            }
            // 系统照片已经被 MediaStore 删除，我们只需要刷新列表
            loadSystemPhotos(reset = true)

            // 调整索引
            if (currentPhotoIndex >= currentPhotos.value.size) {
                currentPhotoIndex = (currentPhotos.value.size - 1).coerceAtLeast(0)
            }

            clearDeleteRequest()
            onComplete(true)
        }
    }

    /**
     * 设为连拍主图
     */
    fun setMainBurstPhoto(photo: MediaData, burstFile: File, onComplete: (Boolean) -> Unit) {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val success = GalleryManager.setMainBurstPhoto(context, photo.id, burstFile)
            if (success) {
                photoRefreshKeys[photo.id] = System.currentTimeMillis()
                loadPhotos()
            }
            withContext(Dispatchers.Main) {
                onComplete(success)
            }
        }
    }

    /**
     * 批量删除选中的照片
     *
     * @param deleteExported 是否同时删除系统相册中的导出图片
     * - true: Android 11+ 会使用 MediaStore.createDeleteRequest 弹出系统确认对话框
     * - false: 只删除应用内部照片，不删除系统相册
     */
    fun deleteSelectedPhotos(deleteExported: Boolean = true) {
        val toDelete = selectedPhotos.toList()
        if (toDelete.isEmpty()) return

        if (selectedTab == GalleryTab.SYSTEM) {
            // 系统相册批量删除
            viewModelScope.launch {
                val context = getApplication<Application>()
                val uris = toDelete.mapNotNull { photo ->
                    val uri = photo.uri
                    if (uri.scheme == "content") uri else {
                        PLog.w(TAG, "Ignoring non-content URI in system delete: $uri")
                        null
                    }
                }

                if (uris.isEmpty()) {
                    exitSelectionMode()
                    return@launch
                }

                try {
                    val pendingIntent = MediaStore.createDeleteRequest(
                        context.contentResolver,
                        uris
                    )
                    pendingDeletePhotos = toDelete
                    batchDeletePendingIntent = pendingIntent
                    PLog.d(
                        TAG,
                        "Set system batch delete pending intent for ${toDelete.size} photos"
                    )
                } catch (e: Exception) {
                    PLog.e(TAG, "Failed to create system batch delete request", e)
                }
            }
            return
        }

        if (!deleteExported) {
            // 不删除系统相册照片，直接删除应用内部照片
            deleteBatchPhotosOnlyInternal(toDelete)
            return
        }

        // Android 11+: 收集所有导出的 URIs
        viewModelScope.launch {
            val context = getApplication<Application>()
            val allExportedUris = mutableListOf<Uri>()

            withContext(Dispatchers.IO) {
                toDelete.forEach { photo ->
                    val metadata = GalleryManager.loadMetadata(context, photo.id)
                    metadata?.exportedUris?.forEach { uriString ->
                        try {
                            allExportedUris.add(uriString.toUri())
                        } catch (e: Exception) {
                            PLog.e(TAG, "Invalid URI: $uriString", e)
                        }
                    }
                }
            }

            val validExportedUris = allExportedUris.filter { it.scheme == "content" }

            if (validExportedUris.isNotEmpty()) {
                // 有导出的照片，需要用户确认
                try {
                    val pendingIntent = MediaStore.createDeleteRequest(
                        context.contentResolver,
                        validExportedUris
                    )
                    pendingDeletePhotos = toDelete
                    batchDeletePendingIntent = pendingIntent
                    PLog.d(TAG, "Set batch delete pending intent for ${toDelete.size} photos")
                } catch (e: Exception) {
                    PLog.e(TAG, "Failed to create batch delete request", e)
                    // 创建请求失败，直接删除应用内照片
                    deleteBatchPhotosOnlyInternal(toDelete)
                }
            } else {
                // 没有导出的照片，直接删除应用内照片
                deleteBatchPhotosOnlyInternal(toDelete)
            }
        }
    }

    /**
     * 仅删除应用内部照片（批量，不删除系统相册）
     */
    private fun deleteBatchPhotosOnlyInternal(photos: List<MediaData>) {
        viewModelScope.launch {
            val context = getApplication<Application>()
            var deletedCount = 0

            withContext(Dispatchers.IO) {
                photos.forEach { photo ->
                    val success = GalleryManager.deletePhotoOnly(context, photo.id)
                    if (success) {
                        deletedCount++
                    }
                }
            }

            exitSelectionMode()
            loadPhotos()
            PLog.d(TAG, "Batch deleted $deletedCount photos (app only)")
        }
    }

    /**
     * 批量删除确认后的回调（在用户确认删除系统相册照片后调用）
     */
    fun deleteBatchPhotosAfterConfirmation(onComplete: (Int) -> Unit = {}) {
        val photos = pendingDeletePhotos
        if (photos.isEmpty()) return

        viewModelScope.launch {
            if (selectedTab == GalleryTab.SYSTEM) {
                val relatedPhotonVideos = photos.mapNotNull { it.relatedPhoto }.filter { it.isVideo }
                if (relatedPhotonVideos.isNotEmpty()) {
                    val context = getApplication<Application>()
                    withContext(Dispatchers.IO) {
                        relatedPhotonVideos.forEach { GalleryManager.deletePhotoOnly(context, it.id) }
                    }
                    loadPhotos()
                }
                // 系统相册刷新
                loadSystemPhotos(reset = true)
            } else {
                // 原有的内部照片删除逻辑
                val context = getApplication<Application>()
                var deletedCount = 0

                withContext(Dispatchers.IO) {
                    photos.forEach { photo ->
                        val success = GalleryManager.deletePhotoOnly(context, photo.id)
                        if (success) {
                            deletedCount++
                        }
                    }
                }
                loadPhotos()
                PLog.d(TAG, "Batch deleted $deletedCount internal photos after confirmation")
            }

            exitSelectionMode()
            clearBatchDeleteRequest()
            onComplete(photos.size)
        }
    }

    /**
     * 获取 Motion Photo 视频文件
     */
    fun getMotionPhotoVideo(photo: MediaData): File? {
        val context = getApplication<Application>()
        val videoFile = GalleryManager.getVideoFile(context, photo.id)
        return if (videoFile.exists()) videoFile else null
    }

    /**
     * 分享照片
     */
    fun sharePhoto(photo: MediaData) {
        viewModelScope.launch {
            _isSharing.value = true
            try {
                val context = getApplication<Application>()
                val shareRequest = prepareShareRequest(photo)
                if (shareRequest != null) {
                    val shareIntent = Intent(Intent.ACTION_SEND).apply {
                        type = shareRequest.mimeType
                        putExtra(Intent.EXTRA_STREAM, shareRequest.uri)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    context.startActivity(
                        Intent.createChooser(shareIntent, null).apply {
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                    )
                }
            } finally {
                _isSharing.value = false
            }
        }
    }

    /**
     * 批量分享选中的照片
     */
    fun shareSelectedPhotos() {
        if (selectedPhotos.isEmpty()) return

        viewModelScope.launch {
            _isSharing.value = true
            try {
                val context = getApplication<Application>()
                val requests = selectedPhotos.mapNotNull { prepareShareRequest(it) }
                if (requests.isNotEmpty()) {
                    val uris = ArrayList(requests.map { it.uri })
                    val mimeType = if (requests.any { it.mimeType.startsWith("video/") }) {
                        "*/*"
                    } else {
                        "image/jpeg"
                    }

                    val shareIntent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                        type = mimeType
                        putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    val chooser = Intent.createChooser(shareIntent, null).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    context.startActivity(chooser)
                }
            } finally {
                _isSharing.value = false
            }
        }
    }

    /**
     * 进入编辑模式
     */
    fun enterEditMode() {
        val targetPhoto = getCurrentPhoto() ?: return
        if (targetPhoto.isVideo) return

        if (currentMediaMetadata == null || currentPhotoMetadataId != targetPhoto.id) {
            val context = getApplication<Application>()
            val metadata = targetPhoto.metadata ?: runBlocking {
                withContext(Dispatchers.IO) {
                    GalleryManager.loadMetadata(context, targetPhoto.id)
                }
            }
            applyMetadataToEditState(metadata)
        }

        isEditing = true
        // 从当前元数据恢复编辑状态
        currentMediaMetadata?.let { metadata ->
            editLutId.value = metadata.lutId
            editFrameId.value = metadata.frameId
            editPhotoRecipeParams.value = metadata.colorRecipeParams
            // 智能初始化：导入的照片默认值为 0，App 拍摄的则回退到当前全局配置
            editSharpening.value =
                metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening.value)
            editNoiseReduction.value =
                metadata.noiseReduction ?: (if (metadata.isImported) 0f else noiseReduction.value)
            editChromaNoiseReduction.value =
                metadata.chromaNoiseReduction
                    ?: (if (metadata.isImported) 0f else chromaNoiseReduction.value)
            editRawDenoise.value = metadata.rawDenoiseValue ?: 0f
            editRawExposureCompensation.value = metadata.rawExposureCompensation ?: 0f
            editRawBlackPointCorrection.value = metadata.rawBlackPointCorrection ?: 0f
            editRawWhitePointCorrection.value = metadata.rawWhitePointCorrection ?: 0f
            editRawDcpId.value = metadata.rawDcpId
            
            editComputationalAperture.value = metadata.computationalAperture
            editFocusPointX.value = metadata.focusPointX
            editFocusPointY.value = metadata.focusPointY
            restoreCropEditState(targetPhoto, metadata)
        } ?: run {
            editLutId.value = null
            editFrameId.value = null
            // 这里一般是本 App 预览或拍摄进入，保持跟随全局
            editSharpening.value = sharpening.value
            editNoiseReduction.value = noiseReduction.value
            editChromaNoiseReduction.value = chromaNoiseReduction.value
            editRawDenoise.value = 0.2f
            editRawExposureCompensation.value = 0f
            editRawBlackPointCorrection.value = 0f
            editRawWhitePointCorrection.value = 0f
            editRawDcpId.value = null
            editComputationalAperture.value = null
            editFocusPointX.value = null
            editFocusPointY.value = null
            restoreCropEditState(targetPhoto, null)
        }

        // 加载当前编辑的 LUT 配置
        editLutId.value?.let { id ->
            viewModelScope.launch {
                editLutConfig = withContext(Dispatchers.IO) {
                    contentRepository.lutManager.loadLut(id)
                }
            }
        }
    }

    /**
     * 退出编辑模式
     */
    fun exitEditMode() {
        isEditing = false
        editLutId.value = null
        editLutConfig = null
        editFrameId.value = null
        editPhotoRecipeParams.value = null
        editCropRect.value = null
        editCropAspectOption.value = CropAspectOption.Free
        editRawDenoise.value = 0.2f
        editRawExposureCompensation.value = 0f
        editRawBlackPointCorrection.value = 0f
        editRawWhitePointCorrection.value = 0f
        editRawDcpId.value = null
    }

    /**
     * 设置照片级别的色彩配方覆盖（null = 清除覆盖，跟随 LUT 默认）
     */
    fun setPhotoRecipeParams(params: ColorRecipeParams?) {
        editPhotoRecipeParams.value = params
    }

    /**
     * 设置 LUT
     */
    fun setEditLut(lutId: String?) {
        if (editLutId.value != lutId) {
            editPhotoRecipeParams.value = null
        }
        editLutId.value = lutId
        if (lutId == null) {
            editLutConfig = null
            return
        }

        viewModelScope.launch {
            editLutConfig = withContext(Dispatchers.IO) {
                contentRepository.lutManager.loadLut(lutId)
            }
        }
    }

    fun switchToNextLut() {
        if (availableLuts.isEmpty()) return
        val currentLut = editLutId.value
        val currentIndex = availableLuts.indexOfFirst { it.id == currentLut }
        val nextIndex = (currentIndex + 1) % availableLuts.size
        setEditLut(availableLuts[nextIndex].id)
    }

    fun switchToPreviousLut() {
        if (availableLuts.isEmpty()) return
        val currentLut = editLutId.value
        val currentIndex = availableLuts.indexOfFirst { it.id == currentLut }
        val prevIndex = if (currentIndex <= 0) availableLuts.size - 1 else currentIndex - 1
        setEditLut(availableLuts[prevIndex].id)
    }


    /**
     * 设置边框
     */
    fun setEditFrame(frameId: String?) {
        editFrameId.value = frameId
    }

    suspend fun getEditCustomProperties(frameId: String): Map<String, String> {
        return contentRepository.frameManager.loadCustomProperties(frameId)
    }

    /**
     * 保存当前边框的自定义属性到持久化存储
     */
    fun saveEditCustomProperties(properties: Map<String, String>) {
        currentMediaMetadata = currentMediaMetadata?.copy(
            customProperties = properties
        )
        viewModelScope.launch {
            editFrameId.value?.let {
                contentRepository.frameManager.saveCustomProperties(
                    it,
                    properties
                )
            }
        }
    }

    /**
     * 设置锐化强度
     */
    fun setSharpening(value: Float) {
        editSharpening.value = value
    }

    /**
     * 设置降噪强度
     */
    fun setNoiseReduction(value: Float) {
        editNoiseReduction.value = value
    }

    /**
     * 设置减少杂色强度
     */
    fun setChromaNoiseReduction(value: Float) {
        editChromaNoiseReduction.value = value
    }

    private fun persistRawEditMetadata(mediaData: MediaData) {
        viewModelScope.launch {
            val metadata = (mediaData.metadata ?: currentMediaMetadata)?.copy(
                rawDenoiseValue = editRawDenoise.value,
                rawExposureCompensation = editRawExposureCompensation.value,
                rawBlackPointCorrection = editRawBlackPointCorrection.value,
                rawWhitePointCorrection = editRawWhitePointCorrection.value,
                rawDcpId = editRawDcpId.value,
            )
            val context = getApplication<Application>()
            metadata?.let {
                GalleryManager.saveMetadata(context, mediaData.id, it)
            }
        }
    }

    fun saveRawDenoiseValue(mediaData: MediaData, value: Float) {
        editRawDenoise.value = value
        persistRawEditMetadata(mediaData)
    }

    fun saveRawExposureCompensationValue(mediaData: MediaData, value: Float) {
        editRawExposureCompensation.value = value
        persistRawEditMetadata(mediaData)
    }

    fun saveRawBlackPointCorrectionValue(mediaData: MediaData, value: Float) {
        editRawBlackPointCorrection.value = value
        persistRawEditMetadata(mediaData)
    }

    fun saveRawWhitePointCorrectionValue(mediaData: MediaData, value: Float) {
        editRawWhitePointCorrection.value = value
        persistRawEditMetadata(mediaData)
    }

    fun saveRawDcpSelection(mediaData: MediaData, dcpId: String?) {
        editRawDcpId.value = dcpId
        persistRawEditMetadata(mediaData)
    }

    suspend fun importRawDcp(uri: Uri): DcpInfo? {
        val importedId = withContext(Dispatchers.IO) {
            contentRepository.getCustomImportManager().importDcp(uri)
        } ?: return null
        contentRepository.refreshCustomContent()
        return contentRepository.getAvailableDcps().firstOrNull { it.id == importedId }?.also {
            editRawDcpId.value = it.id
        }
    }

    /**
     * 设置裁剪矩形（归一化坐标 0-1）
     */
    fun setCropRect(rect: RectF?) {
        editCropRect.value = rect
    }

    /**
     * 设置裁剪比例选项
     */
    fun setCropAspectOption(option: CropAspectOption) {
        editCropAspectOption.value = option
        val photo = getCurrentPhoto() ?: return
        val w = photo.metadata?.width ?: photo.width
        val h = photo.metadata?.height ?: photo.height
        if (w > 0 && h > 0) {
            editCropRect.value = calculateInitialCropRect(w, h, option)
        }
    }

    /**
     * 重置裁剪
     */
    fun resetCrop() {
        editCropRect.value = null
        editCropAspectOption.value = CropAspectOption.Free
    }


    fun isRaw(photoId: String): Boolean {
        val context = getApplication<Application>()
        return GalleryManager.getDngFile(context, photoId).exists()
    }

    /**
     * 获取指定照片的完整转换器（LUT + 边框）
     */
    fun getPhotoTransformation(photo: MediaData): PhotoTransformation? {
        if (photo.isVideo) return null
        val metadata = photo.metadata ?: return null

        // 使用照片自己的 metadata 中保存的处理参数，如果没有则使用全局设置
        // 导入的照片默认不应用处理（除非用户编辑过）
        val photoSharpening =
            metadata.sharpening ?: (if (metadata.isImported) 0f else sharpening.value)
        val photoNoiseReduction =
            metadata.noiseReduction ?: (if (metadata.isImported) 0f else noiseReduction.value)
        val photoChromaNoiseReduction =
            metadata.chromaNoiseReduction
                ?: (if (metadata.isImported) 0f else chromaNoiseReduction.value)

        return PhotoTransformation(
            context = getApplication<Application>(),
            metadata = metadata,
            photoProcessor = contentRepository.photoProcessor,
            sharpening = photoSharpening,
            noiseReduction = photoNoiseReduction,
            chromaNoiseReduction = photoChromaNoiseReduction
        )
    }

    /**
     * 生成预览缓存 key
     */
    private fun previewCacheKey(mediaData: MediaData, metadata: MediaMetadata, showOrigin: Boolean, maxEdge: Int = 4096): String {
        val photoId = mediaData.id
        val metadataHash = metadata.toJson().hashCode()
        val refreshKey = photoRefreshKeys[photoId] ?: 0L
        return if (showOrigin) {
            "${photoId}_${refreshKey}"
        } else if (maxEdge < 4096) {
            "${photoId}_${metadataHash}_${refreshKey}_${maxEdge}"
        } else {
            "${photoId}_${metadataHash}_${refreshKey}"
        }
    }

    private fun detailCacheKey(mediaData: MediaData, metadata: MediaMetadata, showOrigin: Boolean): String {
        return "detail_${previewCacheKey(mediaData, metadata, showOrigin)}"
    }

    private fun shouldUseHdrDetail(metadata: MediaMetadata): Boolean {
        return metadata.manualHdrEffectEnabled
    }

    /**
     * 清除指定照片的预览缓存
     */
    fun invalidatePreviewCache(photoId: String) {
        val snapshot = previewBitmapCache.snapshot()
        snapshot.keys.filter { it.startsWith("${photoId}_") }.forEach {
            previewBitmapCache.remove(it)
        }
        val detailSnapshot = detailBitmapCache.snapshot()
        detailSnapshot.keys.filter { it.contains("detail_${photoId}_") }.forEach {
            detailBitmapCache.remove(it)
        }
    }

    /**
     * 获取应用 LUT 和边框后的预览 Bitmap
     */
    suspend fun getPreviewBitmap(
        photo: MediaData,
        useGlobalEdit: Boolean = false,
        showOrigin: Boolean = false,
        bitmap: Bitmap? = null,
        ignoreCrop: Boolean = false,
        ignoreDenoise: Boolean = false,
        recipeParamsOverride: ColorRecipeParams? = null,
        maxEdge: Int = 4096
    ): Bitmap? {
        if (photo.isVideo) return null
        return withContext(Dispatchers.IO) {
            try {
                val context = getApplication<Application>()

                var finalMetadata: MediaMetadata
                var finalS = 0f
                var finalNR = 0f
                var finalCNR = 0f

                val metadata =
                    photo.metadata
                        ?: photo.relatedPhoto?.metadata
                        ?: GalleryManager.loadMetadata(getApplication(), photo.id)
                        ?: MediaMetadata()

                if (useGlobalEdit) {
                    finalMetadata = (currentMediaMetadata ?: metadata).copy(
                        lutId = editLutId.value,
                        frameId = editFrameId.value,
                        colorRecipeParams = recipeParamsOverride ?: editPhotoRecipeParams.value ?: editLutRecipeParams.value,
                        sharpening = editSharpening.value,
                        noiseReduction = editNoiseReduction.value,
                        chromaNoiseReduction = editChromaNoiseReduction.value,
                        rawDenoiseValue = editRawDenoise.value,
                        rawExposureCompensation = editRawExposureCompensation.value,
                        rawBlackPointCorrection = editRawBlackPointCorrection.value,
                        rawWhitePointCorrection = editRawWhitePointCorrection.value,
                        rawDcpId = editRawDcpId.value,
                        computationalAperture = editComputationalAperture.value,
                        focusPointX = editFocusPointX.value,
                        focusPointY = editFocusPointY.value,
                        postCropRegion = editCropRect.value?.let { rectF ->
                            val cw = photo.metadata?.width ?: photo.width
                            val ch = photo.metadata?.height ?: photo.height
                            android.graphics.Rect(
                                (rectF.left * cw).roundToInt(),
                                (rectF.top * ch).roundToInt(),
                                (rectF.right * cw).roundToInt(),
                                (rectF.bottom * ch).roundToInt()
                            )
                        }
                    )
                    
                    if (ignoreCrop) {
                        finalMetadata = finalMetadata.copy(postCropRegion = null)
                    }
                    finalS = editSharpening.value
                    finalNR = editNoiseReduction.value
                    finalCNR = editChromaNoiseReduction.value
                } else {
                    finalMetadata = metadata

                    if (!ignoreDenoise) {
                        finalS = finalMetadata.sharpening
                            ?: (if (finalMetadata.isImported) 0f else sharpening.value)
                        finalNR =
                            finalMetadata.noiseReduction
                                ?: (if (finalMetadata.isImported) 0f else noiseReduction.value)
                        finalCNR = finalMetadata.chromaNoiseReduction
                            ?: (if (finalMetadata.isImported) 0f else chromaNoiseReduction.value)
                    }
                        
                    if (ignoreCrop) {
                        finalMetadata = finalMetadata.copy(postCropRegion = null)
                    }
                }

                // 使用多级缓存优化性能
                val previewCacheKey = previewCacheKey(photo, finalMetadata, showOrigin, maxEdge)

                val cached = previewBitmapCache.get(previewCacheKey)
                if (cached != null && !cached.isRecycled) {
                    return@withContext cached
                }

                // 2. 原始底图缓存（按 maxEdge 加载，快速预览走小尺寸，正式预览走高分辨率）
                val currentBitmap = bitmap ?: if (selectedTab == GalleryTab.PHOTON) {
                    GalleryManager.loadBitmap(context, photo.id, maxEdge)
                } else {
                    GalleryManager.loadBitmap(context, photo.id, maxEdge)
                        ?: GalleryManager.loadBitmap(context, photo.uri, maxEdge)
                } ?: return@withContext null

                // 只在全分辨率路径下缓存原始底图（避免低分辨率污染 origin 缓存）
                if (maxEdge >= 4096) {
                    previewBitmapCache.put(previewCacheKey(photo, finalMetadata, true), currentBitmap)
                }

                if (showOrigin) {
                    currentBitmap
                } else {
                    // 预览生成
                    val result = contentRepository.photoProcessor.processBitmap(
                        context, photo.id, currentBitmap, finalMetadata,
                        finalS, finalNR, finalCNR,
                        false
                    )
                    // 只在全分辨率路径下更新亮度估计（结果更准确）
                    if (maxEdge >= 4096) {
                        currentBrightness[photo.id] = estimateAverageBrightness(result)
                    }
                    // 存入缓存
                    previewBitmapCache.put(previewCacheKey, result)

                    result
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to create preview", e)
                null
            }
        }
    }

    suspend fun getDetailBitmap(
        photo: MediaData,
    ): Bitmap? {
        if (photo.isVideo) return null
        return withContext(Dispatchers.IO) {
            try {
                val context = getApplication<Application>()
                val metadata =
                    photo.metadata
                        ?: photo.relatedPhoto?.metadata
                        ?: GalleryManager.loadMetadata(getApplication(), photo.id)
                        ?: MediaMetadata()

                val detailCacheKey = detailCacheKey(photo, metadata, false)
                val shouldUseHdrDetail = shouldUseHdrDetail(metadata)

                // HDR detail must be derived from the original render path, not bokeh.jpg,
                // otherwise SDR bokeh cache would override HDR-capable sources.

                if (!shouldUseHdrDetail) {
                    return@withContext null
                }

                if (GalleryManager.isHdrWorkInFlight(photo.id)) {
                    PLog.d(TAG, "getDetailBitmap: HDR work in flight for ${photo.id}, using preview fallback")
                    return@withContext null
                }

                val cachedDetail = detailBitmapCache.get(detailCacheKey)
                if (cachedDetail != null && !cachedDetail.isRecycled) {
                    PLog.d(TAG, "getDetailBitmap: hit detail cache for ${photo.id}")
                    return@withContext cachedDetail
                }

                val detailFile = GalleryManager.getDetailHdrFile(context, photo.id)
                if (detailFile.exists()) {
                    val diskCached = GalleryManager.loadBitmap(context, Uri.fromFile(detailFile), preserveHdr = true)
                    if (diskCached != null) {
                        detailBitmapCache.put(detailCacheKey, diskCached)
//                            PLog.d(
//                                TAG,
//                                "getDetailBitmap: using disk detail HDR cache for ${photo.id}, hasGainmap=${UltraHdrWriter.hasGainmap(diskCached)}"
//                            )
                        return@withContext diskCached
                    }
                }

                return@withContext null
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to create detail bitmap", e)
                null
            }
        }
    }

    suspend fun evaluatePhotoWithAi(photo: MediaData): Result<AiPhotoEvaluation> = withContext(Dispatchers.IO) {
        try {
            val cacheKey = aiEvaluationCacheKey(photo)
            aiEvaluationCache[cacheKey]?.let {
                return@withContext Result.success(it)
            }

            val bitmap = getPreviewBitmap(
                photo = photo,
                showOrigin = false,
                ignoreDenoise = true,
                maxEdge = 1024
            ) ?: return@withContext Result.failure(IllegalStateException("Unable to load photo preview"))

            val context = getApplication<Application>()
            val client = OpenAIApiClient()
            client.initialize(context)
            val result = client.evaluateImageQuality(
                bitmap = bitmap,
                localeTag = Locale.getDefault().toLanguageTag()
            )
            result.onSuccess { aiEvaluationCache[cacheKey] = it }
            result.onFailure { PLog.e(TAG, "AI photo evaluation request failed", it) }
            result
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to evaluate photo with AI", e)
            Result.failure(e)
        }
    }

    private fun aiEvaluationCacheKey(photo: MediaData): String {
        val metadata = photo.metadata ?: photo.relatedPhoto?.metadata
        return listOf(
            photo.id,
            photo.uri.toString(),
            photo.sourceUri?.toString().orEmpty(),
            photo.thumbnailUri.toString(),
            photo.displayName,
            photo.dateAdded.toString(),
            photo.size.toString(),
            photo.width.toString(),
            photo.height.toString(),
            photo.mimeType.orEmpty(),
            metadata?.hashCode()?.toString().orEmpty()
        ).joinToString("|")
    }

    fun shouldPrioritizeDetailBitmap(photo: MediaData): Boolean {
        val context = getApplication<Application>()
        val metadata =
            photo.metadata
                ?: photo.relatedPhoto?.metadata
                ?: MediaMetadata()
        if (!shouldUseHdrDetail(metadata)) {
            return false
        }
        val detailCacheKey = detailCacheKey(photo, metadata, false)
        val cachedDetail = detailBitmapCache.get(detailCacheKey)
        if (cachedDetail != null && !cachedDetail.isRecycled) {
            return true
        }
        if (GalleryManager.getDetailHdrFile(context, photo.id).exists()) {
            return true
        }
        return metadata.hasEmbeddedGainmap
    }

    /**
     * 保存编辑（只更新元数据，不修改原图）
     */
    fun saveEditMetadata(photo: MediaData, onComplete: (Boolean) -> Unit = {}) {
        // 检查 VIP 权限
        val currentLut = availableLuts.find { it.id == editLutId.value }
        if (currentLut?.isVip == true && !isPurchased.value) {
            showPaymentDialog = true
            onComplete(false)
            return
        }

        viewModelScope.launch {
            try {
                val context = getApplication<Application>()

                val targetPhotoId = if (selectedTab == GalleryTab.SYSTEM) {
                    photo.relatedPhoto?.id ?: run {
                        val importedId = GalleryManager.importPhoto(
                            context,
                            photo.uri,
                            editLutId.value
                        )
                        if (importedId != null) {
                            photo.metadata = GalleryManager.loadMetadata(context, importedId)
                            currentMediaMetadata = photo.metadata
                        }
                        importedId
                    }
                } else {
                    photo.id
                }

                if (targetPhotoId == null) {
                    onComplete(false)
                    return@launch
                }

                val w = photo.metadata?.width ?: photo.width
                val h = photo.metadata?.height ?: photo.height
                val finalCropRegion = editCropRect.value?.let { rectF ->
                    android.graphics.Rect(
                        (rectF.left * w).roundToInt(),
                        (rectF.top * h).roundToInt(),
                        (rectF.right * w).roundToInt(),
                        (rectF.bottom * h).roundToInt()
                    )
                }
                
                val metadata = currentMediaMetadata?.copy(
                    lutId = editLutId.value,
                    frameId = editFrameId.value,
                    colorRecipeParams = editPhotoRecipeParams.value,
                    sharpening = editSharpening.value,
                    noiseReduction = editNoiseReduction.value,
                    chromaNoiseReduction = editChromaNoiseReduction.value,
                    rawDenoiseValue = editRawDenoise.value,
                    rawExposureCompensation = editRawExposureCompensation.value,
                    rawBlackPointCorrection = editRawBlackPointCorrection.value,
                    rawWhitePointCorrection = editRawWhitePointCorrection.value,
                    rawDcpId = editRawDcpId.value,
                    computationalAperture = editComputationalAperture.value,
                    focusPointX = editFocusPointX.value,
                    focusPointY = editFocusPointY.value,
                    postCropRegion = finalCropRegion
                ) ?: run {
                    onComplete(false)
                    return@launch
                }
                val success = GalleryManager.saveMetadata(context, targetPhotoId, metadata)

                if (success) {
                    currentMediaMetadata = metadata

                    // 如果是新导入的，刷相册列表
                    if (selectedTab == GalleryTab.SYSTEM) {
                        loadPhotos()
                    } else {
                        // 更新 photos 列表中对应照片的 metadata，触发 UI 刷新
                        val updatedPhotos = _photos.value.map { p ->
                            if (p.id == targetPhotoId) {
                                p.copy(metadata = metadata)
                            } else {
                                p
                            }
                        }
                        _photos.value = updatedPhotos
                    }

                    // 同步更新 latestPhoto
                    if (_latestPhoto.value?.id == targetPhotoId) {
                        _latestPhoto.value = _latestPhoto.value?.copy(metadata = metadata)
                    }

                    exitEditMode()
                    invalidatePreviewCache(targetPhotoId)
                    GalleryManager.deleteDetailHdrFile(context, targetPhotoId)
                    GalleryManager.queueDetailHdrCacheBuild(
                        context = context,
                        photoId = targetPhotoId,
                        metadata = metadata,
                        sharpening = metadata.sharpening ?: 0f,
                        noiseReduction = metadata.noiseReduction ?: 0f,
                        chromaNoiseReduction = metadata.chromaNoiseReduction ?: 0f
                    )
                }
                onComplete(success)
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to save metadata", e)
                onComplete(false)
            }
        }
    }

    /**
     * 刷新 RAW 照片的预览图
     */
    fun refreshRawPreview(photo: MediaData, onComplete: (Boolean) -> Unit = {}) {
        if (refreshingPhotos.contains(photo.id)) return

        viewModelScope.launch {
            refreshingPhotos.add(photo.id)
            try {
                val context = getApplication<Application>()
                val result = GalleryManager.refreshRawPreview(
                    context,
                    photo.id,
                    RawProcessingPreferences.DROMode.valueOf(droMode.value)
                )
                if (result != null) {
                    // 更新刷新密钥以强制 UI 重新加载
                    photoRefreshKeys[photo.id] = System.currentTimeMillis()
                    invalidatePreviewCache(photo.id)

                    // 触发列表更新
                    val updatedPhotos = _photos.value.map { p ->
                        if (p.id == photo.id) {
                            p.copy()
                        } else {
                            p
                        }
                    }
                    _photos.value = updatedPhotos
                    onComplete(true)
                } else {
                    onComplete(false)
                }
            } finally {
                refreshingPhotos.remove(photo.id)
            }
        }
    }

    /**
     * 导出照片到公共目录（带 LUT 烘焙）
     */
    fun exportPhoto(
        photo: MediaData,
        bitmap: Bitmap? = null,
        suffix: String? = null,
        onComplete: (Boolean) -> Unit = {}
    ) {
        if (photo.isVideo) {
            onComplete(false)
            return
        }
        viewModelScope.launch {
            var photoId = photo.id
            if (selectedTab == GalleryTab.SYSTEM) {
                photoId = photo.relatedPhoto?.id ?: photoId
            }
            val metadata = GalleryManager.loadMetadata(getApplication(), photoId) ?: photo.metadata
            ?: MediaMetadata()
            val context = getApplication<Application>()
            val success = GalleryManager.exportPhoto(
                context, photoId, bitmap, contentRepository.photoProcessor, metadata,
                sharpening.value, noiseReduction.value,
                chromaNoiseReduction.value, photoQuality.firstOrNull() ?: 95, suffix
            )
            if (success) {
                exitEditMode()
                loadPhotos()
            }
            onComplete(success)
        }
    }

    fun exportDng(
        photo: MediaData,
        onComplete: (Boolean) -> Unit = {}
    ) {
        if (photo.isVideo) {
            onComplete(false)
            return
        }
        viewModelScope.launch {
            var photoId = photo.id
            if (selectedTab == GalleryTab.SYSTEM) {
                photoId = photo.relatedPhoto?.id ?: photoId
            }
            val context = getApplication<Application>()
            val dngFile = GalleryManager.getDngFile(context, photoId)
            val metadata = GalleryManager.loadMetadata(context, photoId) ?: photo.metadata ?: MediaMetadata()
            val success = if (dngFile.exists() && dngFile.length() > 0L) {
                GalleryManager.exportDng(context, photoId, dngFile, metadata)
                true
            } else {
                false
            }
            if (success) {
                loadPhotos()
            }
            onComplete(success)
        }
    }

    /**
     * 批量导出选中的照片
     */
    fun exportSelectedPhotos(onComplete: (Int) -> Unit = {}) {
        if (selectedPhotos.isEmpty()) return

        viewModelScope.launch {
            _isExporting.value = true
            val toExport = selectedPhotos.filter { it.isImage }
            val total = toExport.size
            exportProgress = 0 to total
            
            try {
                val context = getApplication<Application>()
                val quality = photoQuality.firstOrNull() ?: 95
                
                withContext(Dispatchers.IO) {
                    toExport.forEachIndexed { index, photo ->
                        var photoId = photo.id
                        // 如果在系统相册 Tab，导出其关联的照片
                        if (selectedTab == GalleryTab.SYSTEM) {
                            photoId = photo.relatedPhoto?.id ?: return@forEachIndexed
                        }
                        val metadata = GalleryManager.loadMetadata(context, photoId) ?: photo.metadata ?: MediaMetadata()
                        GalleryManager.exportPhoto(
                            context, photoId, null, contentRepository.photoProcessor, metadata,
                            sharpening.value, noiseReduction.value,
                            chromaNoiseReduction.value, quality
                        )
                        
                        withContext(Dispatchers.Main) {
                            exportProgress = (index + 1) to total
                        }
                    }
                }
                
                exitSelectionMode()
                loadPhotos()
                onComplete(total)
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to batch export photos", e)
                onComplete(0)
            } finally {
                _isExporting.value = false
                exportProgress = 0 to 0
            }
        }
    }

    private data class ShareRequest(val uri: Uri, val mimeType: String)

    private suspend fun prepareShareRequest(photo: MediaData): ShareRequest? = withContext(Dispatchers.IO) {
        try {
            val context = getApplication<Application>()
            if (photo.isVideo) {
                val shareUri = photo.sourceUri ?: photo.metadata?.sourceUri?.toUri() ?: photo.uri
                return@withContext ShareRequest(
                    uri = shareUri,
                    mimeType = photo.mimeType ?: photo.metadata?.mimeType ?: "video/*"
                )
            }
            val metadata =
                photo.metadata ?: GalleryManager.loadMetadata(context, photo.id) ?: MediaMetadata()

            // 处理照片：跟随用户设置
            val processedBitmap = contentRepository.photoProcessor.process(
                context, photo.id, metadata,
                sharpening.value, noiseReduction.value, chromaNoiseReduction.value
            ) ?: return@withContext null

            // 保存到缓存目录
            val sharedDir = File(context.cacheDir, "shared")
            if (!sharedDir.exists()) sharedDir.mkdirs()

            val sharedFile = File(sharedDir, "share_${photo.id}.jpg")
            FileOutputStream(sharedFile).use { out ->
                // 使用用户设置的照片质量
                processedBitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    photoQuality.firstOrNull() ?: 95,
                    out
                )
            }

            processedBitmap.recycle()

            ShareRequest(
                uri = FileProvider.getUriForFile(
                    context,
                    "${context.packageName}.fileprovider",
                    sharedFile
                ),
                mimeType = "image/jpeg"
            )
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to prepare shared photo", e)
            null
        }
    }

    /**
     * 批量导入照片
     */
    fun importPhotos(uris: List<Uri>) {
        if (uris.isEmpty()) return

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val context = getApplication<Application>()
                var successCount = 0

                withContext(Dispatchers.IO) {
                    val lutId = defaultLutId.firstOrNull() ?: contentRepository.getAvailableLuts()
                        .firstOrNull { it.isDefault }?.id
                    val computationalAperture = defaultVirtualAperture.firstOrNull()?.let { if (it > 0f) it else null }
                    uris.forEach { uri ->
                        val photoId = GalleryManager.importPhoto(context, uri, lutId, computationalAperture)
                        if (photoId != null) {
                            successCount++
                        }
                    }
                }

                if (successCount > 0) {
                    loadPhotos()
                }
                PLog.d(TAG, "Imported $successCount of ${uris.size} photos")
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to import photos", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        contentRepository.lutManager.clearCache()
    }

    /**
     * 发起购买
     */
    fun purchase(activity: android.app.Activity) {
        billingManager.purchase(activity)
    }

    /**
     * 获取自定义导入管理器
     */
    fun getCustomImportManager() = contentRepository.getCustomImportManager()

    /**
     * 估计 Bitmap 的平均亮度（调试用）
     */
    private fun estimateAverageBrightness(bitmap: Bitmap): Float {
        return try {
            // 缩小尺寸以快速计算
            val scaledBitmap = bitmap.scale(64, 64, false)
            val pixels = IntArray(64 * 64)
            scaledBitmap.getPixels(pixels, 0, 64, 0, 0, 64, 64)

            var totalLuma = 0f
            for (pixel in pixels) {
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                totalLuma += (0.2126f * r + 0.7152f * g + 0.0722f * b)
            }
            scaledBitmap.recycle()
            totalLuma / (64 * 64) / 255f
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to estimate brightness", e)
            0f
        }
    }

    fun canToggleManualHdrEnhance(photo: MediaData): Boolean {
        if (photo.isVideo) return false
        val metadata = photo.metadata ?: photo.relatedPhoto?.metadata ?: return false
        return metadata.hasEmbeddedGainmap ||
            metadata.dynamicRangeProfile == "HLG10" ||
            GalleryManager.getDngFile(getApplication(), photo.id).exists() ||
            GalleryManager.getPhotoFile(getApplication(), photo.id).exists()
    }

    fun isManualHdrEnhanceEnabled(photo: MediaData): Boolean {
        if (photo.isVideo) return false
        val metadata = photo.metadata ?: photo.relatedPhoto?.metadata ?: return false
        return metadata.manualHdrEffectEnabled
    }

    fun getManualHdrStrength(photo: MediaData): Float {
        val metadata = photo.metadata ?: photo.relatedPhoto?.metadata
        return HdrGainmapStrength.coerce(metadata?.hdrEffectStrength)
    }

    fun toggleManualHdrEnhance(photo: MediaData, onComplete: (Boolean) -> Unit = {}) {
        if (!canToggleManualHdrEnhance(photo)) {
            onComplete(false)
            return
        }
        viewModelScope.launch {
            try {
                val context = getApplication<Application>()
                val currentMetadata = photo.metadata ?: photo.relatedPhoto?.metadata ?: run {
                    onComplete(false)
                    return@launch
                }
                val updatedMetadata = currentMetadata.copy(
                    manualHdrEffectEnabled = !currentMetadata.manualHdrEffectEnabled
                )
                val success = GalleryManager.saveMetadata(context, photo.id, updatedMetadata)
                if (success) {
                    val updatedPhotos = _photos.value.map { p ->
                        if (p.id == photo.id) p.copy(metadata = updatedMetadata) else p
                    }
                    _photos.value = updatedPhotos
                    if (_latestPhoto.value?.id == photo.id) {
                        _latestPhoto.value = _latestPhoto.value?.copy(metadata = updatedMetadata)
                    }
                    if (currentPhotoMetadataId == photo.id) {
                        currentMediaMetadata = updatedMetadata
                    }
                    invalidatePreviewCache(photo.id)
                    if (updatedMetadata.manualHdrEffectEnabled) {
                        GalleryManager.queueDetailHdrCacheBuild(
                            context = context,
                            photoId = photo.id,
                            metadata = updatedMetadata,
                            sharpening = updatedMetadata.sharpening ?: 0f,
                            noiseReduction = updatedMetadata.noiseReduction ?: 0f,
                            chromaNoiseReduction = updatedMetadata.chromaNoiseReduction ?: 0f
                        )
                    } else {
                        GalleryManager.deleteDetailHdrFile(context, photo.id)
                    }
                    photoRefreshKeys[photo.id] = System.currentTimeMillis()
                }
                onComplete(success)
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to toggle manual HDR enhance", e)
                onComplete(false)
            }
        }
    }

    fun setManualHdrStrength(
        photo: MediaData,
        strength: Float,
        onComplete: (Boolean) -> Unit = {}
    ) {
        if (!canToggleManualHdrEnhance(photo)) {
            onComplete(false)
            return
        }
        viewModelScope.launch {
            try {
                val context = getApplication<Application>()
                val currentMetadata = photo.metadata ?: photo.relatedPhoto?.metadata ?: run {
                    onComplete(false)
                    return@launch
                }
                val updatedMetadata = currentMetadata.copy(
                    manualHdrEffectEnabled = true,
                    hdrEffectStrength = HdrGainmapStrength.coerce(strength)
                )
                val success = GalleryManager.saveMetadata(context, photo.id, updatedMetadata)
                if (success) {
                    val updatedPhotos = _photos.value.map { p ->
                        if (p.id == photo.id) p.copy(metadata = updatedMetadata) else p
                    }
                    _photos.value = updatedPhotos
                    if (_latestPhoto.value?.id == photo.id) {
                        _latestPhoto.value = _latestPhoto.value?.copy(metadata = updatedMetadata)
                    }
                    if (currentPhotoMetadataId == photo.id) {
                        currentMediaMetadata = updatedMetadata
                    }
                    invalidatePreviewCache(photo.id)
                    GalleryManager.deleteDetailHdrFile(context, photo.id)
                    GalleryManager.queueDetailHdrCacheBuild(
                        context = context,
                        photoId = photo.id,
                        metadata = updatedMetadata,
                        sharpening = updatedMetadata.sharpening ?: 0f,
                        noiseReduction = updatedMetadata.noiseReduction ?: 0f,
                        chromaNoiseReduction = updatedMetadata.chromaNoiseReduction ?: 0f
                    )
                    photoRefreshKeys[photo.id] = System.currentTimeMillis()
                }
                onComplete(success)
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to update manual HDR strength", e)
                onComplete(false)
            }
        }
    }
}
