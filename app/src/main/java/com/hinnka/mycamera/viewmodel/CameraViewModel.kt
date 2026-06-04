package com.hinnka.mycamera.viewmodel

import android.app.Application
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.hinnka.mycamera.camera.*
import com.hinnka.mycamera.data.ContentRepository
import com.hinnka.mycamera.data.AiFocusTargetMode
import com.hinnka.mycamera.data.CameraFeaturePreferencesUpdate
import com.hinnka.mycamera.data.PreferenceUpdateValue
import com.hinnka.mycamera.data.UserPreferences
import com.hinnka.mycamera.data.VolumeKeyAction
import com.hinnka.mycamera.frame.FrameEditorDraft
import com.hinnka.mycamera.frame.FrameInfo
import com.hinnka.mycamera.frame.FramePreviewFactory
import com.hinnka.mycamera.gallery.GalleryManager
import com.hinnka.mycamera.gallery.MediaMetadata
import com.hinnka.mycamera.lut.BaselineColorCorrectionTarget
import com.hinnka.mycamera.lut.BakedLutExporter
import com.hinnka.mycamera.lut.LutConfig
import com.hinnka.mycamera.lut.LutConverter
import com.hinnka.mycamera.lut.LutInfo
import com.hinnka.mycamera.lut.getBaselineColorCorrectionConfig
import com.hinnka.mycamera.lut.creator.LutGenerator
import com.hinnka.mycamera.lut.creator.OpenAIApiClient
import com.hinnka.mycamera.model.ColorRecipeParams
import com.hinnka.mycamera.model.LutSelectorMode
import com.hinnka.mycamera.model.SafeImage
import com.hinnka.mycamera.phantom.PhantomWidgetProvider
import com.hinnka.mycamera.raw.ColorSpace
import com.hinnka.mycamera.raw.DcpProfileParser
import com.hinnka.mycamera.raw.DcpInfo
import com.hinnka.mycamera.raw.RawDemosaicProcessor
import com.hinnka.mycamera.color.TransferCurve
import com.hinnka.mycamera.model.EffectParams
import com.hinnka.mycamera.raw.RawProcessingPreferences
import com.hinnka.mycamera.raw.RawProfile
import com.hinnka.mycamera.raw.SpectralFilmProfile
import com.hinnka.mycamera.screencapture.PhantomPipCrop
import com.hinnka.mycamera.ui.camera.CameraGLSurfaceView
import com.hinnka.mycamera.utils.*
import com.hinnka.mycamera.video.CaptureMode
import com.hinnka.mycamera.video.VideoAudioInputManager
import com.hinnka.mycamera.video.VideoAudioInputOption
import com.hinnka.mycamera.video.VideoAspectRatio
import com.hinnka.mycamera.video.VideoBitratePreset
import com.hinnka.mycamera.video.VideoFpsPreset
import com.hinnka.mycamera.video.VideoLogProfile
import com.hinnka.mycamera.video.VideoResolutionPreset
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first
import kotlin.math.roundToInt
import kotlinx.coroutines.flow.*
import java.io.File
import java.util.UUID
import kotlin.math.abs

data class MultipleExposureFrame(
    val index: Int,
    val file: File
)

data class MultipleExposureSessionState(
    val enabled: Boolean = false,
    val sessionId: String? = null,
    val targetCount: Int = 2,
    val capturedCount: Int = 0,
    val frames: List<MultipleExposureFrame> = emptyList(),
    val isProcessing: Boolean = false,
    val previewBitmap: Bitmap? = null
) {
    val isSessionActive: Boolean
        get() = sessionId != null

    val canFinish: Boolean
        get() = capturedCount >= 2 && !isProcessing
}

private fun resolvePreviewBaselineTarget(useRaw: Boolean): BaselineColorCorrectionTarget? {
    return if (useRaw) null else BaselineColorCorrectionTarget.JPG
}

private fun UserPreferences.getBaselineLutId(target: BaselineColorCorrectionTarget): String? {
    return getBaselineColorCorrectionConfig(target).lutId
}

private data class RawSpectralFilmSettings(
    val enabled: Boolean,
    val stock: String?,
    val print: String?
)

private data class PresetMatchSnapshot(
    val lutId: String?,
    val colorRecipe: ColorRecipeParams,
    val effects: com.hinnka.mycamera.model.EffectParams,
    val aspectRatio: String,
    val useRaw: Boolean,
    val useMFNR: Boolean,
    val useMFSR: Boolean,
    val frameId: String?,
    val rawDcpId: String?,
    val rawSpectralFilmEnabled: Boolean,
    val rawSpectralFilmStock: String?,
    val rawSpectralFilmPrint: String?,
    val rawDROMode: String,
    val jpgBaselineLutId: String?,
    val rawBaselineLutId: String?,
    val phantomBaselineLutId: String?
) {
    fun matches(preset: com.hinnka.mycamera.model.CameraPreset): Boolean {
        return lutId == preset.lutId &&
            colorRecipe == preset.colorRecipe &&
            effects == preset.effects &&
            aspectRatio == preset.aspectRatio &&
            useRaw == preset.useRaw &&
            useMFNR == preset.useMFNR &&
            useMFSR == preset.useMFSR &&
            frameId == preset.frameId &&
            rawDcpId == preset.rawDcpId &&
            rawSpectralFilmEnabled == preset.rawSpectralFilmEnabled &&
            rawSpectralFilmStock == preset.rawSpectralFilmStock &&
            rawSpectralFilmPrint == preset.rawSpectralFilmPrint &&
            rawDROMode == preset.rawDROMode &&
            jpgBaselineLutId == preset.jpgBaselineLutId &&
            rawBaselineLutId == preset.rawBaselineLutId &&
            phantomBaselineLutId == preset.phantomBaselineLutId
    }
}

private data class ActivePresetMatchState(
    val prefs: UserPreferences,
    val presets: List<com.hinnka.mycamera.model.CameraPreset>,
    val aspectRatio: String,
    val lutId: String,
    val recipe: ColorRecipeParams,
    val effects: com.hinnka.mycamera.model.EffectParams
)

private data class SettingValue<T>(val value: T)

private data class CameraFeatureUpdate(
    val lutId: SettingValue<String?>? = null,
    val colorRecipe: SettingValue<ColorRecipeParams>? = null,
    val effects: SettingValue<com.hinnka.mycamera.model.EffectParams>? = null,
    val aspectRatio: SettingValue<AspectRatio>? = null,
    val useRaw: SettingValue<Boolean>? = null,
    val useMFNR: SettingValue<Boolean>? = null,
    val useMFSR: SettingValue<Boolean>? = null,
    val frameId: SettingValue<String?>? = null,
    val rawDcpId: SettingValue<String?>? = null,
    val rawSpectralFilmEnabled: SettingValue<Boolean>? = null,
    val rawSpectralFilmStock: SettingValue<String?>? = null,
    val rawSpectralFilmPrint: SettingValue<String?>? = null,
    val droMode: SettingValue<String>? = null,
    val jpgBaselineLutId: SettingValue<String?>? = null,
    val rawBaselineLutId: SettingValue<String?>? = null,
    val phantomBaselineLutId: SettingValue<String?>? = null,
    val activePresetId: SettingValue<String?>? = null
)

/**
 * 相机 ViewModel
 * 使用 Camera2Controller 支持隐藏摄像头
 */
@OptIn(ExperimentalCoroutinesApi::class)
class CameraViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "CameraViewModel"
    }

    private val cameraController = Camera2Controller(application)


    // 内容仓库（单例，与 GalleryViewModel 共享）
    private val contentRepository = ContentRepository.getInstance(application)

    private val userPreferencesRepository = contentRepository.userPreferencesRepository

    // 计费管理器
    private val billingManager = com.hinnka.mycamera.billing.BillingManagerImpl(application)
    val isPurchased = billingManager.isPurchased

    // 快门音效播放器
    private val shutterSoundPlayer = ShutterSoundPlayer(application)

    // 震动辅助类
    private val vibrationHelper = VibrationHelper(application)
    private val videoAudioInputManager = VideoAudioInputManager(application)

    private val locationManager = LocationManager(application)

    val state: StateFlow<CameraState> = cameraController.state
    val livePhotoRecorder get() = cameraController.livePhotoRecorder
    val videoRecorder get() = cameraController.videoRecorder

    // 照片保存完成事件
    private val _imageSavedEvent = MutableSharedFlow<Unit>()
    val imageSavedEvent: SharedFlow<Unit> = _imageSavedEvent.asSharedFlow()

    private val _isInitialized = MutableStateFlow(false)
    val isInitialized = _isInitialized.asStateFlow()

    private val _canStartShutterAnimation = MutableStateFlow(false)
    val canStartShutterAnimation = _canStartShutterAnimation.asStateFlow()

    // LUT 相关状态
    var currentLutConfig: LutConfig? by mutableStateOf(null)
        private set

    var currentBaselineLutConfig: LutConfig? by mutableStateOf(null)
        private set

    var currentLutId = MutableStateFlow("standard")
        private set

    @OptIn(ExperimentalCoroutinesApi::class)
    var currentRecipeParams = currentLutId.flatMapLatest { id ->
        contentRepository.lutManager.getColorRecipeParams(id)
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.Eagerly,
        initialValue = ColorRecipeParams.DEFAULT
    )

    private fun recipeFlowFor(lutId: String): StateFlow<ColorRecipeParams> {
        return contentRepository.lutManager.getColorRecipeParams(lutId).stateIn(
            viewModelScope,
            started = SharingStarted.Lazily,
            initialValue = ColorRecipeParams.DEFAULT,
        )
    }

    val currentEffectParams: StateFlow<com.hinnka.mycamera.model.EffectParams> = userPreferencesRepository.userPreferences
        .map { it.activeEffectParams }
        .stateIn(viewModelScope, SharingStarted.Eagerly, com.hinnka.mycamera.model.EffectParams.DEFAULT)

    val customPresets: StateFlow<List<com.hinnka.mycamera.model.CameraPreset>> = userPreferencesRepository.userPreferences
        .map { it.customPresets }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    // 合并内置预设与用户自定义预设
    val allPresets: StateFlow<List<com.hinnka.mycamera.model.CameraPreset>> = userPreferencesRepository.userPreferences
        .map { prefs ->
            val deletedIds = prefs.deletedBuiltInIds.split(",").filter { it.isNotEmpty() }.toSet()
            val builtInsById = com.hinnka.mycamera.model.CameraPreset.BUILT_IN_PRESETS.associateBy { it.id }
            val orderedPresets = prefs.customPresets
                .filter { it.id !in deletedIds }
                .map { saved ->
                    builtInsById[saved.id]?.let { builtin ->
                        builtin.copy(
                            name = saved.name,
                            lutId = saved.lutId,
                            colorRecipe = saved.colorRecipe,
                            effects = saved.effects,
                            aspectRatio = saved.aspectRatio,
                            useRaw = saved.useRaw,
                            useMFNR = saved.useMFNR,
                            useMFSR = saved.useMFSR,
                            frameId = saved.frameId,
                            rawDcpId = saved.rawDcpId,
                            rawSpectralFilmEnabled = saved.rawSpectralFilmEnabled,
                            rawSpectralFilmStock = saved.rawSpectralFilmStock,
                            rawSpectralFilmPrint = saved.rawSpectralFilmPrint,
                            rawDROMode = saved.rawDROMode,
                            jpgBaselineLutId = saved.jpgBaselineLutId,
                            rawBaselineLutId = saved.rawBaselineLutId,
                            phantomBaselineLutId = saved.phantomBaselineLutId
                        )
                    } ?: saved
                }
            val orderedIds = orderedPresets.map { it.id }.toSet()
            val missingBuiltIns = com.hinnka.mycamera.model.CameraPreset.BUILT_IN_PRESETS
                .filter { it.id !in deletedIds && it.id !in orderedIds }
            val visibleBuiltInIds = com.hinnka.mycamera.model.CameraPreset.BUILT_IN_PRESETS
                .filter { it.id !in deletedIds }
                .map { it.id }
                .toSet()
            val hasCompleteSavedBuiltInOrder = visibleBuiltInIds.isNotEmpty() &&
                visibleBuiltInIds.all { builtInId -> orderedPresets.any { it.id == builtInId } }

            if (hasCompleteSavedBuiltInOrder) {
                orderedPresets + missingBuiltIns
            } else {
                val overridesById = orderedPresets.associateBy { it.id }
                val builtInsWithOverrides = com.hinnka.mycamera.model.CameraPreset.BUILT_IN_PRESETS
                    .filter { it.id !in deletedIds }
                    .map { builtin -> overridesById[builtin.id] ?: builtin }
                val customs = orderedPresets.filter { it.id !in visibleBuiltInIds }
                builtInsWithOverrides + customs
            }
        }
        .stateIn(viewModelScope, SharingStarted.Eagerly, com.hinnka.mycamera.model.CameraPreset.BUILT_IN_PRESETS)

    val activePresetId: StateFlow<String?> = userPreferencesRepository.userPreferences
        .map { it.activePresetId }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    var draftPreset: com.hinnka.mycamera.model.CameraPreset? = null

    @Volatile
    private var isApplyingPreset = false

    fun prepareCurrentSettingsPresetDraft(name: String): com.hinnka.mycamera.model.CameraPreset {
        return com.hinnka.mycamera.model.CameraPreset(
            id = UUID.randomUUID().toString(),
            name = name,
            lutId = currentLutId.value,
            colorRecipe = currentRecipeParams.value,
            effects = currentEffectParams.value,
            aspectRatio = state.value.aspectRatio.name,
            useRaw = useRaw.value,
            useMFNR = useMFNR.value,
            useMFSR = useMFSR.value,
            frameId = currentFrameId,
            rawDcpId = rawDcpId.value,
            rawSpectralFilmEnabled = rawSpectralFilmEnabled.value,
            rawSpectralFilmStock = rawSpectralFilmStock.value,
            rawSpectralFilmPrint = rawSpectralFilmPrint.value,
            rawDROMode = droMode.value,
            jpgBaselineLutId = jpgBaselineLutId.value,
            rawBaselineLutId = rawBaselineLutId.value,
            phantomBaselineLutId = phantomBaselineLutId.value,
            isBuiltIn = false
        ).also {
            draftPreset = it
        }
    }

    fun getMergedRecipeParams(recipe: ColorRecipeParams = currentRecipeParams.value): ColorRecipeParams {
        return currentEffectParams.value.applyTo(recipe)
    }

    fun setEffectParams(effects: com.hinnka.mycamera.model.EffectParams) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(effects = SettingValue(effects))
            )
        }
    }

    fun applyPreset(preset: com.hinnka.mycamera.model.CameraPreset?) {
        viewModelScope.launch {
            isApplyingPreset = true
            try {
                applyCameraFeatureUpdate(
                    preset.toCameraFeatureUpdate().copy(activePresetId = SettingValue(preset?.id)),
                    clearActivePresetOnMismatch = false
                )
            } finally {
                isApplyingPreset = false
            }
        }
    }

    private fun com.hinnka.mycamera.model.CameraPreset?.toCameraFeatureUpdate(): CameraFeatureUpdate {
        val ratio = try {
            AspectRatio.valueOf(this?.aspectRatio ?: AspectRatio.RATIO_4_3.name)
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to apply preset aspectRatio: ${this?.aspectRatio}", e)
            AspectRatio.RATIO_4_3
        }
        return CameraFeatureUpdate(
            lutId = SettingValue(this?.lutId),
            colorRecipe = SettingValue(this?.colorRecipe ?: ColorRecipeParams.DEFAULT),
            effects = SettingValue(this?.effects ?: EffectParams.DEFAULT),
            aspectRatio = SettingValue(ratio),
            useRaw = SettingValue(this?.useRaw ?: false),
            useMFNR = SettingValue(this?.useMFNR ?: false),
            useMFSR = SettingValue(this?.useMFSR ?: false),
            frameId = SettingValue(this?.frameId),
            rawDcpId = SettingValue(this?.rawDcpId),
            rawSpectralFilmEnabled = SettingValue(this?.rawSpectralFilmEnabled ?: false),
            rawSpectralFilmStock = SettingValue(this?.rawSpectralFilmStock),
            rawSpectralFilmPrint = SettingValue(this?.rawSpectralFilmPrint),
            droMode = SettingValue(this?.rawDROMode ?: RawProcessingPreferences.DROMode.OFF.name),
            jpgBaselineLutId = SettingValue(this?.jpgBaselineLutId),
            rawBaselineLutId = SettingValue(this?.rawBaselineLutId),
            phantomBaselineLutId = SettingValue(this?.phantomBaselineLutId)
        )
    }

    private suspend fun applyCameraFeatureUpdate(
        update: CameraFeatureUpdate,
        clearActivePresetOnMismatch: Boolean = true
    ) {
        val prefs = userPreferencesRepository.userPreferences.first()
        var desiredUseRaw = prefs.useRaw
        var desiredUseMFNR = prefs.useMFNR
        var desiredUseMFSR = prefs.useMFSR
        var desiredUseMultipleExposure = prefs.useMultipleExposure

        update.useRaw?.let { desiredUseRaw = it.value }
        update.useMFNR?.let { desiredUseMFNR = it.value }
        update.useMFSR?.let { desiredUseMFSR = it.value }

        if (update.useRaw?.value == true) {
            desiredUseMultipleExposure = false
            desiredUseMFSR = false
        }
        if (update.useMFNR?.value == true) {
            desiredUseMultipleExposure = false
            desiredUseMFSR = false
        }
        if (update.useMFSR?.value == true) {
            if (desiredUseRaw) {
                desiredUseMFSR = false
            } else {
                desiredUseMultipleExposure = false
                desiredUseMFNR = false
            }
        }

        val needsCameraReopen =
            update.aspectRatio != null ||
                update.useRaw != null ||
                update.useMFNR != null ||
                update.useMFSR != null

        update.colorRecipe?.let {
            val recipeLutId = if (update.lutId != null) {
                update.lutId.value ?: "none"
            } else {
                currentLutId.value
            }
            contentRepository.lutManager.saveColorRecipeParams(recipeLutId, it.value)
        }

        update.lutId?.let {
            setLut(it.value, persist = false)
        }

        update.aspectRatio?.let {
            cameraController.setAspectRatio(it.value)
        }

        if (desiredUseMultipleExposure != prefs.useMultipleExposure) {
            if (!desiredUseMultipleExposure) {
                cancelMultipleExposureSession()
            }
        }

        if (update.useRaw != null) {
            cameraController.setUseRaw(desiredUseRaw)
        }
        if (update.useMFNR != null || desiredUseMFNR != prefs.useMFNR) {
            cameraController.setUseMFNR(desiredUseMFNR)
        }
        if (update.useMFSR != null || desiredUseMFSR != prefs.useMFSR) {
            cameraController.setUseMFSR(desiredUseMFSR)
        }

        update.frameId?.let {
            currentFrameId = it.value
        }

        update.rawDcpId?.let {
            prewarmRawDcp(it.value)
        }

        userPreferencesRepository.saveCameraFeaturePreferences(
            CameraFeaturePreferencesUpdate(
                lutId = update.lutId?.let { PreferenceUpdateValue(it.value) },
                effects = update.effects?.let { PreferenceUpdateValue(it.value) },
                aspectRatio = update.aspectRatio?.let { PreferenceUpdateValue(it.value.name) },
                useRaw = update.useRaw?.let { PreferenceUpdateValue(desiredUseRaw) },
                useMFNR = if (update.useMFNR != null || desiredUseMFNR != prefs.useMFNR) {
                    PreferenceUpdateValue(desiredUseMFNR)
                } else {
                    null
                },
                useMFSR = if (update.useMFSR != null || desiredUseMFSR != prefs.useMFSR) {
                    PreferenceUpdateValue(desiredUseMFSR)
                } else {
                    null
                },
                useMultipleExposure = if (desiredUseMultipleExposure != prefs.useMultipleExposure) {
                    PreferenceUpdateValue(desiredUseMultipleExposure)
                } else {
                    null
                },
                frameId = update.frameId?.let { PreferenceUpdateValue(it.value) },
                rawDcpId = update.rawDcpId?.let { PreferenceUpdateValue(it.value) },
                rawSpectralFilmEnabled = update.rawSpectralFilmEnabled?.let { PreferenceUpdateValue(it.value) },
                rawSpectralFilmStock = update.rawSpectralFilmStock?.let { PreferenceUpdateValue(it.value) },
                rawSpectralFilmPrint = update.rawSpectralFilmPrint?.let { PreferenceUpdateValue(it.value) },
                droMode = update.droMode?.let {
                    PreferenceUpdateValue(RawProcessingPreferences.DROMode.fromPersistedName(it.value).name)
                },
                jpgBaselineLutId = update.jpgBaselineLutId?.let { PreferenceUpdateValue(it.value) },
                rawBaselineLutId = update.rawBaselineLutId?.let { PreferenceUpdateValue(it.value) },
                phantomBaselineLutId = update.phantomBaselineLutId?.let { PreferenceUpdateValue(it.value) },
                activePresetId = update.activePresetId?.let { PreferenceUpdateValue(it.value) }
            )
        )

        if (needsCameraReopen) {
            reopenCamera()
        }
        if (clearActivePresetOnMismatch) {
            clearActivePresetIfCurrentSettingsMismatch()
        }
    }

    fun savePreset(preset: com.hinnka.mycamera.model.CameraPreset) {
        viewModelScope.launch {
            val currentList = customPresets.value.toMutableList()
            val index = currentList.indexOfFirst { it.id == preset.id }
            if (index >= 0) {
                currentList[index] = preset
            } else {
                currentList.add(preset)
            }
            userPreferencesRepository.saveCustomPresets(currentList)
            applyPreset(preset)
        }
    }

    fun deletePreset(presetId: String) {
        viewModelScope.launch {
            val currentList = customPresets.value.toMutableList()
            currentList.removeAll { it.id == presetId }
            userPreferencesRepository.saveCustomPresets(currentList)

            // 如果删除的是内置预设，将其加入 deletedBuiltInIds
            val isBuiltIn = com.hinnka.mycamera.model.CameraPreset.BUILT_IN_PRESETS.any { it.id == presetId }
            if (isBuiltIn) {
                val currentDeleted = userPreferencesRepository.userPreferences.first().deletedBuiltInIds
                val deletedList = currentDeleted.split(",").filter { it.isNotEmpty() }.toMutableList()
                if (presetId !in deletedList) {
                    deletedList.add(presetId)
                    userPreferencesRepository.saveDeletedBuiltInIds(deletedList.joinToString(","))
                }
            }

            if (activePresetId.value == presetId) {
                userPreferencesRepository.saveActivePresetId(null)
            }
        }
    }

    fun resetToDefaultPresets() {
        viewModelScope.launch {
            userPreferencesRepository.saveDeletedBuiltInIds("")
            val currentList = customPresets.value.toMutableList()
            currentList.removeAll { it.isBuiltIn || it.id.startsWith("builtin_") }
            userPreferencesRepository.saveCustomPresets(currentList)
        }
    }

    fun savePresetOrder(presets: List<com.hinnka.mycamera.model.CameraPreset>) {
        viewModelScope.launch {
            userPreferencesRepository.saveCustomPresets(presets)
        }
    }

    fun refreshActivePresetMatch() {
        viewModelScope.launch {
            clearActivePresetIfCurrentSettingsMismatch()
        }
    }

    private suspend fun clearActivePresetIfCurrentSettingsMismatch() {
        if (isApplyingPreset) return
        val presetId = activePresetId.value ?: return
        val preset = allPresets.value.firstOrNull { it.id == presetId }
        if (preset == null) {
            userPreferencesRepository.saveActivePresetId(null)
            return
        }
        val snapshot = currentPresetMatchSnapshot()
        if (!snapshot.matches(preset)) {
            PLog.d(TAG, "Active preset [$presetId] no longer matches current settings; showing default preset")
            userPreferencesRepository.saveActivePresetId(null)
        }
    }

    private suspend fun clearActivePresetIfCurrentSettingsMismatch(matchState: ActivePresetMatchState) {
        if (isApplyingPreset) return
        val presetId = matchState.prefs.activePresetId ?: return
        val preset = matchState.presets.firstOrNull { it.id == presetId }
        if (preset == null) {
            userPreferencesRepository.saveActivePresetId(null)
            return
        }
        val snapshot = matchState.toPresetMatchSnapshot()
        if (!snapshot.matches(preset)) {
            PLog.d(TAG, "Active preset [$presetId] no longer matches current settings; showing default preset")
            userPreferencesRepository.saveActivePresetId(null)
        }
    }

    private fun currentPresetMatchSnapshot(): PresetMatchSnapshot {
        return PresetMatchSnapshot(
            lutId = currentLutId.value.takeIf { it != "none" },
            colorRecipe = currentRecipeParams.value,
            effects = currentEffectParams.value,
            aspectRatio = state.value.aspectRatio.name,
            useRaw = useRaw.value,
            useMFNR = useMFNR.value,
            useMFSR = useMFSR.value,
            frameId = currentFrameId,
            rawDcpId = rawDcpId.value,
            rawSpectralFilmEnabled = rawSpectralFilmEnabled.value,
            rawSpectralFilmStock = rawSpectralFilmStock.value,
            rawSpectralFilmPrint = rawSpectralFilmPrint.value,
            rawDROMode = droMode.value,
            jpgBaselineLutId = jpgBaselineLutId.value,
            rawBaselineLutId = rawBaselineLutId.value,
            phantomBaselineLutId = phantomBaselineLutId.value
        )
    }

    private fun ActivePresetMatchState.toPresetMatchSnapshot(): PresetMatchSnapshot {
        return PresetMatchSnapshot(
            lutId = lutId.takeIf { it != "none" },
            colorRecipe = recipe,
            effects = effects,
            aspectRatio = aspectRatio,
            useRaw = prefs.useRaw,
            useMFNR = prefs.useMFNR,
            useMFSR = prefs.useMFSR,
            frameId = prefs.frameId,
            rawDcpId = prefs.rawDcpId,
            rawSpectralFilmEnabled = prefs.rawSpectralFilmEnabled,
            rawSpectralFilmStock = prefs.rawSpectralFilmStock,
            rawSpectralFilmPrint = prefs.rawSpectralFilmPrint,
            rawDROMode = prefs.droMode,
            jpgBaselineLutId = prefs.jpgBaselineLutId,
            rawBaselineLutId = prefs.rawBaselineLutId,
            phantomBaselineLutId = prefs.phantomBaselineLutId
        )
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    val currentBaselineRecipeParams: StateFlow<ColorRecipeParams> =
        userPreferencesRepository.userPreferences.flatMapLatest { prefs ->
            if (shouldUseSpectralFilmPreview(prefs)) {
                return@flatMapLatest flowOf(ColorRecipeParams.DEFAULT)
            }
            val target = resolvePreviewBaselineTarget(prefs.useRaw)
            val lutId = target?.let { prefs.getBaselineLutId(it) }
            if (target == null || lutId == null) {
                flowOf(ColorRecipeParams.DEFAULT)
            } else {
                contentRepository.lutManager.getColorRecipeParams(lutId, target)
            }
        }.stateIn(
            scope = viewModelScope,
            started = SharingStarted.Eagerly,
            initialValue = ColorRecipeParams.DEFAULT
        )

    var availableLutList: List<LutInfo> by mutableStateOf(emptyList())
        private set

    // LUT 预览图缓存（lutId -> 预览Bitmap）
    var previewThumbnail by mutableStateOf<Bitmap?>(null)

    // 是否正在生成预览
    private var isGeneratingPreviews = false

    // 边框相关状态
    var currentFrameId: String? by mutableStateOf(null)
        private set

    var showHistogram by mutableStateOf(true)
        private set

    var availableFrameList: List<FrameInfo> by mutableStateOf(emptyList())
        private set

    var zoomRatioByMain by mutableFloatStateOf(1f)
    var isZooming by mutableStateOf(false)
    val globalMinZoom: Float
        get() = state.value.availableCameras.filter { it.lensType != LensType.FRONT }.minOfOrNull { it.minZoom * it.intrinsicZoomRatio } ?: 1f
    val globalMaxZoom: Float
        get() = state.value.availableCameras.filter { it.lensType != LensType.FRONT }.maxOfOrNull { it.maxZoom * it.intrinsicZoomRatio } ?: 20f

    // 付费弹窗状态
    var showPaymentDialog by mutableStateOf(false)

    var isExpanded by mutableStateOf(false)

    var isAiFocusBusy by mutableStateOf(false)
    private var startupPrewarmJob: Job? = null

    // 新增设置项 StateFlow
    val showLevelIndicator: Flow<Boolean> = userPreferencesRepository.userPreferences.map { it.showLevelIndicator }
    val focusPeakingEnabled: Flow<Boolean> = userPreferencesRepository.userPreferences.map { it.focusPeakingEnabled }
    val aiFocusTargetMode: StateFlow<AiFocusTargetMode> =
        userPreferencesRepository.userPreferences.map { it.aiFocusTargetMode }
            .stateIn(viewModelScope, SharingStarted.Eagerly, AiFocusTargetMode.OFF)
    val aiFocusScoreThreshold: StateFlow<Float> =
        userPreferencesRepository.userPreferences.map { it.aiFocusScoreThreshold }
            .stateIn(viewModelScope, SharingStarted.Eagerly, 0.5f)
    val shutterSoundEnabled: Flow<Boolean> = userPreferencesRepository.userPreferences.map { it.shutterSoundEnabled }
    val vibrationEnabled: Flow<Boolean> = userPreferencesRepository.userPreferences.map { it.vibrationEnabled }
    val volumeKeyAction: StateFlow<VolumeKeyAction> =
        userPreferencesRepository.userPreferences.map { it.volumeKeyAction }
            .stateIn(viewModelScope, SharingStarted.Eagerly, initialValue = VolumeKeyAction.NONE)
    val autoSaveAfterCapture: Flow<Boolean> = userPreferencesRepository.userPreferences.map { it.autoSaveAfterCapture }
    val topSheetAspectRatios: StateFlow<List<AspectRatio>> = userPreferencesRepository.userPreferences
        .map { it.topSheetAspectRatios }
        .stateIn(viewModelScope, SharingStarted.Eagerly, AspectRatio.defaultTopSheetRatios)
    val customAspectRatios: StateFlow<List<AspectRatio>> = userPreferencesRepository.userPreferences
        .map { it.customAspectRatios }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
    val availablePhotoAspectRatios: StateFlow<List<AspectRatio>> = userPreferencesRepository.userPreferences
        .map { AspectRatio.entries + it.customAspectRatios }
        .stateIn(viewModelScope, SharingStarted.Eagerly, AspectRatio.entries)
    val nrLevel: StateFlow<Int> = userPreferencesRepository.userPreferences
        .map { it.nrLevel }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 5)
    val useRaw: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useRaw }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val edgeLevel: StateFlow<Int> = userPreferencesRepository.userPreferences
        .map { it.edgeLevel }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 1)
    val rawSpectralFilmEnabled: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.rawSpectralFilmEnabled }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val rawSpectralFilmStock: StateFlow<String?> = userPreferencesRepository.userPreferences
        .map { it.rawSpectralFilmStock }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val rawSpectralFilmPrint: StateFlow<String?> = userPreferencesRepository.userPreferences
        .map { it.rawSpectralFilmPrint }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val photoQuality: Flow<Int> = userPreferencesRepository.userPreferences.map { it.photoQuality }

    val defaultFocalLength: Flow<Float> = userPreferencesRepository.userPreferences.map { it.defaultFocalLength }
    val customFocalLengths: Flow<List<Float>> = userPreferencesRepository.userPreferences.map { it.customFocalLengths }
    val hiddenFocalLengths: Flow<List<Float>> = userPreferencesRepository.userPreferences.map { it.hiddenFocalLengths }
    val customLensIds: Flow<List<String>> = userPreferencesRepository.userPreferences.map { it.customLensIds }
    val lensIdBlacklist: Flow<List<String>> = userPreferencesRepository.userPreferences.map { it.lensIdBlacklist }
    val userPreferences: StateFlow<UserPreferences> = userPreferencesRepository.userPreferences
        .stateIn(viewModelScope, SharingStarted.Eagerly, UserPreferences())
    val jpgBaselineLutId: StateFlow<String?> = userPreferencesRepository.userPreferences
        .map { it.jpgBaselineLutId }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val rawBaselineLutId: StateFlow<String?> = userPreferencesRepository.userPreferences
        .map { it.rawBaselineLutId }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val phantomBaselineLutId: StateFlow<String?> = userPreferencesRepository.userPreferences
        .map { it.phantomBaselineLutId }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val rawDcpId: StateFlow<String?> = userPreferencesRepository.userPreferences
        .map { it.rawDcpId }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val rawNlmNoiseFactor: StateFlow<Float> = userPreferencesRepository.userPreferences
        .map { it.rawNlmNoiseFactor }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)
    val rawExposureCompensation: StateFlow<Float> = userPreferencesRepository.userPreferences
        .map { it.rawExposureCompensation }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)
    val rawAutoExposure: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.rawAutoExposure }
        .stateIn(viewModelScope, SharingStarted.Eagerly, true)
    val rawMinShutterSpeedNs: StateFlow<Long> = userPreferencesRepository.userPreferences
        .map { it.rawMinShutterSpeedNs }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0L)
    val rawDROEnabled: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.rawDROEnabled }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val rawBlackPointCorrection: StateFlow<Float> = userPreferencesRepository.userPreferences
        .map { it.rawBlackPointCorrection }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)
    val rawWhitePointCorrection: StateFlow<Float> = userPreferencesRepository.userPreferences
        .map { it.rawWhitePointCorrection }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0f)
    val rawAutoWhiteBalanceEstimate: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.rawAutoWhiteBalanceEstimate }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val rawBlackLevelMode: StateFlow<String> = combine(
        state.map { it.currentCameraId }.distinctUntilChanged(),
        userPreferencesRepository.userPreferences
    ) { cameraId, prefs ->
        prefs.rawBlackLevelModes[cameraId] ?: "Default"
    }.stateIn(viewModelScope, SharingStarted.Eagerly, "Default")
    val rawCustomBlackLevel: StateFlow<Float> = combine(
        state.map { it.currentCameraId }.distinctUntilChanged(),
        userPreferencesRepository.userPreferences
    ) { cameraId, prefs ->
        prefs.rawCustomBlackLevels[cameraId] ?: 0f
    }.stateIn(viewModelScope, SharingStarted.Eagerly, 0f)
    val exportDngWithRawExport: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.exportDngWithRawExport }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    var availableDcps: List<DcpInfo> by mutableStateOf(emptyList())
        private set
    val useMFNR: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useMFNR }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val useMultipleExposure: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useMultipleExposure }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val multipleExposureCount: StateFlow<Int> = userPreferencesRepository.userPreferences
        .map { it.multipleExposureCount }
        .stateIn(viewModelScope, SharingStarted.Eagerly, 2)
    val multiFrameCount: StateFlow<Int> = userPreferencesRepository.userPreferences
        .map { it.multiFrameCount }
        .stateIn(viewModelScope, SharingStarted.Eagerly, MultiFrameConfig.DEFAULT_FRAME_COUNT)
    val useMFSR: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useMFSR }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val useLivePhoto: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useLivePhoto }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val enableDevelopAnimation: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.enableDevelopAnimation }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val backgroundImage: StateFlow<String> = userPreferencesRepository.userPreferences
        .map { it.backgroundImage }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "camera_bg")
    val useGpuAcceleration: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useGpuAcceleration }
        .stateIn(viewModelScope, SharingStarted.Eagerly, DeviceUtil.defaultGpuAcceleration)
    val droMode: StateFlow<String> = userPreferencesRepository.userPreferences
        .map { it.droMode }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "OFF")
    val tonemapMode: StateFlow<String> = userPreferencesRepository.userPreferences
        .map { it.tonemapMode }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "FAST")
    val fixTonemapPreview: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.fixTonemapPreview }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val applyUltraHDR: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.applyUltraHDR }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val colorSpace: StateFlow<ColorSpace> = userPreferencesRepository.userPreferences
        .map { it.colorSpace }
        .stateIn(viewModelScope, SharingStarted.Eagerly, ColorSpace.SRGB)
    val logCurve: StateFlow<TransferCurve> = userPreferencesRepository.userPreferences
        .map { it.logCurve }
        .stateIn(viewModelScope, SharingStarted.Eagerly, TransferCurve.SRGB)

    val rawLut: StateFlow<String> = userPreferencesRepository.userPreferences
        .map { prefs ->
            prefs.rawLuts[prefs.logCurve.name] ?: RawProfile.defaultLutFor(prefs.colorSpace, prefs.logCurve)
        }
        .stateIn(viewModelScope, SharingStarted.Eagerly, RawProfile.default.rawLut)
    val rawProfile: StateFlow<RawProfile> = userPreferencesRepository.userPreferences
        .map { prefs ->
            RawProfile.fromComponents(
                colorSpace = prefs.colorSpace,
                logCurve = prefs.logCurve,
                rawLut = prefs.rawLuts[prefs.logCurve.name]
            )
        }
        .stateIn(viewModelScope, SharingStarted.Eagerly, RawProfile.default)

    val useP010: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useP010 }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val useHlg10: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useHlg10 }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val hlgHardwareCompatibilityEnabled: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.hlgHardwareCompatibilityEnabled }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val useP3ColorSpace: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useP3ColorSpace }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val autoEnableHdr: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.autoEnableHdr }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    val useHdrScreenMode: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.useHdrScreenMode }
        .stateIn(viewModelScope, SharingStarted.Eagerly, true)

    val phantomMode: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.phantomMode }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    val videoCodec: StateFlow<com.hinnka.mycamera.video.VideoCodec> = userPreferencesRepository.userPreferences
        .map { it.videoCodec }
        .stateIn(viewModelScope, SharingStarted.Eagerly, com.hinnka.mycamera.video.VideoCodec.H264)
    val videoAudioInputOptions: StateFlow<List<VideoAudioInputOption>> = videoAudioInputManager.availableInputs

    val phantomButtonHidden: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.phantomButtonHidden }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    val launchCameraOnPhantomMode: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.launchCameraOnPhantomMode }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val phantomPipPreview: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.phantomPipPreview }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val phantomPipCrop: StateFlow<PhantomPipCrop> = userPreferencesRepository.userPreferences
        .map { it.phantomPipCrop }
        .stateIn(viewModelScope, SharingStarted.Eagerly, PhantomPipCrop())
    val phantomSaveAsNew: StateFlow<Boolean> = userPreferencesRepository.userPreferences
        .map { it.phantomSaveAsNew }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val defaultVirtualAperture: Flow<Float> =
        userPreferencesRepository.userPreferences.map { it.defaultVirtualAperture }

    val mirrorFrontCamera: Flow<Boolean> = userPreferencesRepository.userPreferences.map { it.mirrorFrontCamera }
    val widgetTheme = userPreferencesRepository.userPreferences.map { it.widgetTheme }
        .stateIn(viewModelScope, SharingStarted.Eagerly, com.hinnka.mycamera.data.WidgetTheme.FOLLOW_SYSTEM)
    val saveLocationEnabled = userPreferencesRepository.userPreferences.map { it.saveLocation }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val openAIApiKey = userPreferencesRepository.userPreferences.map { it.openAIApiKey }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val openAIUrl = userPreferencesRepository.userPreferences.map { it.openAIBaseUrl }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val openAIModel = userPreferencesRepository.userPreferences.map { it.openAIModel }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val useBuiltInAiService = userPreferencesRepository.userPreferences.map { it.useBuiltInAiService }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    // 动态获取可用的 AI 模型列表 (UI 层按需触发刷新)
    private val _availableOpenAIModels = MutableStateFlow<List<String>>(emptyList())
    val availableOpenAIModels = _availableOpenAIModels.asStateFlow()

    private val _isFetchingAIModels = MutableStateFlow(false)
    val isFetchingAIModels = _isFetchingAIModels.asStateFlow()

    // 软件处理参数 Flow
    val sharpening: Flow<Float> = userPreferencesRepository.userPreferences.map { it.sharpening }
    val noiseReduction: Flow<Float> = userPreferencesRepository.userPreferences.map { it.noiseReduction }
    val chromaNoiseReduction: Flow<Float> = userPreferencesRepository.userPreferences.map { it.chromaNoiseReduction }

    private var isShutterSoundEnabled = true
    private var isVibrationEnabled = true

    var glSurfaceView: CameraGLSurfaceView? = null

    // 保存当前的 SurfaceTexture 以便切换摄像头时重用
    private var currentSurfaceTexture: SurfaceTexture? = null

    // 用于处理音量键连续按下的时间戳，防止抖动和过快响应
    private var lastVolumeKeyEventTime = 0L
    private val VOLUME_KEY_DEBOUNCE_TIME = 200L // 毫秒

    private var hasAppliedDefaultFocalLength = false

    private val stackingImages = mutableListOf<SafeImage>()
    private var multipleExposureMetadata: MediaMetadata? = null
    var multipleExposureState by mutableStateOf(MultipleExposureSessionState())
        private set

    private val burstImages = mutableListOf<SafeImage>()
    private var burstCaptureInfo: CaptureInfo? = null
    private var burstPhotoId: String? = null
    var burstImageCount by mutableStateOf(0)
        private set

    var showGhostPermissions by mutableStateOf(false)

    init {
        cameraController.initialize()
        cameraController.onImageCaptured = { image, captureInfo, characteristics, captureResult ->
            if (state.value.burstCapturing) {
                if (burstCaptureInfo == null) {
                    burstCaptureInfo = captureInfo
                }
                burstImages.add(image)
            } else if (multipleExposureState.enabled) {
                viewModelScope.launch {
                    handleMultipleExposureFrameCaptured(image, captureInfo)
                }
            } else if (state.value.useMFNR || state.value.useMFSR) {
                val count = state.value.multiFrameCount
                PLog.d(TAG, "Burst frame received: ${stackingImages.size + 1}/$count")
                stackingImages.add(image)
                if (stackingImages.size >= count) {
                    val imagesToProcess = stackingImages.toList()
                    stackingImages.clear()
                    viewModelScope.launch {
                        processStacking(imagesToProcess, captureInfo, characteristics, captureResult)
                    }
                }
            } else {
                PLog.d(
                    TAG,
                    "onImageCaptured callback triggered - image: ${image.width}x${image.height}, format: ${image.format}"
                )
                viewModelScope.launch {
                    saveImage(image, captureInfo, characteristics, captureResult)
                }
            }
        }
        cameraController.onVideoSaved = { uri ->
            if (uri != null) {
                viewModelScope.launch {
                    val mediaId = GalleryManager.recordVideoCapture(getApplication(), uri)
                    if (mediaId != null) {
                        _imageSavedEvent.emit(Unit)
                    }
                }
            }
        }

        cameraController.onCameraError = { code, message, canRetry ->
            // 只记录错误日志，不在这里重试打开相机
            // 相机恢复应该由 CameraScreen 的 ON_RESUME 生命周期事件处理
            // 这样可以避免在相机被其他应用占用时的无限重试循环
            PLog.d(TAG, "onCameraError: code=$code, message=$message, canRetry=$canRetry")
            stackingImages.forEach {
                it.close()
            }
            stackingImages.clear()
            burstImages.forEach {
                it.close()
            }
            burstImages.clear()
            burstImageCount = 0
        }

        // 监听快门声音、震动和软件处理设置
        viewModelScope.launch {
            var firstPreferencesLogged = false
            val preferenceCollectStart = SystemClock.elapsedRealtime()
            userPreferencesRepository.userPreferences.collect {
                if (!firstPreferencesLogged) {
                    firstPreferencesLogged = true

                    if (it.rawAutoWhiteBalanceEstimate) {
                        setRawAutoWhiteBalanceEstimate(false)
                    }

                    StartupTrace.mark(
                        "CameraViewModel.userPreferences first collect",
                        "costMs=${SystemClock.elapsedRealtime() - preferenceCollectStart}"
                    )
                }
                isShutterSoundEnabled = it.shutterSoundEnabled
                isVibrationEnabled = it.vibrationEnabled
                // 同步降噪等级到相机控制器
                val currentCameraState = cameraController.state.value
                if (currentCameraState.nrLevel != it.nrLevel) {
                    cameraController.setNRLevel(it.nrLevel)
                }
                // 同步锐化等级到相机控制器
                cameraController.setEdgeLevel(it.edgeLevel)
                // 同步 RAW 设置到相机控制器
                if (currentCameraState.useRaw != it.useRaw) {
                    cameraController.setUseRaw(it.useRaw)
                }
                if (currentCameraState.rawMinShutterSpeedNs != it.rawMinShutterSpeedNs) {
                    cameraController.setRawMinShutterSpeedNs(it.rawMinShutterSpeedNs)
                }
                val resolvedDroMode = RawProcessingPreferences.DROMode.fromPersistedName(it.droMode).name
                if (currentCameraState.droMode != resolvedDroMode) {
                    cameraController.setDroMode(resolvedDroMode)
                }
                if (currentCameraState.tonemapMode != it.tonemapMode) {
                    cameraController.setTonemapMode(it.tonemapMode)
                }
                if (currentCameraState.fixTonemapPreview != it.fixTonemapPreview) {
                    cameraController.setFixTonemapPreview(it.fixTonemapPreview)
                }
                if (cameraController.state.value.meteringMode != it.meteringMode) {
                    cameraController.setMeteringMode(it.meteringMode)
                }
                cameraController.setCaptureMode(it.captureMode)
                cameraController.setVideoResolution(it.videoResolution)
                cameraController.setVideoFps(it.videoFps)
                cameraController.setVideoAspectRatio(it.videoAspectRatio)
                cameraController.setVideoLogProfile(it.videoLogProfile)
                cameraController.setVideoBitrate(it.videoBitrate)
                cameraController.setVideoAudioInputId(it.videoAudioInputId)
                cameraController.setVideoStabilizationMode(it.videoStabilizationMode)
                cameraController.setVideoTorchEnabled(it.videoTorchEnabled)
                cameraController.setVideoCodec(it.videoCodec)
                cameraController.setMirrorFrontCameraEnabled(it.mirrorFrontCamera)
                multipleExposureState = multipleExposureState.copy(
                    enabled = it.useMultipleExposure,
                    targetCount = it.multipleExposureCount
                )
                // 同步 Live Photo 设置到相机控制器
                cameraController.setUseLivePhoto(it.useLivePhoto && it.captureMode == CaptureMode.PHOTO)
                // 同步 Ultra HDR 设置到相机控制器
                cameraController.setApplyUltraHDR(it.applyUltraHDR)
                // 同步 P010 设置到相机控制器
                cameraController.setUseP010(it.useP010)
                // 同步 HLG10 设置到相机控制器
                cameraController.setUseHlg10(it.useHlg10)
                // 同步 P3 色域设置到相机控制器
                cameraController.setUseP3ColorSpace(it.useP3ColorSpace)
            }
        }

        cameraController.onLivePhotoVideoCaptured = { file, timestamp ->
            // Global listener still available for other UI needs if any
        }

        // 设置快门音效和震动回调
        cameraController.onPlayShutterSound = {
            if (isShutterSoundEnabled) {
                shutterSoundPlayer.play()
            }
            if (isVibrationEnabled) {
                vibrationHelper.vibrate()
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
                availableLutList = sortedLuts
            }
        }

        viewModelScope.launch {
            contentRepository.availableDcps.collect { dcps ->
                availableDcps = dcps.sortedBy { it.getName() }
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
                availableFrameList = sortedFrames
            }
        }

        viewModelScope.launch {
            userPreferencesRepository.userPreferences.collectLatest { prefs ->
                currentBaselineLutConfig = withContext(Dispatchers.IO) {
                    resolvePreviewBaselineLut(prefs)
                }
            }
        }

        viewModelScope.launch {
            val presetInputs = combine(
                userPreferencesRepository.userPreferences,
                allPresets,
                isInitialized
            ) { prefs, presets, initialized ->
                if (initialized) prefs to presets else null
            }

            combine(
                presetInputs,
                state.map { it.aspectRatio.name }.distinctUntilChanged(),
                currentLutId.flatMapLatest { lutId -> contentRepository.lutManager.getColorRecipeParams(lutId) },
                currentEffectParams
            ) { inputs, aspectRatio, recipe, effects ->
                inputs?.let { (prefs, presets) ->
                    ActivePresetMatchState(
                        prefs = prefs,
                        presets = presets,
                        aspectRatio = aspectRatio,
                        lutId = currentLutId.value,
                        recipe = recipe,
                        effects = effects
                    )
                }
            }.collect { matchState ->
                if (matchState != null) {
                    clearActivePresetIfCurrentSettingsMismatch(matchState)
                }
            }
        }

        // 加载用户偏好设置
        viewModelScope.launch {
            val prefs = userPreferencesRepository.userPreferences.firstOrNull()
            if (prefs != null) {
                // 应用保存的画面比例
                try {
                    val savedAspectRatio = AspectRatio.valueOf(prefs.aspectRatio)
                    cameraController.setAspectRatio(savedAspectRatio)
                } catch (e: IllegalArgumentException) {
                    // 如果保存的值无效，使用默认值
                }
                cameraController.setCaptureMode(prefs.captureMode)
                cameraController.setVideoResolution(prefs.videoResolution)
                cameraController.setVideoFps(prefs.videoFps)
                cameraController.setVideoAspectRatio(prefs.videoAspectRatio)
                cameraController.setVideoLogProfile(prefs.videoLogProfile)
                cameraController.setVideoBitrate(prefs.videoBitrate)
                cameraController.setVideoAudioInputId(prefs.videoAudioInputId)
                cameraController.setVideoStabilizationMode(prefs.videoStabilizationMode)
                cameraController.setVideoTorchEnabled(prefs.videoTorchEnabled)
                cameraController.setVideoCodec(prefs.videoCodec)
                cameraController.setMeteringMode(prefs.meteringMode)

                // 应用保存的 LUT 配置
                if (prefs.lutId != null) {
                    setLut(prefs.lutId)
                } else {
                    // 如果没有保存的 LUT，使用配置文件中的默认 LUT（第一个）
                    val defaultLut = availableLutList.firstOrNull { it.isDefault }
                    defaultLut?.let { setLut(it.id, persist = false) }
                }

                // 应用保存的边框配置
                if (prefs.frameId != null) {
                    currentFrameId = prefs.frameId
                }

                showHistogram = prefs.showHistogram

                // 应用保存的网格线设置
                cameraController.setShowGrid(prefs.showGrid)

                cameraController.setUseMFNR(prefs.useMFNR)
                cameraController.setUseMFSR(prefs.useMFSR)
                cameraController.setMultiFrameCount(prefs.multiFrameCount)
                cameraController.setUseLivePhoto(prefs.useLivePhoto && prefs.captureMode == CaptureMode.PHOTO)
                cameraController.setDroMode(prefs.droMode)
                cameraController.setTonemapMode(prefs.tonemapMode)
                cameraController.setFixTonemapPreview(prefs.fixTonemapPreview)

                // 应用保存的虚拟光圈
                if (prefs.defaultVirtualAperture > 0f) {
                    setVirtualApertureAuto(true)
                    setAperture(prefs.defaultVirtualAperture)
                }
            } else {
                // 如果没有任何偏好设置，使用配置文件中的默认 LUT（第一个）
                val defaultLut = availableLutList.firstOrNull { it.isDefault }
                defaultLut?.let { setLut(it.id, persist = false) }
            }

            _isInitialized.value = true
            StartupTrace.mark("CameraViewModel.isInitialized set to true")
        }

        // 监听相机状态，用于同步预览渲染参数
        viewModelScope.launch {
            state.collect { currentState ->
                // 同步焦段
                val availableCameras = currentState.availableCameras
                if (availableCameras.isNotEmpty() && !hasAppliedDefaultFocalLength) {
                    val prefs = userPreferencesRepository.userPreferences.firstOrNull()
                    val defaultFL = prefs?.defaultFocalLength ?: 0f
                    if (defaultFL > 0f) {
                        applyDefaultFocalLength(defaultFL)
                    }
                    hasAppliedDefaultFocalLength = true
                }

                // 同步虚化参数到渲染器
                glSurfaceView?.let { view ->
                    if (currentState.isVirtualApertureEnabled) {
                        view.setAperture(currentState.virtualAperture)
                    } else {
                        view.setAperture(0f)
                    }
                    view.setVideoRecorder(cameraController.videoRecorder)
                    view.setVideoLogProfile(currentState.videoConfig.logProfile)
                    view.setIsHlgInput(shouldTreatPreviewAsHlgInput(currentState))
                    currentState.focusPoint?.let { fp ->
                        view.setFocusPoint(android.graphics.PointF(fp.first, fp.second))
                    }
                    view.setAutoFocus(currentState.isAutoFocus)
                }
            }
        }

        // 实时景深处理
        viewModelScope.launch {
            cameraController.previewDepthProcessor.latestDepthMap.collect { depth ->
                glSurfaceView?.setDepthMap(depth)
            }
        }

        cameraController.previewAiFocusProcessor.onBusyStateChanged = { busy ->
            viewModelScope.launch(Dispatchers.Main) {
                isAiFocusBusy = busy
            }
        }

        StartupTrace.mark("CameraViewModel.init end")
    }

    fun getAvailableRawLutList(context: Context, logCurve: TransferCurve): List<String> {
        try {
            val files = logCurve.rawFolder?.let { context.assets.list(it) }
            return files?.filter { it.endsWith(".plut") }?.toList() ?: emptyList()
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to list raw luts", e)
        }
        return emptyList()
    }

    fun setUseHdrScreenMode(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveUseHdrScreenMode(enabled)
        }
    }

    fun setRawDcpId(dcpId: String?) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(rawDcpId = SettingValue(dcpId))
            )
        }
    }
    fun setRawBaselineLutId(lutId: String?) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(rawBaselineLutId = SettingValue(lutId))
            )
        }
    }
    fun setRawSpectralFilmEnabled(enabled: Boolean) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(rawSpectralFilmEnabled = SettingValue(enabled))
            )
        }
    }
    fun setRawSpectralFilmStock(stock: String?) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(rawSpectralFilmStock = SettingValue(stock))
            )
        }
    }
    fun setRawSpectralFilmPrint(print: String?) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(rawSpectralFilmPrint = SettingValue(print))
            )
        }
    }
    fun setRawNlmNoiseFactor(value: Float) {
        viewModelScope.launch { userPreferencesRepository.saveRawNlmNoiseFactor(value) }
    }
    fun setRawExposureCompensation(value: Float) {
        viewModelScope.launch { userPreferencesRepository.saveRawExposureCompensation(value) }
    }
    fun setRawAutoExposure(enabled: Boolean) {
        viewModelScope.launch { userPreferencesRepository.saveRawAutoExposure(enabled) }
    }
    fun setRawMinShutterSpeedNs(value: Long) {
        viewModelScope.launch { userPreferencesRepository.saveRawMinShutterSpeedNs(value) }
    }
    fun setRawDROEnabled(enabled: Boolean) {
        viewModelScope.launch { userPreferencesRepository.updateRawDROEnabled(enabled) }
    }
    fun setRawBlackPointCorrection(value: Float) {
        viewModelScope.launch { userPreferencesRepository.saveRawBlackPointCorrection(value) }
    }
    fun setRawWhitePointCorrection(value: Float) {
        viewModelScope.launch { userPreferencesRepository.saveRawWhitePointCorrection(value) }
    }
    fun setRawAutoWhiteBalanceEstimate(enabled: Boolean) {
        viewModelScope.launch { userPreferencesRepository.saveRawAutoWhiteBalanceEstimate(enabled) }
    }
    fun setRawBlackLevelMode(mode: String) {
        viewModelScope.launch {
            userPreferencesRepository.saveRawBlackLevelMode(state.value.currentCameraId, mode)
        }
    }
    fun setRawCustomBlackLevel(value: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveRawCustomBlackLevel(state.value.currentCameraId, value)
        }
    }
    fun importRawDcp(uri: Uri, onComplete: (Boolean) -> Unit) {
        viewModelScope.launch {
            val success = contentRepository.getCustomImportManager().importDcp(uri) != null
            if (success) {
                contentRepository.refreshCustomContent()
            }
            onComplete(success)
        }
    }

    fun importRawDcps(uris: List<Uri>, onComplete: (importedDcps: List<DcpInfo>, failedCount: Int) -> Unit) {
        viewModelScope.launch {
            val importedIds = withContext(Dispatchers.IO) {
                uris.mapNotNull { uri ->
                    contentRepository.getCustomImportManager().importDcp(uri)
                }
            }
            val importedDcps = if (importedIds.isNotEmpty()) {
                contentRepository.refreshCustomContent()
                val dcpById = contentRepository.getAvailableDcps().associateBy { it.id }
                importedIds.mapNotNull { dcpById[it] }
            } else {
                emptyList()
            }
            onComplete(importedDcps, uris.size - importedDcps.size)
        }
    }

    fun deleteRawDcp(dcpId: String, onComplete: (Boolean) -> Unit) {
        viewModelScope.launch {
            val success = withContext(Dispatchers.IO) {
                contentRepository.getCustomImportManager().deleteCustomDcp(dcpId)
            }
            if (success) {
                if (rawDcpId.firstOrNull() == dcpId) {
                    userPreferencesRepository.saveRawDcpId(null)
                }
                contentRepository.refreshCustomContent()
            }
            onComplete(success)
        }
    }

    /**
     * 打开相机（Camera2 接口）
     */
    fun openCamera(surfaceTexture: SurfaceTexture) {
        PLog.d(TAG, "openCamera")
        currentSurfaceTexture = surfaceTexture
        cameraController.openCamera(surfaceTexture)
    }

    /**
     * 关闭相机
     */
    fun closeCamera() {
        currentSurfaceTexture = null
        cameraController.closeCamera()
    }

    fun prewarmDepthEstimator() {
        if (startupPrewarmJob?.isActive == true) return

        startupPrewarmJob = viewModelScope.launch {
            prewarmRawDcp(rawDcpId.firstOrNull())
        }
    }

    private suspend fun prewarmRawDcp(dcpId: String?) = withContext(Dispatchers.IO) {
        val dcpInfo = dcpId?.let { id ->
            contentRepository.getAvailableDcps().firstOrNull { it.id == id }
        } ?: return@withContext
        DcpProfileParser.prewarm(getApplication<Application>(), dcpInfo)
    }

    /**
     * 检查相机状态并在必要时恢复
     */
    fun checkAndRecoverCamera() {
        // 如果有保存的 SurfaceTexture，重新打开相机
        currentSurfaceTexture?.let { texture ->
            if (!state.value.isPreviewActive) {
                cameraController.openCamera(texture)
            }
        }
        restorePreviewLutAfterResume()
    }

    private fun restorePreviewLutAfterResume() {
        val lutId = currentLutId.value
        PLog.d(TAG, "restorePreviewLutAfterResume: lutId=$lutId")
        viewModelScope.launch {
            val loadedLut = withContext(Dispatchers.IO) {
                contentRepository.lutManager.loadLut(lutId)
            }
            currentLutConfig = loadedLut
            cameraController.setLutEnabled(loadedLut != null)
            cameraController.setLogLutActive(loadedLut?.curve?.isLog == true)
            glSurfaceView?.let { view ->
                val currentState = state.value
                view.setBaselineLut(currentBaselineLutConfig)
                view.setBaselineLutEnabled(currentBaselineLutConfig != null)
                view.setBaselineParams(currentBaselineRecipeParams.value)
                view.setLut(loadedLut)
                view.setLutEnabled(loadedLut != null)
                view.setParams(currentRecipeParams.value, if (currentState.isVirtualApertureEnabled) {
                    currentState.virtualAperture
                } else {
                    0f
                })
                view.setColorRecipeEnabled(!currentRecipeParams.value.isDefault())
                view.setVideoRecorder(cameraController.videoRecorder)
                view.setVideoLogProfile(currentState.videoConfig.logProfile)
                view.setIsHlgInput(shouldTreatPreviewAsHlgInput(currentState))
                view.restoreRenderStateAfterResume()
            }
        }
    }

    private suspend fun buildPhotoMetadata(
        width: Int,
        height: Int,
        captureInfo: CaptureInfo,
        sharpeningValue: Float = 0f,
        noiseReductionValue: Float = 0f,
        chromaNoiseReductionValue: Float = 0f,
        captureMode: String? = null,
        multipleExposureFrameCount: Int? = null,
        baselineTarget: BaselineColorCorrectionTarget = BaselineColorCorrectionTarget.JPG,
    ): MediaMetadata {
        val lutIdToSave = currentLutId.value
        val aspectRatio = state.value.aspectRatio
        val frameIdToSave = currentFrameId
        val currentCameraId = cameraController.getCurrentCameraId()

        val sensorOrientation = cameraController.getSensorOrientation()
        val lensFacing = cameraController.getLensFacing()
        val deviceRotation = OrientationObserver.captureRotationDegrees.toInt()
        val baseRotation = if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            (sensorOrientation - deviceRotation + 360) % 360
        } else {
            (sensorOrientation + deviceRotation) % 360
        }

        val userPrefs = userPreferencesRepository.userPreferences.firstOrNull()
        val orientationOffset = userPrefs?.cameraOrientationOffsets?.get(currentCameraId) ?: 0
        val rotation = (baseRotation + orientationOffset) % 360
        val shouldMirror = lensFacing == CameraCharacteristics.LENS_FACING_FRONT &&
            (userPreferencesRepository.userPreferences.firstOrNull()?.mirrorFrontCamera ?: true)
        val aperture = if (state.value.isVirtualApertureEnabled) state.value.virtualAperture else null

        val defaultHdrEffectEnabled = defaultHdrEffectEnabled(
            hasEmbeddedGainmap = false,
            userPrefs = userPrefs
        )
        val baselineMetadata = resolveBaselineMetadata(
            target = baselineTarget,
            userPrefs = userPrefs,
        )

        val spectralFilmSettings = resolveRawSpectralFilmSettings(userPrefs)

        return MediaMetadata(
            lutId = lutIdToSave,
            frameId = frameIdToSave,
            colorRecipeParams = getMergedRecipeParams(),
            baselineTarget = baselineMetadata?.first,
            baselineLutId = baselineMetadata?.second,
            baselineColorRecipeParams = baselineMetadata?.third,
            sharpening = sharpeningValue,
            noiseReduction = noiseReductionValue,
            chromaNoiseReduction = chromaNoiseReductionValue,
            rawDcpId = userPrefs?.rawDcpId,
            rawDenoiseValue = userPrefs?.rawNlmNoiseFactor ?: 0f,
            rawExposureCompensation = userPrefs?.rawExposureCompensation ?: 0f,
            rawAutoExposure = userPrefs?.rawAutoExposure ?: true,
            rawBlackPointCorrection = userPrefs?.rawBlackPointCorrection ?: 0f,
            rawWhitePointCorrection = userPrefs?.rawWhitePointCorrection ?: 0f,
            rawAutoWhiteBalanceEstimate = userPrefs?.rawAutoWhiteBalanceEstimate ?: false,
            rawBlackLevelMode = userPrefs?.rawBlackLevelModes?.get(currentCameraId) ?: "Default",
            rawCustomBlackLevel = userPrefs?.rawCustomBlackLevels?.get(currentCameraId) ?: 0f,
            cameraId = currentCameraId,
            spectralFilmEnabled = spectralFilmSettings.enabled,
            spectralFilmStock = spectralFilmSettings.stock,
            spectralFilmPrint = spectralFilmSettings.print,
            width = width,
            height = height,
            ratio = aspectRatio,
            rotation = rotation,
            deviceModel = captureInfo.model,
            brand = captureInfo.make,
            dateTaken = captureInfo.captureTime,
            latitude = captureInfo.latitude,
            longitude = captureInfo.longitude,
            altitude = captureInfo.altitude,
            iso = captureInfo.iso,
            shutterSpeed = captureInfo.formatExposureTime(),
            focalLength = captureInfo.formatFocalLength(),
            focalLength35mm = captureInfo.formatFocalLength35mm(),
            aperture = captureInfo.formatAperture(),
            exposureBias = state.value.exposureBias,
            droMode = droMode.value,
            isMirrored = shouldMirror,
            colorSpace = captureInfo.colorSpace,
            computationalAperture = aperture,
            focusPointX = state.value.focusPoint?.first,
            focusPointY = state.value.focusPoint?.second,
            manualHdrEffectEnabled = defaultHdrEffectEnabled,
            captureMode = captureMode,
            multipleExposureFrameCount = multipleExposureFrameCount
        )
    }

    private fun resolveRawSpectralFilmSettings(
        userPrefs: UserPreferences?
    ): RawSpectralFilmSettings {
        return RawSpectralFilmSettings(
            enabled = userPrefs?.rawSpectralFilmEnabled ?: false,
            stock = userPrefs?.rawSpectralFilmStock ?: "kodak_portra_400",
            print = userPrefs?.rawSpectralFilmPrint ?: "kodak_portra_endura"
        )
    }

    private fun shouldUseSpectralFilmPreview(userPrefs: UserPreferences): Boolean {
        return userPrefs.useRaw && userPrefs.rawSpectralFilmEnabled
    }

    private fun resolvePreviewBaselineLut(userPrefs: UserPreferences): LutConfig? {
        if (shouldUseSpectralFilmPreview(userPrefs)) {
            val settings = resolveRawSpectralFilmSettings(userPrefs)
            val stock = settings.stock ?: return null
            val print = settings.print ?: return null
            return SpectralFilmProfile.loadPreviewLutConfig(
                getApplication<Application>(),
                stock,
                print
            ).also { config ->
                if (config == null) {
                    PLog.w(TAG, "Failed to load spectral film preview LUT: stock=$stock, print=$print")
                } else {
                    PLog.d(TAG, "Loaded spectral film preview LUT: stock=$stock, print=$print")
                }
            }
        }

        val baselineLutId = resolvePreviewBaselineTarget(userPrefs.useRaw)
            ?.let { userPrefs.getBaselineLutId(it) }
        return baselineLutId?.let { contentRepository.lutManager.loadLut(it) }
    }

    private suspend fun resolveBaselineMetadata(
        target: BaselineColorCorrectionTarget,
        userPrefs: UserPreferences? = null
    ): Triple<BaselineColorCorrectionTarget, String, ColorRecipeParams>? {
        val preferences = userPrefs ?: userPreferencesRepository.userPreferences.firstOrNull() ?: return null
        val baselineLutId = when (target) {
            BaselineColorCorrectionTarget.JPG -> preferences.jpgBaselineLutId
            BaselineColorCorrectionTarget.RAW -> preferences.rawBaselineLutId
            BaselineColorCorrectionTarget.PHANTOM -> preferences.phantomBaselineLutId
        } ?: return null
        val params = contentRepository.lutManager.loadColorRecipeParams(baselineLutId, target)
        return Triple(target, baselineLutId, params)
    }

    private fun isRawCaptureFormat(format: Int): Boolean {
        return when (format) {
            ImageFormat.RAW_SENSOR,
            ImageFormat.RAW10,
            ImageFormat.RAW12 -> true
            else -> false
        }
    }

    private fun defaultHdrEffectEnabled(
        hasEmbeddedGainmap: Boolean,
        userPrefs: UserPreferences?
    ): Boolean {
        if (hasEmbeddedGainmap) return true
        return userPrefs?.autoEnableHdr ?: false
    }

    fun setUseMultipleExposure(enabled: Boolean) {
        if (!enabled) {
            cancelMultipleExposureSession()
        }

        if (enabled) {
            cameraController.setUseMFNR(false)
            cameraController.setUseMFSR(false)
            cameraController.setUseLivePhoto(false)
            cameraController.setUseRaw(false)
        }

        viewModelScope.launch {
            userPreferencesRepository.saveUseMultipleExposure(enabled)
            if (enabled) {
                userPreferencesRepository.setUseMFNR(false)
                userPreferencesRepository.saveUseMFSR(false)
                userPreferencesRepository.saveUseLivePhoto(false)
                userPreferencesRepository.saveUseRaw(false)
            }
        }
    }

    fun cancelMultipleExposureSession() {
        multipleExposureState.previewBitmap?.recycle()
        multipleExposureState.sessionId?.let { sessionId ->
            GalleryManager.clearMultipleExposureSession(getApplication(), sessionId)
        }
        multipleExposureMetadata = null
        multipleExposureState = multipleExposureState.copy(
            sessionId = null,
            capturedCount = 0,
            frames = emptyList(),
            isProcessing = false,
            previewBitmap = null
        )
    }

    fun undoLastMultipleExposureFrame() {
        val sessionId = multipleExposureState.sessionId ?: return
        if (!GalleryManager.removeLastMultipleExposureFrame(getApplication(), sessionId)) return
        refreshMultipleExposurePreview(sessionId)
    }

    fun finishMultipleExposureSession() {
        if (!multipleExposureState.canFinish) return
        val sessionId = multipleExposureState.sessionId ?: return
        val baseMetadata = multipleExposureMetadata ?: return
        viewModelScope.launch(Dispatchers.IO) {
            multipleExposureState = multipleExposureState.copy(isProcessing = true)
            try {
                val context = getApplication<Application>()
                val composedBitmap = GalleryManager.composeMultipleExposurePhoto(context, sessionId) ?: run {
                    multipleExposureState = multipleExposureState.copy(isProcessing = false)
                    return@launch
                }
                val shouldAutoSave = autoSaveAfterCapture.firstOrNull() ?: false
                val photoQualityValue = photoQuality.firstOrNull() ?: 95
                val sharpeningValue = sharpening.firstOrNull() ?: 0f
                val noiseReductionValue = noiseReduction.firstOrNull() ?: 0f
                val chromaNoiseReductionValue = chromaNoiseReduction.firstOrNull() ?: 0f

                val photoId = GalleryManager.preparePhoto(
                    context,
                    baseMetadata.copy(
                        width = composedBitmap.width,
                        height = composedBitmap.height,
                        captureMode = "multiple_exposure",
                        multipleExposureFrameCount = multipleExposureState.capturedCount
                    ),
                    null,
                    previewThumbnail,
                    false,
                    1.0f
                ) ?: run {
                    composedBitmap.recycle()
                    multipleExposureState = multipleExposureState.copy(isProcessing = false)
                    return@launch
                }

                GalleryManager.saveBitmapPhoto(
                    context,
                    photoId,
                    composedBitmap,
                    shouldAutoSave,
                    contentRepository.photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQualityValue
                )
                composedBitmap.recycle()
                cancelMultipleExposureSession()
                _imageSavedEvent.emit(Unit)
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to finish multiple exposure session", e)
                multipleExposureState = multipleExposureState.copy(isProcessing = false)
            }
        }
    }

    fun setVideoCodec(codec: com.hinnka.mycamera.video.VideoCodec) {
        viewModelScope.launch {
            userPreferencesRepository.saveVideoCodec(codec)
        }
    }

    fun pauseVideoRecording() {
        cameraController.pauseVideoRecording()
    }

    fun resumeVideoRecording() {
        cameraController.resumeVideoRecording()
    }

    private fun refreshMultipleExposurePreview(sessionId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val context = getApplication<Application>()
            val frameFiles = GalleryManager.getMultipleExposureFrameFiles(context, sessionId)
            val preview = if (frameFiles.isNotEmpty()) {
                GalleryManager.composeMultipleExposurePreview(context, sessionId)
            } else {
                null
            }
            val oldPreview = multipleExposureState.previewBitmap
            multipleExposureState = multipleExposureState.copy(
                capturedCount = frameFiles.size,
                frames = frameFiles.mapIndexed { index, file -> MultipleExposureFrame(index + 1, file) },
                previewBitmap = preview,
                sessionId = if (frameFiles.isEmpty()) null else sessionId,
                isProcessing = false
            )
            if (oldPreview != null && oldPreview !== preview && !oldPreview.isRecycled) {
                oldPreview.recycle()
            }
            if (frameFiles.isEmpty()) {
                multipleExposureMetadata = null
            }
        }
    }

    private suspend fun handleMultipleExposureFrameCaptured(
        image: SafeImage,
        captureInfo: CaptureInfo
    ) {
        try {
            if (isRawCaptureFormat(image.format)) {
                image.close()
                PLog.w(TAG, "Multiple exposure currently supports processed YUV captures only")
                return
            }

            val context = getApplication<Application>()
            val sessionId = multipleExposureState.sessionId ?: UUID.randomUUID().toString()
            val frameIndex = multipleExposureState.capturedCount + 1
            val metadata = multipleExposureMetadata ?: buildPhotoMetadata(
                width = image.width,
                height = image.height,
                captureInfo = captureInfo,
                captureMode = "multiple_exposure",
                multipleExposureFrameCount = multipleExposureState.targetCount
            ).also { multipleExposureMetadata = it }

            val frameFile = GalleryManager.saveMultipleExposureFrame(
                context,
                sessionId,
                frameIndex,
                image,
                metadata.rotation,
                state.value.aspectRatio,
                metadata.isMirrored,
                photoQuality.firstOrNull() ?: 95
            ) ?: return

            multipleExposureState = multipleExposureState.copy(
                sessionId = sessionId,
                frames = multipleExposureState.frames + MultipleExposureFrame(frameIndex, frameFile),
                capturedCount = frameIndex
            )
            refreshMultipleExposurePreview(sessionId)
            if (frameIndex >= multipleExposureState.targetCount) {
                finishMultipleExposureSession()
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to handle multiple exposure frame", e)
        }
    }

    fun capture() {
        if (state.value.captureMode == CaptureMode.VIDEO) {
            if (state.value.videoRecordingState.isRecording) {
                cameraController.stopVideoRecording()
            } else {
                cameraController.startVideoRecording()
            }
            return
        }

        if (userPreferences.value.saveLocation) {
            val location = locationManager.getCurrentLocation()
            cameraController.setLocation(location?.latitude, location?.longitude)
        } else {
            cameraController.setLocation(null, null)
        }

        val timerSeconds = state.value.timerSeconds

        // 检查 VIP 权限
        val currentLut = getLutInfo(currentLutId.value)
        if (currentLut?.isVip == true && !isPurchased.value) {
            showPaymentDialog = true
            return
        }

        if (timerSeconds > 0) {
            // 延时拍摄：开始倒计时
            viewModelScope.launch {
                for (i in timerSeconds downTo 1) {
                    cameraController.setCountdownValue(i)
                    delay(1000)
                }
                generateThumbnail()
                // 倒计时结束，拍照
                cameraController.setCountdownValue(0)
                if (useLivePhoto.value) {
                    cameraController.setCapturingLivePhoto(true)
                    viewModelScope.launch {
                        delay(1500)
                        cameraController.setCapturingLivePhoto(false)
                    }
                }
                cameraController.capture()
            }
        } else {
            generateThumbnail()
            stackingImages.clear()

            if (useLivePhoto.value) {
                cameraController.setCapturingLivePhoto(true)
                viewModelScope.launch {
                    delay(1500)
                    cameraController.setCapturingLivePhoto(false)
                }
                cameraController.snapshotLivePhoto()
            }
            cameraController.capture()
        }
    }

    fun captureVideoFrame() {
        val currentState = state.value
        if (currentState.captureMode != CaptureMode.VIDEO || !currentState.videoRecordingState.isRecording) {
            return
        }

        if (userPreferences.value.saveLocation) {
            val location = locationManager.getCurrentLocation()
            cameraController.setLocation(location?.latitude, location?.longitude)
        } else {
            cameraController.setLocation(null, null)
        }

        if (isShutterSoundEnabled) {
            shutterSoundPlayer.play()
        }
        if (isVibrationEnabled) {
            vibrationHelper.vibrate()
        }

        glSurfaceView?.capturePreviewFrame { bitmap ->
            viewModelScope.launch {
                saveVideoSnapshot(bitmap)
            }
        } ?: PLog.w(TAG, "captureVideoFrame skipped: glSurfaceView unavailable")
    }

    /**
     * 开始连拍
     */
    fun startContinuousCapture() {
        if (state.value.useRaw && state.value.isRawSupported) return
        generateThumbnail()
        burstImages.clear()
        burstImageCount = 0
        burstPhotoId = UUID.randomUUID().toString()
        if (isShutterSoundEnabled) {
            shutterSoundPlayer.playBurst()
        }
        cameraController.startBurstCapture()
        viewModelScope.launch {
            processBurst()
        }
    }

    /**
     * 停止连拍
     */
    fun stopContinuousCapture() {
        if (state.value.useRaw && state.value.isRawSupported) return
        cameraController.stopBurstCapture()
        shutterSoundPlayer.stopBurst()
        viewModelScope.launch {
            _imageSavedEvent.emit(Unit)
        }
    }


    /**
     * 切换摄像头（前后置切换）
     */
    fun switchCamera() {
        cameraController.switchCamera()
        reopenCamera(preserveVideoRecording = true)
        zoomRatioByMain = 1f
    }

    /**
     * 切换到指定的镜头类型
     */
    fun switchToLens(cameraId: String) {
        cameraController.switchToCameraId(cameraId)
        reopenCamera(preserveVideoRecording = true)
    }

    fun switchToLensAndSetZoomRatio(cameraId: String, ratio: Float) {
        cameraController.switchToCameraId(cameraId)
        reopenCamera(preserveVideoRecording = true)
        setZoomRatioForCamera(ratio, cameraId)
    }

    /**
     * 重新打开相机（切换摄像头后使用）
     */
    private fun reopenCamera(preserveVideoRecording: Boolean = false) {
        currentSurfaceTexture?.let { texture ->
            cameraController.openCamera(
                surfaceTexture = texture,
                preserveVideoRecording = preserveVideoRecording
            )
        }
    }

    /**
     * 获取所有后置摄像头
     */
    fun getBackCameras(): List<CameraInfo> {
        return cameraController.getBackCameras()
    }

    /**
     * 设置曝光补偿
     */
    fun setExposureCompensation(value: Int) {
        cameraController.setExposureCompensation(value)
    }

    /**
     * 设置 ISO
     */
    fun setIso(value: Int) {
        cameraController.setIso(value)
    }

    /**
     * 设置快门速度
     */
    fun setShutterSpeed(value: Long) {
        cameraController.setShutterSpeed(value)
    }

    /**
     * 设置计算光圈 (等效虚化)
     */
    fun setAperture(value: Float) {
        cameraController.setAperture(value)
    }

    /**
     * 设置是否开启虚拟光圈 (等效虚化控制)
     */
    fun setVirtualApertureAuto(enabled: Boolean) {
        cameraController.setVirtualApertureEnabled(enabled)
    }

    fun setAutoFocus(auto: Boolean) {
        cameraController.setAutoFocus(auto)
    }

    fun setFocusDistance(distance: Float) {
        cameraController.setFocusDistance(distance)
    }

    /**
     * 设置变焦倍数
     */
    fun setZoomRatio(ratio: Float) {
        zoomRatioByMain = ratio
        val cameraInfo = state.value.getCurrentCameraInfo()
        setCameraControllerZoomRatio(ratio, cameraInfo)
    }

    private fun setZoomRatioForCamera(ratio: Float, cameraId: String) {
        zoomRatioByMain = ratio
        val cameraInfo = state.value.availableCameras.find { it.cameraId == cameraId }
        setCameraControllerZoomRatio(ratio, cameraInfo)
    }

    private fun setCameraControllerZoomRatio(ratio: Float, cameraInfo: CameraInfo?) {
        val intrinsicZoomRatio = cameraInfo?.intrinsicZoomRatio ?: 1.0f
        cameraController.setZoomRatio(ratio / intrinsicZoomRatio)
    }

    /**
     * 设置画面比例
     */
    fun setAspectRatio(ratio: AspectRatio) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(aspectRatio = SettingValue(ratio))
            )
        }
    }

    fun setTopSheetAspectRatios(ratios: List<AspectRatio>) {
        val sanitizedRatios = AspectRatio.sanitizeTopSheetRatios(ratios)
        if (state.value.aspectRatio !in sanitizedRatios) {
            setAspectRatio(sanitizedRatios.first())
        }
        viewModelScope.launch {
            userPreferencesRepository.saveTopSheetAspectRatios(sanitizedRatios)
        }
    }

    fun addCustomAspectRatio(widthRatio: Int, heightRatio: Int) {
        val ratio = AspectRatio.custom(widthRatio, heightRatio)
        val customRatios = AspectRatio.sanitizeCustomRatios(customAspectRatios.value + ratio)
        viewModelScope.launch {
            userPreferencesRepository.saveCustomAspectRatios(customRatios)
            val selectedRatios = AspectRatio.sanitizeTopSheetRatios(topSheetAspectRatios.value + ratio)
            userPreferencesRepository.saveTopSheetAspectRatios(selectedRatios)
        }
    }

    fun deleteCustomAspectRatio(ratio: AspectRatio) {
        val customRatios = customAspectRatios.value.filterNot { it.name == ratio.name }
        val selectedRatios = topSheetAspectRatios.value.filterNot { it.name == ratio.name }
        if (state.value.aspectRatio.name == ratio.name) {
            setAspectRatio(AspectRatio.RATIO_4_3)
        }
        viewModelScope.launch {
            userPreferencesRepository.saveCustomAspectRatios(customRatios)
            userPreferencesRepository.saveTopSheetAspectRatios(selectedRatios)
        }
    }

    fun setCaptureMode(mode: CaptureMode) {
        if (state.value.videoRecordingState.isRecording && mode != state.value.captureMode) return
        val shouldDisableVideoLog = mode == CaptureMode.PHOTO &&
            state.value.videoConfig.logProfile != VideoLogProfile.OFF
        cameraController.setCaptureMode(mode)
        currentSurfaceTexture = null
        cameraController.closeCamera()
        viewModelScope.launch {
            userPreferencesRepository.saveCaptureMode(mode)
            if (shouldDisableVideoLog) {
                userPreferencesRepository.saveVideoLogProfile(VideoLogProfile.OFF)
            }
        }
    }

    fun setVideoResolution(resolution: VideoResolutionPreset) {
        cameraController.setVideoResolution(resolution)
        reopenCamera()
        viewModelScope.launch {
            userPreferencesRepository.saveVideoResolution(resolution)
        }
    }

    fun setVideoFps(fps: VideoFpsPreset) {
        cameraController.setVideoFps(fps)
        reopenCamera()
        viewModelScope.launch {
            userPreferencesRepository.saveVideoFps(fps)
        }
    }

    fun setVideoAspectRatio(aspectRatio: VideoAspectRatio) {
        cameraController.setVideoAspectRatio(aspectRatio)
        reopenCamera()
        viewModelScope.launch {
            userPreferencesRepository.saveVideoAspectRatio(aspectRatio)
        }
    }

    fun setVideoStabilizationMode(mode: com.hinnka.mycamera.video.VideoStabilizationMode) {
        cameraController.setVideoStabilizationMode(mode)
        viewModelScope.launch {
            userPreferencesRepository.saveVideoStabilizationMode(mode)
        }
    }

    fun setVideoLogProfile(logProfile: VideoLogProfile) {
        cameraController.setVideoLogProfile(logProfile)
        reopenCamera()
        viewModelScope.launch {
            userPreferencesRepository.saveVideoLogProfile(logProfile)
        }
        validateAndCancelNonMatchingVideoLut(logProfile, currentLutConfig)
    }

    /**
     * 验证并取消选择非匹配的视频 LUT
     * 如果当前处于视频模式且启用了 Log，若当前选中的 LUT 与 Log 格式不匹配，则取消该 LUT
     */
    private fun validateAndCancelNonMatchingVideoLut(logProfile: VideoLogProfile, lutConfig: LutConfig?) {
        if (logProfile != VideoLogProfile.OFF && lutConfig != null) {
            if (lutConfig.curve != logProfile.logCurve || lutConfig.colorSpace != logProfile.colorSpace) {
                PLog.d(TAG, "Cancelling selected LUT [${lutConfig.title}] because it does not match video log profile [${logProfile.name}] colorSpace/curve")
                setLut(null)
            }
        }
    }

    fun setVideoBitrate(bitrate: VideoBitratePreset) {
        cameraController.setVideoBitrate(bitrate)
        reopenCamera()
        viewModelScope.launch {
            userPreferencesRepository.saveVideoBitrate(bitrate)
        }
    }

    fun setVideoAudioInputId(audioInputId: String) {
        cameraController.setVideoAudioInputId(audioInputId)
        viewModelScope.launch {
            userPreferencesRepository.saveVideoAudioInputId(audioInputId)
        }
    }

    fun cycleVideoStabilizationMode() {
        val currentMode = state.value.videoConfig.stabilizationMode
        val availableModes = state.value.videoCapabilities.availableStabilizationModes
        if (availableModes.isEmpty()) return
        val nextMode = availableModes[(availableModes.indexOf(currentMode) + 1) % availableModes.size]
        cameraController.setVideoStabilizationMode(nextMode)
        viewModelScope.launch {
            userPreferencesRepository.saveVideoStabilizationMode(nextMode)
        }
    }

    fun setVideoTorchEnabled(enabled: Boolean) {
        cameraController.setVideoTorchEnabled(enabled)
        viewModelScope.launch {
            userPreferencesRepository.saveVideoTorchEnabled(enabled)
        }
    }

    /**
     * 点击对焦
     */
    fun focusOnPoint(x: Float, y: Float, viewWidth: Int, viewHeight: Int) {
        cameraController.focusOnPoint(x, y, viewWidth, viewHeight)
    }

    fun toggleFlash() {
        cameraController.setFlashMode(
            when (state.value.flashMode) {
                0 -> 1
                1 -> 2
                2 -> 0
                else -> 0
            }
        )
    }

    /**
     * 设置曝光自动模式
     */
    fun setAutoExposure(enabled: Boolean) {
        cameraController.setAutoExposure(enabled)
    }

    /**
     * 设置 ISO 自动模式
     */
    fun setIsoAuto(enabled: Boolean) {
        cameraController.setIsoAuto(enabled)
    }

    /**
     * 设置快门自动模式
     */
    fun setShutterSpeedAuto(enabled: Boolean) {
        cameraController.setShutterSpeedAuto(enabled)
    }

    /**
     * 设置白平衡模式
     */
    fun setAwbMode(mode: Int) {
        cameraController.setAwbMode(mode)
    }

    /**
     * 设置白平衡色温
     */
    fun setAwbTemperature(kelvin: Int) {
        cameraController.setAwbTemperature(kelvin)
    }

    fun setMeteringMode(mode: MeteringMode) {
        cameraController.setMeteringMode(mode)
        viewModelScope.launch {
            userPreferencesRepository.saveMeteringMode(mode)
        }
    }

    // ==================== 计费相关方法 ====================

    /**
     * 发起购买
     */
    fun purchase(activity: android.app.Activity) {
        billingManager.purchase(activity)
    }

    // ==================== 自定义导入相关方法 ====================

    /**
     * 获取自定义导入管理器
     */
    fun getCustomImportManager() = contentRepository.getCustomImportManager()

    /**
     * 刷新自定义内容（在导入新的LUT或边框后调用）
     * StateFlow 会自动通知订阅者更新
     */
    fun refreshCustomContent() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                // 重新初始化内容仓库
                // StateFlow 会自动更新 availableLutList 和 availableFrameList
                contentRepository.refreshCustomContent()
            }
            PLog.d(TAG, "Custom content refreshed via ContentRepository")
        }
    }

    /**
     * 复制 LUT
     */
    fun copyLut(lut: LutInfo, copyName: String) {
        viewModelScope.launch {
            val newLutId = withContext(Dispatchers.IO) {
                contentRepository.getCustomImportManager().copyLut(lut, copyName)
            }
            if (newLutId != null) {
                withContext(Dispatchers.IO) {
                    // 同时复制色彩配方
                    val params = contentRepository.lutManager.loadColorRecipeParams(lut.id)
                    contentRepository.lutManager.saveColorRecipeParams(newLutId, params)

                    // 更新排序顺序：放在原版下面
                    val currentOrder = userPreferencesRepository.userPreferences.first().filterOrder.toMutableList()
                    if (currentOrder.isEmpty()) {
                        // 如果当前没有排序，则从当前列表初始化并插入
                        val allIds = availableLutList.map { it.id }.toMutableList()
                        val index = allIds.indexOf(lut.id)
                        if (index != -1) {
                            allIds.add(index + 1, newLutId)
                        } else {
                            allIds.add(newLutId)
                        }
                        userPreferencesRepository.saveFilterOrder(allIds)
                    } else {
                        val index = currentOrder.indexOf(lut.id)
                        if (index != -1) {
                            currentOrder.add(index + 1, newLutId)
                        } else {
                            currentOrder.add(newLutId)
                        }
                        userPreferencesRepository.saveFilterOrder(currentOrder)
                    }

                    // 刷新列表
                    contentRepository.refreshCustomContent()
                }
            }
        }
    }

    /**
     * 获取滤镜排序顺序
     */
    val filterOrder: Flow<List<String>> = userPreferencesRepository.userPreferences.map { it.filterOrder }

    /**
     * 获取边框排序顺序
     */
    val frameOrder: Flow<List<String>> = userPreferencesRepository.userPreferences.map { it.frameOrder }

    /**
     * 获取分类排序顺序
     */
    val categoryOrder: Flow<List<String>> = userPreferencesRepository.userPreferences.map { it.categoryOrder }

    val lutSelectorMode: StateFlow<LutSelectorMode> = userPreferencesRepository.userPreferences
        .map { it.lutSelectorMode }
        .stateIn(viewModelScope, SharingStarted.Eagerly, LutSelectorMode.Style)

    /**
     * 保存滤镜排序顺序
     */
    fun saveFilterOrder(order: List<String>) {
        viewModelScope.launch {
            userPreferencesRepository.saveFilterOrder(order)
        }
    }

    /**
     * 保存边框排序顺序
     */
    fun saveFrameOrder(order: List<String>) {
        viewModelScope.launch {
            userPreferencesRepository.saveFrameOrder(order)
        }
    }

    /**
     * 保存分类排序顺序
     */
    fun saveCategoryOrder(order: List<String>) {
        viewModelScope.launch {
            userPreferencesRepository.saveCategoryOrder(order)
        }
    }

    fun setLutSelectorMode(mode: LutSelectorMode) {
        viewModelScope.launch {
            userPreferencesRepository.saveLutSelectorMode(mode)
        }
    }

    // ==================== LUT 相关方法 ====================

    /**
     * 设置当前 LUT
     */
    fun setLut(lutId: String?, persist: Boolean = true) {
        val normalizedLutId = lutId ?: "none"
        currentLutId.value = normalizedLutId
        currentRecipeParams = recipeFlowFor(normalizedLutId)
        if (lutId == null || lutId == "none") {
            currentLutConfig = null
            // LUT 已禁用，通知相机控制器
            cameraController.setLogLutActive(false)
            cameraController.setLutEnabled(false)
        } else {
            val hadActiveLut = currentLutConfig != null
            if (!hadActiveLut) {
                // 首次启动时先保持”未启用”状态，避免 Live Photo 在 LUT 文件尚未加载完成前录入原始画面。
                cameraController.setLutEnabled(false)
            }
            viewModelScope.launch {
                val loadedLut = withContext(Dispatchers.IO) {
                    contentRepository.lutManager.loadLut(lutId)
                }
                if (state.value.captureMode == CaptureMode.VIDEO) {
                    val logProfile = state.value.videoConfig.logProfile
                    if (logProfile != VideoLogProfile.OFF && loadedLut != null) {
                        if (loadedLut.curve != logProfile.logCurve || loadedLut.colorSpace != logProfile.colorSpace) {
                            PLog.d(TAG, "Deselecting newly selected LUT [${loadedLut.title}] because it does not match video log profile [${logProfile.name}] colorSpace/curve")
                            setLut(null)
                            return@launch
                        }
                    }
                }
                currentLutConfig = loadedLut
                cameraController.setLogLutActive(loadedLut?.curve?.isLog == true)
                cameraController.setLutEnabled(loadedLut != null)
            }
            if (hadActiveLut) {
                cameraController.setLutEnabled(true)
            }
        }

        if (persist) {
            viewModelScope.launch {
                userPreferencesRepository.saveLutConfig(lutId)
                clearActivePresetIfCurrentSettingsMismatch()
            }
        }
    }

    private fun shouldUseHlgCapture(): Boolean {
        val state = state.value
        val baseCondition = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                state.isP010Supported &&
                state.isHlg10Supported &&
                !state.useRaw
        if (!baseCondition) return false
        // 用户主动开启 HLG10 录制
        val userHlg = state.useP010 && state.useHlg10
        // Log LUT 需要 HLG 采集获取线性信号（替代 tonemap gamma，提升兼容性）
        val logLutHlg = state.lutEnabled && state.isLogLutActive
        // Video Log 同样需要 HLG 采集获取线性信号
        val videoLogHlg = state.captureMode == CaptureMode.VIDEO && state.videoConfig.logProfile.isEnabled
        return userHlg || logLutHlg || videoLogHlg
    }

    private fun shouldTreatPreviewAsHlgInput(currentState: CameraState): Boolean {
        return hlgHardwareCompatibilityEnabled.value && currentState.isHLG
    }

    /**
     * 切换到下一个滤镜
     */
    fun switchToNextLut() {
        if (availableLutList.isEmpty()) return
        val currentIndex = availableLutList.indexOfFirst { it.id == currentLutId.value }
        val nextIndex = if (currentIndex == -1 || currentIndex == availableLutList.size - 1) 0 else currentIndex + 1
        setLut(availableLutList[nextIndex].id)
        vibrationHelper.vibrate()
    }

    /**
     * 切换到上一个滤镜
     */
    fun switchToPreviousLut() {
        if (availableLutList.isEmpty()) return
        val currentIndex = availableLutList.indexOfFirst { it.id == currentLutId.value }
        val prevIndex = if (currentIndex <= 0) availableLutList.size - 1 else currentIndex - 1
        setLut(availableLutList[prevIndex].id)
        vibrationHelper.vibrate()
    }

    fun updateLut() {
        viewModelScope.launch {
            val newLutId = userPreferencesRepository.userPreferences.map { it.lutId }.firstOrNull() ?: return@launch
            if (currentLutId.value != newLutId) {
                setLut(newLutId)
            }
        }
    }

    /**
     * 设置是否应用 Ultra HDR 策略
     */
    fun setApplyUltraHDR(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveApplyUltraHDR(enabled)
        }
    }

    fun setSaveLocation(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveSaveLocation(enabled)
        }
    }

    fun refreshLocationOnResume() {
        if (userPreferences.value.saveLocation) {
            locationManager.requestCurrentLocation()
        }
    }

    fun setOpenAIApiKey(key: String) {
        viewModelScope.launch {
            userPreferencesRepository.saveOpenAIApiKey(key)
        }
    }

    fun setOpenAIUrl(url: String) {
        viewModelScope.launch {
            userPreferencesRepository.saveOpenAIBaseUrl(url)
        }
    }

    fun setOpenAIModel(model: String) {
        viewModelScope.launch {
            userPreferencesRepository.saveOpenAIModel(model)
        }
    }

    fun setUseBuiltInAiService(use: Boolean) {
        if (use && !isPurchased.value) {
            showPaymentDialog = true
            return
        }
        viewModelScope.launch {
            userPreferencesRepository.saveUseBuiltInAiService(use)
        }
    }

    /**
     * 查询可用的 AI 模型列表
     */
    fun fetchAvailableAIModels() {
        if (_isFetchingAIModels.value) return

        viewModelScope.launch {
            _isFetchingAIModels.value = true

            try {
                val context = getApplication<Application>()
                val client = OpenAIApiClient()
                client.initialize(context)
                val result = client.getAvailableModels()
                result.onSuccess { models ->
                    _availableOpenAIModels.value = models
                    // 如果当前选择的模型为空且有可用模型，自动选择第一个
                    if (openAIModel.value.isNullOrBlank() && models.isNotEmpty()) {
                        setOpenAIModel(models.first())
                    }
                }.onFailure { e ->
                    PLog.e(TAG, "Failed to fetch AI models", e)
                }
            } catch (e: Exception) {
                PLog.e(TAG, "Error initializing OpenAIApiClient for model fetch", e)
            } finally {
                _isFetchingAIModels.value = false
            }
        }
    }

    /**
     * 设置是否启用 P010
     */
    fun setUseP010(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveUseP010(enabled)
        }
    }

    fun setUseHlg10(enabled: Boolean) {
        cameraController.setUseHlg10(enabled)
        viewModelScope.launch {
            userPreferencesRepository.saveUseHlg10(enabled)
        }
        reopenCamera()
    }

    fun setHlgHardwareCompatibilityEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveHlgHardwareCompatibilityEnabled(enabled)
        }
        glSurfaceView?.setIsHlgInput(shouldTreatPreviewAsHlgInput(state.value))
    }

    fun setUseP3ColorSpace(enabled: Boolean) {
        cameraController.setUseP3ColorSpace(enabled)
        viewModelScope.launch {
            userPreferencesRepository.saveUseP3ColorSpace(enabled)
        }
        reopenCamera()
    }

    fun setAutoEnableHdrForHdrCapture(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveAutoEnableHdrForHdrCapture(enabled)
        }
    }

    /**
     * 获取 LUT 信息
     */
    fun getLutInfo(id: String): LutInfo? {
        return contentRepository.lutManager.getLutInfo(id)
    }

    /**
     * 从相机捕获预览帧并生成所有 LUT 的预览图
     */
    fun generateThumbnail() {
        if (isGeneratingPreviews) {
            PLog.d(TAG, "Already generating previews, skipping")
            return
        }

        isGeneratingPreviews = true

        val glView = glSurfaceView
        if (glView != null) {
            val thumbnailRotation = capturePreviewThumbnailRotation()
            glView.capturePreviewFrame { bitmap ->
                previewThumbnail = rotateCapturePreviewThumbnail(bitmap, thumbnailRotation)
                isGeneratingPreviews = false
            }
        } else {
            isGeneratingPreviews = false
        }
    }

    private fun capturePreviewThumbnailRotation(): Float {
        val deviceRotation = OrientationObserver.captureRotationDegrees.toInt()
        val lensFacing = cameraController.getLensFacing()
        val baseRotation = if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            (360 - deviceRotation) % 360
        } else {
            deviceRotation
        }
        val currentCameraId = cameraController.getCurrentCameraId()
        val orientationOffset = userPreferences.value.cameraOrientationOffsets[currentCameraId] ?: 0
        return ((baseRotation + orientationOffset) % 360).toFloat()
    }

    private fun rotateCapturePreviewThumbnail(bitmap: Bitmap, rotationDegrees: Float): Bitmap {
        if (rotationDegrees == 0f) {
            return bitmap
        }
        val sourceWidth = bitmap.width
        val sourceHeight = bitmap.height
        val rotated = BitmapUtils.rotate(bitmap, rotationDegrees)
        PLog.d(
            TAG,
            "Preview thumbnail rotated for capture: ${sourceWidth}x${sourceHeight}, rotation=$rotationDegrees"
        )
        return rotated
    }

    suspend fun applyLut(bitmap: Bitmap): Bitmap = withContext(Dispatchers.IO) {
        currentLutConfig?.let { lut ->
            val params = contentRepository.lutManager.loadColorRecipeParams(currentLutId.value)
            contentRepository.imageProcessor.applyLut(
                bitmap = bitmap,
                isHlgInput = shouldTreatPreviewAsHlgInput(state.value),
                lutConfig = lut,
                colorRecipeParams = params
            )
        } ?: bitmap
    }

    fun handleHistogramUpdate(histogram: IntArray) {
        cameraController.updateHistogram(histogram)
    }

    fun handleMeteringUpdate(totalWeight: Double, weightedSumLuminance: Double) {
        cameraController.calculateAutoMetering(totalWeight, weightedSumLuminance)
    }

    fun handleHighlightPointUpdate(x: Float, y: Float) {
        cameraController.updateHighlightPoint(x, y)
    }

    fun handleDepthMapUpdate(bitmap: Bitmap) {
        cameraController.previewDepthProcessor.processBitmap(bitmap)
    }

    fun handleAiFocusInputUpdate(bitmap: Bitmap) {
        if (isAiFocusBusy) return
        if (!state.value.isAutoFocus || state.value.isFocusing) return
        cameraController.previewAiFocusProcessor.targetMode = aiFocusTargetMode.value
        cameraController.previewAiFocusProcessor.scoreThreshold = aiFocusScoreThreshold.value
        cameraController.previewAiFocusProcessor.onFocusTarget = { target ->
            viewModelScope.launch(Dispatchers.Main) {
                val currentState = state.value
                if (currentState.focusPoint != null &&
                    currentState.focusPointSource == FocusPointSource.MANUAL
                ) {
                    return@launch
                }
                if (!state.value.isAutoFocus || state.value.isFocusing) return@launch
                cameraController.focusOnNormalizedPoint(target.x, target.y)
            }
        }
        cameraController.previewAiFocusProcessor.onTargetSeen = { target ->
            cameraController.notifyAiSubjectSeen(target.x, target.y)
        }
        cameraController.previewAiFocusProcessor.onTargetLost = {
            viewModelScope.launch(Dispatchers.Main) {
                cameraController.cancelSubjectFocus("ai_target_lost")
            }
        }
        cameraController.previewAiFocusProcessor.processBitmap(bitmap)
    }

    // ==================== 边框相关方法 ====================

    /**
     * 设置当前边框
     */
    fun setFrame(frameId: String?) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(frameId = SettingValue(frameId))
            )
        }
    }

    /**
     * 获取边框的自定义属性
     */
    suspend fun getFrameCustomProperties(frameId: String): Map<String, String> {
        return contentRepository.frameManager.loadCustomProperties(frameId)
    }

    /**
     * 保存边框的自定义属性
     */
    suspend fun saveFrameCustomProperties(frameId: String, properties: Map<String, String>) {
        contentRepository.frameManager.saveCustomProperties(frameId, properties)
    }

    fun loadFrameEditorDraft(frameId: String?, imageFrame: Boolean = false): FrameEditorDraft {
        return contentRepository.frameManager.createEditorDraft(frameId, imageFrame)
    }

    suspend fun saveFrameEditorDraft(draft: FrameEditorDraft): String? = withContext(Dispatchers.IO) {
        val savedId = contentRepository.frameManager.saveEditorDraft(draft)
        if (savedId != null) {
            contentRepository.refreshCustomContent()
        }
        savedId
    }

    fun importFrameEditorImage(uri: Uri, frameIdHint: String? = null): String? {
        return contentRepository.frameManager.importEditorFrameImage(uri, frameIdHint)
    }

    suspend fun renderFrameEditorPreview(draft: FrameEditorDraft, portrait: Boolean): Bitmap =
        withContext(Dispatchers.Default) {
            val source = FramePreviewFactory.createPreviewBitmap(portrait)
            val template = draft.toTemplate(draft.editableFrameId ?: draft.sourceFrameId ?: "preview_frame")
            val metadata = FramePreviewFactory.createPreviewMetadata(source.width, source.height)
            contentRepository.frameRenderer.render(source, template, metadata)
        }

    /**
     * 设置是否显示直方图
     */
    fun saveShowHistogram(show: Boolean) {
        showHistogram = show
        // 保存到用户偏好设置
        viewModelScope.launch {
            userPreferencesRepository.saveShowHistogram(show)
        }
    }

    // ==================== 延时拍摄和网格线相关方法 ====================

    /**
     * 设置是否使用多帧降噪
     */
    fun setUseMFNR(enabled: Boolean) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(useMFNR = SettingValue(enabled))
            )
        }
    }

    /**
     * 设置多帧合成帧数
     */
    fun setMultiFrameCount(count: Int) {
        val normalizedCount = count.coerceIn(
            MultiFrameConfig.MIN_FRAME_COUNT,
            MultiFrameConfig.MAX_FRAME_COUNT
        )
        cameraController.setMultiFrameCount(normalizedCount)
        viewModelScope.launch {
            userPreferencesRepository.saveMultiFrameCount(normalizedCount)
            //reopenCamera()
        }
    }

    fun setMultipleExposureCount(count: Int) {
        val normalizedCount = count.coerceIn(2, 9)
        multipleExposureState = multipleExposureState.copy(targetCount = normalizedCount)
        viewModelScope.launch {
            userPreferencesRepository.saveMultipleExposureCount(normalizedCount)
        }
    }

    /**
     * 设置是否使用超分辨率
     */
    fun setUseMFSR(enabled: Boolean) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(useMFSR = SettingValue(enabled))
            )
        }
    }

    fun setSuperResolutionScale(scale: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveSuperResolutionScale(scale)
        }
    }

    /**
     * 设置是否启用 Live Photo
     */
    fun setUseLivePhoto(enabled: Boolean) {
        if (enabled) {
            setUseMultipleExposure(false)
        }
        cameraController.setUseLivePhoto(enabled)
        viewModelScope.launch {
            userPreferencesRepository.saveUseLivePhoto(enabled)
        }
    }

    fun setEnableDevelopAnimation(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveEnableDevelopAnimation(enabled)
        }
    }

    /**
     * 切换延时拍摄档位（0s → 3s → 5s → 10s → 0s）
     */
    fun toggleTimer() {
        val currentTimer = state.value.timerSeconds
        val nextTimer = when (currentTimer) {
            0 -> 3
            3 -> 5
            5 -> 10
            10 -> 0
            else -> 0
        }
        cameraController.setTimerSeconds(nextTimer)
    }

    /**
     * 切换网格线显示
     */
    fun toggleGrid() {
        setShowGrid(!state.value.showGrid)
    }

    /**
     * 设置是否显示网格线
     */
    fun setShowGrid(show: Boolean) {
        cameraController.setShowGrid(show)
        viewModelScope.launch {
            userPreferencesRepository.saveShowGrid(show)
        }
    }

    /**
     * 切换 RAW 格式拍摄
     */
    fun toggleRaw() {
        val nextValue = !useRaw.value
        setUseRaw(nextValue)
    }

    fun setUseRaw(useRaw: Boolean) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(useRaw = SettingValue(useRaw))
            )
        }
    }

    fun setExportDngWithRawExport(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveExportDngWithRawExport(enabled)
        }
    }

    // ==================== 新增设置项方法 ====================

    /**
     * 设置是否显示水平仪
     */
    fun setShowLevelIndicator(show: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveShowLevelIndicator(show)
        }
    }

    /**
     * 设置手动对焦时是否显示峰值对焦
     */
    fun setFocusPeakingEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveFocusPeakingEnabled(enabled)
        }
    }

    /**
     * 复制边框并把副本插入到原边框后面。
     */
    fun copyFrame(frame: FrameInfo, copyName: String) {
        viewModelScope.launch {
            val newFrameId = withContext(Dispatchers.IO) {
                contentRepository.getCustomImportManager().copyFrame(frame, copyName)
            }
            if (newFrameId != null) {
                withContext(Dispatchers.IO) {
                    val currentOrder = userPreferencesRepository.userPreferences.first().frameOrder.toMutableList()
                    if (currentOrder.isEmpty()) {
                        val allIds = availableFrameList.map { it.id }.toMutableList()
                        val index = allIds.indexOf(frame.id)
                        if (index != -1) {
                            allIds.add(index + 1, newFrameId)
                        } else {
                            allIds.add(newFrameId)
                        }
                        userPreferencesRepository.saveFrameOrder(allIds)
                    } else {
                        val index = currentOrder.indexOf(frame.id)
                        if (index != -1) {
                            currentOrder.add(index + 1, newFrameId)
                        } else {
                            currentOrder.add(newFrameId)
                        }
                        userPreferencesRepository.saveFrameOrder(currentOrder)
                    }
                    contentRepository.refreshCustomContent()
                }
            }
        }
    }

    fun setAiFocusTargetMode(mode: AiFocusTargetMode) {
        viewModelScope.launch {
            userPreferencesRepository.saveAiFocusTargetMode(mode)
        }
    }

    fun setAiFocusScoreThreshold(value: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveAiFocusScoreThreshold(value)
        }
    }

    /**
     * 设置是否启用快门声音
     */
    fun setShutterSoundEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveShutterSoundEnabled(enabled)
        }
    }

    /**
     * 设置是否启用拍摄震动
     */
    fun setVibrationEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveVibrationEnabled(enabled)
        }
    }

    /**
     * 设置音量键操作
     */
    fun setVolumeKeyAction(action: VolumeKeyAction) {
        viewModelScope.launch {
            userPreferencesRepository.saveVolumeKeyAction(action)
        }
    }

    /**
     * 处理音量键按下
     * @return 是否消费了该事件
     */
    fun handleVolumeKey(isUp: Boolean): Boolean {
        val action = volumeKeyAction.value
        if (action == VolumeKeyAction.NONE) return false

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastVolumeKeyEventTime < VOLUME_KEY_DEBOUNCE_TIME) {
            return true // 还在冷却时间内，消费事件但不做处理
        }
        lastVolumeKeyEventTime = currentTime

        return when (action) {
            VolumeKeyAction.CAPTURE -> {
                capture()
                true
            }

            VolumeKeyAction.EXPOSURE_COMPENSATION -> {
                val currentEV = state.value.exposureCompensation
                val range = state.value.getExposureCompensationRange()
                if (range.lower == 0 && range.upper == 0) return true // 不支持曝光补偿

                if (isUp) {
                    if (currentEV < range.upper) {
                        setExposureCompensation(currentEV + 1)
                    }
                } else {
                    if (currentEV > range.lower) {
                        setExposureCompensation(currentEV - 1)
                    }
                }
                true
            }

            VolumeKeyAction.ZOOM -> {
                handleVolumeZoom(isUp)
                true
            }
        }
    }

    /**
     * 处理音量键变焦切换
     * 逻辑：切换到下一个/上一个 ZoomStop
     */
    private fun handleVolumeZoom(isUp: Boolean) {
        val currentState = state.value
        val availableCameras = currentState.availableCameras
        val currentCamera = currentState.getCurrentCameraInfo() ?: return

        // 1. 获取主摄
        val mainCamera = availableCameras.find {
            it.lensType == if (currentCamera.lensType == LensType.FRONT) LensType.FRONT else LensType.BACK_MAIN
        } ?: return

        // 2. 计算变焦档位 (逻辑同步自 ZoomControlBar.kt)
        val lensZoomStops = calculateLensZoomStops(availableCameras, currentCamera)
        val zoomStops = allZoomStops(
            lensZoomStops,
            mainCamera,
            currentCamera,
            userPreferences.value.customFocalLengths,
            userPreferences.value.hiddenFocalLengths
        )

        if (zoomStops.isEmpty()) return

        // 3. 找到当前或者最近的档位索引
        val currentZoomRatio = zoomRatioByMain
        var currentIndex = zoomStops.indexOfFirst { abs(it - currentZoomRatio) < 0.05f }

        if (currentIndex == -1) {
            // 如果不在已知档位，找到最近的一个
            currentIndex = zoomStops.indices.minByOrNull { abs(zoomStops[it] - currentZoomRatio) } ?: 0
        }

        // 4. 计算下一个索引
        val nextIndex = if (isUp) {
            (currentIndex + 1).coerceAtMost(zoomStops.lastIndex)
        } else {
            (currentIndex - 1).coerceAtLeast(0)
        }

        if (nextIndex != currentIndex) {
            val targetZoom = zoomStops[nextIndex]

            // 5. 检查是否需要切换镜头 (逻辑同步自 ZoomControlBar.kt)
            val optimalLens = findOptimalLens(targetZoom, availableCameras, currentCamera.cameraId)
            if (optimalLens != null && optimalLens.cameraId != currentCamera.cameraId) {
                switchToLensAndSetZoomRatio(optimalLens.cameraId, targetZoom)
            } else {
                setZoomRatio(targetZoom)
            }
        }
    }

    /**
     * 计算变焦档位
     */
    fun calculateLensZoomStops(
        cameras: List<CameraInfo>,
        currentCamera: CameraInfo?
    ): List<Float> {
        val stops = mutableListOf<Float>()

        val filter: (CameraInfo) -> Boolean = if (currentCamera?.lensType == LensType.FRONT) {
            { it.lensType == LensType.FRONT }
        } else {
            { it.lensType != LensType.FRONT && it.lensType != LensType.BACK_MACRO }
        }

        // 添加各个镜头的固有变焦比例
        cameras.filter(filter).forEach { camera ->
            if (camera.intrinsicZoomRatio > 0) {
                // 避免添加极其接近的变焦倍率（例如 1.0 和 1.0006）
                if (stops.none { abs(it - camera.intrinsicZoomRatio) < 0.01f }) {
                    stops.add(camera.intrinsicZoomRatio)
                }
            }
        }
        return stops.sorted()
    }

    /**
     * 计算变焦档位
     */
    fun allZoomStops(
        lensZoomStops: List<Float>,
        mainCamera: CameraInfo?,
        currentCamera: CameraInfo?,
        customFocalLengths: List<Float> = emptyList(),
        hiddenFocalLengths: List<Float> = emptyList()
    ): List<Float> {
        val stops = mutableListOf<Float>()

        if (currentCamera?.lensType == LensType.FRONT) {
            stops.addAll(lensZoomStops)
            if (stops.none { abs(it - 2f) <= 0.1f }) {
                stops.add(2f)
            }
            return stops.sorted()
        }

        mainCamera ?: return lensZoomStops.sorted()

        // 1. 添加并过滤原生镜头焦段
        if (mainCamera.focalLength35mmEquivalent > 0) {
            val filteredLensStops = lensZoomStops.filter { zoom ->
                val fl = zoom * mainCamera.focalLength35mmEquivalent
                hiddenFocalLengths.none { abs(it - fl) < 0.5f }
            }
            stops.addAll(filteredLensStops)
        } else {
            stops.addAll(lensZoomStops)
        }

        addDefaultMinimumZoomStop(stops, lensZoomStops, mainCamera, hiddenFocalLengths)

        // 2. 添加自定义焦段 (不参与隐藏过滤)
        if (mainCamera.focalLength35mmEquivalent > 0) {
            customFocalLengths.forEach { fl ->
                val zoom = fl / mainCamera.focalLength35mmEquivalent
                if (stops.none { abs(it - zoom) <= 0.01f }) {
                    stops.add(zoom)
                }
            }
        }

        return stops.sorted()
    }

    private fun addDefaultMinimumZoomStop(
        stops: MutableList<Float>,
        lensZoomStops: List<Float>,
        mainCamera: CameraInfo,
        hiddenFocalLengths: List<Float> = emptyList()
    ) {
        val mainZoom = mainCamera.intrinsicZoomRatio
        val hasSmallerLens = lensZoomStops.any { it < mainZoom - 0.01f }
        val minimumZoom = mainCamera.minZoom * mainZoom

        if (hasSmallerLens || minimumZoom >= mainZoom - 0.01f) return

        val isHidden = if (mainCamera.focalLength35mmEquivalent > 0) {
            val minimumFocalLength = minimumZoom * mainCamera.focalLength35mmEquivalent
            hiddenFocalLengths.any { abs(it - minimumFocalLength) < 0.5f }
        } else {
            false
        }

        if (!isHidden && stops.none { abs(it - minimumZoom) <= 0.01f }) {
            stops.add(minimumZoom)
        }
    }

    /**
     * 根据变焦倍率找到最佳镜头
     */
    fun findOptimalLens(
        targetZoom: Float,
        cameras: List<CameraInfo>,
        currentCameraId: String
    ): CameraInfo? {
        val currentLensType = cameras.find { it.cameraId == currentCameraId }?.lensType
        val zoomableCameras =
            cameras.filter { if (currentLensType == LensType.FRONT) it.lensType == LensType.FRONT else (it.lensType != LensType.FRONT && it.lensType != LensType.BACK_MACRO) }
        if (zoomableCameras.isEmpty()) return null
        val candidates = zoomableCameras
            .filter { it.intrinsicZoomRatio <= targetZoom + 0.01f }
        val bestZoom = candidates.maxOfOrNull { it.intrinsicZoomRatio }
            ?: zoomableCameras.minOfOrNull { it.intrinsicZoomRatio }
            ?: return null
        val tiedCandidates = candidates.filter { abs(it.intrinsicZoomRatio - bestZoom) <= 0.01f }
            .ifEmpty { zoomableCameras.filter { abs(it.intrinsicZoomRatio - bestZoom) <= 0.01f } }
        return tiedCandidates.firstOrNull { it.cameraId == currentCameraId }
            ?: tiedCandidates.firstOrNull()
    }

    /**
     * 设置是否拍摄后自动保存
     */
    fun setAutoSaveAfterCapture(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveAutoSaveAfterCapture(enabled)
        }
    }

    fun addCustomFocalLength(focalLength: Float) {
        viewModelScope.launch {
            val prefs = userPreferencesRepository.userPreferences.first()
            val list = prefs.customFocalLengths.toMutableList()
            if (list.none { abs(it - focalLength) < 0.5f }) {
                list.add(focalLength)
                userPreferencesRepository.saveCustomFocalLengths(list.sorted())
            }
        }
    }

    fun removeCustomFocalLength(focalLength: Float) {
        viewModelScope.launch {
            val prefs = userPreferencesRepository.userPreferences.first()
            val list = prefs.customFocalLengths.toMutableList()
            list.removeAll { abs(it - focalLength) < 0.5f }
            userPreferencesRepository.saveCustomFocalLengths(list)

            // 如果删除了当前的默认焦段，重置为0
            if (abs(prefs.defaultFocalLength - focalLength) < 0.5f) {
                userPreferencesRepository.saveDefaultFocalLength(0f)
            }
        }
    }

    fun toggleFocalLengthVisibility(focalLength: Float) {
        viewModelScope.launch {
            val prefs = userPreferencesRepository.userPreferences.first()
            val list = prefs.hiddenFocalLengths.toMutableList()
            val index = list.indexOfFirst { abs(it - focalLength) < 0.5f }
            if (index != -1) {
                list.removeAt(index)
            } else {
                list.add(focalLength)
                // 如果隐藏了当前的默认焦段，重置默认焦段为0
                if (abs(prefs.defaultFocalLength - focalLength) < 0.5f) {
                    userPreferencesRepository.saveDefaultFocalLength(0f)
                }
            }
            userPreferencesRepository.saveHiddenFocalLengths(list)
        }
    }

    /**
     * 设置 RAW 色彩空间
     */
    fun setColorSpace(colorSpace: ColorSpace) {
        viewModelScope.launch {
            userPreferencesRepository.saveColorSpace(colorSpace)
        }
    }

    /**
     * 设置 RAW Log 曲线
     */
    fun setLogCurve(logCurve: TransferCurve) {
        viewModelScope.launch {
            userPreferencesRepository.saveLogCurve(logCurve)
        }
    }

    fun setRawProfile(rawProfile: RawProfile) {
        viewModelScope.launch {
            userPreferencesRepository.saveRawProfile(rawProfile)
        }
    }

    fun setBaselineLut(target: BaselineColorCorrectionTarget, lutId: String?) {
        viewModelScope.launch {
            val update = when (target) {
                BaselineColorCorrectionTarget.JPG -> CameraFeatureUpdate(jpgBaselineLutId = SettingValue(lutId))
                BaselineColorCorrectionTarget.RAW -> CameraFeatureUpdate(rawBaselineLutId = SettingValue(lutId))
                BaselineColorCorrectionTarget.PHANTOM -> CameraFeatureUpdate(phantomBaselineLutId = SettingValue(lutId))
            }
            applyCameraFeatureUpdate(update)
        }
    }

    /**
     * 为指定颜色推荐最合适的 LUT 列表
     */
    suspend fun recommendLutsForColor(color: Int): List<LutInfo> = withContext(Dispatchers.IO) {
        contentRepository.lutManager.recommendLutsForColor(color)
    }

    /**
     * 设置降噪等级
     */
    fun setNRLevel(level: Int) {
        viewModelScope.launch {
            userPreferencesRepository.saveNRLevel(level)
        }
    }

    /**
     * 设置锐化等级
     */
    fun setEdgeLevel(level: Int) {
        viewModelScope.launch {
            userPreferencesRepository.saveEdgeLevel(level)
        }
    }

    /**
     * 设置照片质量
     */
    fun setPhotoQuality(quality: Int) {
        viewModelScope.launch {
            userPreferencesRepository.savePhotoQuality(quality)
        }
    }

    /**
     * 设置锐化强度
     */
    fun setSharpening(value: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveSharpening(value)
        }
    }

    /**
     * 设置降噪强度
     */
    fun setNoiseReduction(value: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveNoiseReduction(value)
        }
    }

    /**
     * 设置减少杂色强度
     */
    fun setChromaNoiseReduction(value: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveChromaNoiseReduction(value)
        }
    }

    /**
     * 设置摄像头方向偏移
     * @param cameraId 摄像头 ID
     * @param offset 旋转偏移角度 (0, 90, 180, 270)
     */
    fun setCameraOrientationOffset(cameraId: String, offset: Int) {
        viewModelScope.launch {
            userPreferencesRepository.saveCameraOrientationOffset(cameraId, offset)
        }
    }

    /**
     * 设置默认焦段
     */
    fun setDefaultFocalLength(focalLength: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveDefaultFocalLength(focalLength)
        }
    }


    fun setCustomLensIds(value: String) {
        viewModelScope.launch {
            val lensIds = value.split(",")
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .distinct()
            userPreferencesRepository.saveCustomLensIds(lensIds)
            cameraController.refreshCameraList()
        }
    }

    fun setLensIdBlacklist(value: String) {
        viewModelScope.launch {
            val lensIds = value.split(",")
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .distinct()
            userPreferencesRepository.saveLensIdBlacklist(lensIds)
            cameraController.refreshCameraList()
        }
    }

    /**
     * 应用默认焦段
     */
    private fun applyDefaultFocalLength(focalLength: Float) {
        val currentState = state.value
        val availableCameras = currentState.availableCameras
        val currentCamera = currentState.getCurrentCameraInfo() ?: return

        // 找到主摄来计算变焦倍率
        val mainCamera = availableCameras.find {
            it.lensType == if (currentCamera.lensType == LensType.FRONT) LensType.FRONT else LensType.BACK_MAIN
        } ?: return

        if (mainCamera.focalLength35mmEquivalent <= 0) return

        val targetZoom = focalLength / mainCamera.focalLength35mmEquivalent

        // 找到该变焦倍率下的最佳镜头
        val optimalLens = findOptimalLens(targetZoom, availableCameras, currentCamera.cameraId)
        if (optimalLens != null && optimalLens.cameraId != currentCamera.cameraId) {
            switchToLensAndSetZoomRatio(optimalLens.cameraId, targetZoom)
        } else {
            setZoomRatio(targetZoom)
        }
        PLog.d(TAG, "Applied default focal length: ${focalLength}mm (zoom: $targetZoom)")
    }

    /**
     * 获取摄像头方向偏移
     * @param cameraId 摄像头 ID
     * @return 旋转偏移角度的 Flow
     */
    fun getCameraOrientationOffset(cameraId: String): Flow<Int> {
        return userPreferencesRepository.userPreferences.map { prefs ->
            prefs.cameraOrientationOffsets[cameraId] ?: 0
        }
    }

    /**
     * 保存图片
     */
    private suspend fun saveImage(
        image: SafeImage,
        captureInfo: CaptureInfo,
        characteristics: CameraCharacteristics?,
        captureResult: CaptureResult?
    ) {
        var ownsImage = true
        try {
            PLog.d(TAG, "saveImage started - dimensions: ${image.width}x${image.height}, format: ${image.format}")
            val context = getApplication<Application>()

            // 保存当前配置信息
            val lutIdToSave = currentLutId.value
            val aspectRatio = state.value.aspectRatio
            val frameIdToSave = currentFrameId
            val shouldAutoSave = autoSaveAfterCapture.firstOrNull() ?: false
            val sharpeningValue = sharpening.firstOrNull() ?: 0f
            val noiseReductionValue = noiseReduction.firstOrNull() ?: 0f
            val chromaNoiseReductionValue = chromaNoiseReduction.firstOrNull() ?: 0f
            val photoQualityValue = photoQuality.firstOrNull() ?: 95
            val droModeString = droMode.value
            val droModeForProcessing =
                com.hinnka.mycamera.raw.RawProcessingPreferences.DROMode.fromPersistedName(droModeString)
            val currentCameraId = cameraController.getCurrentCameraId()

            // 计算旋转角度
            val sensorOrientation = cameraController.getSensorOrientation()
            val lensFacing = cameraController.getLensFacing()
            val deviceRotation = OrientationObserver.captureRotationDegrees.toInt()

            // 基础旋转角度计算
            val baseRotation = if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
                (sensorOrientation - deviceRotation + 360) % 360
            } else {
                (sensorOrientation + deviceRotation) % 360
            }

            // 获取用户配置的摄像头方向偏移
            val userPrefs = userPreferencesRepository.userPreferences.firstOrNull()
            val orientationOffset = userPrefs?.cameraOrientationOffsets?.get(currentCameraId) ?: 0

            // 应用方向偏移
            val rotation = (baseRotation + orientationOffset) % 360

            val shouldMirror = lensFacing == CameraCharacteristics.LENS_FACING_FRONT &&
                    (userPreferencesRepository.userPreferences.firstOrNull()?.mirrorFrontCamera ?: true)

            val aperture = if (state.value.isVirtualApertureEnabled) state.value.virtualAperture else null
            val defaultHdrEffectEnabled = defaultHdrEffectEnabled(
                hasEmbeddedGainmap = false,
                userPrefs = userPrefs
            )
            val baselineTarget = if (isRawCaptureFormat(image.format)) {
                BaselineColorCorrectionTarget.RAW
            } else {
                BaselineColorCorrectionTarget.JPG
            }
            val baselineMetadata = resolveBaselineMetadata(baselineTarget, userPrefs)
            val spectralFilmSettings = resolveRawSpectralFilmSettings(userPrefs)

            // 创建统一的 PhotoMetadata，包含编辑配置和拍摄信息
            val metadata = MediaMetadata(
                lutId = lutIdToSave,
                frameId = frameIdToSave,
                colorRecipeParams = getMergedRecipeParams(),
                baselineTarget = baselineMetadata?.first,
                baselineLutId = baselineMetadata?.second,
                baselineColorRecipeParams = baselineMetadata?.third,
                sharpening = sharpeningValue,
                noiseReduction = noiseReductionValue,
                chromaNoiseReduction = chromaNoiseReductionValue,
                rawDcpId = userPrefs?.rawDcpId,
                rawDenoiseValue = userPrefs?.rawNlmNoiseFactor ?: 0f,
                rawExposureCompensation = userPrefs?.rawExposureCompensation ?: 0f,
                rawAutoExposure = userPrefs?.rawAutoExposure ?: true,
                rawBlackPointCorrection = userPrefs?.rawBlackPointCorrection ?: 0f,
                rawWhitePointCorrection = userPrefs?.rawWhitePointCorrection ?: 0f,
                rawAutoWhiteBalanceEstimate = userPrefs?.rawAutoWhiteBalanceEstimate ?: false,
                rawBlackLevelMode = userPrefs?.rawBlackLevelModes?.get(currentCameraId) ?: "Default",
                rawCustomBlackLevel = userPrefs?.rawCustomBlackLevels?.get(currentCameraId) ?: 0f,
                cameraId = currentCameraId,
                spectralFilmEnabled = spectralFilmSettings.enabled,
                spectralFilmStock = spectralFilmSettings.stock,
                spectralFilmPrint = spectralFilmSettings.print,
                width = image.width,
                height = image.height,
                ratio = aspectRatio,
                rotation = rotation,
                deviceModel = captureInfo.model,
                brand = captureInfo.make,
                dateTaken = captureInfo.captureTime,
                latitude = captureInfo.latitude,
                longitude = captureInfo.longitude,
                altitude = captureInfo.altitude,
                iso = captureInfo.iso,
                shutterSpeed = captureInfo.formatExposureTime(),
                focalLength = captureInfo.formatFocalLength(),
                focalLength35mm = captureInfo.formatFocalLength35mm(),
                aperture = captureInfo.formatAperture(),
                exposureBias = state.value.exposureBias,
                droMode = droModeString,
                isMirrored = shouldMirror,
                colorSpace = captureInfo.colorSpace,
                dynamicRangeProfile = state.value.currentDynamicRangeProfile,
                computationalAperture = aperture,
                focusPointX = state.value.focusPoint?.first,
                focusPointY = state.value.focusPoint?.second,
                manualHdrEffectEnabled = defaultHdrEffectEnabled
            )

            val livePhotoVideoDeferred = if (useLivePhoto.value) {
                val deferred = CompletableDeferred<Pair<File, Long>?>()
                cameraController.recordLivePhotoVideo(image.timestamp / 1000) { file, ts ->
                    deferred.complete(if (file.name == "error") null else Pair(file, ts))
                }
                deferred
            } else null

            val resolvedCharacteristics = characteristics ?: run {
                PLog.e(TAG, "Failed to save image: camera characteristics unavailable")
                return
            }
            val photoId =
                GalleryManager.preparePhoto(
                    context,
                    metadata,
                    captureResult,
                    previewThumbnail,
                    useLivePhoto.value,
                    1.0f,
                    includeCropRegionInOutputSize = shouldIncludeCropRegionInOutputSize(image.format)
                )
            if (photoId == null) {
                PLog.e(TAG, "Failed to save image")
                return
            }
            ownsImage = false
            viewModelScope.launch(Dispatchers.IO) {
                GalleryManager.saveVideo(context, photoId, livePhotoVideoDeferred)

                GalleryManager.savePhoto(
                    context,
                    photoId,
                    image,
                    previewThumbnail,
                    rotation,
                    aspectRatio,
                    resolvedCharacteristics,
                    captureResult,
                    shouldAutoSave,
                    contentRepository.photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQualityValue,
                    exposureBias = state.value.exposureBias,
                    droMode = droModeForProcessing,
                    exportDngWithRawExport = exportDngWithRawExport.value,
                )
            }
            PLog.d(TAG, "Image saved: $photoId, LUT: $lutIdToSave, Frame: $frameIdToSave")
            _imageSavedEvent.emit(Unit)
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to save image", e)
        } finally {
            if (ownsImage) {
                image.close()
            }
        }
    }

    private suspend fun saveVideoSnapshot(bitmap: Bitmap) {
        try {
            val context = getApplication<Application>()
            val currentState = state.value
            val shouldAutoSave = autoSaveAfterCapture.firstOrNull() ?: false
            val sharpeningValue = sharpening.firstOrNull() ?: 0f
            val noiseReductionValue = noiseReduction.firstOrNull() ?: 0f
            val chromaNoiseReductionValue = chromaNoiseReduction.firstOrNull() ?: 0f
            val photoQualityValue = photoQuality.firstOrNull() ?: 95
            val userPrefs = userPreferencesRepository.userPreferences.firstOrNull()
            val shouldMirror = cameraController.getLensFacing() == CameraCharacteristics.LENS_FACING_FRONT &&
                    (userPrefs?.mirrorFrontCamera ?: true)
            val baselineMetadata = resolveBaselineMetadata(BaselineColorCorrectionTarget.JPG, userPrefs)
            val currentCameraId = cameraController.getCurrentCameraId()
            val spectralFilmSettings = resolveRawSpectralFilmSettings(userPrefs)

            val metadata = MediaMetadata(
                lutId = currentLutId.value,
                frameId = currentFrameId,
                colorRecipeParams = currentRecipeParams.value,
                baselineTarget = baselineMetadata?.first,
                baselineLutId = baselineMetadata?.second,
                baselineColorRecipeParams = baselineMetadata?.third,
                sharpening = sharpeningValue,
                noiseReduction = noiseReductionValue,
                chromaNoiseReduction = chromaNoiseReductionValue,
                rawDcpId = userPrefs?.rawDcpId,
                rawDenoiseValue = userPrefs?.rawNlmNoiseFactor ?: 0f,
                rawExposureCompensation = userPrefs?.rawExposureCompensation ?: 0f,
                rawAutoExposure = userPrefs?.rawAutoExposure ?: true,
                rawBlackPointCorrection = userPrefs?.rawBlackPointCorrection ?: 0f,
                rawWhitePointCorrection = userPrefs?.rawWhitePointCorrection ?: 0f,
                rawAutoWhiteBalanceEstimate = userPrefs?.rawAutoWhiteBalanceEstimate ?: false,
                rawBlackLevelMode = userPrefs?.rawBlackLevelModes?.get(currentCameraId) ?: "Default",
                rawCustomBlackLevel = userPrefs?.rawCustomBlackLevels?.get(currentCameraId) ?: 0f,
                cameraId = currentCameraId,
                spectralFilmEnabled = spectralFilmSettings.enabled,
                spectralFilmStock = spectralFilmSettings.stock,
                spectralFilmPrint = spectralFilmSettings.print,
                width = bitmap.width,
                height = bitmap.height,
                ratio = mapVideoAspectRatioToPhotoAspectRatio(currentState.videoConfig.aspectRatio),
                rotation = 0,
                deviceModel = Build.MODEL,
                brand = Build.MANUFACTURER,
                dateTaken = System.currentTimeMillis(),
                latitude = currentState.latitude,
                longitude = currentState.longitude,
                exposureBias = currentState.exposureBias,
                droMode = droMode.value,
                isMirrored = shouldMirror,
                dynamicRangeProfile = currentState.currentDynamicRangeProfile,
                captureMode = "video_snapshot"
            )

            val photoId = GalleryManager.preparePhoto(
                context,
                metadata,
                null,
                bitmap,
                false,
                1.0f,
                includeCropRegionInOutputSize = false
            )
            if (photoId == null) {
                PLog.e(TAG, "Failed to prepare video snapshot")
                return
            }

            withContext(Dispatchers.IO) {
                GalleryManager.saveBitmapPhoto(
                    context,
                    photoId,
                    bitmap,
                    shouldAutoSave,
                    contentRepository.photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQualityValue
                )
            }
            PLog.d(TAG, "Video snapshot saved: $photoId")
            _imageSavedEvent.emit(Unit)
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to save video snapshot", e)
        }
    }

    private fun mapVideoAspectRatioToPhotoAspectRatio(aspectRatio: VideoAspectRatio): AspectRatio? {
        return when (aspectRatio) {
            VideoAspectRatio.RATIO_16_9 -> AspectRatio.RATIO_16_9
            else -> null
        }
    }

    private suspend fun processStacking(
        images: List<SafeImage>,
        captureInfo: CaptureInfo,
        characteristics: CameraCharacteristics?,
        captureResult: CaptureResult?
    ) {
        try {
            PLog.d(TAG, "processStacking started - image size ${images.size}")
            val context = getApplication<Application>()

            // 保存当前配置信息
            val lutIdToSave = currentLutId.value
            val aspectRatio = state.value.aspectRatio
            val frameIdToSave = currentFrameId
            val shouldAutoSave = autoSaveAfterCapture.firstOrNull() ?: false
            val sharpeningValue = maxOf(sharpening.firstOrNull() ?: 0f, 0.4f)
            val noiseReductionValue = noiseReduction.firstOrNull() ?: 0f
            val chromaNoiseReductionValue = chromaNoiseReduction.firstOrNull() ?: 0f
            val photoQualityValue = photoQuality.firstOrNull() ?: 95
            val droModeString = droMode.value
            val currentCameraId = cameraController.getCurrentCameraId()

            // 计算旋转角度
            val sensorOrientation = cameraController.getSensorOrientation()
            val lensFacing = cameraController.getLensFacing()
            val deviceRotation = OrientationObserver.captureRotationDegrees.toInt()

            // 基础旋转角度计算
            val baseRotation = if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
                (sensorOrientation - deviceRotation + 360) % 360
            } else {
                (sensorOrientation + deviceRotation) % 360
            }

            val shouldMirror = lensFacing == CameraCharacteristics.LENS_FACING_FRONT &&
                    (userPreferencesRepository.userPreferences.firstOrNull()?.mirrorFrontCamera ?: true)

            // 获取用户配置的摄像头方向偏移
            val userPrefs = userPreferencesRepository.userPreferences.firstOrNull()
            val orientationOffset = userPrefs?.cameraOrientationOffsets?.get(currentCameraId) ?: 0

            // 应用方向偏移
            val rotation = (baseRotation + orientationOffset) % 360

            val useSuperRes = useMFSR.value
            val superResScale = if (useSuperRes) 2f else 1.0f

            val aperture = if (state.value.isVirtualApertureEnabled) state.value.virtualAperture else null
            val defaultHdrEffectEnabled = defaultHdrEffectEnabled(
                hasEmbeddedGainmap = false,
                userPrefs = userPrefs
            )
            val baselineTarget = if (images.firstOrNull()?.format?.let(::isRawCaptureFormat) == true) {
                BaselineColorCorrectionTarget.RAW
            } else {
                BaselineColorCorrectionTarget.JPG
            }
            val baselineMetadata = resolveBaselineMetadata(baselineTarget, userPrefs)
            val spectralFilmSettings = resolveRawSpectralFilmSettings(userPrefs)

            // 创建统一的 PhotoMetadata，包含编辑配置和拍摄信息
            val metadata = MediaMetadata(
                lutId = lutIdToSave,
                frameId = frameIdToSave,
                colorRecipeParams = currentRecipeParams.value,
                baselineTarget = baselineMetadata?.first,
                baselineLutId = baselineMetadata?.second,
                baselineColorRecipeParams = baselineMetadata?.third,
                sharpening = sharpeningValue,
                noiseReduction = noiseReductionValue,
                chromaNoiseReduction = chromaNoiseReductionValue,
                rawDcpId = userPrefs?.rawDcpId,
                rawDenoiseValue = userPrefs?.rawNlmNoiseFactor ?: 0f,
                rawExposureCompensation = userPrefs?.rawExposureCompensation ?: 0f,
                rawAutoExposure = userPrefs?.rawAutoExposure ?: true,
                rawBlackPointCorrection = userPrefs?.rawBlackPointCorrection ?: 0f,
                rawWhitePointCorrection = userPrefs?.rawWhitePointCorrection ?: 0f,
                rawAutoWhiteBalanceEstimate = userPrefs?.rawAutoWhiteBalanceEstimate ?: false,
                rawBlackLevelMode = userPrefs?.rawBlackLevelModes?.get(currentCameraId) ?: "Default",
                rawCustomBlackLevel = userPrefs?.rawCustomBlackLevels?.get(currentCameraId) ?: 0f,
                cameraId = currentCameraId,
                spectralFilmEnabled = spectralFilmSettings.enabled,
                spectralFilmStock = spectralFilmSettings.stock,
                spectralFilmPrint = spectralFilmSettings.print,
                width = (images[0].width.toFloat() * superResScale).roundToInt(),
                height = (images[0].height.toFloat() * superResScale).roundToInt(),
                ratio = aspectRatio,
                rotation = rotation,
                deviceModel = captureInfo.model,
                brand = captureInfo.make,
                dateTaken = captureInfo.captureTime,
                latitude = captureInfo.latitude,
                longitude = captureInfo.longitude,
                altitude = captureInfo.altitude,
                iso = captureInfo.iso,
                shutterSpeed = captureInfo.formatExposureTime(),
                focalLength = captureInfo.formatFocalLength(),
                focalLength35mm = captureInfo.formatFocalLength35mm(),
                aperture = captureInfo.formatAperture(),
                exposureBias = state.value.exposureBias,
                droMode = droModeString,
                isMirrored = shouldMirror,
                colorSpace = captureInfo.colorSpace,
                dynamicRangeProfile = state.value.currentDynamicRangeProfile,
                computationalAperture = aperture,
                focusPointX = state.value.focusPoint?.first,
                focusPointY = state.value.focusPoint?.second,
                manualHdrEffectEnabled = defaultHdrEffectEnabled
            )

            val livePhotoVideoDeferred = if (useLivePhoto.value) {
                val deferred = CompletableDeferred<Pair<File, Long>?>()
                images.firstOrNull()?.let {
                    cameraController.recordLivePhotoVideo(it.timestamp / 1000) { file, ts ->
                        deferred.complete(if (file.name == "error") null else Pair(file, ts))
                    }
                } ?: deferred.complete(null)
                deferred
            } else null

            characteristics ?: return
            val photoId = GalleryManager.preparePhoto(
                context,
                metadata,
                captureResult,
                previewThumbnail,
                useLivePhoto.value,
                superResScale,
                includeCropRegionInOutputSize = images.firstOrNull()?.let {
                    shouldIncludeCropRegionInOutputSize(it.format)
                } ?: false
            )
            if (photoId == null) {
                PLog.e(TAG, "Failed to save burst image")
                return
            }

            viewModelScope.launch(Dispatchers.IO) {
                GalleryManager.saveVideo(context, photoId, livePhotoVideoDeferred)

                GalleryManager.saveStackedPhoto(
                    context,
                    photoId,
                    images,
                    rotation,
                    aspectRatio,
                    characteristics,
                    captureResult,
                    shouldAutoSave,
                    contentRepository.photoProcessor,
                    sharpeningValue,
                    noiseReductionValue,
                    chromaNoiseReductionValue,
                    photoQualityValue,
                    useSuperResolution = useSuperRes,
                    superResolutionScale = superResScale,
                    useGpuAcceleration = useGpuAcceleration.value,
                    exposureBias = state.value.exposureBias,
                    exportDngWithRawExport = exportDngWithRawExport.value
                )
            }
            PLog.d(TAG, "Image saved: $photoId, LUT: $lutIdToSave, Frame: $frameIdToSave")
            _imageSavedEvent.emit(Unit)
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to save image", e)
        }
    }

    private suspend fun prepareBurst(context: Context, photoId: String, image: SafeImage, captureInfo: CaptureInfo) {
        // 保存当前配置信息
        val lutIdToSave = currentLutId.value
        val aspectRatio = state.value.aspectRatio
        val frameIdToSave = currentFrameId
        val sharpeningValue = sharpening.firstOrNull() ?: 0f
        val noiseReductionValue = noiseReduction.firstOrNull() ?: 0f
        val chromaNoiseReductionValue = chromaNoiseReduction.firstOrNull() ?: 0f
        val currentCameraId = cameraController.getCurrentCameraId()

        // 计算旋转角度
        val sensorOrientation = cameraController.getSensorOrientation()
        val lensFacing = cameraController.getLensFacing()
        val deviceRotation = OrientationObserver.captureRotationDegrees.toInt()

        // 基础旋转角度计算
        val baseRotation = if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            (sensorOrientation - deviceRotation + 360) % 360
        } else {
            (sensorOrientation + deviceRotation) % 360
        }

        // 获取用户配置的摄像头方向偏移
        val userPrefs = userPreferencesRepository.userPreferences.firstOrNull()
        val orientationOffset = userPrefs?.cameraOrientationOffsets?.get(currentCameraId) ?: 0

        // 应用方向偏移
        val rotation = (baseRotation + orientationOffset) % 360

        val shouldMirror = lensFacing == CameraCharacteristics.LENS_FACING_FRONT &&
                (userPreferencesRepository.userPreferences.firstOrNull()?.mirrorFrontCamera ?: true)

        val aperture = if (state.value.isVirtualApertureEnabled) state.value.virtualAperture else null
        val defaultHdrEffectEnabled = defaultHdrEffectEnabled(
            hasEmbeddedGainmap = false,
            userPrefs = userPrefs
        )
        val baselineTarget = if (isRawCaptureFormat(image.format)) {
            BaselineColorCorrectionTarget.RAW
        } else {
            BaselineColorCorrectionTarget.JPG
        }
        val baselineMetadata = resolveBaselineMetadata(baselineTarget, userPrefs)
        val spectralFilmSettings = resolveRawSpectralFilmSettings(userPrefs)

        // 创建统一的 PhotoMetadata，包含编辑配置和拍摄信息
        val metadata = MediaMetadata(
            lutId = lutIdToSave,
            frameId = frameIdToSave,
            colorRecipeParams = currentRecipeParams.value,
            baselineTarget = baselineMetadata?.first,
            baselineLutId = baselineMetadata?.second,
            baselineColorRecipeParams = baselineMetadata?.third,
            sharpening = sharpeningValue,
            noiseReduction = noiseReductionValue,
            chromaNoiseReduction = chromaNoiseReductionValue,
            rawDcpId = userPrefs?.rawDcpId,
            rawDenoiseValue = userPrefs?.rawNlmNoiseFactor ?: 0f,
            rawExposureCompensation = userPrefs?.rawExposureCompensation ?: 0f,
            rawAutoExposure = userPrefs?.rawAutoExposure ?: true,
            rawBlackPointCorrection = userPrefs?.rawBlackPointCorrection ?: 0f,
            rawWhitePointCorrection = userPrefs?.rawWhitePointCorrection ?: 0f,
            rawAutoWhiteBalanceEstimate = userPrefs?.rawAutoWhiteBalanceEstimate ?: false,
            rawBlackLevelMode = userPrefs?.rawBlackLevelModes?.get(currentCameraId) ?: "Default",
            rawCustomBlackLevel = userPrefs?.rawCustomBlackLevels?.get(currentCameraId) ?: 0f,
            cameraId = currentCameraId,
            spectralFilmEnabled = spectralFilmSettings.enabled,
            spectralFilmStock = spectralFilmSettings.stock,
            spectralFilmPrint = spectralFilmSettings.print,
            width = image.width,
            height = image.height,
            ratio = aspectRatio,
            rotation = rotation,
            deviceModel = captureInfo.model,
            brand = captureInfo.make,
            dateTaken = captureInfo.captureTime,
            latitude = captureInfo.latitude,
            longitude = captureInfo.longitude,
            altitude = captureInfo.altitude,
            iso = captureInfo.iso,
            shutterSpeed = captureInfo.formatExposureTime(),
            focalLength = captureInfo.formatFocalLength(),
            focalLength35mm = captureInfo.formatFocalLength35mm(),
            aperture = captureInfo.formatAperture(),
            isMirrored = shouldMirror,
            colorSpace = captureInfo.colorSpace,
            dynamicRangeProfile = state.value.currentDynamicRangeProfile,
            computationalAperture = aperture,
            focusPointX = state.value.focusPoint?.first,
            focusPointY = state.value.focusPoint?.second,
            manualHdrEffectEnabled = defaultHdrEffectEnabled
        )

        GalleryManager.preparePhoto(
            context,
            metadata,
            null,
            previewThumbnail,
            false,
            1.0f,
            photoId = photoId
        )
    }

    private suspend fun processBurst() = withContext(Dispatchers.IO) {
        while (true) {
            if (!state.value.burstCapturing && burstImages.isEmpty()) run {
                burstPhotoId = null
                burstImageCount = 0
                burstCaptureInfo = null
                break
            }
            val image = burstImages.removeFirstOrNull() ?: run {
                delay(33)
                continue
            }
            val captureInfo = burstCaptureInfo ?: run {
                delay(33)
                continue
            }
            val context = getApplication<Application>()
            val photoId = burstPhotoId ?: run {
                delay(33)
                continue
            }
            burstImageCount++
            try {
                val metadata = GalleryManager.loadMetadata(context, photoId)
                if (metadata == null) {
                    prepareBurst(context, photoId, image, captureInfo)
                }
                val shouldAutoSave = autoSaveAfterCapture.firstOrNull() ?: false
                val photoQualityValue = photoQuality.firstOrNull() ?: 95
                GalleryManager.saveBurstPhoto(
                    context,
                    photoId,
                    image,
                    shouldAutoSave,
                    contentRepository.photoProcessor,
                    photoQualityValue,
                )
//                PLog.d(TAG, "Image saved: $photoId")
            } catch (e: Exception) {
                PLog.e(TAG, "Failed to save image", e)
            }
        }
    }

    fun getAvailableFocalLengths(): List<Float> {
        val currentState = state.value
        val availableCameras = currentState.availableCameras
        if (availableCameras.isEmpty()) return emptyList()

        val mainCamera = availableCameras.find {
            it.lensType == LensType.BACK_MAIN
        } ?: availableCameras.firstOrNull { it.lensFacing == CameraCharacteristics.LENS_FACING_BACK }
        ?: return emptyList()

        if (mainCamera.focalLength35mmEquivalent <= 0) return emptyList()

        val lensZoomStops = calculateLensZoomStops(availableCameras, mainCamera)
        val stops = lensZoomStops.toMutableList()
        addDefaultMinimumZoomStop(stops, lensZoomStops, mainCamera)

        return stops
            .map { it * mainCamera.focalLength35mmEquivalent }
            .distinctBy { it.roundToInt() }
            .sorted()
    }

    /**
     * 设置背景图
     */
    fun setBackgroundImage(image: String) {
        viewModelScope.launch {
            userPreferencesRepository.saveBackgroundImage(image)
        }
    }

    /**
     * 设置多帧合成是否使用 GPU 加速
     */
    fun setUseGpuAcceleration(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveUseGpuAcceleration(enabled)
        }
    }

    /**
     * 保存从外部选择的背景图
     */
    fun saveCustomBackgroundImage(uri: Uri) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                try {
                    val context = getApplication<Application>()
                    val inputStream = context.contentResolver.openInputStream(uri)
                    if (inputStream != null) {
                        val backgroundDir = File(context.filesDir, "backgrounds")
                        if (!backgroundDir.exists()) {
                            backgroundDir.mkdirs()
                        }
                        val fileName = "custom_bg_${System.currentTimeMillis()}.jpg"
                        val outputFile = File(backgroundDir, fileName)
                        inputStream.use { input ->
                            outputFile.outputStream().use { output ->
                                input.copyTo(output)
                            }
                        }
                        setBackgroundImage(outputFile.absolutePath)
                    }
                } catch (e: Exception) {
                    PLog.e(TAG, "Failed to save custom background image", e)
                }
            }
        }
    }


    fun onShutterAnimationTriggered() {
        _canStartShutterAnimation.value = true
        viewModelScope.launch {
            // Delay prewarming to avoid stuttering during the initial reveal animation
            // 150ms initial delay + 800ms animation + 250ms buffer
            delay(1200)
            prewarmDepthEstimator()
        }
    }

    override fun onCleared() {
        super.onCleared()
        cameraController.release()
        contentRepository.lutManager.clearCache()
        contentRepository.frameManager.clearCache()
        shutterSoundPlayer.release()
        videoAudioInputManager.release()

        // 清理未处理的连拍图片
        stackingImages.forEach {
            it.close()
        }
        stackingImages.clear()
        burstImages.forEach {
            it.close()
        }
        burstImages.clear()
        burstImageCount = 0
        multipleExposureState.previewBitmap?.recycle()
        multipleExposureState.sessionId?.let { GalleryManager.clearMultipleExposureSession(getApplication(), it) }
    }

    /**
     * 设置当前 RAW 动态范围
     */
    fun setDroMode(mode: String) {
        viewModelScope.launch {
            applyCameraFeatureUpdate(
                CameraFeatureUpdate(droMode = SettingValue(mode))
            )
        }
    }

    /**
     * 设置色调映射模式
     */
    fun setTonemapMode(mode: String) {
        viewModelScope.launch {
            userPreferencesRepository.saveTonemapMode(mode)
        }
    }

    fun setFixTonemapPreview(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveFixTonemapPreview(enabled)
        }
    }

    fun togglePhantomMode() {
        val newMode = !phantomMode.value
        viewModelScope.launch {
            userPreferencesRepository.savePhantomMode(newMode)
        }
    }

    fun setPhantomButtonHidden(hidden: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.savePhantomButtonHidden(hidden)
        }
    }

    fun setLaunchCameraOnPhantomMode(launch: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveLaunchCameraOnPhantomMode(launch)
        }
    }

    fun setPhantomPipPreview(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.savePhantomPipPreview(enabled)
        }
    }

    fun setPhantomPipCrop(crop: PhantomPipCrop) {
        viewModelScope.launch {
            userPreferencesRepository.savePhantomPipCrop(crop)
        }
    }

    fun setPhantomSaveAsNew(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.savePhantomSaveAsNew(enabled)
        }
    }

    fun setDefaultVirtualAperture(aperture: Float) {
        viewModelScope.launch {
            userPreferencesRepository.saveDefaultVirtualAperture(aperture)
        }
    }

    /**
     * 设置是否启用自拍镜像
     */
    fun setMirrorFrontCamera(enabled: Boolean) {
        viewModelScope.launch {
            userPreferencesRepository.saveMirrorFrontCamera(enabled)
        }
    }

    /**
     * 设置 Widget 主题
     */
    fun setWidgetTheme(theme: com.hinnka.mycamera.data.WidgetTheme) {
        viewModelScope.launch {
            userPreferencesRepository.saveWidgetTheme(theme)
            // 通知 Widget 更新
            val intent = Intent(
                getApplication<Application>(),
                PhantomWidgetProvider::class.java
            ).apply {
                action = android.appwidget.AppWidgetManager.ACTION_APPWIDGET_UPDATE
                val ids = android.appwidget.AppWidgetManager.getInstance(getApplication())
                    .getAppWidgetIds(
                        android.content.ComponentName(
                            getApplication(),
                            PhantomWidgetProvider::class.java
                        )
                    )
                putExtra(android.appwidget.AppWidgetManager.EXTRA_APPWIDGET_IDS, ids)
            }
            getApplication<Application>().sendBroadcast(intent)
        }
    }

    /**
     * 获取 LUT 的 .cube 字符串内容
     */
    suspend fun getLutCubeString(lutId: String): String? = withContext(Dispatchers.IO) {
        val lutInfo = contentRepository.lutManager.getLutInfo(lutId) ?: return@withContext null
        val lutConfig = contentRepository.lutManager.loadLut(lutId) ?: return@withContext null

        val floatBuffer = lutConfig.toFloatBuffer()
        val floatArray = FloatArray(floatBuffer.capacity())
        floatBuffer.position(0)
        floatBuffer.get(floatArray)

        LutGenerator.exportToCubeString(floatArray, lutConfig.size, lutInfo.getName())
    }

    /**
     * 将 LUT（含色彩配方）导出为 .plut v4 字节数组
     * 若该 LUT 没有色彩配方则导出为标准 v3 格式
     */
    suspend fun exportLutToPlut(lutId: String): ByteArray? = withContext(Dispatchers.IO) {
        val lutConfig = contentRepository.lutManager.loadLut(lutId) ?: return@withContext null
        val recipe = contentRepository.lutManager.loadColorRecipeParams(lutId)
        val recipeJson = if (!recipe.isDefault()) recipe.toJson() else null

        val outputStream = java.io.ByteArrayOutputStream()
        LutConverter.exportToPlut(lutConfig, outputStream, recipeJson)
        outputStream.toByteArray()
    }

    /**
     * 导出原始无损 .cube 字节数组
     */
    suspend fun exportLutToCube(lutId: String): ByteArray? = withContext(Dispatchers.IO) {
        getLutCubeString(lutId)?.toByteArray(Charsets.UTF_8)
    }

    /**
     * 将 LUT 和色彩配方永久烘焙并导出为标准 .cube 字节数组
     */
    suspend fun exportBakedLutToCube(lutId: String): ByteArray? = withContext(Dispatchers.IO) {
        val lutConfig = contentRepository.lutManager.loadLut(lutId) ?: return@withContext null
        val recipe = contentRepository.lutManager.loadColorRecipeParams(lutId)

        try {
            val lutInfo = contentRepository.lutManager.getLutInfo(lutId)
            val name = lutInfo?.getName() ?: "BakedLUT"
            BakedLutExporter.exportBakedCube(lutConfig, recipe, name)
        } catch (e: Exception) {
            PLog.e("CameraViewModel", "Failed to bake LUT to cube", e)
            null
        }
    }

    /**
     * 将 LUT 和色彩配方永久烘焙并导出为 HALD CLUT .png 字节数组
     */
    suspend fun exportBakedLutToHaldPng(lutId: String): ByteArray? = withContext(Dispatchers.IO) {
        val lutConfig = contentRepository.lutManager.loadLut(lutId) ?: return@withContext null
        val recipe = contentRepository.lutManager.loadColorRecipeParams(lutId)

        try {
            BakedLutExporter.exportBakedHaldPng(lutConfig, recipe)
        } catch (e: Exception) {
            PLog.e("CameraViewModel", "Failed to bake LUT to HALD PNG", e)
            null
        }
    }

    suspend fun exportFrameToJson(frame: FrameInfo): ByteArray? = withContext(Dispatchers.IO) {
        contentRepository.getCustomImportManager().exportFrameJson(frame)
    }

    /**
     * 从导入的 .plut URI 中提取嵌入的色彩配方并保存到指定 LUT（仅 v4 文件含有配方）
     */
    suspend fun extractAndSaveColorRecipeFromPlut(lutId: String, uri: Uri) = withContext(Dispatchers.IO) {
        try {
            val recipeJson = getApplication<Application>().contentResolver
                .openInputStream(uri)?.use { LutConverter.extractRecipeJsonFromPlut(it) }
                ?: return@withContext
            val params = ColorRecipeParams.fromJson(recipeJson)
            contentRepository.lutManager.saveColorRecipeParams(lutId, params)
        } catch (e: Exception) {
            PLog.e("CameraViewModel", "Failed to extract recipe from plut", e)
        }
    }

}

private fun shouldIncludeCropRegionInOutputSize(imageFormat: Int): Boolean {
    return when (imageFormat) {
        ImageFormat.RAW_SENSOR,
        ImageFormat.RAW10,
        ImageFormat.RAW12 -> true

        else -> false
    }
}
