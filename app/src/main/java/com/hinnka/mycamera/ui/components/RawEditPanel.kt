package com.hinnka.mycamera.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import android.graphics.Bitmap
import com.hinnka.mycamera.R
import com.hinnka.mycamera.lut.LutInfo
import com.hinnka.mycamera.raw.DcpInfo
import com.hinnka.mycamera.raw.RawCfaCorrection
import com.hinnka.mycamera.raw.RawProcessingPreferences.DROMode
import com.hinnka.mycamera.raw.RawColorEngine
import com.hinnka.mycamera.raw.SpectralFilmSelection
import com.hinnka.mycamera.raw.SpectralFilmUiInfo
import com.hinnka.mycamera.raw.SpectralFilmTuning
import kotlin.math.roundToInt

enum class RawEditPanelContentMode {
    FULL,
    QUICK,
}

@Composable
fun RawEditPanel(
    selectedDcpId: String?,
    availableDcps: List<DcpInfo>,
    selectedBaselineLutId: String?,
    onSelectBaselineLut: (String?) -> Unit,
    onEditBaselineRecipe: (String) -> Unit,
    availableLuts: List<LutInfo>,
    thumbnail: Bitmap?,
    rawNlmNoiseFactor: Float,
    rawExposureCompensation: Float,
    rawAutoExposure: Boolean,
    rawHighlightsAdjustment: Float,
    rawShadowsAdjustment: Float,
    rawBlackPointCorrection: Float,
    rawWhitePointCorrection: Float,
    rawBlackLevelMode: String = RawCfaCorrection.MODE_DEFAULT,
    rawCustomBlackLevel: Float = 0f,
    rawCfaCorrectionMode: String = RawCfaCorrection.MODE_DEFAULT,
    rawColorEngine: RawColorEngine,
    spectralFilmSelection: SpectralFilmSelection?,
    spectralFilmPrint: String?,
    onSelectDcp: (String?) -> Unit,
    onImportDcp: () -> Unit,
    onDeleteDcp: (DcpInfo) -> Unit,
    onRawNlmNoiseFactorChange: (Float) -> Unit,
    onRawExposureCompensationChange: (Float) -> Unit,
    onRawAutoExposureChange: (Boolean) -> Unit,
    onRawHighlightsAdjustmentChange: (Float) -> Unit,
    onRawShadowsAdjustmentChange: (Float) -> Unit,
    onRawBlackPointCorrectionChange: (Float) -> Unit,
    onRawWhitePointCorrectionChange: (Float) -> Unit,
    onRawBlackLevelModeChange: (String) -> Unit = {},
    onRawCustomBlackLevelChange: (Float) -> Unit = {},
    onRawCfaCorrectionModeChange: (String) -> Unit = {},
    onRawColorEngineChange: (RawColorEngine) -> Unit,
    onSpectralFilmSelectionChange: (SpectralFilmSelection?) -> Unit,
    onSpectralFilmPrintChange: (String?) -> Unit,
    onAdjustmentStart: () -> Unit,
    onAdjustmentEnd: () -> Unit,
    onOpenBaselineLutSheet: (() -> Unit)? = null,
    showDngMetadataControls: Boolean = false,
    contentMode: RawEditPanelContentMode = RawEditPanelContentMode.FULL,
    modifier: Modifier = Modifier
) {
    var localSpectralFilmTuning by remember { mutableStateOf(spectralFilmSelection?.tuning ?: SpectralFilmTuning.DEFAULT) }

    LaunchedEffect(spectralFilmSelection) {
        localSpectralFilmTuning = spectralFilmSelection?.tuning ?: SpectralFilmTuning.DEFAULT
    }

    fun commitSpectralFilmDensityGains() {
        val selection = spectralFilmSelection ?: return
        onSpectralFilmSelectionChange(selection.copy(tuning = localSpectralFilmTuning))
        onAdjustmentEnd()
    }
    var spectralFilmTuningExpanded by remember(spectralFilmSelection?.id) { mutableStateOf(false) }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        RawColorEngineSelector(
            selectedEngine = rawColorEngine,
            onSelectEngine = onRawColorEngineChange
        )
        Spacer(modifier = Modifier.height(16.dp))

        RawDcpSelector(
            selectedDcpId = selectedDcpId,
            availableDcps = availableDcps,
            showNonAdobeCurveWarning = rawColorEngine != RawColorEngine.AdobeCurve,
            onSelectDcp = onSelectDcp,
            onImportDcp = onImportDcp,
            onDeleteDcp = onDeleteDcp
        )
        Spacer(modifier = Modifier.height(16.dp))

        if (rawColorEngine == RawColorEngine.Spektrafilm) {
            val isPositiveFilm = SpectralFilmUiInfo.isPositiveFilm(spectralFilmSelection?.id)
            RawSpectralFilmSelector(
                selectedFilm = spectralFilmSelection?.id,
                onSelectFilm = { film ->
                    onSpectralFilmSelectionChange(film?.let { SpectralFilmSelection(it) })
                }
            )
            if (!isPositiveFilm) {
                Spacer(modifier = Modifier.height(16.dp))
                RawSpectralPrintSelector(
                    selectedPrint = spectralFilmPrint,
                    onSelectPrint = onSpectralFilmPrintChange
                )
                Spacer(modifier = Modifier.height(16.dp))
                if (contentMode != RawEditPanelContentMode.QUICK) {
                    RawSpectralFilmTuningHeader(
                        expanded = spectralFilmTuningExpanded,
                        onClick = { spectralFilmTuningExpanded = !spectralFilmTuningExpanded }
                    )
                    if (spectralFilmTuningExpanded) {
                        SliderSettingItem(
                            title = stringResource(R.string.settings_spectral_film_c_density_gain),
                            value = localSpectralFilmTuning.cDensityGain,
                            valueRange = SpectralFilmTuning.MIN_DENSITY_GAIN..SpectralFilmTuning.MAX_DENSITY_GAIN,
                            resetValue = 1f,
                            valueTextFormatter = { "${(it * 100f).roundToInt()}%" },
                            onValueChange = {
                                onAdjustmentStart()
                                localSpectralFilmTuning = localSpectralFilmTuning.copy(cDensityGain = it)
                            },
                            onValueChangeFinished = ::commitSpectralFilmDensityGains
                        )
                        SliderSettingItem(
                            title = stringResource(R.string.settings_spectral_film_m_density_gain),
                            value = localSpectralFilmTuning.mDensityGain,
                            valueRange = SpectralFilmTuning.MIN_DENSITY_GAIN..SpectralFilmTuning.MAX_DENSITY_GAIN,
                            resetValue = 1f,
                            valueTextFormatter = { "${(it * 100f).roundToInt()}%" },
                            onValueChange = {
                                onAdjustmentStart()
                                localSpectralFilmTuning = localSpectralFilmTuning.copy(mDensityGain = it)
                            },
                            onValueChangeFinished = ::commitSpectralFilmDensityGains
                        )
                        SliderSettingItem(
                            title = stringResource(R.string.settings_spectral_film_y_density_gain),
                            value = localSpectralFilmTuning.yDensityGain,
                            valueRange = SpectralFilmTuning.MIN_DENSITY_GAIN..SpectralFilmTuning.MAX_DENSITY_GAIN,
                            resetValue = 1f,
                            valueTextFormatter = { "${(it * 100f).roundToInt()}%" },
                            onValueChange = {
                                onAdjustmentStart()
                                localSpectralFilmTuning = localSpectralFilmTuning.copy(yDensityGain = it)
                            },
                            onValueChangeFinished = ::commitSpectralFilmDensityGains
                        )
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                }
            } else {
                Spacer(modifier = Modifier.height(16.dp))
            }
        }

        RawBaselineColorCorrectionSelector(
            selectedLutId = selectedBaselineLutId,
            availableLuts = availableLuts,
            thumbnail = thumbnail,
            onSelectLut = onSelectBaselineLut,
            onEditRecipe = onEditBaselineRecipe,
            onOpenSheet = onOpenBaselineLutSheet
        )
        Spacer(modifier = Modifier.height(16.dp))

        if (contentMode != RawEditPanelContentMode.QUICK) {
            RawSwitchSettingItem(
                title = stringResource(R.string.settings_raw_auto_exposure),
                description = stringResource(R.string.settings_raw_auto_exposure_description),
                checked = rawAutoExposure,
                onCheckedChange = onRawAutoExposureChange
            )
            SliderSettingItem(
                title = stringResource(R.string.settings_raw_exposure_compensation),
                value = rawExposureCompensation,
                valueRange = -2f..2f,
                resetValue = 0f,
                onValueChange = {
                    onAdjustmentStart()
                    onRawExposureCompensationChange(it)
                },
                onValueChangeFinished = onAdjustmentEnd
            )
            SliderSettingItem(
                title = stringResource(R.string.settings_raw_highlights_adjustment),
                value = rawHighlightsAdjustment,
                valueRange = -1f..1f,
                resetValue = 0f,
                onValueChange = {
                    onAdjustmentStart()
                    onRawHighlightsAdjustmentChange(it)
                },
                onValueChangeFinished = onAdjustmentEnd
            )
            SliderSettingItem(
                title = stringResource(R.string.settings_raw_shadows_adjustment),
                value = rawShadowsAdjustment,
                valueRange = -1f..1f,
                resetValue = 0f,
                onValueChange = {
                    onAdjustmentStart()
                    onRawShadowsAdjustmentChange(it)
                },
                onValueChangeFinished = onAdjustmentEnd
            )
            SliderSettingItem(
                title = stringResource(R.string.settings_raw_nlm_noise_factor),
                value = rawNlmNoiseFactor,
                valueRange = 0f..1f,
                resetValue = 0f,
                onValueChange = {
                    onAdjustmentStart()
                    onRawNlmNoiseFactorChange(it)
                },
                onValueChangeFinished = onAdjustmentEnd
            )
            SliderSettingItem(
                title = stringResource(R.string.settings_raw_black_point_correction),
                value = rawBlackPointCorrection,
                valueRange = -0.25f..0.25f,
                resetValue = 0f,
                onValueChange = {
                    onAdjustmentStart()
                    onRawBlackPointCorrectionChange(it)
                },
                onValueChangeFinished = onAdjustmentEnd
            )
            SliderSettingItem(
                title = stringResource(R.string.settings_raw_white_point_correction),
                value = rawWhitePointCorrection,
                valueRange = -0.5f..0.5f,
                resetValue = 0f,
                onValueChange = {
                    onAdjustmentStart()
                    onRawWhitePointCorrectionChange(it)
                },
                onValueChangeFinished = onAdjustmentEnd
            )
            if (showDngMetadataControls) {
                RawChoiceSetting(
                    title = stringResource(R.string.settings_raw_black_level_correction),
                    description = stringResource(R.string.settings_raw_black_level_correction_description),
                    levels = listOf(
                        RawCfaCorrection.MODE_DEFAULT to stringResource(R.string.settings_black_level_default),
                        "0" to "0",
                        "16" to "16",
                        "64" to "64",
                        "256" to "256",
                        "512" to "512",
                        "Custom" to stringResource(R.string.settings_black_level_custom)
                    ),
                    currentLevel = rawBlackLevelMode,
                    onLevelSelected = onRawBlackLevelModeChange
                )
                if (rawBlackLevelMode == "Custom") {
                    RawNumberInputSetting(
                        title = stringResource(R.string.settings_black_level_custom),
                        value = rawCustomBlackLevel,
                        onValueChange = onRawCustomBlackLevelChange
                    )
                }
                RawChoiceSetting(
                    title = stringResource(R.string.settings_raw_cfa_correction),
                    description = stringResource(R.string.settings_raw_cfa_correction_description),
                    levels = listOf(
                        RawCfaCorrection.MODE_DEFAULT to stringResource(R.string.settings_cfa_correction_default),
                        RawCfaCorrection.MODE_2X2_RGGB to stringResource(R.string.settings_cfa_correction_2x2_rggb),
                        RawCfaCorrection.MODE_2X2_GRBG to stringResource(R.string.settings_cfa_correction_2x2_grbg),
                        RawCfaCorrection.MODE_2X2_GBRG to stringResource(R.string.settings_cfa_correction_2x2_gbrg),
                        RawCfaCorrection.MODE_2X2_BGGR to stringResource(R.string.settings_cfa_correction_2x2_bggr),
                        RawCfaCorrection.MODE_4X4_RGGB to stringResource(R.string.settings_cfa_correction_4x4_rggb),
                        RawCfaCorrection.MODE_4X4_GRBG to stringResource(R.string.settings_cfa_correction_4x4_grbg),
                        RawCfaCorrection.MODE_4X4_GBRG to stringResource(R.string.settings_cfa_correction_4x4_gbrg),
                        RawCfaCorrection.MODE_4X4_BGGR to stringResource(R.string.settings_cfa_correction_4x4_bggr)
                    ),
                    currentLevel = rawCfaCorrectionMode,
                    onLevelSelected = onRawCfaCorrectionModeChange
                )
            }
        }
    }
}

@Composable
private fun RawChoiceSetting(
    title: String,
    description: String,
    levels: List<Pair<String, String>>,
    currentLevel: String,
    onLevelSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Text(
            text = title,
            color = Color.White,
            fontSize = 16.sp,
            fontWeight = FontWeight.Normal
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = description,
            color = Color.White.copy(alpha = 0.6f),
            fontSize = 13.sp,
            lineHeight = 18.sp
        )
        Spacer(modifier = Modifier.height(12.dp))
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            levels.forEach { (level, label) ->
                val selected = currentLevel == level
                Box(
                    modifier = Modifier
                        .height(36.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (selected) Color(0xFFFF6B35) else Color.White.copy(alpha = 0.1f))
                        .clickable { onLevelSelected(level) }
                        .padding(horizontal = 12.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = label,
                        color = Color.White,
                        fontSize = 12.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}

@Composable
private fun RawNumberInputSetting(
    title: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    var text by remember(value) {
        mutableStateOf(if (value == 0f) "" else value.toString())
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Text(
            text = title,
            color = Color.White,
            fontSize = 16.sp,
            fontWeight = FontWeight.Normal
        )
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedTextField(
            value = text,
            onValueChange = { input ->
                text = input
                input.toFloatOrNull()?.let(onValueChange)
            },
            singleLine = true,
            textStyle = LocalTextStyle.current.copy(color = Color.White),
            modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                cursorColor = Color.White,
                focusedBorderColor = Color(0xFFFF6B35),
                unfocusedBorderColor = Color.White.copy(alpha = 0.25f),
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent
            )
        )
    }
}

@Composable
private fun RawSpectralFilmTuningHeader(
    expanded: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.settings_spectral_film_tuning),
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = stringResource(R.string.settings_spectral_film_tuning_description),
                color = Color.White.copy(alpha = 0.55f),
                fontSize = 13.sp,
                lineHeight = 18.sp
            )
        }
        Icon(
            imageVector = Icons.Default.ChevronRight,
            contentDescription = null,
            tint = Color.White.copy(alpha = 0.75f),
            modifier = Modifier
                .size(22.dp)
                .rotate(if (expanded) 90f else 0f)
        )
    }
}

@Composable
private fun RawDROModeSettingItem(
    title: String,
    description: String,
    currentMode: DROMode,
    onModeSelected: (DROMode) -> Unit,
    modifier: Modifier = Modifier
) {
    val levels = listOf(
        DROMode.OFF to stringResource(R.string.settings_dro_off),
        DROMode.DR100 to stringResource(R.string.settings_dro_dr100),
        DROMode.DR200 to stringResource(R.string.settings_dro_dr200),
        DROMode.DR400 to stringResource(R.string.settings_dro_dr400)
    )

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Text(
            text = title,
            color = Color.White,
            fontSize = 16.sp,
            fontWeight = FontWeight.Normal
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = description,
            color = Color.White.copy(alpha = 0.6f),
            fontSize = 13.sp,
            lineHeight = 18.sp
        )
        Spacer(modifier = Modifier.height(12.dp))
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            levels.forEach { (mode, label) ->
                val isSelected = currentMode == mode
                Box(
                    modifier = Modifier
                        .height(36.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(
                            if (isSelected) Color(0xFFFF6B35) else Color.White.copy(alpha = 0.1f)
                        )
                        .clickable { onModeSelected(mode) },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = label,
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                        textAlign = TextAlign.Center,
                        modifier = Modifier
                            .padding(horizontal = 8.dp)
                            .widthIn(min = 48.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun RawSwitchSettingItem(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = description,
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 13.sp,
                lineHeight = 18.sp
            )
        }
        Spacer(modifier = Modifier.width(16.dp))
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = Color.White,
                checkedTrackColor = Color(0xFFFF6B35),
                uncheckedThumbColor = Color.Gray,
                uncheckedTrackColor = Color.White.copy(alpha = 0.2f),
                uncheckedBorderColor = Color.Transparent
            )
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RawColorEngineSelector(
    selectedEngine: RawColorEngine,
    onSelectEngine: (RawColorEngine) -> Unit
) {
    var showSheet by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { showSheet = true }
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.settings_raw_color_engine),
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = rawColorEngineName(selectedEngine),
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 13.sp,
                lineHeight = 18.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        Spacer(modifier = Modifier.width(16.dp))
        Icon(
            imageVector = Icons.Default.ChevronRight,
            contentDescription = null,
            tint = Color.White.copy(alpha = 0.6f)
        )
    }

    if (showSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSheet = false },
            containerColor = Color(0xFF1E1E1E),
            dragHandle = { BottomSheetDefaults.DragHandle(color = Color.White.copy(alpha = 0.2f)) }
        ) {
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
                Text(
                    text = stringResource(R.string.settings_raw_color_engine),
                    color = Color.White,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(16.dp))
                RawColorEngine.entries.forEach { engine ->
                    RawColorEngineItem(
                        name = rawColorEngineName(engine),
                        description = rawColorEngineDescription(engine),
                        isSelected = selectedEngine == engine,
                        onClick = {
                            onSelectEngine(engine)
                            showSheet = false
                        }
                    )
                }
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
}

@Composable
private fun rawColorEngineName(engine: RawColorEngine): String {
    return when (engine) {
        RawColorEngine.AdobeCurve -> stringResource(R.string.settings_raw_color_engine_adobe_curve)
        RawColorEngine.AgX -> stringResource(R.string.settings_raw_color_engine_agx)
        RawColorEngine.DarktableSigmoid -> stringResource(R.string.settings_raw_color_engine_darktable_sigmoid)
        RawColorEngine.DarktableFilmic -> stringResource(R.string.settings_raw_color_engine_darktable_filmic)
        RawColorEngine.Spektrafilm -> stringResource(R.string.settings_raw_color_engine_spectral_film)
    }
}

@Composable
private fun rawColorEngineDescription(engine: RawColorEngine): String {
    return when (engine) {
        RawColorEngine.AdobeCurve -> stringResource(R.string.settings_raw_color_engine_adobe_curve_description)
        RawColorEngine.AgX -> stringResource(R.string.settings_raw_color_engine_agx_description)
        RawColorEngine.DarktableSigmoid -> stringResource(R.string.settings_raw_color_engine_darktable_sigmoid_description)
        RawColorEngine.DarktableFilmic -> stringResource(R.string.settings_raw_color_engine_darktable_filmic_description)
        RawColorEngine.Spektrafilm -> stringResource(R.string.settings_raw_color_engine_spectral_film_description)
    }
}

@Composable
private fun RawColorEngineItem(
    name: String,
    description: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(if (isSelected) Color(0xFFFF6B35).copy(alpha = 0.15f) else Color.Transparent)
            .clickable(onClick = onClick)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = name,
                color = if (isSelected) Color(0xFFFF6B35) else Color.White,
                fontSize = 14.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = description,
                color = Color.White.copy(alpha = 0.55f),
                fontSize = 12.sp,
                lineHeight = 16.sp
            )
        }
        if (isSelected) {
            Spacer(modifier = Modifier.width(12.dp))
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = null,
                tint = Color(0xFFFF6B35),
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RawDcpSelector(
    selectedDcpId: String?,
    availableDcps: List<DcpInfo>,
    showNonAdobeCurveWarning: Boolean = false,
    onSelectDcp: (String?) -> Unit,
    onImportDcp: () -> Unit,
    onDeleteDcp: (DcpInfo) -> Unit
) {
    var showSheet by remember { mutableStateOf(false) }
    var pendingDeleteDcp by remember { mutableStateOf<DcpInfo?>(null) }
    val selectedName = availableDcps.firstOrNull { it.id == selectedDcpId }?.getName()
        ?: stringResource(R.string.none)

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { showSheet = true }
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.raw_dcp_title),
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = selectedName,
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 13.sp,
                lineHeight = 18.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            if (showNonAdobeCurveWarning && selectedDcpId != null) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = stringResource(R.string.raw_dcp_non_adobe_curve_warning),
                    color = Color(0xFFFFC36A).copy(alpha = 0.9f),
                    fontSize = 12.sp,
                    lineHeight = 16.sp
                )
            }
        }

        Spacer(modifier = Modifier.width(16.dp))

        Icon(
            imageVector = Icons.Default.ChevronRight,
            contentDescription = null,
            tint = Color.White.copy(alpha = 0.6f)
        )
    }

    if (showSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSheet = false },
            containerColor = Color(0xFF1E1E1E),
            dragHandle = { BottomSheetDefaults.DragHandle(color = Color.White.copy(alpha = 0.2f)) }
        ) {
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.raw_dcp_title),
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                    TextButton(onClick = {
                        showSheet = false
                        onImportDcp()
                    }) {
                        Text(
                            text = stringResource(R.string.raw_dcp_import),
                            color = Color(0xFFFF6B35),
                            fontSize = 14.sp
                        )
                    }
                }
                if (showNonAdobeCurveWarning) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = stringResource(R.string.raw_dcp_non_adobe_curve_warning),
                        color = Color(0xFFFFC36A).copy(alpha = 0.9f),
                        fontSize = 12.sp,
                        lineHeight = 16.sp
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                androidx.compose.foundation.lazy.LazyColumn(
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(bottom = 32.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    item {
                        DcpItem(
                            name = stringResource(R.string.none),
                            isSelected = selectedDcpId == null,
                            onClick = {
                                onSelectDcp(null)
                                showSheet = false
                            }
                        )
                    }
                    items(availableDcps.size, key = { availableDcps[it].id }) { index ->
                        val dcp = availableDcps[index]
                        DcpItem(
                            name = dcp.getName(),
                            isSelected = selectedDcpId == dcp.id,
                            onClick = {
                                onSelectDcp(dcp.id)
                                showSheet = false
                            },
                            isCustom = !dcp.isBuiltIn,
                            onDelete = { pendingDeleteDcp = dcp }
                        )
                    }
                }
            }
        }
    }

    pendingDeleteDcp?.let { dcp ->
        AlertDialog(
            onDismissRequest = { pendingDeleteDcp = null },
            title = { Text(stringResource(R.string.delete_confirm_title)) },
            text = { Text(stringResource(R.string.delete_dcp_confirm_message, dcp.getName())) },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDeleteDcp(dcp)
                        pendingDeleteDcp = null
                    }
                ) {
                    Text(stringResource(R.string.delete))
                }
            },
            dismissButton = {
                TextButton(onClick = { pendingDeleteDcp = null }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }
}

@Composable
private fun DcpItem(
    name: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    isCustom: Boolean = false,
    onDelete: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(if (isSelected) Color(0xFFFF6B35).copy(alpha = 0.15f) else Color.Transparent)
            .clickable { onClick() }
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = name,
            color = if (isSelected) Color(0xFFFF6B35) else Color.White,
            fontSize = 12.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f)
        )
        if (isSelected) {
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = null,
                tint = Color(0xFFFF6B35),
                modifier = Modifier.size(20.dp)
            )
        }
        if (isCustom && onDelete != null) {
            Spacer(modifier = Modifier.width(8.dp))
            IconButton(
                onClick = onDelete,
                modifier = Modifier.size(36.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = stringResource(R.string.delete),
                    tint = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RawBaselineColorCorrectionSelector(
    selectedLutId: String?,
    availableLuts: List<LutInfo>,
    thumbnail: Bitmap?,
    onSelectLut: (String?) -> Unit,
    title: String = stringResource(R.string.settings_baseline_raw_title),
    onEditRecipe: ((String) -> Unit)? = null,
    onOpenSheet: (() -> Unit)? = null
) {
    var showSheet by remember { mutableStateOf(false) }
    val selectedLut = availableLuts.find { it.id == selectedLutId }
    val selectedName = selectedLut?.getName() ?: stringResource(R.string.none)

    fun openSheet() {
        if (onOpenSheet != null) {
            onOpenSheet()
        } else {
            showSheet = true
        }
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(
            modifier = Modifier
                .weight(1f)
                .clickable { openSheet() }
        ) {
            Text(
                text = title,
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = selectedName,
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 13.sp,
                lineHeight = 18.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        Spacer(modifier = Modifier.width(16.dp))

        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Default.ChevronRight,
                contentDescription = null,
                tint = Color.White.copy(alpha = 0.6f),
                modifier = Modifier.clickable { openSheet() }
            )
        }
    }

    if (showSheet) {
        RawBaselineColorCorrectionBottomSheet(
            selectedLutId = selectedLutId,
            availableLuts = availableLuts,
            thumbnail = thumbnail,
            title = title,
            onSelectLut = onSelectLut,
            onEditRecipe = onEditRecipe,
            onDismiss = { showSheet = false }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RawBaselineColorCorrectionBottomSheet(
    selectedLutId: String?,
    availableLuts: List<LutInfo>,
    thumbnail: Bitmap?,
    title: String = stringResource(R.string.settings_baseline_raw_title),
    containerColor: Color = Color(0xFF1E1E1E),
    onSelectLut: (String?) -> Unit,
    onEditRecipe: ((String) -> Unit)? = null,
    onDismiss: () -> Unit
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = containerColor,
        scrimColor = Color.Transparent,
        dragHandle = { BottomSheetDefaults.DragHandle(color = Color.White.copy(alpha = 0.2f)) }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title,
                    color = Color.White,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
                IconButton(onClick = onDismiss) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = stringResource(R.string.close),
                        tint = Color.White.copy(alpha = 0.8f)
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = stringResource(R.string.settings_baseline_dialog_description),
                color = Color.White.copy(alpha = 0.7f),
                fontSize = 13.sp,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            LutSelector(
                availableLuts = availableLuts,
                currentLutId = selectedLutId,
                thumbnail = thumbnail,
                onLutSelected = { selected ->
                    onSelectLut(selected)
                }
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextButton(
                    onClick = {
                        onSelectLut(null)
                    }
                ) {
                    Text(stringResource(R.string.settings_baseline_clear))
                }
                if (onEditRecipe != null) {
                    TextButton(
                        onClick = {
                            if (selectedLutId != null) {
                                onDismiss()
                                onEditRecipe(selectedLutId)
                            }
                        },
                        enabled = selectedLutId != null
                    ) {
                        Text(stringResource(R.string.settings_baseline_edit_recipe))
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RawSpectralFilmSelector(
    selectedFilm: String?,
    onSelectFilm: (String?) -> Unit
) {
    var showSheet by remember { mutableStateOf(false) }
    val displayFilm = selectedFilm?.let { SpectralFilmUiInfo.getFilmDisplayName(it) }
        ?: stringResource(R.string.none)

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { showSheet = true }
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.settings_negative_film),
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = displayFilm,
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 13.sp,
                lineHeight = 18.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        Spacer(modifier = Modifier.width(16.dp))

        Icon(
            imageVector = Icons.Default.ChevronRight,
            contentDescription = null,
            tint = Color.White.copy(alpha = 0.6f)
        )
    }

    if (showSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSheet = false },
            containerColor = Color(0xFF1E1E1E),
            dragHandle = { BottomSheetDefaults.DragHandle(color = Color.White.copy(alpha = 0.2f)) }
        ) {
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.settings_negative_film),
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                androidx.compose.foundation.lazy.LazyColumn(
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(bottom = 32.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(SpectralFilmUiInfo.availableFilms.size, key = { SpectralFilmUiInfo.availableFilms[it] }) { index ->
                        val film = SpectralFilmUiInfo.availableFilms[index]
                        val isSelected = selectedFilm == film
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(if (isSelected) Color(0xFFFF6B35).copy(alpha = 0.15f) else Color.Transparent)
                                .clickable {
                                    onSelectFilm(film)
                                    showSheet = false
                                }
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = SpectralFilmUiInfo.getFilmDisplayName(film),
                                color = if (isSelected) Color(0xFFFF6B35) else Color.White,
                                fontSize = 14.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f)
                            )
                            if (isSelected) {
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = null,
                                    tint = Color(0xFFFF6B35),
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RawSpectralPrintSelector(
    selectedPrint: String?,
    onSelectPrint: (String?) -> Unit
) {
    var showSheet by remember { mutableStateOf(false) }
    val displayPrint = selectedPrint?.let { SpectralFilmUiInfo.getPrintDisplayName(it) }
        ?: stringResource(R.string.none)

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { showSheet = true }
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.settings_print_paper),
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = displayPrint,
                color = Color.White.copy(alpha = 0.6f),
                fontSize = 13.sp,
                lineHeight = 18.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        Spacer(modifier = Modifier.width(16.dp))

        Icon(
            imageVector = Icons.Default.ChevronRight,
            contentDescription = null,
            tint = Color.White.copy(alpha = 0.6f)
        )
    }

    if (showSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSheet = false },
            containerColor = Color(0xFF1E1E1E),
            dragHandle = { BottomSheetDefaults.DragHandle(color = Color.White.copy(alpha = 0.2f)) }
        ) {
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.settings_print_paper),
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                androidx.compose.foundation.lazy.LazyColumn(
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(bottom = 32.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(SpectralFilmUiInfo.availablePrints.size, key = { SpectralFilmUiInfo.availablePrints[it] }) { index ->
                        val print = SpectralFilmUiInfo.availablePrints[index]
                        val isSelected = selectedPrint == print
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(if (isSelected) Color(0xFFFF6B35).copy(alpha = 0.15f) else Color.Transparent)
                                .clickable {
                                    onSelectPrint(print)
                                    showSheet = false
                                }
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = SpectralFilmUiInfo.getPrintDisplayName(print),
                                color = if (isSelected) Color(0xFFFF6B35) else Color.White,
                                fontSize = 14.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f)
                            )
                            if (isSelected) {
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = null,
                                    tint = Color(0xFFFF6B35),
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
