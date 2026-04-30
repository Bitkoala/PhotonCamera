package com.hinnka.mycamera.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import com.hinnka.mycamera.R
import com.hinnka.mycamera.raw.DcpInfo

@Composable
fun RawEditPanel(
    selectedDcpId: String?,
    availableDcps: List<DcpInfo>,
    rawNlmNoiseFactor: Float,
    rawExposureCompensation: Float,
    rawBlackPointCorrection: Float,
    rawWhitePointCorrection: Float,
    onSelectDcp: (String?) -> Unit,
    onImportDcp: () -> Unit,
    onDeleteDcp: (DcpInfo) -> Unit,
    onRawNlmNoiseFactorChange: (Float) -> Unit,
    onRawExposureCompensationChange: (Float) -> Unit,
    onRawBlackPointCorrectionChange: (Float) -> Unit,
    onRawWhitePointCorrectionChange: (Float) -> Unit,
    onAdjustmentStart: () -> Unit,
    onAdjustmentEnd: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        RawDcpSelector(
            selectedDcpId = selectedDcpId,
            availableDcps = availableDcps,
            onSelectDcp = onSelectDcp,
            onImportDcp = onImportDcp,
            onDeleteDcp = onDeleteDcp
        )
        Spacer(modifier = Modifier.height(16.dp))
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
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RawDcpSelector(
    selectedDcpId: String?,
    availableDcps: List<DcpInfo>,
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
