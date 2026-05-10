package com.hinnka.mycamera.ui.settings

import android.graphics.Bitmap
import android.graphics.Color as AndroidColor
import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DragHandle
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.ScreenRotation
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.hinnka.mycamera.R
import com.hinnka.mycamera.frame.DividerOrientation
import com.hinnka.mycamera.frame.ElementAlignment
import com.hinnka.mycamera.frame.FontWeight as FrameFontWeight
import com.hinnka.mycamera.frame.FrameEditorDraft
import com.hinnka.mycamera.frame.FrameElementDraft
import com.hinnka.mycamera.frame.FramePosition
import com.hinnka.mycamera.frame.LogoType
import com.hinnka.mycamera.frame.TextType
import com.hinnka.mycamera.ui.theme.AccentOrange
import com.hinnka.mycamera.viewmodel.CameraViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import sh.calvin.reorderable.ReorderableItem
import sh.calvin.reorderable.rememberReorderableLazyListState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FrameEditorScreen(
    viewModel: CameraViewModel,
    frameId: String?,
    imageFrame: Boolean,
    onBack: () -> Unit,
    onSaved: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val scope = rememberCoroutineScope()
    val backgroundColor = Color(0xFF121212)

    var initialDraft by remember(frameId, imageFrame) {
        mutableStateOf(viewModel.loadFrameEditorDraft(frameId, imageFrame))
    }
    var draft by remember(frameId, imageFrame) { mutableStateOf(initialDraft) }
    var selectedTab by rememberSaveable(frameId, imageFrame) { mutableIntStateOf(0) }
    var previewPortrait by rememberSaveable(frameId, imageFrame) { mutableStateOf(true) }
    var previewBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isRenderingPreview by remember { mutableStateOf(false) }
    var isSaving by remember { mutableStateOf(false) }
    var showDiscardDialog by remember { mutableStateOf(false) }
    var validationMessage by remember { mutableStateOf<String?>(null) }
    var showAddElementMenu by remember { mutableStateOf(false) }
    var pendingFontElementId by remember { mutableStateOf<String?>(null) }
    var pendingLogoElementId by remember { mutableStateOf<String?>(null) }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        scope.launch {
            val importedPath = withContext(Dispatchers.IO) {
                viewModel.importFrameEditorImage(uri, draft.editableFrameId)
            }
            if (importedPath != null) {
                draft = draft.copy(
                    layout = draft.layout.copy(
                        imagePath = importedPath,
                        imageResName = null
                    )
                )
            }
        }
    }

    val fontPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        val targetId = pendingFontElementId ?: return@rememberLauncherForActivityResult
        pendingFontElementId = null
        uri ?: return@rememberLauncherForActivityResult
        scope.launch {
            val importedPath = withContext(Dispatchers.IO) {
                viewModel.getCustomImportManager().importFont(uri)
            }
            if (importedPath != null) {
                draft = updateTextElementById(draft, targetId) { it.copy(fontFamily = importedPath) }
            }
        }
    }

    val logoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        val targetId = pendingLogoElementId ?: return@rememberLauncherForActivityResult
        pendingLogoElementId = null
        uri ?: return@rememberLauncherForActivityResult
        scope.launch {
            val importedPath = withContext(Dispatchers.IO) {
                viewModel.getCustomImportManager().importLogo(uri)
            }
            if (importedPath != null) {
                draft = updateLogoElementById(draft, targetId) { it.copy(overrideSource = importedPath) }
            }
        }
    }

    LaunchedEffect(frameId, imageFrame) {
        val loaded = viewModel.loadFrameEditorDraft(frameId, imageFrame)
        val legacyProperties = frameId?.let { viewModel.getFrameCustomProperties(it) }.orEmpty()
        val hydratedDraft = loaded.applyLegacyCustomProperties(legacyProperties)
        initialDraft = hydratedDraft
        draft = hydratedDraft
    }

    LaunchedEffect(draft, previewPortrait) {
        isRenderingPreview = true
        delay(150)
        previewBitmap = runCatching {
            viewModel.renderFrameEditorPreview(draft, previewPortrait)
        }.getOrNull()
        isRenderingPreview = false
    }

    val hasChanges = draft != initialDraft
    val canSave = draft.name.trim().isNotEmpty() &&
        (draft.layout.position != FramePosition.IMAGE ||
            !draft.layout.imagePath.isNullOrBlank() ||
            !draft.layout.imageResName.isNullOrBlank())
    val validationNameRequiredMessage = stringResource(R.string.frame_editor_validation_name_required)
    val saveFailedMessage = stringResource(R.string.frame_editor_save_failed)

    fun handleBack() {
        if (hasChanges) {
            showDiscardDialog = true
        } else {
            onBack()
        }
    }

    BackHandler(onBack = ::handleBack)

    ScaffoldWithBottomBar(
        modifier = modifier
            .fillMaxSize()
            .background(backgroundColor),
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = when {
                            frameId == null && imageFrame -> stringResource(R.string.frame_editor_new_image_title)
                            frameId == null -> stringResource(R.string.frame_editor_new_title)
                            else -> stringResource(R.string.frame_editor_edit_title)
                        },
                        color = Color.White
                    )
                },
                navigationIcon = {
                    IconButton(onClick = ::handleBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.back),
                            tint = Color.White
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = backgroundColor
                )
            )
        },
        bottomBar = {
            Surface(
                color = backgroundColor,
                tonalElevation = 8.dp,
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    validationMessage?.let { message ->
                        Text(
                            text = message,
                            color = Color(0xFFFF8A80),
                            fontSize = 12.sp
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        OutlinedButton(
                            onClick = ::handleBack,
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(stringResource(R.string.cancel))
                        }

                        Button(
                            onClick = {
                                val validationErrors = draft.validate()
                                if (!canSave || validationErrors.isNotEmpty()) {
                                    validationMessage = validationErrors.joinToString()
                                        .ifBlank { validationNameRequiredMessage }
                                    return@Button
                                }
                                validationMessage = null
                                isSaving = true
                                scope.launch {
                                    val savedId = withContext(Dispatchers.IO) {
                                        viewModel.saveFrameEditorDraft(draft)
                                    }
                                    isSaving = false
                                    if (savedId != null) {
                                        if (draft.editableFrameId == savedId) {
                                            viewModel.saveFrameCustomProperties(savedId, emptyMap())
                                        }
                                        onSaved(savedId)
                                    } else {
                                        validationMessage = saveFailedMessage
                                    }
                                }
                            },
                            enabled = !isSaving,
                            colors = ButtonDefaults.buttonColors(containerColor = AccentOrange),
                            modifier = Modifier.weight(1f)
                        ) {
                            if (isSaving) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(18.dp),
                                    strokeWidth = 2.dp,
                                    color = Color.White
                                )
                            } else {
                                Text(
                                    text = if (draft.isBuiltInSource) {
                                        stringResource(R.string.frame_editor_save_as_copy)
                                    } else {
                                        stringResource(R.string.save)
                                    }
                                )
                            }
                        }
                    }
                }
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(backgroundColor)
                .padding(innerPadding)
        ) {
            PreviewCard(
                previewBitmap = previewBitmap,
                isRendering = isRenderingPreview,
                portrait = previewPortrait,
                onOrientationChange = { previewPortrait = it },
                modifier = Modifier.padding(horizontal = 16.dp)
            )

            ScrollableTabRow(
                selectedTabIndex = selectedTab,
                containerColor = Color.Transparent,
                contentColor = Color.White,
                edgePadding = 16.dp,
                divider = {},
                indicator = { positions ->
                    if (selectedTab < positions.size) {
                        TabRowDefaults.SecondaryIndicator(
                            modifier = Modifier.tabIndicatorOffset(positions[selectedTab]),
                            color = AccentOrange
                        )
                    }
                }
            ) {
                listOf(
                    stringResource(R.string.frame_editor_tab_basic),
                    stringResource(R.string.frame_editor_tab_elements),
                    stringResource(R.string.frame_editor_tab_rules)
                ).forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(title) }
                    )
                }
            }

            when (selectedTab) {
                0 -> FrameBasicTab(
                    draft = draft,
                    onDraftChange = { draft = it },
                    onPickImage = { imagePicker.launch(arrayOf("image/png", "image/webp", "image/*")) },
                    modifier = Modifier.weight(1f)
                )

                1 -> FrameElementsTab(
                    draft = draft,
                    onDraftChange = { draft = it },
                    showAddMenu = showAddElementMenu,
                    onShowAddMenuChange = { showAddElementMenu = it },
                    onImportFont = { draftId ->
                        pendingFontElementId = draftId
                        fontPicker.launch(arrayOf("font/*", "application/octet-stream"))
                    },
                    onImportLogo = { draftId ->
                        pendingLogoElementId = draftId
                        logoPicker.launch("image/*")
                    },
                    modifier = Modifier.weight(1f)
                )

                else -> FrameRulesTab(modifier = Modifier.weight(1f))
            }
        }
    }

    if (showDiscardDialog) {
        AlertDialog(
            onDismissRequest = { showDiscardDialog = false },
            title = { Text(stringResource(R.string.frame_editor_discard_title)) },
            text = { Text(stringResource(R.string.frame_editor_discard_message)) },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDiscardDialog = false
                        onBack()
                    }
                ) {
                    Text(stringResource(R.string.discard))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDiscardDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }
}

@Composable
private fun ScaffoldWithBottomBar(
    modifier: Modifier = Modifier,
    topBar: @Composable () -> Unit,
    bottomBar: @Composable () -> Unit,
    content: @Composable (PaddingValues) -> Unit
) {
    androidx.compose.material3.Scaffold(
        modifier = modifier,
        topBar = topBar,
        bottomBar = bottomBar,
        containerColor = Color(0xFF121212),
        content = content
    )
}

@Composable
private fun PreviewCard(
    previewBitmap: Bitmap?,
    isRendering: Boolean,
    portrait: Boolean,
    onOrientationChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        color = Color.White.copy(alpha = 0.04f),
        shape = RoundedCornerShape(24.dp),
        modifier = modifier.fillMaxWidth()
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp)
                    .background(Color.White.copy(alpha = 0.06f)),
                contentAlignment = Alignment.Center
            ) {
                when {
                    isRendering -> CircularProgressIndicator(color = AccentOrange)
                    previewBitmap != null -> androidx.compose.foundation.Image(
                        bitmap = previewBitmap.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxSize()
                    )

                    else -> Text(
                        text = stringResource(R.string.frame_editor_preview_unavailable),
                        color = Color.White.copy(alpha = 0.6f)
                    )
                }
            }

            Surface(
                color = Color.Black.copy(alpha = 0.36f),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.align(Alignment.TopEnd)
            ) {
                IconButton(
                    onClick = { onOrientationChange(!portrait) },
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ScreenRotation,
                        contentDescription = if (portrait) {
                            stringResource(R.string.frame_editor_preview_landscape)
                        } else {
                            stringResource(R.string.frame_editor_preview_portrait)
                        },
                        tint = if (portrait) AccentOrange else Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun FrameBasicTab(
    draft: FrameEditorDraft,
    onDraftChange: (FrameEditorDraft) -> Unit,
    onPickImage: () -> Unit,
    modifier: Modifier = Modifier
) {
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            SectionCard(title = stringResource(R.string.frame_editor_section_identity)) {
                TextFieldSection(
                    label = stringResource(R.string.name),
                    value = draft.name,
                    onValueChange = { onDraftChange(draft.copy(name = it)) }
                )
                DropdownSelectionField(
                    label = stringResource(R.string.frame_editor_layout_position),
                    currentLabel = framePositionLabel(draft.layout.position),
                    options = FramePosition.entries,
                    optionLabel = { framePositionLabel(it) },
                    onSelected = { position ->
                        onDraftChange(draft.copy(layout = draft.layout.copy(position = position)))
                    }
                )
            }
        }

        item {
            SectionCard(title = stringResource(R.string.frame_editor_section_layout)) {
                if (draft.layout.position != FramePosition.IMAGE) {
                    IntField(
                        label = stringResource(R.string.frame_editor_layout_height),
                        value = draft.layout.heightDp,
                        onValueChange = {
                            onDraftChange(draft.copy(layout = draft.layout.copy(heightDp = it.coerceAtLeast(0))))
                        }
                    )
                    IntField(
                        label = stringResource(R.string.frame_editor_layout_padding),
                        value = draft.layout.paddingDp,
                        onValueChange = {
                            onDraftChange(draft.copy(layout = draft.layout.copy(paddingDp = it.coerceAtLeast(0))))
                        }
                    )
                    ColorField(
                        label = stringResource(R.string.frame_editor_layout_background),
                        value = draft.layout.backgroundColor,
                        onValueChange = {
                            onDraftChange(draft.copy(layout = draft.layout.copy(backgroundColor = it)))
                        }
                    )
                    ColorField(
                        label = stringResource(R.string.frame_editor_layout_border_color),
                        value = draft.layout.borderColor,
                        onValueChange = {
                            onDraftChange(draft.copy(layout = draft.layout.copy(borderColor = it)))
                        }
                    )
                    IntField(
                        label = stringResource(R.string.frame_editor_layout_line_spacing),
                        value = draft.layout.lineSpacingDp,
                        onValueChange = {
                            onDraftChange(draft.copy(layout = draft.layout.copy(lineSpacingDp = it.coerceAtLeast(0))))
                        }
                    )
                    if (draft.layout.position == FramePosition.BORDER) {
                        IntField(
                            label = stringResource(R.string.frame_editor_layout_border_width),
                            value = draft.layout.borderWidthDp,
                            onValueChange = {
                                onDraftChange(draft.copy(layout = draft.layout.copy(borderWidthDp = it.coerceAtLeast(0))))
                            }
                        )
                    }
                } else {
                    Text(
                        text = stringResource(R.string.frame_editor_image_hint),
                        color = Color.White.copy(alpha = 0.65f),
                        fontSize = 13.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = onPickImage,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(
                            imageVector = Icons.Default.PhotoLibrary,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.frame_editor_pick_image))
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = when {
                            !draft.layout.imagePath.isNullOrBlank() -> {
                                val imagePath = draft.layout.imagePath.orEmpty()
                                stringResource(R.string.frame_editor_image_source_path, imagePath.substringAfterLast('/'))
                            }
                            !draft.layout.imageResName.isNullOrBlank() -> {
                                val imageResName = draft.layout.imageResName.orEmpty()
                                stringResource(R.string.frame_editor_image_source_builtin, imageResName)
                            }
                            else -> stringResource(R.string.frame_editor_image_source_missing)
                        },
                        color = Color.White.copy(alpha = 0.8f),
                        fontSize = 13.sp
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FrameElementsTab(
    draft: FrameEditorDraft,
    onDraftChange: (FrameEditorDraft) -> Unit,
    showAddMenu: Boolean,
    onShowAddMenuChange: (Boolean) -> Unit,
    onImportFont: (String) -> Unit,
    onImportLogo: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    if (draft.layout.position == FramePosition.IMAGE) {
        Box(
            modifier = modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            SectionCard(title = stringResource(R.string.frame_editor_elements_disabled_title)) {
                Text(
                    text = stringResource(R.string.frame_editor_elements_disabled_message),
                    color = Color.White.copy(alpha = 0.7f)
                )
            }
        }
        return
    }

    val lazyListState = rememberLazyListState()
    var editingElementId by remember(draft.sourceFrameId, draft.editableFrameId) {
        mutableStateOf<String?>(null)
    }
    val reorderableState = rememberReorderableLazyListState(lazyListState) { from, to ->
        val fromId = from.key as? String ?: return@rememberReorderableLazyListState
        val toId = to.key as? String ?: return@rememberReorderableLazyListState
        val fromIndex = draft.elements.indexOfFirst { it.draftId == fromId }
        val toIndex = draft.elements.indexOfFirst { it.draftId == toId }
        if (fromIndex == -1 || toIndex == -1) return@rememberReorderableLazyListState

        val updated = draft.elements.toMutableList().apply {
            add(toIndex, removeAt(fromIndex))
        }
        onDraftChange(draft.copy(elements = updated))
    }

    val selectedElement = draft.elements.firstOrNull { it.draftId == draft.effectiveSelectedElementId }
    val editingElement = draft.elements.firstOrNull { it.draftId == editingElementId }

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        state = lazyListState,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            SectionCard(title = stringResource(R.string.frame_editor_section_elements)) {
                Box {
                    OutlinedButton(
                        onClick = { onShowAddMenuChange(true) },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(imageVector = Icons.Default.Add, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.frame_editor_add_element))
                    }
                    DropdownMenu(
                        expanded = showAddMenu,
                        onDismissRequest = { onShowAddMenuChange(false) }
                    ) {
                        FrameElementType.entries.forEach { type ->
                            DropdownMenuItem(
                                text = { Text(frameElementTypeLabel(type)) },
                                onClick = {
                                    val newElement = createDraftElement(type)
                                    onDraftChange(
                                        draft.copy(
                                            elements = draft.elements + newElement,
                                            selectedElementId = newElement.draftId
                                        )
                                    )
                                    editingElementId = newElement.draftId
                                    onShowAddMenuChange(false)
                                }
                            )
                        }
                    }
                }
            }
        }

        items(draft.elements, key = { it.draftId }) { element ->
            ReorderableItem(reorderableState, key = element.draftId) { isDragging ->
                ElementListItem(
                    element = element,
                    selected = element.draftId == draft.effectiveSelectedElementId,
                    dragging = isDragging,
                    onSelect = {
                        onDraftChange(draft.withSelectedElement(element.draftId))
                        editingElementId = element.draftId
                    },
                    onEdit = {
                        onDraftChange(draft.withSelectedElement(element.draftId))
                        editingElementId = element.draftId
                    },
                    onDuplicate = {
                        val duplicated = duplicateElement(element)
                        onDraftChange(
                            draft.copy(
                                elements = draft.elements + duplicated,
                                selectedElementId = duplicated.draftId
                            )
                        )
                        editingElementId = duplicated.draftId
                    },
                    onDelete = {
                        val updated = draft.elements.filterNot { it.draftId == element.draftId }
                        if (editingElementId == element.draftId) {
                            editingElementId = null
                        }
                        onDraftChange(
                            draft.copy(
                                elements = updated,
                                selectedElementId = updated.firstOrNull()?.draftId
                            )
                        )
                    },
                    dragModifier = Modifier.draggableHandle()
                )
            }
        }

    }

    LaunchedEffect(editingElementId, draft.elements) {
        if (editingElementId != null && draft.elements.none { it.draftId == editingElementId }) {
            editingElementId = null
        }
    }

    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    if (editingElement != null) {
        ModalBottomSheet(
            onDismissRequest = { editingElementId = null },
            containerColor = Color(0xFF1A1A1A),
            contentColor = Color.White,
            sheetState = sheetState
        ) {
            Box(modifier = Modifier.height(400.dp)) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                        .padding(bottom = 24.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = elementTypeLabel(editingElement),
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = elementSummary(editingElement),
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 12.sp
                    )
                    ElementEditor(
                        element = editingElement,
                        onElementChange = { updated ->
                            onDraftChange(updateElement(draft, updated))
                        },
                        onImportFont = onImportFont,
                        onImportLogo = onImportLogo
                    )
                }
            }
        }
    }
}

@Composable
private fun FrameRulesTab(modifier: Modifier = Modifier) {
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            SectionCard(title = stringResource(R.string.frame_editor_rules_title_render)) {
                RuleLine(stringResource(R.string.frame_editor_rule_line_global))
                RuleLine(stringResource(R.string.frame_editor_rule_vertical_divider))
                RuleLine(stringResource(R.string.frame_editor_rule_image_mode))
            }
        }
        item {
            SectionCard(title = stringResource(R.string.frame_editor_rules_title_fonts)) {
                RuleLine(stringResource(R.string.frame_editor_rule_font_default))
                RuleLine(stringResource(R.string.frame_editor_rule_font_dsdigib))
                RuleLine(stringResource(R.string.frame_editor_rule_font_slackside))
            }
        }
    }
}

@Composable
private fun ElementListItem(
    element: FrameElementDraft,
    selected: Boolean,
    dragging: Boolean,
    onSelect: () -> Unit,
    onEdit: () -> Unit,
    onDuplicate: () -> Unit,
    onDelete: () -> Unit,
    dragModifier: Modifier,
) {
    val borderColor = if (selected) AccentOrange else Color.White.copy(alpha = 0.15f)
    val background = when {
        dragging -> Color.White.copy(alpha = 0.12f)
        selected -> Color.White.copy(alpha = 0.08f)
        else -> Color.White.copy(alpha = 0.04f)
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(background)
            .border(1.dp, borderColor, RoundedCornerShape(16.dp))
            .clickable(onClick = onSelect)
            .padding(horizontal = 10.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Default.DragHandle,
            contentDescription = null,
            tint = Color.White.copy(alpha = 0.5f),
            modifier = dragModifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = elementTypeLabel(element),
                color = Color.White,
                fontWeight = FontWeight.Medium,
                fontSize = 14.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = elementSummary(element),
                color = Color.White.copy(alpha = 0.65f),
                fontSize = 11.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
        IconButton(
            onClick = onEdit,
            modifier = Modifier.size(32.dp)
        ) {
            Icon(
                Icons.Default.Edit,
                contentDescription = null,
                tint = Color.White.copy(alpha = 0.7f),
                modifier = Modifier.size(16.dp)
            )
        }
        IconButton(
            onClick = onDuplicate,
            modifier = Modifier.size(32.dp)
        ) {
            Icon(
                Icons.Default.ContentCopy,
                contentDescription = null,
                tint = Color.White.copy(alpha = 0.7f),
                modifier = Modifier.size(16.dp)
            )
        }
        IconButton(
            onClick = onDelete,
            modifier = Modifier.size(32.dp)
        ) {
            Icon(
                Icons.Default.Delete,
                contentDescription = null,
                tint = Color.White.copy(alpha = 0.7f),
                modifier = Modifier.size(16.dp)
            )
        }
    }
}

@Composable
private fun ElementEditor(
    element: FrameElementDraft,
    onElementChange: (FrameElementDraft) -> Unit,
    onImportFont: (String) -> Unit,
    onImportLogo: (String) -> Unit
) {
    when (element) {
        is FrameElementDraft.Text -> {
            val isCustomText = element.textType == TextType.CUSTOM
            val supportsFormat = when (element.textType) {
                TextType.DATE, TextType.TIME, TextType.DATETIME -> true
                else -> false
            }
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_text_type),
                currentLabel = textTypeLabel(element.textType),
                options = TextType.entries,
                optionLabel = { textTypeLabel(it) },
                onSelected = { onElementChange(element.copy(textType = it)) }
            )
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_alignment),
                currentLabel = alignmentLabel(element.alignment),
                options = ElementAlignment.entries,
                optionLabel = { alignmentLabel(it) },
                onSelected = { onElementChange(element.copy(alignment = it)) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_line),
                value = element.line,
                allowNegative = true,
                onValueChange = { onElementChange(element.copy(line = it)) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_text_size),
                value = element.fontSizeSp,
                onValueChange = { onElementChange(element.copy(fontSizeSp = it.coerceAtLeast(0))) }
            )
            ColorField(
                label = stringResource(R.string.frame_editor_text_color),
                value = element.color,
                onValueChange = { onElementChange(element.copy(color = it)) }
            )
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_font_weight),
                currentLabel = fontWeightLabel(element.fontWeight),
                options = FrameFontWeight.entries,
                optionLabel = { fontWeightLabel(it) },
                onSelected = { onElementChange(element.copy(fontWeight = it)) }
            )
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_font_family),
                currentLabel = fontFamilyLabel(element.fontFamily),
                options = frameFontOptions(),
                optionLabel = { it.second },
                onSelected = { onElementChange(element.copy(fontFamily = it.first)) }
            )
            OutlinedButton(
                onClick = { onImportFont(element.draftId) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(imageVector = Icons.Default.Add, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.custom_import))
            }
            val customFontPath = element.fontFamily?.takeIf { it.startsWith("/") }
            if (customFontPath != null) {
                Text(
                    text = customFontPath.substringAfterLast('/'),
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 12.sp
                )
                TextButton(
                    onClick = { onElementChange(element.copy(fontFamily = null)) }
                ) {
                    Text(stringResource(R.string.default_text))
                }
            }
            if (isCustomText) {
                TextFieldSection(
                    label = stringResource(R.string.frame_editor_text_custom_value),
                    value = element.overrideText.orEmpty(),
                    onValueChange = { onElementChange(element.copy(overrideText = it.ifBlank { null })) }
                )
            } else {
                TextFieldSection(
                    label = stringResource(R.string.frame_editor_text_override),
                    value = element.overrideText.orEmpty(),
                    onValueChange = { onElementChange(element.copy(overrideText = it.ifBlank { null })) }
                )
            }
            if (supportsFormat) {
                TextFieldSection(
                    label = stringResource(R.string.frame_editor_text_format),
                    value = element.format.orEmpty(),
                    onValueChange = { onElementChange(element.copy(format = it.ifBlank { null })) }
                )
            }
            TextFieldSection(
                label = stringResource(R.string.frame_editor_text_prefix),
                value = element.prefix.orEmpty(),
                onValueChange = { onElementChange(element.copy(prefix = it.ifBlank { null })) }
            )
            TextFieldSection(
                label = stringResource(R.string.frame_editor_text_suffix),
                value = element.suffix.orEmpty(),
                onValueChange = { onElementChange(element.copy(suffix = it.ifBlank { null })) }
            )
        }

        is FrameElementDraft.Logo -> {
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_logo_type),
                currentLabel = logoTypeLabel(element.logoType),
                options = LogoType.entries,
                optionLabel = { logoTypeLabel(it) },
                onSelected = { onElementChange(element.copy(logoType = it)) }
            )
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_logo_source),
                currentLabel = logoSourceLabel(element.overrideSource),
                options = logoSourceOptions(),
                optionLabel = { it.second },
                onSelected = { onElementChange(element.copy(overrideSource = it.first)) }
            )
            OutlinedButton(
                onClick = { onImportLogo(element.draftId) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(imageVector = Icons.Default.Image, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.custom_import))
            }
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_alignment),
                currentLabel = alignmentLabel(element.alignment),
                options = ElementAlignment.entries,
                optionLabel = { alignmentLabel(it) },
                onSelected = { onElementChange(element.copy(alignment = it)) }
            )
            val customLogoSource = element.overrideSource?.takeIf {
                it.startsWith("/") || it.startsWith("content://")
            }
            if (customLogoSource != null) {
                Text(
                    text = customLogoSource.substringAfterLast('/'),
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 12.sp
                )
                TextButton(
                    onClick = { onElementChange(element.copy(overrideSource = null)) }
                ) {
                    Text(stringResource(R.string.default_text))
                }
            }
            IntField(
                label = stringResource(R.string.frame_editor_line),
                value = element.line,
                allowNegative = true,
                onValueChange = { onElementChange(element.copy(line = it)) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_logo_size),
                value = element.sizeDp,
                onValueChange = { onElementChange(element.copy(sizeDp = it.coerceAtLeast(0))) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_logo_max_width),
                value = element.maxWidth,
                onValueChange = { onElementChange(element.copy(maxWidth = it.coerceAtLeast(0))) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_margin),
                value = element.marginDp,
                onValueChange = { onElementChange(element.copy(marginDp = it.coerceAtLeast(0))) }
            )
            SwitchRow(
                label = stringResource(R.string.frame_editor_logo_light),
                checked = element.light,
                onCheckedChange = { onElementChange(element.copy(light = it)) }
            )
        }

        is FrameElementDraft.Divider -> {
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_divider_orientation),
                currentLabel = dividerOrientationLabel(element.orientation),
                options = DividerOrientation.entries,
                optionLabel = { dividerOrientationLabel(it) },
                onSelected = { onElementChange(element.copy(orientation = it)) }
            )
            DropdownSelectionField(
                label = stringResource(R.string.frame_editor_alignment),
                currentLabel = alignmentLabel(element.alignment),
                options = ElementAlignment.entries,
                optionLabel = { alignmentLabel(it) },
                onSelected = { onElementChange(element.copy(alignment = it)) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_line),
                value = element.line,
                allowNegative = true,
                onValueChange = { onElementChange(element.copy(line = it)) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_divider_length),
                value = element.lengthDp,
                onValueChange = { onElementChange(element.copy(lengthDp = it.coerceAtLeast(0))) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_divider_thickness),
                value = element.thicknessDp,
                onValueChange = { onElementChange(element.copy(thicknessDp = it.coerceAtLeast(0))) }
            )
            ColorField(
                label = stringResource(R.string.frame_editor_divider_color),
                value = element.color,
                onValueChange = { onElementChange(element.copy(color = it)) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_margin),
                value = element.marginDp,
                onValueChange = { onElementChange(element.copy(marginDp = it.coerceAtLeast(0))) }
            )
        }

        is FrameElementDraft.Spacer -> {
            IntField(
                label = stringResource(R.string.frame_editor_line),
                value = element.line,
                allowNegative = true,
                onValueChange = { onElementChange(element.copy(line = it)) }
            )
            IntField(
                label = stringResource(R.string.frame_editor_spacer_width),
                value = element.widthDp,
                onValueChange = { onElementChange(element.copy(widthDp = it.coerceAtLeast(0))) }
            )
        }
    }
}

@Composable
private fun SectionCard(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Surface(
        color = Color.White.copy(alpha = 0.04f),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            content = {
                Text(
                    text = title,
                    color = Color.White,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Medium
                )
                content()
            }
        )
    }
}

@Composable
private fun TextFieldSection(
    label: String,
    value: String,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun IntField(
    label: String,
    value: Int,
    onValueChange: (Int) -> Unit,
    allowNegative: Boolean = false
) {
    var text by remember(value) { mutableStateOf(value.toString()) }
    OutlinedTextField(
        value = text,
        onValueChange = {
            text = it
            it.toIntOrNull()?.let { number ->
                if (allowNegative || number >= 0) {
                    onValueChange(number)
                }
            }
        },
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun ColorField(
    label: String,
    value: Int,
    onValueChange: (Int) -> Unit
) {
    var text by remember(value) { mutableStateOf(colorToHex(value)) }
    OutlinedTextField(
        value = text,
        onValueChange = {
            text = it
            parseColorOrNull(it)?.let(onValueChange)
        },
        label = { Text(label) },
        singleLine = true,
        trailingIcon = {
            Box(
                modifier = Modifier
                    .size(18.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(Color(value))
            )
        },
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun SwitchRow(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, color = Color.White)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun <T> DropdownSelectionField(
    label: String,
    currentLabel: String,
    options: List<T>,
    optionLabel: @Composable (T) -> String,
    onSelected: (T) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = label,
            color = Color.White.copy(alpha = 0.7f),
            fontSize = 12.sp
        )
        Box {
            OutlinedButton(
                onClick = { expanded = true },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = currentLabel,
                    modifier = Modifier.fillMaxWidth(),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                options.forEach { option ->
                    DropdownMenuItem(
                        text = { Text(optionLabel(option)) },
                        onClick = {
                            expanded = false
                            onSelected(option)
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun RuleLine(text: String) {
    Text(
        text = text,
        color = Color.White.copy(alpha = 0.72f),
        fontSize = 13.sp
    )
}

private enum class FrameElementType {
    TEXT,
    LOGO,
    DIVIDER,
    SPACER
}

private fun createDraftElement(type: FrameElementType): FrameElementDraft {
    return when (type) {
        FrameElementType.TEXT -> FrameElementDraft.Text()
        FrameElementType.LOGO -> FrameElementDraft.Logo()
        FrameElementType.DIVIDER -> FrameElementDraft.Divider()
        FrameElementType.SPACER -> FrameElementDraft.Spacer()
    }
}

private fun duplicateElement(element: FrameElementDraft): FrameElementDraft {
    return when (element) {
        is FrameElementDraft.Text -> element.copy(draftId = java.util.UUID.randomUUID().toString())
        is FrameElementDraft.Logo -> element.copy(draftId = java.util.UUID.randomUUID().toString())
        is FrameElementDraft.Divider -> element.copy(draftId = java.util.UUID.randomUUID().toString())
        is FrameElementDraft.Spacer -> element.copy(draftId = java.util.UUID.randomUUID().toString())
    }
}

private fun updateElement(draft: FrameEditorDraft, updated: FrameElementDraft): FrameEditorDraft {
    return draft.copy(
        elements = draft.elements.map { if (it.draftId == updated.draftId) updated else it },
        selectedElementId = updated.draftId
    )
}

private fun updateTextElementById(
    draft: FrameEditorDraft,
    elementId: String,
    updater: (FrameElementDraft.Text) -> FrameElementDraft.Text
): FrameEditorDraft {
    return draft.copy(
        elements = draft.elements.map { element ->
            if (element is FrameElementDraft.Text && element.draftId == elementId) {
                updater(element)
            } else {
                element
            }
        },
        selectedElementId = elementId
    )
}

private fun updateLogoElementById(
    draft: FrameEditorDraft,
    elementId: String,
    updater: (FrameElementDraft.Logo) -> FrameElementDraft.Logo
): FrameEditorDraft {
    return draft.copy(
        elements = draft.elements.map { element ->
            if (element is FrameElementDraft.Logo && element.draftId == elementId) {
                updater(element)
            } else {
                element
            }
        },
        selectedElementId = elementId
    )
}

private fun FrameEditorDraft.applyLegacyCustomProperties(properties: Map<String, String>): FrameEditorDraft {
    if (properties.isEmpty()) return this

    val updatedElements = elements.map { element ->
        when (element) {
            is FrameElementDraft.Text -> {
                val overrideText = properties[element.textType.name]
                val fontFamily = if (element.textType == TextType.DEVICE_MODEL) {
                    properties["DEVICE_MODEL_FONT"]?.let { legacyFont ->
                        when (legacyFont) {
                            "Default" -> null
                            "SlacksideOne" -> "SlacksideOne.ttf"
                            else -> legacyFont
                        }
                    } ?: element.fontFamily
                } else {
                    element.fontFamily
                }
                element.copy(
                    overrideText = overrideText ?: element.overrideText,
                    fontFamily = fontFamily
                )
            }

            is FrameElementDraft.Logo -> {
                if (element.logoType == LogoType.BRAND) {
                    element.copy(overrideSource = properties["LOGO"] ?: element.overrideSource)
                } else {
                    element
                }
            }

            else -> element
        }
    }

    return copy(elements = updatedElements)
}

private fun colorToHex(color: Int): String {
    return if ((color ushr 24) == 0xFF) {
        String.format("#%06X", color and 0xFFFFFF)
    } else {
        String.format("#%08X", color)
    }
}

private fun parseColorOrNull(value: String): Int? {
    return runCatching { AndroidColor.parseColor(value) }.getOrNull()
}

@Composable
private fun frameElementTypeLabel(type: FrameElementType): String = when (type) {
    FrameElementType.TEXT -> stringResource(R.string.frame_editor_element_text)
    FrameElementType.LOGO -> stringResource(R.string.frame_editor_element_logo)
    FrameElementType.DIVIDER -> stringResource(R.string.frame_editor_element_divider)
    FrameElementType.SPACER -> stringResource(R.string.frame_editor_element_spacer)
}

@Composable
private fun framePositionLabel(position: FramePosition): String = when (position) {
    FramePosition.TOP -> stringResource(R.string.frame_editor_position_top)
    FramePosition.BOTTOM -> stringResource(R.string.frame_editor_position_bottom)
    FramePosition.BOTH -> stringResource(R.string.frame_editor_position_both)
    FramePosition.OVERLAY -> stringResource(R.string.frame_editor_position_overlay)
    FramePosition.BORDER -> stringResource(R.string.frame_editor_position_border)
    FramePosition.IMAGE -> stringResource(R.string.frame_editor_position_image)
}

@Composable
private fun textTypeLabel(type: TextType): String = when (type) {
    TextType.DEVICE_MODEL -> stringResource(R.string.frame_editor_text_type_device_model)
    TextType.BRAND -> stringResource(R.string.frame_editor_text_type_brand)
    TextType.DATE -> stringResource(R.string.frame_editor_text_type_date)
    TextType.TIME -> stringResource(R.string.frame_editor_text_type_time)
    TextType.DATETIME -> stringResource(R.string.frame_editor_text_type_datetime)
    TextType.LOCATION -> stringResource(R.string.frame_editor_text_type_location)
    TextType.ISO -> stringResource(R.string.frame_editor_text_type_iso)
    TextType.SHUTTER_SPEED -> stringResource(R.string.frame_editor_text_type_shutter_speed)
    TextType.FOCAL_LENGTH -> stringResource(R.string.frame_editor_text_type_focal_length)
    TextType.FOCAL_LENGTH_35MM -> stringResource(R.string.frame_editor_text_type_focal_length_35mm)
    TextType.APERTURE -> stringResource(R.string.frame_editor_text_type_aperture)
    TextType.RESOLUTION -> stringResource(R.string.frame_editor_text_type_resolution)
    TextType.CUSTOM -> stringResource(R.string.frame_editor_text_type_custom)
    TextType.APP_NAME -> stringResource(R.string.frame_editor_text_type_app_name)
    TextType.FILTER_NAME -> stringResource(R.string.frame_editor_text_type_filter_name)
}

@Composable
private fun logoTypeLabel(type: LogoType): String = when (type) {
    LogoType.BRAND -> stringResource(R.string.frame_editor_logo_type_brand)
    LogoType.APP -> stringResource(R.string.frame_editor_logo_type_app)
}

@Composable
private fun alignmentLabel(alignment: ElementAlignment): String = when (alignment) {
    ElementAlignment.START -> stringResource(R.string.frame_editor_alignment_start)
    ElementAlignment.CENTER -> stringResource(R.string.frame_editor_alignment_center)
    ElementAlignment.END -> stringResource(R.string.frame_editor_alignment_end)
}

@Composable
private fun fontWeightLabel(weight: FrameFontWeight): String = when (weight) {
    FrameFontWeight.NORMAL -> stringResource(R.string.frame_editor_font_weight_normal)
    FrameFontWeight.MEDIUM -> stringResource(R.string.frame_editor_font_weight_medium)
    FrameFontWeight.BOLD -> stringResource(R.string.frame_editor_font_weight_bold)
}

@Composable
private fun dividerOrientationLabel(orientation: DividerOrientation): String = when (orientation) {
    DividerOrientation.HORIZONTAL -> stringResource(R.string.frame_editor_divider_orientation_horizontal)
    DividerOrientation.VERTICAL -> stringResource(R.string.frame_editor_divider_orientation_vertical)
}

@Composable
private fun frameFontOptions(): List<Pair<String?, String>> = listOf(
    null to stringResource(R.string.default_text),
    "DS-DIGIB.TTF" to "DS-DIGIB.TTF",
    "SlacksideOne.ttf" to "SlacksideOne.ttf"
)

@Composable
private fun fontFamilyLabel(fontFamily: String?): String {
    return frameFontOptions().firstOrNull { it.first == fontFamily }?.second
        ?: fontFamily?.substringAfterLast('/')
        ?: stringResource(R.string.default_text)
}

@Composable
private fun logoSourceOptions(): List<Pair<String?, String>> = listOf(
    null to stringResource(R.string.default_text),
    "none" to stringResource(R.string.none),
    "apple" to "Apple",
    "samsung" to "Samsung",
    "xiaomi" to "Xiaomi",
    "huawei" to "Huawei",
    "honor" to "Honor",
    "oppo" to "OPPO",
    "vivo" to "Vivo",
    "sony" to "Sony",
    "canon" to "Canon",
    "nikon" to "Nikon",
    "fujifilm" to "Fujifilm",
    "leica" to "Leica",
    "hasselblad" to "Hasselblad",
    "hasselblad_l" to "Hasselblad L",
    "dji" to "DJI",
    "panasonic" to "Panasonic",
    "olympus" to "Olympus",
    "pentax" to "Pentax",
    "ricoh" to "Ricoh"
)

@Composable
private fun logoSourceLabel(source: String?): String {
    return logoSourceOptions().firstOrNull { it.first == source }?.second
        ?: source?.substringAfterLast('/')
        ?: stringResource(R.string.default_text)
}

@Composable
private fun elementTypeLabel(element: FrameElementDraft): String = when (element) {
    is FrameElementDraft.Text -> stringResource(R.string.frame_editor_element_text)
    is FrameElementDraft.Logo -> stringResource(R.string.frame_editor_element_logo)
    is FrameElementDraft.Divider -> stringResource(R.string.frame_editor_element_divider)
    is FrameElementDraft.Spacer -> stringResource(R.string.frame_editor_element_spacer)
}

@Composable
private fun elementSummary(element: FrameElementDraft): String = when (element) {
    is FrameElementDraft.Text -> stringResource(
        R.string.frame_editor_summary_text,
        textTypeLabel(element.textType),
        alignmentLabel(element.alignment),
        element.line
    )
    is FrameElementDraft.Logo -> stringResource(
        R.string.frame_editor_summary_logo,
        logoTypeLabel(element.logoType),
        alignmentLabel(element.alignment),
        element.sizeDp
    )
    is FrameElementDraft.Divider -> stringResource(
        R.string.frame_editor_summary_divider,
        dividerOrientationLabel(element.orientation),
        alignmentLabel(element.alignment),
        element.lengthDp
    )
    is FrameElementDraft.Spacer -> stringResource(
        R.string.frame_editor_summary_spacer,
        element.line,
        element.widthDp
    )
}
