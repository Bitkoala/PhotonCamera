package com.hinnka.mycamera.ui.gallery

import android.app.Activity
import android.graphics.Bitmap
import android.widget.Toast
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.staggeredgrid.LazyVerticalStaggeredGrid
import androidx.compose.foundation.lazy.staggeredgrid.LazyStaggeredGridState
import androidx.compose.foundation.lazy.staggeredgrid.StaggeredGridCells
import androidx.compose.foundation.lazy.staggeredgrid.items
import androidx.compose.foundation.lazy.staggeredgrid.StaggeredGridItemSpan
import androidx.compose.foundation.lazy.staggeredgrid.rememberLazyStaggeredGridState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.BurstMode
import androidx.compose.material.icons.filled.Output
import androidx.compose.material3.*
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import com.hinnka.mycamera.R
import com.hinnka.mycamera.gallery.MediaData
import com.hinnka.mycamera.ui.camera.autoRotate
import com.hinnka.mycamera.ui.theme.AccentOrange
import com.hinnka.mycamera.utils.OrientationObserver
import com.hinnka.mycamera.viewmodel.GalleryTab
import com.hinnka.mycamera.viewmodel.GalleryViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * 相册浏览界面
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GalleryScreen(
    viewModel: GalleryViewModel,
    onBack: () -> Unit,
    onPhotoClick: (GalleryTab, Int) -> Unit,
    modifier: Modifier = Modifier
) {
    val photos by viewModel.currentPhotos.collectAsState()
    val context = LocalContext.current
    val isLoading by viewModel.isLoading.collectAsState()
    val isSystemLoadingMore by viewModel.isSystemLoadingMore.collectAsState()
    val isPhotonLoadingMore by viewModel.isPhotonLoadingMore.collectAsState()
    val isSharing by viewModel.isSharing.collectAsState()
    val isSelectionMode = viewModel.isSelectionMode
    val selectedPhotos = viewModel.selectedPhotos
    val selectedTab = viewModel.selectedTab
    val hasPermission = viewModel.hasGalleryPermission
    val scope = rememberCoroutineScope()
    val isFastScrolling = remember { mutableStateOf(false) }
    val photonGridState = rememberLazyStaggeredGridState()
    val systemGridState = rememberLazyStaggeredGridState()
    val currentGridState = when (selectedTab) {
        GalleryTab.PHOTON -> photonGridState
        GalleryTab.SYSTEM -> systemGridState
    }
    var shouldRenderGrid by rememberSaveable { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isNotEmpty()) {
            viewModel.importPhotos(uris)
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        if (grants.values.all { it }) {
            viewModel.checkGalleryPermission()
        }
    }

    // Activity Result Launcher for batch delete confirmation
    val batchDeleteLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            // User confirmed deletion, delete internal photos
            viewModel.deleteBatchPhotosAfterConfirmation()
        } else {
            // User cancelled deletion
            viewModel.clearBatchDeleteRequest()
        }
    }

    // Monitor batchDeletePendingIntent and launch system delete dialog
    LaunchedEffect(viewModel.batchDeletePendingIntent) {
        viewModel.batchDeletePendingIntent?.let { pendingIntent ->
            try {
                batchDeleteLauncher.launch(
                    IntentSenderRequest.Builder(pendingIntent.intentSender).build()
                )
            } catch (e: Exception) {
                // Failed to launch, clear the request
                viewModel.clearBatchDeleteRequest()
            }
        }
    }

    var showDeleteDialog by remember { mutableStateOf(false) }

    // 首次进入时先让导航过渡完成，再挂载重型网格，避免首屏动画和列表构建抢主线程。
    LaunchedEffect(Unit) {
        viewModel.loadCurrentTabData()
        delay(300L)
        shouldRenderGrid = true
    }

    fun loadCurrentTabData() {
        scope.launch {
            viewModel.loadCurrentTabData()
        }
    }

    // 监听生命周期，onResume 时刷新列表
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                loadCurrentTabData()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }



    Scaffold(
        topBar = {
            TopAppBar(
                modifier = Modifier,
                title = {
                    if (isSelectionMode) {
                        Text(
                            text = stringResource(R.string.items_selected, selectedPhotos.size),
                            color = Color.White
                        )
                    } else {
                        TabRow(
                            selectedTabIndex = selectedTab.ordinal,
                            containerColor = Color.Transparent,
                            contentColor = Color.White,
                            indicator = { tabPositions ->
                                TabRowDefaults.SecondaryIndicator(
                                    Modifier.tabIndicatorOffset(tabPositions[selectedTab.ordinal]),
                                    color = AccentOrange
                                )
                            },
                            divider = {}
                        ) {
                            Tab(
                                selected = selectedTab == GalleryTab.PHOTON,
                                onClick = {
                                    scope.launch {
                                        viewModel.selectTab(GalleryTab.PHOTON)
                                    }
                                },
                                text = {
                                    Text(
                                        text = stringResource(R.string.gallery_photon),
                                        fontSize = 14.sp,
                                        fontWeight = if (selectedTab == GalleryTab.PHOTON) FontWeight.Bold else FontWeight.Normal
                                    )
                                }
                            )
                            Tab(
                                selected = selectedTab == GalleryTab.SYSTEM,
                                onClick = {
                                    scope.launch {
                                        viewModel.selectTab(GalleryTab.SYSTEM)
                                    }
                                },
                                text = {
                                    Text(
                                        text = stringResource(R.string.gallery_system),
                                        fontSize = 14.sp,
                                        fontWeight = if (selectedTab == GalleryTab.SYSTEM) FontWeight.Bold else FontWeight.Normal
                                    )
                                }
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = {
                        if (isSelectionMode) {
                            viewModel.exitSelectionMode()
                        } else {
                            onBack()
                        }
                    }) {
                        Icon(
                            imageVector = if (isSelectionMode) Icons.Default.Close else Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.back),
                            tint = Color.White
                        )
                    }
                },
                actions = {
                    if (isSelectionMode) {
                        IconButton(onClick = { viewModel.toggleSelectAll() }) {
                            Icon(
                                imageVector = Icons.Default.SelectAll,
                                contentDescription = stringResource(R.string.select_all),
                                tint = Color.White
                            )
                        }
                    } else {
                        IconButton(onClick = { launcher.launch("image/*") }) {
                            Icon(
                                imageVector = Icons.Default.AddPhotoAlternate,
                                contentDescription = stringResource(R.string.import_photo),
                                tint = Color.White
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF151515)
                )
            )
        },
        bottomBar = {
            // 多选模式下显示操作栏
            AnimatedVisibility(
                visible = isSelectionMode && selectedPhotos.isNotEmpty(),
                enter = slideInVertically(initialOffsetY = { it }) + fadeIn(),
                exit = slideOutVertically(targetOffsetY = { it }) + fadeOut()
            ) {
                Surface(
                    color = Color(0xFF1A1A1A),
                    modifier = Modifier.fillMaxWidth().navigationBarsPadding()
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        // 删除按钮
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.clickable { showDeleteDialog = true }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = stringResource(R.string.delete),
                                tint = Color.White,
                                modifier = Modifier.size(28.dp)
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = stringResource(R.string.delete),
                                color = Color.White,
                                fontSize = 12.sp
                            )
                        }

                        // 分享按钮
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.clickable(enabled = !isSharing) { viewModel.shareSelectedPhotos() }
                        ) {
                            if (isSharing) {
                                CircularProgressIndicator(
                                    color = Color.White,
                                    modifier = Modifier.size(28.dp),
                                    strokeWidth = 2.dp
                                )
                            } else {
                                Icon(
                                    imageVector = Icons.Default.Share,
                                    contentDescription = stringResource(R.string.share),
                                    tint = Color.White,
                                    modifier = Modifier.size(28.dp)
                                )
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = stringResource(R.string.share),
                                color = Color.White,
                                fontSize = 12.sp
                            )
                        }

                        // 导出按钮
                        if (viewModel.selectedTab == GalleryTab.PHOTON) {
                            val isExporting by viewModel.isExporting.collectAsState()
                            val context = LocalContext.current
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier.clickable(enabled = !isExporting) {
                                    viewModel.exportSelectedPhotos { count ->
                                        if (count > 0) {
                                            Toast.makeText(context, R.string.export_complete, Toast.LENGTH_SHORT).show()
                                        }
                                    }
                                }
                            ) {
                                if (isExporting) {
                                    val progress = viewModel.exportProgress
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        CircularProgressIndicator(
                                            color = Color.White,
                                            modifier = Modifier.size(28.dp),
                                            strokeWidth = 2.dp
                                        )
                                        Text(
                                            text = "${progress.first}/${progress.second}",
                                            color = Color.White,
                                            fontSize = 10.sp
                                        )
                                    }
                                } else {
                                    Icon(
                                        imageVector = Icons.Default.Output,
                                        contentDescription = stringResource(R.string.export),
                                        tint = AccentOrange,
                                        modifier = Modifier.size(28.dp)
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = stringResource(R.string.export),
                                        color = Color.White,
                                        fontSize = 12.sp
                                    )
                                }
                            }
                        }
                    }
                }
            }
        },
        containerColor = Color(0xFF151515),
        modifier = modifier
    ) { paddingValues ->
        PullToRefreshBox(
            isRefreshing = isLoading,
            onRefresh = {
                loadCurrentTabData()
            },
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            if (selectedTab == GalleryTab.SYSTEM && !hasPermission) {
                // 权限缺失提示
                Column(
                    modifier = Modifier.align(Alignment.Center).padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = stringResource(R.string.permission_required_gallery),
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 16.sp,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(
                        onClick = {
                            val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                arrayOf(
                                    android.Manifest.permission.READ_MEDIA_IMAGES,
                                    android.Manifest.permission.READ_MEDIA_VIDEO
                                )
                            } else {
                                arrayOf(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                            }
                            permissionLauncher.launch(permission)
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = AccentOrange)
                    ) {
                        Text(stringResource(R.string.grant_permission))
                    }
                }
            } else if (photos.isEmpty()) {
                // 空状态
                Column(
                    modifier = Modifier.align(Alignment.Center),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = stringResource(R.string.no_photos),
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 16.sp,
                        textAlign = TextAlign.Center
                    )
                }
            } else if (!shouldRenderGrid) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        color = AccentOrange,
                        modifier = Modifier.size(28.dp),
                        strokeWidth = 2.dp
                    )
                }
            } else {
                val currentPhotos = photos
                val gridEntries = remember(currentPhotos, context) {
                    buildGalleryGridEntries(
                        context = context,
                        photos = currentPhotos
                    )
                }
                Box(modifier = Modifier.fillMaxSize()) {
                    // 照片网格
                    LazyVerticalStaggeredGrid(
                        columns = StaggeredGridCells.Fixed(3),
                        state = currentGridState,
                        contentPadding = PaddingValues(4.dp),
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalItemSpacing = 4.dp,
                        modifier = Modifier.fillMaxSize()
                    ) {
                        items(
                            items = gridEntries,
                            key = { entry ->
                                when (entry) {
                                    is GalleryGridEntry.Header -> "${selectedTab.name}_${entry.key}"
                                    is GalleryGridEntry.Photo -> "${selectedTab.name}_${entry.photo.id}"
                                }
                            },
                            contentType = { entry ->
                                when (entry) {
                                    is GalleryGridEntry.Header -> "header"
                                    is GalleryGridEntry.Photo -> "photo"
                                }
                            },
                            span = { entry ->
                                if (
                                    entry is GalleryGridEntry.Header ||
                                    (entry is GalleryGridEntry.Photo && entry.photo.shouldUseFullLineSpan())
                                ) {
                                    StaggeredGridItemSpan.FullLine
                                } else {
                                    StaggeredGridItemSpan.SingleLane
                                }
                            }
                        ) { entry ->
                            when (entry) {
                                is GalleryGridEntry.Header -> {
                                    GalleryDateHeader(title = entry.title)
                                }

                                is GalleryGridEntry.Photo -> {
                                    val photo = entry.photo
                                    PhotoGridItem(
                                        photo = photo,
                                        viewModel = viewModel,
                                        isSelected = selectedPhotos.contains(photo),
                                        isSelectionMode = isSelectionMode,
                                        onClick = {
                                            if (isSelectionMode) {
                                                viewModel.togglePhotoSelection(photo)
                                            } else {
                                                viewModel.setCurrentPhoto(entry.index)
                                                onPhotoClick(selectedTab, entry.index)
                                            }
                                        },
                                        onLongClick = {
                                            if (!isSelectionMode) {
                                                viewModel.enterSelectionMode()
                                            }
                                            viewModel.togglePhotoSelection(photo)
                                        }
                                    )

                                    // 触底加载更多 (仅限系统相册)
                                    if (photo == currentPhotos.lastOrNull()) {
                                        LaunchedEffect(photo.id) {
                                            viewModel.loadCurrentTabMore()
                                        }
                                    }
                                }
                            }
                        }

                        if ((selectedTab == GalleryTab.SYSTEM && isSystemLoadingMore) ||
                            (selectedTab == GalleryTab.PHOTON && isPhotonLoadingMore)
                        ) {
                            item(span = StaggeredGridItemSpan.FullLine) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(16.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    CircularProgressIndicator(
                                        color = AccentOrange,
                                        modifier = Modifier.size(24.dp),
                                        strokeWidth = 2.dp
                                    )
                                }
                            }
                        }
                    }

                    GalleryFastScrollbar(
                        gridState = currentGridState,
                        isDragging = isFastScrolling.value,
                        onDragStateChange = { isFastScrolling.value = it },
                        modifier = Modifier
                            .align(Alignment.CenterEnd)
                            .fillMaxHeight()
                            .padding(vertical = 8.dp)
                    )
                }
            }
        }
    }

    // 删除确认对话框
    if (showDeleteDialog) {
        var deleteExported by remember { mutableStateOf(false) }

        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text(stringResource(R.string.delete)) },
            text = {
                Column {
                    Text(
                        text = stringResource(R.string.delete_multiple_confirm, selectedPhotos.size)
                    )
                    if (viewModel.selectedTab == GalleryTab.PHOTON) {
                        Spacer(modifier = Modifier.height(12.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.clickable { deleteExported = !deleteExported }
                        ) {
                            Checkbox(
                                checked = deleteExported,
                                onCheckedChange = { deleteExported = it },
                                colors = CheckboxDefaults.colors(
                                    checkedColor = Color(0xFFFF6B35),
                                    uncheckedColor = Color.White.copy(alpha = 0.6f)
                                )
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = stringResource(R.string.delete_exported_photos),
                                color = Color.White.copy(alpha = 0.9f),
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteSelectedPhotos(deleteExported)
                        showDeleteDialog = false
                    }
                ) {
                    Text(stringResource(R.string.delete), color = Color.Red)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            },
            containerColor = Color(0xFF2D2D2D),
            titleContentColor = Color.White,
            textContentColor = Color.White
        )
    }
}

@Composable
private fun GalleryFastScrollbar(
    gridState: LazyStaggeredGridState,
    isDragging: Boolean,
    onDragStateChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    val scope = rememberCoroutineScope()
    val density = LocalDensity.current
    val layoutInfo = gridState.layoutInfo
    val totalItems = layoutInfo.totalItemsCount
    val visibleItems = layoutInfo.visibleItemsInfo
    val scrollIndicatorState = gridState.scrollIndicatorState
    var scrollJob by remember { mutableStateOf<Job?>(null) }

    if (
        totalItems <= 0 ||
        visibleItems.isEmpty() ||
        scrollIndicatorState == null ||
        totalItems <= visibleItems.size
    ) {
        return
    }

    val currentScrollPx = scrollIndicatorState.scrollOffset
        .takeIfKnown()
        ?.toFloat()
        ?: return
    val contentSizePx = scrollIndicatorState.contentSize
        .takeIfKnown()
        ?.toFloat()
        ?: return
    val viewportSizePx = scrollIndicatorState.viewportSize
        .takeIfKnown()
        ?.toFloat()
        ?: return
    val maxScrollPx = (contentSizePx - viewportSizePx).coerceAtLeast(1f)

    var isVisible by remember { mutableStateOf(false) }
    LaunchedEffect(gridState.isScrollInProgress, isDragging, currentScrollPx) {
        if (gridState.isScrollInProgress || isDragging) {
            isVisible = true
        } else {
            delay(900L)
            isVisible = false
        }
    }
    val alpha by animateFloatAsState(
        targetValue = if (isVisible || isDragging) 1f else 0f,
        animationSpec = tween(durationMillis = 180),
        label = "galleryFastScrollbarAlpha"
    )

    BoxWithConstraints(
        modifier = modifier
            .width(40.dp)
            .graphicsLayer { this.alpha = alpha },
        contentAlignment = Alignment.TopEnd
    ) {
        val trackHeightPx = constraints.maxHeight.toFloat().coerceAtLeast(1f)
        val thumbHeightPx = with(density) { 56.dp.toPx() }.coerceAtMost(trackHeightPx)
        val availableTravelPx = (trackHeightPx - thumbHeightPx).coerceAtLeast(1f)
        val progress = currentScrollPx / maxScrollPx
        val thumbOffsetPx = (availableTravelPx * progress).coerceIn(0f, availableTravelPx)
        val animatedThumbOffsetPx by animateFloatAsState(
            targetValue = thumbOffsetPx,
            animationSpec = tween(durationMillis = 90),
            label = "galleryFastScrollbarOffset"
        )
        val displayedThumbOffsetPx = if (isDragging) thumbOffsetPx else animatedThumbOffsetPx

        fun scrollToTrackPosition(yPx: Float) {
            val targetProgress = ((yPx - thumbHeightPx / 2f) / availableTravelPx).coerceIn(0f, 1f)
            val targetIndex = (targetProgress * (totalItems - visibleItems.size))
                .roundToInt()
                .coerceIn(0, totalItems - 1)
            scrollJob?.cancel()
            scrollJob = scope.launch {
                gridState.scrollToItem(targetIndex)
            }
        }

        Box(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .fillMaxHeight()
                .width(40.dp)
                .pointerInput(totalItems) {
                    detectDragGestures(
                        onDragStart = { offset ->
                            onDragStateChange(true)
                            scrollToTrackPosition(offset.y)
                        },
                        onDragEnd = {
                            onDragStateChange(false)
                            scrollJob = null
                        },
                        onDragCancel = {
                            onDragStateChange(false)
                            scrollJob?.cancel()
                            scrollJob = null
                        },
                        onDrag = { change, _ ->
                            change.consume()
                            scrollToTrackPosition(change.position.y)
                        }
                    )
                }
        ) {
            Box(
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .fillMaxHeight()
                    .width(6.dp)
                    .clip(RoundedCornerShape(999.dp))
                    .background(Color.White.copy(alpha = 0.16f))
            )
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(end = 1.dp)
                    .offset(y = with(density) { displayedThumbOffsetPx.toDp() })
                    .width(4.dp)
                    .height(with(density) { thumbHeightPx.toDp() })
                    .clip(RoundedCornerShape(999.dp))
                    .background(AccentOrange.copy(alpha = 0.9f))
            )
        }
    }
}

private fun Int.takeIfKnown(): Int? =
    takeIf { it != Int.MAX_VALUE && it >= 0 }

private fun MediaData.shouldUseFullLineSpan(): Boolean {
    if (!isImage) return false
    val width = (metadata?.width ?: width).takeIf { it > 0 } ?: return false
    val height = (metadata?.height ?: height).takeIf { it > 0 } ?: return false
    return width.toFloat() / height.toFloat() >= 2.2f
}

/**
 * 日期分组标题
 */
@Composable
private fun GalleryDateHeader(
    title: String
) {
    Text(
        text = title,
        color = Color.White.copy(alpha = 0.9f),
        fontSize = 14.sp,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 6.dp, vertical = 10.dp)
    )
}

/**
 * 照片网格项
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PhotoGridItem(
    photo: MediaData,
    viewModel: GalleryViewModel,
    isSelected: Boolean,
    isSelectionMode: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit
) {
    val refreshKey = viewModel.photoRefreshKeys[photo.id] ?: 0L
    val thumbnail by produceState<Bitmap?>(
        initialValue = null,
        key1 = photo.id,
        key2 = photo.thumbnailUri,
        key3 = refreshKey
    ) {
        value = viewModel.getGridThumbnailBitmap(photo)
    }

    val aspectRatio = remember(photo.width, photo.height, photo.metadata, OrientationObserver.isLandscape) {
        val rawWidth = if (photo.isVideo) {
            photo.metadata?.videoWidth ?: photo.width
        } else {
            photo.metadata?.width ?: photo.width
        }
        val rawHeight = if (photo.isVideo) {
            photo.metadata?.videoHeight ?: photo.height
        } else {
            photo.metadata?.height ?: photo.height
        }
        val rotationDegrees = if (photo.isVideo) {
            photo.metadata?.rotationDegrees ?: 0
        } else {
            0
        }
        val shouldSwapDimensions = rotationDegrees == 90 || rotationDegrees == 270
        val width = if (shouldSwapDimensions) rawHeight else rawWidth
        val height = if (shouldSwapDimensions) rawWidth else rawHeight
        if (width > 0 && height > 0) {
            if (OrientationObserver.isLandscape) height.toFloat() / width.toFloat() else width.toFloat() / height.toFloat()
        } else {
            1f
        }
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(aspectRatio)
            .clip(RoundedCornerShape(4.dp))
            .background(Color(0xFF1A1A1A))
            .combinedClickable(
                onClick = onClick,
                onLongClick = onLongClick
            )
            .autoRotate(matchParentSize = true)
    ) {
        // 照片缩略图
        thumbnail?.takeIf { !it.isRecycled }?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = photo.displayName,
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize()
            )
        }

        // 选择指示器
        if (isSelectionMode) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        if (isSelected) AccentOrange.copy(alpha = 0.3f) else Color.Transparent
                    )
            )

            Icon(
                imageVector = if (isSelected) Icons.Default.CheckCircle else Icons.Default.RadioButtonUnchecked,
                contentDescription = if (isSelected) stringResource(R.string.selected) else stringResource(R.string.not_selected),
                tint = if (isSelected) AccentOrange else Color.White.copy(alpha = 0.7f),
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(6.dp)
                    .size(24.dp)
            )
        }

        // 媒体类型标记
        if (photo.isVideo) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(4.dp)
                    .background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(999.dp))
                    .padding(horizontal = 6.dp, vertical = 4.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = stringResource(R.string.video),
                        tint = Color.White,
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(2.dp))
                    Text(
                        text = photo.getFormattedDuration(),
                        color = Color.White,
                        fontSize = 10.sp,
                        lineHeight = 14.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        } else if (photo.isMotionPhoto) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(4.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        painterResource(R.drawable.ic_live_photo),
                        contentDescription = stringResource(R.string.settings_use_live_photo),
                        tint = Color.White,
                        modifier = Modifier.size(13.dp)
                    )
                }
            }
        }

        if (photo.isBurstPhoto) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(4.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.BurstMode,
                        contentDescription = "Burst photo",
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }

        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
        ) {
            // RAW 标记
            val isRaw = photo.isImage && viewModel.isRawInGallery(photo.id)
            if (isRaw) {
                Box(
                    modifier = Modifier
                        .padding(3.dp)
                        .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(2.dp))
                        .padding(horizontal = 4.dp)
                ) {
                    Text(
                        text = "RAW",
                        color = Color.White,
                        fontSize = 8.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // 导入标记
            if (viewModel.selectedTab == GalleryTab.PHOTON && photo.metadata?.isImported == true) {
                Box(
                    modifier = Modifier
                        .padding(3.dp)
                        .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(2.dp))
                        .padding(horizontal = 4.dp)
                ) {
                    Text(
                        text = stringResource(R.string.imported),
                        color = Color.White,
                        fontSize = 8.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        // 关联标记 (System Tab 专用，表示该照片在 App 内有对应版本)
        if (viewModel.selectedTab == GalleryTab.SYSTEM && photo.relatedPhoto != null) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(4.dp)
                    .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(2.dp))
                    .padding(2.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.AutoFixHigh,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(12.dp)
                )
            }
        }
    }
}
