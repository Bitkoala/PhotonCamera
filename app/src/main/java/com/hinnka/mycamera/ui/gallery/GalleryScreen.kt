package com.hinnka.mycamera.ui.gallery

import android.app.Activity
import android.content.Context
import android.graphics.Rect
import android.graphics.Bitmap
import android.widget.Toast
import android.view.LayoutInflater
import android.widget.ImageView
import android.widget.TextView
import android.os.Build
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
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
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.AddPhotoAlternate
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.viewinterop.AndroidView
import androidx.recyclerview.widget.RecyclerView
import androidx.recyclerview.widget.StaggeredGridLayoutManager
import com.hinnka.mycamera.R
import com.hinnka.mycamera.gallery.MediaData
import com.hinnka.mycamera.ui.theme.AccentOrange
import com.hinnka.mycamera.utils.OrientationObserver
import com.hinnka.mycamera.viewmodel.GalleryTab
import com.hinnka.mycamera.viewmodel.GalleryViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import androidx.recyclerview.widget.DiffUtil
import android.os.Parcelable

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
                GalleryRecyclerGrid(
                    entries = gridEntries,
                    selectedTab = selectedTab,
                    viewModel = viewModel,
                    selectedPhotos = selectedPhotos,
                    isSelectionMode = isSelectionMode,
                    isLoadingMore = (selectedTab == GalleryTab.SYSTEM && isSystemLoadingMore) ||
                        (selectedTab == GalleryTab.PHOTON && isPhotonLoadingMore),
                    onPhotoClick = onPhotoClick,
                    onLoadMore = { scope.launch { viewModel.loadCurrentTabMore() } },
                    modifier = Modifier.fillMaxSize()
                )
            }
        }
    }

    // 删除确认对话框
    if (showDeleteDialog) {
//        var deleteExported by remember { mutableStateOf(true) }

        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text(stringResource(R.string.delete)) },
            text = {
                Column {
                    Text(
                        text = stringResource(R.string.delete_multiple_confirm, selectedPhotos.size)
                    )
                    /*if (viewModel.selectedTab == GalleryTab.PHOTON) {
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
                    }*/
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteSelectedPhotos(true)
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
private fun GalleryRecyclerGrid(
    entries: List<GalleryGridEntry>,
    selectedTab: GalleryTab,
    viewModel: GalleryViewModel,
    selectedPhotos: List<MediaData>,
    isSelectionMode: Boolean,
    isLoadingMore: Boolean,
    onPhotoClick: (GalleryTab, Int) -> Unit,
    onLoadMore: () -> Unit,
    modifier: Modifier = Modifier
) {
    val selectedPhotoIds = selectedPhotos.map { it.id }.toSet()
    val adapter = remember {
        GalleryRecyclerAdapter()
    }

    // Preserve scroll state for each tab separately
    var photonScrollState by rememberSaveable { mutableStateOf<Parcelable?>(null) }
    var systemScrollState by rememberSaveable { mutableStateOf<Parcelable?>(null) }

    val currentScrollState = if (selectedTab == GalleryTab.PHOTON) photonScrollState else systemScrollState

    // Use key(selectedTab) to ensure we have separate RecyclerView state management for each tab
    key(selectedTab) {
        AndroidView(
            modifier = modifier,
            factory = { context ->
                val spacingPx = (4f * context.resources.displayMetrics.density).toInt().coerceAtLeast(1)
                RecyclerView(context).apply {
                    layoutManager = StaggeredGridLayoutManager(3, RecyclerView.VERTICAL).apply {
                        gapStrategy = StaggeredGridLayoutManager.GAP_HANDLING_MOVE_ITEMS_BETWEEN_SPANS
                        // Restore state for this specific tab
                        currentScrollState?.let { onRestoreInstanceState(it) }
                    }
                    setPadding(spacingPx / 2, spacingPx / 2, spacingPx / 2, spacingPx / 2)
                    clipToPadding = false
                    overScrollMode = RecyclerView.OVER_SCROLL_NEVER
                    isVerticalScrollBarEnabled = true
                    itemAnimator = null
                    setHasFixedSize(false)
                    addItemDecoration(GalleryGridSpacingDecoration(spacingPx))
                    this.adapter = adapter

                    // Save state when scrolling stops
                    addOnScrollListener(object : RecyclerView.OnScrollListener() {
                        override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                            if (newState == RecyclerView.SCROLL_STATE_IDLE) {
                                val state = layoutManager?.onSaveInstanceState()
                                if (selectedTab == GalleryTab.PHOTON) {
                                    photonScrollState = state
                                } else {
                                    systemScrollState = state
                                }
                            }
                        }
                    })
                }
            },
            update = { recyclerView ->
                val spanCount = 3 // Keep 3 columns as requested
                val layoutManager = recyclerView.layoutManager as? StaggeredGridLayoutManager
                if (layoutManager != null && layoutManager.spanCount != spanCount) {
                    layoutManager.spanCount = spanCount
                }

                adapter.bindState(
                    entries = entries,
                    selectedTab = selectedTab,
                    viewModel = viewModel,
                    selectedPhotoIds = selectedPhotoIds,
                    isSelectionMode = isSelectionMode,
                    isLoadingMore = isLoadingMore,
                    isLandscape = OrientationObserver.isLandscape,
                    rotationDegrees = OrientationObserver.rotationDegrees,
                    onPhotoClick = onPhotoClick,
                    onLoadMore = onLoadMore
                )
                if (recyclerView.adapter == null) {
                    recyclerView.adapter = adapter
                }
                // Ensure layout is recalculated if entries changed to prevent gaps
                (recyclerView.layoutManager as? StaggeredGridLayoutManager)?.invalidateSpanAssignments()
            }
        )
    }
}

private class GalleryGridSpacingDecoration(
    private val spacingPx: Int
) : RecyclerView.ItemDecoration() {
    private val halfSpacingPx = spacingPx / 2

    override fun getItemOffsets(
        outRect: Rect,
        view: View,
        parent: RecyclerView,
        state: RecyclerView.State
    ) {
        outRect.set(halfSpacingPx, halfSpacingPx, halfSpacingPx, halfSpacingPx)
    }
}

private class GalleryRecyclerAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var entries: List<GalleryGridEntry> = emptyList()
    private var selectedTab: GalleryTab = GalleryTab.PHOTON
    private var viewModel: GalleryViewModel? = null
    private var selectedPhotoIds: Set<String> = emptySet()
    private var isSelectionMode: Boolean = false
    private var isLoadingMore: Boolean = false
    private var isLandscape: Boolean = false
    private var rotationDegrees: Float = 0f
    private var onPhotoClick: (GalleryTab, Int) -> Unit = { _, _ -> }
    private var onLoadMore: () -> Unit = {}

    init {
        setHasStableIds(true)
    }

    fun bindState(
        entries: List<GalleryGridEntry>,
        selectedTab: GalleryTab,
        viewModel: GalleryViewModel,
        selectedPhotoIds: Set<String>,
        isSelectionMode: Boolean,
        isLoadingMore: Boolean,
        isLandscape: Boolean,
        rotationDegrees: Float,
        onPhotoClick: (GalleryTab, Int) -> Unit,
        onLoadMore: () -> Unit
    ) {
        val oldEntries = this.entries
        val oldLoadingMore = this.isLoadingMore
        val oldSelectedPhotoIds = this.selectedPhotoIds
        val oldIsSelectionMode = this.isSelectionMode
        val oldIsLandscape = this.isLandscape
        val oldRotationDegrees = this.rotationDegrees

        this.entries = entries
        this.selectedTab = selectedTab
        this.viewModel = viewModel
        this.selectedPhotoIds = selectedPhotoIds
        this.isSelectionMode = isSelectionMode
        this.isLoadingMore = isLoadingMore
        this.isLandscape = isLandscape
        this.rotationDegrees = rotationDegrees
        this.onPhotoClick = onPhotoClick
        this.onLoadMore = onLoadMore

        val diffResult = DiffUtil.calculateDiff(object : DiffUtil.Callback() {
            override fun getOldListSize(): Int = oldEntries.size + if (oldLoadingMore) 1 else 0
            override fun getNewListSize(): Int = entries.size + if (isLoadingMore) 1 else 0

            override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
                val oldIsLoading = oldItemPosition >= oldEntries.size
                val newIsLoading = newItemPosition >= entries.size
                if (oldIsLoading && newIsLoading) return true
                if (oldIsLoading || newIsLoading) return false

                val oldEntry = oldEntries[oldItemPosition]
                val newEntry = entries[newItemPosition]
                return when {
                    oldEntry is GalleryGridEntry.Header && newEntry is GalleryGridEntry.Header -> oldEntry.key == newEntry.key
                    oldEntry is GalleryGridEntry.Photo && newEntry is GalleryGridEntry.Photo -> oldEntry.photo.id == newEntry.photo.id
                    else -> false
                }
            }

            override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
                if (oldItemPosition >= oldEntries.size || newItemPosition >= entries.size) return true

                val oldEntry = oldEntries[oldItemPosition]
                val newEntry = entries[newItemPosition]
                
                // If anything about the UI state changed, we should rebind
                if (oldIsSelectionMode != isSelectionMode || oldIsLandscape != isLandscape || oldRotationDegrees != rotationDegrees) return false

                return when {
                    oldEntry is GalleryGridEntry.Header && newEntry is GalleryGridEntry.Header -> oldEntry.title == newEntry.title
                    oldEntry is GalleryGridEntry.Photo && newEntry is GalleryGridEntry.Photo -> {
                        val id = oldEntry.photo.id
                        val wasSelected = id in oldSelectedPhotoIds
                        val isSelected = id in selectedPhotoIds
                        wasSelected == isSelected && oldEntry.photo.dateAdded == newEntry.photo.dateAdded
                    }
                    else -> false
                }
            }
        })
        diffResult.dispatchUpdatesTo(this)
    }

    override fun getItemCount(): Int = entries.size + if (isLoadingMore) 1 else 0

    override fun getItemId(position: Int): Long {
        if (position >= entries.size) return "loading_more".hashCode().toLong()
        return when (val entry = entries[position]) {
            is GalleryGridEntry.Header -> entry.key.hashCode().toLong()
            is GalleryGridEntry.Photo -> entry.photo.id.hashCode().toLong()
        }
    }

    override fun getItemViewType(position: Int): Int {
        if (position >= entries.size) return VIEW_TYPE_LOADING
        return when (entries[position]) {
            is GalleryGridEntry.Header -> VIEW_TYPE_HEADER
            is GalleryGridEntry.Photo -> VIEW_TYPE_PHOTO
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return when (viewType) {
            VIEW_TYPE_HEADER -> HeaderHolder(
                LayoutInflater.from(parent.context).inflate(R.layout.item_gallery_header, parent, false) as TextView
            )
            VIEW_TYPE_LOADING -> LoadingHolder(
                LayoutInflater.from(parent.context).inflate(R.layout.item_gallery_loading, parent, false)
            )
            else -> PhotoHolder(GalleryPhotoItemView(parent.context))
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val params = (holder.itemView.layoutParams as? StaggeredGridLayoutManager.LayoutParams)
            ?: StaggeredGridLayoutManager.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        params.isFullSpan = position >= entries.size || entries[position].isFullSpan()
        holder.itemView.layoutParams = params

        if (position >= entries.size) {
            return
        }

        when (val entry = entries[position]) {
            is GalleryGridEntry.Header -> {
                (holder.itemView as TextView).text = entry.title
            }

            is GalleryGridEntry.Photo -> {
                val photo = entry.photo
                val model = viewModel ?: return
                if (position == entries.lastIndex) {
                    onLoadMore()
                }
                (holder as PhotoHolder).bind(
                    photo = photo,
                    viewModel = model,
                    isSelected = photo.id in selectedPhotoIds,
                    isSelectionMode = isSelectionMode,
                    isLandscape = isLandscape,
                    rotationDegrees = rotationDegrees,
                    scope = scope,
                    onClick = {
                        if (isSelectionMode) {
                            model.togglePhotoSelection(photo)
                        } else {
                            model.setCurrentPhoto(entry.index)
                            onPhotoClick(selectedTab, entry.index)
                        }
                    },
                    onLongClick = {
                        if (!isSelectionMode) {
                            model.enterSelectionMode()
                        }
                        model.togglePhotoSelection(photo)
                    }
                )
            }
        }
    }

    override fun onViewRecycled(holder: RecyclerView.ViewHolder) {
        if (holder is PhotoHolder) {
            holder.recycle()
        }
        super.onViewRecycled(holder)
    }

    private fun GalleryGridEntry.isFullSpan(): Boolean =
        this is GalleryGridEntry.Header ||
            (this is GalleryGridEntry.Photo && photo.shouldUseFullLineSpan())

    private class HeaderHolder(view: TextView) : RecyclerView.ViewHolder(view)
    private class LoadingHolder(view: View) : RecyclerView.ViewHolder(view)
    private class PhotoHolder(
        private val view: GalleryPhotoItemView
    ) : RecyclerView.ViewHolder(view) {
        private var job: Job? = null

        fun bind(
            photo: MediaData,
            viewModel: GalleryViewModel,
            isSelected: Boolean,
            isSelectionMode: Boolean,
            isLandscape: Boolean,
            rotationDegrees: Float,
            scope: CoroutineScope,
            onClick: () -> Unit,
            onLongClick: () -> Unit
        ) {
            job?.cancel()
            view.bindStatic(
                photo = photo,
                viewModel = viewModel,
                isSelected = isSelected,
                isSelectionMode = isSelectionMode,
                isLandscape = isLandscape,
                rotationDegrees = rotationDegrees,
                onClick = onClick,
                onLongClick = onLongClick
            )
            job = scope.launch {
                val bitmap = viewModel.getGridThumbnailBitmap(photo)
                view.bindThumbnail(photo.id, bitmap)
            }
        }

        fun recycle() {
            job?.cancel()
            job = null
            view.clearThumbnail()
        }
    }

    private companion object {
        const val VIEW_TYPE_HEADER = 0
        const val VIEW_TYPE_PHOTO = 1
        const val VIEW_TYPE_LOADING = 2
    }
}

private fun MediaData.shouldUseFullLineSpan(): Boolean {
    if (!isImage) return false
    val width = (metadata?.width ?: width).takeIf { it > 0 } ?: return false
    val height = (metadata?.height ?: height).takeIf { it > 0 } ?: return false
    return width.toFloat() / height.toFloat() >= 2.2f
}

private class GalleryPhotoItemView(context: Context) : FrameLayout(context) {
    private val contentLayer: FrameLayout
    private val imageView: ImageView
    private val selectionOverlay: View
    private val selectionIcon: ImageView
    private val videoBadge: LinearLayout
    private val videoDuration: TextView
    private val motionIcon: ImageView
    private val burstIcon: ImageView
    private val rawBadge: TextView
    private val importedBadge: TextView
    private val relatedBadge: View
    private var aspectRatio: Float = 1f
    private var boundPhotoId: String? = null

    init {
        LayoutInflater.from(context).inflate(R.layout.item_gallery_photo, this, true)
        clipToOutline = true
        background = androidx.core.content.ContextCompat.getDrawable(context, R.drawable.bg_gallery_photo_item)
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        contentLayer = findViewById(R.id.gallery_photo_content)
        imageView = findViewById(R.id.gallery_photo_image)
        selectionOverlay = findViewById(R.id.gallery_photo_selection_overlay)
        selectionIcon = findViewById(R.id.gallery_photo_selection_icon)
        videoBadge = findViewById(R.id.gallery_photo_video_badge)
        videoDuration = findViewById(R.id.gallery_photo_video_duration)
        motionIcon = findViewById(R.id.gallery_photo_motion_icon)
        burstIcon = findViewById(R.id.gallery_photo_burst_icon)
        rawBadge = findViewById(R.id.gallery_photo_raw_badge)
        importedBadge = findViewById(R.id.gallery_photo_imported_badge)
        relatedBadge = findViewById(R.id.gallery_photo_related_badge)
        clipChildren = false
        clipToPadding = false
    }

    fun bindStatic(
        photo: MediaData,
        viewModel: GalleryViewModel,
        isSelected: Boolean,
        isSelectionMode: Boolean,
        isLandscape: Boolean,
        rotationDegrees: Float,
        onClick: () -> Unit,
        onLongClick: () -> Unit
    ) {
        boundPhotoId = photo.id
        aspectRatio = photo.galleryAspectRatio(isLandscape)
        imageView.setImageDrawable(null)
        selectionOverlay.visibility = if (isSelectionMode && isSelected) VISIBLE else GONE
        selectionIcon.visibility = if (isSelectionMode) VISIBLE else GONE
        selectionIcon.setImageResource(
            if (isSelected) R.drawable.ic_gallery_check_circle else R.drawable.ic_gallery_radio_unchecked
        )

        videoBadge.visibility = if (photo.isVideo) VISIBLE else GONE
        videoDuration.text = photo.getFormattedDuration()
        motionIcon.visibility = if (photo.isMotionPhoto) VISIBLE else GONE
        burstIcon.visibility = if (photo.isBurstPhoto) VISIBLE else GONE
        rawBadge.visibility = if (photo.isImage && viewModel.isRawInGallery(photo.id)) VISIBLE else GONE
        importedBadge.visibility = if (viewModel.selectedTab == GalleryTab.PHOTON && photo.metadata?.isImported == true) VISIBLE else GONE
        importedBadge.text = context.getString(R.string.imported)
        relatedBadge.visibility = if (viewModel.selectedTab == GalleryTab.SYSTEM && photo.relatedPhoto != null) VISIBLE else GONE

        setOnClickListener { onClick() }
        setOnLongClickListener {
            onLongClick()
            true
        }

        val targetDegrees = if (rotationDegrees != 0f) rotationDegrees - 180f else 0f
        if (contentLayer.rotation != targetDegrees) {
            contentLayer.animate()
                .rotation(targetDegrees)
                .setDuration(300)
                .start()
        }
        requestLayout()
    }

    fun bindThumbnail(photoId: String, bitmap: Bitmap?) {
        if (boundPhotoId != photoId || bitmap == null || bitmap.isRecycled) return
        imageView.setImageBitmap(bitmap)
    }

    fun clearThumbnail() {
        boundPhotoId = null
        imageView.setImageDrawable(null)
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec).coerceAtLeast(1)
        val height = (width / aspectRatio.coerceAtLeast(0.1f)).toInt().coerceAtLeast(1)

        setMeasuredDimension(width, height)

        val isRotated = OrientationObserver.isLandscape
        val childWidth: Int
        val childHeight: Int

        if (isRotated) {
            childWidth = height
            childHeight = width
        } else {
            childWidth = width
            childHeight = height
        }

        val childWidthSpec = MeasureSpec.makeMeasureSpec(childWidth, MeasureSpec.EXACTLY)
        val childHeightSpec = MeasureSpec.makeMeasureSpec(childHeight, MeasureSpec.EXACTLY)

        // Measure contentLayer and other potential children
        for (i in 0 until childCount) {
            val child = getChildAt(i)
            if (child.visibility != GONE) {
                child.measure(childWidthSpec, childHeightSpec)
            }
        }
    }

    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        val width = right - left
        val height = bottom - top

        for (i in 0 until childCount) {
            val child = getChildAt(i)
            if (child.visibility != GONE) {
                val childWidth = child.measuredWidth
                val childHeight = child.measuredHeight
                val childLeft = (width - childWidth) / 2
                val childTop = (height - childHeight) / 2
                child.layout(childLeft, childTop, childLeft + childWidth, childTop + childHeight)
            }
        }
    }
}

private fun MediaData.galleryAspectRatio(isLandscape: Boolean): Float {
    val rawWidth = if (isVideo) {
        metadata?.videoWidth ?: width
    } else {
        metadata?.width ?: width
    }
    val rawHeight = if (isVideo) {
        metadata?.videoHeight ?: height
    } else {
        metadata?.height ?: height
    }
    val rotationDegrees = if (isVideo) metadata?.rotationDegrees ?: 0 else 0
    val shouldSwapDimensions = rotationDegrees == 90 || rotationDegrees == 270
    val resolvedWidth = if (shouldSwapDimensions) rawHeight else rawWidth
    val resolvedHeight = if (shouldSwapDimensions) rawWidth else rawHeight
    return if (resolvedWidth > 0 && resolvedHeight > 0) {
        if (isLandscape) {
            resolvedHeight.toFloat() / resolvedWidth.toFloat()
        } else {
            resolvedWidth.toFloat() / resolvedHeight.toFloat()
        }
    } else {
        1f
    }
}
