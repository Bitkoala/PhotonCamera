package com.hinnka.mycamera

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.SystemBarStyle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalWindowInfo
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.repeatOnLifecycle
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.hinnka.mycamera.camera.AspectRatio
import com.hinnka.mycamera.screencapture.ScreenCaptureRenderConfigStore
import com.hinnka.mycamera.ui.settings.PhantomPipCropScreen
import com.hinnka.mycamera.ui.camera.CameraScreen
import com.hinnka.mycamera.ui.gallery.BurstDetailScreen
import com.hinnka.mycamera.ui.gallery.GalleryScreen
import com.hinnka.mycamera.ui.gallery.GalleryDetailScreen
import com.hinnka.mycamera.ui.gallery.PhotoEditScreen
import com.hinnka.mycamera.ui.settings.FilterManagementScreen
import com.hinnka.mycamera.ui.settings.FrameEditorScreen
import com.hinnka.mycamera.ui.settings.FrameManagementScreen
import com.hinnka.mycamera.ui.settings.SettingsScreen
import com.hinnka.mycamera.ui.theme.PhotonCameraTheme
import com.hinnka.mycamera.update.AppUpdateManager
import com.hinnka.mycamera.utils.BuglyHelper
import com.hinnka.mycamera.utils.OrientationObserver
import com.hinnka.mycamera.viewmodel.CameraViewModel
import com.hinnka.mycamera.viewmodel.GalleryTab
import com.hinnka.mycamera.viewmodel.GalleryViewModel
import com.hinnka.mycamera.lut.creator.LutCreatorScreen
import com.hinnka.mycamera.lut.creator.LutCreatorViewModel
import com.hinnka.mycamera.utils.DeviceUtil
import com.hinnka.mycamera.gallery.GalleryManager
import com.hinnka.mycamera.utils.PLog
import com.hinnka.mycamera.utils.StartupTrace

/**
 * 路由常量
 */
object Routes {
    const val CAMERA = "camera"
    const val GALLERY = "gallery"
    const val PHOTO_DETAIL = "photo_detail/{tab}/{index}?photoId={photoId}"
    const val BURST_DETAIL = "burst_detail/{photoId}"
    const val PHOTO_EDIT = "photo_edit"
    const val SETTINGS = "settings"
    const val FILTER_MANAGEMENT = "filter_management?locateLutId={locateLutId}"
    const val FRAME_MANAGEMENT = "frame_management"
    const val FRAME_EDITOR = "frame_editor?frameId={frameId}&imageFrame={imageFrame}"
    const val LUT_CREATOR = "lut_creator"
    const val PHANTOM_PIP_CROP = "phantom_pip_crop"

    fun photoDetail(tab: GalleryTab = GalleryTab.PHOTON, index: Int = 0, photoId: String? = null) =
        "photo_detail/$tab/$index" + (if (photoId != null) "?photoId=$photoId" else "")

    fun filterManagement(locateLutId: String? = null) = 
        "filter_management" + (if (locateLutId != null) "?locateLutId=$locateLutId" else "")

    fun burstDetail(photoId: String) = "burst_detail/$photoId"

    fun frameEditor(frameId: String? = null, imageFrame: Boolean = false): String {
        return buildString {
            append("frame_editor?imageFrame=$imageFrame")
            if (frameId != null) {
                append("&frameId=$frameId")
            }
        }
    }
}

class MainActivity : ComponentActivity() {

    private val cameraViewModel: CameraViewModel by viewModels()
    private val galleryViewModel: GalleryViewModel by viewModels()

    private val permissions =
        arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

    private var hasPermissions by mutableStateOf(false)
    private var pendingRoute by mutableStateOf<String?>(null)
    private var pendingZipImportUris by mutableStateOf<List<Uri>>(emptyList())

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { result ->
        hasPermissions = result.values.all { it }
    }

    private fun applyPreferredWindowColorMode() {
        val configuration = resources.configuration
        when {
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE && !DeviceUtil.isHarmonyOS && configuration.isScreenHdr -> {
                window.colorMode = ActivityInfo.COLOR_MODE_HDR
            }
            configuration.isScreenWideColorGamut -> {
                window.colorMode = ActivityInfo.COLOR_MODE_WIDE_COLOR_GAMUT
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        StartupTrace.mark("MainActivity.onCreate start")

        // 启用全屏模式
        StartupTrace.measure("MainActivity.enableEdgeToEdge") {
            enableEdgeToEdge(
                statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
                navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT)
            )
        }
        hideSystemUI()
        StartupTrace.mark("MainActivity.hideSystemUI applied")
        applyPreferredWindowColorMode()
        StartupTrace.mark("MainActivity.applyPreferredWindowColorMode applied")

        OrientationObserver.observe(this)
        StartupTrace.mark("MainActivity.OrientationObserver.observe applied")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            GalleryManager.hdrSdrRatio = display?.hdrSdrRatio ?: 0f
        }
        PLog.d("MainActivity", "hdrSdrRatio=${GalleryManager.hdrSdrRatio}")

        // 检查权限
        hasPermissions = permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        StartupTrace.mark("MainActivity.permissions checked", "hasPermissions=$hasPermissions")

        handleIntent(intent)
        StartupTrace.mark("MainActivity.intent handled", "pendingRoute=$pendingRoute")


        StartupTrace.mark("MainActivity.setContent start")
        setContent {
            StartupComposeReadyEffect()
            val currentRecipeParams by cameraViewModel.currentRecipeParams.collectAsState()
            val phantomPipCrop by cameraViewModel.phantomPipCrop.collectAsState()
            ScreenCaptureRenderConfigStore.save(
                baselineLutConfig = cameraViewModel.currentBaselineLutConfig,
                baselineColorRecipeParams = cameraViewModel.currentBaselineRecipeParams.value,
                creativeLutConfig = cameraViewModel.currentLutConfig,
                creativeColorRecipeParams = currentRecipeParams,
                crop = phantomPipCrop
            )

            PhotonCameraTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color.Black
                ) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        if (hasPermissions) {
                            NavigationHost(
                                cameraViewModel = cameraViewModel,
                                galleryViewModel = galleryViewModel,
                                pendingRoute = pendingRoute,
                                onRouteHandled = { pendingRoute = null },
                                pendingZipImportUris = pendingZipImportUris,
                                onZipImportHandled = { pendingZipImportUris = emptyList() }
                            )
                        } else {
                            PermissionScreen(
                                onRequestPermission = {
                                    permissionLauncher.launch(permissions)
                                }
                            )
                        }
                        AppUpdateInstallPrompt()
                    }
                }
            }
        }
        StartupTrace.mark("MainActivity.setContent end")
        window.decorView.post {
            StartupTrace.mark("MainActivity.decorView.post")
            reportFullyDrawn()
            StartupTrace.reportFullyDrawn("MainActivity.reportFullyDrawn")
            cameraViewModel.prewarmDepthEstimator()
        }
    }

    override fun onResume() {
        super.onResume()
        applyPreferredWindowColorMode()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            if (cameraViewModel.handleVolumeKey(keyCode == KeyEvent.KEYCODE_VOLUME_UP)) {
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        intent ?: return
        val zipUris = getZipImportUris(intent)
        if (zipUris.isNotEmpty()) {
            pendingZipImportUris = zipUris
            pendingRoute = Routes.FILTER_MANAGEMENT
        } else if (intent.action == Intent.ACTION_SEND && intent.type?.startsWith("image/") == true) {
            val uri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
            }
            if (uri != null) {
                galleryViewModel.importSharedImage(uri) { photoId ->
                    pendingRoute = Routes.photoDetail(photoId = photoId)
                }
            }
        } else if (intent.action == Intent.ACTION_SEND_MULTIPLE && intent.type?.startsWith("image/") == true) {
            val uris = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM)
            }
            if (!uris.isNullOrEmpty()) {
                galleryViewModel.importSharedImages(uris)
                pendingRoute = Routes.GALLERY
            }
        }
        intent.getStringExtra("route")?.let {
            pendingRoute = it
        }
        intent.getBooleanExtra("show_ghost_permissions", false).let {
            cameraViewModel.showGhostPermissions = it
        }
    }

    private fun getZipImportUris(intent: Intent): List<Uri> {
        val isZipMimeType = intent.type?.let { type ->
            type == "application/zip" ||
                    type == "application/x-zip-compressed" ||
                    type == "application/octet-stream"
        } ?: false

        val dataUri = intent.data
        val dataLooksLikeZip = dataUri?.lastPathSegment?.endsWith(".zip", ignoreCase = true) == true

        return when (intent.action) {
            Intent.ACTION_VIEW -> {
                if (dataUri != null && (isZipMimeType || dataLooksLikeZip)) listOf(dataUri) else emptyList()
            }
            Intent.ACTION_SEND -> {
                if (!isZipMimeType) return emptyList()
                val uri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
                }
                listOfNotNull(uri)
            }
            Intent.ACTION_SEND_MULTIPLE -> {
                if (!isZipMimeType) return emptyList()
                val uris = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM)
                }
                uris.orEmpty()
            }
            else -> emptyList()
        }
    }

    private fun hideSystemUI() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            controller.hide(WindowInsetsCompat.Type.statusBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }
}

@Composable
private fun StartupComposeReadyEffect() {
    LaunchedEffect(Unit) {
        StartupTrace.mark("MainActivity.first composition")
    }
}

@Composable
private fun AppUpdateInstallPrompt() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var readyApk by remember { mutableStateOf<java.io.File?>(null) }

    LaunchedEffect(lifecycleOwner) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            AppUpdateManager.readyApk.collect { apkFile ->
                readyApk = apkFile
            }
        }
    }

    val apkFile = readyApk ?: return
    AlertDialog(
        onDismissRequest = {
            AppUpdateManager.consumeReadyApk(apkFile)
            readyApk = null
        },
        title = {
            Text(
                text = stringResource(R.string.update_ready_title),
                style = MaterialTheme.typography.titleLarge
            )
        },
        text = {
            Text(
                text = stringResource(R.string.update_ready_message),
                style = MaterialTheme.typography.bodyMedium
            )
        },
        confirmButton = {
            TextButton(
                onClick = {
                    val installStarted = AppUpdateManager.startInstall(context, apkFile)
                    if (installStarted) {
                        AppUpdateManager.consumeReadyApk(apkFile)
                        readyApk = null
                    } else {
                        android.widget.Toast.makeText(
                            context,
                            R.string.update_install_permission_hint,
                            android.widget.Toast.LENGTH_LONG
                        ).show()
                    }
                }
            ) {
                Text(stringResource(R.string.update_install_now))
            }
        },
        dismissButton = {
            TextButton(
                onClick = {
                    AppUpdateManager.consumeReadyApk(apkFile)
                    readyApk = null
                }
            ) {
                Text(stringResource(R.string.cancel))
            }
        }
    )
}

@Composable
fun NavigationHost(
    cameraViewModel: CameraViewModel,
    galleryViewModel: GalleryViewModel,
    pendingRoute: String? = null,
    onRouteHandled: () -> Unit = {},
    pendingZipImportUris: List<Uri> = emptyList(),
    onZipImportHandled: () -> Unit = {}
) {
    val context = LocalContext.current
    val navController = rememberNavController()

    androidx.compose.runtime.LaunchedEffect(pendingRoute) {
        pendingRoute?.let {
            navController.navigate(it)
            onRouteHandled()
        }
    }
    navController.addOnDestinationChangedListener { _, destination, _ ->
        BuglyHelper.setUserScene(context, destination.route.hashCode())
    }
    val containerSize = LocalWindowInfo.current.containerSize
    cameraViewModel.isExpanded =
        (containerSize.width * 1f / containerSize.height) > AspectRatio.RATIO_4_3.getValue(false)

    Box(modifier = Modifier.fillMaxSize()) {
        NavHost(
            navController = navController,
            startDestination = Routes.CAMERA,
            enterTransition = {
                slideInHorizontally(initialOffsetX = { it }) + fadeIn()
            },
            exitTransition = {
                slideOutHorizontally(targetOffsetX = { -it }) + fadeOut()
            },
            popEnterTransition = {
                slideInHorizontally(initialOffsetX = { -it }) + fadeIn()
            },
            popExitTransition = {
                slideOutHorizontally(targetOffsetX = { it }) + fadeOut()
            }
        ) {
            composable(Routes.CAMERA) {
                if (cameraViewModel.isExpanded) {
                    Row {
                        CameraScreen(
                            viewModel = cameraViewModel,
                            galleryViewModel = galleryViewModel,
                            onGalleryClick = {
                                navController.navigate(Routes.GALLERY)
                                val latestPhoto = galleryViewModel.latestPhoto.value
                                /*if (latestPhoto != null && System.currentTimeMillis() - latestPhoto.dateAdded < 3 * 60 * 1000) {
                                    galleryViewModel.setCurrentPhotoById(latestPhoto.id)
                                    navController.navigate(Routes.photoDetail(photoId = latestPhoto.id))
                                }*/
                            },
                            onSettingsClick = {
                                navController.navigate(Routes.SETTINGS)
                            },
                            onFilterManagementClick = { lutId ->
                                navController.navigate(Routes.filterManagement(lutId))
                            },
                            onFrameManagementClick = {
                                navController.navigate(Routes.FRAME_MANAGEMENT)
                            },
                            modifier = Modifier.weight(1f)
                        )
                        GalleryDetailScreen(
                            viewModel = galleryViewModel,
                            isExpanded = true,
                            onEdit = {
                                navController.navigate(Routes.PHOTO_EDIT)
                            },
                            onViewBurst = { photoId ->
                                navController.navigate(Routes.burstDetail(photoId))
                            },
                            modifier = Modifier.weight(1f)
                        )
                    }
                } else {
                    CameraScreen(
                        viewModel = cameraViewModel,
                        galleryViewModel = galleryViewModel,
                        onGalleryClick = {
                            navController.navigate(Routes.GALLERY)
                            val latestPhoto = galleryViewModel.latestPhoto.value
                            if (latestPhoto != null && System.currentTimeMillis() - latestPhoto.dateAdded < 3 * 60 * 1000) {
                                galleryViewModel.setCurrentPhotoById(latestPhoto.id)
                                navController.navigate(Routes.photoDetail(photoId = latestPhoto.id))
                            }
                        },
                        onSettingsClick = {
                            navController.navigate(Routes.SETTINGS)
                        },
                        onFilterManagementClick = { lutId ->
                            navController.navigate(Routes.filterManagement(lutId))
                        },
                        onFrameManagementClick = {
                            navController.navigate(Routes.FRAME_MANAGEMENT)
                        },
                    )
                }
            }

            composable(Routes.GALLERY) {
                GalleryScreen(
                    viewModel = galleryViewModel,
                    onBack = {
                        navController.popBackStack()
                    },
                    onPhotoClick = { tab, index ->
                        navController.navigate(Routes.photoDetail(tab, index))
                    }
                )
            }

            composable(
                route = Routes.PHOTO_DETAIL,
                arguments = listOf(
                    navArgument("tab") { type = NavType.StringType },
                    navArgument("index") { type = NavType.IntType },
                    navArgument("photoId") {
                        type = NavType.StringType
                        nullable = true
                        defaultValue = null
                    }
                )
            ) { backStackEntry ->
                val index = backStackEntry.arguments?.getInt("index") ?: 0
                val tab = backStackEntry.arguments?.getString("tab") ?: GalleryTab.PHOTON.name
                val photoId = backStackEntry.arguments?.getString("photoId")
                GalleryDetailScreen(
                    viewModel = galleryViewModel,
                    initialIndex = index,
                    selectedTab = GalleryTab.valueOf(tab),
                    photoId = photoId,
                    onBack = {
                        navController.popBackStack()
                    },
                    onEdit = {
                        navController.navigate(Routes.PHOTO_EDIT)
                    },
                    onViewBurst = { id ->
                        navController.navigate(Routes.burstDetail(id))
                    }
                )
            }

            composable(
                route = Routes.BURST_DETAIL,
                arguments = listOf(navArgument("photoId") { type = NavType.StringType })
            ) { backStackEntry ->
                val photoId = backStackEntry.arguments?.getString("photoId") ?: ""
                BurstDetailScreen(
                    viewModel = galleryViewModel,
                    photoId = photoId,
                    onEdit = {
                        navController.navigate(Routes.PHOTO_EDIT)
                    },
                    onBack = {
                        navController.popBackStack()
                    }
                )
            }

            composable(Routes.PHOTO_EDIT) {
                PhotoEditScreen(
                    viewModel = galleryViewModel,
                    cameraViewModel = cameraViewModel,
                    onBack = {
                        navController.popBackStack()
                    },
                    onOpenFrameEditor = { frameId ->
                        navController.navigate(Routes.frameEditor(frameId = frameId))
                    },
                    onFilterManagementClick = { lutId ->
                        navController.navigate(Routes.filterManagement(lutId))
                    }
                )
            }

            composable(Routes.SETTINGS) {
                SettingsScreen(
                    viewModel = cameraViewModel,
                    onBack = {
                        navController.popBackStack()
                    },
                    onFilterManagementClick = {
                        navController.navigate(Routes.FILTER_MANAGEMENT)
                    },
                    onFrameManagementClick = {
                        navController.navigate(Routes.FRAME_MANAGEMENT)
                    },
                    onPhantomPipCropClick = {
                        navController.navigate(Routes.PHANTOM_PIP_CROP)
                    }
                )
            }

            composable(Routes.PHANTOM_PIP_CROP) {
                val crop by cameraViewModel.phantomPipCrop.collectAsState()
                PhantomPipCropScreen(
                    initialCrop = crop,
                    onBack = { navController.popBackStack() },
                    onSave = {
                        cameraViewModel.setPhantomPipCrop(it)
                        navController.popBackStack()
                    }
                )
            }

            composable(
                route = Routes.FILTER_MANAGEMENT,
                arguments = listOf(
                    navArgument("locateLutId") {
                        type = NavType.StringType
                        nullable = true
                        defaultValue = null
                    }
                )
            ) { backStackEntry ->
                val locateLutId = backStackEntry.arguments?.getString("locateLutId")
                FilterManagementScreen(
                    viewModel = cameraViewModel,
                    onBack = {
                        navController.popBackStack()
                    },
                    onLutCreatorClick = {
                        navController.navigate(Routes.LUT_CREATOR)
                    },
                    pendingZipImportUris = pendingZipImportUris,
                    onZipImportHandled = onZipImportHandled,
                    locateLutId = locateLutId
                )
            }

            composable(Routes.LUT_CREATOR) {
                val lutCreatorViewModel: LutCreatorViewModel = androidx.lifecycle.viewmodel.compose.viewModel()
                LutCreatorScreen(
                    viewModel = lutCreatorViewModel,
                    onNavigateBack = { navController.popBackStack() },
                    onSuccess = { lutId ->
                        cameraViewModel.refreshCustomContent()
                        navController.popBackStack()
                    }
                )
            }

            composable(Routes.FRAME_MANAGEMENT) {
                FrameManagementScreen(
                    viewModel = cameraViewModel,
                    onBack = {
                        navController.popBackStack()
                    },
                    onCreateFrameClick = {
                        navController.navigate(Routes.frameEditor())
                    },
                    onEditFrameStyle = { frameId ->
                        navController.navigate(Routes.frameEditor(frameId = frameId))
                    }
                )
            }

            composable(
                route = Routes.FRAME_EDITOR,
                arguments = listOf(
                    navArgument("frameId") {
                        type = NavType.StringType
                        nullable = true
                        defaultValue = null
                    },
                    navArgument("imageFrame") {
                        type = NavType.BoolType
                        defaultValue = false
                    }
                )
            ) { backStackEntry ->
                FrameEditorScreen(
                    viewModel = cameraViewModel,
                    frameId = backStackEntry.arguments?.getString("frameId"),
                    imageFrame = backStackEntry.arguments?.getBoolean("imageFrame") ?: false,
                    onBack = { navController.popBackStack() },
                    onSaved = { savedId ->
                        cameraViewModel.setFrame(savedId)
                        navController.popBackStack()
                    }
                )
            }
        }
    }
}

@Composable
fun PermissionScreen(
    onRequestPermission: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black)
            .navigationBarsPadding(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            Text(
                text = stringResource(R.string.camera_permission_required),
                color = Color.White,
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center
            )

            Button(
                onClick = onRequestPermission,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFFF6B35)
                )
            ) {
                Text(text = stringResource(R.string.grant_permission))
            }
        }
    }
}
