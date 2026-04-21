package com.hinnka.mycamera.ui.components

import android.graphics.Bitmap
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FilterNone
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import coil.size.Size
import coil.transform.Transformation
import com.hinnka.mycamera.R
import com.hinnka.mycamera.data.ContentRepository
import com.hinnka.mycamera.lut.LutInfo
import com.hinnka.mycamera.model.ColorRecipeParams
import com.hinnka.mycamera.ui.camera.LutEditBottomSheet
import com.hinnka.mycamera.utils.PLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private sealed class LutCategoryTab {
    data object BuiltIn : LutCategoryTab()
    data object Uncategorized : LutCategoryTab()
    data class Category(val name: String) : LutCategoryTab()
}

/**
 * LUT 选择器组件
 *
 * 显示可用的 LUT 列表，支持选择和预览
 */
@Composable
fun LutSelector(
    availableLuts: List<LutInfo>,
    currentLutId: String?,
    thumbnail: Bitmap?,
    onLutSelected: (String?) -> Unit,
    onEditClick: (() -> Unit)? = null,
    categoryOrder: List<String> = emptyList(),
    modifier: Modifier = Modifier
) {
    val scrollState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()
    var showLutEditDialogState by remember { mutableStateOf(false) }
    val builtInText = stringResource(R.string.built_in)
    val uncategorizedText = stringResource(R.string.uncategorized)

    // 分类逻辑
    val categoryTabs = remember(availableLuts, categoryOrder, builtInText, uncategorizedText) {
        val dynamicCategories = availableLuts.map { it.category }
            .distinct()
            .filter { it.isNotEmpty() }
        val hasUncategorizedLuts = availableLuts.any { !it.isBuiltIn && it.category.isEmpty() }
        val orderedEntries = categoryOrder.filter { it == builtInText || dynamicCategories.contains(it) }
        val remainingDynamic = dynamicCategories.filterNot { it in orderedEntries }.sorted()

        buildList {
            if (orderedEntries.isEmpty()) {
                add(LutCategoryTab.BuiltIn)
                addAll(remainingDynamic.map(LutCategoryTab::Category))
            } else {
                orderedEntries.forEach { entry ->
                    when (entry) {
                        builtInText -> add(LutCategoryTab.BuiltIn)
                        else -> add(LutCategoryTab.Category(entry))
                    }
                }
                if (builtInText !in orderedEntries) add(LutCategoryTab.BuiltIn)
                addAll(remainingDynamic.map(LutCategoryTab::Category))
            }

            if (hasUncategorizedLuts) {
                add(LutCategoryTab.Uncategorized)
            }
        }
    }
    var selectedCategory by remember(currentLutId, availableLuts) {
        mutableStateOf(
            availableLuts.find { it.id == currentLutId }?.let { lut ->
                if (lut.isBuiltIn) {
                    LutCategoryTab.BuiltIn
                } else if (lut.category.isNotEmpty()) {
                    LutCategoryTab.Category(lut.category)
                } else {
                    LutCategoryTab.Uncategorized
                }
            } ?: LutCategoryTab.BuiltIn
        )
    }

    val filteredLuts = remember(selectedCategory, availableLuts) {
        when (selectedCategory) {
            LutCategoryTab.BuiltIn -> availableLuts.filter { it.isBuiltIn }
            LutCategoryTab.Uncategorized -> availableLuts.filter { !it.isBuiltIn && it.category.isEmpty() }
            is LutCategoryTab.Category -> {
                val categoryName = (selectedCategory as LutCategoryTab.Category).name
                availableLuts.filter { it.category == categoryName }
            }
        }
    }

    val actualShowLutEditDialog = onEditClick == null && showLutEditDialogState

    // 在组件首次加载时滚动到当前选中的 LUT
    LaunchedEffect(currentLutId) {
        currentLutId?.let { lutId ->
            val selectedIndex = filteredLuts.indexOfFirst { it.id == lutId }
            if (selectedIndex >= 2) {
                coroutineScope.launch {
                    scrollState.scrollToItem(selectedIndex - 2)
                }
            }
        }
    }

    // 如果选中的 LUT 不在当前分类中，可选：不自动切换分类，或者在切换分类时如果是“全部”则保持选中

    // 全局 LUT 编辑底部弹窗
    if (actualShowLutEditDialog && currentLutId != null) {
        LutEditBottomSheet(
            lutId = currentLutId,
            onDismiss = {
                showLutEditDialogState = false
            }
        )
    }

    Column(modifier = modifier.fillMaxWidth()) {
        // 分类选择器 (小芯片样式)
        LazyRow(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(horizontal = 8.dp)
        ) {
            items(categoryTabs) { category ->
                val isSelected = selectedCategory == category
                val categoryName = when (category) {
                    LutCategoryTab.BuiltIn -> builtInText
                    LutCategoryTab.Uncategorized -> uncategorizedText
                    is LutCategoryTab.Category -> category.name
                }

                Text(
                    text = categoryName,
                    color = if (isSelected) Color(0xFFFF6B35) else Color.White.copy(alpha = 0.5f),
                    fontSize = 12.sp,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                    modifier = Modifier
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null
                        ) {
                            selectedCategory = category
                        }
                        .padding(vertical = 4.dp)
                )
            }
        }

        LazyRow(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            state = scrollState
        ) {
            // LUT 列表
            items(filteredLuts, key = { it.id }) { lut ->
                LutItem(
                    id = lut.id,
                    name = lut.getName(),
                    previewBitmap = thumbnail,
                    isSelected = currentLutId == lut.id,
                    isVip = lut.isVip,
                    isCustom = !lut.isBuiltIn,
                    onClick = {
                        if (currentLutId == lut.id) {
                            if (onEditClick != null) {
                                onEditClick()
                            } else {
                                showLutEditDialogState = true
                            }
                        } else {
                            onLutSelected(lut.id)
                        }
                    }
                )
            }
        }
    }
}

/**
 * 单个 LUT 选项
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun LutItem(
    id: String,
    name: String,
    previewBitmap: Bitmap?,
    isSelected: Boolean,
    isVip: Boolean,
    onClick: () -> Unit,
    isNone: Boolean = false,
    isCustom: Boolean = false,  // 添加自定义标识参数
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val backgroundColor = if (isSelected) {
        Color.White.copy(alpha = 0.3f)
    } else {
        Color.Black.copy(alpha = 0.5f)
    }

    val borderColor = if (isSelected) {
        Color.White
    } else {
        Color.Gray.copy(alpha = 0.5f)
    }

    Column(
        modifier = modifier
            .width(60.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(backgroundColor)
            .border(
                width = if (isSelected) 2.dp else 1.dp,
                color = borderColor,
                shape = RoundedCornerShape(8.dp)
            )
            .clickable { onClick() }
            .padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // 预览区域
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(RoundedCornerShape(4.dp))
                .then(
                    if (isNone) {
                        Modifier.background(Color.DarkGray)
                    } else if (previewBitmap != null) {
                        // 显示真实预览图
                        Modifier
                    } else {
                        // 占位符：模拟滤镜预览的渐变色
                        Modifier.background(
                            Brush.linearGradient(
                                colors = listOf(
                                    Color(0xFF4A148C),
                                    Color(0xFF00897B),
                                    Color(0xFFFF6F00)
                                )
                            )
                        )
                    }
                ),
            contentAlignment = Alignment.Center
        ) {
            // 显示预览图片
            if (!isNone && previewBitmap != null) {
                // 照片缩略图
                val transformation = remember(id, previewBitmap) {
                    object : Transformation {
                        override val cacheKey: String = "previewTransformation_${previewBitmap.hashCode()}_$id"
                        val contentRepository = ContentRepository.getInstance(context)

                        override suspend fun transform(
                            input: Bitmap,
                            size: Size
                        ): Bitmap {
                            val lutConfig = withContext(Dispatchers.IO) {
                                contentRepository.lutManager.loadLut(id)
                            }
                            if (lutConfig != null) {
                                val colorRecipeParams = contentRepository.lutManager.loadColorRecipeParams(id)
                                return contentRepository.imageProcessor.applyLut(
                                    bitmap = input,
                                    lutConfig = lutConfig,
                                    colorRecipeParams = colorRecipeParams
                                )
                            }
                            return input
                        }

                    }
                }
                val imageRequest = remember(previewBitmap, transformation) {
                    ImageRequest.Builder(context)
                        .data(previewBitmap)
                        .crossfade(true)
                        .transformations(transformation)
                        .build()
                }

                AsyncImage(
                    model = imageRequest,
                    contentDescription = name,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            }
            if (isNone) {
                Icon(
                    imageVector = Icons.Default.FilterNone,
                    contentDescription = null,
                    tint = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.size(24.dp)
                )
            }

            if (isSelected) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.3f)),
                    contentAlignment = Alignment.Center
                ) {
                    Box(
                        modifier = Modifier
                            .size(24.dp)
                            .background(Color.White.copy(alpha = 0.2f), CircleShape)
                            .border(1.dp, Color.White.copy(alpha = 0.5f), CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Tune,
                            contentDescription = stringResource(R.string.edit),
                            tint = Color.White,
                            modifier = Modifier.size(14.dp)
                        )
                    }
                }
            }

            if (isVip) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .background(
                            color = Color(0xFFFFD700),
                            shape = RoundedCornerShape(bottomStart = 4.dp)
                        )
                        .padding(horizontal = 4.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = stringResource(R.string.billing_vip_tag),
                        color = Color.Black,
                        fontSize = 8.sp,
                        fontWeight = FontWeight.Bold,
                        lineHeight = 8.sp
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(4.dp))

        // 名称
        Text(
            text = name,
            color = Color.White,
            fontSize = 9.sp,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth().basicMarquee()
        )
    }
}
