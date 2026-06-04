package com.hinnka.mycamera.raw

import android.content.Context
import com.hinnka.mycamera.lut.LutConfig
import com.hinnka.mycamera.utils.PLog
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.pow

data class SpectralFilmLut(
    val stock: String,
    val name: String,
    val type: String,
    val referenceIlluminant: String,
    val viewingIlluminant: String,
    val sourceKey: String,
    val size: Int,
    val values: FloatArray
)

data class FilmStockInfo(
    val type: String,
    val referenceIlluminant: String,
    val viewingIlluminant: String
)

data class FilmDensityWire(
    val dMin: FloatArray,
    val dMax: FloatArray
)

data class PrintPaperInfo(
    val referenceIlluminant: String,
    val viewingIlluminant: String
)

data class SpectralFilmPrintModel(
    val version: Int,
    val observerCmfs: FloatArray,
    val viewingIlluminants: Map<String, FloatArray>,
    val xyzToProPhoto: Map<String, FloatArray>,
    val films: Map<String, SpectralFilmModelData>,
    val papers: Map<String, SpectralPaperModelData>,
    val combinations: Map<String, Map<String, SpectralPrintCombinationData>>
)

data class SpectralFilmModelData(
    val channelDensity: FloatArray,
    val baseDensity: FloatArray,
    val densityMin: FloatArray,
    val densityMax: FloatArray
)

data class SpectralPaperModelData(
    val viewingIlluminant: String,
    val sensitivity: FloatArray,
    val channelDensity: FloatArray,
    val baseDensity: FloatArray,
    val logExposure: FloatArray,
    val densityCurves: FloatArray
)

data class SpectralPrintCombinationData(
    val printIlluminant: FloatArray,
    val exposureFactor: Float,
    val neutralFilters: FloatArray
)

object FilmStockRegistry {
    private val default = FilmStockInfo(
        type = "negative",
        referenceIlluminant = "D55",
        viewingIlluminant = "D50"
    )

    val stocks = mapOf(
        "fujifilm_c200" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "fujifilm_pro_400h" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "fujifilm_provia_100f" to FilmStockInfo(type = "positive", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "fujifilm_velvia_100" to FilmStockInfo(type = "positive", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "fujifilm_xtra_400" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_ektachrome_100" to FilmStockInfo(type = "positive", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_ektar_100" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_gold_200" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_kodachrome_64" to FilmStockInfo(type = "positive", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_portra_160" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_portra_400" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_portra_800" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_portra_800_push1" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_portra_800_push2" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_ultramax_400" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_verita_200d" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_vision3_200t" to FilmStockInfo(type = "negative", referenceIlluminant = "T", viewingIlluminant = "D50"),
        "kodak_vision3_250d" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50"),
        "kodak_vision3_500t" to FilmStockInfo(type = "negative", referenceIlluminant = "T", viewingIlluminant = "D50"),
        "kodak_vision3_50d" to FilmStockInfo(type = "negative", referenceIlluminant = "D55", viewingIlluminant = "D50")
    )

    fun get(stock: String): FilmStockInfo = stocks[stock] ?: default
}

object FilmDensityWireRegistry {
    private val defaultDMin = floatArrayOf(-0.2f, -0.2f, -0.2f)

    val wires = mapOf(
        "fujifilm_c200" to wire(1.7393f, 1.9257f, 2.2892f),
        "fujifilm_pro_400h" to wire(1.5215f, 1.6171f, 2.0555f),
        "fujifilm_xtra_400" to wire(2.0234f, 2.1502f, 2.6683f),
        "kodak_ektar_100" to wire(1.8008f, 1.7775f, 2.2188f),
        "kodak_gold_200" to wire(1.6538f, 1.5270f, 1.6814f),
        "kodak_portra_160" to wire(1.6281f, 1.6692f, 1.9841f),
        "kodak_portra_400" to wire(1.8590f, 1.7816f, 2.0896f),
        "kodak_portra_800" to wire(1.8386f, 1.7134f, 1.8858f),
        "kodak_portra_800_push1" to wire(2.1669f, 1.9517f, 2.2205f),
        "kodak_portra_800_push2" to wire(2.3529f, 2.0343f, 2.4457f),
        "kodak_ultramax_400" to wire(1.7868f, 1.6910f, 1.9068f),
        "kodak_verita_200d" to wire(1.6719f, 1.6319f, 1.8600f),
        "kodak_vision3_200t" to wire(1.5222f, 1.5838f, 1.8118f),
        "kodak_vision3_250d" to wire(1.5678f, 1.6333f, 1.8032f),
        "kodak_vision3_500t" to wire(1.5709f, 1.5010f, 1.8260f),
        "kodak_vision3_50d" to wire(1.5802f, 1.6924f, 1.9061f)
    )

    fun get(stock: String): FilmDensityWire? = wires[stock]

    private fun wire(cMax: Float, mMax: Float, yMax: Float): FilmDensityWire {
        return FilmDensityWire(
            dMin = defaultDMin.copyOf(),
            dMax = floatArrayOf(cMax, mMax, yMax)
        )
    }
}

object PrintPaperRegistry {
    private val default = PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "D50")

    val papers = mapOf(
        "fujifilm_crystal_archive_typeii" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "D50"),
        "kodak_2383" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "K75P"),
        "kodak_2393" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "K75P"),
        "kodak_ektacolor_edge" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "D50"),
        "kodak_endura_premier" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "D50"),
        "kodak_portra_endura" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "D50"),
        "kodak_supra_endura" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "D50"),
        "kodak_ultra_endura" to PrintPaperInfo(referenceIlluminant = "TH-KG3", viewingIlluminant = "D50")
    )

    fun get(paper: String): PrintPaperInfo = papers[paper] ?: default
}

object SpectralPrintModelRegistry {
    private const val TAG = "SpectralPrintModel"
    private const val MODEL_ASSET = "spektrafilm/print_model.json"
    private const val WAVELENGTH_COUNT = 81
    private const val CHANNEL_COUNT = 3

    @Volatile
    private var cached: SpectralFilmPrintModel? = null

    fun load(context: Context): SpectralFilmPrintModel? {
        cached?.let { return it }
        return synchronized(this) {
            cached ?: loadInternal(context)?.also { cached = it }
        }
    }

    private fun loadInternal(context: Context): SpectralFilmPrintModel? {
        return try {
            val jsonText = context.assets.open(MODEL_ASSET).bufferedReader().use { it.readText() }
            val root = JSONObject(jsonText)
            SpectralFilmPrintModel(
                version = root.optInt("version", 1),
                observerCmfs = root.getJSONArray("observerCmfs").toFloatArray2d(WAVELENGTH_COUNT, CHANNEL_COUNT),
                viewingIlluminants = root.getJSONObject("viewingIlluminants").toFloatArrayMap1d(),
                xyzToProPhoto = root.getJSONObject("xyzToProPhoto").toFloatArrayMap2d(CHANNEL_COUNT, CHANNEL_COUNT),
                films = parseFilms(root.getJSONObject("films")),
                papers = parsePapers(root.getJSONObject("papers")),
                combinations = parseCombinations(root.getJSONObject("combinations"))
            )
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to load Spektrafilm print model", e)
            null
        }
    }

    private fun parseFilms(obj: JSONObject): Map<String, SpectralFilmModelData> {
        val result = mutableMapOf<String, SpectralFilmModelData>()
        obj.keys().forEach { key ->
            val film = obj.getJSONObject(key)
            result[key] = SpectralFilmModelData(
                channelDensity = film.getJSONArray("channelDensity").toFloatArray2d(WAVELENGTH_COUNT, CHANNEL_COUNT),
                baseDensity = film.getJSONArray("baseDensity").toFloatArray1d(),
                densityMin = film.getJSONArray("densityMin").toFloatArray1d(),
                densityMax = film.getJSONArray("densityMax").toFloatArray1d()
            )
        }
        return result
    }

    private fun parsePapers(obj: JSONObject): Map<String, SpectralPaperModelData> {
        val result = mutableMapOf<String, SpectralPaperModelData>()
        obj.keys().forEach { key ->
            val paper = obj.getJSONObject(key)
            result[key] = SpectralPaperModelData(
                viewingIlluminant = paper.getString("viewingIlluminant"),
                sensitivity = paper.getJSONArray("sensitivity").toFloatArray2d(WAVELENGTH_COUNT, CHANNEL_COUNT),
                channelDensity = paper.getJSONArray("channelDensity").toFloatArray2d(WAVELENGTH_COUNT, CHANNEL_COUNT),
                baseDensity = paper.getJSONArray("baseDensity").toFloatArray1d(),
                logExposure = paper.getJSONArray("logExposure").toFloatArray1d(),
                densityCurves = paper.getJSONArray("densityCurves").toFloatArray2d(paper.getJSONArray("logExposure").length(), CHANNEL_COUNT)
            )
        }
        return result
    }

    private fun parseCombinations(obj: JSONObject): Map<String, Map<String, SpectralPrintCombinationData>> {
        val result = mutableMapOf<String, Map<String, SpectralPrintCombinationData>>()
        obj.keys().forEach { filmKey ->
            val paperMap = mutableMapOf<String, SpectralPrintCombinationData>()
            val papers = obj.getJSONObject(filmKey)
            papers.keys().forEach { paperKey ->
                val combo = papers.getJSONObject(paperKey)
                paperMap[paperKey] = SpectralPrintCombinationData(
                    printIlluminant = combo.getJSONArray("printIlluminant").toFloatArray1d(),
                    exposureFactor = combo.getJSONArray("exposureFactor").firstNumber().toFloat(),
                    neutralFilters = combo.getJSONArray("neutralFilters").toFloatArray1d()
                )
            }
            result[filmKey] = paperMap
        }
        return result
    }

    private fun JSONObject.toFloatArrayMap1d(): Map<String, FloatArray> {
        val result = mutableMapOf<String, FloatArray>()
        keys().forEach { key -> result[key] = getJSONArray(key).toFloatArray1d() }
        return result
    }

    private fun JSONObject.toFloatArrayMap2d(rows: Int, cols: Int): Map<String, FloatArray> {
        val result = mutableMapOf<String, FloatArray>()
        keys().forEach { key -> result[key] = getJSONArray(key).toFloatArray2d(rows, cols) }
        return result
    }

    private fun JSONArray.toFloatArray1d(): FloatArray {
        val result = FloatArray(length())
        for (i in 0 until length()) {
            result[i] = optDouble(i, 0.0).toFloat()
        }
        return result
    }

    private fun JSONArray.toFloatArray2d(rows: Int, cols: Int): FloatArray {
        val result = FloatArray(rows * cols)
        for (row in 0 until rows) {
            val rowArray = getJSONArray(row)
            for (col in 0 until cols) {
                result[row * cols + col] = rowArray.optDouble(col, 0.0).toFloat()
            }
        }
        return result
    }

    private fun JSONArray.firstNumber(): Double {
        var current: Any = this
        while (current is JSONArray) {
            current = current.get(0)
        }
        return (current as Number).toDouble()
    }
}

object SpectralFilmProfile {
    private const val TAG = "SpectralFilmProfile"
    private const val LUT_SIZE = 17
    private const val RAW_SPECTRAL_FILM_INPUT_SCALE = 2.88f
    private const val PREVIEW_EXPOSURE_ANCHOR_SRGB = 0.5f

    private const val DEFAULT_FILM_ASSET = "spektrafilm/profiles/kodak_portra_400_film.cube"

    @Volatile
    private var cachedDefault: SpectralFilmLut? = null

    @Volatile
    private var cachedCombined: SpectralFilmLut? = null
    private var cachedCombinedFilmKey: String? = null
    private var cachedCombinedPrintKey: String? = null

    @Volatile
    private var cachedPreviewConfig: LutConfig? = null
    private var cachedPreviewFilmKey: String? = null
    private var cachedPreviewPrintKey: String? = null

    fun loadDefaultLut(context: Context): SpectralFilmLut? {
        cachedDefault?.let { return it }
        return synchronized(this) {
            cachedDefault ?: loadCombinedLutInternal(
                context,
                "kodak_portra_400",
                "kodak_portra_endura",
                DEFAULT_FILM_ASSET
            )?.also { cachedDefault = it }
        }
    }

    fun loadCombinedLut(
        context: Context,
        filmStock: String,
        printPaper: String
    ): SpectralFilmLut? {
        if (cachedCombinedFilmKey == filmStock && cachedCombinedPrintKey == printPaper) {
            cachedCombined?.let { return it }
        }
        return synchronized(this) {
            if (cachedCombinedFilmKey == filmStock && cachedCombinedPrintKey == printPaper) {
                cachedCombined?.let { return it }
            }
            val filmAsset = "spektrafilm/profiles/${filmStock}_film.cube"
            val lut = loadCombinedLutInternal(context, filmStock, printPaper, filmAsset)
            if (lut != null) {
                cachedCombined = lut
                cachedCombinedFilmKey = filmStock
                cachedCombinedPrintKey = printPaper
            }
            lut
        }
    }

    fun loadPreviewLutConfig(
        context: Context,
        filmStock: String,
        printPaper: String
    ): LutConfig? {
        if (cachedPreviewFilmKey == filmStock && cachedPreviewPrintKey == printPaper) {
            cachedPreviewConfig?.let { return it }
        }
        return synchronized(this) {
            if (cachedPreviewFilmKey == filmStock && cachedPreviewPrintKey == printPaper) {
                cachedPreviewConfig?.let { return it }
            }
            loadCombinedLut(context, filmStock, printPaper)?.toPreviewLutConfig()?.also { config ->
                cachedPreviewConfig = config
                cachedPreviewFilmKey = filmStock
                cachedPreviewPrintKey = printPaper
            }
        }
    }

    private fun SpectralFilmLut.toPreviewLutConfig(): LutConfig {
        val rgbValues = FloatArray(size * size * size * 3)
        val srgbToProPhoto = computeWorkingToOutputTransform(ColorSpace.SRGB, ColorSpace.ProPhoto)
        val proPhotoToSrgb = computeWorkingToOutputTransform(ColorSpace.ProPhoto, ColorSpace.SRGB)
        val previewExposureGain = computePreviewExposureGain(srgbToProPhoto, proPhotoToSrgb)
        var dst = 0
        for (b in 0 until size) {
            val bEncoded = b.toFloat() / (size - 1).toFloat()
            for (g in 0 until size) {
                val gEncoded = g.toFloat() / (size - 1).toFloat()
                for (r in 0 until size) {
                    val rEncoded = r.toFloat() / (size - 1).toFloat()
                    val srgbLinear = samplePreviewLinearSrgb(
                        rEncoded,
                        gEncoded,
                        bEncoded,
                        srgbToProPhoto,
                        proPhotoToSrgb
                    )

                    rgbValues[dst++] = encodeSrgb(srgbLinear[0] * previewExposureGain)
                    rgbValues[dst++] = encodeSrgb(srgbLinear[1] * previewExposureGain)
                    rgbValues[dst++] = encodeSrgb(srgbLinear[2] * previewExposureGain)
                }
            }
        }
        return LutConfig(
            size = size,
            data = rgbValues,
            title = "Spectral Film Preview: $name",
            configDataType = LutConfig.CONFIG_DATA_TYPE_UINT16
        )
    }

    private fun SpectralFilmLut.computePreviewExposureGain(
        srgbToProPhoto: FloatArray,
        proPhotoToSrgb: FloatArray
    ): Float {
        val targetLinear = decodeSrgb(PREVIEW_EXPOSURE_ANCHOR_SRGB)
        val previewLinear = samplePreviewLinearSrgb(
            PREVIEW_EXPOSURE_ANCHOR_SRGB,
            PREVIEW_EXPOSURE_ANCHOR_SRGB,
            PREVIEW_EXPOSURE_ANCHOR_SRGB,
            srgbToProPhoto,
            proPhotoToSrgb
        )
        val previewLuma = linearSrgbLuma(previewLinear)
        return (targetLinear / previewLuma.coerceAtLeast(1e-6f)).coerceIn(0.25f, 4f)
    }

    private fun SpectralFilmLut.samplePreviewLinearSrgb(
        rEncoded: Float,
        gEncoded: Float,
        bEncoded: Float,
        srgbToProPhoto: FloatArray,
        proPhotoToSrgb: FloatArray
    ): FloatArray {
        val rLinear = decodeSrgb(rEncoded)
        val gLinear = decodeSrgb(gEncoded)
        val bLinear = decodeSrgb(bEncoded)

        val proPhotoInputR = srgbToProPhoto[0] * rLinear +
            srgbToProPhoto[1] * gLinear +
            srgbToProPhoto[2] * bLinear
        val proPhotoInputG = srgbToProPhoto[3] * rLinear +
            srgbToProPhoto[4] * gLinear +
            srgbToProPhoto[5] * bLinear
        val proPhotoInputB = srgbToProPhoto[6] * rLinear +
            srgbToProPhoto[7] * gLinear +
            srgbToProPhoto[8] * bLinear

        val spectralCoordR = encodeProPhoto(proPhotoInputR / RAW_SPECTRAL_FILM_INPUT_SCALE)
        val spectralCoordG = encodeProPhoto(proPhotoInputG / RAW_SPECTRAL_FILM_INPUT_SCALE)
        val spectralCoordB = encodeProPhoto(proPhotoInputB / RAW_SPECTRAL_FILM_INPUT_SCALE)

        val spectralEncoded = sampleSpectralLut(
            spectralCoordR,
            spectralCoordG,
            spectralCoordB
        )
        val proPhotoR = decodeProPhoto(spectralEncoded[0])
        val proPhotoG = decodeProPhoto(spectralEncoded[1])
        val proPhotoB = decodeProPhoto(spectralEncoded[2])

        return floatArrayOf(
            proPhotoToSrgb[0] * proPhotoR +
                proPhotoToSrgb[1] * proPhotoG +
                proPhotoToSrgb[2] * proPhotoB,
            proPhotoToSrgb[3] * proPhotoR +
                proPhotoToSrgb[4] * proPhotoG +
                proPhotoToSrgb[5] * proPhotoB,
            proPhotoToSrgb[6] * proPhotoR +
                proPhotoToSrgb[7] * proPhotoG +
                proPhotoToSrgb[8] * proPhotoB
        )
    }

    private fun linearSrgbLuma(rgb: FloatArray): Float {
        return rgb[0] * 0.2126f + rgb[1] * 0.7152f + rgb[2] * 0.0722f
    }

    private fun SpectralFilmLut.sampleSpectralLut(
        r: Float,
        g: Float,
        b: Float
    ): FloatArray {
        val maxIndex = size - 1
        val x = r.coerceIn(0f, 1f) * maxIndex
        val y = g.coerceIn(0f, 1f) * maxIndex
        val z = b.coerceIn(0f, 1f) * maxIndex

        val x0 = x.toInt().coerceIn(0, maxIndex)
        val y0 = y.toInt().coerceIn(0, maxIndex)
        val z0 = z.toInt().coerceIn(0, maxIndex)
        val x1 = (x0 + 1).coerceAtMost(maxIndex)
        val y1 = (y0 + 1).coerceAtMost(maxIndex)
        val z1 = (z0 + 1).coerceAtMost(maxIndex)

        val tx = x - x0
        val ty = y - y0
        val tz = z - z0

        fun valueAt(ix: Int, iy: Int, iz: Int, channel: Int): Float {
            return values[(((iz * size + iy) * size + ix) * 4) + channel]
        }

        val result = FloatArray(3)
        for (channel in 0..2) {
            val c000 = valueAt(x0, y0, z0, channel)
            val c100 = valueAt(x1, y0, z0, channel)
            val c010 = valueAt(x0, y1, z0, channel)
            val c110 = valueAt(x1, y1, z0, channel)
            val c001 = valueAt(x0, y0, z1, channel)
            val c101 = valueAt(x1, y0, z1, channel)
            val c011 = valueAt(x0, y1, z1, channel)
            val c111 = valueAt(x1, y1, z1, channel)

            val c00 = c000 * (1f - tx) + c100 * tx
            val c10 = c010 * (1f - tx) + c110 * tx
            val c01 = c001 * (1f - tx) + c101 * tx
            val c11 = c011 * (1f - tx) + c111 * tx
            val c0 = c00 * (1f - ty) + c10 * ty
            val c1 = c01 * (1f - ty) + c11 * ty
            result[channel] = c0 * (1f - tz) + c1 * tz
        }
        return result
    }

    private fun decodeSrgb(value: Float): Float {
        val clamped = value.coerceIn(0f, 1f)
        return if (clamped <= 0.04045f) {
            clamped / 12.92f
        } else {
            ((clamped + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat()
        }
    }

    private fun decodeProPhoto(value: Float): Float {
        val clamped = value.coerceIn(0f, 1f)
        return if (clamped < 0.03125f) {
            clamped / 16f
        } else {
            clamped.toDouble().pow(1.8).toFloat()
        }
    }

    private fun encodeSrgb(value: Float): Float {
        val clamped = value.coerceIn(0f, 1f)
        return if (clamped <= 0.0031308f) {
            clamped * 12.92f
        } else {
            1.055f * clamped.toDouble().pow(1.0 / 2.4).toFloat() - 0.055f
        }.coerceIn(0f, 1f)
    }

    private fun computeWorkingToOutputTransform(
        workingSpace: ColorSpace,
        outputSpace: ColorSpace
    ): FloatArray {
        val workingFromXyz = computeXyzD50ToGamut(workingSpace) ?: return identityMatrix3x3()
        val xyzFromWorking = invertMatrix3x3(workingFromXyz) ?: return identityMatrix3x3()
        val outputFromXyz = computeXyzD50ToGamut(outputSpace) ?: return identityMatrix3x3()
        return multiplyMatrix3x3(outputFromXyz, xyzFromWorking)
    }

    private fun computeXyzD50ToGamut(colorSpace: ColorSpace): FloatArray? {
        val xr = colorSpace.xr
        val yr = colorSpace.yr
        val xg = colorSpace.xg
        val yg = colorSpace.yg
        val xb = colorSpace.xb
        val yb = colorSpace.yb
        val xw = colorSpace.xw
        val yw = colorSpace.yw

        val mS = floatArrayOf(
            xr / yr, xg / yg, xb / yb,
            1f, 1f, 1f,
            (1 - xr - yr) / yr, (1 - xg - yg) / yg, (1 - xb - yb) / yb
        )
        val invS = invertMatrix3x3(mS) ?: return null

        val xWhite = xw / yw
        val yWhite = 1f
        val zWhite = (1 - xw - yw) / yw

        val sR = invS[0] * xWhite + invS[1] * yWhite + invS[2] * zWhite
        val sG = invS[3] * xWhite + invS[4] * yWhite + invS[5] * zWhite
        val sB = invS[6] * xWhite + invS[7] * yWhite + invS[8] * zWhite

        val gamutToXyzNative = floatArrayOf(
            mS[0] * sR, mS[1] * sG, mS[2] * sB,
            mS[3] * sR, mS[4] * sG, mS[5] * sB,
            mS[6] * sR, mS[7] * sG, mS[8] * sB
        )

        val bradfordD65ToD50 = floatArrayOf(
            1.0478112f, 0.0228866f, -0.0501270f,
            0.0295424f, 0.9904844f, -0.0170491f,
            -0.0092345f, 0.0150436f, 0.7521316f
        )

        val gamutToXyzD50 = if (isD50WhitePoint(xw, yw)) {
            gamutToXyzNative
        } else {
            multiplyMatrix3x3(bradfordD65ToD50, gamutToXyzNative)
        }
        return invertMatrix3x3(gamutToXyzD50)
    }

    private fun isD50WhitePoint(x: Float, y: Float): Boolean {
        return abs(x - 0.3457f) < 0.002f && abs(y - 0.3585f) < 0.002f
    }

    private fun multiplyMatrix3x3(lhs: FloatArray, rhs: FloatArray): FloatArray {
        return FloatArray(9) { index ->
            val row = index / 3
            val col = index % 3
            lhs[row * 3] * rhs[col] +
                lhs[row * 3 + 1] * rhs[3 + col] +
                lhs[row * 3 + 2] * rhs[6 + col]
        }
    }

    private fun invertMatrix3x3(matrix: FloatArray): FloatArray? {
        val det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
            matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
            matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])

        if (abs(det) < 1e-12f) return null

        val invDet = 1.0f / det
        return floatArrayOf(
            (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * invDet,
            (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * invDet,
            (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * invDet,
            (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * invDet,
            (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * invDet,
            (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * invDet,
            (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * invDet,
            (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * invDet,
            (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * invDet
        )
    }

    private fun identityMatrix3x3(): FloatArray = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    )

    private fun loadCombinedLutInternal(
        context: Context,
        filmStock: String,
        printPaper: String,
        filmAssetPath: String
    ): SpectralFilmLut? {
        val startTime = System.currentTimeMillis()
        val filmInfo = FilmStockRegistry.get(filmStock)
        val printInfo = PrintPaperRegistry.get(printPaper)

        if (filmInfo.type == "positive") {
            val positiveData = loadRawCube(context, filmAssetPath) ?: return null
            val positiveValues = packRgbLut(positiveData)
            val elapsed = System.currentTimeMillis() - startTime
            PLog.d(
                TAG,
                "Loaded positive film 1-LUT ($filmStock) from film asset without print paper ($printPaper ignored) in ${elapsed}ms"
            )
            return SpectralFilmLut(
                stock = filmStock,
                name = filmStock,
                type = filmInfo.type,
                referenceIlluminant = filmInfo.referenceIlluminant,
                viewingIlluminant = filmInfo.viewingIlluminant,
                sourceKey = "$filmStock:${filmInfo.type}:${filmInfo.referenceIlluminant}:${filmInfo.viewingIlluminant}",
                size = LUT_SIZE,
                values = positiveValues
            )
        }

        val filmData = loadRawCube(context, filmAssetPath) ?: return null
        val printModel = SpectralPrintModelRegistry.load(context) ?: return null
        val densityWire = FilmDensityWireRegistry.get(filmStock)

        val combinedValues = combineWithDynamicPrint(filmStock, printPaper, filmData, printModel, densityWire)
            ?: return null
        val elapsed = System.currentTimeMillis() - startTime
        PLog.d(
            TAG,
            "Dynamically combined film ($filmStock/${filmInfo.referenceIlluminant}) and " +
                "print ($printPaper/${printInfo.referenceIlluminant}) using Spektrafilm print model " +
                "with ${if (densityWire != null) "cmy_film wire" else "film density range fallback"} in ${elapsed}ms"
        )

        return SpectralFilmLut(
            stock = filmStock,
            name = "$filmStock + $printPaper",
            type = filmInfo.type,
            referenceIlluminant = filmInfo.referenceIlluminant,
            viewingIlluminant = printInfo.viewingIlluminant,
            sourceKey = "$filmStock:$printPaper:${filmInfo.type}:${filmInfo.referenceIlluminant}:${printInfo.referenceIlluminant}:${printInfo.viewingIlluminant}:wire=${densityWire?.dMax?.joinToString() ?: "fallback"}",
            size = LUT_SIZE,
            values = combinedValues
        )
    }

    private fun loadRawCube(context: Context, assetPath: String): FloatArray? {
        return try {
            val totalSize = LUT_SIZE * LUT_SIZE * LUT_SIZE * 3
            val values = FloatArray(totalSize)
            var dst = 0

            context.assets.open(assetPath).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).useLines { lines ->
                    for (line in lines) {
                        val trimmed = line.trim()
                        if (trimmed.isEmpty() || trimmed.startsWith("#") ||
                            trimmed.startsWith("TITLE") || trimmed.startsWith("LUT_3D_SIZE") ||
                            trimmed.startsWith("DOMAIN_MIN") || trimmed.startsWith("DOMAIN_MAX")
                        ) {
                            continue
                        }
                        val parts = trimmed.split("\\s+".toRegex())
                        if (parts.size >= 3 && dst < totalSize) {
                            values[dst++] = parts[0].toFloat()
                            values[dst++] = parts[1].toFloat()
                            values[dst++] = parts[2].toFloat()
                        }
                    }
                }
            }
            if (dst == totalSize) {
                values
            } else {
                PLog.e(TAG, "Cube file $assetPath has incorrect size: $dst elements, expected $totalSize")
                null
            }
        } catch (e: Exception) {
            PLog.e(TAG, "Failed to load raw cube from $assetPath", e)
            null
        }
    }

    private fun packRgbLut(rgbData: FloatArray): FloatArray {
        val totalSize = LUT_SIZE * LUT_SIZE * LUT_SIZE * 4
        val result = FloatArray(totalSize)
        for (i in 0 until (LUT_SIZE * LUT_SIZE * LUT_SIZE)) {
            val dstOffset = i * 4
            result[dstOffset] = rgbData[i * 3]
            result[dstOffset + 1] = rgbData[i * 3 + 1]
            result[dstOffset + 2] = rgbData[i * 3 + 2]
            result[dstOffset + 3] = 1f
        }
        return result
    }

    private fun combineWithDynamicPrint(
        filmStock: String,
        printPaper: String,
        filmData: FloatArray,
        model: SpectralFilmPrintModel,
        densityWire: FilmDensityWire?
    ): FloatArray? {
        val filmModel = model.films[filmStock]
        val paperModel = model.papers[printPaper]
        val combo = model.combinations[filmStock]?.get(printPaper)
        if (filmModel == null || paperModel == null || combo == null) {
            PLog.e(TAG, "Missing Spektrafilm print model data for $filmStock + $printPaper")
            return null
        }
        val viewingIlluminant = model.viewingIlluminants[paperModel.viewingIlluminant]
        val xyzToProPhoto = model.xyzToProPhoto[paperModel.viewingIlluminant]
        if (viewingIlluminant == null || xyzToProPhoto == null) {
            PLog.e(TAG, "Missing viewing illuminant ${paperModel.viewingIlluminant} for $printPaper")
            return null
        }

        val totalSize = LUT_SIZE * LUT_SIZE * LUT_SIZE * 4
        val combined = FloatArray(totalSize)
        val wavelengths = viewingIlluminant.size
        val paperSensitivity = FloatArray(wavelengths * 3)
        for (i in paperSensitivity.indices) {
            paperSensitivity[i] = paperModel.sensitivity[i]
        }
        val scanNormalization = computeScanNormalization(viewingIlluminant, model.observerCmfs, wavelengths)
        val dMinFilm = densityWire?.dMin ?: filmModel.densityMin
        val dMaxFilm = densityWire?.dMax ?: filmModel.densityMax
        if (densityWire == null) {
            PLog.w(TAG, "Missing cmy_film density wire for $filmStock; falling back to film profile density range")
        }
        val dRange0 = dMaxFilm[0] - dMinFilm[0]
        val dRange1 = dMaxFilm[1] - dMinFilm[1]
        val dRange2 = dMaxFilm[2] - dMinFilm[2]

        val filmSpectralDensity = FloatArray(wavelengths)
        val printDensity = FloatArray(3)
        val paperSpectralDensity = FloatArray(wavelengths)

        for (i in 0 until (LUT_SIZE * LUT_SIZE * LUT_SIZE)) {
            val cDensity = dMinFilm[0] + filmData[i * 3] * dRange0
            val mDensity = dMinFilm[1] + filmData[i * 3 + 1] * dRange1
            val yDensity = dMinFilm[2] + filmData[i * 3 + 2] * dRange2

            for (w in 0 until wavelengths) {
                val base = w * 3
                filmSpectralDensity[w] = filmModel.baseDensity[w] +
                    cDensity * filmModel.channelDensity[base] +
                    mDensity * filmModel.channelDensity[base + 1] +
                    yDensity * filmModel.channelDensity[base + 2]
            }

            for (ch in 0..2) {
                var raw = 0f
                for (w in 0 until wavelengths) {
                    raw += pow10(-filmSpectralDensity[w]) *
                        combo.printIlluminant[w] *
                        paperSensitivity[w * 3 + ch]
                }
                val logRaw = log10((raw * combo.exposureFactor).coerceAtLeast(1e-10f))
                printDensity[ch] = interpolateDensity(logRaw, paperModel.logExposure, paperModel.densityCurves, ch)
            }

            for (w in 0 until wavelengths) {
                val base = w * 3
                paperSpectralDensity[w] = paperModel.baseDensity[w] +
                    printDensity[0] * paperModel.channelDensity[base] +
                    printDensity[1] * paperModel.channelDensity[base + 1] +
                    printDensity[2] * paperModel.channelDensity[base + 2]
            }

            var x = 0f
            var y = 0f
            var z = 0f
            for (w in 0 until wavelengths) {
                val light = pow10(-paperSpectralDensity[w]) * viewingIlluminant[w]
                val cmfBase = w * 3
                x += light * model.observerCmfs[cmfBase]
                y += light * model.observerCmfs[cmfBase + 1]
                z += light * model.observerCmfs[cmfBase + 2]
            }
            x /= scanNormalization
            y /= scanNormalization
            z /= scanNormalization

            val rLinear = xyzToProPhoto[0] * x + xyzToProPhoto[1] * y + xyzToProPhoto[2] * z
            val gLinear = xyzToProPhoto[3] * x + xyzToProPhoto[4] * y + xyzToProPhoto[5] * z
            val bLinear = xyzToProPhoto[6] * x + xyzToProPhoto[7] * y + xyzToProPhoto[8] * z

            val dstOffset = i * 4
            combined[dstOffset] = encodeProPhoto(rLinear)
            combined[dstOffset + 1] = encodeProPhoto(gLinear)
            combined[dstOffset + 2] = encodeProPhoto(bLinear)
            combined[dstOffset + 3] = 1f
        }
        return combined
    }

    private fun computeScanNormalization(
        illuminant: FloatArray,
        observerCmfs: FloatArray,
        wavelengths: Int
    ): Float {
        var normalization = 0f
        for (w in 0 until wavelengths) {
            normalization += illuminant[w] * observerCmfs[w * 3 + 1]
        }
        return normalization.coerceAtLeast(1e-10f)
    }

    private fun interpolateDensity(
        logRaw: Float,
        logExposure: FloatArray,
        densityCurves: FloatArray,
        channel: Int
    ): Float {
        if (logRaw <= logExposure.first()) return densityCurves[channel]
        val lastIndex = logExposure.lastIndex
        if (logRaw >= logExposure[lastIndex]) return densityCurves[lastIndex * 3 + channel]
        var high = 1
        while (high < logExposure.size && logRaw > logExposure[high]) {
            high++
        }
        val low = high - 1
        val t = ((logRaw - logExposure[low]) / (logExposure[high] - logExposure[low])).coerceIn(0f, 1f)
        val y0 = densityCurves[low * 3 + channel]
        val y1 = densityCurves[high * 3 + channel]
        return y0 * (1f - t) + y1 * t
    }

    private fun pow10(value: Float): Float = 10.0.pow(value.toDouble()).toFloat()

    private fun log10(value: Float): Float = (ln(value.toDouble()) / LN_10).toFloat()

    private fun encodeProPhoto(value: Float): Float {
        val clamped = value.coerceAtLeast(0f)
        val encoded = if (clamped < 0.001953125f) {
            clamped * 16f
        } else {
            clamped.toDouble().pow(1.0 / 1.8).toFloat()
        }
        return encoded.coerceIn(0f, 1f)
    }

    private const val LN_10 = 2.302585092994046
}
