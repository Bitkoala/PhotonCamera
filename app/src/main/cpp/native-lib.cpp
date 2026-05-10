/**
 * native-lib.cpp
 *
 * JNI Interface for PhotonCamera native functions.
 */
#include <algorithm>
#include <android/bitmap.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <jni.h>
#include <limits>
#include <map>
#include <omp.h>
#include <string>
#include <turbojpeg.h>
#include <vector>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>

#include "common.h"
#include "jxl_utils.h"
#include "libraw/libraw.h"
#include "math_utils.h"
#include "stacking_utils.h"
#include "vulkan_raw_stacker.h"
#include "vulkan_stacker.h"
#include <android/hardware_buffer_jni.h>

#ifndef LOG_TAG
#define LOG_TAG "native-lib"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#endif

static jbyteArray encodeBitmapThumbnailToJpeg(JNIEnv *env,
                                              const libraw_processed_image_t *thumb,
                                              int quality = 90) {
  if (!thumb || !thumb->data || thumb->width <= 0 || thumb->height <= 0) {
    return nullptr;
  }

  if (thumb->colors != 1 && thumb->colors != 3) {
    LOGI("encodeBitmapThumbnailToJpeg: unsupported colors=%d", thumb->colors);
    return nullptr;
  }

  const int pixelFormat = thumb->colors == 3 ? TJPF_RGB : TJPF_GRAY;
  const int rowStride = thumb->width * thumb->colors;
  std::vector<unsigned char> converted;
  const unsigned char *src = thumb->data;

  if (thumb->bits == 16) {
    converted.resize(static_cast<size_t>(rowStride) * thumb->height);
    const auto *src16 = reinterpret_cast<const unsigned short *>(thumb->data);
    const size_t sampleCount =
        static_cast<size_t>(thumb->width) * thumb->height * thumb->colors;
    for (size_t i = 0; i < sampleCount; ++i) {
      converted[i] = static_cast<unsigned char>(src16[i] >> 8);
    }
    src = converted.data();
  } else if (thumb->bits != 8) {
    LOGI("encodeBitmapThumbnailToJpeg: unsupported bits=%d", thumb->bits);
    return nullptr;
  }

  tjhandle handle = tjInitCompress();
  if (!handle) {
    LOGE("encodeBitmapThumbnailToJpeg: tjInitCompress failed: %s",
         tjGetErrorStr());
    return nullptr;
  }

  unsigned char *jpegBuf = nullptr;
  unsigned long jpegSize = 0;
  const int ret =
      tjCompress2(handle, src, thumb->width, rowStride, thumb->height,
                  pixelFormat, &jpegBuf, &jpegSize, TJSAMP_420, quality,
                  TJFLAG_NOREALLOC | TJFLAG_FASTDCT);
  if (ret != 0 || !jpegBuf || jpegSize == 0) {
    LOGE("encodeBitmapThumbnailToJpeg: tjCompress2 failed: %s",
         tjGetErrorStr());
    if (jpegBuf) {
      tjFree(jpegBuf);
    }
    tjDestroy(handle);
    return nullptr;
  }

  jbyteArray result = env->NewByteArray(static_cast<jsize>(jpegSize));
  env->SetByteArrayRegion(result, 0, static_cast<jsize>(jpegSize),
                          reinterpret_cast<const jbyte *>(jpegBuf));
  tjFree(jpegBuf);
  tjDestroy(handle);
  return result;
}

static jobject createArgb8888Bitmap(JNIEnv *env, int width, int height) {
  jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
  jclass configClass = env->FindClass("android/graphics/Bitmap$Config");
  jfieldID argb8888Field =
      env->GetStaticFieldID(configClass, "ARGB_8888",
                            "Landroid/graphics/Bitmap$Config;");
  jobject argb8888 = env->GetStaticObjectField(configClass, argb8888Field);
  jmethodID createBitmapMethod =
      env->GetStaticMethodID(bitmapClass, "createBitmap",
                             "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
  return env->CallStaticObjectMethod(bitmapClass, createBitmapMethod, width,
                                     height, argb8888);
}

static jobject createBitmapFromRgba(JNIEnv *env, int width, int height,
                                    const unsigned char *rgbaData,
                                    int strideBytes) {
  if (!rgbaData || width <= 0 || height <= 0) {
    return nullptr;
  }

  jobject bitmap = createArgb8888Bitmap(env, width, height);
  if (!bitmap) {
    LOGE("createBitmapFromRgba: failed to allocate Bitmap %dx%d", width,
         height);
    return nullptr;
  }

  void *pixels = nullptr;
  if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
      !pixels) {
    LOGE("createBitmapFromRgba: failed to lock pixels");
    return nullptr;
  }

  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
    AndroidBitmap_unlockPixels(env, bitmap);
    LOGE("createBitmapFromRgba: failed to query bitmap info");
    return nullptr;
  }

  const auto *src = rgbaData;
  auto *dst = reinterpret_cast<unsigned char *>(pixels);
  for (int y = 0; y < height; ++y) {
    std::memcpy(dst + y * info.stride, src + y * strideBytes,
                static_cast<size_t>(width) * 4);
  }

  AndroidBitmap_unlockPixels(env, bitmap);
  return bitmap;
}

static jobject decodeJpegPreviewToBitmap(JNIEnv *env, const unsigned char *jpegData,
                                         unsigned long jpegSize) {
  if (!jpegData || jpegSize == 0) {
    return nullptr;
  }

  tjhandle handle = tjInitDecompress();
  if (!handle) {
    LOGE("decodeJpegPreviewToBitmap: tjInitDecompress failed: %s",
         tjGetErrorStr());
    return nullptr;
  }

  int width = 0;
  int height = 0;
  int subsamp = 0;
  auto *mutableJpegData = const_cast<unsigned char *>(jpegData);
  if (tjDecompressHeader2(handle, mutableJpegData, jpegSize, &width, &height,
                          &subsamp) != 0) {
    LOGE("decodeJpegPreviewToBitmap: header decode failed: %s",
         tjGetErrorStr());
    tjDestroy(handle);
    return nullptr;
  }

  std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);
  if (tjDecompress2(handle, mutableJpegData, jpegSize, rgba.data(), width, width * 4,
                    height, TJPF_RGBA, TJFLAG_FASTUPSAMPLE |
                                            TJFLAG_FASTDCT) != 0) {
    LOGE("decodeJpegPreviewToBitmap: jpeg decode failed: %s", tjGetErrorStr());
    tjDestroy(handle);
    return nullptr;
  }

  tjDestroy(handle);
  return createBitmapFromRgba(env, width, height, rgba.data(), width * 4);
}

static jobject convertBitmapThumbnailToBitmap(JNIEnv *env,
                                              const libraw_processed_image_t *thumb) {
  if (!thumb || !thumb->data || thumb->width <= 0 || thumb->height <= 0) {
    return nullptr;
  }

  if (thumb->colors != 1 && thumb->colors != 3) {
    LOGI("convertBitmapThumbnailToBitmap: unsupported colors=%d", thumb->colors);
    return nullptr;
  }

  const size_t pixelCount = static_cast<size_t>(thumb->width) * thumb->height;
  std::vector<unsigned char> rgba(pixelCount * 4);

  if (thumb->bits == 8) {
    for (size_t i = 0; i < pixelCount; ++i) {
      const size_t srcIndex = i * thumb->colors;
      const unsigned char r = thumb->colors == 3 ? thumb->data[srcIndex] : thumb->data[i];
      const unsigned char g = thumb->colors == 3 ? thumb->data[srcIndex + 1] : thumb->data[i];
      const unsigned char b = thumb->colors == 3 ? thumb->data[srcIndex + 2] : thumb->data[i];
      const size_t dstIndex = i * 4;
      rgba[dstIndex] = r;
      rgba[dstIndex + 1] = g;
      rgba[dstIndex + 2] = b;
      rgba[dstIndex + 3] = 255;
    }
  } else if (thumb->bits == 16) {
    const auto *src16 = reinterpret_cast<const unsigned short *>(thumb->data);
    for (size_t i = 0; i < pixelCount; ++i) {
      const size_t srcIndex = i * thumb->colors;
      const unsigned char r =
          static_cast<unsigned char>((thumb->colors == 3 ? src16[srcIndex] : src16[i]) >> 8);
      const unsigned char g = static_cast<unsigned char>(
          (thumb->colors == 3 ? src16[srcIndex + 1] : src16[i]) >> 8);
      const unsigned char b = static_cast<unsigned char>(
          (thumb->colors == 3 ? src16[srcIndex + 2] : src16[i]) >> 8);
      const size_t dstIndex = i * 4;
      rgba[dstIndex] = r;
      rgba[dstIndex + 1] = g;
      rgba[dstIndex + 2] = b;
      rgba[dstIndex + 3] = 255;
    }
  } else {
    LOGI("convertBitmapThumbnailToBitmap: unsupported bits=%d", thumb->bits);
    return nullptr;
  }

  return createBitmapFromRgba(env, thumb->width, thumb->height, rgba.data(),
                              thumb->width * 4);
}

struct Matrix3x3 {
  float m[9];

  Matrix3x3() {
    for (int i = 0; i < 9; i++)
      m[i] = 0;
  }

  static Matrix3x3 identity() {
    Matrix3x3 res;
    res.m[0] = res.m[4] = res.m[8] = 1.0f;
    return res;
  }

  Matrix3x3 multiply(const Matrix3x3 &other) const {
    Matrix3x3 res;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        res.m[i * 3 + j] = m[i * 3 + 0] * other.m[0 * 3 + j] +
                           m[i * 3 + 1] * other.m[1 * 3 + j] +
                           m[i * 3 + 2] * other.m[2 * 3 + j];
      }
    }
    return res;
  }

  Matrix3x3 invert() const {
    float det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                m[1] * (m[3] * m[8] - m[5] * m[6]) +
                m[2] * (m[3] * m[7] - m[4] * m[6]);

    if (std::abs(det) < 1e-12f)
      return identity();

    float invDet = 1.0f / det;
    Matrix3x3 res;
    res.m[0] = (m[4] * m[8] - m[5] * m[7]) * invDet;
    res.m[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
    res.m[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
    res.m[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
    res.m[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
    res.m[5] = (m[2] * m[3] - m[0] * m[5]) * invDet;
    res.m[6] = (m[3] * m[7] - m[4] * m[6]) * invDet;
    res.m[7] = (m[1] * m[6] - m[0] * m[7]) * invDet;
    res.m[8] = (m[0] * m[4] - m[1] * m[3]) * invDet;
    return res;
  }
};


struct DngGainMap {
  uint32_t top = 0;
  uint32_t left = 0;
  uint32_t bottom = 0;
  uint32_t right = 0;
  uint32_t plane = 0;
  uint32_t planes = 0;
  uint32_t rowPitch = 0;
  uint32_t colPitch = 0;
  uint32_t mapPointsV = 0;
  uint32_t mapPointsH = 0;
  double mapSpacingV = 0.0;
  double mapSpacingH = 0.0;
  double mapOriginV = 0.0;
  double mapOriginH = 0.0;
  uint32_t mapPlanes = 0;
  std::vector<float> mapGain;
};

static float illuminantToTemp(int illuminant);

static bool parseDngGainMapOpcode(const uint8_t *data, size_t size, size_t &offset,
                                  DngGainMap &gainMap) {
  if (offset + 12 > size) {
    return false;
  }
  offset += 12; // skip opcode version, flags and payload size

  auto readUInt = [&](uint32_t &out) -> bool {
    if (offset + 4 > size) {
      return false;
    }
    out = readBigEndian<uint32_t>(data + offset);
    offset += 4;
    return true;
  };
  auto readDouble = [&](double &out) -> bool {
    if (offset + 8 > size) {
      return false;
    }
    out = readBigEndian<double>(data + offset);
    offset += 8;
    return true;
  };
  auto readFloat = [&](float &out) -> bool {
    if (offset + 4 > size) {
      return false;
    }
    out = readBigEndian<float>(data + offset);
    offset += 4;
    return true;
  };

  if (!readUInt(gainMap.top) || !readUInt(gainMap.left) || !readUInt(gainMap.bottom) ||
      !readUInt(gainMap.right) || !readUInt(gainMap.plane) || !readUInt(gainMap.planes) ||
      !readUInt(gainMap.rowPitch) || !readUInt(gainMap.colPitch) ||
      !readUInt(gainMap.mapPointsV) || !readUInt(gainMap.mapPointsH) ||
      !readDouble(gainMap.mapSpacingV) || !readDouble(gainMap.mapSpacingH) ||
      !readDouble(gainMap.mapOriginV) || !readDouble(gainMap.mapOriginH) ||
      !readUInt(gainMap.mapPlanes)) {
    return false;
  }

  const size_t count = static_cast<size_t>(gainMap.mapPointsV) *
                       static_cast<size_t>(gainMap.mapPointsH) *
                       static_cast<size_t>(gainMap.mapPlanes);
  gainMap.mapGain.resize(count);
  for (size_t i = 0; i < count; ++i) {
    if (!readFloat(gainMap.mapGain[i])) {
      return false;
    }
  }
  return true;
}

static bool parseDngGainMaps(const libraw_dng_rawopcode_t &opcodeData,
                             std::vector<DngGainMap> &gainMaps) {
  gainMaps.clear();
  if (!opcodeData.data || opcodeData.len < 4) {
    return false;
  }

  const auto *bytes = static_cast<const uint8_t *>(opcodeData.data);
  size_t offset = 0;
  const uint32_t opcodeCount = readBigEndian<uint32_t>(bytes + offset);
  offset += 4;

  for (uint32_t i = 0; i < opcodeCount && offset + 4 <= opcodeData.len; ++i) {
    const uint32_t opcode = readBigEndian<uint32_t>(bytes + offset);
    offset += 4;
    if (opcode == 9 && gainMaps.size() < 4) {
      DngGainMap gainMap;
      if (!parseDngGainMapOpcode(bytes, opcodeData.len, offset, gainMap)) {
        return false;
      }
      gainMaps.push_back(std::move(gainMap));
    } else {
      if (offset + 12 > opcodeData.len) {
        return false;
      }
      offset += 8;
      const uint32_t payloadSize = readBigEndian<uint32_t>(bytes + offset);
      offset += 4;
      if (offset + payloadSize > opcodeData.len) {
        return false;
      }
      offset += payloadSize;
    }
  }

  return !gainMaps.empty();
}

static float sampleDngGainMapBilinear(const DngGainMap &gainMap, float x, float y,
                                      int plane = 0) {
  const float maxX = static_cast<float>(std::max<int>(0, gainMap.mapPointsH - 1));
  const float maxY = static_cast<float>(std::max<int>(0, gainMap.mapPointsV - 1));
  const float fx = std::clamp(x, 0.0f, maxX);
  const float fy = std::clamp(y, 0.0f, maxY);
  const int mapPlane =
      std::clamp(plane, 0, static_cast<int>(std::max<uint32_t>(1, gainMap.mapPlanes)) - 1);

  const int x0 = static_cast<int>(std::floor(fx));
  const int y0 = static_cast<int>(std::floor(fy));
  const int x1 = std::min(x0 + 1, static_cast<int>(gainMap.mapPointsH) - 1);
  const int y1 = std::min(y0 + 1, static_cast<int>(gainMap.mapPointsV) - 1);
  const float tx = fx - static_cast<float>(x0);
  const float ty = fy - static_cast<float>(y0);

  auto at = [&](int sx, int sy) -> float {
    const size_t index =
        (static_cast<size_t>(sy) * gainMap.mapPointsH + static_cast<size_t>(sx)) *
            gainMap.mapPlanes +
        static_cast<size_t>(mapPlane);
    return gainMap.mapGain[index];
  };

  const float v00 = at(x0, y0);
  const float v10 = at(x1, y0);
  const float v01 = at(x0, y1);
  const float v11 = at(x1, y1);
  const float top = v00 + (v10 - v00) * tx;
  const float bottom = v01 + (v11 - v01) * tx;
  return top + (bottom - top) * ty;
}

static float sampleDngGainMapAtPixel(const DngGainMap &gainMap, int x, int y,
                                     int rawWidth, int rawHeight,
                                     int plane = 0) {
  const double spacingH = gainMap.mapSpacingH > 0.0 ? gainMap.mapSpacingH : 1.0;
  const double spacingV = gainMap.mapSpacingV > 0.0 ? gainMap.mapSpacingV : 1.0;
  const double relX =
      (static_cast<double>(x) + 0.5) / static_cast<double>(std::max(1, rawWidth));
  const double relY =
      (static_cast<double>(y) + 0.5) / static_cast<double>(std::max(1, rawHeight));
  const float mapX = static_cast<float>((relX - gainMap.mapOriginH) / spacingH);
  const float mapY = static_cast<float>((relY - gainMap.mapOriginV) / spacingV);
  return sampleDngGainMapBilinear(gainMap, mapX, mapY, plane);
}

static void computeEffectiveBlackLevels(LibRaw &rawProcessor, float out[4]) {
  const float base = static_cast<float>(rawProcessor.imgdata.color.black);
  const unsigned cols = rawProcessor.imgdata.color.cblack[4];
  const unsigned rows = rawProcessor.imgdata.color.cblack[5];
  for (int ch = 0; ch < 4; ++ch) {
    float total = base + static_cast<float>(rawProcessor.imgdata.color.cblack[ch]);
    // Add the repeat pattern contribution for this CFA channel.
    if (cols > 0 && rows > 0) {
      float sum = 0.0f;
      int count = 0;
      for (unsigned r = 0; r < rows; ++r) {
        for (unsigned c = 0; c < cols; ++c) {
          if (rawProcessor.FC(r, c) == ch) {
            sum += static_cast<float>(
                rawProcessor.imgdata.color.cblack[6 + r * cols + c]);
            ++count;
          }
        }
      }
      if (count > 0) {
        total += sum / static_cast<float>(count);
      }
    }
    out[ch] = total;
  }
}


// Metadata-based black level calculation is now handled by computeEffectiveBlackLevels.


static bool applySupportedDngGainMaps(LibRaw &rawProcessor, const float blackLevels[4]) {
  if (!rawProcessor.imgdata.rawdata.raw_image || !rawProcessor.imgdata.idata.filters) {
    return false;
  }

  std::vector<DngGainMap> gainMaps;
  if (!parseDngGainMaps(rawProcessor.imgdata.color.dng_levels.rawopcodes[1], gainMaps)) {
    return false;
  }

  if (gainMaps.size() != 4) {
    LOGI("dng gain map: unsupported map count=%zu", gainMaps.size());
    return false;
  }

  unsigned check = 0;
  bool isNoOp = true;
  for (const auto &gainMap : gainMaps) {
    if (gainMap.rowPitch != 2 || gainMap.colPitch != 2 || gainMap.mapPlanes != 1 ||
        gainMap.mapGain.empty() ||
        gainMap.mapGain.size() != static_cast<size_t>(gainMap.mapPointsV) *
                                     static_cast<size_t>(gainMap.mapPointsH) *
                                     static_cast<size_t>(gainMap.mapPlanes)) {
      LOGI("dng gain map: unsupported layout top=%u left=%u rowPitch=%u colPitch=%u mapPlanes=%u size=%zu",
           gainMap.top, gainMap.left, gainMap.rowPitch, gainMap.colPitch, gainMap.mapPlanes,
           gainMap.mapGain.size());
      return false;
    }

    if ((gainMap.top & 1u) == 0u) {
      check += ((gainMap.left & 1u) == 0u) ? 1u : 2u;
    } else {
      check += ((gainMap.left & 1u) == 0u) ? 4u : 8u;
    }

    for (float value : gainMap.mapGain) {
      if (std::abs(value - 1.0f) > 1e-6f) {
        isNoOp = false;
        break;
      }
    }
  }

  if (isNoOp || check != 15u) {
    LOGI("dng gain map: unsupported combination noop=%d check=%u", isNoOp ? 1 : 0, check);
    return false;
  }

  const int rawWidth = rawProcessor.imgdata.sizes.raw_width;
  const int rawHeight = rawProcessor.imgdata.sizes.raw_height;
  const int rawPitch = rawProcessor.imgdata.sizes.raw_pitch > 0
                           ? rawProcessor.imgdata.sizes.raw_pitch / static_cast<int>(sizeof(ushort))
                           : rawWidth;
  auto *rawImage = rawProcessor.imgdata.rawdata.raw_image;

  const DngGainMap *mapByParity[2][2] = {};
  for (const auto &gainMap : gainMaps) {
    mapByParity[gainMap.top & 1u][gainMap.left & 1u] = &gainMap;
    LOGI("dng gain map: rect=%u,%u-%u,%u pitch=%u,%u points=%ux%u spacing=%f,%f origin=%f,%f planes=%u",
         gainMap.left, gainMap.top, gainMap.right, gainMap.bottom,
         gainMap.colPitch, gainMap.rowPitch, gainMap.mapPointsH,
         gainMap.mapPointsV, gainMap.mapSpacingH, gainMap.mapSpacingV,
         gainMap.mapOriginH, gainMap.mapOriginV, gainMap.mapPlanes);
  }

  double gainSum = 0.0;
  float gainMin = std::numeric_limits<float>::max();
  float gainMax = 0.0f;
  uint64_t gainCount = 0;
  for (int y = 0; y < rawHeight; ++y) {
    const float rowBlack[2] = {blackLevels[rawProcessor.FC(y, 0)], blackLevels[rawProcessor.FC(y, 1)]};
    for (int x = 0; x < rawWidth; ++x) {
      const DngGainMap *gainMap = mapByParity[y & 1][x & 1];
      if (!gainMap) {
        continue;
      }
      if (static_cast<uint32_t>(y) < gainMap->top ||
          static_cast<uint32_t>(y) >= gainMap->bottom ||
          static_cast<uint32_t>(x) < gainMap->left ||
          static_cast<uint32_t>(x) >= gainMap->right ||
          ((static_cast<uint32_t>(y) - gainMap->top) % gainMap->rowPitch) != 0u ||
          ((static_cast<uint32_t>(x) - gainMap->left) % gainMap->colPitch) != 0u) {
        continue;
      }
      const int mapPlane =
          std::max(0, rawProcessor.FC(y, x) - static_cast<int>(gainMap->plane));
      const float gain =
          sampleDngGainMapAtPixel(*gainMap, x, y, rawWidth, rawHeight, mapPlane);
      const float black = rowBlack[x & 1];
      float corrected = (static_cast<float>(rawImage[y * rawPitch + x]) - black) * gain + black;
      corrected = std::max(corrected, 0.0f);
      rawImage[y * rawPitch + x] = static_cast<ushort>(std::min(corrected, 65535.0f));
      gainSum += static_cast<double>(gain);
      gainMin = std::min(gainMin, gain);
      gainMax = std::max(gainMax, gain);
      ++gainCount;
    }
  }

  LOGI("dng gain map: applied 4 maps size=%ux%u grid=%ux%u gain avg=%f min=%f max=%f count=%llu",
       rawWidth, rawHeight, gainMaps[0].mapPointsH, gainMaps[0].mapPointsV,
       gainCount > 0 ? static_cast<float>(gainSum / static_cast<double>(gainCount)) : 0.0f,
       gainCount > 0 ? gainMin : 0.0f, gainMax,
       static_cast<unsigned long long>(gainCount));
  return true;
}

static void logProcessedImageStats(const char *label,
                                   const libraw_processed_image_t *image) {
  if (!image || !image->data || image->width <= 0 || image->height <= 0 ||
      image->colors < 3 || image->bits != 16) {
    return;
  }

  const auto *pixels = reinterpret_cast<const ushort *>(image->data);
  const size_t pixelCount =
      static_cast<size_t>(image->width) * static_cast<size_t>(image->height);
  double sumR = 0.0;
  double sumG = 0.0;
  double sumB = 0.0;
  uint64_t clipped = 0;
  for (size_t i = 0; i < pixelCount; ++i) {
    const ushort r = pixels[i * image->colors];
    const ushort g = pixels[i * image->colors + 1];
    const ushort b = pixels[i * image->colors + 2];
    sumR += r;
    sumG += g;
    sumB += b;
    if (r >= 65535 || g >= 65535 || b >= 65535) {
      ++clipped;
    }
  }

  const double invCount = pixelCount > 0 ? 1.0 / static_cast<double>(pixelCount) : 0.0;
  LOGI("%s rgb16 mean R=%f G=%f B=%f luma=%f clipped=%llu/%llu",
       label, static_cast<float>(sumR * invCount),
       static_cast<float>(sumG * invCount), static_cast<float>(sumB * invCount),
       static_cast<float>((sumR * 0.2126 + sumG * 0.7152 + sumB * 0.0722) *
                          invCount),
       static_cast<unsigned long long>(clipped),
       static_cast<unsigned long long>(pixelCount));
}

static bool estimateRawAutoWhiteBalance(LibRaw &rawProcessor, const float blackLevels[4],
                                        float outUserMul[4], int &sampleCount) {
  sampleCount = 0;
  if (!rawProcessor.imgdata.rawdata.raw_image || !rawProcessor.imgdata.idata.filters) {
    return false;
  }

  const int rawWidth = rawProcessor.imgdata.sizes.raw_width;
  const int rawHeight = rawProcessor.imgdata.sizes.raw_height;
  const int rawPitch = rawProcessor.imgdata.sizes.raw_pitch > 0
                           ? rawProcessor.imgdata.sizes.raw_pitch / static_cast<int>(sizeof(ushort))
                           : rawWidth;
  const int left = std::max(0, static_cast<int>(rawProcessor.imgdata.sizes.left_margin));
  const int top = std::max(0, static_cast<int>(rawProcessor.imgdata.sizes.top_margin));
  const int right = std::min(rawWidth - 1, left + static_cast<int>(rawProcessor.imgdata.sizes.width) - 1);
  const int bottom = std::min(rawHeight - 1, top + static_cast<int>(rawProcessor.imgdata.sizes.height) - 1);
  if (right <= left + 1 || bottom <= top + 1) {
    return false;
  }

  float whiteLevel = static_cast<float>(rawProcessor.imgdata.color.dng_levels.dng_whitelevel[0]);
  if (whiteLevel <= 0.0f) {
    whiteLevel = static_cast<float>(rawProcessor.imgdata.color.maximum);
  }
  if (whiteLevel <= 0.0f) {
    whiteLevel = 65535.0f;
  }

  double sums[4] = {0.0, 0.0, 0.0, 0.0};
  auto *rawImage = rawProcessor.imgdata.rawdata.raw_image;
  const int step = 4;
  for (int y = top; y + 1 <= bottom; y += step) {
    for (int x = left; x + 1 <= right; x += step) {
      float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      bool seen[4] = {false, false, false, false};
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const int yy = y + dy;
          const int xx = x + dx;
          int c = rawProcessor.FC(yy, xx);
          if (c < 0 || c > 3) {
            continue;
          }
          const float black = blackLevels[c];
          const float range = std::max(1.0f, whiteLevel - black);
          const float raw = static_cast<float>(rawImage[yy * rawPitch + xx]);
          v[c] = std::clamp((raw - black) / range, 0.0f, 1.0f);
          seen[c] = true;
        }
      }

      if (!seen[0] || !seen[1] || !seen[2]) {
        continue;
      }
      if (!seen[3]) {
        v[3] = v[1];
      }

      const float g = 0.5f * (v[1] + v[3]);
      const float minValue = std::min(std::min(v[0], v[1]), std::min(v[2], v[3]));
      const float maxValue = std::max(std::max(v[0], v[1]), std::max(v[2], v[3]));
      const float luma = 0.25f * (v[0] + v[1] + v[2] + v[3]);
      if (luma < 0.025f || luma > 0.82f || minValue < 0.004f || maxValue > 0.96f) {
        continue;
      }
      if (maxValue / std::max(minValue, 0.004f) > 3.0f || g <= 0.0f) {
        continue;
      }

      sums[0] += v[0];
      sums[1] += v[1];
      sums[2] += v[2];
      sums[3] += v[3];
      ++sampleCount;
    }
  }

  if (sampleCount < 512 || sums[0] <= 0.0 || sums[1] <= 0.0 ||
      sums[2] <= 0.0 || sums[3] <= 0.0) {
    return false;
  }

  const float r = static_cast<float>(sums[0] / sampleCount);
  const float gr = static_cast<float>(sums[1] / sampleCount);
  const float b = static_cast<float>(sums[2] / sampleCount);
  const float gb = static_cast<float>(sums[3] / sampleCount);
  const float g = 0.5f * (gr + gb);
  outUserMul[0] = std::clamp(g / std::max(r, 1e-4f), 0.5f, 4.0f);
  outUserMul[1] = 1.0f;
  outUserMul[2] = std::clamp(g / std::max(b, 1e-4f), 0.5f, 4.0f);
  outUserMul[3] = std::clamp(g / std::max(gb, 1e-4f), 0.75f, 1.33f);
  return std::isfinite(outUserMul[0]) && std::isfinite(outUserMul[2]) &&
         std::isfinite(outUserMul[3]);
}

static float illuminantToTemp(int illuminant) {
  switch (illuminant) {
  case 1:
    return 5500.0f;
  case 2:
    return 4150.0f;
  case 3:
    return 2850.0f;
  case 4:
    return 5500.0f;
  case 9:
    return 5500.0f;
  case 10:
    return 6500.0f;
  case 11:
    return 7500.0f;
  case 12:
    return 6400.0f;
  case 13:
    return 5050.0f;
  case 14:
    return 4150.0f;
  case 15:
    return 3525.0f;
  case 16:
    return 2925.0f;
  case 17:
    return 2850.0f;
  case 18:
    return 5500.0f;
  case 19:
    return 6500.0f;
  case 20:
    return 5500.0f;
  case 21:
    return 6500.0f;
  case 22:
    return 7500.0f;
  case 23:
    return 5000.0f;
  case 24:
    return 3200.0f;
  default:
    return 0.0f;
  }
}

static Matrix3x3 computeXYZD50ToGamut(float xr, float yr, float xg, float yg,
                                      float xb, float yb, float xw, float yw) {

  Matrix3x3 mS;
  mS.m[0] = xr / yr;
  mS.m[1] = xg / yg;
  mS.m[2] = xb / yb;
  mS.m[3] = 1.0f;
  mS.m[4] = 1.0f;
  mS.m[5] = 1.0f;
  mS.m[6] = (1 - xr - yr) / yr;
  mS.m[7] = (1 - xg - yg) / yg;
  mS.m[8] = (1 - xb - yb) / yb;

  Matrix3x3 invS = mS.invert();
  float Xw = xw / yw, Yw = 1.0f, Zw = (1 - xw - yw) / yw;
  float sR = invS.m[0] * Xw + invS.m[1] * Yw + invS.m[2] * Zw;
  float sG = invS.m[3] * Xw + invS.m[4] * Yw + invS.m[5] * Zw;
  float sB = invS.m[6] * Xw + invS.m[7] * Yw + invS.m[8] * Zw;

  Matrix3x3 gamutToXYZNative;
  gamutToXYZNative.m[0] = mS.m[0] * sR;
  gamutToXYZNative.m[1] = mS.m[1] * sG;
  gamutToXYZNative.m[2] = mS.m[2] * sB;
  gamutToXYZNative.m[3] = mS.m[3] * sR;
  gamutToXYZNative.m[4] = mS.m[4] * sG;
  gamutToXYZNative.m[5] = mS.m[5] * sB;
  gamutToXYZNative.m[6] = mS.m[6] * sR;
  gamutToXYZNative.m[7] = mS.m[7] * sG;
  gamutToXYZNative.m[8] = mS.m[8] * sB;

  float BRADFORD_D65_TO_D50[9] = {1.0478112f,  0.0228866f, -0.0501270f,
                                  0.0295424f,  0.9904844f, -0.0170491f,
                                  -0.0092345f, 0.0150436f, 0.7521316f};
  Matrix3x3 bMat;
  for (int i = 0; i < 9; i++)
    bMat.m[i] = BRADFORD_D65_TO_D50[i];

  const bool isD50WhitePoint = std::abs(xw - 0.3457f) < 0.002f &&
                               std::abs(yw - 0.3585f) < 0.002f;
  Matrix3x3 gamutToXYZD50 =
      isD50WhitePoint ? gamutToXYZNative : bMat.multiply(gamutToXYZNative);
  return gamutToXYZD50.invert();
}

static inline float hlgToLinear(float value) {
  constexpr float kHlgA = 0.17883277f;
  constexpr float kHlgB = 0.28466892f;
  constexpr float kHlgC = 0.55991073f;
  constexpr float kHdrReferenceScale = 6.0f;

  const float v = std::max(0.0f, value);
  const float linear = (v <= 0.5f)
                           ? (v * v) / 3.0f
                           : static_cast<float>(
                                 (std::exp((v - kHlgC) / kHlgA) + kHlgB) /
                                 12.0f);
  return linear * kHdrReferenceScale;
}

static unsigned char mapCfaPatternToLibRaw(int cfaPattern) {
  switch (cfaPattern) {
  case 0:
    return LIBRAW_OPENBAYER_RGGB;
  case 1:
    return LIBRAW_OPENBAYER_GRBG;
  case 2:
    return LIBRAW_OPENBAYER_GBRG;
  case 3:
    return LIBRAW_OPENBAYER_BGGR;
  default:
    return 0;
  }
}

extern "C" {

/**
 * Multi-Frame Stacking JNI Interface
 */
JNIEXPORT jlong JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_createStackerNative(
    JNIEnv *env, jobject /* this */, jint width, jint height,
    jboolean enableSuperRes) {
  auto *stacker = new ImageStacker(width, height, enableSuperRes);
  return reinterpret_cast<jlong>(stacker);
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_addToStackNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject yBuffer,
    jobject uBuffer, jobject vBuffer, jint yRowStride, jint uvRowStride,
    jint uvPixelStride, jint format) {

  auto *stacker = reinterpret_cast<ImageStacker *>(stackerPtr);
  if (!stacker)
    return;

  auto *yData = static_cast<uint8_t *>(env->GetDirectBufferAddress(yBuffer));
  auto *uData = static_cast<uint8_t *>(env->GetDirectBufferAddress(uBuffer));
  auto *vData = static_cast<uint8_t *>(env->GetDirectBufferAddress(vBuffer));

  if (yData && uData && vData) {
    jlong yCap = env->GetDirectBufferCapacity(yBuffer);
    jlong uCap = env->GetDirectBufferCapacity(uBuffer);
    jlong vCap = env->GetDirectBufferCapacity(vBuffer);

    // Basic sanity check for capacity. Actual check depends on strides,
    // but at least check it's not empty.
    if (yCap <= 0 || uCap <= 0 || vCap <= 0) {
      LOGE("addToStackNative: Buffer capacity is zero");
      return;
    }

    stacker->addFrame(yData, uData, vData, yRowStride, uvRowStride,
                      uvPixelStride, format);
  } else {
    LOGE("addToStackNative: Failed to get buffer addresses");
  }
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_stageFrameNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject yBuffer,
    jobject uBuffer, jobject vBuffer, jint yRowStride, jint uvRowStride,
    jint uvPixelStride, jint format) {
  auto *stacker = reinterpret_cast<ImageStacker *>(stackerPtr);
  if (!stacker)
    return;
  auto *yData = static_cast<uint8_t *>(env->GetDirectBufferAddress(yBuffer));
  auto *uData = static_cast<uint8_t *>(env->GetDirectBufferAddress(uBuffer));
  auto *vData = static_cast<uint8_t *>(env->GetDirectBufferAddress(vBuffer));
  if (yData && uData && vData) {
    stacker->stageFrame(yData, uData, vData, yRowStride, uvRowStride,
                        uvPixelStride, format);
  }
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_processFrameNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jint index) {
  auto *stacker = reinterpret_cast<ImageStacker *>(stackerPtr);
  if (stacker)
    stacker->processFrame(index);
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_clearStagedFramesNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr) {
  auto *stacker = reinterpret_cast<ImageStacker *>(stackerPtr);
  if (stacker)
    stacker->clearStagedFrames();
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_processStackNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject outBitmap,
    jint rotation, jint targetWR, jint targetHR, jstring outputPath) {

  auto *stacker = reinterpret_cast<ImageStacker *>(stackerPtr);
  if (!stacker)
    return;

  const char *path = nullptr;
  if (outputPath) {
    path = env->GetStringUTFChars(outputPath, nullptr);
  }

  AndroidBitmapInfo info;
  void *bitmapPixels = nullptr;
  if (outBitmap &&
      (AndroidBitmap_getInfo(env, outBitmap, &info) < 0 ||
       AndroidBitmap_lockPixels(env, outBitmap, &bitmapPixels) < 0)) {
    if (path)
      env->ReleaseStringUTFChars(outputPath, path);
    return;
  }

  stacker->writeResult(static_cast<uint32_t *>(bitmapPixels), info.width,
                       info.height, rotation, targetWR, targetHR, path);

  if (outBitmap) {
    AndroidBitmap_unlockPixels(env, outBitmap);
  }
  if (path) {
    env->ReleaseStringUTFChars(outputPath, path);
  }
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_releaseStackerNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr) {
  auto *stacker = reinterpret_cast<ImageStacker *>(stackerPtr);
  delete stacker;
}

/**
 * Vulkan Stacking JNI Interface
 */
JNIEXPORT jlong JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_createVulkanStackerNative(
    JNIEnv *env, jobject /* this */, jint width, jint height,
    jboolean enableSuperRes) {
  try {
    auto *stacker = new VulkanImageStacker(width, height, enableSuperRes);
    return reinterpret_cast<jlong>(stacker);
  } catch (const std::exception &e) {
    LOGE("createVulkanStackerNative failed: %s", e.what());
  } catch (...) {
    LOGE("createVulkanStackerNative failed with unknown error");
  }
  return 0;
}

JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_addVulkanFrameNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject hardwareBuffer) {
  auto *stacker = reinterpret_cast<VulkanImageStacker *>(stackerPtr);
  if (!stacker || !hardwareBuffer)
    return JNI_FALSE;

  AHardwareBuffer *buffer =
      AHardwareBuffer_fromHardwareBuffer(env, hardwareBuffer);
  if (!buffer)
    return JNI_FALSE;

  bool success = stacker->addFrame(buffer);
  return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_processVulkanStackNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject outBitmap,
    jint rotation) {
  auto *stacker = reinterpret_cast<VulkanImageStacker *>(stackerPtr);
  if (!stacker)
    return JNI_FALSE;

  AndroidBitmapInfo info;
  void *bitmapPixels = nullptr;
  if (outBitmap &&
      (AndroidBitmap_getInfo(env, outBitmap, &info) < 0 ||
       AndroidBitmap_lockPixels(env, outBitmap, &bitmapPixels) < 0)) {
    return JNI_FALSE;
  }

  bool success =
      stacker->processStack(static_cast<uint32_t *>(bitmapPixels), info.width,
                            info.height, info.stride, rotation);

  if (outBitmap) {
    AndroidBitmap_unlockPixels(env, outBitmap);
  }
  return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_releaseVulkanStackerNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr) {
  auto *stacker = reinterpret_cast<VulkanImageStacker *>(stackerPtr);
  delete stacker;
}

JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_resetVulkanStackerNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr) {
  auto *stacker = reinterpret_cast<VulkanImageStacker *>(stackerPtr);
  if (!stacker)
    return JNI_FALSE;
  return stacker->resetForReuse() ? JNI_TRUE : JNI_FALSE;
}

/**
 * Raw Stacking JNI Interface
 */
JNIEXPORT jlong JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_createRawStackerNative(
    JNIEnv *env, jobject /* this */, jint width, jint height,
    jboolean enableSuperRes) {
  auto *stacker = new RawStacker(width, height, enableSuperRes);
  return reinterpret_cast<jlong>(stacker);
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_addToRawStackNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject rawData,
    jint rowStride, jint cfaPattern) {

  auto *stacker = reinterpret_cast<RawStacker *>(stackerPtr);
  if (!stacker)
    return;

  auto *data = static_cast<uint16_t *>(env->GetDirectBufferAddress(rawData));
  if (data) {
    stacker->addFrame(data, rowStride, cfaPattern);
  } else {
    LOGE("addToRawStackNative: Failed to get buffer address");
  }
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_stageRawFrameNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject rawData,
    jint rowStride, jint cfaPattern) {
  auto *stacker = reinterpret_cast<RawStacker *>(stackerPtr);
  if (!stacker)
    return;
  auto *data = static_cast<uint16_t *>(env->GetDirectBufferAddress(rawData));
  if (data) {
    stacker->stageFrame(data, rowStride, cfaPattern);
  }
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_processRawFrameNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jint index) {
  auto *stacker = reinterpret_cast<RawStacker *>(stackerPtr);
  if (stacker)
    stacker->processFrame(index);
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_clearStagedRawFramesNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr) {
  auto *stacker = reinterpret_cast<RawStacker *>(stackerPtr);
  if (stacker)
    stacker->clearStagedFrames();
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_processRawStackWithBufferNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr, jobject outputBuffer) {

  auto *stacker = reinterpret_cast<RawStacker *>(stackerPtr);
  if (!stacker)
    return;

  auto *outData =
      static_cast<uint16_t *>(env->GetDirectBufferAddress(outputBuffer));
  if (!outData)
    return;

  std::vector<uint16_t> result = stacker->process();

  jlong capacity = env->GetDirectBufferCapacity(outputBuffer);
  if (capacity >= result.size() * sizeof(uint16_t)) {
    memcpy(outData, result.data(), result.size() * sizeof(uint16_t));
  } else {
    LOGE("Output buffer too small: capacity=%ld, required=%ld", (long)capacity,
         (long)(result.size() * sizeof(uint16_t)));
  }
}

/**
 * Vulkan Raw Stacking JNI Interface
 */
JNIEXPORT jlong JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_createVulkanRawStackerNative(
    JNIEnv *env, jobject, jint width, jint height, jboolean enableSuperRes,
    jfloat superResScale,
    jfloatArray blackLevel, jint whiteLevel, jfloatArray wbGains,
    jfloatArray noiseModel, jfloatArray lensShadingMap, jint lscWidth,
    jint lscHeight) {

  float bl[4] = {0, 0, 0, 0};
  if (blackLevel) {
    jfloat *body = env->GetFloatArrayElements(blackLevel, nullptr);
    memcpy(bl, body, 4 * sizeof(float));
    env->ReleaseFloatArrayElements(blackLevel, body, 0);
  }

  float wb[4] = {1, 1, 1, 1};
  if (wbGains) {
    jfloat *body = env->GetFloatArrayElements(wbGains, nullptr);
    memcpy(wb, body, 4 * sizeof(float));
    env->ReleaseFloatArrayElements(wbGains, body, 0);
  }

  float noise[2] = {0, 0};
  if (noiseModel) {
    jfloat *body = env->GetFloatArrayElements(noiseModel, nullptr);
    memcpy(noise, body, 2 * sizeof(float));
    env->ReleaseFloatArrayElements(noiseModel, body, 0);
  }

  float *lsc = nullptr;
  if (lensShadingMap && lscWidth > 0 && lscHeight > 0) {
    lsc = env->GetFloatArrayElements(lensShadingMap, nullptr);
  }

  VulkanRawStacker *stacker = nullptr;
  try {
    stacker = new VulkanRawStacker(width, height, enableSuperRes,
                                   superResScale, bl, (float)whiteLevel, wb,
                                   noise, lsc, (uint32_t)lscWidth,
                                   (uint32_t)lscHeight);
  } catch (const std::exception &e) {
    LOGE("createVulkanRawStackerNative failed: %s", e.what());
  } catch (...) {
    LOGE("createVulkanRawStackerNative failed with unknown error");
  }

  if (lsc) {
    env->ReleaseFloatArrayElements(lensShadingMap, lsc, 0);
  }

  if (!stacker) {
    return 0;
  }
  return reinterpret_cast<jlong>(stacker);
}

JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_addVulkanRawFrameNative(
    JNIEnv *env, jobject, jlong stackerPtr, jobject rawData, jint rowStride,
    jint cfaPattern) {
  auto *stacker = reinterpret_cast<VulkanRawStacker *>(stackerPtr);
  if (!stacker)
    return JNI_FALSE;

  auto *data = static_cast<uint16_t *>(env->GetDirectBufferAddress(rawData));
  if (!data) {
    LOGE("addVulkanRawFrameNative: Failed to get buffer address");
    return JNI_FALSE;
  }

  bool success = stacker->addFrame(data, rowStride, cfaPattern);
  return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_processVulkanRawStackNative(
    JNIEnv *env, jobject, jlong stackerPtr, jobject outputBuffer) {
  auto *stacker = reinterpret_cast<VulkanRawStacker *>(stackerPtr);
  if (!stacker)
    return JNI_FALSE;

  auto *outData =
      static_cast<uint16_t *>(env->GetDirectBufferAddress(outputBuffer));
  if (!outData) {
    LOGE("processVulkanRawStackNative: Failed to get buffer address");
    return JNI_FALSE;
  }

  jlong capacity = env->GetDirectBufferCapacity(outputBuffer);
  bool success = stacker->processStack(outData, (size_t)capacity);
  return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_releaseVulkanRawStackerNative(
    JNIEnv *env, jobject, jlong stackerPtr) {
  auto *stacker = reinterpret_cast<VulkanRawStacker *>(stackerPtr);
  delete stacker;
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_processor_MultiFrameStacker_releaseRawStackerNative(
    JNIEnv *env, jobject /* this */, jlong stackerPtr) {
  auto *stacker = reinterpret_cast<RawStacker *>(stackerPtr);
  delete stacker;
}

/**
 * 处理 YUV_420_888 或 P010 图像：旋转、裁切、转换为 RGBA16
 */
/**
 * 处理 YUV_420_888 或 P010 图像：旋转、裁切，并直接保存为 FP16 格式的 JPEG XL
 */
JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_utils_YuvProcessor_processAndSaveYuv(
    JNIEnv *env, jobject /* this */, jobject yBuffer, jobject uBuffer,
    jobject vBuffer, jint width, jint height, jint yRowStride, jint uvRowStride,
    jint uvPixelStride, jint rotation, jint targetWR, jint targetHR,
    jint format, jstring outputPath, jstring hdrSidecarPath, jobject outBitmap8) {
  constexpr int kHdrSidecarDownsample = 2;
  const char *path = env->GetStringUTFChars(outputPath, nullptr);
  const char *hdrPath =
      hdrSidecarPath ? env->GetStringUTFChars(hdrSidecarPath, nullptr) : nullptr;

  // 1. 锁定 Bitmap 地址 (8-bit) 用于预览
  void *bitmapPixels;
  if (AndroidBitmap_lockPixels(env, outBitmap8, &bitmapPixels) < 0) {
    env->ReleaseStringUTFChars(outputPath, path);
    if (hdrPath)
      env->ReleaseStringUTFChars(hdrSidecarPath, hdrPath);
    return JNI_FALSE;
  }
  auto *ptr8 = static_cast<uint32_t *>(bitmapPixels);
  AndroidBitmapInfo info;
  AndroidBitmap_getInfo(env, outBitmap8, &info);
  const int bitmapStridePixels = static_cast<int>(info.stride / 4);

  // 获取 buffer 指针
  auto *yData = static_cast<uint8_t *>(env->GetDirectBufferAddress(yBuffer));
  auto *uData = static_cast<uint8_t *>(env->GetDirectBufferAddress(uBuffer));
  auto *vData = static_cast<uint8_t *>(env->GetDirectBufferAddress(vBuffer));

  if (yData == nullptr || uData == nullptr || vData == nullptr) {
    LOGE("Failed to get buffer addresses");
    AndroidBitmap_unlockPixels(env, outBitmap8);
    env->ReleaseStringUTFChars(outputPath, path);
    if (hdrPath)
      env->ReleaseStringUTFChars(hdrSidecarPath, hdrPath);
    return JNI_FALSE;
  }

  bool isP010 = (format == 0x36);
  int rotatedWidth = (rotation == 90 || rotation == 270) ? height : width;
  int rotatedHeight = (rotation == 90 || rotation == 270) ? width : height;

  // === 裁切计算 ===
  bool currentIsLandscape = (rotatedWidth >= rotatedHeight);
  int tw, th;
  if (currentIsLandscape) {
    tw = (targetWR >= targetHR) ? targetWR : targetHR;
    th = (targetWR >= targetHR) ? targetHR : targetWR;
  } else {
    tw = (targetWR >= targetHR) ? targetHR : targetWR;
    th = (targetWR >= targetHR) ? targetWR : targetHR;
  }

  int finalWidth, finalHeight;
  if ((long long)rotatedWidth * th > (long long)tw * rotatedHeight) {
    finalHeight = (rotatedHeight / 2) * 2;
    finalWidth = (int)(((long long)finalHeight * tw / th) / 2) * 2;
  } else {
    finalWidth = (rotatedWidth / 2) * 2;
    finalHeight = (int)(((long long)finalWidth * th / tw) / 2) * 2;
  }
  finalWidth = std::min(finalWidth, (rotatedWidth / 2) * 2);
  finalHeight = std::min(finalHeight, (rotatedHeight / 2) * 2);
  if (finalWidth > (int)info.width || finalHeight > (int)info.height) {
    LOGI("processAndSaveYuv clamping output from %dx%d to bitmap bounds %ux%u",
         finalWidth, finalHeight, info.width, info.height);
  }
  finalWidth = std::min(finalWidth, (int)info.width);
  finalHeight = std::min(finalHeight, (int)info.height);
  if (finalWidth > 1) {
    finalWidth &= ~1;
  }
  if (finalHeight > 1) {
    finalHeight &= ~1;
  }

  int cropX = ((rotatedWidth - finalWidth) / 4) * 2;
  int cropY = ((rotatedHeight - finalHeight) / 4) * 2;

  // === 转换并存储为 RGB ===
  std::vector<uint16_t> fp16Pixels(finalWidth * finalHeight * 4);
  std::vector<uint16_t> hdrReferencePixels;
  if (isP010 && hdrPath) {
    hdrReferencePixels.resize(finalWidth * finalHeight * 4);
  }

#pragma omp parallel for num_threads(4)
  for (int y = 0; y < finalHeight; y++) {
    for (int x = 0; x < finalWidth; x++) {
      int rx = x + cropX;
      int ry = y + cropY;

      int sx, sy;
      if (rotation == 90) {
        sx = ry;
        sy = height - 1 - rx;
      } else if (rotation == 180) {
        sx = width - 1 - rx;
        sy = height - 1 - ry;
      } else if (rotation == 270) {
        sx = width - 1 - ry;
        sy = rx;
      } else { // 0
        sx = rx;
        sy = ry;
      }

      float Y_val, U_val, V_val;
      if (isP010) {
        Y_val = (float)readValue<uint16_t>(yData + sy * yRowStride + sx * 2,
                                           false) /
                65535.0f;
        int uv_sx = sx / 2;
        int uv_sy = sy / 2;
        U_val =
            (static_cast<float>(readValue<uint16_t>(
                 uData + uv_sy * uvRowStride + uv_sx * uvPixelStride, false)) -
             32768.0f) /
            65535.0f;
        V_val =
            (static_cast<float>(readValue<uint16_t>(
                 vData + uv_sy * uvRowStride + uv_sx * uvPixelStride, false)) -
             32768.0f) /
            65535.0f;
      } else {
        Y_val = (float)yData[sy * yRowStride + sx] / 255.0f;
        int uv_sx = sx / 2;
        int uv_sy = sy / 2;
        U_val = (static_cast<float>(
                     uData[uv_sy * uvRowStride + uv_sx * uvPixelStride]) -
                 128.0f) /
                255.0f;
        V_val = (static_cast<float>(
                     vData[uv_sy * uvRowStride + uv_sx * uvPixelStride]) -
                 128.0f) /
                255.0f;
      }

      float hdrRInput, hdrGInput, hdrBInput;
      if (isP010) {
        hdrRInput = Y_val + 1.4746f * V_val;
        hdrGInput = Y_val - 0.16455f * U_val - 0.57135f * V_val;
        hdrBInput = Y_val + 1.8814f * U_val;
      } else {
        hdrRInput = Y_val + 1.402f * V_val;
        hdrGInput = Y_val - 0.344136f * U_val - 0.714136f * V_val;
        hdrBInput = Y_val + 1.772f * U_val;
      }

      const float R = std::max(0.0f, std::min(1.0f, hdrRInput));
      const float G = std::max(0.0f, std::min(1.0f, hdrGInput));
      const float B = std::max(0.0f, std::min(1.0f, hdrBInput));

      int idx = y * finalWidth + x;

      if (!hdrReferencePixels.empty()) {
        const float hdrR = hlgToLinear(hdrRInput);
        const float hdrG = hlgToLinear(hdrGInput);
        const float hdrB = hlgToLinear(hdrBInput);
        const int hdrIdx = idx * 4;
        hdrReferencePixels[hdrIdx + 0] = floatToHalf(hdrR);
        hdrReferencePixels[hdrIdx + 1] = floatToHalf(hdrG);
        hdrReferencePixels[hdrIdx + 2] = floatToHalf(hdrB);
        hdrReferencePixels[hdrIdx + 3] = floatToHalf(1.0f);
      }

      // --- 输出 A: UINT16 (保存到本地) ---
      int idx16 = idx * 4;
      fp16Pixels[idx16 + 0] = static_cast<uint16_t>(R * 65535.0f);
      fp16Pixels[idx16 + 1] = static_cast<uint16_t>(G * 65535.0f);
      fp16Pixels[idx16 + 2] = static_cast<uint16_t>(B * 65535.0f);
      fp16Pixels[idx16 + 3] = 65535; // Alpha

      // --- 输出 B: 8-bit (预览) ---
      uint32_t r8 = static_cast<uint32_t>(R * 255.0f);
      uint32_t g8 = static_cast<uint32_t>(G * 255.0f);
      uint32_t b8 = static_cast<uint32_t>(B * 255.0f);
      uint32_t a8 = 255;
      ptr8[y * bitmapStridePixels + x] =
          (a8 << 24) | (b8 << 16) | (g8 << 8) | r8;
    }
  }

  AndroidBitmap_unlockPixels(env, outBitmap8);

  // 保存为 JXL
  bool success = saveJxl(
      fp16Pixels.data(), finalWidth, finalHeight, JXL_TYPE_UINT16, path,
      isP010 ? JxlEncodingProfile::BT2100_HLG : JxlEncodingProfile::ORIGINAL);

  if (success && isP010 && hdrPath) {
    const int sidecarWidth =
        std::max(1, (finalWidth + kHdrSidecarDownsample - 1) /
                        kHdrSidecarDownsample);
    const int sidecarHeight =
        std::max(1, (finalHeight + kHdrSidecarDownsample - 1) /
                        kHdrSidecarDownsample);
    std::vector<uint16_t> hdrSidecarPixels(sidecarWidth * sidecarHeight * 4);
    for (int y = 0; y < sidecarHeight; ++y) {
      const int srcY = std::min(
          finalHeight - 1,
          (y * finalHeight + sidecarHeight / 2) / sidecarHeight);
      for (int x = 0; x < sidecarWidth; ++x) {
        const int srcX = std::min(
            finalWidth - 1,
            (x * finalWidth + sidecarWidth / 2) / sidecarWidth);
        const int srcIdx = (srcY * finalWidth + srcX) * 4;
        const int dstIdx = (y * sidecarWidth + x) * 4;
        hdrSidecarPixels[dstIdx + 0] = hdrReferencePixels[srcIdx + 0];
        hdrSidecarPixels[dstIdx + 1] = hdrReferencePixels[srcIdx + 1];
        hdrSidecarPixels[dstIdx + 2] = hdrReferencePixels[srcIdx + 2];
        hdrSidecarPixels[dstIdx + 3] = hdrReferencePixels[srcIdx + 3];
      }
    }

    success = saveJxl(hdrSidecarPixels.data(), sidecarWidth, sidecarHeight,
                      JXL_TYPE_FLOAT16, hdrPath);
    if (!success) {
      LOGE("Failed to write compressed HDR sidecar: %s", hdrPath);
    }
  }

  env->ReleaseStringUTFChars(outputPath, path);
  if (hdrPath)
    env->ReleaseStringUTFChars(hdrSidecarPath, hdrPath);
  return success ? JNI_TRUE : JNI_FALSE;
}

/**
 * 带有保存到本地文件的 JPG 压缩版本的 processToBitmap
 */
JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_utils_YuvProcessor_processToFile(
    JNIEnv *env, jobject /* this */, jobject yBuffer, jobject uBuffer,
    jobject vBuffer, jint width, jint height, jint yRowStride, jint uvRowStride,
    jint uvPixelStride, jint rotation, jint format, jstring outputPath) {

  const char *path = env->GetStringUTFChars(outputPath, nullptr);

  auto *yData = static_cast<uint8_t *>(env->GetDirectBufferAddress(yBuffer));
  auto *uData = static_cast<uint8_t *>(env->GetDirectBufferAddress(uBuffer));
  auto *vData = static_cast<uint8_t *>(env->GetDirectBufferAddress(vBuffer));

  if (yData == nullptr || uData == nullptr || vData == nullptr) {
    LOGE("Failed to get buffer addresses");
    env->ReleaseStringUTFChars(outputPath, path);
    return JNI_FALSE;
  }

  bool isP010 = (format == 0x36);
  int rotatedWidth = (rotation == 90 || rotation == 270) ? height : width;
  int rotatedHeight = (rotation == 90 || rotation == 270) ? width : height;

  std::vector<uint8_t> yDest(rotatedWidth * rotatedHeight);
  std::vector<uint8_t> uDest((rotatedWidth / 2) * (rotatedHeight / 2));
  std::vector<uint8_t> vDest((rotatedWidth / 2) * (rotatedHeight / 2));

#pragma omp parallel for num_threads(4)
  for (int y = 0; y < rotatedHeight; y++) {
    for (int x = 0; x < rotatedWidth; x++) {
      int sx, sy;
      if (rotation == 90) {
        sx = y;
        sy = height - 1 - x;
      } else if (rotation == 180) {
        sx = width - 1 - x;
        sy = height - 1 - y;
      } else if (rotation == 270) {
        sx = width - 1 - y;
        sy = x;
      } else { // 0
        sx = x;
        sy = y;
      }

      if (isP010) {
        yDest[y * rotatedWidth + x] =
            readValue<uint16_t>(yData + sy * yRowStride + sx * 2, false) >> 8;
      } else {
        yDest[y * rotatedWidth + x] = yData[sy * yRowStride + sx];
      }
    }
  }

#pragma omp parallel for num_threads(4)
  for (int y = 0; y < rotatedHeight / 2; y++) {
    for (int x = 0; x < rotatedWidth / 2; x++) {
      int rx = x * 2;
      int ry = y * 2;
      int sx, sy;
      if (rotation == 90) {
        sx = ry;
        sy = height - 1 - rx;
      } else if (rotation == 180) {
        sx = width - 1 - rx;
        sy = height - 1 - ry;
      } else if (rotation == 270) {
        sx = width - 1 - ry;
        sy = rx;
      } else { // 0
        sx = rx;
        sy = ry;
      }

      int uv_sx = sx / 2;
      int uv_sy = sy / 2;

      if (isP010) {
        uDest[y * (rotatedWidth / 2) + x] =
            readValue<uint16_t>(
                uData + uv_sy * uvRowStride + uv_sx * uvPixelStride, false) >>
            8;
        vDest[y * (rotatedWidth / 2) + x] =
            readValue<uint16_t>(
                vData + uv_sy * uvRowStride + uv_sx * uvPixelStride, false) >>
            8;
      } else {
        uDest[y * (rotatedWidth / 2) + x] =
            uData[uv_sy * uvRowStride + uv_sx * uvPixelStride];
        vDest[y * (rotatedWidth / 2) + x] =
            vData[uv_sy * uvRowStride + uv_sx * uvPixelStride];
      }
    }
  }

  tjhandle tjInstance = tj3Init(TJINIT_COMPRESS);
  if (!tjInstance) {
    LOGE("Failed to init turbojpeg: %s", tj3GetErrorStr(nullptr));
    env->ReleaseStringUTFChars(outputPath, path);
    return JNI_FALSE;
  }

  tj3Set(tjInstance, TJPARAM_QUALITY, 90);
  tj3Set(tjInstance, TJPARAM_SUBSAMP, TJSAMP_420);

  const unsigned char *srcPlanes[3] = {yDest.data(), uDest.data(),
                                       vDest.data()};
  int strides[3] = {rotatedWidth, rotatedWidth / 2, rotatedWidth / 2};

  unsigned char *jpegBuf = nullptr;
  size_t jpegSize = 0;

  if (tj3CompressFromYUVPlanes8(tjInstance, srcPlanes, rotatedWidth, strides,
                                rotatedHeight, &jpegBuf, &jpegSize) < 0) {
    LOGE("Failed to compress turbojpeg: %s", tj3GetErrorStr(tjInstance));
    tj3Destroy(tjInstance);
    env->ReleaseStringUTFChars(outputPath, path);
    return JNI_FALSE;
  }

  FILE *file = fopen(path, "wb");
  if (!file) {
    LOGE("Failed to open file for writing: %s", path);
    tj3Free(jpegBuf);
    tj3Destroy(tjInstance);
    env->ReleaseStringUTFChars(outputPath, path);
    return JNI_FALSE;
  }
  fwrite(jpegBuf, 1, jpegSize, file);
  fclose(file);

  tj3Free(jpegBuf);
  tj3Destroy(tjInstance);
  env->ReleaseStringUTFChars(outputPath, path);
  return JNI_TRUE;
}

/**
 * 仅处理预览：旋转、裁切，并输出为 8-bit Bitmap
 */
JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_utils_YuvProcessor_processToBitmap(
    JNIEnv *env, jobject /* this */, jobject yBuffer, jobject uBuffer,
    jobject vBuffer, jint width, jint height, jint yRowStride, jint uvRowStride,
    jint uvPixelStride, jint rotation, jint targetWR, jint targetHR,
    jint format, jobject outBitmap8) {

  void *bitmapPixels;
  if (AndroidBitmap_lockPixels(env, outBitmap8, &bitmapPixels) < 0) {
    LOGE("Failed to lock bitmap pixels");
    return;
  }
  auto *ptr8 = static_cast<uint32_t *>(bitmapPixels);

  AndroidBitmapInfo info;
  AndroidBitmap_getInfo(env, outBitmap8, &info);

  auto *yData = static_cast<uint8_t *>(env->GetDirectBufferAddress(yBuffer));
  auto *uData = static_cast<uint8_t *>(env->GetDirectBufferAddress(uBuffer));
  auto *vData = static_cast<uint8_t *>(env->GetDirectBufferAddress(vBuffer));

  if (yData == nullptr || uData == nullptr || vData == nullptr) {
    LOGE("Failed to get buffer addresses");
    AndroidBitmap_unlockPixels(env, outBitmap8);
    return;
  }

  bool isP010 = (format == 0x36);
  int rotatedWidth = (rotation == 90 || rotation == 270) ? height : width;
  int rotatedHeight = (rotation == 90 || rotation == 270) ? width : height;

  // === 裁切计算 ===
  bool currentIsLandscape = (rotatedWidth >= rotatedHeight);
  int tw, th;
  if (currentIsLandscape) {
    tw = (targetWR >= targetHR) ? targetWR : targetHR;
    th = (targetWR >= targetHR) ? targetHR : targetWR;
  } else {
    tw = (targetWR >= targetHR) ? targetHR : targetWR;
    th = (targetWR >= targetHR) ? targetWR : targetHR;
  }

  int finalWidth, finalHeight;
  if ((long long)rotatedWidth * th > (long long)tw * rotatedHeight) {
    finalHeight = (rotatedHeight / 2) * 2;
    finalWidth = (int)(((long long)finalHeight * tw / th) / 2) * 2;
  } else {
    finalWidth = (rotatedWidth / 2) * 2;
    finalHeight = (int)(((long long)finalWidth * th / tw) / 2) * 2;
  }

  // 匹配 Bitmap 尺寸
  finalWidth = std::min(finalWidth, (int)info.width);
  finalHeight = std::min(finalHeight, (int)info.height);

  int cropX = ((rotatedWidth - finalWidth) / 4) * 2;
  int cropY = ((rotatedHeight - finalHeight) / 4) * 2;

  int stride = info.stride / 4;

#pragma omp parallel for num_threads(4)
  for (int y = 0; y < finalHeight; y++) {
    for (int x = 0; x < finalWidth; x++) {
      int rx = x + cropX;
      int ry = y + cropY;
      int sx, sy;
      if (rotation == 90) {
        sx = ry;
        sy = height - 1 - rx;
      } else if (rotation == 180) {
        sx = width - 1 - rx;
        sy = height - 1 - ry;
      } else if (rotation == 270) {
        sx = width - 1 - ry;
        sy = rx;
      } else { // 0
        sx = rx;
        sy = ry;
      }

      float Y_val, U_val, V_val;
      if (isP010) {
        Y_val = (float)readValue<uint16_t>(yData + sy * yRowStride + sx * 2,
                                           false) /
                65535.0f;
        int uv_sx = sx / 2;
        int uv_sy = sy / 2;
        U_val =
            (static_cast<float>(readValue<uint16_t>(
                 uData + uv_sy * uvRowStride + uv_sx * uvPixelStride, false)) -
             32768.0f) /
            65535.0f;
        V_val =
            (static_cast<float>(readValue<uint16_t>(
                 vData + uv_sy * uvRowStride + uv_sx * uvPixelStride, false)) -
             32768.0f) /
            65535.0f;
      } else {
        Y_val = (float)yData[sy * yRowStride + sx] / 255.0f;
        int uv_sx = sx / 2;
        int uv_sy = sy / 2;
        U_val = (static_cast<float>(
                     uData[uv_sy * uvRowStride + uv_sx * uvPixelStride]) -
                 128.0f) /
                255.0f;
        V_val = (static_cast<float>(
                     vData[uv_sy * uvRowStride + uv_sx * uvPixelStride]) -
                 128.0f) /
                255.0f;
      }

      float R, G, B;
      if (isP010) {
        R = Y_val + 1.4746f * V_val;
        G = Y_val - 0.16455f * U_val - 0.57135f * V_val;
        B = Y_val + 1.8814f * U_val;
      } else {
        R = Y_val + 1.402f * V_val;
        G = Y_val - 0.344136f * U_val - 0.714136f * V_val;
        B = Y_val + 1.772f * U_val;
      }

      uint32_t r8 =
          static_cast<uint32_t>(std::max(0.0f, std::min(1.0f, R)) * 255.0f);
      uint32_t g8 =
          static_cast<uint32_t>(std::max(0.0f, std::min(1.0f, G)) * 255.0f);
      uint32_t b8 =
          static_cast<uint32_t>(std::max(0.0f, std::min(1.0f, B)) * 255.0f);
      ptr8[y * stride + x] = (255u << 24) | (b8 << 16) | (g8 << 8) | r8;
    }
  }

  AndroidBitmap_unlockPixels(env, outBitmap8);
}

/**
 * 从文件中读取并解压缩 RGBA 数据 (FP16)
 */
JNIEXPORT jobject JNICALL
Java_com_hinnka_mycamera_utils_YuvProcessor_loadCompressedArgb(
    JNIEnv *env, jobject /* this */, jstring inputPath) {

  const char *path = env->GetStringUTFChars(inputPath, nullptr);
  int32_t width, height;
  size_t dataSize = 0;

  // 使用 JXL_TYPE_FLOAT16 读取数据，以便于 OpenGL GLES 3.0 处理
  void *pixels = loadJxlRaw(path, width, height, JXL_TYPE_FLOAT16, dataSize);
  env->ReleaseStringUTFChars(inputPath, path);

  if (pixels == nullptr) {
    return nullptr;
  }

  // 直接返回像素数据，不再添加 4 字节宽高头，以保持与旧版本兼容
  return env->NewDirectByteBuffer(pixels, dataSize);
}

JNIEXPORT jintArray JNICALL
Java_com_hinnka_mycamera_utils_YuvProcessor_getCompressedArgbDimensions(
    JNIEnv *env, jobject /* this */, jstring inputPath) {
  const char *path = env->GetStringUTFChars(inputPath, nullptr);

  std::ifstream is(path, std::ios::binary | std::ios::ate);
  if (!is) {
    env->ReleaseStringUTFChars(inputPath, path);
    return nullptr;
  }
  std::streamsize size = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if (!is.read(reinterpret_cast<char *>(buffer.data()), size)) {
    env->ReleaseStringUTFChars(inputPath, path);
    return nullptr;
  }
  env->ReleaseStringUTFChars(inputPath, path);

  auto decoder = JxlDecoderMake(nullptr);
  size_t num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
  if (num_threads > 4)
    num_threads = 4;
  auto runner = JxlThreadParallelRunnerMake(nullptr, num_threads);
  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(decoder.get(),
                                                     JxlThreadParallelRunner,
                                                     runner.get())) {
    return nullptr;
  }

  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_BASIC_INFO)) {
    return nullptr;
  }

  JxlDecoderSetInput(decoder.get(), buffer.data(), buffer.size());
  JxlDecoderCloseInput(decoder.get());

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(decoder.get());
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(decoder.get(), &info)) {
        return nullptr;
      }
      jintArray result = env->NewIntArray(2);
      if (result == nullptr) {
        return nullptr;
      }
      const jint dims[2] = {static_cast<jint>(info.xsize),
                            static_cast<jint>(info.ysize)};
      env->SetIntArrayRegion(result, 0, 2, dims);
      return result;
    }
    if (status == JXL_DEC_SUCCESS || status == JXL_DEC_ERROR ||
        status == JXL_DEC_NEED_MORE_INPUT) {
      return nullptr;
    }
  }
}

/**
 * 将 RGBA 数据 (FP16) 压缩并保存到文件
 * 注意：输入 buffer 应该直接包含像素数据，不含宽高头
 */
JNIEXPORT jboolean JNICALL
Java_com_hinnka_mycamera_utils_YuvProcessor_saveCompressedArgb(
    JNIEnv *env, jobject /* this */, jobject buffer, jint width, jint height,
    jstring outputPath) {

  if (buffer == nullptr || outputPath == nullptr)
    return JNI_FALSE;

  void *pixels = env->GetDirectBufferAddress(buffer);
  if (pixels == nullptr) {
    LOGE("saveCompressedArgb: Failed to get buffer address");
    return JNI_FALSE;
  }

  const char *path = env->GetStringUTFChars(outputPath, nullptr);
  bool success = saveJxl(pixels, width, height, JXL_TYPE_FLOAT16, path);
  env->ReleaseStringUTFChars(outputPath, path);

  return success ? JNI_TRUE : JNI_FALSE;
}

/**
 * 释放内存
 */
JNIEXPORT void JNICALL Java_com_hinnka_mycamera_utils_YuvProcessor_free(
    JNIEnv *env, jobject /* this */, jobject buffer) {
  if (buffer == nullptr)
    return;
  void *nativePtr = env->GetDirectBufferAddress(buffer);
  if (nativePtr != nullptr) {
    free(nativePtr);
  }
}

struct ExifData {
  int iso = 0;
  float noiseProfile[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  bool hasNoiseProfile = false;
  int subjectLocation[4] = {0, 0, 0, 0};
  int subjectLocationLen = 0;
};

static int sget2(unsigned int ord, LibRaw_abstract_datastream *stream) {
  if (!stream)
    return 0;
  unsigned char s[2];
  if (stream->read(s, 1, 2) != 2)
    return 0;
  if (ord == 0x4d4d) // MM (Big Endian)
    return s[0] << 8 | s[1];
  else // II (Little Endian)
    return s[1] << 8 | s[0];
}

static int sget4(unsigned int ord, LibRaw_abstract_datastream *stream) {
  if (!stream)
    return 0;
  unsigned char s[4];
  if (stream->read(s, 1, 4) != 4)
    return 0;
  if (ord == 0x4d4d) // MM (Big Endian)
    return (s[0] << 24) | (s[1] << 16) | (s[2] << 8) | s[3];
  else // II (Little Endian)
    return (s[3] << 24) | (s[2] << 16) | (s[1] << 8) | s[0];
}

static void exif_callback(void *datap, int tag, int type, int len,
                          unsigned int ord, void *ifp, long long offset) {
  auto *ed = static_cast<ExifData *>(datap);
  auto *stream = static_cast<LibRaw_abstract_datastream *>(ifp);

  int actual_tag = tag & 0xFFFF;
  INT64 current_pos = stream->tell();

  if (offset != 0) {
    stream->seek(offset, SEEK_SET);
  }

  if (actual_tag == 0x8827) { // ISOSpeedRatings
    if (len > 0) {
      if (type == 3)
        ed->iso = sget2(ord, stream);
      else if (type == 4)
        ed->iso = sget4(ord, stream);
    }
  } else if (actual_tag == 0xC635 || actual_tag == 0xC761) { // NoiseProfile
    if (len > 0) {
      int count = std::min(len, 8);
      for (int i = 0; i < count; i++) {
        if (type == 12) { // DOUBLE
          double val = 0;
          stream->read(&val, 8, 1);
          ed->noiseProfile[i] = (float)val;
        } else if (type == 11) { // FLOAT
          float val = 0;
          stream->read(&val, 4, 1);
          ed->noiseProfile[i] = val;
        }
      }
      if (count > 0)
        ed->hasNoiseProfile = true;
    }
  } else if (actual_tag == 0x9214 ||
             actual_tag == 0xA214) { // SubjectLocation / SubjectArea
    if (len > 0) {
      int count = std::min(len, 4);
      for (int i = 0; i < count; i++) {
        if (type == 3)
          ed->subjectLocation[i] = sget2(ord, stream);
        else if (type == 4)
          ed->subjectLocation[i] = sget4(ord, stream);
      }
      ed->subjectLocationLen = count;
    }
  }

  // Always restore the stream position
  stream->seek(current_pos, SEEK_SET);
}

/**
 * Native 方法：直接从内存像素 Buffer 处理 RAW 数据
 * 接收纯 Bayer 数据（无 DNG 头），因此所有元数据必须由 Kotlin 提供
 */
JNIEXPORT jobject JNICALL
Java_com_hinnka_mycamera_raw_RawDemosaicProcessor_processRawBufferNative(
    JNIEnv *env, jobject /* this */, jobject rawBuffer, jint width, jint height,
    jint rowStride, jint cfaPattern, jfloatArray blackLevel, jfloat whiteLevel,
    jfloatArray wbGains, jfloatArray colorMatrix, jfloatArray noiseProfile,
    jfloatArray lensShadingMap, jint lensShadingMapWidth,
    jint lensShadingMapHeight, jfloat xr, jfloat yr, jfloat xg, jfloat yg,
    jfloat xb, jfloat yb, jfloat xw, jfloat yw,
    jboolean useRawAutoWhiteBalanceEstimate) {

  if (!rawBuffer) {
    LOGE("processRawBufferNative: rawBuffer is null");
    return nullptr;
  }

  auto *data = static_cast<ushort *>(env->GetDirectBufferAddress(rawBuffer));
  if (!data) {
    LOGE("processRawBufferNative: Failed to get buffer address");
    return nullptr;
  }

  if (width <= 1 || height <= 1 || rowStride < width * 2 || whiteLevel <= 0.0f) {
    LOGE("processRawBufferNative: invalid input size=%dx%d rowStride=%d white=%f",
         width, height, rowStride, whiteLevel);
    return nullptr;
  }

  float bl[4] = {0, 0, 0, 0};
  if (blackLevel) {
    const jsize count = std::min<jsize>(4, env->GetArrayLength(blackLevel));
    env->GetFloatArrayRegion(blackLevel, 0, count, bl);
  }

  float wb[4] = {1, 1, 1, 1};
  if (wbGains) {
    const jsize count = std::min<jsize>(4, env->GetArrayLength(wbGains));
    env->GetFloatArrayRegion(wbGains, 0, count, wb);
  }
  float wbBase = (wb[1] > 0.0f && std::isfinite(wb[1])) ? wb[1] : 1.0f;
  for (float &gain : wb) {
    gain = (gain > 0.0f && std::isfinite(gain)) ? gain / wbBase : 1.0f;
  }

  float cm[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f,
  };
  if (colorMatrix) {
    const jsize count = std::min<jsize>(9, env->GetArrayLength(colorMatrix));
    env->GetFloatArrayRegion(colorMatrix, 0, count, cm);
  }

  std::vector<float> noiseValues;
  if (noiseProfile) {
    const jsize count = env->GetArrayLength(noiseProfile);
    noiseValues.resize(static_cast<size_t>(count));
    if (count > 0) {
      env->GetFloatArrayRegion(noiseProfile, 0, count, noiseValues.data());
    }
  }

  std::vector<float> lscValues;
  const bool hasLsc =
      lensShadingMap && lensShadingMapWidth > 0 && lensShadingMapHeight > 0 &&
      env->GetArrayLength(lensShadingMap) >= lensShadingMapWidth * lensShadingMapHeight * 4;
  if (hasLsc) {
    const jsize count = lensShadingMapWidth * lensShadingMapHeight * 4;
    lscValues.resize(static_cast<size_t>(count));
    env->GetFloatArrayRegion(lensShadingMap, 0, count, lscValues.data());
  }

  {
  LibRaw rawProcessor;
  const auto *rawBytes = reinterpret_cast<const unsigned char *>(data);
  int ret = rawProcessor.open_bayer(
      rawBytes, static_cast<unsigned>(width * height * sizeof(ushort)),
      static_cast<ushort>(width), static_cast<ushort>(height), 0, 0, 0, 0, 0,
      mapCfaPatternToLibRaw(cfaPattern), 0, 0, 0);
  if (ret != LIBRAW_SUCCESS) {
    LOGE("processRawBufferNative: open_bayer failed ret=%d", ret);
    return nullptr;
  }

  rawProcessor.imgdata.color.black = 0;
  rawProcessor.imgdata.color.cblack[0] = static_cast<unsigned>(std::max(0.0f, bl[0]));
  rawProcessor.imgdata.color.cblack[1] = static_cast<unsigned>(std::max(0.0f, bl[1]));
  rawProcessor.imgdata.color.cblack[2] = static_cast<unsigned>(std::max(0.0f, bl[3]));
  rawProcessor.imgdata.color.cblack[3] = static_cast<unsigned>(std::max(0.0f, bl[2]));
  rawProcessor.imgdata.color.maximum = static_cast<unsigned>(std::max(1.0f, whiteLevel));
  rawProcessor.imgdata.color.dng_levels.dng_whitelevel[0] =
      static_cast<unsigned>(std::max(1.0f, whiteLevel));
  rawProcessor.imgdata.color.cam_mul[0] = wb[0];
  rawProcessor.imgdata.color.cam_mul[1] = wb[1];
  rawProcessor.imgdata.color.cam_mul[2] = wb[3];
  rawProcessor.imgdata.color.cam_mul[3] = wb[2];
  LOGI("processRawBufferNative: black RGGB=%f,%f,%f,%f white=%f cam_mul RGBG=%f,%f,%f,%f",
       bl[0], bl[1], bl[2], bl[3], whiteLevel,
       rawProcessor.imgdata.color.cam_mul[0],
       rawProcessor.imgdata.color.cam_mul[1],
       rawProcessor.imgdata.color.cam_mul[2],
       rawProcessor.imgdata.color.cam_mul[3]);
  LOGI("processRawBufferNative: inputCCM=%f,%f,%f,%f,%f,%f,%f,%f,%f",
       cm[0], cm[1], cm[2], cm[3], cm[4], cm[5], cm[6], cm[7], cm[8]);

  ret = rawProcessor.unpack();
  if (ret != LIBRAW_SUCCESS) {
    LOGE("processRawBufferNative: unpack failed ret=%d", ret);
    return nullptr;
  }

  if (hasLsc && rawProcessor.imgdata.rawdata.raw_image) {
    const int rawWidth = rawProcessor.imgdata.sizes.raw_width;
    const int rawHeight = rawProcessor.imgdata.sizes.raw_height;
    const int rawPitch = rawProcessor.imgdata.sizes.raw_pitch > 0
                             ? rawProcessor.imgdata.sizes.raw_pitch /
                                   static_cast<int>(sizeof(ushort))
                             : rawWidth;
    auto *rawImage = rawProcessor.imgdata.rawdata.raw_image;
    auto libRawChannelToAppChannel = [](int channel) -> int {
      switch (channel) {
      case 0:
        return 0; // R
      case 1:
        return 1; // Gr
      case 2:
        return 3; // B
      case 3:
        return 2; // Gb
      default:
        return 1;
      }
    };
    auto sampleAndroidLsc = [&](int appChannel, int x, int y) -> float {
      const float gx = (static_cast<float>(x) + 0.5f) *
                       static_cast<float>(std::max(0, lensShadingMapWidth - 1)) /
                       static_cast<float>(std::max(1, rawWidth));
      const float gy = (static_cast<float>(y) + 0.5f) *
                       static_cast<float>(std::max(0, lensShadingMapHeight - 1)) /
                       static_cast<float>(std::max(1, rawHeight));
      const int x0 = std::clamp(static_cast<int>(std::floor(gx)), 0,
                                static_cast<int>(lensShadingMapWidth) - 1);
      const int y0 = std::clamp(static_cast<int>(std::floor(gy)), 0,
                                static_cast<int>(lensShadingMapHeight) - 1);
      const int x1 =
          std::min(x0 + 1, static_cast<int>(lensShadingMapWidth) - 1);
      const int y1 =
          std::min(y0 + 1, static_cast<int>(lensShadingMapHeight) - 1);
      const float tx = gx - static_cast<float>(x0);
      const float ty = gy - static_cast<float>(y0);
      auto at = [&](int sx, int sy) -> float {
        const size_t index =
            (static_cast<size_t>(sy) * lensShadingMapWidth + sx) * 4 +
            appChannel;
        return lscValues[index];
      };
      const float v00 = at(x0, y0);
      const float v10 = at(x1, y0);
      const float v01 = at(x0, y1);
      const float v11 = at(x1, y1);
      const float top = v00 + (v10 - v00) * tx;
      const float bottom = v01 + (v11 - v01) * tx;
      return std::max(0.0f, top + (bottom - top) * ty);
    };

    double gainSum = 0.0;
    float gainMin = std::numeric_limits<float>::max();
    float gainMax = 0.0f;
    uint64_t gainCount = 0;
    for (int y = 0; y < rawHeight; ++y) {
      for (int x = 0; x < rawWidth; ++x) {
        const int libRawChannel = rawProcessor.FC(y, x);
        const int appChannel = libRawChannelToAppChannel(libRawChannel);
        const float black = bl[appChannel];
        const float gain = sampleAndroidLsc(appChannel, x, y);
        float corrected =
            (static_cast<float>(rawImage[y * rawPitch + x]) - black) * gain +
            black;
        rawImage[y * rawPitch + x] =
            static_cast<ushort>(std::clamp(corrected, 0.0f, 65535.0f));
        gainSum += static_cast<double>(gain);
        gainMin = std::min(gainMin, gain);
        gainMax = std::max(gainMax, gain);
        ++gainCount;
      }
    }
    LOGI("processRawBufferNative: android lsc grid=%dx%d gain avg=%f min=%f max=%f count=%llu",
         lensShadingMapWidth, lensShadingMapHeight,
         gainCount > 0 ? static_cast<float>(gainSum / static_cast<double>(gainCount)) : 0.0f,
         gainCount > 0 ? gainMin : 0.0f, gainMax,
         static_cast<unsigned long long>(gainCount));
  }

  rawProcessor.imgdata.params.output_bps = 16;
  rawProcessor.imgdata.params.gamm[0] = 1.0;
  rawProcessor.imgdata.params.gamm[1] = 1.0;
  rawProcessor.imgdata.params.no_auto_bright = 1;
  rawProcessor.imgdata.params.use_camera_wb = 1;
  rawProcessor.imgdata.params.output_color = 0;
  rawProcessor.imgdata.params.user_qual = 14;
  rawProcessor.imgdata.params.fbdd_noiserd = 0;
  rawProcessor.imgdata.params.threshold = 0;
  rawProcessor.imgdata.params.med_passes = 0;

  ret = rawProcessor.dcraw_process();
  if (ret != LIBRAW_SUCCESS) {
    LOGE("processRawBufferNative: dcraw_process failed ret=%d", ret);
    return nullptr;
  }

  libraw_processed_image_t *image = rawProcessor.dcraw_make_mem_image(&ret);
  if (!image || ret != LIBRAW_SUCCESS) {
    LOGE("processRawBufferNative: dcraw_make_mem_image failed ret=%d", ret);
    return nullptr;
  }
  logProcessedImageStats("processRawBufferNative", image);

  const size_t libRawOutputSize =
      static_cast<size_t>(image->width) * image->height * 3u * sizeof(ushort);
  void *libRawOutData = malloc(libRawOutputSize);
  if (!libRawOutData) {
    LibRaw::dcraw_clear_mem(image);
    LOGE("processRawBufferNative: malloc failed size=%zu", libRawOutputSize);
    return nullptr;
  }
  memcpy(libRawOutData, image->data, libRawOutputSize);
  jobject libRawBuffer = env->NewDirectByteBuffer(libRawOutData, libRawOutputSize);
  if (!libRawBuffer) {
    free(libRawOutData);
    LibRaw::dcraw_clear_mem(image);
    return nullptr;
  }

  jclass dngDataClass = env->FindClass("com/hinnka/mycamera/raw/DngRawData");
  jmethodID constructor =
      env->GetMethodID(dngDataClass, "<init>",
                       "(Ljava/nio/ByteBuffer;IIIF[F[F[F[FIIF[FIIFIJF[I[FLandroid/graphics/Bitmap;)V");

  jfloatArray blackLevelArray = env->NewFloatArray(4);
  float linearBlack[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  env->SetFloatArrayRegion(blackLevelArray, 0, 4, linearBlack);

  jfloatArray preMulArray = env->NewFloatArray(4);
  float oneMul[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  env->SetFloatArrayRegion(preMulArray, 0, 4, oneMul);

  jfloatArray wbArray = env->NewFloatArray(4);
  env->SetFloatArrayRegion(wbArray, 0, 4, oneMul);

  jfloatArray colorMatrixArray = env->NewFloatArray(9);
  env->SetFloatArrayRegion(colorMatrixArray, 0, 9, cm);

  jfloatArray noiseProfileArray = nullptr;
  if (!noiseValues.empty()) {
    noiseProfileArray = env->NewFloatArray(static_cast<jsize>(noiseValues.size()));
    env->SetFloatArrayRegion(noiseProfileArray, 0,
                             static_cast<jsize>(noiseValues.size()),
                             noiseValues.data());
  }

  jintArray activeArray = env->NewIntArray(4);
  jint aa[4] = {0, 0, static_cast<jint>(image->width),
                static_cast<jint>(image->height)};
  env->SetIntArrayRegion(activeArray, 0, 4, aa);

  jobject dngData = env->NewObject(
      dngDataClass, constructor, libRawBuffer, static_cast<jint>(image->width),
      static_cast<jint>(image->height), static_cast<jint>(image->width * 6),
      65535.0f, blackLevelArray, preMulArray, wbArray, colorMatrixArray, -1, 0,
      0.0f, nullptr, 0, 0, 0.0f, 0, 0L, 0.0f, activeArray,
      noiseProfileArray, nullptr);
  LOGI("processRawBufferNative: LibRaw processed %dx%d lsc=%d cam_mul RGBG=%f,%f,%f,%f",
       image->width, image->height, hasLsc ? 1 : 0,
       rawProcessor.imgdata.color.cam_mul[0],
       rawProcessor.imgdata.color.cam_mul[1],
       rawProcessor.imgdata.color.cam_mul[2],
       rawProcessor.imgdata.color.cam_mul[3]);
  LibRaw::dcraw_clear_mem(image);
  return dngData;
  }

  const int srcStride = rowStride / static_cast<int>(sizeof(ushort));
  auto cfaChannelAt = [cfaPattern](int x, int y) -> int {
    const int xp = x & 1;
    const int yp = y & 1;
    switch (cfaPattern) {
    case 1: // GRBG
      return yp == 0 ? (xp == 0 ? 1 : 0) : (xp == 0 ? 3 : 2);
    case 2: // GBRG
      return yp == 0 ? (xp == 0 ? 2 : 3) : (xp == 0 ? 0 : 1);
    case 3: // BGGR
      return yp == 0 ? (xp == 0 ? 3 : 2) : (xp == 0 ? 1 : 0);
    case 0: // RGGB
    default:
      return yp == 0 ? (xp == 0 ? 0 : 1) : (xp == 0 ? 2 : 3);
    }
  };

  auto sampleLsc = [&](int channel, int x, int y) -> float {
    if (!hasLsc) {
      return 1.0f;
    }
    const float gx = (static_cast<float>(x) + 0.5f) *
                         static_cast<float>(lensShadingMapWidth) /
                         std::max(1, width) -
                     0.5f;
    const float gy = (static_cast<float>(y) + 0.5f) *
                         static_cast<float>(lensShadingMapHeight) /
                         std::max(1, height) -
                     0.5f;
    const int x0 = std::clamp(static_cast<int>(std::floor(gx)), 0,
                              static_cast<int>(lensShadingMapWidth) - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(gy)), 0,
                              static_cast<int>(lensShadingMapHeight) - 1);
    const int x1 = std::min(x0 + 1, static_cast<int>(lensShadingMapWidth) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(lensShadingMapHeight) - 1);
    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    auto at = [&](int sx, int sy) -> float {
      const size_t index =
          (static_cast<size_t>(sy) * lensShadingMapWidth + sx) * 4 + channel;
      return lscValues[index];
    };
    const float v00 = at(x0, y0);
    const float v10 = at(x1, y0);
    const float v01 = at(x0, y1);
    const float v11 = at(x1, y1);
    const float top = v00 + (v10 - v00) * tx;
    const float bottom = v01 + (v11 - v01) * tx;
    return std::max(0.0f, top + (bottom - top) * ty);
  };

  auto correctedSensor = [&](int x, int y, int channel) -> float {
    const int sx = std::clamp(x, 0, static_cast<int>(width) - 1);
    const int sy = std::clamp(y, 0, static_cast<int>(height) - 1);
    const float raw = static_cast<float>(data[sy * srcStride + sx]);
    const float black = bl[channel];
    const float range = std::max(1.0f, whiteLevel - black);
    const float gain = sampleLsc(channel, sx, sy) * std::max(0.0f, wb[channel]);
    return std::clamp((raw - black) * gain / range, 0.0f, 1.0f);
  };

  auto sampleChannel = [&](int x, int y, int channel) -> float {
    const int ownChannel = cfaChannelAt(x, y);
    if (ownChannel == channel ||
        ((ownChannel == 1 || ownChannel == 2) && (channel == 1 || channel == 2))) {
      return correctedSensor(x, y, ownChannel);
    }

    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        const int sx = std::clamp(x + dx, 0, static_cast<int>(width) - 1);
        const int sy = std::clamp(y + dy, 0, static_cast<int>(height) - 1);
        const int neighborChannel = cfaChannelAt(sx, sy);
        if (neighborChannel == channel ||
            ((neighborChannel == 1 || neighborChannel == 2) &&
             (channel == 1 || channel == 2))) {
          sum += correctedSensor(sx, sy, neighborChannel);
          ++count;
        }
      }
    }
    if (count == 0) {
      return correctedSensor(x, y, ownChannel);
    }
    return sum / static_cast<float>(count);
  };

  const size_t outputSize =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 3u * sizeof(ushort);
  void *outData = malloc(outputSize);
  if (!outData) {
    LOGE("processRawBufferNative: malloc failed size=%zu", outputSize);
    return nullptr;
  }
  auto *out = static_cast<ushort *>(outData);

#pragma omp parallel for num_threads(8)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t base = (static_cast<size_t>(y) * width + x) * 3u;
      out[base] = static_cast<ushort>(
          std::clamp(sampleChannel(x, y, 0) * 65535.0f, 0.0f, 65535.0f));
      out[base + 1] = static_cast<ushort>(
          std::clamp(sampleChannel(x, y, 1) * 65535.0f, 0.0f, 65535.0f));
      out[base + 2] = static_cast<ushort>(
          std::clamp(sampleChannel(x, y, 3) * 65535.0f, 0.0f, 65535.0f));
    }
  }

  jobject rawDataBuffer = env->NewDirectByteBuffer(outData, outputSize);
  if (!rawDataBuffer) {
    free(outData);
    return nullptr;
  }

  jclass dngDataClass = env->FindClass("com/hinnka/mycamera/raw/DngRawData");
  jmethodID constructor =
      env->GetMethodID(dngDataClass, "<init>",
                       "(Ljava/nio/ByteBuffer;IIIF[F[F[F[FIIF[FIIFIJF[I[FLandroid/graphics/Bitmap;)V");

  jfloatArray blackLevelArray = env->NewFloatArray(4);
  float linearBlack[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  env->SetFloatArrayRegion(blackLevelArray, 0, 4, linearBlack);

  jfloatArray preMulArray = env->NewFloatArray(4);
  float preMul[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  env->SetFloatArrayRegion(preMulArray, 0, 4, preMul);

  jfloatArray wbArray = env->NewFloatArray(4);
  env->SetFloatArrayRegion(wbArray, 0, 4, preMul);

  jfloatArray colorMatrixArray = env->NewFloatArray(9);
  env->SetFloatArrayRegion(colorMatrixArray, 0, 9, cm);

  jfloatArray noiseProfileArray = nullptr;
  if (!noiseValues.empty()) {
    noiseProfileArray = env->NewFloatArray(static_cast<jsize>(noiseValues.size()));
    env->SetFloatArrayRegion(noiseProfileArray, 0,
                             static_cast<jsize>(noiseValues.size()),
                             noiseValues.data());
  }

  jintArray activeArray = env->NewIntArray(4);
  jint aa[4] = {0, 0, width, height};
  env->SetIntArrayRegion(activeArray, 0, 4, aa);

  jobject dngData = env->NewObject(
      dngDataClass, constructor, rawDataBuffer, width, height,
      static_cast<jint>(width * 6), 65535.0f, blackLevelArray, preMulArray,
      wbArray, colorMatrixArray, -1, 0, 0.0f, nullptr, 0, 0, 0.0f, 0, 0L,
      0.0f, activeArray, noiseProfileArray, nullptr);

  LOGI("processRawBufferNative: demosaiced %dx%d rowStride=%d lsc=%d",
       width, height, rowStride, hasLsc ? 1 : 0);
  LOGI("processRawBufferNative: wb RGGB=%f,%f,%f,%f", wb[0], wb[1], wb[2],
       wb[3]);
  return dngData;
}

/**
 * 使用 LibRaw 处理 DNG 文件
 */
JNIEXPORT jobject JNICALL
Java_com_hinnka_mycamera_raw_RawDemosaicProcessor_processDngNative(
    JNIEnv *env, jobject /* this */, jstring filePath, jfloat xr, jfloat yr,
    jfloat xg, jfloat yg, jfloat xb, jfloat yb, jfloat xw, jfloat yw,
    jboolean useRawAutoWhiteBalanceEstimate) {

  const char *path = env->GetStringUTFChars(filePath, nullptr);
  if (path == nullptr) {
    LOGE("Failed to get file path");
    return nullptr;
  }

  LibRaw RawProcessor;
  ExifData ed;
  RawProcessor.set_exifparser_handler(exif_callback, &ed);

  int ret = RawProcessor.open_file(path);
  if (ret != LIBRAW_SUCCESS) {
    LOGE("processDngNative: Failed to open file %s, ret=%d", path, ret);
    env->ReleaseStringUTFChars(filePath, path);
    return nullptr;
  }

  ret = RawProcessor.unpack();
  if (ret != LIBRAW_SUCCESS) {
    LOGE("processDngNative: Failed to unpack %s, ret=%d", path, ret);
    env->ReleaseStringUTFChars(filePath, path);
    return nullptr;
  }

  jobject embeddedPreviewBitmap = nullptr;
  /*ret = RawProcessor.unpack_thumb();
  if (ret == LIBRAW_SUCCESS) {
    libraw_processed_image_t *thumb = RawProcessor.dcraw_make_mem_thumb(&ret);
    if (thumb && ret == LIBRAW_SUCCESS) {
      if (thumb->type == LIBRAW_IMAGE_JPEG && thumb->data &&
          thumb->data_size > 0) {
        embeddedPreviewBitmap =
            decodeJpegPreviewToBitmap(env, thumb->data, thumb->data_size);
        LOGI("processDngNative: extracted embedded JPEG preview (%d bytes)",
             (int)thumb->data_size);
      } else if (thumb->type == LIBRAW_IMAGE_BITMAP) {
        embeddedPreviewBitmap = convertBitmapThumbnailToBitmap(env, thumb);
        if (embeddedPreviewBitmap) {
          LOGI("processDngNative: converted embedded bitmap preview %dx%d c=%d b=%d",
               thumb->width, thumb->height, thumb->colors, thumb->bits);
        } else {
          LOGI("processDngNative: failed to convert embedded bitmap preview %dx%d c=%d b=%d",
               thumb->width, thumb->height, thumb->colors, thumb->bits);
        }
      } else {
        LOGI("processDngNative: embedded preview present but unsupported type=%d",
             thumb->type);
      }
      LibRaw::dcraw_clear_mem(thumb);
    } else {
      LOGI("processDngNative: dcraw_make_mem_thumb failed ret=%d", ret);
    }
  } else {
    LOGI("processDngNative: unpack_thumb failed ret=%d err=%s", ret,
         libraw_strerror(ret));
  }*/

  // Read the TOTAL effective black level per channel.
  // LibRaw stores it as: color.black + cblack[channel] + repeat_pattern[position].
  float dngBlackLevels[4] = {};
  computeEffectiveBlackLevels(RawProcessor, dngBlackLevels);
  LOGI("dng black levels (metadata): %f %f %f %f", dngBlackLevels[0], dngBlackLevels[1], dngBlackLevels[2], dngBlackLevels[3]);
  LOGI("dng levels: black=%u cblack=%u,%u,%u,%u maximum=%u white=%u cam_mul RGBG=%f,%f,%f,%f",
       RawProcessor.imgdata.color.black, RawProcessor.imgdata.color.cblack[0],
       RawProcessor.imgdata.color.cblack[1], RawProcessor.imgdata.color.cblack[2],
       RawProcessor.imgdata.color.cblack[3], RawProcessor.imgdata.color.maximum,
       RawProcessor.imgdata.color.dng_levels.dng_whitelevel[0],
       RawProcessor.imgdata.color.cam_mul[0], RawProcessor.imgdata.color.cam_mul[1],
       RawProcessor.imgdata.color.cam_mul[2], RawProcessor.imgdata.color.cam_mul[3]);
  const bool appliedDngGainMap = applySupportedDngGainMaps(RawProcessor, dngBlackLevels);
  LOGI("dng gain map: opcode2_len=%u applied=%d", RawProcessor.imgdata.color.dng_levels.rawopcodes[1].len,
       appliedDngGainMap ? 1 : 0);

  RawProcessor.imgdata.params.output_bps = 16;
  RawProcessor.imgdata.params.gamm[0] = 1.0; // Linear
  RawProcessor.imgdata.params.gamm[1] = 1.0;
  RawProcessor.imgdata.params.no_auto_bright = 1;
  RawProcessor.imgdata.params.use_camera_wb = 1;
  RawProcessor.imgdata.params.output_color = 0; // Raw color space
  RawProcessor.imgdata.params.user_qual = 14;
  RawProcessor.imgdata.params.fbdd_noiserd = 0;
  RawProcessor.imgdata.params.threshold = 0;
  RawProcessor.imgdata.params.med_passes = 0;

  float selectedWb[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for (int i = 0; i < 4; i++) {
    const float val = RawProcessor.imgdata.color.cam_mul[i];
    selectedWb[i] = val > 0.0f && std::isfinite(val) ? val : 1.0f;
  }
  float selectedBase = selectedWb[1] > 0.0f ? selectedWb[1] : 1.0f;
  for (int i = 0; i < 4; i++) {
    selectedWb[i] /= selectedBase;
  }

  if (useRawAutoWhiteBalanceEstimate == JNI_TRUE) {
    float estimatedWb[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    int awbSamples = 0;
    if (estimateRawAutoWhiteBalance(RawProcessor, dngBlackLevels, estimatedWb, awbSamples)) {
      RawProcessor.imgdata.params.use_camera_wb = 0;
      RawProcessor.imgdata.params.use_auto_wb = 0;
      for (int i = 0; i < 4; ++i) {
        RawProcessor.imgdata.params.user_mul[i] = estimatedWb[i];
        selectedWb[i] = estimatedWb[i];
      }
      LOGI("raw awb estimate: enabled samples=%d wb=%f,%f,%f,%f", awbSamples,
           selectedWb[0], selectedWb[1], selectedWb[2], selectedWb[3]);
    } else {
      LOGI("raw awb estimate: enabled but insufficient samples, using camera wb");
    }
  } else {
    LOGI("raw awb estimate: disabled, using camera wb");
  }

  ret = RawProcessor.dcraw_process();
  if (ret != LIBRAW_SUCCESS) {
    LOGE("processDngNative: Failed to process %s, ret=%d", path, ret);
    env->ReleaseStringUTFChars(filePath, path);
    return nullptr;
  }

  // 获取处理后的图像
  libraw_processed_image_t *image = RawProcessor.dcraw_make_mem_image(&ret);
  if (!image || ret != 0) {
    LOGE("processDngNative: Failed to make mem image, ret=%d", ret);
    env->ReleaseStringUTFChars(filePath, path);
    return nullptr;
  }
  logProcessedImageStats("processDngNative", image);

  // 准备返回结果
  size_t outputSize = (size_t)image->width * image->height * 3 * 2;
  void *outData = malloc(outputSize);
  memcpy(outData, image->data, outputSize);
  jobject rawDataBuffer = env->NewDirectByteBuffer(outData, outputSize);

  // 提取元数据
  jclass dngDataClass = env->FindClass("com/hinnka/mycamera/raw/DngRawData");
  jmethodID constructor =
      env->GetMethodID(dngDataClass, "<init>",
                       "(Ljava/nio/ByteBuffer;IIIF[F[F[F[FIIF[FIIFIJF[I[FLandroid/graphics/Bitmap;)V");

  jfloatArray blackLevelArray = env->NewFloatArray(4);
  for (int i = 0; i < 4; i++) {
    float val = dngBlackLevels[i];
    env->SetFloatArrayRegion(blackLevelArray, i, 1, &val);
  }

  jfloatArray preMulArray = env->NewFloatArray(4);
  float preMul[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  for (int i = 0; i < 4; i++) {
    const float val = RawProcessor.imgdata.color.pre_mul[i];
    preMul[i] = val > 0.0f ? val : 1.0f;
  }
  env->SetFloatArrayRegion(preMulArray, 0, 4, preMul);

  // 白平衡
  jfloatArray wbArray = env->NewFloatArray(4);
  float wb[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  for (int i = 0; i < 4; i++)
    wb[i] = selectedWb[i];
  LOGI("wb: %f, %f, %f, %f", wb[0], wb[1], wb[2], wb[3]); // RGB0 or RGBG
  LOGI("cam_xyz: %f", RawProcessor.imgdata.color.cam_xyz[0][0]);
  LOGI("ccm: %f", RawProcessor.imgdata.color.ccm[0][0]);
  LOGI("cmatrix: %f", RawProcessor.imgdata.color.cmatrix[0][0]);
  LOGI("rgb_cam: %f", RawProcessor.imgdata.color.rgb_cam[0][0]);

  env->SetFloatArrayRegion(wbArray, 0, 1, &wb[0]);
  env->SetFloatArrayRegion(wbArray, 1, 1, &wb[1]);
  if (wb[3] > 0.0f) {
    env->SetFloatArrayRegion(wbArray, 2, 1, &wb[3]);
  } else {
    env->SetFloatArrayRegion(wbArray, 2, 1, &wb[1]);
  }
  env->SetFloatArrayRegion(wbArray, 3, 1, &wb[2]);

  // CCM (从 DNG ForwardMatrix 转换为目标色域)
  Matrix3x3 targetTransform =
      computeXYZD50ToGamut(xr, yr, xg, yg, xb, yb, xw, yw);
  Matrix3x3 m1 = Matrix3x3::identity();
  Matrix3x3 m2 = Matrix3x3::identity();
  bool hasM1 = false, hasM2 = false;

  auto getMatrix = [&](int index, Matrix3x3 &m) {
    float sumFM = 0.0f;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        m.m[i * 3 + j] =
            RawProcessor.imgdata.color.dng_color[index].forwardmatrix[i][j];
        sumFM += std::abs(m.m[i * 3 + j]);
      }
    }
    if (sumFM > 0.01f)
      return true;

    float sumCM = 0.0f;
    Matrix3x3 xyzToCam;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        xyzToCam.m[i * 3 + j] =
            RawProcessor.imgdata.color.dng_color[index].colormatrix[i][j];
        sumCM += std::abs(xyzToCam.m[i * 3 + j]);
      }
    }
    if (sumCM > 0.01f) {
      // 1. 确定参考光源的 XYZ 白点 (根据 DNG 规范)
      float lx, ly, lz;
      int ill = RawProcessor.imgdata.color.dng_color[index].illuminant;

      if (ill == 17) { // Standard Light A
        lx = 1.0985f;
        ly = 1.0000f;
        lz = 0.3558f;
      } else { // Assume D65
        lx = 0.9504f;
        ly = 1.0000f;
        lz = 1.0888f;
      }

      // 2. 计算相机对该光源的响应 (Camera Neutral / White Balance)
      // 这是 ColorMatrix 作用于光源 XYZ 的结果
      float cameraNeutral[3];
      for (int i = 0; i < 3; i++) {
        cameraNeutral[i] = xyzToCam.m[i * 3 + 0] * lx +
                           xyzToCam.m[i * 3 + 1] * ly +
                           xyzToCam.m[i * 3 + 2] * lz;
      }

      // 3. 构造中间矩阵：ColorMatrix * ReferenceDiagonal
      // 在 DNG 逻辑中，我们需要对 ColorMatrix 的每一列乘以对应的 CameraNeutral
      // 分量 这样做的目的是为了让矩阵在处理该光源下的“白点”时，输出为 [1, 1, 1]
      Matrix3x3 referenceMatrix = xyzToCam;
      for (int col = 0; col < 3; col++) {
        // 这一步非常关键：为了求逆后能还原，这里其实是预补偿白平衡
        referenceMatrix.m[0 * 3 + col] /= cameraNeutral[0];
        referenceMatrix.m[1 * 3 + col] /= cameraNeutral[1];
        referenceMatrix.m[2 * 3 + col] /= cameraNeutral[2];
      }

      // 4. 求逆：从 Camera 空间转回该光源下的 XYZ 空间
      // 现在 m 是从 Camera (White Balanced) -> XYZ (Illuminant Relative)
      m = referenceMatrix.invert();

      // 5. 应用色度适应 (Chromatic Adaptation) 映射到 D50
      // ForwardMatrix 必须映射到 D50 空间
      Matrix3x3 adapt;
      if (ill == 17) { // A to D50 (Bradford Transform)
        float a2d50[9] = {0.8924f,  -0.0157f, 0.0529f,  -0.1111f, 1.0505f,
                          -0.0151f, 0.0522f,  -0.0077f, 2.2396f};
        memcpy(adapt.m, a2d50, 9 * sizeof(float));
      } else { // D65 to D50 (Bradford Transform)
        float d652d50[9] = {1.0478f,  0.0229f,  -0.0501f, 0.0295f, 0.9905f,
                            -0.0170f, -0.0092f, 0.0150f,  0.7521f};
        memcpy(adapt.m, d652d50, 9 * sizeof(float));
      }
      m = adapt.multiply(m);
      return true;
    }
    return false;
  };

  hasM1 = getMatrix(0, m1);
  hasM2 = getMatrix(1, m2);

  LOGI("hasM1 = %d hasM2 = %d", hasM1, hasM2);

  float weight = 0.5f;
  if (hasM1 && hasM2) {
    float t1 =
        illuminantToTemp(RawProcessor.imgdata.color.dng_color[0].illuminant);
    float t2 =
        illuminantToTemp(RawProcessor.imgdata.color.dng_color[1].illuminant);
    float currentRatio = wb[0] / wb[2]; // R/B
    float rWarm = 0.5f, rCool = 1.6f;
    auto getTargetRatio = [&](float temp) {
      if (temp <= 2856.0f)
        return rWarm;
      if (temp >= 6504.0f)
        return rCool;
      return rWarm + (rCool - rWarm) * (temp - 2856.0f) / (6504.0f - 2856.0f);
    };
    float r1 = getTargetRatio(t1), r2 = getTargetRatio(t2);
    if (std::abs(r1 - r2) > 0.01f) {
      weight = (currentRatio - r2) / (r1 - r2);
      weight = std::max(0.0f, std::min(1.0f, weight));
    }
  }

  Matrix3x3 camToXYZ;
  if (hasM1 && hasM2) {
    for (int i = 0; i < 9; i++)
      camToXYZ.m[i] = m1.m[i] * weight + m2.m[i] * (1.0f - weight);
  } else if (hasM1)
    camToXYZ = m1;
  else if (hasM2)
    camToXYZ = m2;
  else {
    // 没有任何 DNG ForwardMatrix。
    // Bradford 变换 (D65 to D50)
    float d652d50[9] = {1.0478112f,  0.0228866f, -0.0501270f,
                        0.0295424f,  0.9904844f, -0.0170491f,
                        -0.0092345f, 0.0150436f, 0.7521316f};
    Matrix3x3 adapt;
    memcpy(adapt.m, d652d50, 9 * sizeof(float));

    Matrix3x3 xyzToCam;
    bool hasCamXYZ = false;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        // LibRaw 的 cam_xyz[chan][xyz] 是 XYZ-to-Camera 映射，与 ColorMatrix1/2
        // 结构相同
        xyzToCam.m[i * 3 + j] = RawProcessor.imgdata.color.cam_xyz[i][j];
        if (std::abs(xyzToCam.m[i * 3 + j]) > 0.0001f)
          hasCamXYZ = true;
      }
    }

    if (hasCamXYZ) {
      // 1. 视为 D65 下的 XYZ-to-Camera 矩阵进行标准化 (同 ColorMatrix 处理方式)
      float lx = 0.9504f, ly = 1.0000f, lz = 1.0888f; // D65
      float cameraNeutral[3];
      for (int i = 0; i < 3; i++) {
        cameraNeutral[i] = xyzToCam.m[i * 3 + 0] * lx +
                           xyzToCam.m[i * 3 + 1] * ly +
                           xyzToCam.m[i * 3 + 2] * lz;
      }
      Matrix3x3 referenceMatrix = xyzToCam;
      for (int col = 0; col < 3; col++) {
        if (std::abs(cameraNeutral[0]) > 0.001f)
          referenceMatrix.m[0 * 3 + col] /= cameraNeutral[0];
        if (std::abs(cameraNeutral[1]) > 0.001f)
          referenceMatrix.m[1 * 3 + col] /= cameraNeutral[1];
        if (std::abs(cameraNeutral[2]) > 0.001f)
          referenceMatrix.m[2 * 3 + col] /= cameraNeutral[2];
      }
      // 2. 求逆得到 Camera-to-XYZ
      camToXYZ = referenceMatrix.invert();
      // 3. 应用色度适应 D65 -> D50
      camToXYZ = adapt.multiply(camToXYZ);
      LOGI(
          "Using LibRaw cam_xyz (treated as ColorMatrix) converted to XYZ D50");
    } else {
      // 2. 尝试通过 LibRaw 的 ccm 反算。
      // 当 output_color = 0 时，ccm 通常仍然是针对默认 sRGB (D65) 的矩阵。
      Matrix3x3 camToSRGB;
      bool hasCCM = false;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          camToSRGB.m[i * 3 + j] = RawProcessor.imgdata.color.ccm[i][j];
          if (std::abs(camToSRGB.m[i * 3 + j]) > 0.0001f)
            hasCCM = true;
        }
      }

      if (hasCCM) {
        float srgb2xyz[9] = {0.4124564f, 0.3575761f, 0.1804375f,
                             0.2126729f, 0.7151522f, 0.0721750f,
                             0.0193339f, 0.1191920f, 0.9503041f};
        Matrix3x3 mSRGB2XYZ;
        memcpy(mSRGB2XYZ.m, srgb2xyz, 9 * sizeof(float));
        camToXYZ = adapt.multiply(mSRGB2XYZ.multiply(camToSRGB));
        LOGI("Using LibRaw ccm converted to XYZ D50");
      } else {
        // 3. 最后尝试 cmatrix (XYZ D65 to Camera)
        Matrix3x3 xyzToCam;
        bool hasCMatrix = false;
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            xyzToCam.m[i * 3 + j] = RawProcessor.imgdata.color.cmatrix[i][j];
            if (std::abs(xyzToCam.m[i * 3 + j]) > 0.0001f)
              hasCMatrix = true;
          }
        }

        if (hasCMatrix) {
          float lx = 0.9504f, ly = 1.0000f, lz = 1.0888f;
          float cameraNeutral[3];
          for (int i = 0; i < 3; i++) {
            cameraNeutral[i] = xyzToCam.m[i * 3 + 0] * lx +
                               xyzToCam.m[i * 3 + 1] * ly +
                               xyzToCam.m[i * 3 + 2] * lz;
          }
          Matrix3x3 referenceMatrix = xyzToCam;
          for (int col = 0; col < 3; col++) {
            if (std::abs(cameraNeutral[0]) > 0.001f)
              referenceMatrix.m[0 * 3 + col] /= cameraNeutral[0];
            if (std::abs(cameraNeutral[1]) > 0.001f)
              referenceMatrix.m[1 * 3 + col] /= cameraNeutral[1];
            if (std::abs(cameraNeutral[2]) > 0.001f)
              referenceMatrix.m[2 * 3 + col] /= cameraNeutral[2];
          }
          camToXYZ = referenceMatrix.invert();
          camToXYZ = adapt.multiply(camToXYZ);
          LOGI("Using cmatrix fallback converted to XYZ D50");
        } else {
          camToXYZ = Matrix3x3::identity();
          LOGE("No color metadata found at all, using identity");
        }
      }
    }
  }

  Matrix3x3 finalCCM = targetTransform.multiply(camToXYZ);
  jfloatArray colorMatrixArray = env->NewFloatArray(9);
  env->SetFloatArrayRegion(colorMatrixArray, 0, 9, finalCCM.m);

  LOGI("finalCCM: %f, %f, %f, %f, %f, %f, %f, %f, %f", finalCCM.m[0],
       finalCCM.m[1], finalCCM.m[2], finalCCM.m[3], finalCCM.m[4],
       finalCCM.m[5], finalCCM.m[6], finalCCM.m[7], finalCCM.m[8]);

  // 其它
  jint width = image->width;
  jint height = image->height;
  jint rowStride = width * 6; // RGB16
  jfloat whiteLevel =
      (jfloat)RawProcessor.imgdata.color.dng_levels.dng_whitelevel[0];
  if (whiteLevel <= 0)
    whiteLevel = (jfloat)RawProcessor.imgdata.color.maximum;
  jint cfaPattern = -1; // CFA_LINEAR_RGB

  jfloat baselineExposure =
      RawProcessor.imgdata.color.dng_levels.baseline_exposure;
  jfloat exposureBias =
      RawProcessor.imgdata.makernotes.common.ExposureCalibrationShift;
  int iso = RawProcessor.imgdata.other.iso_speed;
  if (iso == 0)
    iso = ed.iso;

  jlong shutterSpeedLong =
      (jlong)(RawProcessor.imgdata.other.shutter * 1e9); // ns
  jfloat aperture = RawProcessor.imgdata.other.aperture;

  LOGI("iso = %d, shutterSpeed = %lld aperture = %f baselineExposure = %f "
       "exposureBias = %f",
       iso, (long long)shutterSpeedLong, aperture, baselineExposure,
       exposureBias);

  // ActiveArray: use margins to define the actual active sensor area
  jintArray activeArray = env->NewIntArray(4);
  jint aa[4] = {(jint)RawProcessor.imgdata.sizes.left_margin,
                (jint)RawProcessor.imgdata.sizes.top_margin,
                (jint)RawProcessor.imgdata.sizes.left_margin +
                    (jint)RawProcessor.imgdata.sizes.width,
                (jint)RawProcessor.imgdata.sizes.top_margin +
                    (jint)RawProcessor.imgdata.sizes.height};
  env->SetIntArrayRegion(activeArray, 0, 4, aa);

  // LOGI("aa: %d, %d, %d, %d", aa[0], aa[1], aa[2], aa[3]);

  jfloatArray afRegions = nullptr;
  jfloatArray noiseProfileArray = nullptr;
  if (ed.hasNoiseProfile) {
    noiseProfileArray = env->NewFloatArray(8);
    env->SetFloatArrayRegion(noiseProfileArray, 0, 8, ed.noiseProfile);
  }

  jobject dngData = env->NewObject(
      dngDataClass, constructor, rawDataBuffer, width, height, rowStride,
      whiteLevel, blackLevelArray, preMulArray, wbArray, colorMatrixArray,
      cfaPattern, 0, baselineExposure, nullptr, 0, 0, exposureBias, iso,
      shutterSpeedLong, aperture, activeArray, noiseProfileArray,
      embeddedPreviewBitmap);

  // 释放资源
  LibRaw::dcraw_clear_mem(image);
  env->ReleaseStringUTFChars(filePath, path);

  return dngData;
}

/**
 * 释放 DNG RAW 数据的 native 内存
 */
JNIEXPORT void JNICALL Java_com_hinnka_mycamera_raw_DngRawData_freeNativeBuffer(
    JNIEnv *env, jobject /* this */, jobject rawDataBuffer) {
  if (rawDataBuffer == nullptr)
    return;
  void *nativePtr = env->GetDirectBufferAddress(rawDataBuffer);
  if (nativePtr != nullptr) {
    LOGI("Freeing DNG RAW data native buffer: %p", nativePtr);
    free(nativePtr);
  }
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_ml_DnCNNDenoiseEstimator_preprocessNative(
    JNIEnv *env, jobject, jobject bitmap, jint x, jint y, jint w, jint h,
    jobject outBuffer, jboolean isRgb, jboolean channelsFirst) {

  AndroidBitmapInfo info;
  void *pixels = nullptr;
  if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 ||
      AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
    LOGE("DnCNN preprocess: failed to lock bitmap");
    return;
  }

  auto *out = static_cast<float *>(env->GetDirectBufferAddress(outBuffer));
  if (!out || w <= 0 || h <= 0 || info.width <= 0 || info.height <= 0) {
    AndroidBitmap_unlockPixels(env, bitmap);
    return;
  }

  const int stride = static_cast<int>(info.stride / 4);
  const auto *src = static_cast<const uint32_t *>(pixels);
  const bool rgb = isRgb == JNI_TRUE;
  const bool nchw = channelsFirst == JNI_TRUE;
  const int pixelCount = w * h;

#pragma omp parallel for num_threads(8)
  for (int py = 0; py < h; ++py) {
    for (int px = 0; px < w; ++px) {
      const int sx = std::clamp(static_cast<int>(x + px), 0,
                                static_cast<int>(info.width) - 1);
      const int sy = std::clamp(static_cast<int>(y + py), 0,
                                static_cast<int>(info.height) - 1);
      const uint32_t pixel = src[sy * stride + sx];
      const float r = static_cast<float>((pixel >> 0) & 0xFF) / 255.0f;
      const float g = static_cast<float>((pixel >> 8) & 0xFF) / 255.0f;
      const float b = static_cast<float>((pixel >> 16) & 0xFF) / 255.0f;
      const int base = py * w + px;

      if (rgb) {
        if (nchw) {
          out[base] = r;
          out[pixelCount + base] = g;
          out[pixelCount * 2 + base] = b;
        } else {
          const int outIdx = base * 3;
          out[outIdx] = r;
          out[outIdx + 1] = g;
          out[outIdx + 2] = b;
        }
      } else {
        out[base] = 0.299f * r + 0.587f * g + 0.114f * b;
      }
    }
  }

  AndroidBitmap_unlockPixels(env, bitmap);
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_ml_DnCNNDenoiseEstimator_postprocessNative(
    JNIEnv *env, jobject, jobject inBuffer, jobject srcBitmap, jobject dstBitmap,
    jint patchX, jint patchY, jint srcX, jint srcY, jint dstX, jint dstY,
    jint w, jint h, jint patchW, jint patchH, jfloat strength, jboolean isRgb,
    jboolean channelsFirst) {

  AndroidBitmapInfo srcInfo, dstInfo;
  void *srcPixels = nullptr;
  void *dstPixels = nullptr;
  if (AndroidBitmap_getInfo(env, srcBitmap, &srcInfo) < 0 ||
      AndroidBitmap_lockPixels(env, srcBitmap, &srcPixels) < 0) {
    LOGE("DnCNN postprocess: failed to lock source bitmap");
    return;
  }
  if (AndroidBitmap_getInfo(env, dstBitmap, &dstInfo) < 0 ||
      AndroidBitmap_lockPixels(env, dstBitmap, &dstPixels) < 0) {
    AndroidBitmap_unlockPixels(env, srcBitmap);
    LOGE("DnCNN postprocess: failed to lock destination bitmap");
    return;
  }

  auto *in = static_cast<float *>(env->GetDirectBufferAddress(inBuffer));
  if (!in || w <= 0 || h <= 0 || patchW <= 0 || patchH <= 0 ||
      srcInfo.width <= 0 || srcInfo.height <= 0 || dstInfo.width <= 0 ||
      dstInfo.height <= 0) {
    AndroidBitmap_unlockPixels(env, srcBitmap);
    AndroidBitmap_unlockPixels(env, dstBitmap);
    return;
  }

  const int srcStride = static_cast<int>(srcInfo.stride / 4);
  const int dstStride = static_cast<int>(dstInfo.stride / 4);
  const auto *src = static_cast<const uint32_t *>(srcPixels);
  auto *dst = static_cast<uint32_t *>(dstPixels);
  const bool rgb = isRgb == JNI_TRUE;
  const bool nchw = channelsFirst == JNI_TRUE;
  const float blend = std::clamp(static_cast<float>(strength), 0.0f, 1.0f);
  const float invBlend = 1.0f - blend;
  const int patchPixelCount = patchW * patchH;

#pragma omp parallel for num_threads(8)
  for (int py = 0; py < h; ++py) {
    for (int px = 0; px < w; ++px) {
      const int srcPx = std::clamp(static_cast<int>(srcX + px), 0,
                                   static_cast<int>(srcInfo.width) - 1);
      const int srcPy = std::clamp(static_cast<int>(srcY + py), 0,
                                   static_cast<int>(srcInfo.height) - 1);
      const int dstPx = static_cast<int>(dstX + px);
      const int dstPy = static_cast<int>(dstY + py);
      if (dstPx < 0 || dstPy < 0 || dstPx >= static_cast<int>(dstInfo.width) ||
          dstPy >= static_cast<int>(dstInfo.height)) {
        continue;
      }

      const int patchPx = std::clamp(static_cast<int>(patchX + px), 0,
                                     static_cast<int>(patchW) - 1);
      const int patchPy = std::clamp(static_cast<int>(patchY + py), 0,
                                     static_cast<int>(patchH) - 1);
      const int patchBase = patchPy * patchW + patchPx;
      const uint32_t pixel = src[srcPy * srcStride + srcPx];
      const float origR = static_cast<float>((pixel >> 0) & 0xFF);
      const float origG = static_cast<float>((pixel >> 8) & 0xFF);
      const float origB = static_cast<float>((pixel >> 16) & 0xFF);

      float r;
      float g;
      float b;
      if (rgb) {
        float denR;
        float denG;
        float denB;
        if (nchw) {
          denR = in[patchBase] * 255.0f;
          denG = in[patchPixelCount + patchBase] * 255.0f;
          denB = in[patchPixelCount * 2 + patchBase] * 255.0f;
        } else {
          const int inIdx = patchBase * 3;
          denR = in[inIdx] * 255.0f;
          denG = in[inIdx + 1] * 255.0f;
          denB = in[inIdx + 2] * 255.0f;
        }

        r = origR * invBlend + denR * blend;
        g = origG * invBlend + denG * blend;
        b = origB * invBlend + denB * blend;
      } else {
        const float cb =
            -0.1687f * origR - 0.3313f * origG + 0.5f * origB + 128.0f;
        const float cr =
            0.5f * origR - 0.4187f * origG - 0.0813f * origB + 128.0f;
        const float originalY = 0.299f * origR + 0.587f * origG + 0.114f * origB;
        const float denoisedY = in[patchBase] * 255.0f;
        const float newY = originalY * invBlend + denoisedY * blend;

        r = newY + 1.402f * (cr - 128.0f);
        g = newY - 0.344136f * (cb - 128.0f) - 0.714136f * (cr - 128.0f);
        b = newY + 1.772f * (cb - 128.0f);
      }

      const auto resR = static_cast<uint32_t>(std::clamp(r, 0.0f, 255.0f));
      const auto resG = static_cast<uint32_t>(std::clamp(g, 0.0f, 255.0f));
      const auto resB = static_cast<uint32_t>(std::clamp(b, 0.0f, 255.0f));
      dst[dstPy * dstStride + dstPx] =
          0xFF000000u | (resB << 16) | (resG << 8) | resR;
    }
  }

  AndroidBitmap_unlockPixels(env, srcBitmap);
  AndroidBitmap_unlockPixels(env, dstBitmap);
}

JNIEXPORT jobject JNICALL
Java_com_hinnka_mycamera_utils_DirectBufferAllocator_allocateNative(
    JNIEnv *env, jobject, jlong capacity) {
  void* ptr = malloc(capacity);
  if (!ptr) {
    LOGE("Failed to allocate %lld bytes", (long long)capacity);
    return nullptr;
  }
  return env->NewDirectByteBuffer(ptr, capacity);
}

JNIEXPORT void JNICALL
Java_com_hinnka_mycamera_utils_DirectBufferAllocator_freeNative(
    JNIEnv *env, jobject, jobject buffer) {
  if (!buffer) return;
  void* ptr = env->GetDirectBufferAddress(buffer);
  if (ptr) {
    free(ptr);
  }
}

} // extern "C"
