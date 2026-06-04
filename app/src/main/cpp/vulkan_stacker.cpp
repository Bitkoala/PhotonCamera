#include "vulkan_stacker.h"
#include "accumulate.comp.h"
#include "common.h"
#include "normalize.comp.h"
#include "structure_tensor.comp.h"
#include "yuv_to_rgba.comp.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = x;                                                          \
    if (err) {                                                                 \
      LOGE("Detected Vulkan error: %d at %s:%d", err, __FILE__, __LINE__);     \
      throw std::runtime_error("Vulkan call failed");                          \
    }                                                                          \
  } while (0)

namespace {

struct ScalarStats {
  float minValue = 0.0f;
  float maxValue = 0.0f;
  float p50 = 0.0f;
  float p90 = 0.0f;
};

struct LocalReliabilityMap {
  std::vector<float> tileWeights;
  float flatGoodFraction = 0.0f;
  float flatAreaFraction = 0.0f;
  float detailAreaFraction = 0.0f;
};

constexpr bool kEnableYuvStackPerfLogs = false;

inline void logYuvStackStage(const char *stage, double elapsedMs) {
  if (!kEnableYuvStackPerfLogs)
    return;
  LOGI("YuvStackPerf[%s]: %.3f ms", stage, elapsedMs);
}

#undef TIME_START
#undef TIME_END
#define TIME_START(name) const auto start_##name = std::chrono::steady_clock::now()
#define TIME_END(name)                                                         \
  const double elapsed_##name = std::chrono::duration<double, std::milli>(     \
                                     std::chrono::steady_clock::now() -        \
                                     start_##name)                             \
                                     .count();                                 \
  logYuvStackStage(#name, elapsed_##name)

inline float clamp01(float v) { return std::clamp(v, 0.0f, 1.0f); }

inline float smoothstepf(float edge0, float edge1, float x) {
  float width = std::max(edge1 - edge0, 1e-6f);
  float t = clamp01((x - edge0) / width);
  return t * t * (3.0f - 2.0f * t);
}

inline VkDeviceSize yuvAccumTileBytes(uint32_t fullW, uint32_t fullH,
                                      int tilesX, int tilesY) {
  uint32_t tileW = (fullW + tilesX - 1) / tilesX;
  uint32_t tileH = (fullH + tilesY - 1) / tilesY;
  return (VkDeviceSize)(tileW + 16) * (tileH + 16) * sizeof(float) * 4;
}

inline VkDeviceSize yuvKernelTileRangeBytes(uint32_t width, uint32_t height,
                                            int tilesY) {
  uint32_t tileH = (height + tilesY - 1) / tilesY;
  return (VkDeviceSize)width * tileH * sizeof(float) * 4;
}

void constrainYuvTileGridForStorageRange(uint32_t width, uint32_t height,
                                         uint32_t scale,
                                         VkDeviceSize maxStorageBufferRange,
                                         int &tilesX, int &tilesY) {
  if (maxStorageBufferRange == 0)
    return;

  const uint32_t fullW = width * scale;
  const uint32_t fullH = height * scale;
  while (yuvAccumTileBytes(fullW, fullH, tilesX, tilesY) >
             maxStorageBufferRange ||
         yuvKernelTileRangeBytes(width, height, tilesY) >
             maxStorageBufferRange) {
    if (yuvKernelTileRangeBytes(width, height, tilesY) >
        maxStorageBufferRange) {
      ++tilesY;
    } else {
      uint32_t tileW = (fullW + tilesX - 1) / tilesX;
      uint32_t tileH = (fullH + tilesY - 1) / tilesY;
      if (tileW >= tileH) {
        ++tilesX;
      } else {
        ++tilesY;
      }
    }

    if (tilesX > 16 || tilesY > 16) {
      throw std::runtime_error("Unable to tile Vulkan stacker buffers");
    }
  }
}

inline ScalarStats computeScalarStats(const std::vector<float> &values) {
  ScalarStats stats;
  if (values.empty())
    return stats;
  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  auto percentile = [&](float q) -> float {
    size_t idx =
        std::min(sorted.size() - 1, (size_t)std::floor(q * (sorted.size() - 1)));
    return sorted[idx];
  };
  stats.minValue = sorted.front();
  stats.maxValue = sorted.back();
  stats.p50 = percentile(0.50f);
  stats.p90 = percentile(0.90f);
  return stats;
}

LocalReliabilityMap buildLocalReliabilityMap(const GrayImage &reference,
                                             const std::vector<float> &errorMap,
                                             uint32_t gridW, uint32_t gridH) {
  LocalReliabilityMap result;
  if (reference.data.empty() || errorMap.empty() || gridW == 0 || gridH == 0)
    return result;

  result.tileWeights.resize((size_t)gridW * gridH, 1.0f);
  std::vector<float> detailScores;
  detailScores.reserve((size_t)gridW * gridH);
  uint32_t tileW = (reference.width + gridW - 1) / gridW;
  uint32_t tileH = (reference.height + gridH - 1) / gridH;

  for (uint32_t gy = 0; gy < gridH; ++gy) {
    for (uint32_t gx = 0; gx < gridW; ++gx) {
      uint32_t x0 = gx * tileW;
      uint32_t y0 = gy * tileH;
      uint32_t x1 = std::min<uint32_t>(x0 + tileW, reference.width);
      uint32_t y1 = std::min<uint32_t>(y0 + tileH, reference.height);
      double detailSum = 0.0;
      size_t count = 0;
      for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = x0; x < x1; ++x) {
          size_t idx = (size_t)y * reference.width + x;
          float center = (float)reference.data[idx];
          float xp = (float)reference
                         .data[(size_t)y * reference.width +
                               std::min<uint32_t>(x + 1, reference.width - 1)];
          float xm = (float)reference
                         .data[(size_t)y * reference.width + (x > 0 ? x - 1 : 0)];
          float yp = (float)reference.data[(size_t)std::min<uint32_t>(
                                               y + 1, reference.height - 1) *
                                               reference.width +
                                           x];
          float ym = (float)reference
                         .data[(size_t)(y > 0 ? y - 1 : 0) * reference.width + x];
          detailSum += std::abs(xp - xm) + std::abs(yp - ym) +
                       0.5 * std::abs(4.0f * center - xp - xm - yp - ym);
          ++count;
        }
      }
      float detail = (count > 0) ? (float)(detailSum / (double)count) : 0.0f;
      detailScores.push_back(detail);
    }
  }

  ScalarStats detailStats = computeScalarStats(detailScores);
  float flatThreshold = detailStats.minValue +
                        (detailStats.p50 - detailStats.minValue) * 0.70f;
  float detailThreshold = std::max(
      detailStats.p50, detailStats.minValue +
                           (detailStats.p90 - detailStats.minValue) * 0.55f);
  size_t flatCount = 0;
  size_t flatGood = 0;
  size_t detailCount = 0;

  for (size_t idx = 0; idx < errorMap.size() && idx < detailScores.size(); ++idx) {
    float detail = detailScores[idx];
    float err = errorMap[idx];
    float errRms = std::sqrt(std::max(err, 0.0f));
    float detailPenalty = smoothstepf(flatThreshold, detailThreshold, detail);
    float errorWeight = clamp01(1.0f - std::max(0.0f, errRms - 2.0f) / 4.5f);
    float tileWeight = clamp01(errorWeight);

    if (detailPenalty > 0.60f) {
      ++detailCount;
    } else {
      ++flatCount;
      if (errRms < 4.0f)
        ++flatGood;
    }

    if (errRms > 5.5f) {
      tileWeight *= 0.60f;
    }

    if (detailPenalty > 0.78f && errRms > 7.5f) {
      tileWeight = 0.0f;
    }

    result.tileWeights[idx] = clamp01(tileWeight);
  }

  float tileCount = static_cast<float>(detailScores.size());
  result.flatGoodFraction =
      (flatCount > 0) ? (float)flatGood / (float)flatCount : 0.0f;
  result.flatAreaFraction = (tileCount > 0.0f) ? (float)flatCount / tileCount : 0.0f;
  result.detailAreaFraction =
      (tileCount > 0.0f) ? (float)detailCount / tileCount : 0.0f;
  return result;
}

float computeFrameFusionWeight(const std::vector<float> &errorMap,
                               float sharpnessRatio,
                               const LocalReliabilityMap &localMap) {
  if (errorMap.empty())
    return clamp01(0.70f + 0.25f * sharpnessRatio);

  ScalarStats errStats = computeScalarStats(errorMap);
  float errP90Rms = std::sqrt(std::max(errStats.p90, 0.0f));
  float errorWeight =
      clamp01(1.0f - std::max(0.0f, errP90Rms - 2.5f) / 5.0f);
  float sharpnessGuard = clamp01(0.88f + 0.12f * sharpnessRatio);
  float flatCoverage = clamp01(0.75f + 0.25f * localMap.flatAreaFraction);
  float flatReliability = clamp01(0.44f + 0.55f * localMap.flatGoodFraction);
  float detailPenalty = clamp01(1.0f - 0.15f * localMap.detailAreaFraction);
  float baseWeight =
      0.6f + 0.40f * errorWeight * flatCoverage * flatReliability * detailPenalty;
  return clamp01(baseWeight * sharpnessGuard);
}

} // namespace

void YuvStackPerfStats::logSummary(uint32_t outputW, uint32_t outputH,
                                  float scale) const {
  if (!kEnableYuvStackPerfLogs)
    return;
  LOGI("YuvStackPerf summary: total=%.3f ms output=%ux%u scale=%.2f frames=%zu "
       "kept=%zu skipped=%zu effFrames=%.2f",
       totalMs, outputW, outputH, scale, totalFrames, keptFrames, skippedFrames,
       totalEffectiveFrames);
  LOGI("YuvStackPerf phases: score=%.3f process=%.3f norm=%.3f copy=%.3f",
       scoreCalculationMs, allFramesProcessingMs, normalizationDispatchMs,
       outputCopyMs);
}

VulkanImageStacker::VulkanImageStacker(uint32_t w, uint32_t h, bool sr)
    : width(w), height(h), enableSuperRes(sr) {

  if (enableSuperRes) {
    numTilesX = 2;
    numTilesY = 2;
  } else {
    numTilesX = 1;
    numTilesY = 1;
  }

  if (width == 0 || height == 0) {
    throw std::runtime_error("Invalid Vulkan stacker dimensions");
  }
  if (!VulkanManager::getInstance().init()) {
    throw std::runtime_error("Failed to initialize VulkanManager");
  }
  try {
    initVulkanResources();
  } catch (...) {
    releaseVulkanResources();
    throw;
  }
}

VulkanImageStacker::~VulkanImageStacker() {
  releasePendingFrames();
  releaseVulkanResources();
}

void VulkanImageStacker::initVulkanResources() {
  VulkanManager &vm = VulkanManager::getInstance();
  VkDevice device = vm.getDevice();
  VkPhysicalDevice physicalDevice = vm.getPhysicalDevice();
  if (device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE) {
    throw std::runtime_error("VulkanManager is not ready");
  }

  VkPhysicalDeviceProperties deviceProperties{};
  vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
  LOGI("initVulkanResources: device=%s vendor=0x%x device=0x%x api=%u.%u.%u "
       "maxStorageBufferRange=%u",
       deviceProperties.deviceName, deviceProperties.vendorID,
       deviceProperties.deviceID, VK_VERSION_MAJOR(deviceProperties.apiVersion),
       VK_VERSION_MINOR(deviceProperties.apiVersion),
       VK_VERSION_PATCH(deviceProperties.apiVersion),
       deviceProperties.limits.maxStorageBufferRange);

  // Create Accumulator Tiles
  uint32_t scale = enableSuperRes ? 2 : 1;
  uint32_t fullW = width * scale;
  uint32_t fullH = height * scale;
  VkDeviceSize maxStorageBufferRange =
      deviceProperties.limits.maxStorageBufferRange;

  constrainYuvTileGridForStorageRange(width, height, scale,
                                      maxStorageBufferRange, numTilesX,
                                      numTilesY);

  // Each tile is roughly fullW / numTilesX ...
  uint32_t tileW = (fullW + numTilesX - 1) / numTilesX;
  uint32_t tileH = (fullH + numTilesY - 1) / numTilesY;

  // Pad tile size slightly for safety
  VkDeviceSize accumBufferSize =
      (VkDeviceSize)(tileW + 16) * (tileH + 16) * sizeof(float) * 4;
  VkDeviceSize stagingSize = (VkDeviceSize)fullW * fullH * 4;
  VkDeviceSize kpSize =
      (VkDeviceSize)width * height * sizeof(float) * 4; // vec4
  VkDeviceSize kpTileRangeSize = yuvKernelTileRangeBytes(width, height, numTilesY);
  if (maxStorageBufferRange > 0 &&
      (accumBufferSize > maxStorageBufferRange ||
       kpTileRangeSize > maxStorageBufferRange)) {
    LOGE("Vulkan stacker buffers exceed device storage buffer range. "
         "accum=%llu kernelRange=%llu limit=%llu size=%ux%u SR=%d tiles=%dx%d",
         (unsigned long long)accumBufferSize,
         (unsigned long long)kpTileRangeSize,
         (unsigned long long)maxStorageBufferRange, width, height,
         enableSuperRes ? 1 : 0, numTilesX, numTilesY);
    throw std::runtime_error("Vulkan storage buffer range is too small");
  }

  int numTiles = numTilesX * numTilesY;
  LOGI("initVulkanResources: Tiles: %d (%dx%d) accum=%llu staging=%llu "
       "kernel=%llu kernelRange=%llu",
       numTiles, numTilesX, numTilesY, (unsigned long long)accumBufferSize,
       (unsigned long long)stagingSize, (unsigned long long)kpSize,
       (unsigned long long)kpTileRangeSize);

  accumBuffers.resize(numTiles);
  accumMemories.resize(numTiles);
  accumSets.resize(numTiles);
  normalizeSets.resize(numTiles);

  for (int i = 0; i < numTiles; ++i) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = accumBufferSize;
    bufferInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &accumBuffers[i]));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, accumBuffers[i], &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = vm.findMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &accumMemories[i]));
    vkBindBufferMemory(device, accumBuffers[i], accumMemories[i], 0);
  }

  // Initialize Alignment Grid Buffer early
  // Use finer grid (16px tiles) for super-res to capture sub-pixel variation
  uint32_t alignTileSize = enableSuperRes ? 16 : 32;
  gridW = (width + alignTileSize - 1) / alignTileSize;
  gridH = (height + alignTileSize - 1) / alignTileSize;
  VkDeviceSize alignBufferSize = gridW * gridH * sizeof(Point);
  VkBufferCreateInfo alignInfo{};
  alignInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  alignInfo.size = alignBufferSize;
  alignInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  alignInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &alignInfo, nullptr, &alignmentBuffer));

  VkMemoryRequirements alignMemReqs;
  vkGetBufferMemoryRequirements(device, alignmentBuffer, &alignMemReqs);
  VkMemoryAllocateInfo alignAlloc{};
  alignAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alignAlloc.allocationSize = alignMemReqs.size;
  alignAlloc.memoryTypeIndex = vm.findMemoryType(
      alignMemReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(device, &alignAlloc, nullptr, &alignmentMemory));
  vkBindBufferMemory(device, alignmentBuffer, alignmentMemory, 0);

  VkBufferCreateInfo alignUploadInfo{};
  alignUploadInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  alignUploadInfo.size = alignBufferSize;
  alignUploadInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  alignUploadInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(
      vkCreateBuffer(device, &alignUploadInfo, nullptr, &alignmentUploadBuffer));

  VkMemoryRequirements alignUploadReqs;
  vkGetBufferMemoryRequirements(device, alignmentUploadBuffer,
                                &alignUploadReqs);
  VkMemoryAllocateInfo alignUploadAlloc{};
  alignUploadAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alignUploadAlloc.allocationSize = alignUploadReqs.size;
  alignUploadAlloc.memoryTypeIndex = vm.findMemoryType(
      alignUploadReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(device, &alignUploadAlloc, nullptr,
                            &alignmentUploadMemory));
  vkBindBufferMemory(device, alignmentUploadBuffer, alignmentUploadMemory, 0);

  // Initial Clear for Alignment Buffer
  void *ptr;
  vkMapMemory(device, alignmentUploadMemory, 0, alignBufferSize, 0, &ptr);
  memset(ptr, 0, (size_t)alignBufferSize);
  vkUnmapMemory(device, alignmentUploadMemory);

  // Staging Buffer
  VkBufferCreateInfo stagingInfo{};
  stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  stagingInfo.size = stagingSize;
  stagingInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VK_CHECK(vkCreateBuffer(device, &stagingInfo, nullptr, &stagingBuffer));

  VkMemoryRequirements stagingReqs;
  vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingReqs);

  VkMemoryAllocateInfo stagingAlloc{};
  stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  stagingAlloc.allocationSize = stagingReqs.size;
  stagingAlloc.memoryTypeIndex = vm.findMemoryType(
      stagingReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VK_CHECK(vkAllocateMemory(device, &stagingAlloc, nullptr, &stagingMemory));
  vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

  // Descriptor Pool
  VkDescriptorPoolSize poolSizes[3] = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[0].descriptorCount = (uint32_t)numTiles * 64;
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[1].descriptorCount = (uint32_t)numTiles * 64;
  poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  poolSizes[2].descriptorCount = (uint32_t)numTiles * 64;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  poolInfo.maxSets = (uint32_t)numTiles * 64;
  poolInfo.poolSizeCount = 3;
  poolInfo.pPoolSizes = poolSizes;
  VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));

  gpuTimestampPeriodNs = deviceProperties.limits.timestampPeriod;
  gpuTimestampSupported =
      (deviceProperties.limits.timestampComputeAndGraphics == VK_TRUE);
  if (gpuTimestampSupported) {
    VkQueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = 8;
    VK_CHECK(vkCreateQueryPool(device, &queryPoolInfo, nullptr,
                               &gpuTimingQueryPool));
  }

  // Initial Clear
  VkCommandBuffer cb = vm.beginSingleTimeCommands();
  for (int i = 0; i < numTiles; ++i) {
    vkCmdFillBuffer(cb, accumBuffers[i], 0, accumBufferSize, 0);
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.buffer = accumBuffers[i];
    barrier.size = accumBufferSize;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
  }
  vm.endSingleTimeCommands(cb);

  // Phase 2: Kernel Params Buffer (Native Resolution - much safer for VRAM)
  VkBufferCreateInfo kpInfo{};
  kpInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  kpInfo.size = kpSize;
  kpInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  kpInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &kpInfo, nullptr, &kernelParamsBuffer));

  VkMemoryRequirements kpReqs;
  vkGetBufferMemoryRequirements(device, kernelParamsBuffer, &kpReqs);

  VkMemoryAllocateInfo kpAlloc{};
  kpAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  kpAlloc.allocationSize = kpReqs.size;
  kpAlloc.memoryTypeIndex = vm.findMemoryType(
      kpReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(device, &kpAlloc, nullptr, &kernelParamsMemory));
  vkBindBufferMemory(device, kernelParamsBuffer, kernelParamsMemory, 0);

  // Allocate Descriptor Sets for Structure Tensor (One per tile? No, global
  // pass or tile based?) For simplicity, let's make Structure Tensor a global
  // pass or tile based. The shader uses inputSampler and kernelParams. Let's
  // use 1 descriptor set for the whole image if possible, or per tile. Since we
  // run ST on full image (or tiles), let's allocate 'numTiles' sets for it too
  // to be consistent with dispatch pattern.
  tensorSets.resize(numTiles);

  // Phase 3: Motion Prior Buffer (Grid resolution, float)
  // Grid size: max (width+31)/32 * (height+31)/32
  // We reuse the 'gridW * gridH' logic, but max possible grid size is
  // alignedW/32 * alignedH/32
  VkDeviceSize mpSize = (VkDeviceSize)gridW * gridH * sizeof(float);

  VkBufferCreateInfo mpInfo{};
  mpInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  mpInfo.size = mpSize;
  mpInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  mpInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &mpInfo, nullptr, &motionPriorBuffer));

  VkMemoryRequirements mpReqs;
  vkGetBufferMemoryRequirements(device, motionPriorBuffer, &mpReqs);

  VkMemoryAllocateInfo mpAlloc{};
  mpAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mpAlloc.allocationSize = mpReqs.size;
  mpAlloc.memoryTypeIndex = vm.findMemoryType(
      mpReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(device, &mpAlloc, nullptr, &motionPriorMemory));
  vkBindBufferMemory(device, motionPriorBuffer, motionPriorMemory, 0);

  VkBufferCreateInfo mpUploadInfo{};
  mpUploadInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  mpUploadInfo.size = mpSize;
  mpUploadInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  mpUploadInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &mpUploadInfo, nullptr,
                          &motionPriorUploadBuffer));

  VkMemoryRequirements mpUploadReqs;
  vkGetBufferMemoryRequirements(device, motionPriorUploadBuffer,
                                &mpUploadReqs);
  VkMemoryAllocateInfo mpUploadAlloc{};
  mpUploadAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mpUploadAlloc.allocationSize = mpUploadReqs.size;
  mpUploadAlloc.memoryTypeIndex = vm.findMemoryType(
      mpUploadReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(device, &mpUploadAlloc, nullptr,
                            &motionPriorUploadMemory));
  vkBindBufferMemory(device, motionPriorUploadBuffer, motionPriorUploadMemory,
                     0);

  VkBufferCreateInfo lmInfo{};
  lmInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  lmInfo.size = mpSize;
  lmInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  lmInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &lmInfo, nullptr, &localTileMaskBuffer));

  VkMemoryRequirements lmReqs;
  vkGetBufferMemoryRequirements(device, localTileMaskBuffer, &lmReqs);

  VkMemoryAllocateInfo lmAlloc{};
  lmAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  lmAlloc.allocationSize = lmReqs.size;
  lmAlloc.memoryTypeIndex = vm.findMemoryType(
      lmReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(device, &lmAlloc, nullptr, &localTileMaskMemory));
  vkBindBufferMemory(device, localTileMaskBuffer, localTileMaskMemory, 0);

  VkBufferCreateInfo lmUploadInfo{};
  lmUploadInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  lmUploadInfo.size = mpSize;
  lmUploadInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  lmUploadInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &lmUploadInfo, nullptr,
                          &localTileMaskUploadBuffer));

  VkMemoryRequirements lmUploadReqs;
  vkGetBufferMemoryRequirements(device, localTileMaskUploadBuffer,
                                &lmUploadReqs);
  VkMemoryAllocateInfo lmUploadAlloc{};
  lmUploadAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  lmUploadAlloc.allocationSize = lmUploadReqs.size;
  lmUploadAlloc.memoryTypeIndex = vm.findMemoryType(
      lmUploadReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(device, &lmUploadAlloc, nullptr,
                            &localTileMaskUploadMemory));
  vkBindBufferMemory(device, localTileMaskUploadBuffer,
                     localTileMaskUploadMemory, 0);

  // Initialize RGB Intermediate Image
  rgbFrame.width = width;
  rgbFrame.height = height;
  rgbFrame.format = VK_FORMAT_R16G16B16A16_SFLOAT;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = rgbFrame.format;
  imageInfo.extent = {width, height, 1};
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &rgbFrame.image));

  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(device, rgbFrame.image, &memReqs);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = vm.findMemoryType(
      memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &rgbFrame.memory));
  vkBindImageMemory(device, rgbFrame.image, rgbFrame.memory, 0);

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = rgbFrame.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = rgbFrame.format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &rgbFrame.viewY));

  // Initialize Reference RGB Image
  referenceRgbFrame.width = width;
  referenceRgbFrame.height = height;
  referenceRgbFrame.format = VK_FORMAT_R16G16B16A16_SFLOAT;

  imageInfo.format = referenceRgbFrame.format;
  VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &referenceRgbFrame.image));

  vkGetImageMemoryRequirements(device, referenceRgbFrame.image, &memReqs);
  allocInfo.allocationSize = memReqs.size;
  VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &referenceRgbFrame.memory));
  vkBindImageMemory(device, referenceRgbFrame.image, referenceRgbFrame.memory, 0);

  viewInfo.image = referenceRgbFrame.image;
  viewInfo.format = referenceRgbFrame.format;
  VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &referenceRgbFrame.viewY));

  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

  VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &rgbFrame.sampler));
  VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &referenceRgbFrame.sampler));
}

void VulkanImageStacker::createPipelines(VkSampler sampler) {
  VulkanManager &vm = VulkanManager::getInstance();
  VkDevice device = vm.getDevice();

  // 1. Descriptor Set Layout with Immutable Sampler
  VkSampler samplers[1] = {sampler};
  VkDescriptorSetLayoutBinding bindings[4] = {}; // Increased to 4
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[0].pImmutableSamplers =
      nullptr; // Regular RGB image, no immutable sampler needed

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[2].binding = 2;
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[3].binding = 3; // KernelParams (Read Only)
  bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[3].descriptorCount = 1;
  bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding motionPriorBinding{};
  motionPriorBinding.binding = 4; // Motion Prior
  motionPriorBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  motionPriorBinding.descriptorCount = 1;
  motionPriorBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding localMaskBinding{};
  localMaskBinding.binding = 5;
  localMaskBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  localMaskBinding.descriptorCount = 1;
  localMaskBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding referenceBinding{};
  referenceBinding.binding = 6;
  referenceBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  referenceBinding.descriptorCount = 1;
  referenceBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding bindingsFinal[7];
  bindingsFinal[0] = bindings[0];
  bindingsFinal[1] = bindings[1];
  bindingsFinal[2] = bindings[2];
  bindingsFinal[3] = bindings[3];
  bindingsFinal[4] = motionPriorBinding;
  bindingsFinal[5] = localMaskBinding;
  bindingsFinal[6] = referenceBinding;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 7;
  layoutInfo.pBindings = bindingsFinal;
  vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                              &descriptorSetLayout);

  // 2. Pipeline Layout with Push Constants
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(PushConstants);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
  vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

  // 3. Create Compute Pipeline
  // Direct use of generated uint32_t array
  std::vector<uint32_t> shaderCode(accumulate_comp_spv,
                                   accumulate_comp_spv +
                                       (accumulate_comp_spv_size / 4));

  VkShaderModule shaderModule = vm.createShaderModule(shaderCode);

  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shaderModule;
  pipelineInfo.stage.pName = "main";

  vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                           &accumulatePipeline);

  vkDestroyShaderModule(device, shaderModule, nullptr);

  // 4. Create Normalization Pipeline
  VkDescriptorSetLayoutBinding normalizeBindings[2] = {};
  normalizeBindings[0].binding = 0; // Output Buffer
  normalizeBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  normalizeBindings[0].descriptorCount = 1;
  normalizeBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  normalizeBindings[1].binding = 1; // Accumulator Buffer
  normalizeBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  normalizeBindings[1].descriptorCount = 1;
  normalizeBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo normalizeLayoutInfo{};
  normalizeLayoutInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  normalizeLayoutInfo.bindingCount = 2;
  normalizeLayoutInfo.pBindings = normalizeBindings;
  vkCreateDescriptorSetLayout(device, &normalizeLayoutInfo, nullptr,
                              &normalizeSetLayout);

  VkPipelineLayoutCreateInfo normalizePipelineLayoutInfo{};
  normalizePipelineLayoutInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  normalizePipelineLayoutInfo.setLayoutCount = 1;
  normalizePipelineLayoutInfo.pSetLayouts = &normalizeSetLayout;
  normalizePipelineLayoutInfo.pushConstantRangeCount = 1;
  normalizePipelineLayoutInfo.pPushConstantRanges =
      &pushConstantRange; // Reuse push constant range
  vkCreatePipelineLayout(device, &normalizePipelineLayoutInfo, nullptr,
                         &normalizePipelineLayout);

  std::vector<uint32_t> normShaderCode(
      normalize_comp_spv, normalize_comp_spv + (normalize_comp_spv_size / 4));
  VkShaderModule normShaderModule = vm.createShaderModule(normShaderCode);

  VkComputePipelineCreateInfo normPipelineInfo{};
  normPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  normPipelineInfo.layout = normalizePipelineLayout;
  normPipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  normPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  normPipelineInfo.stage.module = normShaderModule;
  normPipelineInfo.stage.pName = "main";

  vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &normPipelineInfo,
                           nullptr, &normalizePipeline);
  vkDestroyShaderModule(device, normShaderModule, nullptr);

  // 5. Create Structure Tensor Pipeline
  // Layout: Binding 0 (Sampler), Binding 1 (KernelParams Out)
  VkDescriptorSetLayoutBinding tensorBindings[2] = {};
  tensorBindings[0].binding = 0;
  tensorBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  tensorBindings[0].descriptorCount = 1;
  tensorBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  tensorBindings[0].pImmutableSamplers = nullptr; // Regular RGB image

  tensorBindings[1].binding = 1;
  tensorBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  tensorBindings[1].descriptorCount = 1;
  tensorBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo tensorLayoutInfo{};
  tensorLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  tensorLayoutInfo.bindingCount = 2;
  tensorLayoutInfo.pBindings = tensorBindings;
  vkCreateDescriptorSetLayout(device, &tensorLayoutInfo, nullptr,
                              &tensorSetLayout);

  VkPipelineLayoutCreateInfo tensorPLInfo{};
  tensorPLInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  tensorPLInfo.setLayoutCount = 1;
  tensorPLInfo.pSetLayouts = &tensorSetLayout;
  tensorPLInfo.pushConstantRangeCount = 1;
  tensorPLInfo.pPushConstantRanges = &pushConstantRange;
  vkCreatePipelineLayout(device, &tensorPLInfo, nullptr, &tensorPipelineLayout);

  std::vector<uint32_t> stCode(structure_tensor_comp_spv,
                               structure_tensor_comp_spv +
                                   (structure_tensor_comp_spv_size / 4));
  VkShaderModule stModule = vm.createShaderModule(stCode);

  VkComputePipelineCreateInfo stPipelineInfo{};
  stPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  stPipelineInfo.layout = tensorPipelineLayout;
  stPipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stPipelineInfo.stage.module = stModule;
  stPipelineInfo.stage.pName = "main";

  vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &stPipelineInfo, nullptr,
                           &tensorPipeline);
  vkDestroyShaderModule(device, stModule, nullptr);

  // 6. Create YUV to RGBA Pipeline
  VkDescriptorSetLayoutBinding yuvToRgbaBindings[2] = {};
  yuvToRgbaBindings[0].binding = 0; // Input YUV Sampler
  yuvToRgbaBindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  yuvToRgbaBindings[0].descriptorCount = 1;
  yuvToRgbaBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  yuvToRgbaBindings[0].pImmutableSamplers = samplers;

  yuvToRgbaBindings[1].binding = 1; // Output RGBA Image
  yuvToRgbaBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  yuvToRgbaBindings[1].descriptorCount = 1;
  yuvToRgbaBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo yuvToRgbaLayoutInfo{};
  yuvToRgbaLayoutInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  yuvToRgbaLayoutInfo.bindingCount = 2;
  yuvToRgbaLayoutInfo.pBindings = yuvToRgbaBindings;
  vkCreateDescriptorSetLayout(device, &yuvToRgbaLayoutInfo, nullptr,
                              &yuvToRgbaLayout);

  VkPipelineLayoutCreateInfo yuvToRgbaPLInfo{};
  yuvToRgbaPLInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  yuvToRgbaPLInfo.setLayoutCount = 1;
  yuvToRgbaPLInfo.pSetLayouts = &yuvToRgbaLayout;
  yuvToRgbaPLInfo.pushConstantRangeCount = 1;
  yuvToRgbaPLInfo.pPushConstantRanges = &pushConstantRange;
  vkCreatePipelineLayout(device, &yuvToRgbaPLInfo, nullptr,
                         &yuvToRgbaPipelineLayout);

  std::vector<uint32_t> yuvToRgbaCode(yuv_to_rgba_comp_spv,
                                      yuv_to_rgba_comp_spv +
                                          (yuv_to_rgba_comp_spv_size / 4));
  VkShaderModule yuvToRgbaModule = vm.createShaderModule(yuvToRgbaCode);

  VkComputePipelineCreateInfo yuvToRgbaPipelineInfo{};
  yuvToRgbaPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  yuvToRgbaPipelineInfo.layout = yuvToRgbaPipelineLayout;
  yuvToRgbaPipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  yuvToRgbaPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  yuvToRgbaPipelineInfo.stage.module = yuvToRgbaModule;
  yuvToRgbaPipelineInfo.stage.pName = "main";

  vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &yuvToRgbaPipelineInfo,
                           nullptr, &yuvToRgbaPipeline);
  vkDestroyShaderModule(device, yuvToRgbaModule, nullptr);
}

bool VulkanImageStacker::addFrame(AHardwareBuffer *buffer) {
  if (buffer == nullptr)
    return false;
  AHardwareBuffer_acquire(buffer);
  pendingFrames.push_back({buffer, 0.0f});
  return true;
}

bool VulkanImageStacker::processFrame(AHardwareBuffer *buffer, float frameScore,
                                      GrayImage &cachedGray, YuvStackPerfStats &perf) {
  using PerfClock = std::chrono::steady_clock;
  auto elapsedMs = [](const PerfClock::time_point &start,
                      const PerfClock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
  };
  VulkanImage input;
  perf.totalFrames++;

  // Use existing conversion if available (from first frame) to compatible with
  // immutable sampler
  TIME_START(importBuffer);
  if (!VulkanBufferImporter::importHardwareBuffer(buffer, input,
                                                  this->ycbcrConversion)) {
    LOGE("processFrame: Failed to import hardware buffer");
    return false;
  }
  TIME_END(importBuffer);

  bool currentIsFirstFrame = isFirstFrame;
  // LOGI("processFrame: Start. isFirstFrame=%d", currentIsFirstFrame);

  VulkanManager &vm = VulkanManager::getInstance();
  VkDevice device = vm.getDevice();

  if (immutableSampler == VK_NULL_HANDLE) {
    // Steal the first frame's sampler and conversion for long-lived pipelines
    this->immutableSampler = input.sampler;
    this->ycbcrConversion = input.ycbcrConversion;
    input.sampler = VK_NULL_HANDLE;
    input.ycbcrConversion = VK_NULL_HANDLE; // Steal ownership

    createPipelines(this->immutableSampler);
  } else {
    // If not first frame, we reused the conversion.
    // We must NOT let input.release() destroy it, because
    // 'this->ycbcrConversion' owns it now.
    input.ycbcrConversion = VK_NULL_HANDLE;
  }

  // 1. Reuse or Allocate Descriptor Sets for each tile
  int numTiles = numTilesX * numTilesY;

  if (yuvToRgbaSet == VK_NULL_HANDLE) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &yuvToRgbaLayout;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &yuvToRgbaSet));
  }

  // Descriptions for accum/tensor sets are constant for the life of the stacker
  for (int i = 0; i < numTiles; ++i) {
    if (accumSets[i] == VK_NULL_HANDLE) {
      VkDescriptorSetAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocInfo.descriptorPool = descriptorPool;
      allocInfo.descriptorSetCount = 1;
      allocInfo.pSetLayouts = &descriptorSetLayout;
      VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &accumSets[i]));

      VkDescriptorImageInfo imageInfo{};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = rgbFrame.viewY;
      imageInfo.sampler = rgbFrame.sampler;

      VkDescriptorImageInfo refImageInfo{};
      refImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      refImageInfo.imageView = referenceRgbFrame.viewY;
      refImageInfo.sampler = referenceRgbFrame.sampler;

      uint32_t scale = enableSuperRes ? 2 : 1;
      uint32_t fullH = height * scale;
      uint32_t fullTileH = (fullH + numTilesY - 1) / numTilesY;
      uint32_t tileRow = i / numTilesX;
      uint32_t fullTileY = tileRow * fullTileH;
      uint32_t fullActualTileH = std::min(fullTileH, fullH - fullTileY);
      uint32_t nativeTileY = fullTileY / scale;
      uint32_t nativeTileEndY =
          (fullTileY + fullActualTileH + scale - 1) / scale;
      VkDeviceSize kernelRowBytes = (VkDeviceSize)width * sizeof(float) * 4;
      VkDescriptorBufferInfo accumBufferInfo{accumBuffers[i], 0,
                                             VK_WHOLE_SIZE};
      VkDescriptorBufferInfo alignBufferInfo{alignmentBuffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo kpBufferInfo{
          kernelParamsBuffer, nativeTileY * kernelRowBytes,
          (nativeTileEndY - nativeTileY) * kernelRowBytes};
      VkDescriptorBufferInfo mpBufferInfo{motionPriorBuffer, 0, VK_WHOLE_SIZE};
      VkDescriptorBufferInfo lmBufferInfo{localTileMaskBuffer, 0, VK_WHOLE_SIZE};

      VkWriteDescriptorSet descriptorWrites[7] = {};
      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[0].dstSet = accumSets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].pImageInfo = &imageInfo;

      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = accumSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pBufferInfo = &accumBufferInfo;

      descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[2].dstSet = accumSets[i];
      descriptorWrites[2].dstBinding = 2;
      descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[2].descriptorCount = 1;
      descriptorWrites[2].pBufferInfo = &alignBufferInfo;

      descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[3].dstSet = accumSets[i];
      descriptorWrites[3].dstBinding = 3;
      descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[3].descriptorCount = 1;
      descriptorWrites[3].pBufferInfo = &kpBufferInfo;

      descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[4].dstSet = accumSets[i];
      descriptorWrites[4].dstBinding = 4;
      descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[4].descriptorCount = 1;
      descriptorWrites[4].pBufferInfo = &mpBufferInfo;

      descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[5].dstSet = accumSets[i];
      descriptorWrites[5].dstBinding = 5;
      descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[5].descriptorCount = 1;
      descriptorWrites[5].pBufferInfo = &lmBufferInfo;

      descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[6].dstSet = accumSets[i];
      descriptorWrites[6].dstBinding = 6;
      descriptorWrites[6].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[6].descriptorCount = 1;
      descriptorWrites[6].pImageInfo = &refImageInfo;

      vkUpdateDescriptorSets(device, 7, descriptorWrites, 0, nullptr);
    }

    if (tensorSets[i] == VK_NULL_HANDLE) {
      VkDescriptorSetAllocateInfo tensorAlloc{};
      tensorAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      tensorAlloc.descriptorPool = descriptorPool;
      tensorAlloc.descriptorSetCount = 1;
      tensorAlloc.pSetLayouts = &tensorSetLayout;
      VK_CHECK(vkAllocateDescriptorSets(device, &tensorAlloc, &tensorSets[i]));

      VkDescriptorImageInfo imageInfo{};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = rgbFrame.viewY;
      imageInfo.sampler = rgbFrame.sampler;

      uint32_t stTileH = (height + numTilesY - 1) / numTilesY;
      uint32_t tileRow = i / numTilesX;
      uint32_t stTileY = tileRow * stTileH;
      uint32_t stActualTileH = std::min(stTileH, height - stTileY);
      VkDeviceSize kernelRowBytes = (VkDeviceSize)width * sizeof(float) * 4;
      VkDescriptorBufferInfo kpBufferInfo{
          kernelParamsBuffer, stTileY * kernelRowBytes,
          stActualTileH * kernelRowBytes};

      VkWriteDescriptorSet tensorWrites[2] = {};
      tensorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      tensorWrites[0].dstSet = tensorSets[i];
      tensorWrites[0].dstBinding = 0;
      tensorWrites[0].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      tensorWrites[0].descriptorCount = 1;
      tensorWrites[0].pImageInfo = &imageInfo;

      tensorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      tensorWrites[1].dstSet = tensorSets[i];
      tensorWrites[1].dstBinding = 1;
      tensorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      tensorWrites[1].descriptorCount = 1;
      tensorWrites[1].pBufferInfo = &kpBufferInfo;

      vkUpdateDescriptorSets(device, 2, tensorWrites, 0, nullptr);
    }
  }

  // 2. Alignment logic on CPU (to get offsets)
  float offsetX = 0.0f, offsetY = 0.0f;
  std::vector<Point> alignmentOffsets;
  std::vector<float> alignmentErrorMap;
  std::vector<float> localTileMask;
  size_t copyCount = 0;
  float frameWeight = 1.0f;

  TIME_START(cpuAlignment);
  {
    // Use cached grayscale from score calculation (no second HardwareBuffer lock)
    auto currentPyramid =
        buildPyramid(cachedGray.data.data(), width, height, 4);
    if (currentIsFirstFrame) {
      referencePyramid = std::move(currentPyramid);
      localTileMask.assign((size_t)gridW * gridH, 1.0f);
    } else {
      int alignTileSize = enableSuperRes ? 16 : 32;
      TileAlignment alignment =
          computeTileAlignment(referencePyramid, currentPyramid, 64, alignTileSize);

      gridW = alignment.gridW;
      gridH = alignment.gridH;

      uint32_t allocatedGridW = (width + alignTileSize - 1) / alignTileSize;
      uint32_t allocatedGridH = (height + alignTileSize - 1) / alignTileSize;
      uint32_t maxAllocatedPoints = allocatedGridW * allocatedGridH;

      copyCount =
          std::min(alignment.offsets.size(), (size_t)maxAllocatedPoints);
      alignmentOffsets = std::move(alignment.offsets);
      alignmentErrorMap = std::move(alignment.errorMap);

      float sumX = 0, sumY = 0;
      for (const auto &p : alignmentOffsets) {
        sumX += p.x;
        sumY += p.y;
      }
      if (!alignmentOffsets.empty()) {
        offsetX = sumX / alignmentOffsets.size();
        offsetY = sumY / alignmentOffsets.size();
      }

      LocalReliabilityMap localMap = buildLocalReliabilityMap(
          referencePyramid.front(), alignmentErrorMap, gridW, gridH);
      localTileMask = std::move(localMap.tileWeights);
      float sharpnessRatio = (referenceFrameScore > 0.0f)
                                 ? std::min(frameScore / referenceFrameScore, 1.0f)
                                 : 1.0f;
      frameWeight = computeFrameFusionWeight(alignmentErrorMap, sharpnessRatio,
                                             localMap);
      float p90Err = computeScalarStats(alignmentErrorMap).p90;
      LOGI("YUV fusion: frame=%zu sharpnessRatio=%.3f frameWeight=%.3f flatGood=%.3f flatArea=%.3f detailArea=%.3f p90Err=%.3f p90ErrRms=%.3f",
           perf.totalFrames - 1, sharpnessRatio, frameWeight, localMap.flatGoodFraction,
           localMap.flatAreaFraction, localMap.detailAreaFraction,
           p90Err, std::sqrt(std::max(p90Err, 0.0f)));
    }
  }
  TIME_END(cpuAlignment);  // closes cpuAlignment + the cached-gray block

  // 3. Prepare Push Constants
  float transform[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  PushConstants pc{};
  pc.t0 = transform[0];
  pc.t1 = transform[1];
  pc.t2 = transform[2];
  pc.t3 = transform[3];
  pc.t4 = transform[4];
  pc.t5 = transform[5];
  pc.offsetX = offsetX;
  pc.offsetY = offsetY;
  uint32_t scale = enableSuperRes ? 2 : 1;
  pc.scale = static_cast<float>(scale);
  pc.width = width * scale;
  pc.height = height * scale;
  pc.baseNoise = 0.001f;
  pc.isFirstFrame = currentIsFirstFrame ? 1 : 0;
  pc.gridW = gridW;
  pc.gridH = gridH;
  pc.noiseAlpha = 0.005f;
  pc.noiseBeta = 0.001f;
  pc.frameWeight = frameWeight;

  if (!currentIsFirstFrame && pc.frameWeight < 0.12f) {
    input.release(device);
    isFirstFrame = false;
    perf.skippedFrames++;
    return false;
  }
  perf.keptFrames++;
  perf.totalEffectiveFrames += pc.frameWeight;

  double yuvToRgbaGpuMs = -1.0;
  double tensorGpuMs = -1.0;
  double accumulateGpuMs = -1.0;

  // --- START PIPELINING SYNC POINT ---
  // Wait for the PREVIOUS frame's GPU task to finish before we start recording
  // new commands or mapping shared buffers.
  vkQueueWaitIdle(vm.getComputeQueue());

  // Safe to map alignment buffers and update frame-varying descriptors now
  if (!currentIsFirstFrame && copyCount > 0) {
    void *mapPtr = nullptr;
    VkResult res =
        vkMapMemory(device, alignmentUploadMemory, 0, VK_WHOLE_SIZE, 0, &mapPtr);
    if (res == VK_SUCCESS && mapPtr != nullptr) {
      memcpy(mapPtr, alignmentOffsets.data(), copyCount * sizeof(Point));
      vkUnmapMemory(device, alignmentUploadMemory);
    }

    void *mpMapPtr = nullptr;
    VkResult resMP =
        vkMapMemory(device, motionPriorUploadMemory, 0, VK_WHOLE_SIZE, 0,
                    &mpMapPtr);
    if (resMP == VK_SUCCESS && mpMapPtr != nullptr) {
      memcpy(mpMapPtr, alignmentErrorMap.data(), copyCount * sizeof(float));
      vkUnmapMemory(device, motionPriorUploadMemory);
    }
  }

  if (localTileMask.empty()) {
    localTileMask.assign((size_t)gridW * gridH, 1.0f);
  }
  void *lmMapPtr = nullptr;
  VkResult resLM =
      vkMapMemory(device, localTileMaskUploadMemory, 0, VK_WHOLE_SIZE, 0,
                  &lmMapPtr);
  if (resLM == VK_SUCCESS && lmMapPtr != nullptr && !localTileMask.empty()) {
    memcpy(lmMapPtr, localTileMask.data(),
           localTileMask.size() * sizeof(float));
    vkUnmapMemory(device, localTileMaskUploadMemory);
  }

  // Update yuvToRgbaSet with current input
  {
    VkDescriptorImageInfo imageInfo{input.sampler, input.viewY,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo outInfo{VK_NULL_HANDLE, rgbFrame.viewY,
                                  VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = yuvToRgbaSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = yuvToRgbaSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &outInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  }

  TIME_START(gpuDispatch);
  // 3. Command Buffer Recording
  VkCommandBuffer cb = vm.beginSingleTimeCommands();
  if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
    vkCmdResetQueryPool(cb, gpuTimingQueryPool, 0, 8);
  }

  // Transition Input Image Layout
  VkImageMemoryBarrier imageBarrier{};
  imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  imageBarrier.image = input.image;
  imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageBarrier.subresourceRange.baseMipLevel = 0;
  imageBarrier.subresourceRange.levelCount = 1;
  imageBarrier.subresourceRange.baseArrayLayer = 0;
  imageBarrier.subresourceRange.layerCount = 1;
  imageBarrier.srcAccessMask = 0;
  imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &imageBarrier);

  // Transition RGB frame to GENERAL for writing
  VkImageMemoryBarrier rgbWriteBarrier{};
  rgbWriteBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  rgbWriteBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  rgbWriteBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  rgbWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  rgbWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  rgbWriteBarrier.image = rgbFrame.image;
  rgbWriteBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  rgbWriteBarrier.subresourceRange.baseMipLevel = 0;
  rgbWriteBarrier.subresourceRange.levelCount = 1;
  rgbWriteBarrier.subresourceRange.baseArrayLayer = 0;
  rgbWriteBarrier.subresourceRange.layerCount = 1;
  rgbWriteBarrier.srcAccessMask = 0;
  rgbWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &rgbWriteBarrier);

  // Dispatch YUV to RGBA Conversion
  if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        gpuTimingQueryPool, 0);
  }
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, yuvToRgbaPipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          yuvToRgbaPipelineLayout, 0, 1, &yuvToRgbaSet, 0,
                          nullptr);

  struct {
    uint32_t w;
    uint32_t h;
  } convPc = {width, height};
  vkCmdPushConstants(cb, yuvToRgbaPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(convPc), &convPc);
  vkCmdDispatch(cb, (width + 15) / 16, (height + 15) / 16, 1);
  if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        gpuTimingQueryPool, 1);
  }

  // Transition RGB frame to SHADER_READ_ONLY_OPTIMAL for sampling
  VkImageMemoryBarrier rgbReadBarrier{};
  rgbReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  rgbReadBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  rgbReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  rgbReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  rgbReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  rgbReadBarrier.image = rgbFrame.image;
  rgbReadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  rgbReadBarrier.subresourceRange.baseMipLevel = 0;
  rgbReadBarrier.subresourceRange.levelCount = 1;
  rgbReadBarrier.subresourceRange.baseArrayLayer = 0;
  rgbReadBarrier.subresourceRange.layerCount = 1;
  rgbReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  rgbReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &rgbReadBarrier);

  // Copy RGB frame to referenceRgbFrame if first frame
  if (pc.isFirstFrame) {
    VkImageMemoryBarrier copyBarriers[2] = {};
    copyBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    copyBarriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    copyBarriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    copyBarriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    copyBarriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    copyBarriers[0].image = rgbFrame.image;
    copyBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyBarriers[0].subresourceRange.levelCount = 1;
    copyBarriers[0].subresourceRange.layerCount = 1;

    copyBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    copyBarriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    copyBarriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copyBarriers[1].srcAccessMask = 0;
    copyBarriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copyBarriers[1].image = referenceRgbFrame.image;
    copyBarriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyBarriers[1].subresourceRange.levelCount = 1;
    copyBarriers[1].subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 2, copyBarriers);

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.extent = {width, height, 1};

    vkCmdCopyImage(cb, rgbFrame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   referenceRgbFrame.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    VkImageMemoryBarrier postCopyBarriers[2] = {};
    postCopyBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    postCopyBarriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    postCopyBarriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    postCopyBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    postCopyBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    postCopyBarriers[0].image = rgbFrame.image;
    postCopyBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    postCopyBarriers[0].subresourceRange.levelCount = 1;
    postCopyBarriers[0].subresourceRange.layerCount = 1;

    postCopyBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    postCopyBarriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    postCopyBarriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    postCopyBarriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    postCopyBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    postCopyBarriers[1].image = referenceRgbFrame.image;
    postCopyBarriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    postCopyBarriers[1].subresourceRange.levelCount = 1;
    postCopyBarriers[1].subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 2, postCopyBarriers);
  }

  // Clear Accumulators if first frame
  if (pc.isFirstFrame) {
    for (int i = 0; i < numTiles; ++i) {
      vkCmdFillBuffer(cb, accumBuffers[i], 0, VK_WHOLE_SIZE, 0);
    }
    // Barrier to ensure clear is done before usage?
    // SingleTimeCommands end with a queue submission which has implicit
    // ordering if we record linearly? No, we need a barrier if we use it in the
    // SAME command buffer as Dispatch. Yes, we are in the same CB.

    VkMemoryBarrier memBarrier = {};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBarrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);
  }

  bool needAlignmentUpload = (!currentIsFirstFrame && copyCount > 0);
  VkBufferMemoryBarrier uploadSrcBarriers[3] = {};
  uint32_t uploadSrcBarrierCount = 0;
  auto appendUploadSrcBarrier = [&](VkBuffer buffer) {
    VkBufferMemoryBarrier &barrier = uploadSrcBarriers[uploadSrcBarrierCount++];
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
  };
  if (needAlignmentUpload) {
    appendUploadSrcBarrier(alignmentUploadBuffer);
    appendUploadSrcBarrier(motionPriorUploadBuffer);
  }
  appendUploadSrcBarrier(localTileMaskUploadBuffer);
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                       uploadSrcBarrierCount, uploadSrcBarriers, 0, nullptr);

  if (needAlignmentUpload) {
    VkBufferCopy alignCopy{};
    alignCopy.size = copyCount * sizeof(Point);
    vkCmdCopyBuffer(cb, alignmentUploadBuffer, alignmentBuffer, 1, &alignCopy);

    VkBufferCopy motionCopy{};
    motionCopy.size = copyCount * sizeof(float);
    vkCmdCopyBuffer(cb, motionPriorUploadBuffer, motionPriorBuffer, 1,
                    &motionCopy);
  }
  if (!localTileMask.empty()) {
    VkBufferCopy localMaskCopy{};
    localMaskCopy.size = localTileMask.size() * sizeof(float);
    vkCmdCopyBuffer(cb, localTileMaskUploadBuffer, localTileMaskBuffer, 1,
                    &localMaskCopy);
  }

  VkBufferMemoryBarrier uploadDstBarriers[3] = {};
  uint32_t uploadDstBarrierCount = 0;
  auto appendUploadDstBarrier = [&](VkBuffer buffer) {
    VkBufferMemoryBarrier &barrier = uploadDstBarriers[uploadDstBarrierCount++];
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
  };
  if (needAlignmentUpload) {
    appendUploadDstBarrier(alignmentBuffer);
    appendUploadDstBarrier(motionPriorBuffer);
  }
  if (!localTileMask.empty()) {
    appendUploadDstBarrier(localTileMaskBuffer);
  }
  if (uploadDstBarrierCount > 0) {
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         uploadDstBarrierCount, uploadDstBarriers, 0, nullptr);
  }

  // Phase 2: Compute Structure Tensor (only on first/reference frame)
  // The kernel shape is driven by the reference frame's local structure.
  // Subsequent frames reuse it — avoids (N-1) expensive GPU dispatches.
  if (currentIsFirstFrame) {
    if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          gpuTimingQueryPool, 2);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tensorPipeline);

    for (int y = 0; y < numTilesY; ++y) {
      for (int x = 0; x < numTilesX; ++x) {
        int i = y * numTilesX + x;

        uint32_t stFullW = width;
        uint32_t stFullH = height;
        uint32_t stTileW = (stFullW + numTilesX - 1) / numTilesX;
        uint32_t stTileH = (stFullH + numTilesY - 1) / numTilesY;

        PushConstants stPC = pc;
        stPC.tileX = x * stTileW;
        stPC.tileY = y * stTileH;
        stPC.tileW = std::min(stTileW, stFullW - stPC.tileX);
        stPC.tileH = std::min(stTileH, stFullH - stPC.tileY);
        stPC.width = stFullW;
        stPC.height = stFullH;
        stPC.offsetY = static_cast<float>(stPC.tileY);

        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                tensorPipelineLayout, 0, 1, &tensorSets[i], 0,
                                nullptr);
        vkCmdPushConstants(cb, tensorPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(stPC), &stPC);
        vkCmdDispatch(cb, (stPC.tileW + 15) / 16, (stPC.tileH + 15) / 16, 1);
      }
    }

    // Barrier: ST writes must finish before Accumulate reads kernelParams
    VkBufferMemoryBarrier stBarrier{};
    stBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    stBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    stBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    stBarrier.buffer = kernelParamsBuffer;
    stBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &stBarrier, 0, nullptr);
    if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          gpuTimingQueryPool, 3);
    }
  }

  // The previous frame is already fully retired by vkQueueWaitIdle() above.
  // Avoid issuing redundant per-tile carry-over barriers here.

  // Pass 2: Accumulate
  if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        gpuTimingQueryPool, 4);
  }
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, accumulatePipeline);

  uint32_t fullW = pc.width;
  uint32_t fullH = pc.height;
  uint32_t tileW = (fullW + numTilesX - 1) / numTilesX;
  uint32_t tileH = (fullH + numTilesY - 1) / numTilesY;

  for (int y = 0; y < numTilesY; ++y) {
    for (int x = 0; x < numTilesX; ++x) {
      int i = y * numTilesX + x;
      pc.tileX = x * tileW;
      pc.tileY = y * tileH;
      pc.tileW = std::min(tileW, fullW - pc.tileX);
      pc.tileH = std::min(tileH, fullH - pc.tileY);
      pc.bufferStride = tileW + 16; // The allocated stride
      pc.offsetY = static_cast<float>(pc.tileY / (enableSuperRes ? 2 : 1));

      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout, 0, 1, &accumSets[i], 0, nullptr);
      vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(pc), &pc);
      vkCmdDispatch(cb, (pc.tileW + 15) / 16, (pc.tileH + 15) / 16, 1);
    }
  }

  // Global barrier between frames to ensure ALL accumulator tiles are finished
  // writing before the next frame begins reading or before normalization
  VkMemoryBarrier postBarrier{};
  postBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  postBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  postBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &postBarrier,
                       0, nullptr, 0, nullptr);
  if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        gpuTimingQueryPool, 5);
  }

  vm.endSingleTimeCommands(cb);
  if (gpuTimestampSupported && gpuTimingQueryPool != VK_NULL_HANDLE) {
    auto ticksToMs = [&](uint64_t begin, uint64_t end) {
      return (double)(end - begin) * (double)gpuTimestampPeriodNs / 1.0e6;
    };

    uint64_t yuvTimestamps[2] = {};
    VkResult yuvResult = vkGetQueryPoolResults(
        device, gpuTimingQueryPool, 0, 2, sizeof(yuvTimestamps), yuvTimestamps,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (yuvResult == VK_SUCCESS && gpuTimestampPeriodNs > 0.0f) {
      yuvToRgbaGpuMs = ticksToMs(yuvTimestamps[0], yuvTimestamps[1]);
    }

    if (currentIsFirstFrame) {
      uint64_t tensorTimestamps[2] = {};
      VkResult tensorResult = vkGetQueryPoolResults(
          device, gpuTimingQueryPool, 2, 2, sizeof(tensorTimestamps),
          tensorTimestamps, sizeof(uint64_t),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      if (tensorResult == VK_SUCCESS && gpuTimestampPeriodNs > 0.0f) {
        tensorGpuMs = ticksToMs(tensorTimestamps[0], tensorTimestamps[1]);
      }
    }

    uint64_t accumulateTimestamps[2] = {};
    VkResult accumulateResult = vkGetQueryPoolResults(
        device, gpuTimingQueryPool, 4, 2, sizeof(accumulateTimestamps),
        accumulateTimestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (accumulateResult == VK_SUCCESS && gpuTimestampPeriodNs > 0.0f) {
      accumulateGpuMs =
          ticksToMs(accumulateTimestamps[0], accumulateTimestamps[1]);
    }
  }
  TIME_END(gpuDispatch);

  isFirstFrame = false;
  input.release(device);
  return true;
}

bool VulkanImageStacker::processStack(uint32_t *outBitmap, uint32_t outWidth,
                                      uint32_t outHeight, uint32_t stride,
                                      int rotation) {
  // Lower CPU priority for this thread during heavy processing
  setpriority(PRIO_PROCESS, 0, 10);

  YuvStackPerfStats perf;
  const auto startTotal = std::chrono::steady_clock::now();

  // Process all queued frames first
  isFirstFrame = true;
  if (pendingFrames.empty()) {
    return false;
  }

  // 1. Calculate scores and extract grayscale in a single lock per frame
  TIME_START(scoreCalculation);
  for (auto &frame : pendingFrames) {
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(frame.buffer, &desc);
    void *lockedData = nullptr;
    if (AHardwareBuffer_lock(frame.buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                             -1, nullptr, &lockedData) == 0) {
      uint8_t *srcY = (uint8_t *)lockedData;
      int stride = desc.stride;
      bool is10bit = (desc.format == 0x36);

      // Extract grayscale
      frame.grayY.width = width;
      frame.grayY.height = height;
      frame.grayY.data.resize((size_t)width * height);
      for (uint32_t y = 0; y < height; ++y) {
        if (is10bit) {
          uint16_t *rowPtr = (uint16_t *)(srcY + y * stride * 2);
          for (uint32_t x = 0; x < width; ++x) {
            frame.grayY.data[y * width + x] = (uint8_t)(rowPtr[x] >> 8);
          }
        } else {
          memcpy(frame.grayY.data.data() + y * width, srcY + y * stride, width);
        }
      }

      // Calculate score from the same locked data
      long long score = 0;
      int step = 8;
      for (uint32_t y = step; y < height - step; y += step) {
        for (uint32_t x = step; x < width - step; x += step) {
          int val = frame.grayY.data[y * width + x];
          int valR = frame.grayY.data[y * width + x + 1];
          int valB = frame.grayY.data[(y + 1) * width + x];
          score += std::abs(val - valR);
          score += std::abs(val - valB);
        }
      }
      frame.score = (float)score;

      AHardwareBuffer_unlock(frame.buffer, nullptr);
    }
  }
  TIME_END(scoreCalculation);
  perf.scoreCalculationMs = elapsed_scoreCalculation;

  // 2. Sort pendingFrames by score descending
  std::sort(
      pendingFrames.begin(), pendingFrames.end(),
      [](const FrameData &a, const FrameData &b) { return a.score > b.score; });
  referenceFrameScore = pendingFrames.empty() ? 0.0f : pendingFrames.front().score;

  //  LOGI("processStack: Processing %zu frames", pendingFrames.size());

  VulkanManager &vm = VulkanManager::getInstance();
  VkDevice device = vm.getDevice();
  if (device == VK_NULL_HANDLE) {
    LOGE("processStack: Vulkan device is NULL or NOT INITIALIZED");
    return false;
  }

  TIME_START(allFramesProcessing);
  int frameIdx = 0;
  for (auto &frame : pendingFrames) {
    //    LOGI("processStack: Processing frame %d, score %f", frameIdx,
    //    frame.score);
    if (!processFrame(frame.buffer, frame.score, frame.grayY, perf)) {
      // LOGW("processStack: Skipped or failed frame %d", frameIdx);
    }
    // Release grayscale cache to free memory as soon as it's consumed
    frame.grayY.data.clear();
    frame.grayY.data.shrink_to_fit();
    AHardwareBuffer_release(frame.buffer);
    frameIdx++;
  }
  pendingFrames.clear();
  vkQueueWaitIdle(vm.getComputeQueue());
  vkResetDescriptorPool(device, descriptorPool, 0);
  resetDescriptorHandles();

  TIME_END(allFramesProcessing);
  perf.allFramesProcessingMs = elapsed_allFramesProcessing;

  if (!outBitmap || isFirstFrame) // Check again after processing
    return false;

  uint32_t scale = enableSuperRes ? 2 : 1;
  uint32_t sensorW = width * scale;
  uint32_t sensorH = height * scale;

  // Calculate the crop and rotation matrix
  float transform[6] = {1, 0, 0, 0, 1, 0};

  // Logic from RotatePlane + Cropping in legacy code:
  // 1. Determine rotated dimensions of sensor
  int rotW = (rotation == 90 || rotation == 270) ? sensorH : sensorW;
  int rotH = (rotation == 90 || rotation == 270) ? sensorW : sensorH;

  float cropX = (rotW - (float)outWidth) / 2.0f;
  float cropY = (rotH - (float)outHeight) / 2.0f;

  float sW = (float)sensorW;
  float sH = (float)sensorH;

  switch (rotation) {
  case 0:
    transform[0] = 1.0f;
    transform[1] = 0.0f;
    transform[2] = cropX;
    transform[3] = 0.0f;
    transform[4] = 1.0f;
    transform[5] = cropY;
    break;
  case 90:
    // Inverse of 90 deg clockwise: (x,y) -> (y, H-1-x)
    transform[0] = 0.0f;
    transform[1] = 1.0f;
    transform[2] = cropY;
    transform[3] = -1.0f;
    transform[4] = 0.0f;
    transform[5] = sH - 1.0f - cropX;
    break;
  case 180:
    // Inverse of 180 deg: (x,y) -> (W-1-x, H-1-y)
    transform[0] = -1.0f;
    transform[1] = 0.0f;
    transform[2] = sW - 1.0f - cropX;
    transform[3] = 0.0f;
    transform[4] = -1.0f;
    transform[5] = sH - 1.0f - cropY;
    break;
  case 270:
    // Inverse of 270 deg clockwise: (x,y) -> (W-1-y, x)
    transform[0] = 0.0f;
    transform[1] = -1.0f;
    transform[2] = sW - 1.0f - cropY;
    transform[3] = 1.0f;
    transform[4] = 0.0f;
    transform[5] = cropX;
    break;
  }

  VkDeviceSize outSize = outWidth * outHeight * 4;
  VkPhysicalDeviceProperties deviceProperties{};
  vkGetPhysicalDeviceProperties(vm.getPhysicalDevice(), &deviceProperties);
  VkDeviceSize maxStorageBufferRange =
      deviceProperties.limits.maxStorageBufferRange;
  if (maxStorageBufferRange > 0 && outSize > maxStorageBufferRange) {
    LOGE("processStack: output storage buffer range is too large. out=%llu "
         "limit=%llu output=%ux%u",
         (unsigned long long)outSize,
         (unsigned long long)maxStorageBufferRange, outWidth, outHeight);
    return false;
  }

  // 1. Dispatch Normalization for each tile
  VkCommandBuffer cb = vm.beginSingleTimeCommands();
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, normalizePipeline);

  uint32_t fullW = sensorW;
  uint32_t fullH = sensorH;
  uint32_t tileW = (fullW + numTilesX - 1) / numTilesX;
  uint32_t tileH = (fullH + numTilesY - 1) / numTilesY;

  int numTiles = numTilesX * numTilesY;

  for (int i = 0; i < numTiles; ++i) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &normalizeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &normalizeSets[i]));

    VkDescriptorBufferInfo outBufferInfo{stagingBuffer, 0, outSize};
    VkDescriptorBufferInfo accumBufferInfo{accumBuffers[i], 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = normalizeSets[i];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &outBufferInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = normalizeSets[i];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &accumBufferInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    // Let's re-map manually to be safe
    struct {
      float t0, t1, t2, t3, t4, t5;
      uint32_t outWidth;
      uint32_t outHeight;
      uint32_t outOriginX;
      uint32_t outOriginY;
      uint32_t sensorWidth;
      uint32_t sensorHeight;
      uint32_t tileX;
      uint32_t tileY;
      uint32_t tileW;
      uint32_t tileH;
      uint32_t bufferStride;
    } npc;
    memcpy(&npc, transform, 6 * sizeof(float));
    npc.outWidth = outWidth;
    npc.outHeight = outHeight;
    int tx = i % numTilesX;
    int ty = i / numTilesX;
    npc.tileX = tx * tileW;
    npc.tileY = ty * tileH;
    npc.tileW = std::min(tileW, fullW - npc.tileX);
    npc.tileH = std::min(tileH, fullH - npc.tileY);
    npc.bufferStride = tileW + 16;
    npc.sensorWidth = fullW;
    npc.sensorHeight = fullH;

    const int tileStartX = static_cast<int>(npc.tileX);
    const int tileStartY = static_cast<int>(npc.tileY);
    const int tileEndX = tileStartX + static_cast<int>(npc.tileW);
    const int tileEndY = tileStartY + static_cast<int>(npc.tileH);
    int dispatchMinX = 0;
    int dispatchMinY = 0;
    int dispatchMaxX = static_cast<int>(outWidth);
    int dispatchMaxY = static_cast<int>(outHeight);

    switch (rotation) {
    case 0:
      dispatchMinX = std::max(0, static_cast<int>(std::floor(tileStartX - cropX)) - 1);
      dispatchMaxX = std::min(static_cast<int>(outWidth),
                              static_cast<int>(std::ceil(tileEndX - cropX)) + 1);
      dispatchMinY = std::max(0, static_cast<int>(std::floor(tileStartY - cropY)) - 1);
      dispatchMaxY = std::min(static_cast<int>(outHeight),
                              static_cast<int>(std::ceil(tileEndY - cropY)) + 1);
      break;
    case 90:
      dispatchMinX =
          std::max(0, static_cast<int>(std::floor(sH - cropX - tileEndY)) - 1);
      dispatchMaxX =
          std::min(static_cast<int>(outWidth),
                   static_cast<int>(std::ceil(sH - cropX - tileStartY)) + 1);
      dispatchMinY = std::max(0, static_cast<int>(std::floor(tileStartX - cropY)) - 1);
      dispatchMaxY = std::min(static_cast<int>(outHeight),
                              static_cast<int>(std::ceil(tileEndX - cropY)) + 1);
      break;
    case 180:
      dispatchMinX =
          std::max(0, static_cast<int>(std::floor(sW - cropX - tileEndX)) - 1);
      dispatchMaxX =
          std::min(static_cast<int>(outWidth),
                   static_cast<int>(std::ceil(sW - cropX - tileStartX)) + 1);
      dispatchMinY =
          std::max(0, static_cast<int>(std::floor(sH - cropY - tileEndY)) - 1);
      dispatchMaxY =
          std::min(static_cast<int>(outHeight),
                   static_cast<int>(std::ceil(sH - cropY - tileStartY)) + 1);
      break;
    case 270:
      dispatchMinX = std::max(0, static_cast<int>(std::floor(tileStartY - cropX)) - 1);
      dispatchMaxX = std::min(static_cast<int>(outWidth),
                              static_cast<int>(std::ceil(tileEndY - cropX)) + 1);
      dispatchMinY =
          std::max(0, static_cast<int>(std::floor(sW - cropY - tileEndX)) - 1);
      dispatchMaxY =
          std::min(static_cast<int>(outHeight),
                   static_cast<int>(std::ceil(sW - cropY - tileStartX)) + 1);
      break;
    default:
      break;
    }

    if (dispatchMinX >= dispatchMaxX || dispatchMinY >= dispatchMaxY)
      continue;

    const uint32_t dispatchW =
        static_cast<uint32_t>(dispatchMaxX - dispatchMinX);
    const uint32_t dispatchH =
        static_cast<uint32_t>(dispatchMaxY - dispatchMinY);
    npc.outOriginX = static_cast<uint32_t>(dispatchMinX);
    npc.outOriginY = static_cast<uint32_t>(dispatchMinY);

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            normalizePipelineLayout, 0, 1, &normalizeSets[i], 0,
                            nullptr);

    vkCmdPushConstants(cb, normalizePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(npc), &npc);

    vkCmdDispatch(cb, (dispatchW + 15) / 16, (dispatchH + 15) / 16, 1);
  }

  // Transition staging buffer
  VkBufferMemoryBarrier hostBarrier{};
  hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  hostBarrier.buffer = stagingBuffer;
  hostBarrier.size = outSize;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                       &hostBarrier, 0, nullptr);

  TIME_START(normalizationDispatch);
  vm.endSingleTimeCommands(cb);
  vkQueueWaitIdle(vm.getComputeQueue());
  TIME_END(normalizationDispatch);
  perf.normalizationDispatchMs = elapsed_normalizationDispatch;

  // 2. Map and copy to bitmap
  TIME_START(outputCopy);
  void *hostData;
  vkMapMemory(device, stagingMemory, 0, outSize, 0, &hostData);

  if (stride == outWidth * 4) {
    memcpy(outBitmap, hostData, (size_t)outSize);
  } else {
    for (uint32_t y = 0; y < outHeight; ++y) {
      memcpy((uint8_t *)outBitmap + y * stride,
             (uint8_t *)hostData + y * outWidth * 4, outWidth * 4);
    }
  }
  vkUnmapMemory(device, stagingMemory);
  TIME_END(outputCopy);
  perf.outputCopyMs = elapsed_outputCopy;

  vkFreeDescriptorSets(device, descriptorPool, numTiles, normalizeSets.data());
  std::fill(normalizeSets.begin(), normalizeSets.end(),
            (VkDescriptorSet)VK_NULL_HANDLE);

  perf.totalMs = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - startTotal)
                     .count();
  perf.logSummary(outWidth, outHeight, scale);

  return true;
}

bool VulkanImageStacker::resetForReuse() {
  VulkanManager &vm = VulkanManager::getInstance();
  VkDevice device = vm.getDevice();
  if (device == VK_NULL_HANDLE) {
    LOGE("resetForReuse: Vulkan device is NULL");
    return false;
  }

  vkQueueWaitIdle(vm.getComputeQueue());
  releasePendingFrames();
  referencePyramid.clear();
  referenceFrameScore = 0.0f;
  isFirstFrame = true;

  if (descriptorPool != VK_NULL_HANDLE) {
    vkResetDescriptorPool(device, descriptorPool, 0);
  }
  resetDescriptorHandles();
  return true;
}

void VulkanImageStacker::resetDescriptorHandles() {
  yuvToRgbaSet = VK_NULL_HANDLE;
  std::fill(accumSets.begin(), accumSets.end(), (VkDescriptorSet)VK_NULL_HANDLE);
  std::fill(tensorSets.begin(), tensorSets.end(), (VkDescriptorSet)VK_NULL_HANDLE);
  std::fill(normalizeSets.begin(), normalizeSets.end(),
            (VkDescriptorSet)VK_NULL_HANDLE);
}

void VulkanImageStacker::releasePendingFrames() {
  for (auto &frame : pendingFrames) {
    frame.grayY.data.clear();
    if (frame.buffer != nullptr) {
      AHardwareBuffer_release(frame.buffer);
    }
  }
  pendingFrames.clear();
}

void VulkanImageStacker::releaseVulkanResources() {
  VkDevice device = VulkanManager::getInstance().getDevice();
  if (device == VK_NULL_HANDLE)
    return;
  if (descriptorPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  if (descriptorSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
  if (pipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  if (accumulatePipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(device, accumulatePipeline, nullptr);

  if (normalizeSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, normalizeSetLayout, nullptr);
  if (normalizePipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(device, normalizePipelineLayout, nullptr);
  if (normalizePipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(device, normalizePipeline, nullptr);
  if (gpuTimingQueryPool != VK_NULL_HANDLE)
    vkDestroyQueryPool(device, gpuTimingQueryPool, nullptr);

  if (immutableSampler != VK_NULL_HANDLE)
    vkDestroySampler(device, immutableSampler, nullptr);

  if (ycbcrConversion != VK_NULL_HANDLE) {
    auto pfnDestroy = (PFN_vkDestroySamplerYcbcrConversion)vkGetDeviceProcAddr(
        device, "vkDestroySamplerYcbcrConversion");
    if (pfnDestroy)
      pfnDestroy(device, ycbcrConversion, nullptr);
  }

  for (size_t i = 0; i < accumBuffers.size(); ++i) {
    if (accumBuffers[i] != VK_NULL_HANDLE)
      vkDestroyBuffer(device, accumBuffers[i], nullptr);
    if (accumMemories[i] != VK_NULL_HANDLE)
      vkFreeMemory(device, accumMemories[i], nullptr);
  }

  if (stagingBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, stagingBuffer, nullptr);
  if (stagingMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, stagingMemory, nullptr);

  if (alignmentUploadBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, alignmentUploadBuffer, nullptr);
  if (alignmentUploadMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, alignmentUploadMemory, nullptr);
  if (alignmentBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, alignmentBuffer, nullptr);
  if (alignmentMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, alignmentMemory, nullptr);

  if (kernelParamsBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, kernelParamsBuffer, nullptr);
  if (kernelParamsMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, kernelParamsMemory, nullptr);

  if (motionPriorBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, motionPriorBuffer, nullptr);
  if (motionPriorMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, motionPriorMemory, nullptr);
  if (motionPriorUploadBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, motionPriorUploadBuffer, nullptr);
  if (motionPriorUploadMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, motionPriorUploadMemory, nullptr);
  if (localTileMaskBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, localTileMaskBuffer, nullptr);
  if (localTileMaskMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, localTileMaskMemory, nullptr);
  if (localTileMaskUploadBuffer != VK_NULL_HANDLE)
    vkDestroyBuffer(device, localTileMaskUploadBuffer, nullptr);
  if (localTileMaskUploadMemory != VK_NULL_HANDLE)
    vkFreeMemory(device, localTileMaskUploadMemory, nullptr);

  if (tensorSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, tensorSetLayout, nullptr);
  if (tensorPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(device, tensorPipelineLayout, nullptr);
  if (tensorPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(device, tensorPipeline, nullptr);

  if (yuvToRgbaLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(device, yuvToRgbaLayout, nullptr);
  if (yuvToRgbaPipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(device, yuvToRgbaPipelineLayout, nullptr);
  if (yuvToRgbaPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(device, yuvToRgbaPipeline, nullptr);

  rgbFrame.release(device);
  referenceRgbFrame.release(device);
}
