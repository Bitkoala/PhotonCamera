# GLES 驱动兼容性记录

本文记录项目在真实 Android 设备上确认过的 GLES 驱动限制。修改 compute shader、image
load/store、纹理格式或 accumulator 架构前必须复查本文，并在目标设备上运行
`GlesRawStackerShaderTest`。不能仅凭 GLES 版本号或桌面 GLSL 编译结果判断移动端可用性。

## PMA110

### `RGBA16F` image 的访问限制

PMA110 的 GLES 驱动可以编译和运行以下组合：

- `layout(rgba16f) writeonly image2D`
- `glBindImageTexture(..., GL_WRITE_ONLY, GL_RGBA16F)`

但不能编译以下组合：

- `layout(rgba16f) image2D`，即 read/write image
- 对该 image 同时使用 `imageLoad` 与 `imageStore`
- `glBindImageTexture(..., GL_READ_WRITE, GL_RGBA16F)`

驱动编译错误为：

```text
Shader raw_accumulate compute compilation failed:
unsupported format on read/write image
```

因此不得使用单张 `RGBA16F` texture 原位完成 accumulator 的读取和写入。即使同一个
compute invocation 只访问唯一像素、没有跨 invocation 竞争，该格式仍会在 shader 编译阶段
被驱动拒绝。

### 当前兼容策略

需要原位 read/write 的 half-float accumulator 使用规范支持的整数 image：

- `R32UI`
- `layout(r32ui) uimage2D`
- `GL_READ_WRITE + GL_R32UI`
- 使用 `packHalf2x16` / `unpackHalf2x16` 在一个 texel 中保存两个 half

RAW base accumulator 使用两张 `R32UI`：

1. `weightedValue + weight`
2. `squareSum + clipMass`

总显存仍为 8 B/像素，与单张 `RGBA16F` 相同，但不需要两张 `RGBA16F` ping-pong。
MFSR accumulator 只保存 `weightedValue + weight`，使用一张 `R32UI`，为 4 B/输出像素。

所有读取 accumulator 的 fragment/compute shader 必须使用 `usampler2D`，取出 `uint` 后再
调用 `unpackHalf2x16`。禁止把 `R32UI` texture 绑定给普通 `sampler2D`。

### 验收要求

相关修改至少验证以下路径：

- shader compile/link
- `GL_R32UI` clear dispatch
- `GL_READ_WRITE` accumulator dispatch
- `R32UI` imageLoad/imageStore
- `usampler2D` normalize
- MFNR 与 MFSR 两种模式

桌面静态检查和 Kotlin 编译不能替代真机验证。新增格式前应优先扩展
`app/src/androidTest/java/com/hinnka/mycamera/processor/GlesRawStackerShaderTest.kt`，让测试实际
dispatch 对应 pass，而不只编译 program。

## 通用规则

- 不根据 `GL_MAX_TEXTURE_SIZE` 推断 image load/store 格式支持。
- 不根据 write-only image 可用推断同格式 read/write image 也可用。
- 不以增加补偿 pass 掩盖驱动格式错误；应选择驱动明确支持的存储格式和访问模型。
- 每个新增 image format 都需要在代表设备上验证 compile、bind、dispatch 和后续采样。
- 运行时错误日志必须保留 shader 名称、访问模式、internal format 和驱动错误文本。
