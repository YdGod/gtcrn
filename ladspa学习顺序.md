# LADSPA 插件代码学习顺序

> GTCRN LADSPA 插件共约 1737 行代码（Rust + Shell + Docker），分为 **源码**、**构建**、**配置** 三个层次。按以下顺序阅读，从高层到底层，从简单到复杂。

---

## 学习路线图

```
第1步  Cargo.toml          (75行)  ← 项目全景：依赖和构建策略
第2步  src/lib.rs          (128行) ← 插件入口：端口定义和描述符
第3步  src/stft.rs         (213行) ← 独立模块：STFT/iSTFT 实现
第4步  src/model.rs        (301行) ← 模型封装：ONNX Runtime 推理
第5步  src/plugin.rs       (559行) ← 核心逻辑：双线程 + 数据流
第6步  build.rs            (161行) ← 构建脚本：静态链接 + 模型转换
第7步  build.sh            (162行) ← 构建入口：3 种策略
第8步  build-minimal-docker.sh (65行) ← Docker 构建精简 ORT
第9步  Dockerfile.minimal-ort (138行) ← ORT 裁剪编译镜像
第10步 required_ops.config (2335行) ← 算子清单（参考）
第11步 README.md           (90行)  ← 使用文档
```

---

## 第 1 步：[Cargo.toml](ladspa/Cargo.toml)（75 行）

**为什么先读它**：这是整个 Rust 项目的"目录"，读完就知道项目用什么库、怎么编译、有哪些构建模式。

**重点理解**：

| 内容 | 说明 |
|------|------|
| `[package]` | 项目名 `gtcrn-ladspa-ort`，编译为 C 动态库 (`cdylib`) |
| `[features]` | 4 种构建策略：`dynamic` / `download` / `minimal` / `static` |
| `[dependencies]` | 5 个关键 crate：`ladspa`（插件接口）、`ort`（ONNX Runtime）、`rubato`（重采样）、`ringbuf`（环形缓冲）、`realfft`（FFT） |

**带着问题读**：这个项目依赖了哪些外部库？各个库是干什么的？

---

## 第 2 步：[src/lib.rs](ladspa/src/lib.rs)（128 行）

**为什么第二个读它**：这是插件的"入口文件"，短小精悍，定义了插件的接口——几个端口、什么类型、叫什么名字。读完就能理解"这个插件对外暴露了什么"。

**重点理解**：

1. **模块声明**（22-24行）：`pub mod model/plugin/stft` —— 三个子模块的关系
2. **端口定义**（38-51行）：5 个端口的索引常量
3. **`get_ladspa_descriptor()`**（62-125行）：这是 LADSPA 主机发现插件的**唯一入口函数**
   - 5 个端口的具体定义：2 个音频端口 + 3 个控制端口
   - 每个端口的类型、默认值、取值范围
4. **属性声明**（71行）：`PROP_HARD_REALTIME_CAPABLE | PROP_REALTIME` 表示这是一个实时插件

**带着问题读**：插件有哪些端口？每个端口是干什么的？LADSPA 主机怎么发现这个插件？

---

## 第 3 步：[src/stft.rs](ladspa/src/stft.rs)（213 行）

**为什么第三个读它**：STFT 是一个**自包含的 DSP 模块**，不依赖 `model.rs` 和 `plugin.rs`，可以独立理解。它是整个音频处理链的第一步和最后一步。

**重点理解**：

1. **常量定义**（29-30行）：`NFFT=512`、`HOP_SIZE=256`、`NUM_FREQ_BINS=257`
2. **为什么用 `sqrt(hann)` 窗**（16-20行注释）：保证 50% 重叠时的完美重构
3. **`StftProcessor` 结构体**：预分配所有缓冲区（零堆分配，满足实时要求）
4. **`process_frame()` 方法**：输入 256 个时域采样点 → 输出 257 个频域 bin（复数）
5. **`inverse_frame()` 方法**：输入增强后的频谱 → 重叠相加合成 → 输出 256 个时域采样点
6. **`realfft` crate 的使用**：纯实数 FFT，比复数 FFT 快近一倍

**带着问题读**：STFT 怎么把 256 个采样点变成 257 个频率 bin？iSTFT 怎么把频谱变回音频？重叠相加是什么意思？

---

## 第 4 步：[src/model.rs](ladspa/src/model.rs)（301 行）

**为什么第四个读它**：依赖 STFT 模块的概念（输入是频谱帧），但不依赖 `plugin.rs`。读完就能理解"模型怎么被加载和调用的"。

**重点理解**：

1. **模型嵌入**（21-25行）：`include_bytes!` 编译时把 `.ort` 文件嵌入二进制，无需外部文件
2. **缓存常量**（31-42行）：`CONV_SIZE=16896`、`TRA_SIZE=96`、`INTER_SIZE=1056`——这些和 Python 流式模型的缓存一一对应
3. **`GtcrnModel` 结构体**：
   - 两个 ONNX Session（轻量/全质量）
   - 所有状态缓存数组（`conv_cache`、`tra_cache`、`inter_cache`）
4. **`new()` 方法**：从嵌入的字节数组创建 ONNX Runtime Session
5. **`warmup()` 方法**：预热推理（避免首帧延迟）
6. **`process_frame()` 方法**：输入一帧频谱 + 缓存 → ONNX 推理 → 输出增强频谱 + 更新缓存
7. **`reset_state()` 方法**：重置所有缓存为零（处理新音频流时）

**带着问题读**：模型是怎么加载的（不需要外部文件）？流式推理的缓存是怎么管理的？两个模型（轻量/全质量）怎么切换？

---

## 第 5 步：[src/plugin.rs](ladspa/src/plugin.rs)（559 行）

**为什么最后读它**：这是**最复杂的模块**，把 `stft.rs` 和 `model.rs` 串联起来，加上重采样和双线程架构。只有理解了前两个模块，才能看懂这里的数据流。

**重点理解**：

1. **双线程架构**（注释第 10-27 行的 ASCII 图）：
   ```
   音频线程（实时）             工作线程（非实时）
   run() 回调                   worker_thread()
     ├─ 输入 → ringbuf ──────>   ├─ 降采样 48k→16k
     └─ ringbuf → 输出 <──────   ├─ STFT → 模型 → iSTFT
                                 └─ 升采样 16k→48k
   ```
2. **为什么用双线程**（23-28行注释）：LADSPA 音频回调不能阻塞，ONNX 推理有延迟抖动，必须分离
3. **`SharedState` 结构体**（74-82行）：原子变量实现无锁通信（`AtomicBool`、`AtomicU32`）
4. **`GtcrnPlugin` 结构体**：持有两个环形缓冲区、两个重采样器、模型实例、工作线程句柄
5. **`new()` 方法**：创建所有组件 + 启动工作线程
6. **`run()` 方法**（LADSPA 回调）：音频线程入口
   - 读输入 → 写入 `input_ring`
   - 从 `output_ring` 读取 → 强度混合（干湿比）→ 施加增益 → 写输出
   - **绝不阻塞**（没有锁，没有内存分配）
7. **`worker_thread()` 函数**：工作线程主循环
   - 从 `input_ring` 取数据 → `resampler_in` 降采样 48k→16k
   - 累积 256 个采样点 → `stft.process_frame()` → `model.process_frame()` → `stft.inverse_frame()`
   - `resampler_out` 升采样 16k→48k → 写入 `output_ring`
8. **重采样器**（`rubato` crate）：sinc 插值，高质量 48kHz ↔ 16kHz 转换

**带着问题读**：音频数据从输入到输出经过了哪些步骤？为什么需要双线程？环形缓冲区怎么避免锁？强度控制（干湿混合）是怎么实现的？

---

## 第 6 步：[build.rs](ladspa/build.rs)（161 行）

**为什么接着读构建**：理解了源码后，再看怎么把它编译成可用的 `.so` 文件。

**重点理解**：

1. **`main()` 函数**（16-84行）：
   - 默认模式：调用 `convert_models()` 转换 ONNX→ORT
   - `static` feature：查找 `onnxruntime-minimal/lib/`，链接所有静态库
2. **`convert_models()` 函数**（92-161行）：
   - 用 Python 的 `onnxruntime.tools.convert_onnx_models_to_ort` 工具
   - 源文件：`stream/onnx_models/gtcrn.onnx` 和 `gtcrn_simple.onnx`
   - 目标：`$OUT_DIR/gtcrn.ort` 和 `gtcrn_simple.ort`
   - 增量构建：只在源文件更新时才重新转换

**带着问题读**：ORT 格式是什么？为什么比 ONNX 好？静态链接时链接了哪些库？

---

## 第 7 步：[build.sh](ladspa/build.sh)（162 行）

**为什么读它**：这是用户直接运行的构建入口。

**重点理解**：

1. **`build_dynamic()`**（44-62行）：`cargo build --features dynamic`
2. **`build_static()`**（65-84行）：`cargo build --features download`（从微软下载 ORT）
3. **`build_minimal()`**（87-141行）：先用 MRI 脚本合并多个 `.a` 为单个 `libonnxruntime.a`，再 `cargo build --features static`

**带着问题读**：三种构建策略有什么区别？什么时候用哪种？

---

## 第 8 步：[build-minimal-docker.sh](ladspa/build-minimal-docker.sh)（65 行）

**为什么读它**：最小化构建的前置步骤，理解 Docker 如何编译精简版 ONNX Runtime。

**重点理解**：

1. 使用 `Dockerfile.minimal-ort` 构建镜像
2. 从容器中提取：静态库（`.a`）、头文件、转换后的 `.ort` 模型、`required_ops.config`
3. 输出目录：`ladspa/onnxruntime-minimal/`

---

## 第 9 步：[Dockerfile.minimal-ort](ladspa/Dockerfile.minimal-ort)（138 行）

**为什么读它**：理解 ONNX Runtime 的裁剪编译过程。

**重点理解**：

1. 基于 `ubuntu:rolling`，安装 CMake + protobuf + Eigen 等编译依赖
2. Clone ONNX Runtime v1.23.2 源码
3. 用 `required_ops.config` 进行**最小化编译**——只保留 GTCRN 需要的算子
4. 编译完成后提取产物

---

## 第 10 步：[required_ops.config](ladspa/required_ops.config)（2335 行，浏览即可）

**为什么最后看**：这是一个纯配置文件，不需要逐行阅读，浏览前面几行就能理解它的作用。

**了解即可**：
- 第一行列出了所有算子类型：`Add, BatchNormalization, Cast, Concat, Conv, ConvTranspose, GRU, MatMul, Mul, PRelu, Pad, Sigmoid, Slice, Sqrt, Tanh, Transpose...`
- 后续行是每个算子的具体实例配置
- ONNX Runtime 编译时根据这个清单裁剪掉不需要的算子（如 Attention、Gelu），将体积从 50MB 压缩到 6MB

---

## 第 11 步：[README.md](ladspa/README.md)（90 行）

**使用文档**，确认你理解了整体工作流程。重点关注 PipeWire 配置示例——理解插件在实际系统中怎么被使用。

---

## 模块依赖关系图

```
src/lib.rs  (入口 ─ 定义接口)
   │
   ├── src/stft.rs    (自包含, 无内部依赖)
   │      └── 提供: StftProcessor
   │
   ├── src/model.rs   (依赖: 概念上理解 STFT 频谱)
   │      └── 提供: GtcrnModel
   │
   └── src/plugin.rs  (依赖: stft + model + lib.rs 常量)
          └── 使用: StftProcessor + GtcrnModel + PORT_*
                 ↑
         外部 crate: ringbuf, rubato, ladspa

build.rs ─── 模型转换 (ONNX → ORT) + 静态链接
build.sh ─── 构建入口 (调用 cargo + 环境配置)
build-minimal-docker.sh ─── Docker 构建精简 ORT
Dockerfile.minimal-ort ─── ORT 裁剪编译
required_ops.config ─── 算子白名单
```

**阅读顺序严格对应依赖关系**：从叶子节点（无依赖的 `stft.rs`、`lib.rs`）逐步过渡到根节点（整合一切的 `plugin.rs`），最后再读构建系统。
