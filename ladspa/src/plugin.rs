//! GTCRN LADSPA 插件实现
//!
//! 本模块实现 LADSPA 插件接口，处理：
//! - 音频 I/O（主机采样率，通常 48kHz）
//! - 高质量重采样到模型采样率（16kHz），使用 sinc 插值
//! - 无锁环形缓冲区实现实时音频通信
//! - 可调节的增强强度（原始/处理音频混合）
//! - 通过控制端口切换模型
//!
//! ## 架构
//!
//! ```text
//! 音频线程（实时）                  工作线程（非实时）
//! ┌─────────────────────────┐       ┌─────────────────────────────┐
//! │ run() 回调               │       │ worker_thread()             │
//! │   ├─ 输入 → input_ring  │──────>│   ├─ 降采样 48k → 16k     │
//! │   └─ output_ring → 输出 │<──────│   ├─ STFT → 模型 → iSTFT   │
//! └─────────────────────────┘       │   └─ 升采样 16k → 48k     │
//!                                   └─────────────────────────────┘
//! ```
//!
//! ## 为什么用双线程架构？
//!
//! LADSPA 的音频回调 `run()` 运行在实时线程中，不能阻塞。
//! ONNX Runtime 推理可能偶尔花费较长时间（~1ms），
//! 如果将推理放在音频回调中，可能导致音频丢帧（xrun）。
//! 因此推理放在独立的工作线程中，通过无锁环形缓冲区通信。

use crate::model::{GtcrnModel, ModelType, NUM_FREQ_BINS};
use crate::stft::{StftProcessor, HOP_SIZE, NFFT};
use crate::{PORT_ENABLE, PORT_INPUT, PORT_MODEL, PORT_OUTPUT, PORT_STRENGTH};
use ladspa::{Plugin, PluginDescriptor, PortConnection};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapRb,
};
use rubato::{FftFixedIn, FftFixedOut, Resampler};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

// =============================================================================
// 常量
// =============================================================================

/// GTCRN 模型要求的采样率（16kHz）
const MODEL_SAMPLE_RATE: usize = 16_000;

/// 输出增益 —— 补偿模型自然导致的音量降低
/// 1.45 是经验值，使输出音量与输入大致匹配
const OUTPUT_GAIN: f32 = 1.45;

/// 工作线程轮询的最短睡眠时间（微秒）
const WORKER_SLEEP_MIN_US: u64 = 500;

/// 工作线程轮询的最长睡眠时间（微秒）
/// ~5ms 匹配典型音频缓冲区大小
/// 自适应回退：从最小睡眠开始，无数据时逐渐增加
const WORKER_SLEEP_MAX_US: u64 = 5000;

/// 环形缓冲区容量（采样点数）
/// 48kHz 下，1024 采样点/块的音频块约需 170ms 覆盖
const RING_BUFFER_SIZE: usize = 8192;

// =============================================================================
// 共享状态（原子操作）
// =============================================================================

/// 音频线程与工作线程之间的共享状态
///
/// 使用原子操作实现无锁通信，避免传统互斥锁带来的优先级反转问题。
/// 所有字段都是原子的 —— 一个线程写入，另一个读取，互不阻塞。
struct SharedState {
    /// 处理启用标志（true=降噪，false=旁路直通）
    enabled: AtomicBool,
    /// 强度值，用 u32 位表示（原子操作不支持 f32）
    /// 使用 f32::to_bits() / f32::from_bits() 转换
    strength_bits: AtomicU32,
    /// 模型类型值（同上，用 u32 原子存储）
    model_bits: AtomicU32,
    /// 关闭信号 —— 插件卸载时通知工作线程退出
    shutdown: AtomicBool,
    /// 初始化标志 —— 第一次 run() 调用后设置为 true
    /// 工作线程需要等待此标志才创建模型（需要端口值）
    initialized: AtomicBool,
}

impl SharedState {
    fn new(model_type: ModelType) -> Self {
        Self {
            enabled: AtomicBool::new(true),  // 默认启用
            strength_bits: AtomicU32::new(1.0_f32.to_bits()),  // 默认满强度
            model_bits: AtomicU32::new((model_type as u32 as f32).to_bits()),
            shutdown: AtomicBool::new(false),
            initialized: AtomicBool::new(false),
        }
    }

    // ---- 启用/禁用 ----
    #[inline]
    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    #[inline]
    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    // ---- 强度 ----
    #[inline]
    fn set_strength(&self, strength: f32) {
        let clamped = strength.clamp(0.0, 1.0);  // 限制在 [0.0, 1.0]
        self.strength_bits
            .store(clamped.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn get_strength(&self) -> f32 {
        f32::from_bits(self.strength_bits.load(Ordering::Relaxed))
    }

    // ---- 模型选择 ----
    #[inline]
    fn set_model(&self, value: f32) {
        self.model_bits.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn get_model_type(&self) -> ModelType {
        let value = f32::from_bits(self.model_bits.load(Ordering::Relaxed));
        ModelType::from_control(value)
    }

    // ---- 关闭信号 ----
    #[inline]
    fn should_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    #[inline]
    fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    // ---- 初始化 ----
    #[inline]
    fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    #[inline]
    fn set_initialized(&self) {
        self.initialized.store(true, Ordering::Release);
    }
}

// =============================================================================
// 工作线程
// =============================================================================

/// 音频处理工作线程
///
/// 在独立线程中处理音频，避免阻塞实时音频回调。
/// 使用预分配的缓冲区来防止处理过程中发生堆内存分配。
///
/// # 处理流水线
/// 1. 从环形缓冲区读取输入 → 降采样 48k→16k（如需要）
/// 2. 积累到 HOP_SIZE(256) 采样点 → STFT → 模型推理 → iSTFT
/// 3. 升采样 16k→48k（如需要）→ 应用增益和强度 → 写入输出环形缓冲区
fn worker_thread(
    mut input_consumer: ringbuf::HeapCons<f32>,    // 输入环形缓冲区消费者
    mut output_producer: ringbuf::HeapProd<f32>,    // 输出环形缓冲区生产者
    state: Arc<SharedState>,                        // 共享状态（原子操作）
    host_sample_rate: usize,                        // 主机采样率
) {
    // 判断是否需要重采样（主机采样率 != 16kHz）
    let needs_resample = host_sample_rate != MODEL_SAMPLE_RATE;
    let resample_ratio = host_sample_rate as f64 / MODEL_SAMPLE_RATE as f64;

    // 计算重采样块大小
    // 例如：48kHz 主机, 16kHz 模型, 比率 = 3.0
    // 每帧处理 HOP_SIZE(256) 个模型采样 → 主机需 768 个采样
    let host_chunk_size = (HOP_SIZE as f64 * resample_ratio).ceil() as usize;

    // ---- 初始化高质量 sinc 重采样器 ----
    // 降采样器：48k → 16k（使用 FFT 加速的 sinc 插值）
    let mut downsampler: Option<FftFixedIn<f32>> = if needs_resample {
        Some(
            FftFixedIn::new(host_sample_rate, MODEL_SAMPLE_RATE, host_chunk_size, 1, 1)
                .expect("降采样器创建失败"),
        )
    } else {
        None
    };

    // 升采样器：16k → 48k
    let mut upsampler: Option<FftFixedOut<f32>> = if needs_resample {
        Some(
            FftFixedOut::new(MODEL_SAMPLE_RATE, host_sample_rate, host_chunk_size, 1, 1)
                .expect("升采样器创建失败"),
        )
    } else {
        None
    };

    // ---- STFT 处理器（512点FFT, 256帧移）----
    let mut stft = StftProcessor::new(NFFT, HOP_SIZE);

    // 等待第一次 run() 调用设置端口值后再创建模型
    while !state.is_initialized() {
        if state.should_shutdown() {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }

    // 根据当前控制端口值创建模型
    let initial_model_type = state.get_model_type();
    let mut model = GtcrnModel::new(initial_model_type);

    // ---- 预分配固定大小的缓冲区（处理期间零堆分配）----
    // 注意：Vec 在初始化后只要不 push/pop，就不会触发堆分配
    let mut input_buffer = vec![0.0_f32; host_chunk_size + 64];     // 输入缓冲区（+64安全余量）
    let mut window = vec![0.0_f32; NFFT];                            // 时域分析窗
    let mut model_accum = vec![0.0_f32; HOP_SIZE * 4];              // 降采样后的累积器
    let mut model_accum_len: usize = 0;                              // 累积器已用长度

    // 重采样器 I/O 缓冲区（单通道，固定大小）
    let mut resample_in = vec![vec![0.0_f32; host_chunk_size + 64]];    // 降采样输入
    let mut resample_out = vec![vec![0.0_f32; HOP_SIZE + 64]];          // 降采样输出
    let mut upsample_in = vec![vec![0.0_f32; HOP_SIZE + 64]];           // 升采样输入
    let mut upsample_out = vec![vec![0.0_f32; host_chunk_size + 64]];   // 升采样输出

    // 输出累积器（升采样后的数据，等待写入环形缓冲区）
    let mut output_accum = vec![0.0_f32; host_chunk_size * 4];
    let mut output_accum_len: usize = 0;

    // 自适应回退：从最短睡眠开始，无数据时逐渐增加
    let mut current_sleep_us = WORKER_SLEEP_MIN_US;

    // 预分配频谱缓冲区（避免热路径上的内存分配）
    let mut spectrum_buffer: [(f32, f32); NUM_FREQ_BINS] = [(0.0, 0.0); NUM_FREQ_BINS];

    loop {
        // 检查关闭信号
        if state.should_shutdown() {
            break;
        }

        // 计算所需的采样数
        let required_samples = if needs_resample {
            downsampler.as_ref().unwrap().input_frames_next()
        } else {
            HOP_SIZE
        };

        // 检查环形缓冲区中可用的采样数（无锁读取）
        let available = input_consumer.occupied_len();
        if available < required_samples {
            // 数据不足 → 睡眠等待（自适应回退策略）
            thread::sleep(Duration::from_micros(current_sleep_us));
            // 每次无数据时加倍睡眠时间，最大到 WORKER_SLEEP_MAX_US
            current_sleep_us = (current_sleep_us * 2).min(WORKER_SLEEP_MAX_US);
            continue;
        }

        // 数据可用 → 重置回退到最短睡眠
        current_sleep_us = WORKER_SLEEP_MIN_US;

        // 从环形缓冲区读取采样（无锁操作）
        let samples_read = input_consumer.pop_slice(&mut input_buffer[..required_samples]);
        if samples_read < required_samples {
            // 竞态条件：检查后、读取前缓冲区被清空
            thread::sleep(Duration::from_micros(WORKER_SLEEP_MIN_US));
            continue;
        }

        // ---- 读取控制值（原子读取）----
        let is_enabled = state.is_enabled();
        let strength = state.get_strength();

        // 检查模型是否需要切换
        let requested_model = state.get_model_type();
        if model.model_type() != requested_model {
            model.set_model_type(requested_model);
        }

        // ---- 旁路模式：直通输入 ----
        if !is_enabled {
            // 直接将输入写入输出（需要保持同步的重采样）
            let written = output_producer.push_slice(&input_buffer[..samples_read]);
            if written < samples_read {
                // 输出缓冲区满 → 丢弃多余采样点，防止堆积
            }
            continue;
        }

        // ---- 降采样至 16kHz（如需要）----
        if needs_resample {
            // 将输入复制到降采样器缓冲区
            resample_in[0][..samples_read].copy_from_slice(&input_buffer[..samples_read]);

            let ds = downsampler.as_mut().unwrap();
            let frames_needed = ds.output_frames_next();
            resample_out[0].resize(frames_needed, 0.0);

            match ds.process_into_buffer(&resample_in, &mut resample_out, None) {
                Ok((_, out_frames)) => {
                    // 将降采样结果复制到模型累积器
                    let space_available = model_accum.len() - model_accum_len;
                    let to_copy = out_frames.min(space_available);
                    model_accum[model_accum_len..model_accum_len + to_copy]
                        .copy_from_slice(&resample_out[0][..to_copy]);
                    model_accum_len += to_copy;
                }
                Err(_) => {
                    // 降采样失败时回退：直接使用输入（采样率不对但防止静音）
                    let to_copy = samples_read.min(model_accum.len() - model_accum_len);
                    model_accum[model_accum_len..model_accum_len + to_copy]
                        .copy_from_slice(&input_buffer[..to_copy]);
                    model_accum_len += to_copy;
                }
            }
        } else {
            // 无需重采样 → 直接复制到模型累积器
            let to_copy = samples_read.min(model_accum.len() - model_accum_len);
            model_accum[model_accum_len..model_accum_len + to_copy]
                .copy_from_slice(&input_buffer[..to_copy]);
            model_accum_len += to_copy;
        }

        // ---- 逐帧处理：STFT → 模型 → iSTFT ----
        while model_accum_len >= HOP_SIZE {
            // 移位窗并添加新采样点
            window.copy_within(HOP_SIZE.., 0);                   // 旧数据左移
            window[NFFT - HOP_SIZE..].copy_from_slice(&model_accum[..HOP_SIZE]); // 新数据填入窗尾

            // 从累积器移除已使用的采样点
            model_accum.copy_within(HOP_SIZE..model_accum_len, 0);
            model_accum_len -= HOP_SIZE;

            // STFT 分析 → 模型推理 → iSTFT 合成
            let spectrum = stft.analyze(&window);
            spectrum_buffer.copy_from_slice(spectrum);

            let enhanced = model
                .process_frame(&spectrum_buffer)
                .unwrap_or(spectrum_buffer);  // 推理失败时回退到原始频谱
            let processed = stft.synthesize(&enhanced);  // 256个时域采样点

            // ---- 升采样回主机采样率（如需要）----
            if needs_resample {
                // 将处理结果复制到升采样器输入
                upsample_in[0][..processed.len()].copy_from_slice(processed);

                let us = upsampler.as_mut().unwrap();
                let frames_needed = us.output_frames_next();
                upsample_out[0].resize(frames_needed, 0.0);

                match us.process_into_buffer(&upsample_in, &mut upsample_out, None) {
                    Ok((_, out_frames)) => {
                        // 应用输出增益和强度，写入输出累积器
                        let space_available = output_accum.len() - output_accum_len;
                        let to_copy = out_frames.min(space_available);
                        let gain = OUTPUT_GAIN * strength;
                        for i in 0..to_copy {
                            output_accum[output_accum_len + i] = upsample_out[0][i] * gain;
                        }
                        output_accum_len += to_copy;
                    }
                    Err(_) => {
                        // 升采样失败回退：简单重复采样（低质量但可接受）
                        let space_available = output_accum.len() - output_accum_len;
                        let to_copy = (processed.len() * 3).min(space_available);
                        let gain = OUTPUT_GAIN * strength;
                        for i in 0..to_copy {
                            output_accum[output_accum_len + i] = processed[i / 3] * gain;
                        }
                        output_accum_len += to_copy;
                    }
                }
            } else {
                // 无需升采样 → 直接复制并应用增益
                let space_available = output_accum.len() - output_accum_len;
                let to_copy = processed.len().min(space_available);
                let gain = OUTPUT_GAIN * strength;
                for i in 0..to_copy {
                    output_accum[output_accum_len + i] = processed[i] * gain;
                }
                output_accum_len += to_copy;
            }
        }

        // ---- 将累积的输出写入环形缓冲区（无锁）----
        if output_accum_len > 0 {
            let written = output_producer.push_slice(&output_accum[..output_accum_len]);
            if written < output_accum_len {
                // 部分写入 → 保留剩余采样点
                output_accum.copy_within(written..output_accum_len, 0);
                output_accum_len -= written;
            } else {
                output_accum_len = 0;  // 全部写入，清空累积器
            }
        }
    }
}

// =============================================================================
// 插件实现
// =============================================================================

/// GTCRN LADSPA 插件实例
///
/// 每个插件实例拥有：
/// - 一对环形缓冲区（输入/输出），连接音频线程和工作线程
/// - 一个工作线程，执行实际的处理流水线
/// - 共享状态（原子变量），实现线程间通信
pub struct GtcrnPlugin {
    /// 输入环形缓冲区生产者端（音频线程写入）
    input_producer: ringbuf::HeapProd<f32>,
    /// 输出环形缓冲区消费者端（音频线程读取）
    output_consumer: ringbuf::HeapCons<f32>,
    /// 工作线程句柄（用于 Drop 时 join）
    worker: Option<JoinHandle<()>>,
    /// 共享状态
    state: Arc<SharedState>,
    /// 主机采样率（仅用于调试，此处标记为允许未使用）
    #[allow(dead_code)]
    host_sample_rate: usize,
}

impl GtcrnPlugin {
    /// 创建新的插件实例
    ///
    /// LADSPA 主机调用此函数时传入采样率等信息。
    ///
    /// # 初始化步骤
    /// 1. 创建无锁环形缓冲区
    /// 2. 预填充输出缓冲区（延迟补偿，平滑启动）
    /// 3. 初始化共享状态（默认使用轻量模型）
    /// 4. 启动工作线程
    ///
    /// # 返回
    /// Box<dyn Plugin> 满足 LADSPA 插件接口要求
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(_descriptor: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
        let host_sr = sample_rate as usize;

        // ---- 创建无锁环形缓冲区 ----
        let input_ring = HeapRb::<f32>::new(RING_BUFFER_SIZE);
        let output_ring = HeapRb::<f32>::new(RING_BUFFER_SIZE);

        let (input_producer, input_consumer) = input_ring.split();
        let (mut output_producer, output_consumer) = output_ring.split();

        // ---- 预填充输出缓冲区（延迟补偿）----
        // 两个模型帧（HOP_SIZE）的主机速率采样点
        // 确保音频开始时有足够的缓冲数据
        let resample_ratio = host_sr as f64 / MODEL_SAMPLE_RATE as f64;
        let prefill_samples = ((HOP_SIZE as f64 * resample_ratio) * 2.0) as usize;
        let zeros = vec![0.0_f32; prefill_samples];
        output_producer.push_slice(&zeros);

        // ---- 共享状态（默认使用轻量模型，启动后由控制端口决定）----
        let state = Arc::new(SharedState::new(ModelType::Simple));

        // ---- 启动工作线程 ----
        let worker = {
            let st = Arc::clone(&state);
            Some(thread::spawn(move || {
                worker_thread(input_consumer, output_producer, st, host_sr)
            }))
        };

        Box::new(Self {
            input_producer,
            output_consumer,
            worker,
            state,
            host_sample_rate: host_sr,
        })
    }
}

/// 插件销毁时的清理逻辑
impl Drop for GtcrnPlugin {
    fn drop(&mut self) {
        // 发送关闭信号
        self.state.request_shutdown();

        // 等待工作线程完成
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();  // 忽略 join 错误（线程可能已 panic）
        }
    }
}

/// LADSPA 插件接口实现
impl Plugin for GtcrnPlugin {
    /// 插件激活时调用（如 JACK transport 开始）
    fn activate(&mut self) {
        // 注意：环形缓冲区不能轻易清空（生产者/消费者独立）
        // 预填充确保了以已知状态启动
    }

    /// 插件停用时调用
    fn deactivate(&mut self) {
        // 无需操作 —— 工作线程继续运行
    }

    /// 音频处理回调 —— 实时线程中运行！
    ///
    /// LADSPA 主机定期调用此函数处理音频块。
    /// 关键：此函数中不能有任何阻塞操作（如互斥锁、内存分配）。
    /// 所有操作使用无锁原子操作和环形缓冲区。
    ///
    /// # 参数
    /// * `sample_count` - 此音频块的采样点数
    /// * `ports` - 插件端口数组（音频 + 控制）
    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        // ---- 提取端口数据 ----
        let input = ports[PORT_INPUT].unwrap_audio();        // &[f32]
        let mut output = ports[PORT_OUTPUT].unwrap_audio_mut(); // &mut [f32]
        let enable_control = *ports[PORT_ENABLE].unwrap_control();     // f32
        let strength_control = *ports[PORT_STRENGTH].unwrap_control();  // f32
        let model_control = *ports[PORT_MODEL].unwrap_control();        // f32

        // ---- 更新共享状态（原子写入，无锁）----
        self.state.set_enabled(enable_control >= 0.5);
        self.state.set_strength(strength_control);
        self.state.set_model(model_control);

        // ---- 通知工作线程端口值已就绪（首次run后允许创建模型）----
        if !self.state.is_initialized() {
            self.state.set_initialized();
        }

        // ---- 推送输入采样到环形缓冲区（无锁）----
        let written = self.input_producer.push_slice(&input[..sample_count]);
        if written < sample_count {
            // 环形缓冲区满 —— 丢弃部分数据（此处无法恢复旧数据）
        }

        // ---- 从环形缓冲区获取输出采样（无锁）----
        let read = self.output_consumer.pop_slice(&mut output[..sample_count]);

        // ---- 填充剩余输出（降级回退）----
        // 当输出缓冲区数据不足时（启动或缓冲区欠载），
        // 用略微衰减的直通输入填充，表示处理延迟
        if read < sample_count {
            for i in read..sample_count {
                output[i] = input[i] * 0.95;  // -0.45dB 衰减提示用户
            }
        }
    }
}
