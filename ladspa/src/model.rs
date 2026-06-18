//! GTCRN 神经网络模型封装 —— 基于 ONNX Runtime
//!
//! 本模块提供 GTCRN ONNX 模型的高级接口：
//! - 从嵌入的二进制数据加载模型（编译时嵌入）
//! - 管理流式推理的状态缓存
//! - 逐帧频谱处理
//!
//! 模型格式：ORT (ONNX Runtime 优化格式)，编译时从 ONNX 转换而来

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

// =============================================================================
// 嵌入的模型数据（编译时嵌入到二进制中）
// =============================================================================

/// 嵌入的精简 ONNX 模型 (`gtcrn_simple.onnx`) — 更轻量、更快
/// `include_bytes!` 在编译时将文件内容嵌入为静态字节数组
static EMBEDDED_MODEL_SIMPLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gtcrn_simple.ort"));

/// 嵌入的全质量 ONNX 模型 (`gtcrn.onnx`)
/// 经过 ONNX Runtime 预优化，消除了 Range 等冗余算子
static EMBEDDED_MODEL_FULL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gtcrn.ort"));

// =============================================================================
// 常量定义
// =============================================================================

/// 频率 bins 数量（NFFT/2 + 1 = 512/2 + 1 = 257）
pub const NUM_FREQ_BINS: usize = 257;

/// 各状态数组的扁平化大小：
/// - conv_cache: (2, 1, 16, 16, 33) = 2*1*16*16*33 = 16896 个 f32
/// - tra_cache:  (2, 3, 1, 1, 16)  = 2*3*1*1*16  = 96 个 f32
/// - inter_cache:(2, 1, 33, 16)    = 2*1*33*16   = 1056 个 f32
/// - input_buf:  (1, 257, 1, 2)    = 1*257*1*2   = 514 个 f32
const CONV_SIZE: usize = 16896;   // 卷积缓存大小
const TRA_SIZE: usize = 96;       // 时间注意力缓存大小
const INTER_SIZE: usize = 1056;   // 帧间RNN缓存大小
const INPUT_SIZE: usize = NUM_FREQ_BINS * 2;  // 输入缓冲区大小

// =============================================================================
// 模型类型枚举
// =============================================================================

/// 模型类型选择
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    /// 简单/快速模型 (`gtcrn_simple.onnx`) —— 质量较低但速度更快
    Simple = 0,
    /// 全质量模型 (`gtcrn.onnx`) —— 质量最高
    Full = 1,
}

impl ModelType {
    /// 从控制值（浮点数）创建 ModelType
    /// 值 >= 0.5 → Full, 值 < 0.5 → Simple
    #[must_use]
    pub fn from_control(value: f32) -> Self {
        if value >= 0.5 {
            Self::Full
        } else {
            Self::Simple
        }
    }
}

// =============================================================================
// ONNX Runtime 会话创建
// =============================================================================

/// 从嵌入的模型字节创建推理会话
///
/// 配置：
/// - Level3 图优化（最高级别）
/// - 单线程推理（避免干扰实时音频线程）
/// - 从内存加载（无需磁盘文件）
fn create_session(model_bytes: &[u8]) -> Result<Session, ort::Error> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?  // 图优化 Level 3
        .with_intra_threads(1)?   // 单线程内操作（避免线程切换开销）
        .with_inter_threads(1)?   // 单线程间操作
        .commit_from_memory(model_bytes)  // 从内存字节创建会话
}

// =============================================================================
// 流式推理状态
// =============================================================================

/// GTCRN 循环状态 —— 流式推理的核心
///
/// 每次推理需要传入上一帧的状态，并返回更新后的状态。
/// 这保证了模型输出的连续性。
pub struct GtcrnState {
    /// 卷积状态: 形状 (2, 1, 16, 16, 33)，扁平化为 f32 数组
    /// [0] 编码器缓存, [1] 解码器缓存
    conv: Vec<f32>,
    /// 时间循环注意力状态: 形状 (2, 3, 1, 1, 16)，扁平化
    /// [0] 编码器TRA缓存, [1] 解码器TRA缓存
    tra: Vec<f32>,
    /// 帧间 DPGRNN 状态: 形状 (2, 1, 33, 16)，扁平化
    /// [0] DPGRNN1 缓存, [1] DPGRNN2 缓存
    inter: Vec<f32>,
    /// 预分配的输入缓冲区: 形状 (1, 257, 1, 2)
    input_buf: Vec<f32>,
    /// 预分配的输出缓冲区（避免每次推理时分配内存）
    output_buf: [(f32, f32); NUM_FREQ_BINS],
}

impl GtcrnState {
    /// 创建全零初始状态（流式推理开始时的状态）
    #[must_use]
    pub fn new() -> Self {
        Self {
            conv: vec![0.0_f32; CONV_SIZE],
            tra: vec![0.0_f32; TRA_SIZE],
            inter: vec![0.0_f32; INTER_SIZE],
            input_buf: vec![0.0_f32; INPUT_SIZE],
            output_buf: [(0.0_f32, 0.0_f32); NUM_FREQ_BINS],
        }
    }

    /// 重置状态为全零（例如切换模型时）
    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.tra.fill(0.0);
        self.inter.fill(0.0);
    }
}

impl Default for GtcrnState {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GtcrnState {
    fn clone(&self) -> Self {
        Self {
            conv: self.conv.clone(),
            tra: self.tra.clone(),
            inter: self.inter.clone(),
            // 缓冲区和输出数组不克隆数据（每次重新分配）
            input_buf: vec![0.0_f32; INPUT_SIZE],
            output_buf: [(0.0_f32, 0.0_f32); NUM_FREQ_BINS],
        }
    }
}

// =============================================================================
// 模型封装
// =============================================================================

/// GTCRN 模型封装 —— ONNX Runtime 推理接口
///
/// 生命周期：
/// 1. 创建实例 → 加载嵌入模型 → 初始化 ONNX 会话
/// 2. 逐帧调用 `process_frame()` 进行流式推理
/// 3. 可在运行时切换模型类型（轻量 ↔ 全质量）
pub struct GtcrnModel {
    /// 当前使用的模型类型
    model_type: ModelType,
    /// 流式推理状态（跨帧缓存）
    state: GtcrnState,
    /// ONNX Runtime 推理会话
    session: Session,
}

impl GtcrnModel {
    /// 创建指定类型的新模型实例
    ///
    /// 加载嵌入的模型字节 → 创建 ONNX Runtime 会话
    #[must_use]
    pub fn new(model_type: ModelType) -> Self {
        eprintln!("GTCRN-ORT: 正在创建 {:?} 模型实例", model_type);

        let model_bytes = match model_type {
            ModelType::Simple => EMBEDDED_MODEL_SIMPLE,
            ModelType::Full => EMBEDDED_MODEL_FULL,
        };

        let session = create_session(model_bytes).expect("创建 ONNX 会话失败");

        Self {
            model_type,
            state: GtcrnState::new(),
            session,
        }
    }

    /// 使用默认模型（轻量）创建实例
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(ModelType::Simple)
    }

    /// 获取当前模型类型
    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// 切换模型类型（轻量 ↔ 全质量）
    ///
    /// 如果目标类型与当前相同，不做操作。
    /// 切换后重置状态以避免不兼容的缓存。
    pub fn set_model_type(&mut self, model_type: ModelType) {
        if self.model_type != model_type {
            eprintln!("GTCRN-ORT: 正在切换到 {:?} 模型", model_type);

            let model_bytes = match model_type {
                ModelType::Simple => EMBEDDED_MODEL_SIMPLE,
                ModelType::Full => EMBEDDED_MODEL_FULL,
            };

            // 重新加载 ONNX Runtime 会话
            match create_session(model_bytes) {
                Ok(new_session) => {
                    self.session = new_session;
                    self.model_type = model_type;
                    self.state.reset();  // 切换模型后重置状态
                }
                Err(e) => {
                    eprintln!("GTCRN-ORT: 切换模型会话失败! 错误: {:?}", e);
                }
            }
        }
    }

    /// 重置流式推理状态
    ///
    /// 例如：音频流中断后重新开始时调用
    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    /// 处理单帧频谱 —— 流式推理的核心函数
    ///
    /// # 参数
    /// * `spectrum` - 输入复数频谱 [(real, imag); NUM_FREQ_BINS]，共257个频率bin
    ///
    /// # 返回
    /// * `Ok([(real, imag); 257])` - 增强后的复数频谱
    /// * `Err(...)` - ONNX 推理错误
    ///
    /// # 工作流程
    /// 1. 频谱数据扁平化填充到输入缓冲区
    /// 2. 创建所有输入张量（频谱 + 3个状态缓存）
    /// 3. ONNX Runtime 推理
    /// 4. 从输出中更新状态缓存
    /// 5. 提取增强频谱并返回
    pub fn process_frame(
        &mut self,
        spectrum: &[(f32, f32); NUM_FREQ_BINS],
    ) -> Result<[(f32, f32); NUM_FREQ_BINS], Box<dyn std::error::Error + Send + Sync>> {
        // 步骤1: 将频谱填充到输入缓冲区（扁平化：real/imag 交替）
        // 模型期望输入形状: (1, 257, 1, 2)
        for (i, &(re, im)) in spectrum.iter().enumerate() {
            self.state.input_buf[i * 2] = re;      // 实部
            self.state.input_buf[i * 2 + 1] = im;  // 虚部
        }

        // 步骤2: 创建 ONNX Runtime 张量引用（零拷贝，直接引用缓冲区）
        let input_tensor =
            TensorRef::from_array_view(([1usize, NUM_FREQ_BINS, 1, 2], &self.state.input_buf[..]))?;
        let conv_tensor =
            TensorRef::from_array_view(([2usize, 1, 16, 16, 33], &self.state.conv[..]))?;
        let tra_tensor = TensorRef::from_array_view(([2usize, 3, 1, 1, 16], &self.state.tra[..]))?;
        let inter_tensor =
            TensorRef::from_array_view(([2usize, 1, 33, 16], &self.state.inter[..]))?;

        // 步骤3: 运行 ONNX 推理
        let outputs = self.session.run(ort::inputs![
            input_tensor,   // mix: 输入频谱
            conv_tensor,    // conv_cache: 卷积状态
            tra_tensor,     // tra_cache: 时间注意力状态
            inter_tensor,   // inter_cache: 帧间RNN状态
        ])?;

        // 步骤4: 提取输出张量数据
        // 输出顺序: [enh, conv_cache_out, tra_cache_out, inter_cache_out]
        let (_, output_enh_data) = outputs[0].try_extract_tensor::<f32>()?;    // 增强频谱
        let (_, output_conv_data) = outputs[1].try_extract_tensor::<f32>()?;   // 更新后的卷积状态
        let (_, output_tra_data) = outputs[2].try_extract_tensor::<f32>()?;    // 更新后的TRA状态
        let (_, output_inter_data) = outputs[3].try_extract_tensor::<f32>()?;  // 更新后的帧间状态

        // 步骤5: 更新状态缓存（供下一帧使用）
        self.state.conv.copy_from_slice(output_conv_data);
        self.state.tra.copy_from_slice(output_tra_data);
        self.state.inter.copy_from_slice(output_inter_data);

        // 步骤6: 将增强频谱数据转回 (real, imag) 元组格式
        for (i, pair) in self.state.output_buf.iter_mut().enumerate() {
            *pair = (output_enh_data[i * 2], output_enh_data[i * 2 + 1]);
        }

        Ok(self.state.output_buf)
    }
}
