//! GTCRN LADSPA 插件 —— 使用 ONNX Runtime 进行实时语音增强
//!
//! 本插件为 Linux 音频系统（如 PipeWire、JACK、PulseAudio）提供
//! 实时语音降噪功能。使用 GTCRN 神经网络模型通过 ONNX Runtime 推理。
//! 支持 OpenVINO 执行提供器以在 Intel 硬件上获得最佳 CPU 性能。
//!
//! ## 架构
//!
//! 插件由三个主要模块组成：
//! - `model`: GTCRN ONNX 模型加载与推理
//! - `plugin`: LADSPA 插件接口实现（音频线程 + 工作线程）
//! - `stft`: 实时短时傅里叶变换 (STFT/iSTFT)
//!
//! ## 端口
//!
//! 1. **音频输入** (Audio Input) — 单声道音频输入
//! 2. **音频输出** (Audio Output) — 增强后的单声道音频输出
//! 3. **开关** (Enable) — 切换开关，启用/禁用降噪处理
//! 4. **强度** (Strength) — 0.0~1.0，控制增强强度（干湿混合）
//! 5. **模型选择** (Model) — 0=轻量模型, 1=全质量模型

pub mod model;   // 模型加载与推理模块
pub mod plugin;  // LADSPA 插件接口实现模块
pub mod stft;    // STFT 短时傅里叶变换模块

use ladspa::{
    DefaultValue, PluginDescriptor, Port, PortDescriptor, HINT_INTEGER, HINT_TOGGLED,
    PROP_HARD_REALTIME_CAPABLE, PROP_REALTIME,
};

/// 唯一的 LADSPA 插件 ID
/// "ORTC" 的 ASCII 十六进制表示（ONNX RunTime C++）
pub const PLUGIN_ID: u64 = 0x4F52_5443;

/// 插件版本号（从 Cargo.toml 读取）
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ========== 端口索引定义 ==========
/// 音频输入端口索引
pub const PORT_INPUT: usize = 0;

/// 音频输出端口索引
pub const PORT_OUTPUT: usize = 1;

/// 启用/禁用控制端口索引
pub const PORT_ENABLE: usize = 2;

/// 增强强度控制端口索引
pub const PORT_STRENGTH: usize = 3;

/// 模型选择控制端口索引
pub const PORT_MODEL: usize = 4;

/// 返回 LADSPA 插件描述符（LADSPA 主机发现的入口函数）
///
/// LADSPA 主机通过调用此函数来发现插件：
/// - index=0 返回插件描述符
/// - index>0 返回 None（表示只有1个插件）
#[no_mangle]
#[allow(unsafe_code)]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn get_ladspa_descriptor(index: u64) -> Option<PluginDescriptor> {
    if index != 0 {
        return None;
    }

    Some(PluginDescriptor {
        unique_id: PLUGIN_ID,
        label: "gtcrn_mono",
        // 声明插件支持硬实时和实时处理
        properties: PROP_HARD_REALTIME_CAPABLE | PROP_REALTIME,
        name: "GTCRN 语音增强 (ORT)",
        maker: "GTCRN 模型 (c) 2024 Rong Xiaobin | LADSPA 插件 (c) 2026 Bruno Gonçalves",
        copyright: "MIT License",
        ports: vec![
            // 端口1: 音频输入
            Port {
                name: "输入",
                desc: PortDescriptor::AudioInput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            // 端口2: 音频输出
            Port {
                name: "输出",
                desc: PortDescriptor::AudioOutput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            // 端口3: 启用/禁用开关（拨动式）
            Port {
                name: "启用",
                desc: PortDescriptor::ControlInput,
                hint: Some(HINT_TOGGLED),  // 开关风格控件
                default: Some(DefaultValue::Value1),  // 默认：启用
                lower_bound: None,
                upper_bound: None,
            },
            // 端口4: 增强强度滑块
            Port {
                name: "强度",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::High),  // 默认：最大
                lower_bound: Some(0.0),              // 范围 [0.0, 1.0]
                upper_bound: Some(1.0),
            },
            // 端口5: 模型选择（整数型）
            Port {
                name: "模型 (0=轻量 1=全质量)",
                desc: PortDescriptor::ControlInput,
                hint: Some(HINT_INTEGER),   // 整数选择器
                default: Some(DefaultValue::Value1), // 默认：全质量
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
        ],
        // 插件实例化函数
        new: plugin::GtcrnPlugin::new,
    })
}

/// 重新导出 LADSPA 描述符函数（供 LADSPA 主机发现）
pub use ladspa::ladspa_descriptor;
