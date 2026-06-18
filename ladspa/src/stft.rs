//! 实时 STFT/iSTFT 处理 —— 重叠相加法合成
//!
//! 实现短时傅里叶变换及其逆变换，使用 `sqrt(hann)` 窗函数
//! 保证 50% 重叠时的完美重构（PR, Perfect Reconstruction）。
//!
//! ## 设计要点
//! - 预分配所有缓冲区（初始化后零堆分配）
//! - 返回内部缓冲区的切片引用（零拷贝）
//! - 512点 FFT，256采样点帧移（50% 重叠，匹配 GTCRN）
//!
//! ## STFT 参数
//! - NFFT = 512（FFT大小）
//! - HOP_SIZE = 256（帧移 = NFFT/2，50%重叠）
//! - 窗函数: sqrt(hann) — 分析窗和合成窗都用 sqrt(hann) = 完美重构
//!
//! ## 为什么用 sqrt(hann)？
//! 标准的汉宁窗在 50% 重叠时本身就能完美重构。
//! 但 GTCRN 模型在分析和合成阶段都使用 sqrt(hann) 窗
//! （等效于分析窗 = 合成窗 = sqrt(hann)，乘积 = hann），
//! 这样即使在模型修改了频谱后也能保持较好的重构质量。

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

use crate::model::NUM_FREQ_BINS;

/// 帧移大小（采样点数），256 对应 GTCRN 的 50% 重叠
pub const HOP_SIZE: usize = 256;

/// FFT 大小，512 为 GTCRN 的标准配置
pub const NFFT: usize = 512;

/// 实时流式音频的 STFT 处理器
///
/// 使用 512 点 FFT 和 256 采样点帧移（50% 重叠），
/// 与 GTCRN 模型的要求一致。
///
/// # 性能
/// 通过复用内部缓冲区最小化内存分配。
/// 所有方法返回内部缓冲区的切片引用，避免数据拷贝。
pub struct StftProcessor {
    /// FFT 大小（512）
    nfft: usize,
    /// 帧移大小（256）
    hop_size: usize,
    /// 分析/合成窗函数: sqrt(hann)
    /// 窗函数在初始化时计算一次，之后不再修改
    window: Vec<f32>,
    /// 前向 FFT 计划（实数 → 复数）
    fft: Arc<dyn RealToComplex<f32>>,
    /// 逆向 FFT 计划（复数 → 实数）
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// 重叠相加缓冲区（合成阶段用）
    /// 长度 = NFFT
    overlap_buffer: Vec<f32>,
    /// FFT 输入暂存缓冲区（应用窗函数后的数据）
    fft_scratch: Vec<f32>,
    /// FFT 输出频谱暂存缓冲区
    spectrum_scratch: Vec<Complex<f32>>,
    /// iFFT 输出时域暂存缓冲区
    ifft_scratch: Vec<f32>,
    /// 预分配的分析输出缓冲区（避免每次 analyze 都分配）
    output_spectrum: [(f32, f32); NUM_FREQ_BINS],
    /// 预分配的合成输出缓冲区（避免每次 synthesize 都分配）
    output_samples: Vec<f32>,
}

impl StftProcessor {
    /// 创建新的 STFT 处理器
    ///
    /// # 参数
    /// * `nfft` - FFT 大小（GTCRN 使用 512）
    /// * `hop_size` - 帧移（GTCRN 使用 256）
    ///
    /// 初始化所有缓冲区并预计算窗函数。
    #[must_use]
    pub fn new(nfft: usize, hop_size: usize) -> Self {
        // 创建 FFT 计划器
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);   // 前向实数FFT
        let ifft = planner.plan_fft_inverse(nfft);  // 逆向复数→实数FFT

        // 预计算 sqrt(Hann) 窗函数（周期性风格，匹配 PyTorch 默认行为）
        // hann[n] = 0.5 * (1 - cos(2πn/N))
        // window[n] = sqrt(hann[n])
        let window: Vec<f32> = (0..nfft)
            .map(|i| {
                let phase = 2.0 * PI * i as f32 / nfft as f32;
                let hann = 0.5 * (1.0 - phase.cos());  // 标准汉宁窗
                hann.sqrt()                              // 平方根汉宁窗
            })
            .collect();

        Self {
            nfft,
            hop_size,
            window,
            fft,
            ifft,
            overlap_buffer: vec![0.0; nfft],               // 重叠相加缓冲区
            fft_scratch: vec![0.0; nfft],                   // FFT输入暂存
            spectrum_scratch: vec![Complex::new(0.0, 0.0); nfft / 2 + 1], // 频谱（含Nyquist）
            ifft_scratch: vec![0.0; nfft],                  // iFFT输出
            output_spectrum: [(0.0, 0.0); NUM_FREQ_BINS],  // 257个频率bin
            output_samples: vec![0.0; hop_size],            // 256个时域采样点
        }
    }

    /// 分析：时域帧 → 频域频谱
    ///
    /// # 参数
    /// * `frame` - 时域采样点（必须恰好 `nfft` 个采样点）
    ///
    /// # 返回
    /// 内部频谱缓冲区的引用（(real, imag) 元组数组）。
    /// 引用有效期至下一次 `analyze` 调用。
    ///
    /// # 处理流程
    /// 1. 应用分析窗函数（sqrt(hann)）
    /// 2. 前向实数FFT
    /// 3. 转换为 (real, imag) 元组格式
    ///
    /// # Panics
    /// 仅在 FFT 内部出错时 panic（正常输入不会发生）
    #[inline]
    pub fn analyze(&mut self, frame: &[f32]) -> &[(f32, f32); NUM_FREQ_BINS] {
        // 步骤1: 应用分析窗函数
        for (i, (&sample, &win)) in frame.iter().zip(&self.window).enumerate() {
            self.fft_scratch[i] = sample * win;
        }

        // 步骤2: 前向 FFT（实数 → 复数）
        self.fft
            .process(&mut self.fft_scratch, &mut self.spectrum_scratch)
            .expect("FFT 处理失败");

        // 步骤3: 转换为 (real, imag) 元组 → 预分配的输出缓冲区
        for (i, c) in self.spectrum_scratch.iter().enumerate().take(NUM_FREQ_BINS) {
            self.output_spectrum[i] = (c.re, c.im);
        }

        &self.output_spectrum
    }

    /// 合成：频域频谱 → 时域采样点
    ///
    /// 使用重叠相加法 (Overlap-Add) 实现帧间平滑过渡。
    ///
    /// # 参数
    /// * `spectrum` - 复数频谱 [(real, imag); NUM_FREQ_BINS]
    ///
    /// # 返回
    /// 时域采样点切片（`hop_size` 个采样点）。
    /// 引用有效期至下一次 `synthesize` 调用。
    ///
    /// # 处理流程
    /// 1. (real, imag) 元组 → Complex 格式
    /// 2. 确保 DC 和 Nyquist 频率的虚部为零
    /// 3. 逆向 FFT（复数 → 实数）
    /// 4. 归一化 + 应用合成窗函数
    /// 5. 重叠相加到缓冲区
    /// 6. 输出 `hop_size` 个采样点
    #[inline]
    pub fn synthesize(&mut self, spectrum: &[(f32, f32); NUM_FREQ_BINS]) -> &[f32] {
        // 步骤1: 将 (real, imag) 元组转为 Complex 格式
        for (i, &(re, im)) in spectrum.iter().enumerate() {
            self.spectrum_scratch[i] = Complex::new(re, im);
        }

        // 步骤2: DC (0 Hz) 和 Nyquist (fs/2) 频率的虚部必须为0
        // 因为实数信号的频谱在 DC/Nyquist 处是纯实数
        self.spectrum_scratch[0].im = 0.0;
        self.spectrum_scratch[NUM_FREQ_BINS - 1].im = 0.0;

        // 步骤3: 逆向 FFT（复数 → 实数）
        if self
            .ifft
            .process(&mut self.spectrum_scratch, &mut self.ifft_scratch)
            .is_err()
        {
            // FFT 失败时输出静音（安全回退）
            self.output_samples.fill(0.0);
            return &self.output_samples;
        }

        // 步骤4: 归一化（除以 NFFT）+ 应用合成窗函数
        // FFTW/realfft 的逆向FFT需要手动除以N来归一化
        let scale = 1.0 / self.nfft as f32;
        for (i, sample) in self.ifft_scratch.iter_mut().enumerate() {
            *sample *= scale * self.window[i];  // 归一化 + 合成窗
        }

        // 步骤5: 重叠相加积累到缓冲区
        // 将加窗后的NFFT个采样点累加到overlap缓冲区
        for (i, &sample) in self.ifft_scratch.iter().enumerate() {
            self.overlap_buffer[i] += sample;
        }

        // 步骤6: 输出前 hop_size 个采样点（已完成重叠相加的部分）
        self.output_samples
            .copy_from_slice(&self.overlap_buffer[..self.hop_size]);

        // 步骤7: 移位缓冲区 —— 移除已输出的采样点，为下一帧腾空间
        // 将 [hop_size..nfft] 移到 [0..nfft-hop_size]
        self.overlap_buffer.copy_within(self.hop_size.., 0);
        // 清空剩余部分（为下一帧的累加做准备）
        self.overlap_buffer[self.nfft - self.hop_size..].fill(0.0);

        &self.output_samples
    }
}
