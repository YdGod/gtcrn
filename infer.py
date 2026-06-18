"""
GTCRN 语音增强推理脚本

功能：加载预训练模型，对嘈杂音频进行降噪增强，输出清晰的语音。
使用流程：
  1. 加载预训练模型权重
  2. 读取嘈杂音频 (16kHz)
  3. STFT变换为频谱
  4. 模型推理得到增强频谱
  5. iSTFT逆变换得到增强音频
  6. 保存增强后的音频文件
"""
import os
import torch
import soundfile as sf  # 音频文件读写库
from gtcrn import GTCRN


## ================= 加载预训练模型 =================
# 使用CPU推理（如有GPU可改为 cuda）
device = torch.device("cpu")
model = GTCRN().eval()  # 创建模型并设为评估模式（禁用dropout/batchnorm）
# 加载在DNS3数据集上训练的模型权重
ckpt = torch.load(os.path.join('checkpoints', 'model_trained_on_dns3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

## ================= 加载测试音频 =================
# 读取16kHz单声道嘈杂音频
mix, fs = sf.read(os.path.join('test_wavs', 'Game-noise.wav'), dtype='float32')
assert fs == 16000  # 确保采样率正确，模型仅支持16kHz

## ================= 模型推理 =================
# 步骤1: STFT短时傅里叶变换（时域→频域）
# n_fft=512: FFT点数, hop=256: 帧移, win_length=512: 窗长
# window=sqrt(hann): 平方根汉宁窗（用于完美重构）
# return_complex=True: 返回复数频谱
input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)

# 步骤2: 复数转实数表示 (F,T) → (F,T,2)，最后一维[实部, 虚部]
input = torch.view_as_real(input)

# 步骤3: 模型推理（禁用梯度计算以节省内存）
with torch.no_grad():
    output = model(input[None])[0]  # 添加batch维度 → 推理 → 去除batch维度

# 步骤4: 实数转回复数 (F,T,2) → (F,T)
output = torch.view_as_complex(output.contiguous())

# 步骤5: iSTFT逆短时傅里叶变换（频域→时域）
enh = torch.istft(output, 512, 256, 512, torch.hann_window(512).pow(0.5))

## ================= 保存增强音频 =================
sf.write(os.path.join('test_wavs', 'Game-noise-enhanced.wav'), enh.detach().cpu().numpy(), fs)
print("增强完成！输出文件：test_wavs/Game-noise-enhanced.wav")
