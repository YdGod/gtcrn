"""
GTCRN 流式模型 - 完整推理与 ONNX 导出脚本

功能：
  1. 加载离线模型并转换为流式模型
  2. 离线推理验证（基准）
  3. 流式逐帧推理验证
  4. 导出 ONNX 格式模型
  5. 使用 ONNX Runtime 进行推理验证

ONNX (Open Neural Network Exchange) 导出意义：
  - 将 PyTorch 模型转为跨平台通用格式
  - 支持 C/C++/Rust 等多种语言部署
  - ONNX Runtime 提供更快的 CPU 推理速度
  - 支持在 LADSPA 插件等环境中使用
"""
import os
import time
import soundfile as sf
from tqdm import tqdm
from gtcrn import GTCRN
from modules.convert import convert_to_stream


device = torch.device("cpu")

# ================= 加载模型 =================
model = GTCRN().to(device).eval()
model.load_state_dict(torch.load('onnx_models/model_trained_on_dns3.tar', map_location=device)['model'])
stream_model = StreamGTCRN().to(device).eval()
# 将离线模型的权重转换到流式模型
convert_to_stream(stream_model, model)

"""流式转换验证"""

# ================= 离线推理（参考基准）=================
x = torch.from_numpy(sf.read('test_wavs/mix.wav', dtype='float32')[0])
x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)[None]
with torch.no_grad():
    y = model(x)
y = torch.istft(y, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
sf.write('test_wavs/enh.wav', y.squeeze(), 16000)

# ================= 流式逐帧推理 =================
# 初始化缓存
conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
inter_cache = torch.zeros(2, 1, 33, 16).to(device)
# ys = []  # 存储每帧输出
# times = []  # 测量每帧推理时间
# for i in tqdm(range(x.shape[2])):
#     xi = x[:,:,i:i+1]
#     tic = time.perf_counter()
#     with torch.no_grad():
#         yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache)
#     toc = time.perf_counter()
#     times.append((toc-tic)*1000)  # 毫秒
#     ys.append(yi)
# ys = torch.cat(ys, dim=2)

# ys = torch.istft(ys, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
# sf.write('test_wavs/enh_stream.wav', ys.squeeze(), 16000)
# print(">>> 推理时间: 均值: {:.1f}ms, 最大: {:.1f}ms, 最小: {:.1f}ms".format(sum(times)/len(times), max(times), min(times)))
# print(">>> 流式误差:", np.abs(y-ys).max())


# ================= ONNX 模型导出 =================
import onnx
import onnxruntime
from onnxsim import simplify  # ONNX模型简化工具
from librosa import istft

## 导出为ONNX格式
file = 'onnx_models/gtcrn.onnx'
if not os.path.exists(file):
    # 创建示例输入（一帧频谱 + 所有缓存）
    input = torch.randn(1, 257, 1, 2, device=device)
    torch.onnx.export(stream_model,
                    (input, conv_cache, tra_cache, inter_cache),  # 模型输入元组
                    file,
                    input_names = ['mix', 'conv_cache', 'tra_cache', 'inter_cache'],
                    output_names = ['enh', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out'],
                    opset_version=11,  # ONNX操作集版本
                    verbose = False)

    # 验证ONNX模型有效性
    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)

# 简化ONNX模型（减少冗余操作，提升推理速度）
if not os.path.exists(file.split('.onnx')[0]+'_simple.onnx'):
    model_simp, check = simplify(onnx_model)
    assert check, "简化后的ONNX模型验证失败"
    onnx.save(model_simp, file.split('.onnx')[0] + '_simple.onnx')


# ================= ONNX Runtime 推理验证 =================
# 创建ONNX Runtime推理会话（使用CPU执行提供器）
# session = onnxruntime.InferenceSession(file, None, providers=['CPUExecutionProvider'])
session = onnxruntime.InferenceSession(file.split('.onnx')[0]+'_simple.onnx', None, providers=['CPUExecutionProvider'])

# 初始化ONNX Runtime缓存（numpy数组）
conv_cache = np.zeros([2, 1, 16, 16, 33],  dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 16],  dtype="float32")
inter_cache = np.zeros([2, 1, 33, 16],  dtype="float32")

T_list = []  # 推理时间记录
outputs = []  # 每帧输出

inputs = x.numpy()
for i in tqdm(range(inputs.shape[-2])):
    tic = time.perf_counter()

    # ONNX Runtime 逐帧推理
    out_i,  conv_cache, tra_cache, inter_cache \
            = session.run([], {'mix': inputs[..., i:i+1, :],
                'conv_cache': conv_cache,
                'tra_cache': tra_cache,
                'inter_cache': inter_cache})

    toc = time.perf_counter()
    T_list.append(toc-tic)
    outputs.append(out_i)

# 拼接所有帧
outputs = np.concatenate(outputs, axis=2)
# iSTFT 恢复时域信号
enhanced = istft(outputs[...,0] + 1j * outputs[...,1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)
sf.write('test_wavs/enh_onnx.wav', enhanced.squeeze(), 16000)

# 打印性能指标
print(">>> ONNX误差:", np.abs(y - enhanced).max())  # 与PyTorch模型的误差
print(">>> 推理时间: 均值: {:.1f}ms, 最大: {:.1f}ms, 最小: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))
print(">>> RTF (实时因子):", 1e3*np.mean(T_list) / 16)  # RTF < 1 表示实时处理
