"""
GTCRN 流式模型 (Streaming Model)
ShuffleNetV2 + SFE + TRA + 2 DPGRNN
超轻量级，33.0 MMACs，23.67 K参数

与 gtcrn.py 的区别：
  - 使用 StreamConv2d 替代普通 Conv2d（支持逐帧因果处理）
  - 使用 StreamConvTranspose2d 替代普通 ConvTranspose2d
  - TRA 改为 StreamTRA（带隐藏状态缓存）
  - DPGRNN 的 Inter-RNN 支持缓存状态传递
  - Encoder/Decoder 传递并更新状态缓存

流式处理的核心理念：
  音频逐帧到达 → 模型逐帧处理 → 不需要等待未来帧 → 实时输出
  每处理一帧，更新状态缓存传给下一帧使用
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from modules.convolution import StreamConv2d, StreamConvTranspose2d


class ERB(nn.Module):
    """
    等效矩形带宽 (ERB) 滤波器组
    用于频率维度的子带压缩/恢复，与离线模型完全一致
    """
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1  # 257 = 512/2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        """频率(Hz) → ERB尺度"""
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        """ERB尺度 → 频率(Hz)"""
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        """生成ERB三角滤波器组矩阵"""
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        # 第一个三角滤波器（上升沿）
        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        # 中间滤波器（上升+下降）
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        # 最后一个三角滤波器（下降沿）
        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """波段合并 (Band Merge): 线性频率 → ERB子带"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """波段分离 (Band Split): ERB子带 → 线性频率"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """
    子带特征提取 (Subband Feature Extraction)
    展开相邻频率信息，kernel_size=3时展开为3倍通道
    """
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))

    def forward(self, x):
        """x: (B,C,T,F) → (B, C*kernel_size, T, F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class StreamTRA(nn.Module):
    """
    流式时间递归注意力 (Streaming Temporal Recurrent Attention)

    与离线TRA的区别：接收并返回GRU的隐藏状态缓存，
    实现跨帧的状态传递，保证流式处理的连续性。
    """
    def __init__(self, channels):
        super().__init__()
        # GRU: 输入channels维，输出channels*2（门控信号），1层
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x, h_cache):
        """
        参数:
            x: (B,C,T,F) 输入特征，流式模式下T=1
            h_cache: (1,B,C) GRU隐藏状态缓存

        返回:
            x * At: 加权后的特征
            h_cache: 更新后的隐藏状态（供下一帧使用）
        """
        # 计算能量（每帧平方均值）
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        # GRU处理时序信号，传入并更新隐藏状态
        at, h_cache = self.att_gru(zt.transpose(1,2), h_cache)
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1) 广播到频率维度

        return x * At, h_cache


class ConvBlock(nn.Module):
    """
    基础卷积块 (Conv → BN → Act)
    注意：这里使用普通Conv2d（非流式），因为频率维度的操作不需要因果关系
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StreamGTConvBlock(nn.Module):
    """
    流式分组时间卷积块 (Streaming Group Temporal Convolution Block)

    与离线版GTConvBlock的区别：
    1. 深度卷积使用 StreamConv2d/StreamConvTranspose2d（支持因果缓存）
    2. TRA使用 StreamTRA（支持隐藏状态传递）
    3. forward接收并返回 conv_cache 和 tra_cache

    这些缓存使得模型能够逐帧处理，每帧只依赖过去的帧信息。
    """
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        # 选择卷积模块类型（普通卷积 or 转置卷积，以及对应的流式版本）
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        stream_conv_module = StreamConvTranspose2d if use_deconv else StreamConv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        # 逐点卷积1：1x1卷积（不需要流式，因为kernel_size=1无时间依赖）
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        # 深度卷积：使用流式版本，需要缓存历史帧
        self.depth_conv = stream_conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        # 逐点卷积2：1x1卷积
        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)

        # 流式时间注意力
        self.tra = StreamTRA(in_channels//2)

    def shuffle(self, x1, x2):
        """Channel Shuffle: 交错合并两组通道"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])  # (B,2C,T,F)
        return x

    def forward(self, x, conv_cache, tra_cache):
        """
        流式前向传播

        参数:
            x: (B, C, T, F) 输入，流式模式下T=1
            conv_cache: (B, C, (kT-1)*dT, F) 卷积缓存（历史帧）
            tra_cache: (1, B, C) TRA的GRU隐藏状态缓存

        返回:
            x: (B, C, T, F) 输出
            conv_cache: 更新后的卷积缓存
            tra_cache: 更新后的TRA缓存
        """
        # 通道分组
        x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]

        # 分支1：SFE + 深度可分离卷积 + TRA
        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        # 流式深度卷积：传入并更新缓存
        h1, conv_cache = self.depth_conv(h1, conv_cache)
        h1 = self.depth_act(self.depth_bn(h1))
        h1 = self.point_bn2(self.point_conv2(h1))

        # 流式TRA：传入并更新GRU隐藏状态
        h1, tra_cache = self.tra(h1, tra_cache)

        # Channel Shuffle合并
        x =  self.shuffle(h1, x2)

        return x, conv_cache, tra_cache


class GRNN(nn.Module):
    """
    分组循环神经网络 (Grouped RNN)
    将输入分成两组，用两个独立GRU处理，减少计算量
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size) 或 None
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """
    流式分组双路径循环神经网络 (Dual-Path GRNN)

    与离线版的区别：
    Inter-RNN (帧间RNN) 接收并返回缓存状态 inter_cache，
    实现跨帧的时间依赖性传递。

    两路径处理：
    1. Intra-RNN: 沿频率维度（帧内），不缓存（每帧独立处理完整频率）
    2. Inter-RNN: 沿时间维度（帧间），缓存隐藏状态传递到下一帧
    """
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        # Intra-RNN: 双向GRU沿频率处理（每帧内完整处理，无需缓存）
        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter-RNN: 单向GRU沿时间处理（需要缓存隐藏状态跨帧传递）
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x, inter_cache):
        """
        流式前向传播

        参数:
            x: (B, C, T, F)
            inter_cache: (1, B*F, hidden_size) Inter-RNN的隐藏状态缓存

        返回:
            dual_out: (B, C, T, F) 输出
            inter_cache: 更新后的Inter-RNN缓存
        """
        ## Intra-RNN: 沿频率维度的帧内处理
        # (B,C,T,F) → (B,T,F,C) → (B*T,F,C)
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]  # 双向GRU沿频率
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)  # 残差连接

        ## Inter-RNN: 沿时间维度的帧间处理（带缓存）
        # (B,T,F,C) → (B,F,T,C) → (B*F,T,C)
        x = intra_out.permute(0,2,1,3)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        # 传入并更新缓存
        inter_x, inter_cache = self.inter_rnn(inter_x, inter_cache)
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0,2,1,3)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)  # 残差连接

        # 恢复形状 (B,C,T,F)
        dual_out = inter_out.permute(0,3,1,2)

        return dual_out, inter_cache


class StreamEncoder(nn.Module):
    """
    流式编码器

    结构: ConvBlock×2 + StreamGTConvBlock×3
    - 前两个ConvBlock不变（频率维度操作，无时间依赖）
    - 后三个StreamGTConvBlock需要缓存（有时序因果依赖）

    缓存说明:
      conv_cache: (B, C, (kT-1)*dT, F)
        - [:2]  对应第3层 (kT=3,dT=1 → 缓存2帧)
        - [2:6] 对应第4层 (kT=3,dT=2 → 缓存4帧)
        - [6:16] 对应第5层 (kT=3,dT=5 → 缓存10帧)
      tra_cache: (3, 1, B, C) 三层StreamTRA的GRU隐藏状态
    """
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            # 第1层：频率下采样×2，9ch→16ch
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            # 第2层：频率下采样×2，分组卷积
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            # 第3-5层：流式GTConvBlock，不同膨胀率
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x, conv_cache, tra_cache):
        """
        流式前向

        参数:
            x: (B,C,T,F)
            conv_cache: (B, C, 16, F) 卷积缓存（前两层padding后的）
            tra_cache: (3, 1, B, C) TRA缓存

        返回:
            x, en_outs, conv_cache, tra_cache
        """
        en_outs = []
        # 前两层：普通卷积，无缓存
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)

        # 后三层：流式卷积，每层处理对应区域的缓存
        x, conv_cache[:,:, :2, :], tra_cache[0] = self.en_convs[2](x, conv_cache[:,:, :2, :], tra_cache[0]); en_outs.append(x)
        x, conv_cache[:,:, 2:6, :], tra_cache[1] = self.en_convs[3](x, conv_cache[:,:, 2:6, :], tra_cache[1]); en_outs.append(x)
        x, conv_cache[:,:, 6:16, :], tra_cache[2] = self.en_convs[4](x, conv_cache[:,:, 6:16, :], tra_cache[2]); en_outs.append(x)

        return x, en_outs, conv_cache, tra_cache


class StreamDecoder(nn.Module):
    """
    流式解码器

    结构: StreamGTConvBlock×3 + ConvBlock×2
    与encoder对称，通过跳跃连接恢复细节。

    缓存布局与encoder对称。
    """
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            # 膨胀率5，对应encoder第5层
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=True),
            # 膨胀率2，对应encoder第4层
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=True),
            # 膨胀率1，对应encoder第3层
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=True),
            # 频率上采样×2
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            # 最后一次上采样，输出2通道（实部+虚部mask）
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs, conv_cache, tra_cache):
        """
        参数:
            x: (B,C,T,F) 来自DPGRNN的特征
            en_outs: encoder各层输出（用于跳跃连接）
            conv_cache: decoder的卷积缓存
            tra_cache: decoder的TRA缓存

        返回:
            x, conv_cache, tra_cache
        """
        # 前三层：流式GTConvBlock + 跳跃连接
        x, conv_cache[:,:, 6:16, :], tra_cache[0] = self.de_convs[0](x + en_outs[4], conv_cache[:,:, 6:16, :], tra_cache[0])
        x, conv_cache[:,:, 2:6, :], tra_cache[1] = self.de_convs[1](x + en_outs[3], conv_cache[:,:, 2:6, :], tra_cache[1])
        x, conv_cache[:,:, :2, :], tra_cache[2] = self.de_convs[2](x + en_outs[2], conv_cache[:,:, :2, :], tra_cache[2])

        # 后两层：普通转置卷积 + 跳跃连接（上采样恢复频率维度）
        for i in range(3, 5):
            x = self.de_convs[i](x + en_outs[4-i])
        return x, conv_cache, tra_cache


class Mask(nn.Module):
    """
    复数比率掩码 (Complex Ratio Mask, CRM)

    将预测的复数mask与原始频谱相乘：
    增强频谱 = mask * 原始频谱 （复数乘法）
    """
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        # 复数乘法：(M_r + j*M_i) × (S_r + j*S_i)
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class StreamGTCRN(nn.Module):
    """
    流式 GTCRN 语音增强模型

    整体流程（逐帧处理）：
    1. 输入一帧频谱 (1, 257, 1, 2)
    2. ERB子带压缩：257→129
    3. SFE特征展开：3ch→9ch
    4. StreamEncoder：逐步提取特征 129→33
    5. DPGRNN×2：双路径RNN建模
    6. StreamDecoder：恢复频率维度 33→129
    7. ERB逆映射：129→257
    8. CRM掩码：复数乘法得到增强频谱
    9. 输出增强帧 + 更新后的所有缓存

    缓存设计（流式处理的核心）：
    - conv_cache: 卷积缓存，存储历史帧用于因果填充
      encoder: (1, 16, 16, 33)  decoder: (1, 16, 16, 33)
    - tra_cache: TRA的GRU隐藏状态
      encoder: (3, 1, 1, 16)  decoder: (3, 1, 1, 16)
    - inter_cache: DPGRNN的Inter-RNN隐藏状态
      [dpgrnn1_cache, dpgrnn2_cache]: each (1, 33, 16)
    """
    def __init__(self):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)

        self.encoder = StreamEncoder()

        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)

        self.decoder = StreamDecoder()

        self.mask = Mask()

    def forward(self, spec, conv_cache, tra_cache, inter_cache):
        """
        流式一帧前向推理

        参数:
            spec: (B, F, T, 2) = (1, 257, 1, 2) 一帧复数频谱
            conv_cache: [en_cache, de_cache] 编码器+解码器的卷积缓存
            tra_cache: [en_cache, de_cache] 编码器+解码器的TRA缓存
            inter_cache: [cache1, cache2] DPGRNN的Inter-RNN缓存

        返回:
            spec_enh: 增强后的频谱
            conv_cache, tra_cache, inter_cache: 更新后的缓存（供下一帧使用）
        """
        spec_ref = spec  # 保存原始频谱 (B,F,T,2)

        # 提取幅度、实部、虚部三通道特征
        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        # ERB压缩 + SFE展开
        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        # 流式编码器
        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat, conv_cache[0], tra_cache[0])

        # 双路径RNN（带Inter-RNN缓存）
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0])  # (B,16,T,33)
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1])  # (B,16,T,33)

        # 流式解码器
        m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])

        # ERB逆映射恢复频率维度
        m = self.erb.bs(m_feat)

        # CRM复数掩码应用
        spec_enh = self.mask(m, spec_ref.permute(0,3,2,1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)

        return spec_enh, conv_cache, tra_cache, inter_cache


if __name__ == "__main__":
    import os
    import time
    import soundfile as sf
    from tqdm import tqdm
    from gtcrn import GTCRN
    from modules.convert import convert_to_stream

    device = torch.device("cpu")

    # 加载离线模型 → 转换为流式模型
    model = GTCRN().to(device).eval()
    model.load_state_dict(torch.load('onnx_models/model_trained_on_dns3.tar', map_location=device)['model'])
    stream_model = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model, model)  # 将权重从离线模型转换到流式模型

    """流式转换验证"""
    ### 离线推理（参考基准）
    x = torch.from_numpy(sf.read('test_wavs/mix.wav', dtype='float32')[0])
    x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)[None]
    with torch.no_grad():
        y = model(x)
    y = torch.istft(y, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
    sf.write('test_wavs/enh.wav', y.squeeze(), 16000)

    ### 在线（流式）推理
    # 初始化所有缓存为零
    conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)  # [en, de]
    tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)     # [en, de]
    inter_cache = torch.zeros(2, 1, 33, 16).to(device)      # [dpgrnn1, dpgrnn2]
    # 逐帧推理代码（注释状态，需要时取消注释）
    # ys = []
    # times = []
    # for i in tqdm(range(x.shape[2])):
    #     xi = x[:,:,i:i+1]
    #     tic = time.perf_counter()
    #     with torch.no_grad():
    #         yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache)
    #     toc = time.perf_counter()
    #     times.append((toc-tic)*1000)
    #     ys.append(yi)
    # ys = torch.cat(ys, dim=2)

    # ys = torch.istft(ys, 512, 256, 512, torch.hann_window(512).pow(0.5)).detach().cpu().numpy()
    # sf.write('test_wavs/enh_stream.wav', ys.squeeze(), 16000)
    # print(">>> 推理时间: 均值: {:.1f}ms, 最大: {:.1f}ms, 最小: {:.1f}ms".format(sum(times)/len(times), max(times), min(times)))
    # print(">>> 流式误差:", np.abs(y-ys).max())
