"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
超轻量级语音增强模型，33.0 MMACs 计算量，23.67 K 参数
参考论文: GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


class ERB(nn.Module):
    """
    等效矩形带宽 (Equivalent Rectangular Bandwidth) 滤波器组
    用于在频率维度上进行子带压缩和恢复

    核心思想：人耳对不同频率的感知不是线性的，低频分辨能力更强，
    ERB尺度模拟了这种特性。通过ERB滤波器组将257个线性频率bin
    压缩到64个ERB子带（保留低频65个bin不压缩），减少计算量。
    """            # 65个低频bin+64个ERB子带  = 129个输入频率bin
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        """
        初始化ERB滤波器组

        参数:
            erb_subband_1: 低频直接保留的bin数量 (不经过ERB压缩)
            erb_subband_2: ERB压缩后的高频子带数量
            nfft: FFT点数，默认512
            high_lim: ERB映射的高频上限 (Hz)，默认8000
            fs: 采样率，默认16000
        """
        super().__init__()
        # 生成ERB滤波器矩阵
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1  # 频率bin总数 (257 = 512/2 + 1)
        self.erb_subband_1 = erb_subband_1
        # 前向映射：高频线性频率 → ERB子带 (压缩)
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        # 逆向映射：ERB子带 → 高频线性频率 (恢复)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        # 使用预计算的滤波器矩阵作为权重，不参与梯度训练
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        """将频率(Hz)转换为ERB尺度"""
        # ERB转换公式：ERB = 21.4 * log10(0.00437 * freq + 1)
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        """将ERB尺度转换回频率(Hz)"""
        # 逆转换公式：freq = (10^(ERB/21.4) - 1) / 0.00437
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        """
        生成ERB滤波器组矩阵

        原理：在ERB尺度上均匀分布64个中心频率，每个中心频率对应一个
        三角滤波器，映射回线性频率尺度。64个滤波器相互重叠排列，
        形成重叠的三角形滤波器组。

        具体步骤：
        1. 将目标频率范围(2031Hz~8000Hz)转换到ERB尺度
        2. 在ERB尺度上均匀取64个点 → 转回Hz → 转回FFT bin索引
        3. 以每个bin索引为中心，生成三角形加权系数

        滤波器形状示意（查看对应频率bin的加权系数）：
          滤波器0:  ●╲             (只有下降沿，从bins[0]的1.0降到bins[1]的~0)
          滤波器1:  ╱●╲            (上升沿+下降沿，完整三角形)
          滤波器2:    ╱●╲
            ...
          滤波器62:              ╱●╲
          滤波器63:                ╱●  (只有上升沿，从~0升到bins[-1]的1.0)

        注意：相邻滤波器在边界处重叠，保证从高频bin到ERB子带的映射是平滑的。
        """
        # 低频65个bin直接保留，ERB压缩从第65个bin对应的频率开始
        low_lim = erb_subband_1/nfft * fs  # 65/512 × 16000 ≈ 2031 Hz
        # 将起止频率从Hz转换到ERB尺度
        erb_low = self.hz2erb(low_lim)     # 低端ERB值
        erb_high = self.hz2erb(high_lim)   # 高端ERB值 (8000 Hz对应)
        # 在ERB尺度上均匀采样64个点（人耳感知均匀）
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        # 将ERB采样点转回Hz → 再转回FFT bin索引
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        # 初始化滤波器矩阵: [64个ERB子带, 257个频率bin]
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        # 第一个滤波器（只有下降沿: bins[0]→bins[1] 从1降到~0）
        # 公式: (bins[1] - bin索引) / (bins[1] - bins[0])
        # 当bin=bins[0]时值为1.0，bin=bins[1]时值趋近于0
        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        # 中间62个滤波器：完整三角形（上升沿0→1 + 下降沿1→0）
        for i in range(erb_subband_2-2):
            # 上升沿: bins[i]→bins[i+1]，从0升到1
            # 公式: (bin索引 - bins[i]) / (bins[i+1] - bins[i])
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            # 下降沿: bins[i+1]→bins[i+2]，从1降到0
            # 公式: (bins[i+2] - bin索引) / (bins[i+2] - bins[i+1])
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        # 最后一个滤波器（只有上升沿: bins[-2]→bins[-1] 从~0升到1）
        # 与倒数第二个滤波器(滤波器62)的下降沿互补：
        # 滤波器62在bins[-2]~bins[-1]是下降沿(1→0)，取1减去它得到上升沿(0→1)
        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]

        # 裁掉低频前65个bin（它们直接保留，不参与ERB压缩）
        # 最终矩阵形状: [64, 192]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """
        波段合并 (Band Merge)：将线性频率压缩到ERB子带

        输入 x: (B,C,T,F) - Batch, Channel, Time, Frequency
        输出: (B,C,T, low_freqs + erb_subbands)
        """
        # 低频部分直接保留（人耳对此区域更敏感）
        x_low = x[..., :self.erb_subband_1]
        # 高频部分通过ERB滤波器组压缩
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """
        波段分离 (Band Split)：将ERB子带恢复为线性频率

        输入 x_erb: (B,C,T,F_erb) - F_erb是压缩后的频率维度
        输出: (B,C,T, nfreqs)
        """
        # 低频部分不变
        x_erb_low = x_erb[..., :self.erb_subband_1]
        # 高频部分从ERB子带逆映射回线性频率
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """
    子带特征提取 (Subband Feature Extraction)

    将相邻频率子带的信息展开拼接，类似于在频率方向做一个小滑动窗口，
    让每个位置能看到相邻频率的上下文信息。
    例如：kernel_size=3时，每个位置展开后包含自己和左右邻居共3个值。
    """
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        # nn.Unfold 将滑动窗口展开为列向量，类似卷积的im2col操作
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))

    def forward(self, x):
        """
        输入 x: (B,C,T,F)
        输出: (B, C*kernel_size, T, F)
        例如输入(B,3,T,129) → 输出(B,9,T,129)
        """
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """
    时间递归注意力 (Temporal Recurrent Attention)

    作用：在时间维度上学习注意力权重，让模型能够关注重要的时间帧，
    抑制不重要的帧。使用GRU捕获时序依赖关系。

    工作流程：
    1. 计算每帧的能量（平方均值）
    2. 通过GRU建模时间依赖
    3. 通过全连接层+Sigmoid生成注意力权重
    4. 将权重乘回原始特征
    """
    def __init__(self, channels):
        super().__init__()
        # GRU用于捕获时序依赖，输出2倍channel（用于门控）
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        # 将GRU输出映射回原始channel数
        self.att_fc = nn.Linear(channels*2, channels)
        # Sigmoid生成[0,1]范围的注意力权重
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """
        输入 x: (B,C,T,F)
        输出: x * 注意力权重，形状不变 (B,C,T,F)
        """
        # 计算每个时间帧的能量：对频率维度求平方均值
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        # GRU建模时间依赖
        at = self.att_gru(zt.transpose(1,2))[0]
        # 全连接层映射
        at = self.att_fc(at).transpose(1,2)
        # Sigmoid激活得到注意力权重
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1) 扩展维度用于广播

        return x * At


class ConvBlock(nn.Module):
    """
    基础卷积块

    结构: Conv → BatchNorm → Activation
    - 中间层使用 PReLU (可学习的参数化ReLU)
    - 最后一层使用 Tanh (输出范围[-1,1])
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        # 根据use_deconv选择使用转置卷积还是普通卷积
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        # 最后一层用Tanh，其他用PReLU
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """
    分组时间卷积块 (Group Temporal Convolution Block)

    这是GTCRN的核心模块，结合了ShuffleNetV2的思想：
    1. Channel Split：将输入特征按通道分成两半
    2. 一半经过深度可分离卷积处理（逐点卷积→深度卷积→逐点卷积）
    3. TRA时间注意力
    4. Channel Shuffle：将两半打乱重组

    这种设计大幅减少了计算量和参数量。
    """
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        # 时间维度的padding量 = (时间核大小-1) * 时间膨胀率 (用于因果卷积)
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        # 子带特征提取：3个相邻频率子带展开
        self.sfe = SFE(kernel_size=3, stride=1)

        # 逐点卷积1：1x1卷积扩展通道 (9 → hidden_channels)
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        # 深度卷积：分组卷积，每个通道独立进行时空卷积
        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        # 逐点卷积2：1x1卷积压缩回 in_channels//2
        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)

        # 时间递归注意力
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """
        Channel Shuffle操作

        将两个分组的信息交错混合，促进信息流通。
        x1, x2: (B,C,T,F) → 交叉拼接 → (B,2C,T,F)
        """
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """
        ShuffleNetV2风格的前向传播

        输入 x: (B, C, T, F)
        输出: (B, C, T, F)
        """
        # 通道分组：分成两半
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        # 分支1：经过SFE + 深度可分离卷积 + TRA处理
        x1 = self.sfe(x1)  # 子带特征展开
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))  # 逐点卷积1
        # 因果填充：只在时间维度的左侧填充，确保不泄露未来信息
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))  # 深度卷积
        h1 = self.point_bn2(self.point_conv2(h1))  # 逐点卷积2

        h1 = self.tra(h1)  # 时间注意力

        # Channel Shuffle：将处理过的h1和未处理的x2交错合并
        x =  self.shuffle(h1, x2)

        return x


class GRNN(nn.Module):
    """
    分组循环神经网络 (Grouped RNN)

    将输入和隐藏状态分成两组，分别用两个独立的GRU处理，
    然后合并。减少了每个GRU的参数量和计算量。
    类似于分组卷积的思想。
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # 两组独立的GRU，每组处理一半的特征
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, 序列长度, input_size)
        h: (层数, B, hidden_size)  可选，默认全零初始化
        """
        # 如果未提供隐藏状态，初始化为全零
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        # 将输入和隐藏状态各分成两组
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        # 两组GRU独立处理
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        # 合并输出和隐藏状态
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """
    分组双路径循环神经网络 (Grouped Dual-Path RNN)

    这是语音分离/增强中常用的架构，通过对特征图进行两个方向(RNN)处理
    来捕获长距离依赖：

    1. Intra-RNN (帧内RNN)：沿频率维度处理，捕获同一帧内不同频率的依赖
       输入形状: (B*T, F, C)  → 将B和T合并，在F维度上用RNN建模

    2. Inter-RNN (帧间RNN)：沿时间维度处理，捕获同一频率在不同时间的依赖
       输入形状: (B*F, T, C)  → 将B和F合并，在T维度上用RNN建模

    这种双路径设计既减少了计算量又有效建模了时频依赖。
    """
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        # Intra-RNN：沿频率维度的双向GRU（分组）
        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter-RNN：沿时间维度的单向GRU（分组）
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x):
        """
        输入 x: (B, C, T, F)
        输出: (B, C, T, F)
        """
        ## Intra-RNN：频率维度处理
        # 维度变换：(B,C,T,F) → (B,T,F,C) → (B*T,F,C)
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # 沿频率维度RNN处理 (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # 全连接层映射 (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # 恢复形状 (B,T,F,C)
        intra_x = self.intra_ln(intra_x)  # LayerNorm归一化
        intra_out = torch.add(x, intra_x)  # 残差连接

        ## Inter-RNN：时间维度处理
        # 维度变换：(B,T,F,C) → (B,F,T,C) → (B*F,T,C)
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]  # 沿时间维度RNN处理 (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # 全连接层映射 (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # 恢复 (B,T,F,C)
        inter_x = self.inter_ln(inter_x)  # LayerNorm归一化
        inter_out = torch.add(intra_out, inter_x)  # 残差连接

        # 输出恢复为 (B,C,T,F)
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)

        return dual_out


class Encoder(nn.Module):
    """
    编码器

    结构：5个卷积块（2个基础卷积 + 3个GTConvBlock）
    - 前两个ConvBlock用于下采样频率维度
    - 后三个GTConvBlock在不同膨胀率下提取时频特征

    输入: (B, 9, T, 129)
    输出: (B, 16, T, 33)  频率维度从129压缩到33
    """
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            # 第1层：初始卷积，将9通道扩展到16通道，频率下采样2倍
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            # 第2层：分组卷积，频率继续下采样2倍
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            # 第3层：GTConvBlock，膨胀率1（局部特征）
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            # 第4层：GTConvBlock，膨胀率2（中等感受野）
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            # 第5层：GTConvBlock，膨胀率5（大感受野）
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []  # 保存每层输出，用于decoder的跳跃连接
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    """
    解码器

    结构：与编码器对称，5个卷积块（3个GTConvBlock + 2个转置卷积）
    - 使用转置卷积上采样频率维度
    - 跳跃连接：每个decoder层接收对应encoder层的输出

    输入: (B, 16, T, 33) + 编码器中间输出
    输出: (B, 2, T, 129)  恢复为2通道（实部+虚部mask）
    """
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            # 膨胀率5，对应encoder最后一层
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            # 膨胀率2
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            # 膨胀率1
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            # 频率上采样2倍
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            # 最后一次上采样，输出2通道（实部和虚部的mask）
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            # 跳跃连接：当前输入 + 对应encoder层的输出
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x


class Mask(nn.Module):
    """
    复数比率掩码 (Complex Ratio Mask, CRM)

    将预测的复数mask与原始频谱相乘，得到增强后的频谱。
    CRM = (M_r + j*M_i) * (S_r + j*S_i)
        = (M_r*S_r - M_i*S_i) + j*(M_i*S_r + M_r*S_i)

    相比于只增强幅度的掩码，CRM同时修正相位信息。
    """
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        # 复数乘法：实部 = mask实*spec实 - mask虚*spec虚
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        # 复数乘法：虚部 = spec虚*mask实 + spec实*mask虚
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    """
    GTCRN 语音增强模型

    整体架构：
    1. ERB子带压缩：降低频率维度计算量
    2. SFE子带特征提取：展开相邻频率信息
    3. Encoder编码器：逐步提取时频特征
    4. DPGRNN×2：双路径循环神经网络建模长距离依赖
    5. Decoder解码器：恢复频率分辨率
    6. ERB逆映射：恢复原始频率维度
    7. CRM掩码：复数乘法得到增强频谱

    输入: (B, F, T, 2) 复数频谱
    输出: (B, F, T, 2) 增强后的复数频谱
    """
    def __init__(self):
        super().__init__()
        # ERB滤波器组：低频65bin直接保留，高频压缩到64个ERB子带
        self.erb = ERB(65, 64)
        # 子带特征提取：3个相邻频率bin展开
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()

        # 两个DPGRNN堆叠，加深时频建模能力
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)

        self.decoder = Decoder()

        self.mask = Mask()

    def forward(self, spec):
        """
        前向推理

        参数:
            spec: (B, F, T, 2) - 复数频谱，最后一维 [实部, 虚部]

        返回:
            spec_enh: (B, F, T, 2) - 增强后的复数频谱
        """
        spec_ref = spec  # 保存原始频谱用于mask (B,F,T,2)

        # 提取幅度、实部、虚部作为输入特征
        spec_real = spec[..., 0].permute(0,2,1)  # (B,T,F)
        spec_imag = spec[..., 1].permute(0,2,1)  # (B,T,F)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)  # 幅度谱
        # 拼接三个特征：(B,3,T,257)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)

        # ERB子带压缩：257 → 129 (65低频 + 64 ERB高频)
        feat = self.erb.bm(feat)  # (B,3,T,129)
        # SFE展开相邻频率：3通道 → 9通道
        feat = self.sfe(feat)     # (B,9,T,129)

        # 编码器：逐步提取特征，频率维度129→33
        feat, en_outs = self.encoder(feat)

        # 双路径RNN：建模时频长距离依赖
        feat = self.dpgrnn1(feat) # (B,16,T,33)
        feat = self.dpgrnn2(feat) # (B,16,T,33)

        # 解码器：恢复频率维度 33→129
        m_feat = self.decoder(feat, en_outs)

        # ERB逆映射：恢复原始频率维度 129→257
        m = self.erb.bs(m_feat)

        # 复数比率掩码：(B,2,T,F)
        spec_enh = self.mask(m, spec_ref.permute(0,3,2,1))
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)

        return spec_enh


if __name__ == "__main__":
    model = GTCRN().eval()

    """计算模型复杂度（计算量和参数量）"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print(flops, params)

    """因果性检查：验证模型是否严格遵守因果约束"""
    # 生成三组随机音频
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    # 拼接：前半部分相同(a)，后半部分不同(b vs c)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    # STFT变换
    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    x2 = torch.stft(x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    # 模型推理
    y1 = model(x1)[0]
    y2 = model(x2)[0]
    # iSTFT逆变换
    y1 = torch.istft(y1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y2 = torch.istft(y2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

    # 前半部分应该相同（因果性保证），后半部分可以不同
    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())  # 应该接近0
    print((y1[16000:] - y2[16000:]).abs().max())              # 应该较大
