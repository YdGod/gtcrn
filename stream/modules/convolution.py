# -*- coding: utf-8 -*-
"""
流式卷积模块 (Streaming Convolution)

为了支持实时逐帧处理而设计的因果卷积层。

核心思想：
  音频信号是时间序列，实时处理时不能"看到"未来。
  流式卷积通过在时间维度左侧缓存历史帧来实现因果填充，
  每帧只计算当前输入，不依赖未来帧。

Created on Sat Dec  3 17:32:08 2022
Modified on Fri Mar  7 21:25:18 2025

@author: Xiaohuai Le, Xiaobin Rong

导出ONNX时的注意事项：
  确保缓存以tensor形式保存（而非list），因为ONNX不支持动态list。
  本实现中使用torch.cat拼接缓存和输入，然后用普通卷积处理。
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Union

"""
当导出为 ONNX 格式时，确保缓存以张量 (tensor) 保存，而非列表 (list)。
因为 ONNX 不支持动态长度的列表结构。
"""

class StreamConv1d(nn.Module):
    """
    流式一维卷积

    用于沿时间维度的因果卷积。
    缓存机制：保留输入的历史帧，与新帧拼接后进行卷积。

    与非流式的区别：
      非流式: Conv1d(pad(input))  —— 一次性padding所有帧
      流式:   Conv1d(cat(cache, x)) —— 逐帧缓存历史拼接
    """
    def __init__(self,
                in_channels: int,       # 输入通道数
                out_channels: int,      # 输出通道数
                kernel_size: int,       # 卷积核大小（仅时间维度）
                stride: int=1,          # 步长
                padding: int=0,         # 填充（必须为0，由缓存替代）
                dilation: int=1,        # 膨胀率
                groups: int=1,          # 分组数
                bias: bool=True,        # 是否使用偏置
                *args, **kargs):
        super(StreamConv1d, self).__init__(*args, *kargs)

        # 因果流式要求padding=0，时间维度的因果填充由缓存实现
        assert padding == 0, "为满足因果流式要求，padding必须为0"

        self.StreamConv1d = nn.Conv1d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)

    def forward(self, x, cache):
        """
        流式卷积前向

        参数:
            x:     [bs, C, T_size] 当前输入（T_size通常为1，逐帧处理）
            cache: [bs, C, T_size-1] 历史帧缓存

        返回:
            oup:       卷积输出
            out_cache: 更新后的缓存（供下一帧使用）
        """
        # 缓存+新帧拼接 → 形成完整的时间上下文
        inp = torch.cat([cache, x], dim=-1)
        oup = self.StreamConv1d(inp)
        # 更新缓存：保留最近的历史帧（滑动窗口）
        out_cache = inp[..., 1:]
        return oup, out_cache


class StreamConv2d(nn.Module):
    """
    流式二维卷积

    用于时频域的因果卷积，仅在时间维度有缓存依赖，
    频率维度的填充使用普通padding。

    设计要点：
      - 时间维度 padding=0（因果要求），用缓存替代
      - 频率维度 padding 可以非零（频率方向不需要因果约束）
      - 输入形状：[bs, C, 1, F]，T=1表示逐帧处理
    """
    def __init__(self,
                 in_channels: int,                           # 输入通道数
                 out_channels: int,                          # 输出通道数
                 kernel_size: Union[int, Tuple[int, int]],   # [时间核大小, 频率核大小]
                 stride: Union[int, Tuple[int, int]] = 1,    # [时间步长, 频率步长]
                 padding: Union[str, int, Tuple[int, int]] = 0,  # [时间填充, 频率填充]
                 dilation: Union[int, Tuple[int, int]] = 1,  # [时间膨胀, 频率膨胀]
                 groups: int = 1,                             # 分组数
                 bias: bool = True,                           # 偏置
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        kernel_size 默认顺序: [T_size, F_size]  (时间维度在前，频率维度在后)
        """
        # 解析padding参数（支持int或tuple两种形式）
        if type(padding) is int:
            self.T_pad = padding   # 时间维度填充
            self.F_pad = padding   # 频率维度填充
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError('无效的填充大小。')

        # 时间维度padding必须为0（因果性要求，由缓存实现）
        assert self.T_pad == 0, "为满足因果流式要求，时间维度padding必须为0"

        self.StreamConv2d = nn.Conv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = groups,
                                bias = bias)

    def forward(self, x, cache):
        """
        流式二维卷积前向

        参数:
            x:     [bs, C, 1, F] 当前输入帧（T=1）
            cache: [bs, C, T_cache, F] 历史帧缓存（T_cache = 时间核大小-1的膨胀后长度）

        返回:
            outp:      卷积输出 [bs, C_out, 1, F_out]
            out_cache: 更新后的缓存 [bs, C, T_cache, F]
        """
        # 缓存 + 当前帧在时间维度拼接
        inp = torch.cat([cache, x], dim=2)
        outp = self.StreamConv2d(inp)
        # 移除最旧的一帧，保留最近的T_cache帧
        out_cache = inp[:,:, 1:]
        return outp, out_cache

## 版本 1
## 此实现推理速度较慢，详情参见 https://github.com/Xiaobin-Rong/gtcrn/issues/37
# class StreamConvTranspose2d(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: Union[int, Tuple[int, int]],
#                  stride: Union[int, Tuple[int, int]] = 1,
#                  padding: Union[str, int, Tuple[int, int]] = 0,
#                  dilation: Union[int, Tuple[int, int]] = 1,
#                  groups: int = 1,
#                  bias: bool = True,
#                  *args, **kargs):
#         super().__init__(*args, **kargs)
#         """
#         kernel_size 默认顺序: [T_size, F_size]
#         stride 默认顺序: [T_stride, F_stride]，且要求 T_stride == 1
#         """
#         if type(kernel_size) is int:
#             self.T_size = kernel_size
#             self.F_size = kernel_size
#         elif type(kernel_size) in [list, tuple]:
#             self.T_size, self.F_size = kernel_size
#         else:
#             raise ValueError('无效的卷积核大小。')

#         if type(stride) is int:
#             self.T_stride = stride
#             self.F_stride = stride
#         elif type(stride) in [list, tuple]:
#             self.T_stride, self.F_stride = stride
#         else:
#             raise ValueError('无效的步长大小。')

#         assert self.T_stride == 1   # 时间维度步长必须为1

#         if type(padding) is int:
#             self.T_pad = padding
#             self.F_pad = padding
#         elif type(padding) in [list, tuple]:
#             self.T_pad, self.F_pad = padding
#         else:
#             raise ValueError('无效的填充大小。')

#         if type(dilation) is int:
#             self.T_dilation = dilation
#             self.F_dilation = dilation
#         elif type(dilation) in [list, tuple]:
#             self.T_dilation, self.F_dilation = dilation
#         else:
#             raise ValueError('无效的膨胀率大小。')

#         assert self.T_pad == (self.T_size-1) * self.T_dilation, "为满足因果流式要求，..."

#         self.ConvTranspose2d = nn.ConvTranspose2d(in_channels = in_channels,
#                                                 out_channels = out_channels,
#                                                 kernel_size = kernel_size,
#                                                 stride = stride,
#                                                 padding = padding,
#                                                 dilation = dilation,
#                                                 groups = groups,
#                                                 bias = bias)

#     def forward(self, x, cache):
#         """
#         x: [bs, C, 1, F]
#         cache: [bs, C, T_size-1, F]
#         """
#         inp = torch.cat([cache, x], dim=2)
#         outp = self.ConvTranspose2d(inp)
#         out_cache = inp[:,:, 1:]
#         return outp, out_cache

## 版本 2 (推荐使用)
class StreamConvTranspose2d(nn.Module):
    """
    流式二维转置卷积 (Version 2)

    关键技巧：使用普通 Conv2d 实现转置卷积的效果

    为什么不用 nn.ConvTranspose2d？
      - 转置卷积的输出在因果模式下难以正确实现
      - 用 Conv2d + 权重翻转 + 上采样来等价实现更简单

    实现原理：
      1. 将转置卷积权重在时间+频率维度翻转
      2. 在频率维度手动上采样（插入零 + padding）
      3. 用普通Conv2d处理（带0填充的因果卷积）

    为什么在频率维度上采样？
      Encoder中频率维度被下采样了（stride=(1,2)），
      Decoder需要恢复频率维度，所以要在频率方向插入零值来上采样。
    """
    def __init__(self,
                 in_channels: int,                           # 输入通道数
                 out_channels: int,                          # 输出通道数
                 kernel_size: Union[int, Tuple[int, int]],   # [T, F] 卷积核大小
                 stride: Union[int, Tuple[int, int]] = 1,    # [T, F] 步长
                 padding: Union[str, int, Tuple[int, int]] = 0,  # [T, F] 填充
                 dilation: Union[int, Tuple[int, int]] = 1,  # [T, F] 膨胀率
                 groups: int = 1,                             # 分组数
                 bias: bool = True,                           # 偏置
                 *args, **kargs):
        super().__init__(*args, **kargs)
        """
        kernel_size 默认顺序: [T_size, F_size]
        stride 默认顺序: [T_stride, F_stride]，要求 T_stride == 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 解析 kernel_size
        if type(kernel_size) is int:
            self.T_size = kernel_size
            self.F_size = kernel_size
        elif type(kernel_size) in [list, tuple]:
            self.T_size, self.F_size = kernel_size
        else:
            raise ValueError('无效的卷积核大小。')

        # 解析 stride
        if type(stride) is int:
            self.T_stride = stride
            self.F_stride = stride
        elif type(stride) in [list, tuple]:
            self.T_stride, self.F_stride = stride
        else:
            raise ValueError('无效的步长大小。')

        assert self.T_stride == 1  # 时间步长必须为1（流式要求）

        # 解析 padding
        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError('无效的填充大小。')
        assert(self.T_pad == 0)  # 时间维度padding为0（因果要求）

        # 解析 dilation
        if type(dilation) is int:
            self.T_dilation = dilation
            self.F_dilation = dilation
        elif type(dilation) in [list, tuple]:
            self.T_dilation, self.F_dilation = dilation
        else:
            raise ValueError('无效的膨胀率大小。')

        # 使用 Conv2d 实现转置卷积 —— 核心技巧！
        # 权重会在 convert_to_stream() 中进行时间维度翻转
        # 频率维度的上采样在forward中手动实现
        self.StreamConvTranspose2d = nn.Conv2d(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        stride = (self.T_stride, 1),  # 频率维度的上采样需要手动做
                                        padding = (self.T_pad, 0),    # 频率维度padding手动做
                                        dilation = dilation,
                                        groups = groups,
                                        bias = bias)

    def forward(self, x, cache):
        """
        流式转置卷积前向

        参数:
            x:     [bs, C, 1, F] 当前输入帧
            cache: [bs, C, T-1, F] 历史缓存

        返回:
            outp:      输出 [bs, C_out, 1, F_out]
            out_cache: 更新后的缓存
        """
        # [bs, C, T, F] 缓存+输入拼接
        inp = torch.cat([cache, x], dim = 2)
        out_cache = inp[:, :, 1:]  # 更新缓存（滑动窗口）
        bs, C, T, F = inp.shape

        # 频率维度的上采样操作（如果F_stride > 1）
        if self.F_stride > 1:
            # 在频率维度插入F_stride-1个零 → 实现上采样
            # [bs,C,T,F] → [bs,C,T,F,1] → [bs,C,T,F,F_stride] → [bs,C,T,F*F_stride]
            inp = torch.cat([inp[:,:,:,:,None], torch.zeros([bs,C,T,F,self.F_stride-1])], dim = -1).reshape([bs,C,T,-1])
            left_pad = self.F_stride - 1
            if self.F_size > 1:
                if left_pad <= self.F_size - 1:
                    # 频率维度的因果填充（右对齐）
                    inp = torch.nn.functional.pad(inp, pad = [(self.F_size - 1)*self.F_dilation-self.F_pad, (self.F_size - 1)*self.F_dilation-self.F_pad - left_pad, 0, 0])
                else:
                    # 未实现的边界情况
                    raise(NotImplementedError)
            else:
                raise(NotImplementedError)

        else: # F_stride = 1（无需上采样）
            # 频率维度对称padding
            inp = torch.nn.functional.pad(inp, pad=[(self.F_size-1)*self.F_dilation-self.F_pad, (self.F_size-1)*self.F_dilation-self.F_pad])

        outp = self.StreamConvTranspose2d(inp)

        return outp, out_cache


if __name__ == '__main__':
    from convert import convert_to_stream

    # ========== 测试1: StreamConv1d 正确性验证 ==========
    Sconv = StreamConv1d(1, 1, 3)
    Conv = nn.Conv1d(1, 1, 3)
    convert_to_stream(Sconv, Conv)

    test_input = torch.randn([1, 1, 10])
    with torch.no_grad():
        ## 非流式（参考输出）
        test_out1 = Conv(torch.nn.functional.pad(test_input, [2,0]))  # 左侧padding2

        ## 流式（逐帧处理）
        cache = torch.zeros([1, 1, 2])  # 初始缓存=2帧
        test_out2 = []
        for i in range(10):
            out, cache = Sconv(test_input[..., i:i+1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=-1)
        print(">>> StreamConv1d 误差:", (test_out1 - test_out2).abs().max())  # 应接近0

    # ========== 测试2: StreamConv2d 正确性验证 ==========
    Sconv = StreamConv2d(1, 1, [3,3])
    Conv = nn.Conv2d(1, 1, (3,3))
    convert_to_stream(Sconv, Conv)

    test_input = torch.randn([1,1,10,6])

    with torch.no_grad():
        ## 非流式（参考输出）
        test_out1 = Conv(torch.nn.functional.pad(test_input,[0,0,2,0]))

        ## 流式（逐帧处理）
        cache = torch.zeros([1,1,2,6])  # 时间维度缓存2帧
        test_out2 = []
        for i in range(10):
            out, cache = Sconv(test_input[:,:, i:i+1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=2)
        print(">>> StreamConv2d 误差:", (test_out1 - test_out2).abs().max())  # 应接近0


    # ========== 测试3: StreamConvTranspose2d 正确性验证 ==========
    kt = 3   # 时间维度的卷积核大小
    dt = 2   # 时间维度的膨胀率
    pt = (kt-1) * dt  # 时间维度的因果填充量
    # 创建普通转置卷积和对应的流式版本
    DeConv = torch.nn.ConvTranspose2d(4, 8, (kt,3), stride=(1,2), padding=(pt,1), dilation=(dt,2), groups=2)
    SDeconv = StreamConvTranspose2d(4, 8, (kt,3), stride=(1,2), padding=(0,1), dilation=(dt,2), groups=2)
    convert_to_stream(SDeconv, DeConv)

    test_input = torch.randn([1, 4, 100, 6])
    with torch.no_grad():
        ## 非流式（参考输出）
        test_out1 = DeConv(nn.functional.pad(test_input, [0,0,pt,0]))  # 因果padding

        ## 流式（逐帧处理）
        test_out2 = []
        cache = torch.zeros([1, 4, pt, 6])  # 时间维度缓存 pt 帧
        for i in range(100):
            out, cache = SDeconv(test_input[:,:, i:i+1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=2)

        print(">>> StreamConvTranspose2d 误差:", (test_out1 - test_out2).abs().max())  # 应接近0
