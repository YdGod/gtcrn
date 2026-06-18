"""
GTCRN 混合损失函数 (Hybrid Loss)

结合了三种损失来训练语音增强模型：

1. 幅度压缩的MSE损失 (Magnitude Loss)
   - 对幅度谱取0.3次方进行压缩，平衡大值和小值的影响
   - 小值（安静部分）不会被大值（响亮部分）掩盖

2. 压缩复数MSE损失 (Compressed Complex Loss)
   - 将实部/虚部除以幅度^0.7进行幅度压缩
   - 保留相位信息的同时平衡各频率贡献

3. SI-SNR 损失 (Scale-Invariant Signal-to-Noise Ratio)
   - 尺度不变的信噪比，对整体音量变化不敏感
   - 衡量增强语音与干净语音的相似度

总损失 = 30*(实部损失+虚部损失) + 70*幅度损失 + SI-SNR损失

参考论文: A Hybrid Loss for Speech Enhancement
"""
import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """混合损失函数"""
    def __init__(self):
        super().__init__()

    def forward(self, pred_stft, true_stft):
        """
        计算混合损失

        参数:
            pred_stft: (B, F, T, 2) 预测的复数频谱
            true_stft: (B, F, T, 2) 干净的复数频谱

        返回:
            loss: 标量损失值
        """
        device = pred_stft.device

        # ---- 分离实部和虚部 ----
        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]

        # ---- 计算幅度谱 ----
        # 加1e-12防止除零
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)

        # ---- 压缩复数损失 ----
        # 除以幅度^0.7进行幅度归一化，平衡不同能量的时频点
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = nn.MSELoss()(pred_real_c, true_real_c)  # 实部MSE
        imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)  # 虚部MSE

        # ---- 幅度损失 ----
        # 取0.3次方压缩幅度范围，防止大值主导损失
        mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))

        # ---- SI-SNR损失 ----
        # 将频谱通过iSTFT转回时域
        y_pred = torch.istft(pred_stft_real+1j*pred_stft_imag, 512, 256, 512,
                             window=torch.hann_window(512).pow(0.5).to(device))
        y_true = torch.istft(true_stft_real+1j*true_stft_imag, 512, 256, 512,
                             window=torch.hann_window(512).pow(0.5).to(device))

        # 将预测信号投影到目标信号方向（尺度不变处理）
        # s_target = (y_true·y_pred) / ||y_true||^2 * y_true
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / \
                 (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-8)

        # SI-SNR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )
        # 取负号因为要最小化损失，而SI-SNR越大越好
        sisnr = - torch.log10(
            torch.norm(y_true, dim=-1, keepdim=True)**2 /
            (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2 + 1e-8) + 1e-8
        ).mean()

        # 加权求和：复数损失权重30+30，幅度损失权重70，SI-SNR权重1
        return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr


if __name__ == "__main__":
    # 测试损失函数
    loss_func = HybridLoss()

    # 随机生成预测和目标频谱
    pred_stft = torch.randn(1, 257, 63, 2)
    true_stft = torch.randn(1, 257, 63, 2)
    loss = loss_func(pred_stft, true_stft)
    print(f"混合损失值: {loss.item():.4f}")
