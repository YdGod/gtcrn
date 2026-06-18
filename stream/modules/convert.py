"""
权重转换模块：将离线（非流式）模型的权重转换到流式模型

为什么需要转换？
  流式模型使用了 StreamConv2d、StreamConvTranspose2d 等自定义层，
  这些层在结构上与普通 Conv2d/ConvTranspose2d 不同，
  但底层的 Conv2d/Conv1d 权重可以复用。

转换策略：
  1. 同名参数直接复制
  2. StreamConv2d.weight → Conv2d.weight (去掉 "Stream" 前缀匹配)
  3. StreamConvTranspose2d.weight 需要特殊处理：
     - 流式转置卷积使用普通卷积实现，权重需要时间维度的翻转
     - 如果形状不匹配，还需要维度置换 [1,0,2,3]
"""
import torch


def convert_to_stream(stream_model, model):
    """
    将离线模型的权重复制到流式模型

    参数:
        stream_model: 流式模型（目标）
        model: 离线模型（源），提供预训练权重

    工作流程：
       遍历流式模型的所有参数名，在离线模型中查找对应参数并复制
    """
    state_dict = model.state_dict()       # 离线模型权重字典
    new_state_dict = stream_model.state_dict()  # 流式模型权重字典（目标）

    for key in stream_model.state_dict().keys():
        # 情况1：参数名完全匹配 → 直接复制
        if key in state_dict.keys():
            new_state_dict[key] = state_dict[key]

        # 情况2：键名包含 "StreamConv1d." 前缀 → 去掉前缀后匹配
        elif key.replace('StreamConv1d.', '') in state_dict.keys():
            new_state_dict[key] = state_dict[key.replace('StreamConv1d.', '')]

        # 情况3：键名包含 "StreamConv2d." 前缀 → 去掉前缀后匹配
        elif key.replace('StreamConv2d.', '') in state_dict.keys():
            new_state_dict[key] = state_dict[key.replace('StreamConv2d.', '')]

        # 情况4：StreamConvTranspose2d（用Conv2d实现转置卷积，需特殊处理）
        # For StreamConvTranspose2d Version 1:
        # elif key.replace('ConvTranspose2d.', '') in state_dict.keys():
        #     new_state_dict[key] = state_dict[key.replace('ConvTranspose2d.', '')]

        ## For StreamConvTranspose2d Version 2:
        elif key.replace('StreamConvTranspose2d.', '') in state_dict.keys():
            # 使用Conv2d实现转置卷积时，权重需要在时间维度上翻转
            if key.endswith('weight'):
                # 离线模型的ConvTranspose2d权重形状为 [in_ch, out_ch/groups, kT, kF]
                # 流式模型用Conv2d模拟，权重需要维度置换和翻转
                src_weight = state_dict[key.replace('StreamConvTranspose2d.', '')]
                tgt_shape = new_state_dict[key].shape  # [out_ch, in_ch/groups, kT, kF]
                if tgt_shape != src_weight.shape:
                    # 形状不匹配（含分组卷积的情况）：
                    # ConvTranspose2d: [in, out/g, kT, kF] → Conv2d: [out, in/g, kT, kF]
                    # 通过 reshape→permute→reshape 正确重排各分组的权重
                    in_ch, out_per_group, kT, kF = src_weight.shape
                    out_ch, in_per_group = tgt_shape[0], tgt_shape[1]
                    groups = out_ch // out_per_group  # 从形状推导分组数
                    new_state_dict[key] = torch.flip(
                        src_weight
                        .reshape(groups, in_ch // groups, out_per_group, kT, kF)
                        .permute(0, 2, 1, 3, 4)  # (groups, out/g, in/g, kT, kF)
                        .reshape(out_ch, in_per_group, kT, kF),
                        dims=[-2, -1]
                    )
                else:
                    # 形状匹配（groups=1）：直接翻转最后两维（时间+频率维度）
                    new_state_dict[key] = torch.flip(src_weight, dims=[-2,-1])

            else:
                # bias等参数直接复制，无需翻转
                new_state_dict[key] = state_dict[key.replace('StreamConvTranspose2d.', '')]


        else:
            # 无法匹配的参数名 → 报错
            raise(ValueError('键匹配错误! 找不到参数: ' + key))

    # 加载转换后的权重
    stream_model.load_state_dict(new_state_dict)
