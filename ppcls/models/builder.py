import torch
from ppcls.core.utils import Registry, build_module

# 创建网络注册器
MODEL = Registry("model")


def build_model(cfg):

    pretrained = cfg.pop("pretrained", None)
    model =  build_module(cfg, MODEL)
    if pretrained is not None:
        state_dict = torch.load(pretrained)     # 读取网络模型预训练参数
        model_state_dict = model.state_dict()   # 获取网络模型参数

        # 将预训练网络模型参数循环赋值给网络模型参数
        for key in model_state_dict.keys():
            if key in state_dict and model_state_dict[key].shape == state_dict[key].shape:
                model_state_dict[key] = state_dict[key]     # 如果网络模型参数和预训练网络参数形状一致，则赋值
        model.load_state_dict(model_state_dict)

    return model


