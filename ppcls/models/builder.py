from ppcls.core.utils import Registry, build_module

# 创建网络注册器
MODEL = Registry("model")


def build_model(cfg):
    return build_module(cfg, MODEL)

