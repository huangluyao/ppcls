from ppcls.core.utils import Registry, build_module

OPTIM = Registry("optimizer")


def build_optimizer(cfg, model_params):
    default_params = {"params": model_params}       # 每个优化器都需要传入需要优化的网络参数
    return build_module(cfg, OPTIM, default_params)
