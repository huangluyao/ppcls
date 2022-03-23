from ppcls.core.utils import Registry, build_module

OPTIM = Registry("optimizer")


def build_optimizer(cfg, model_params):
    default_params = {"params": model_params}
    return build_module(cfg, OPTIM, default_params)
