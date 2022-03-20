from ppcls.core.utils import Registry, build_module

OPTIM = Registry("augmentation")


def build_optimzer(cfg):
    return build_module(cfg, OPTIM)
