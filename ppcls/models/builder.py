from ppcls.core.utils import Registry, build_module

MODEL = Registry("model")


def build_model(cfg):
    return build_module(cfg, MODEL)

