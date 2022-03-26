from ppcls.core.utils import Registry, build_module
AUGMENTATION = Registry("augmentation")


def build_aug(cfg):
    return build_module(cfg, AUGMENTATION)


