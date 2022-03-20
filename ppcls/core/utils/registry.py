
class Registry:
    def __init__(self, name=None):
        self._module_dict = {}
        self.name = name

    def registry(self, module, name=None):
        if name is None:
            self._module_dict[module.__name__] = module
        else:
            self._module_dict[name] = module

    def get(self, key):
        return self._module_dict.get(key, None)


def build_module(cfg: dict,
                 register: Registry
                 ):

    args = cfg.copy() # 防止cfg中的type 这个key消失
    obj_type = args.pop("type", None)
    if obj_type is None:
        raise KeyError(f"cfg must contain the key 'type'")

    if isinstance(obj_type, str):
        obj = register.get(obj_type) # 获取类对象
        if obj is None:
            raise KeyError(f"{obj} is not in the {register.name} registry")
        return obj(**args)      # 实例化对象
    else:
        raise KeyError(f"type must be string but got {type(obj_type)}")


if __name__ == "__main__":
    import torchvision.transforms.transforms as ts
    a = Registry()
    a.registry(ts.Normalize)

    print(a.get("Normalize"))
    pass
