class Registry:
    """
    注册器，里面存放着字典，用于记录传入的模块
    模块可以是一个类或者一个函数
    """
    def __init__(self, name=None):
        self._module_dict = {}      # 一个字典，用于存放注册到里面的模块， 模块可以是一个类或者一个函数
        self.name = name            # 这个注册器的名字

    def registry(self, module, name=None):
        if name is None:
            self._module_dict[module.__name__] = module # 如果名字为空，字典保存这个模块的名字和这个模块
        else:
            self._module_dict[name] = module    # 字典保存传入的名字和这个模块

    def get(self, key):
        # 通过传入的模块名字获取到这个模块
        return self._module_dict.get(key, None)


def build_module(cfg: dict,
                 register: Registry,
                 default_params: dict = None,
                 ):

    args = cfg.copy() # 防止cfg中的type 这个key消失
    obj_type = args.pop("type", None)   # 删除并获取args中的type对应的模块名称
    if obj_type is None:
        raise KeyError(f"cfg must contain the key 'type'")

    if isinstance(obj_type, str):
        obj = register.get(obj_type) # 获取类对象
        if obj is None:
            raise KeyError(f"{obj} is not in the {register.name} registry")

        if default_params is not None:
            args.update(default_params)     # 如果有其他参数要传入，则更新一下要传入的参数
            pass
        return obj(**args)      # 实例化对象
    else:
        raise KeyError(f"type must be string but got {type(obj_type)}")


if __name__ == "__main__":
    import torchvision.transforms.transforms as ts
    a = Registry()
    a.registry(ts.Normalize)

    print(a.get("Normalize"))
    pass
