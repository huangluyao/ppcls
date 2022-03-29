# 递归， 函数他自己调用自己

def show_config_values(cfg, prefix="", logger=print):

    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            logger(f"{prefix}{key}")
            show_config_values(cfg[key], prefix="    "+prefix)
        elif isinstance(cfg[key], list):
            if len(cfg[key]) > 0 and isinstance(cfg[key][0], dict):
                for info_dict in cfg[key]:
                    logger(f"{prefix} \t {info_dict}")
            else:
                logger(f"{prefix}{key.ljust(20)}\t{cfg[key]}")
        else:
            logger(f"{prefix}{key.ljust(20)}\t{cfg[key]}")


if __name__ == "__main__":

    import json
    with open("D:\pycode\github\ppcls\config\\test.json") as f:
        cfg = json.load(f)

    show_config_values(cfg)