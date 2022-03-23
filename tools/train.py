from ppcls.models import build_model
from ppcls.core import build_optimizer
import json


if __name__=="__main__":

    # model_cfg = {"type": "mobilenet_v2", "num_classes": 4}
    config_path = "config/test.json"

    with open(config_path, "r") as f:
        cfg = json.load(f)
    model_cfg = cfg["model"]
    model = build_model(model_cfg)
    optim = build_optimizer(cfg["optimizer"], model.parameters())
    print(model)


