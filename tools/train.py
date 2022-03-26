from ppcls.core import setup_logger, SimpleTrainer
import json


if __name__=="__main__":

    logger = setup_logger("log.txt")
    config_path = "config/test.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    trainer = SimpleTrainer(cfg, logger)
    trainer.train()

    # model_cfg = cfg["model"]
    # logger.info("build model")
    # model = build_model(model_cfg)
    # logger.info("build optimizer")
    # optim = build_optimizer(cfg["optimizer"], model.parameters())


