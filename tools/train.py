# 训练的主函数
from ppcls.core import setup_logger, SimpleTrainer
import json


if __name__=="__main__":
    # 创建一个logger用于记录训练日志
    logger = setup_logger("log.txt")

    # 解析配置文件
    config_path = "config/test.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # 创建一个训练器并训练
    trainer = SimpleTrainer(cfg, logger)
    trainer.train()
