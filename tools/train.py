# 训练的主函数
import time, os
from ppcls.core import setup_logger, SimpleTrainer
import json



def create_workdir():
    # 获取当前时间，作为文件夹名称
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_dir = os.path.join("export", cur_time)

    # 创建文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 返回文件夹名称
    return output_dir

if __name__=="__main__":

    # 解析配置文件
    config_path = "config/test.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # 创建一个工作目录，用于保存训练的模型参数以及其他训练相关的文件
    cfg["output_dir"] = create_workdir()

    # 创建一个logger用于记录训练日志
    logger = setup_logger(os.path.join(cfg["output_dir"],"log.txt"))

    # 创建一个训练器并训练
    trainer = SimpleTrainer(cfg, logger)
    trainer.train()
