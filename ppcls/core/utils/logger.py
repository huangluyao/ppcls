import logging
import sys
from termcolor import colored


def setup_logger(file_name, name="ppcls"):
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 创建打印输出格式
    stream_formatter = logging.Formatter(
        colored("[%(name)s %(asctime)s %(levelname)s %(filename)s:%(lineno)d]:", "green") + " %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )
    # 创建文本输出格式
    file_formatter = logging.Formatter(
        "[%(name)s %(asctime)s %(levelname)s %(filename)s:%(lineno)d]: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )
    # 添加打印处理器
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    # 添加文本处理器
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger


if __name__=="__main__":
    logger = setup_logger("log.txt")
    logger.info("setup")
    logger.warning("hello")
    logger.error("error")

