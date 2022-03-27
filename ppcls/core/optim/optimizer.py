import torch
from .builder import OPTIM

# 注册优化器
OPTIM.registry(torch.optim.SGD, "SGD")
OPTIM.registry(torch.optim.AdamW, "AdamW")
OPTIM.registry(torch.optim.Adam, "Adam")
