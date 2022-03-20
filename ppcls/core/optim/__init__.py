from .builder import OPTIM
from torch import optim

OPTIM.registry(optim.SGD)
OPTIM.registry(optim.AdamW)
OPTIM.registry(optim.Adam)

