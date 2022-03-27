from torchvision.models import mobilenet_v2
from .builder import MODEL

# 注册mobilenet_v2
MODEL.registry(mobilenet_v2, "mobilenet_v2")
