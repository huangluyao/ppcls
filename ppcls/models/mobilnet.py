from torchvision.models import mobilenet_v2
from .builder import MODEL

MODEL.registry(mobilenet_v2, "mobilenet_v2")
