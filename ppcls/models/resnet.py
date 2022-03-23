from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from .builder import MODEL

MODEL.registry(resnet18, "resnet18")
MODEL.registry(resnet34, "resnet34")
MODEL.registry(resnet50, "resnet50")
MODEL.registry(resnet101, "resnet101")
MODEL.registry(resnet152, "resnet152")
