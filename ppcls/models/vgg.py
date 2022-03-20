from torchvision.models import vgg11, vgg16, vgg19
from .builder import MODEL

MODEL.registry(vgg11, "vgg11")
MODEL.registry(vgg16, "vgg16")
MODEL.registry(vgg19, "vgg19")
