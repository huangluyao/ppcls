import torchvision.transforms.transforms as ts   # torchvision 的数据增强
from PIL import Image                            # 图片读取
import numpy as np
import cv2
from ppcls.core.dataset.builder import AUGMENTATION, build_aug


class Compose:
    def __init__(self, transforms):
        """
        初始化类对象
        """
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                op = build_aug(transform)
                self.transforms.append(op)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


AUGMENTATION.registry(ts.Resize)
AUGMENTATION.registry(ts.Normalize)
AUGMENTATION.registry(ts.RandomRotation, "Rotate")
AUGMENTATION.registry(ts.RandomCrop, "Crop")
AUGMENTATION.registry(ts.RandomHorizontalFlip, "HorizontalFlip")
AUGMENTATION.registry(ts.ToTensor)


if __name__ == "__main__":
    import json
    image_path = "E:\my_code\pp_cls\data\\flower4\\train\huanghuacai\image_0565.jpg"
    img = Image.open(image_path)
    config_path = "E:\my_code\pp_cls\config\\test.json"

    with open(config_path) as f:
        aug_config = json.load(f)

    comp = Compose(aug_config["augmentation"])
    img = comp(img)

    cv2.imshow("result", np.array(img))
    cv2.waitKey()

