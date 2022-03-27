import torchvision.transforms.transforms as ts   # torchvision 的数据增强
from torch.utils.data import DataLoader          # 导入torch的数据集加载器
from PIL import Image                            # 图片读取
import numpy as np
import cv2
from ppcls.core.dataset.builder import AUGMENTATION, build_aug
from .classification_dataset import ClassificationDataset


class Compose:
    """
    将各种增强方法融合在一起的类
    """
    def __init__(self, transforms):
        """
        初始化类对象
        """
        # 创建一个增强方法的列表
        self.transforms = []

        # 循环导入配置文件的增强方式并创建增强方法
        for transform in transforms:
            if isinstance(transform, dict):
                op = build_aug(transform)   # 如果transform是一个字典，就创建字典中对应的增强方法
                self.transforms.append(op)  # 将该方法添加到列表中

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)        # 循环执行列表中各个增强方法
        return img


# 注册数据集增强方法
AUGMENTATION.registry(ts.Resize)
AUGMENTATION.registry(ts.Normalize)
AUGMENTATION.registry(ts.RandomRotation, "Rotate")
AUGMENTATION.registry(ts.RandomCrop, "Crop")
AUGMENTATION.registry(ts.RandomHorizontalFlip, "HorizontalFlip")
AUGMENTATION.registry(ts.ToTensor)


def build_loader(cfg):
    # 加载训练集和验证集的数据增强方式
    train_comp = Compose(cfg["augmentation"]["train"])
    val_comp = Compose(cfg["augmentation"]["val"])
    # 根据路径和增强方式创建训练集和验证集的数据集
    train_dataset = ClassificationDataset(cfg["train_data_path"], transform=train_comp)
    val_dataset = ClassificationDataset(cfg["val_data_path"], transform=val_comp)
    # 创建训练集加载器和验证集加载器，并返回
    train_data_loader = DataLoader(dataset=train_dataset,           # 选择数据集
                                   batch_size=cfg["batch_size"],    # 一批图像数据的数量
                                   shuffle=True,                    # 是否打乱顺序
                                   num_workers=cfg["num_workers"])  # 多进程处理图片，0代表不开启进程
    val_data_loader = DataLoader(dataset=val_dataset,
                                 batch_size=cfg["batch_size"])

    return train_data_loader, val_data_loader


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

