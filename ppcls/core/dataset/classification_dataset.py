import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np


class ClassificationDataset(Dataset):
    """
    创建一个分类网络的数据集, 创建完这个数据集是一个可迭代对象
    有 __getitem__就必然要写 __len__
    __getitem__的作用 是可以被for训练遍历，也可以用下表索引
    __len__的作用是 可以知道这个迭代器可以让for循环知道循环多少次结束
    """

    def __init__(self, data_path, transform=None):
        # 创建一个类别名称列表，记录每一个类别的名称
        self.class_names = []
        # 根据数据集路径获取每张图片对应的图片路径和对应的标签索引
        self.image_paths, self.image_labels = self.read_image_info(data_path)
        self.transform = transform

    def __getitem__(self, item):
        # 根据item的获取到对应的图片路径和对应的标签索引
        image_path = self.image_paths[item]
        label = self.image_labels[item]
        # 打开图片
        image = Image.open(image_path)
        # 如果有数据增强，则进行数据增强
        if self.transform is not None:
            image = self.transform(image)
        # 转化成numpy格式
        image = np.array(image)
        # 返回图片和索引
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def read_image_info(self, data_path):
        # 支持的图片格式
        IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
        image_paths = list()        # 存放图片的列表
        image_labels = list()       # 存放标签索引的列表
        categories_folder = os.listdir(data_path)   # 获取数据路径下，所有文件和文件夹的名字
        # 遍历一下获取到的文件夹和文件名称，保留是文件夹的名称
        categories_folder = [category for category in categories_folder if
                             os.path.isdir(os.path.join(data_path, category))]  #

        # 文件夹的名称就是对应类别的名称
        self.class_names = categories_folder

        # 遍历每个文件夹
        for cls_indx, category_folder in enumerate(categories_folder):
            category_path = os.path.join(data_path, category_folder)        # 得到文件夹路径
            images_name = os.listdir(category_path)                         # 获取到该文件夹下所有文件的名称
            for image_name in images_name:
                suffix = image_name.split('.')[-1].upper()                  # 获取该文件的后缀
                if suffix in IMAGE_FORMER:                                  # 如果后缀是一个图片格式
                    image_path = os.path.join(category_path, image_name)    # 获取图片路径
                    image_paths.append(image_path)                          # 添加到图片路径列表
                    image_labels.append(cls_indx)                           # 添加到索引列表
        return image_paths, image_labels


if __name__=="__main__":
    import json
    import cv2
    from ppcls.core.dataset.augmentation import Compose
    data_path = "E:/my_code/pp_cls/data/flower4/train"
    config_path = "E:/my_code/pp_cls/config/test.json"

    with open(config_path) as f:
        aug_config = json.load(f)

    comp = Compose(aug_config["augmentation"])
    train_dataset = ClassificationDataset(data_path, transform=comp)

    for images, labels in train_dataset:
        image = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", image)
        print(f"label {labels}")
        cv2.waitKey()

        pass




