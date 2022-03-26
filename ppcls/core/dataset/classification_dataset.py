import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np


class ClassificationDataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.class_names = []
        self.image_paths, self.image_labels = self.read_image_info(data_path)
        self.transform = transform

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.image_labels[item]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        image = np.array(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def read_image_info(self, data_path):
        IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
        image_paths = list()
        image_labels = list()
        categories_folder = os.listdir(data_path)
        categories_folder = [category for category in categories_folder if
                             os.path.isdir(os.path.join(data_path, category))]
        self.class_names = categories_folder
        for cls_indx, category_folder in enumerate(categories_folder):
            category_path = os.path.join(data_path, category_folder)
            images_name = os.listdir(category_path)
            for image_name in images_name:
                suffix = image_name.split('.')[-1].upper()
                if suffix in IMAGE_FORMER:
                    image_path = os.path.join(category_path, image_name)
                    image_paths.append(image_path)
                    image_labels.append(cls_indx)
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




