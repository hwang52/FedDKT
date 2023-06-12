import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

# 自定义数据集读取mini-imagenet
class MyDataSet(Dataset):
    def __init__(self, root_dir: str, csv_name: str, json_path: str, transform=None):
        super(MyDataSet, self).__init__()

        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i) for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)
        self.targets = csv_data["label"].values

        self.transfrom = transform
    
    def __len__(self):
        return self.total_num
    
    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB彩色, L灰度图
        if img.mode != 'RGB':
            raise ValueError("image {} is not RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transfrom is not None:
            img = self.transfrom(img)
        
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考:
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)  # 行方向进行拼接 [batch_size, dim]
        labels = torch.as_tensor(labels)

        return images, labels
