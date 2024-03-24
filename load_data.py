import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



class ImageDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.images_path = images_path
        self.labels_path = labels_path
        self.images_name = sorted(os.listdir(images_path))

    def __getitem__(self, index):
        imn = self.images_name[index % len(self.images_name)]
        image = Image.open(self.images_path + imn)
        label = Image.open(self.labels_path + imn.replace('x2', ''))
        img_lr = self.lr_transform(image)
        img_hr = self.hr_transform(label)
        return {"lr": img_lr, "hr": img_hr, "hs": np.array(img_hr).shape}

    def __len__(self):
        return len(self.images_name)


def get_div2k_data(tip, tlp, bs, sh):
    dataloader = DataLoader(
        ImageDataset(tip, tlp),
        batch_size=bs, 
        shuffle=sh, 
    )
    return dataloader

