import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, images_path, labels_path, lr_size, rate):
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(lr_size)
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((lr_size[0] * rate, lr_size[1] * rate))
            ]
        )
        self.images_path = images_path
        self.labels_path = labels_path
        self.images_name = sorted(os.listdir(images_path))
        self.rate = rate

    def __getitem__(self, index):
        imn = self.images_name[index % len(self.images_name)]
        image = Image.open(self.images_path + imn)
        label = Image.open(self.labels_path + imn.replace('x' + str(self.rate), ''))
        img_lr = self.lr_transform(image)
        img_hr = self.hr_transform(label)
        # return {"lr": img_lr, "hr": img_hr, "hs": np.array(img_hr).shape}
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.images_name)


def get_div2k_data(tip, tlp, vip, vlp, bs, lr_size, rate):
    return DataLoader(
        ImageDataset(tip, tlp, lr_size, rate),
        batch_size=bs, 
        shuffle=True, 
    ), DataLoader(
        ImageDataset(vip, vlp, lr_size, rate),
        batch_size=bs
    )

