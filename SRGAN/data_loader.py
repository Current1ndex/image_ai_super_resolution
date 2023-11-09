import os

import numpy as np
from PIL import Image

dir_path = 'F:/DIV2K/'
data_path_x1 = 'DIV2K_train_HR_sub/'
data_path_x2 = 'DIV2K_train_LR_bicubic/X2_sub/'
data_path_x4 = 'DIV2K_train_LR_bicubic/X4_sub/'

active_data_path = data_path_x4

"""
x1 -> 0001_s001.png -> 0800_s040.png
x2 -> 0001_s001.png -> 0800_s040.png
"""
imn_l = os.listdir(dir_path + data_path_x1)
np.random.shuffle(imn_l)
max_count = 32592


def load_image(img_path):
    return Image.open(img_path).convert('RGB')


def load_data(batch_size=1, epoch=0):
    epoch = epoch % (32592 - batch_size)
    ims_lr = []
    ims_hr = []
    for i in range(epoch, epoch + batch_size):
        ims_lr.append(load_image(dir_path + active_data_path + imn_l[i]))
        ims_hr.append(load_image(dir_path + data_path_x1 + imn_l[i]))
    ims_lr = np.array(ims_lr, np.float64) / 127.5 - 1
    ims_hr = np.array(ims_hr, np.float64) / 127.5 - 1
    return ims_hr, ims_lr












