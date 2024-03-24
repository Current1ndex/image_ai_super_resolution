import os
import sys

sys.path.append('')

import numpy as np
from PIL import Image

from configs import get_div2k_config


config = get_div2k_config()
train_image_path = config.data_path + config.data_train_image_path

t = np.array([])
for index, image_name in enumerate(os.listdir(train_image_path)):
    image = np.array(Image.open(train_image_path + image_name)) / 255.
    image = np.reshape(image, (-1, 3))
    if index == 0:
        t = image
    else:
        t = np.concatenate([t, image], axis=0)
        
print(np.mean(t, axis=0))
print(np.std(t, axis=0))

