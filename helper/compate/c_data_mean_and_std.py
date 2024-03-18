import os
import sys

sys.path.append('')

import PIL

from configs.DIV2K_config import get_config


config = get_config()
train_image_path = config.data_path + config.data_train_image_path
train_label_path = config.data_path + config.data_train_label_path + 'x' + str(config.re_rate) + '/'
train_image_path = config.data_path + config.data_train_image_path
train_label_path = config.data_path + config.data_train_label_path + 'x' + str(config.re_rate) + '/'

