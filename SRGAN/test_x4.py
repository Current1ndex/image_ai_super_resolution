import os

import numpy as np
from PIL import Image
from tensorflow import keras


image_path = 'test/images/0801x4.png'

image = Image.open(image_path).convert('RGB')
image = np.array(image, np.float32) / 127.5 - 1
# 339 510 -> 1356 2040
sr = [[0, 110, 219], [0, 97, 195, 292, 390]]
st_h = [[110, 120], [219, 230]]
st_w = [[97, 120], [195, 217], [292, 315], [390, 412]]

model = keras.models.load_model('./saves/x4_42000/model.h5')


def split_image(img):
    il = []
    for h in sr[0]:
        for w in sr[1]:
            il.append(img[h: h+120, w: w+120, :])
    return np.array(il, np.float64)


def stack_image(iml, rs):
    for i, h in enumerate(sr[0]):
        for j, w in enumerate(sr[1]):
            rs[h * 4: h * 4 + 480, w * 4: w * 4 + 480, :] += iml[i * len(sr[1]) + j]
    for w in st_w:
        rs[:, w[0] * 4: w[1] * 4, :] /= 2.
    for h in st_h:
        rs[h[0] * 4: h[1] * 4, :, :] /= 2.
    return rs


result = np.zeros((image.shape[0] * 4, image.shape[1] * 4, image.shape[2]))
image_split_list = split_image(image)

image = Image.fromarray(np.uint8((image_split_list[0] + 1) * 127.5))
image.save("test/x4/output2_x4.png")

image_split_list = model(image_split_list)

image = Image.fromarray(np.uint8((image_split_list[0] + 1) * 127.5))
image.save("test/x4/output1_x4.png")

result = stack_image(image_split_list, result)
image = Image.fromarray(np.uint8((result + 1) * 127.5))
image.save("test/x4/output_x4.png")
