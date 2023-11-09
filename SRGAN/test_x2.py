import os

import numpy as np
from PIL import Image
from tensorflow import keras

image_path = 'test/images/0802x2.png'

image = Image.open(image_path).convert('RGB')
image = np.array(image, np.float32) / 127.5 - 1

model = keras.models.load_model('./saves/x2_46000/model.h5')

# (678, 1020, 3) (1356, 2040, 3)
sr = [[0, 219, 438], [0, 195, 390, 585, 780]]
st_h = [[219, 240], [438, 459]]
st_w = [[195, 240], [390, 435], [585, 630], [780, 825]]


def split_image(img):
    il = []
    for h in sr[0]:
        for w in sr[1]:
            il.append(img[h: h+240, w: w+240, :])
    return np.array(il, np.float64)


def stack_image(iml, rs):
    for i, h in enumerate(sr[0]):
        for j, w in enumerate(sr[1]):
            rs[h * 2: h * 2 + 480, w * 2: w * 2 + 480, :] += iml[i * len(sr[1]) + j]
    for w in st_w:
        rs[:, w[0] * 2: w[1] * 2, :] /= 2.
    for h in st_h:
        rs[h[0] * 2: h[1] * 2, :, :] /= 2.
    return rs


result = np.zeros((image.shape[0] * 2, image.shape[1] * 2, image.shape[2]))
# if index == 0: size => (678, 1020)
# last_index_start [438, 780] => [[0, 219, 438], [0, 195, 390, 585, 780]]
# {h: 20, w: 44}
image_split_list = split_image(image)

image = Image.fromarray(np.uint8((image_split_list[0] + 1) * 127.5))
image.save("test/x2/output2_x2.png")

image_split_list = model(image_split_list)

image = Image.fromarray(np.uint8((image_split_list[0] + 1) * 127.5))
image.save("test/x2/output1_x2.png")

result = stack_image(image_split_list, result)
image = Image.fromarray(np.uint8((result + 1) * 127.5))
image.save("test/x2/output_x2.png")

print(result)

