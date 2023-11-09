import os

from PIL import Image

dir_path = 'F:/DIV2K/'
dir_x1_path = dir_path + 'DIV2K_train_HR/'
dir_x2_path = dir_path + 'DIV2K_train_LR_bicubic/X2/'
dir_x3_path = dir_path + 'DIV2K_train_LR_bicubic/X3/'
dir_x4_path = dir_path + 'DIV2K_train_LR_bicubic/X4/'

imn_l = os.listdir(dir_x1_path)


def test_x_rate_prefect_match():
    for imn in imn_l:
        img_x1 = Image.open(dir_x1_path + imn)
        img_x2 = Image.open(dir_x2_path + imn[:-4] + 'x2.png')
        img_x3 = Image.open(dir_x3_path + imn[:-4] + 'x3.png')
        img_x4 = Image.open(dir_x4_path + imn[:-4] + 'x4.png')
        if img_x1.size[0] != img_x2.size[0] * 2 or img_x1.size[1] != img_x2.size[1] * 2:
            print("2 ->", imn[:-4])
        if img_x1.size[0] != img_x3.size[0] * 3 or img_x1.size[1] != img_x3.size[1] * 3:
            print("3 ->", imn[:-4])
        if img_x1.size[0] != img_x4.size[0] * 4 or img_x1.size[1] != img_x4.size[1] * 4:
            print("4 ->", imn[:-4])


# 实际上不应该去寻找公约数，输入图像的尺寸是不应该规定的
def test_size(n=10):
    max_img_h = 0
    max_img_w = 0
    for imn in imn_l:
        img_x3 = Image.open(dir_x3_path + imn[:-4] + 'x3.png')
        img_x4 = Image.open(dir_x4_path + imn[:-4] + 'x4.png')
        if max_img_h < img_x4.size[0]:
            max_img_h = img_x4.size[0]
        if max_img_w < img_x4.size[1]:
            max_img_w = img_x4.size[1]
        # if img_x3.size[0] % n != 0 or img_x3.size[1] % n != 0:
        #     print("3 ->", img_x3.size)
        # if img_x4.size[0] % n != 0 or img_x4.size[1] % n != 0:
        #     print("4 ->", img_x4.size)
    print(max_img_h, max_img_w)


test_size()




