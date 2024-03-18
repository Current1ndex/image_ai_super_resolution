import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_owner', type=str, default='DIV2K dataset config.')

    parser.add_argument('--data_path', type=str, default='F:/DIV2K/DIV2K/')
    parser.add_argument('--data_train_image_path', type=str, default='DIV2K_train_HR/')
    parser.add_argument('--data_train_label_path', type=str, default='DIV2K_train_LR_bicubic/')
    parser.add_argument('--data_valid_image_path', type=str, default='DIV2K_valid_HR/')
    parser.add_argument('--data_valid_label_path', type=str, default='DIV2K_valid_LR_bicubic/')

    parser.add_argument('--re_rate', type=int, default=2)

    parser.add_argument('--mean', type=list, default=[])
    parser.add_argument('--std', type=list, default=[])

    args = parser.parse_args()
    return args

