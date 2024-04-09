import argparse


def get_srgan_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch_start", type=int, default=0)
    parser.add_argument("--epoch_count", type=int, default=50)
    parser.add_argument('--init_count', type=int, default=10)

    parser.add_argument("--data_name", type=str, default="div2k")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--lr_size", type=tuple, default=(96, 96))
    parser.add_argument("--rate", type=int, default=4)

    parser.add_argument("--rb_count", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--sample_interval", type=int, default=200)
    parser.add_argument("--checkpoint_interval", type=int, default=-1)
    args = parser.parse_args()
    return args

