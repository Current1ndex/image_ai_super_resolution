import argparse


def get_srgan_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch_start", type=int, default=0)
    parser.add_argument("--epoch_count", type=int, default=10)

    parser.add_argument("--data_name", type=str, default="div2k")
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--adam_b1", type=float, default=0.5)
    parser.add_argument("--adam_b2", type=float, default=0.999)

    parser.add_argument("--sample_interval", type=int, default=200)
    parser.add_argument("--checkpoint_interval", type=int, default=-1)
    args = parser.parse_args()
    return args

