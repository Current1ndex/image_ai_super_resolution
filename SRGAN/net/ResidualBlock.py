import torch.nn as nn

from utils import default_init_weights


class ResidualBlock(nn.Module):
    def __init__(self, mdc=64, rss=1.):
        super(ResidualBlock, self).__init__()
        self.c_1 = nn.Conv2d(mdc, mdc, 3, 1, 1, bias=True)
        self.c_2 = nn.Conv2d(mdc, mdc, 3, 1, 1, bias=True)
        self.r = nn.ReLU(inplace=True)
        if rss == 1.:
            self.init_weights()

    def init_weights(self):
        for ms in [self.c_1, self.c_2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        out = self.c_2(self.r(self.c_1(x)))
        return x + out * self.rss



