import torch.nn as nn

from ResidualBlock import ResidualBlock
from PixelShufflePack import PixelShufflePack
from utils import make_layers

from utils import default_init_weights


class ModifiedResNet():
    def __init__(self, inc, ouc, mdc=64, block_count=16, upscale_factor=4):
        super(ModifiedResNet, self).__init__()
        self.inc = inc
        self.ouc = ouc
        self.mdc = mdc
        self.bct = block_count
        self.usf = upscale_factor
        self.c_f = nn.Conv2d(inc, mdc, 3, 1, 1, bias=True)
        self.rbs = make_layers(ResidualBlock, self.bct, mdc=self.mdc)
        # self.usf in [2, 3, 4]
        if self.usf in [2, 3]:
            self.us = PixelShufflePack(mdc, mdc, self.usf, upsample_kernel=3)
        elif self.usf == 4:
            self.us_1 = PixelShufflePack(mdc, mdc, 2, upsample_kernel=3)
            self.us_2 = PixelShufflePack(mdc, mdc, 2, upsample_kernel=3)
        else:
            pass
        self.c_h = nn.Conv2d(mdc, mdc, 3, 1, 1, bias=True)
        self.c_l = nn.Conv2d(mdc, ouc, 3, 1, 1, bias=True)
        self.nus = nn.Upsample(scale_factor=self.usf, mode='bilinear', align_corners=False)
        self.l_relu = nn.LeakyReLU(negative_slope=.1, inplace=True)
        self.init_weights()

    def init_weights(self):
        for m in [self.c_f, self.c_h, self.c_l]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        out = self.l_relu(self.c_f(x))
        out = self.rbs(out)
        if self.usf in [2, 3]:
            out = self.us(out)
        elif self.usf == 4:
            out = self.us_1(out)
            out = self.us_2(out)
        else:
            pass
        out = self.c_l(self.l_relu(self.c_h(out)))
        usd_out = self.nus(out)
        return out + usd_out









