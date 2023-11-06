import torch.nn as nn
import torch.nn.functional as F

from utils import default_init_weights


class PixelShufflePack(nn.Module):
    def __init__(self, inc, ouc, scale_factor, upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.inc = inc
        self.ouc = ouc
        self.sf = scale_factor
        self.usk = upsample_kernel
        self.usc = nn.Conv2d(inc, ouc * self.sf * self.sf, self.usk, padding=(self.usk - 1) // 2)
        self.init_weights()

    def init_weights(self):
        default_init_weights(self, 1)

    def forward(self, x):
        x = self.usc(x)
        x = F.pixel_shuffle(x, self.sf)
        return x


