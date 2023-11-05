import torch.nn as nn


class ModifiedVGG(nn.Module):
    def __init__(self, inc, mdc):
        super(ModifiedVGG, self).__init__()
        self.c_0_0 = nn.Conv2d(inc, mdc, 3, 1, 1, bias=True)
        self.c_0_1 = nn.Conv2d(mdc, mdc, 4, 2, 1, bias=False)
        self.b_0_1 = nn.BatchNorm2d(mdc, affine=True)
        self.c_1_0 = nn.Conv2d(mdc, mdc * 2, 3, 1, 1, bias=False)
        self.b_1_0 = nn.BatchNorm2d(mdc * 2, affine=True)
        self.c_1_1 = nn.Conv2d(mdc * 2, mdc * 2, 4, 2, 1, bias=False)
        self.b_1_1 = nn.BatchNorm2d(mdc * 2, affine=True)
        self.c_2_0 = nn.Conv2d(mdc * 2, mdc * 4, 3, 1, 1, bias=False)
        self.b_2_0 = nn.BatchNorm2d(mdc * 4, affine=True)
        self.c_2_1 = nn.Conv2d(mdc * 4, mdc * 4, 4, 2, 1, bias=False)
        self.b_2_1 = nn.BatchNorm2d(mdc * 4, affine=True)
        self.c_3_0 = nn.Conv2d(mdc * 4, mdc * 8, 3, 1, 1, bias=False)
        self.b_3_0 = nn.BatchNorm2d(mdc * 8, affine=True)
        self.c_3_1 = nn.Conv2d(mdc * 8, mdc * 8, 4, 2, 1, bias=False)
        self.b_3_1 = nn.BatchNorm2d(mdc * 8, affine=True)
        self.c_4_0 = nn.Conv2d(mdc * 8, mdc * 8, 3, 1, 1, bias=False)
        self.b_4_0 = nn.BatchNorm2d(mdc * 8, affine=True)
        self.c_4_1 = nn.Conv2d(mdc * 8, mdc * 8, 4, 2, 1, bias=False)
        self.b_4_1 = nn.BatchNorm2d(mdc * 8, affine=True)
        self.l_1 = nn.Linear(mdc * 8 * 4 * 4, 100)
        self.l_2 = nn.Linear(100, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=.2, inplace=True)

    def forward(self, x):
        feature = self.leaky_relu(self.c_0_0(x))
        feature = self.leaky_relu(self.b_0_1(self.c_0_1(feature)))
        feature = self.leaky_relu(self.b_1_0(self.c_1_0(feature)))
        feature = self.leaky_relu(self.b_1_1(self.c_1_1(feature)))
        feature = self.leaky_relu(self.b_2_0(self.c_2_0(feature)))
        feature = self.leaky_relu(self.b_2_1(self.c_2_1(feature)))
        feature = self.leaky_relu(self.b_3_0(self.c_3_0(feature)))
        feature = self.leaky_relu(self.b_3_1(self.c_3_1(feature)))
        feature = self.leaky_relu(self.b_4_0(self.c_4_0(feature)))
        feature = self.leaky_relu(self.b_4_1(self.c_4_1(feature)))
        feature = feature.view(feature.size(0), -1)
        feature = self.leaky_relu(self.l_1(feature))
        feature = self.l_2(feature)
        return feature



































