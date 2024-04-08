import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super(ResidualBlock, self).__init__()
        self.cb = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d()
        )
        for module in self.cb:
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module, std=.02)

    def forward(self, x):
        return self.cb(x) + x


class SRGAN_G(nn.Module):
    def __init__(self, inc, n) -> None:
        super(SRGAN_G, self).__init__()
        self.c1 = nn.Conv2d(inc, 64, 3, 1, 1)
        self.r1 = nn.ReLU()
        rb = []
        for _ in range(n):
            rb.append(ResidualBlock())
        self.rb = nn.Sequential(*rb)
        self.c2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b1 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 256, 3, 1, 1)
        # SubpixelConv2d = 卷积后减小通道数量，增加图像尺寸
        # 或者直接使用 UpSampling 层
        self.r2 = nn.ReLU()
        self.c4 = nn.Conv2d(64, 256, 3, 1, 1)
        self.r3 = nn.ReLU()
        self.c5 = nn.Conv2d(64, 3, 1, 1)
        self.th = nn.Tanh()
        for module in self.modules:
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module, std=.02)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.trunc_normal_(module, mean=1., std=.02)

    def forward(self, x):
        x = self.r1(self.c1(x))
        t = x
        x = self.rb(x)
        x = self.b1(self.c2(x))
        x = x + t
        x = self.c3(x)
        _, c, h, w = x.size()
        x = torch.reshape(x, (c // 4, h * 2, w * 2))
        x = self.r2(x)
        x = self.c4(x)
        _, c, h, w = x.size()
        x = torch.reshape(x, (c // 4, h * 2, w * 2))
        x = self.r3(x)
        x = self.c5(x)
        y = self.th(x)
        return y


class SRGAN_D(nn.Module):
    def __init__(self) -> None:
        super(SRGAN_D, self).__init__()
        self.sample_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 2048, 3, 2, 1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
            nn.Conv2d(2048, 1024, 1, 1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
            nn.Conv2d(2048, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 512, 3, 1, 1),
            nn.BatchNorm2d(512)
        )
        self.ft = nn.Flatten()
        self.l1 = nn.Linear()

    def forward(self, x):
        return self.l1(self.ft(self.sample_block(x) + x))


