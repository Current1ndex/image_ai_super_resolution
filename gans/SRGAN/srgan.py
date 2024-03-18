import torch
import torch.nn as nn
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            *list(
                vgg19(weights='IMAGENET1K_V1').features.children()
            )[:18]
        )

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, inc):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.BatchNorm2d(inc, 0.8),
            nn.PReLU(),
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.BatchNorm2d(inc, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, inc=3, onc=3, rn=16):
        super(GeneratorResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc, 64, 9, 1, 4), 
            nn.PReLU()
        )
        res_blocks = []
        for _ in range(rn):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), 
            nn.BatchNorm2d(64, 0.8)
        )
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, onc, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        inc, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
        def discriminator_block(inc, ouc, first_block=False):
            layers = []
            layers.append(nn.Conv2d(inc, ouc, 3, 1, 1))
            if not first_block:
                layers.append(nn.BatchNorm2d(ouc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(ouc, ouc, 3, 2, 1))
            layers.append(nn.BatchNorm2d(ouc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        layers = []
        for i, ouc in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(inc, ouc, first_block=(i == 0)))
            inc = ouc
        layers.append(nn.Conv2d(ouc, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
