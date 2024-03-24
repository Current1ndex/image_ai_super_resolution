import os
import numpy as np
import sys

from torchvision.utils import save_image, make_grid

from torch.autograd import Variable

from configs import *

from gans.SRGAN import *
from load_data import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

config = get_srgan_config()

if config.data_name == 'div2k':
    data_config = get_div2k_config()
else:
    data_config = None

tip = data_config.data_path + data_config.data_train_image_path
tlp = data_config.data_path + data_config.data_train_label_path

dataloader = get_div2k_data(tip, tlp, config.batch_size, True)

generator = GeneratorResNet()
discriminator = Discriminator()
feature_extractor = FeatureExtractor()
feature_extractor.eval()

criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

generator = generator.cuda()
discriminator = discriminator.cuda()
feature_extractor = feature_extractor.cuda()
criterion_GAN = criterion_GAN.cuda()
criterion_content = criterion_content.cuda()

if config.epoch_start != 0:
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.adam_b1, config.adam_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.adam_b1, config.adam_b2))


for epoch in range(config.epoch_start, config.epoch_count):
    for i, imgs in enumerate(dataloader):

        imgs_lr = Variable(imgs["lr"].type(torch.cuda.FloatTensor))
        imgs_hr = Variable(imgs["hr"].type(torch.cuda.FloatTensor))
        c, h, w = imgs['hs']
        h = int(h / 2 ** 4)
        w = int(w / 2 ** 4)
        valid = Variable(torch.cuda.FloatTensor(np.ones((imgs_lr.size(0), 1, h, w))), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(np.zeros((imgs_lr.size(0), 1, h, w))), requires_grad=False)

        optimizer_G.zero_grad()
        gen_hr = generator(imgs_lr)
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, config.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % config.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if config.checkpoint_interval != -1 and epoch % config.checkpoint_interval == 0:
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)