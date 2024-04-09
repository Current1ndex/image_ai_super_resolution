import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models.vgg import vgg19, VGG19_Weights

from configs import get_srgan_config, get_div2k_config
from gans.SRGAN import SRGAN_G, SRGAN_D
from load_data import get_div2k_data
from losses import LossInit, LossG, LossD

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dc = get_div2k_config()
nc = get_srgan_config()

net_G = SRGAN_G(nc.input_channel, nc.rb_count).to(device)
net_D = SRGAN_D().to(device)
net_vgg = vgg19(weights=VGG19_Weights.DEFAULT).features

train_dl, valid_dl = get_div2k_data(
    dc.data_path + dc.data_train_image_path,
    dc.data_path + dc.data_train_label_path,
    dc.data_path + dc.data_valid_image_path,
    dc.data_path + dc.data_valid_label_path,
    nc.batch_size,
    nc.lr_size,
    nc.rate
)

loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()

optim_I = optim.Adam(net_G.parameters(), lr=nc.lr)
optim_G = optim.Adam(net_G.parameters(), lr=nc.lr)
optim_D = optim.Adam(net_D.parameters(), lr=nc.lr)


def init():
    for _ in range(nc.init_count):
        for image in train_dl:
            hr = image['hr'].to(device)
            lr = image['lr'].to(device)
            optim_G.zero_grad()
            loss_i = LossInit(net_G, loss_mse)(lr, hr)
            loss_i.backward()
            optim_G.step()


def train():
    for epoch in range(nc.epoch_count):
        for image in train_dl:
            hr = image['hr'].to(device)
            lr = image['lr'].to(device)
            optim_G.zero_grad()
            loss_g = LossG(net_G, net_D, net_vgg, loss_ce, loss_mse)(lr, hr)
            loss_g.backward()
            optim_G.step()
            optim_D.zero_grad()
            loss_d = LossD(net_G, net_D, loss_ce)(lr, hr)
            loss_d.backward()
            optim_D.step()
            print('[Epoch: %d]: [G-loss: %f] [D-loss: %f]' % (epoch, loss_g, loss_d))


# init()
train()

