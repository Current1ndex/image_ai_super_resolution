import torch
import torch.nn as nn


class LossInit(nn.Module):
    def __init__(self, net_G, loss_fn) -> None:
        super().__init__()
        self.g = net_G
        self.f = loss_fn

    def forward(self, lr, hr):
        return self.f(self.g(lr), hr)


class LossG(nn.Module):
    def __init__(self, net_G, net_D, vgg, loss_fn1, loss_fn2) -> None:
        super().__init__()
        self.g = net_G
        self.d = net_D
        self.v = vgg
        self.f1 = loss_fn1
        self.f2 = loss_fn2
    
    def forward(self, lr, hr):
        fake = self.g(lr)
        fake_r = self.d(fake)
        fake_f = self.vgg((fake + 1) / 2.)
        rely_f = self.vgg((hr + 1) / 2.)
        loss_1 = 1e-3 * self.f1(fake_r, torch.ones_like(fake_r))
        loss_1 = torch.mean(loss_1)
        loss_2 = self.f2(fake, hr)
        loss_3 = 2e-6 * self.f2(fake_f, rely_f)
        return loss_1 + loss_2 + loss_3



class LossD(nn.Module):
    def __init__(self, net_G, net_D, loss_fn) -> None:
        super().__init__()
        self.g = net_G
        self.d = net_D
        self.f = loss_fn

    def forward(self, lr, hr):
        fake = self.g(lr)
        fake_r = self.d(fake)
        rely_r = self.d(hr)
        loss_1 = self.f(rely_r, torch.ones_like(rely_r))
        loss_1 = torch.mean(loss_1)
        loss_2 = self.f(fake_r, torch.zeros_like(fake_r))
        loss_2 = torch.mean(loss_2)
        return loss_1 + loss_2


