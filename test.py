import torch
from torchvision.models import vgg19


vgg19_model = vgg19(weights='IMAGENET1K_V1')
print(vgg19_model)
print(list(vgg19_model.features.children())[:18])
