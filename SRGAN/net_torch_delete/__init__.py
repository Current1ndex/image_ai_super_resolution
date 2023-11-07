# Copyright (c) OpenMMLab. All rights reserved.
from .ModifiedVGG import ModifiedVGG
from .sr_resnet import MSRResNet
from .srgan import SRGAN

__all__ = [
    'ModifiedVGG',
    'MSRResNet',
    'SRGAN',
]
