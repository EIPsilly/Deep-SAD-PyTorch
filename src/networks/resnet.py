from torchvision import models

from base.base_net import BaseNet

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchsummary import summary

from .res_encoder import ResNet as resnet_encoder
from .res_decoder import ResNet as resnet_decoder
from .res_encoder import Bottleneck as Bottleneck_encoder
from .res_decoder import Bottleneck as Bottleneck_decoder

class Resnet_Autoencoder(BaseNet):
    
    def __init__(self):
        super().__init__()
        self.encoder = resnet_encoder(Bottleneck_encoder, [3, 4, 6, 3], return_indices=True, width_per_group = 64 * 2)
        self.decoder = resnet_decoder(Bottleneck_decoder, [3, 6, 4, 3], width_per_group = 64 * 2)
    
    def forward(self, x):
        x, idx = self.encoder(x)
        x = self.decoder(x, idx)
        return x


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_net = Resnet_Autoencoder().to(device)
    test_input = torch.rand(2, 3, 253, 253).to(device)
    out = ae_net(test_input)
    out
    summary(ae_net.encoder, (3, 253, 253))
    summary(ae_net.decoder, [[1000]])