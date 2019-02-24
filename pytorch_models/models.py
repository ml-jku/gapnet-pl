import math

import torch
import torch.nn as nn
from pyll.base import TorchModel, calc_out_shape


class MCNN(TorchModel):
    def __init__(self, num_classes=13, input_shape=None):
        super(MCNN, self).__init__()
        assert input_shape
        in_c = input_shape[0]
        in_h = input_shape[1]
        in_w = input_shape[2]

        for i, (n, c) in enumerate(zip(range(0, 7), [16, 16, 16, 32, 32, 32, 64])):
            s = int(math.pow(2, n))  # downscaling factor before conv
            p = int(math.pow(2, 6 - n))  # downscaling factor after conv
            setattr(self, "scale{}".format(i), nn.Sequential(
                nn.MaxPool2d(kernel_size=s, stride=s),
                nn.Conv2d(in_c, out_channels=c, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, out_channels=c, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, out_channels=c, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=p, stride=p)
            ))

        self.combine = nn.Sequential(
            nn.Conv2d(in_channels=208, out_channels=1024, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # calc shape after multiscale encoder
        h, w = calc_out_shape(in_h, in_w, self.scale0)
        h, w = calc_out_shape(h, w, self.combine)
        # FC layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=h * w * 1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=num_classes)
        )

        # init
        self.init_parameters()

    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        scales = []
        for i in range(0, 7):
            scales.append(getattr(self, "scale{}".format(i))(x))
        x = torch.cat(scales, dim=1)
        x = self.combine(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def loss(self, prediction, target):
        bce = prediction.clamp(min=0) - prediction * target + torch.log(1.0 + torch.exp(-prediction.abs()))
        return bce.mean()
