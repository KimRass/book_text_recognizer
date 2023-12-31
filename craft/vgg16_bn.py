from collections import namedtuple
import torch.nn as nn
import torchvision
from torchvision import models
from packaging import version

from craft.torch_utils import initialize_weights


class vgg16_bn(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else:
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        # conv2_2
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # conv3_3
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # conv4_3
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # conv5_3
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            initialize_weights(self.slice1.modules())
            initialize_weights(self.slice2.modules())
            initialize_weights(self.slice3.modules())
            initialize_weights(self.slice4.modules())

        # No pretrained model for fc6 and fc7
        initialize_weights(self.slice5.modules())

        if freeze:
            # Only first conv
            for param in self.slice1.parameters():
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
