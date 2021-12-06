import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50


class FCN_ResNet50(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):
        super(FCN_ResNet50, self).__init__()
        fcn = fcn_resnet50(pretrained=pretrained, num_classes=num_classes)
        self.fcn = nn.Sequential(*list(fcn.children()))
        self.fcn = self.fcn[:]
        print(type(self.fcn))

    def forward(self, x):
        output = self.fcn(x)
        return output


if __name__ == "__main__":
    net = FCN_ResNet50()
    input = torch.randn((4, 3, 256, 256))
    output = net(input)
    print(output.shape)
