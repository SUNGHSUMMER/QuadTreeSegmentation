import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BasedSegmentation.deeplabv3p.aspp import ASPP
from models.BasedSegmentation.deeplabv3p.resnet_muti import ResNet18_Muti


class DeepLabV3P(nn.Module):
    def __init__(self):
        super(DeepLabV3P, self).__init__()

        self.out_channels = 64

        self.resnet = ResNet18_Muti() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.out_channels) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.feat4xrefine = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1,padding=0, stride=1)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feat16x, feat4x = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        feat16x = self.aspp(feat16x) # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.interpolate(feat16x, size=feat4x.size()[2:], mode='bilinear', align_corners=True)

        concated = torch.cat([feat4x, output], 1)

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

if __name__ == "__main__":
    net = DeepLabV3P()
    test = torch.randn((2, 3, 256, 256))
    out = net(test)

