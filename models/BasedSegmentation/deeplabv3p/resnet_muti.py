import numpy
import torch
import torch.nn as nn
import torchvision.models as models
import os


class ResNet18_Muti(nn.Module):
    def __init__(self):
        super(ResNet18_Muti, self).__init__()
        resnet = models.resnet18()

        # load pretrained model:
        # path_name = os.path.join('model', 'resnet', 'resnet18-5c106cde.pth')
        # resnet.load_state_dict(torch.load(path_name))

        # remove fully connected layer, avg pool, layer4 and layer5:
        self.resnet = nn.Sequential(*list(resnet.children()))
        self.resnet4x = self.resnet[0:4]
        self.resnet8x = self.resnet[4:6]
        self.resnet16x = self.resnet[6]

    def forward(self, x):
        feat4x = self.resnet4x(x)

        feat8x = self.resnet8x(feat4x)

        feat16x = self.resnet16x(feat8x)

        return feat16x, feat8x, feat4x

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    net = ResNet18_Muti()
    f32x, f16x, f8x = net(x)
    print("feature32x: " + str(f32x.shape))
    print("feature16x: " + str(f16x.shape))
    print("feature8x: " + str(f8x.shape))

    # summary(net, (3, 256, 256), batch_size=1, device='cpu')