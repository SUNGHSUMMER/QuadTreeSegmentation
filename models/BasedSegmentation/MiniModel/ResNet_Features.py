import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


class ResNet18_Features(nn.Module):
    def __init__(self, pretrain=False):
        super(ResNet18_Features, self).__init__()
        backbone = resnet18()
        if pretrain:
            # 以项目目录索引
            pretrained_model_path = os.path.join("models", "PretrainedModel", "resnet18-5c106cde.pth")
            # 以当前目录索引
            # pretrained_model_path = "../PretrainedModel/resnet18-5c106cde.pth"

            backbone.load_state_dict(torch.load(pretrained_model_path))
            print("Using the Pretrain ResNet18 as BackBone")

        self.resnet = nn.Sequential(*list(backbone.children()))

        self.backbone = self.resnet[0:6]

    def forward(self, local_img):
        feature = self.backbone(local_img)
        return feature


class ResNet50_Features(nn.Module):
    def __init__(self, pretrain=False):
        super(ResNet50_Features, self).__init__()
        backbone = resnet50()
        if pretrain:
            pretrained_model_path = os.path.join("model",
                                                 "PretrainedModel",
                                                 "resnet50-19c8e357.pth")
            backbone.load_state_dict(torch.load(pretrained_model_path))
            print("Using the Pretrain ResNet50 as BackBone")

        self.resnet = nn.Sequential(*list(backbone.children()))

        self.backbone = self.resnet[0:6]

    def forward(self, local_img):
        feature = self.backbone(local_img)
        return feature


if __name__ == "__main__":
    local_img = torch.rand(4, 3, 256, 256)
    net = ResNet18_Features(pretrain=True)
    output = net(local_img)
    print(output.shape)
