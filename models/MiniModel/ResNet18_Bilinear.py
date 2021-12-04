import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MiniModel.ResNet_Features import ResNet18_Features


class ResNet18_Bilinear(nn.Module):
    """
    这个模型只有M1部分的ResNet18（预训练过的）
    八倍下采样之后，经过一个1*1卷积，将通道数压缩到类别数之后，
    直接双线性插值上采样得到最终的预测结果
    """
    def __init__(self, num_classes=7, pretrain=True):
        super(ResNet18_Bilinear, self).__init__()
        self.num_classes = num_classes
        self.feature_extract = ResNet18_Features(pretrain=pretrain)
        self.class_predict = nn.Conv2d(in_channels=128,
                                       out_channels=num_classes,
                                       kernel_size=(1, 1))

    def forward(self, input):
        h, w = input.size()[2:]
        local_feature = self.feature_extract(input)
        output = F.interpolate(local_feature, size=(h, w), mode='bilinear', align_corners=True)
        output = self.class_predict(output)

        return output


if __name__ == "__main__":
    x = torch.randn(4, 3, 256, 256)
    net = ResNet18_Bilinear()
    output = net(x)
    print(output.shape)
