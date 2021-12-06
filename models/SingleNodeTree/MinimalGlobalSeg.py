import torch.nn as nn
import torch.nn.functional as F
from models.BasedSegmentation.MiniModel.ResNet18_Bilinear import ResNet18_Bilinear


class SingleNodeTree(nn.Module):
    def __init__(self,num_classes=7, patch_size=306, backbone=ResNet18_Bilinear, pretrain=True):
        super(SingleNodeTree, self).__init__()
        self.patch_size = patch_size
        self.backbone = backbone(num_classes=num_classes, pretrain=pretrain)

    def forward(self, image):
        _, _, H, W = image.shape
        patch_H = self.patch_size
        patch_W = self.patch_size
        scaled_images = F.interpolate(image, size=(patch_H, patch_W),
                                      mode='bilinear', align_corners=True)
        output = self.backbone(scaled_images)
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)

        return output
