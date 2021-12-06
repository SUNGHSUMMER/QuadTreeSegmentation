import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasedSegmentation.deeplabv3plus.deeplab import DeepLabV3P
from einops.layers.torch import Rearrange
from einops import rearrange
from models.Tools.QuadTree import QuadTree


class SingleSubTreeSeg(nn.Module):
    """
    该网络是四叉树分割的一个逻辑非常简化的版本，不再动了。
    具体地，网络在预测完整张图像之后，随之将网络分成了四块，不需要任何中间的判断条件，直接进行预测
    最后预测的结果拼接起来，覆盖掉原来的整张图片的结果
    """

    def __init__(self, num_classes=7, patch_size=306, backbone=DeepLabV3P):
        super(SingleSubTreeSeg, self).__init__()
        self.patch_size = patch_size
        self.backbone = backbone(num_classes=num_classes)

    def forward(self, image):
        tree_list = []
        final_predict = []
        batch_size, _, H0, W0 = image.shape

        # return a tuple of tensor with the shape:(1, channel, H0, W0)
        # tuple size is: batch_size

        image_tuple = torch.chunk(image, batch_size, dim=0)

        patch_H = self.patch_size
        patch_W = self.patch_size
        scaled_images = F.interpolate(image, size=(patch_H, patch_W),
                                      mode='bilinear', align_corners=True)
        output_stage0 = self.backbone(scaled_images)

        outputs_stage0 = torch.chunk(output_stage0, batch_size, dim=0)

        # output_stage0_full = F.interpolate(output_stage0,
        #                                    size=(H0, W0), mode='bilinear', align_corners=True)

        for index in range(batch_size):
            # create a new QuadTree and append into treelist
            quad_tree = QuadTree()
            tree_list.append(quad_tree)

            # the base image and prediction
            single_image = image_tuple[0]
            single_predict = outputs_stage0[0]

            quad_tree.base_image = single_image
            quad_tree.base_predict = single_predict

            sub_images = rearrange(quad_tree.base_image,
                                   '1 c (p1 h) (p2 w) -> (p1 p2) c h w', p1=2, p2=2)
            _, _, H1, W1 = sub_images.shape
            sub_images = F.interpolate(sub_images, size=(patch_H, patch_W),
                                       mode='bilinear', align_corners=True)

            sub_predict = self.backbone(sub_images)

            sub_predict_tuple = torch.chunk(sub_predict, 4, dim=0)

            quad_tree.predict00 = sub_predict_tuple[0]
            quad_tree.predict01 = sub_predict_tuple[1]
            quad_tree.predict10 = sub_predict_tuple[2]
            quad_tree.predict11 = sub_predict_tuple[3]

            sub_predict = rearrange(sub_predict,
                                    '(p1 p2) c h w -> 1 c (p1 h) (p2 w)', p1=2, p2=2)

            sub_predict = F.interpolate(sub_predict, size=(H0, W0),
                                        mode='bilinear', align_corners=True)
            final_predict.append(sub_predict)

        # output.base_predict = output_stage0_full

        final_predict = torch.concat(final_predict, 0)
        print(final_predict.shape)

        return final_predict
