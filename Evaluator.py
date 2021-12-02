from Utils.metrics import ConfusionMatrix
import torch
import sys
from Utils.metrics import ConfusionMatrix
from Utils.transform import masks_transform, images_transform
import torch.nn.functional as F
import sys


class Evaluator(object):
    def __init__(self, n_class, sub_batch_size=6, val=True):
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.sub_batch_size = sub_batch_size
        self.val = val

    @staticmethod
    def set_eval(model):
        model.module.eval()

    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def eval_test(self, sample, model):
        images, labels = sample['image'], sample['label']  # PIL images
        images = images_transform(images)  # list of PIL to Tensor
        labels_numpy = masks_transform(labels, numpy=True)  # list of PIL to numpy

        _, _, H, W = images.shape

        scaled_images = F.interpolate(images, size=(306, 306), mode='bilinear', align_corners=True)

        output = model(scaled_images)
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)

        predictions = output.argmax(1).cpu().numpy()  # b, h, ws

        self.metrics.update(labels_numpy, predictions)

        return predictions
