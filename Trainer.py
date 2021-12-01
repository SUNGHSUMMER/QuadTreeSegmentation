import torch
import sys
from Utils.metrics import ConfusionMatrix
from Utils.transform import masks_transform
import sys

class Trainer(object):
    def __init__(self, criterion, optimizer, n_class, sub_batch_size=6, mode=1):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.sub_batch_size = sub_batch_size

    @staticmethod
    def set_train(model):
        model.module.train()

    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def train(self, sample, model):
        images, labels = sample['image'], sample['label']  # PIL images
        labels_npy = masks_transform(labels, numpy=True)  # label of origin size in numpy

        self.optimizer.step()
        self.optimizer.zero_grad()
        ####################################################################################
        scores = np.array(patch2global(predicted_patches, self.n_class, sizes, coordinates,
                                       self.size_p))  # merge softmax scores from patches (overlaps)
        predictions = scores.argmax(1)  # b, h, w
        self.metrics.update(labels_npy, predictions)
        return loss
