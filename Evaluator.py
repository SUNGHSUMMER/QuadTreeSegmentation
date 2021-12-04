from Utils.metrics import ConfusionMatrix
from Utils.transform import masks_transform, images_transform
import torch.nn.functional as F


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
        """

        :param sample:sample is a dict, loaded by dataloader,it have three keys, id image and label
        :param model: the model to use
        :return: predictions is the output prediction of format of numpy
        """
        images, labels = sample['image'], sample['label']  # PIL images
        images = images_transform(images)  # list of PIL to Tensor
        labels_numpy = masks_transform(labels, numpy=True)  # list of PIL to numpy

        output = model(images)

        predictions = output.argmax(1).cpu().numpy()  # b, h, ws

        self.metrics.update(labels_numpy, predictions)

        return predictions
