from Utils.metrics import ConfusionMatrix
from Utils.transform import masks_transform, images_transform


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
        images = images_transform(images)  # list of PIL to Tensor
        labels_tensor = masks_transform(labels, numpy=False)  # list of PIL to numpy
        labels_numpy = masks_transform(labels, numpy=True)  # list of PIL to numpy


        output = model(images)

        # _, _, H, W = images.shape

        # scaled_images = F.interpolate(images, size=(612, 612), mode='bilinear', align_corners=True)
        #
        # output = model(scaled_images)
        # output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)

        # print("output", output.shape)
        # print("labels", labels_npy.shape)

        loss = self.criterion(output, labels_tensor)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        predictions = output.argmax(1).cpu().numpy()  # b, h, ws

        # print(labels_numpy.shape)
        # print(predictions.shape)

        self.metrics.update(labels_numpy, predictions)

        return loss
