import os.path
import sys

from prettytable import PrettyTable
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from Utils.transform import masks_transform
from data_utils.deepglobe_colormap import deepglobe_color_map
from PIL import Image


def show_score_as_table(*score):
    """
    show score as table

    """
    data_length = len(score)
    tb = PrettyTable()
    if data_length == 2:
        train_score = score[0]
        train_iou = train_score["iou"]
        train_miou = train_score["iou_mean"]

        val_score = score[1]
        val_iou = val_score["iou"]
        val_miou = val_score["iou_mean"]

        tb.field_names = ["type", "train", "val"]
        tb.add_row(["mIoU", train_miou, val_miou])
        tb.add_row(["urban", train_iou[0], val_iou[0]])
        tb.add_row(["agriculture", train_iou[1], val_iou[1]])
        tb.add_row(["rangeland", train_iou[2], val_iou[2]])
        tb.add_row(["forest", train_iou[3], val_iou[3]])
        tb.add_row(["water", train_iou[4], val_iou[4]])
        tb.add_row(["barren", train_iou[5], val_iou[5]])
        print(tb)
    if data_length == 1:
        test_score = score[0]
        test_iou = test_score["iou"]
        test_miou = test_score["iou_mean"]

        tb.field_names = ["type", "test"]
        tb.add_row(["mIoU", test_miou])
        tb.add_row(["urban", test_iou[0]])
        tb.add_row(["agriculture", test_iou[1]])
        tb.add_row(["rangeland", test_iou[2]])
        tb.add_row(["forest", test_iou[3]])
        tb.add_row(["water", test_iou[4]])
        tb.add_row(["barren", test_iou[5]])
        print(tb)


def numpy2Image(prediction):
    colorize = deepglobe_color_map()
    label = colorize[prediction, :].reshape([prediction.shape[0], prediction.shape[1], 3])
    return label


def show_results_as_plt(sample, prediction, plts_path):
    id, image, label = sample['id'], sample['image'], sample['label']  # PIL images

    labels_numpy = masks_transform(label, numpy=True)  # list of PIL to numpy

    id = id[0]
    image = image[0]
    label_numpy = labels_numpy[0]

    prediction = np.squeeze(prediction, axis=0)

    # print(label_numpy.shape)
    # print(prediction.shape)

    label_img = numpy2Image(label_numpy)
    prediction_img = numpy2Image(prediction)

    plt.subplot(131)
    plt.axis('off')
    plt.title('Original Picture')
    plt.imshow(image)

    plt.subplot(132)
    plt.axis('off')
    plt.title('Ground Truth')
    plt.imshow(label_img)

    plt.subplot(133)
    plt.axis('off')
    plt.title('Predict Label')
    plt.imshow(prediction_img)

    plt_root = os.path.join(plts_path, id + ".png")
    plt.savefig(plt_root)
    plt.close()


def show_raw_prediction(sample, prediction, prediction_path):
    id = sample['id'][0]  # find the image id

    prediction = np.squeeze(prediction, axis=0)
    prediction_img = numpy2Image(prediction)
    image = Image.fromarray(np.uint8(prediction_img))
    image_root = os.path.join(prediction_path, id + ".png")
    image.save(image_root)
