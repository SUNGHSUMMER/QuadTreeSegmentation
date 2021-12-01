#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import torch.utils.data as data

from Evaluator import Evaluator
from Trainer import Trainer
from data_utils.dataloader import DeepGlobe
from Utils.loss import FocalLoss
from Utils.lr_scheduler import LR_Scheduler
from Utils.optimizer import get_Adam_optimizer
from data_utils.getindex import txt2list
from models.MiniModel.ResNet18_Bilinear import ResNet18_Bilinear
from tensorboardX import SummaryWriter

args = {
    "n_class": 7,
    "data_path": "./DeepGlobe/",
    "batch_size": 6,
    "model_path": "./logs/saved_models/",
    "log_path": "./logs/runs/",
    "device": "cuda",
    "devices": "0, 1",
    "learning_rate": 0.001,
    "num_epochs": 50,
    "num_worker": 0,
    "transform": True,
    "sub_batch_size": 6
}

n_class = args["n_class"]

device = torch.device(args["device"])
if args["device"] == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = args["device"]

data_path = args["data_path"]
img_path = os.path.join(data_path, "data")
index_path = os.path.join(data_path, "data_split")

log_path = args["log_path"]
model_path = args["model_path"]

print(img_path)
print(index_path)

ids_train = txt2list(os.path.join(index_path, "train.txt"))
ids_test = txt2list(os.path.join(index_path, "crossvali.txt"))
ids_val = txt2list(os.path.join(index_path, "test.txt"))

transform = args["transform"]

num_epochs = args["num_epochs"]
learning_rate = args["learning_rate"]

batch_size = args["batch_size"]
sub_batch_size = args["sub_batch_size"]

num_worker = args["num_worker"]

###################################
print("preparing datasets and dataloaders......")

dataset_train = DeepGlobe(img_path, ids_train, label=True, transform=transform)
dataloader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=num_worker,
                                               shuffle=True, pin_memory=True)

dataset_val = DeepGlobe(img_path, ids_val, label=True)
dataloader_val = data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=num_worker,
                                             shuffle=False, pin_memory=True)

dataset_test = DeepGlobe(img_path, ids_test, label=False)
dataloader_test = data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=num_worker,
                                              shuffle=False, pin_memory=True)

print("Length of Train Dataset:", len(dataloader_train))
print("Length of Val Dataset:", len(dataloader_val))
print("Length of Test Dataset:", len(dataloader_test))

###################################

model = ResNet18_Bilinear()
model = nn.DataParallel(model)
model = model.to(device)

optimizer = get_Adam_optimizer(model, learning_rate)
scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

criterion1 = FocalLoss(gamma=3)
criterion = lambda x, y: criterion1(x, y)

trainer = Trainer(criterion, optimizer, n_class, sub_batch_size, mode=1)
# evaluator = Evaluator(n_class, size_p, size_g, sub_batch_size, mode, train, dataset, context10, context15)

writer = SummaryWriter(log_dir=log_path + "DeepGlobe")
f_log = open(log_path + "DeepGlobe" + ".log", 'w')

for epoch in range(num_epochs):

    trainer.set_train(model)
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train)
    train_loss = 0
    for i_batch, sample_batched in enumerate(tbar):
        scheduler(optimizer, i_batch, epoch, best_pred)  # update lr
        loss = trainer.train(sample_batched, model)
        train_loss += loss.item()
        score_train = trainer.get_scores()
        tbar.set_description('epoch:%d Train loss: %.3f;mIoU: %.3f'
                             % (epoch + 1, train_loss / (i_batch + 1), score_train["iou_mean"]))
        writer.add_scalar('train_loss', loss, epoch * len(dataloader_train) + i_batch)
        writer.add_scalar('train_miou', score_train["iou_mean"],
                          epoch * len(dataloader_train) + i_batch)

    score_train = trainer.get_scores()
    trainer.reset_metrics()
    # torch.cuda.empty_cache()

    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            print("evaling...")
            model.eval()
            tbar = tqdm(dataloader_val)
            for i_batch, sample_batched in enumerate(tbar):
                predictions = evaluator.eval_test(sample_batched, model, global_fixed_medium, global_fixed_large)
                score_val = evaluator.get_scores()
                # use [1:] since class0 is not considered in deep_globe metric
                tbar.set_description('mIoU: %.3f' % (np.mean(np.nan_to_num(score_val["iou"])[1:])))
                images = sample_batched['image']
                labels = sample_batched['label']  # PIL images

                if i_batch * batch_size + len(images) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (
                        epoch % len(dataloader_val)):
                    writer.add_image('image', transforms.ToTensor()(
                        images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)
                    writer.add_image('mask', classToRGB(dataset, np.array(
                        labels[(epoch % len(dataloader_val)) - i_batch * batch_size])), epoch)
                    writer.add_image('prediction', classToRGB(dataset, predictions[
                        (epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)

            torch.save(model.state_dict(), model_path + task_name + ".epoch" + str(epoch) + ".pth")

            score_val = evaluator.get_scores()
            evaluator.reset_metrics()

            if np.mean(np.nan_to_num(score_val["iou"][1:])) > best_pred: best_pred = np.mean(
                np.nan_to_num(score_val["iou"][1:]))
            log = ""
            log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch + 1, num_epochs, np.mean(
                np.nan_to_num(score_train["iou"][1:])), np.mean(np.nan_to_num(score_val["iou"][1:]))) + "\n"
            log = log + "train: " + str(score_train["iou"]) + "\n"
            log = log + "val:" + str(score_val["iou"]) + "\n"
            log += "================================\n"
            print(log)

            f_log.write(log)
            f_log.flush()
            writer.add_scalars('IoU', {'train iou': score_train["iou_mean"],
                                       'validation iou': score_val["iou_mean"]}, epoch)
f_log.close()
