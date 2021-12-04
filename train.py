#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os

import torch
from torch import nn
from tqdm import tqdm
import torch.utils.data as data

from Evaluator import Evaluator
from Trainer import Trainer
from data_utils.dataloader import DeepGlobe, collate
from Utils.loss import FocalLoss
from Utils.lr_scheduler import LR_Scheduler
from Utils.optimizer import get_Adam_optimizer
from data_utils.getindex import txt2list
# from models.minimal_global_seg.MinimalGlobalSeg import MinimalGlobalSeg
from models.minimal_global_seg.MinimalGlobalSeg_DeepLabV3plus_ResNet101 import MinimalGlobalSeg
from tensorboardX import SummaryWriter
from Utils.visual import show_score_as_table

args = {
    "n_class": 7,
    "data_path": "./DeepGlobe/",
    "batch_size": 6,
    "model_path": "./saved_models/",
    "log_path": "./logs/",
    "device": "cuda",
    "devices": "0, 1",
    "learning_rate": 0.001,
    "num_epochs": 50,
    "num_worker": 0,
    "augment": True,
    "sub_batch_size": 6,
    "model_name": "MGS",
    "model_notes": "DeepLabV3P_ResNet101",
    "val_frequency": 5
}
print("----------------------------------------------------------------")
n_class = args["n_class"]

#######################################################################
model = MinimalGlobalSeg(num_classes=n_class, patch_size=306)
#######################################################################
model_name = args["model_name"]
model_notes = args["model_notes"]
print("Using model: " + model_name + " And " + model_notes)
epoch_name = ""

device = torch.device(args["device"])
if args["device"] == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = args["devices"]

data_path = args["data_path"]
img_path = os.path.join(data_path, "data")
index_path = os.path.join(data_path, "data_split")

log_path = args["log_path"]
if not os.path.exists(log_path):
    os.makedirs(log_path)

model_path = args["model_path"]
if not os.path.exists(model_path):
    os.makedirs(model_path)

ids_train = txt2list(os.path.join(index_path, "train.txt"))
ids_val = txt2list(os.path.join(index_path, "crossvali.txt"))

augment = args["augment"]

num_epochs = args["num_epochs"]
learning_rate = args["learning_rate"]
val_frequency = args["val_frequency"]

save_path = os.path.join(model_path, model_name + model_notes)
if not os.path.exists(save_path):
    os.makedirs(save_path)

batch_size = args["batch_size"]
sub_batch_size = args["sub_batch_size"]

num_worker = args["num_worker"]

###################################
print("preparing datasets and dataloaders......")

dataset_train = DeepGlobe(img_path, ids_train, label=True, augment=augment)
dataloader_train = data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=num_worker,
                                   collate_fn=collate, shuffle=True, pin_memory=True)

dataset_val = DeepGlobe(img_path, ids_val, label=True)
dataloader_val = data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=num_worker,
                                 collate_fn=collate, shuffle=False, pin_memory=True)

print("Length of Train Dataset:", len(dataset_train))
print("Length of Val Dataset:", len(dataset_val))

##############################################
model = nn.DataParallel(model)
model = model.to(device)
##############################################

optimizer = get_Adam_optimizer(model, learning_rate)
scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))

criterion1 = FocalLoss(gamma=3)
criterion = lambda x, y: criterion1(x, y)

trainer = Trainer(criterion, optimizer, n_class, sub_batch_size, mode=1)
evaluator = Evaluator(n_class, sub_batch_size)

writer = SummaryWriter(log_dir=log_path + "DeepGlobe")
f_log = open(log_path + "DeepGlobe" + ".log", 'w')

best_miou = 0.0
best_epoch = 0

print("----------------------------------------------------------------")
print("start training......")

for epoch in range(num_epochs):

    trainer.set_train(model)
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train)
    train_loss = 0
    for i_batch, sample_batched in enumerate(tbar):
        scheduler(optimizer, i_batch, epoch)  # update lr
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

    if (epoch + 1) % val_frequency == 0:
        with torch.no_grad():
            print("evaling...")
            model.eval()
            tbar = tqdm(dataloader_val)
            for i_batch, sample_batched in enumerate(tbar):
                predictions = evaluator.eval_test(sample_batched, model)
                score_val = evaluator.get_scores()
                tbar.set_description('mIoU: %.3f' % score_val["iou_mean"])
            pth_path = os.path.join(save_path, "epoch_" + str(epoch + 1) + ".pth")
            torch.save(model.state_dict(), pth_path)

            score_val = evaluator.get_scores()
            evaluator.reset_metrics()

            if score_val["iou_mean"] > best_miou:
                best_miou = score_val["iou_mean"]
                best_epoch = epoch + 1

            log = ""
            log = log + 'epoch [{}/{}] mIoU: train = {:.4f}, val = {:.4f}'.format(
                epoch + 1, num_epochs, score_train["iou_mean"], score_val["iou_mean"]) + "\n"
            log = log + "train: " + str(score_train["iou"]) + "\n"
            log = log + "val:" + str(score_val["iou"]) + "\n"
            log += "================================\n"

            # print(log)
            print("epoch [{}/{}]".format(epoch + 1, num_epochs))
            show_score_as_table(score_train, score_val)

            f_log.write(log)
            f_log.flush()
            writer.add_scalars('IoU', {'train iou': score_train["iou_mean"],
                                       'validation iou': score_val["iou_mean"]}, epoch + 1)

final_log = "Best mIoU in Val Dataset is: " + str(best_miou) + " in Epoch: " + str(best_epoch)

print(final_log)
f_log.write(final_log)

f_log.close()
