#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import sys

import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm

from Evaluator import Evaluator
from Utils.visual import show_score_as_table, show_results_as_plt, show_raw_prediction
from data_utils.dataloader import DeepGlobe, collate
from data_utils.getindex import txt2list
from models.minimal_global_seg.MinimalGlobalSeg import MinimalGlobalSeg

args = {
    "n_class": 7,
    "data_path": "./DeepGlobe/",
    "batch_size": 6,
    "loaded_model_name": "",
    "log_path": "./runs/",
    "device": "cuda",
    "devices": "0",
    "num_worker": 0,
    "model_path": "./saved_models/",
    "model_name": "ResNet18_Bilinear",
    "model_notes": "612_patch_global",
    "using_epoch": 44,
    "visualization_path": "./visualization/",
    "show_plt": True,
    "show_prediction": True
}

print("----------------------------------------------------------------")
n_class = args["n_class"]

#######################################################################
model = MinimalGlobalSeg(num_classes=n_class, patch_size=306)
#######################################################################

model_name = args["model_name"]
model_notes = args["model_notes"]
using_epoch = args["using_epoch"]
print("Using model: " + model_name + " And " + model_notes)
using_epoch = "epoch_" + str(using_epoch) + ".pth"
print("Test the epoch of " + str(using_epoch))

model_path = args["model_path"]
save_path = os.path.join(model_path, model_name + model_notes, using_epoch)
assert os.path.isfile(save_path), "Can't find this saved_model in: [" + save_path + "]"

show_plt = args["show_plt"]
show_prediction = args["show_prediction"]

visualization_path = args["visualization_path"]
plts_path = os.path.join(visualization_path, "plt", model_name + model_notes)
prediction_path = os.path.join(visualization_path, "prediction", model_name + model_notes)

device = torch.device(args["device"])
if args["device"] == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = args["devices"]

data_path = args["data_path"]
img_path = os.path.join(data_path, "data")
index_path = os.path.join(data_path, "data_split")

batch_size = args["batch_size"]

num_worker = args["num_worker"]

ids_test = txt2list(os.path.join(index_path, "test.txt"))
dataset_test = DeepGlobe(img_path, ids_test, label=True)
dataloader_test = data.DataLoader(dataset=dataset_test, batch_size=1, num_workers=num_worker,
                                  collate_fn=collate, shuffle=False, pin_memory=True)

print("Length of Test Dataset:", len(dataset_test))

##############################################
model = nn.DataParallel(model)
model.load_state_dict(torch.load(save_path, map_location=device))
model = model.to(device)
##############################################

evaluator = Evaluator(n_class)

tbar = tqdm(dataloader_test)

if not os.path.exists(plts_path):
    os.makedirs(plts_path)
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)

print("evaling...")
for i_batch, sample_batched in enumerate(tbar):
    with torch.no_grad():
        evaluator.set_eval(model)
        predictions = evaluator.eval_test(sample_batched, model)

        if show_plt:
            show_results_as_plt(sample_batched, predictions, plts_path=plts_path)
        if show_prediction:
            show_raw_prediction(sample_batched, predictions, prediction_path=prediction_path)

        score_test = evaluator.get_scores()
        tbar.set_description('mIoU: %.3f' % score_test["iou_mean"])

score_test = evaluator.get_scores()
evaluator.reset_metrics()
show_score_as_table(score_test)
