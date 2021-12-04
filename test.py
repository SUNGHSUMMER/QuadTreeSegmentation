#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import torch
from torch import nn
import torch.utils.data as data

from Evaluator import Evaluator
from data_utils.dataloader import DeepGlobe, collate
from data_utils.getindex import txt2list
from models.minimal_global_seg.MinimalGlobalSeg import MinimalGlobalSeg

args = {
    "n_class": 7,
    "data_path": "./DeepGlobe/",
    "batch_size": 6,
    "model_path": "./logs/saved_models/",
    "log_path": "./logs/runs/",
    "device": "cuda",
    "devices": "0",
    "learning_rate": 0.001,
    "num_worker": 0,
    "sub_batch_size": 6,
    "model_name": "ResNet18_Bilinear",
}

n_class = args["n_class"]
model_name = args["model_name"]
print("Using model: " + model_name)

device = torch.device(args["device"])
if args["device"] == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = args["devices"]

data_path = args["data_path"]
img_path = os.path.join(data_path, "data")
index_path = os.path.join(data_path, "data_split")

batch_size = args["batch_size"]
sub_batch_size = args["sub_batch_size"]

num_worker = args["num_worker"]

ids_test = txt2list(os.path.join(index_path, "test.txt"))
dataset_test = DeepGlobe(img_path, ids_test, label=False)
dataloader_test = data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=num_worker,
                                  collate_fn=collate, shuffle=False, pin_memory=True)

print("Length of Test Dataset:", len(dataloader_test))

##############################################
model = MinimalGlobalSeg(num_classes=7, patch_size=306)
model = nn.DataParallel(model)
model = model.to(device)
##############################################

evaluator = Evaluator(n_class, sub_batch_size)



