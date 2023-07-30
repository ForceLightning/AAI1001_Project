import os
import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import accuracy, Recall, F1Score
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import fit_one_cycle
from fastai.callback.progress import ProgressCallback, ShowGraphCallback, CSVLogger
from fastai.callback.tracker import SaveModelCallback
import fastai.callback.schedule

from scripts.utils import *


def main(
    batch_size=2048,
    shuffle_train=True,
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    model_dir="./models/",
    num_splits=10
):
    train_set = ECGDataset("./data/heartbeats_evensplit_train/",
                           item_transform=hb_transform, target_transform=label_encode)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,
                              pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers)

    # json dump model training progress
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if os.path.exists(f"{model_dir}/model.json"):
        with open(f'{model_dir}/model.json', 'r') as f:
            interruptable_info = json.load(f)
    else:
        interruptable_info = {
            "fold": 0,
            "epoch": 0
        }
        with open(f'{model_dir}/model.json', 'w') as f:
            json.dump(interruptable_info, f, indent=4)

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    train_ys = np.array([y for y in train_set.y])

    learners = []
    use_cuda = torch.cuda.is_available()
    for idx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(train_ys)), train_ys)):
        if not os.path.exists(f"{model_dir}/tcn_fold_{idx+1}"):
            os.makedirs(f"{model_dir}/tcn_fold_{idx+1}")
        tcn_model = TCN(360, 5, [32]*9, 2, 0.125, use_skip_connections=True)
        if use_cuda:
            tcn_model = tcn_model.cuda()

        if idx < interruptable_info["fold"]:
            continue

        train_fold_set = Subset(train_set, train_index)
        val_fold_set = Subset(train_set, val_index)
        train_fold_loader = DataLoader(train_fold_set, batch_size=batch_size, shuffle=shuffle_train,
                                       pin_memory=pin_memory, num_workers=num_workers, persistent_workers=persistent_workers)
        val_fold_loader = DataLoader(
            val_fold_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        fold_dls = DataLoaders(train_fold_loader, val_fold_loader)
        best_model_cb = SaveModelCallback(monitor="valid_loss", fname="best")
        every_epoch_save_cb = SaveModelCallback(
            monitor="valid_loss", fname="epoch", every_epoch=True, with_opt=True)
        csv_logger_cb = CSVLogger(
            fname=f"{model_dir}/tcn_fold_{idx+1}/log.csv", append=True)
        learner_cbs = [MixedPrecision()] if use_cuda else None
        learn = Learner(
            dls=fold_dls,
            model=tcn_model,
            model_dir=f"{model_dir}/tcn_fold_{idx+1}",
            loss_func=nn.CrossEntropyLoss(),
            cbs=learner_cbs,
            metrics=[accuracy, fastai_precision_score(average="macro", zero_division=0.0), Recall(
                average="macro"), F1Score(average="macro")]
        )
        if interruptable_info["epoch"] != 0 and os.path.exists(f"{model_dir}/tcn_fold_{idx+1}/epoch_{interruptable_info['epoch']}.pth"):
            learn.load(
                f"{model_dir}/tcn_fold_{idx+1}/epoch_{interruptable_info['epoch'] + 1}.pth")
        learn.fit_one_cycle(
            n_epoch=100,
            lr_max=3e-3,
            div=10.0,
            start_epoch=interruptable_info["epoch"],
            wd=1e-5,
            cbs=[LogInterruptable(filename=f"{model_dir}/model.json"), best_model_cb,
                 every_epoch_save_cb, csv_logger_cb, ShowGraphCallback()]
        )
        if persistent_workers:
            train_fold_loader._iterator._shutdown_workers()
            val_fold_loader._iterator._shutdown_workers()
        learners.append(learn)
        interruptable_info["fold"] += 1
        interruptable_info["epoch"] = 0
        json.dump(interruptable_info, open(
            f"{model_dir}/model.json", "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN model")
    parser.add_argument("--batch_size", type=int,
                        default=2048, help="Batch size")
    parser.add_argument("--shuffle_train", type=bool,
                        default=True, help="Shuffle train set")
    parser.add_argument("--pin_memory", type=bool,
                        default=True, help="Pin memory")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="Number of workers")
    parser.add_argument("--persistent_workers", type=bool,
                        default=True, help="Persistent workers")
    parser.add_argument("--model_dir", type=str,
                        default="./models/", help="Model directory")
    parser.add_argument("--num_splits", type=int,
                        default=10, help="Number of splits")
    args = parser.parse_args()
    main(**vars(args))
