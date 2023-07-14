import sys
import os

import json
import signal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, precision_recall_curve
from sklearn.metrics import roc_curve as sk_roc_curve

from fastai.learner import Learner
from fastai.metrics import accuracy, Precision, Recall, F1Score, RocAuc
from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
from fastai.metrics import skm_to_fastai

sys.path.append("../" + os.path.dirname(os.path.realpath(__file__)))
from TCN.TCN.tcn import TCN_DimensionalityReduced

class ECGDataset(Dataset):
    def __init__(self, data_dir, item_transform=None, target_transform=None, in_memory=True) -> None:
        self.data_dir = data_dir
        self.lengths = {}
        self.item_transform = item_transform
        self.target_transform = target_transform
        self.data = []
        self.in_memory = in_memory
        for _, _, files in os.walk(data_dir):
            for file in files:
                npzfile = np.load(data_dir + file)
                self.lengths[file] = len(npzfile["labels"])
                if in_memory:
                    hbs = npzfile["heartbeats"]
                    labels = npzfile["labels"]
                    self.data += zip(hbs, labels)
        if in_memory:
            self.X = np.array([hb for (hb, _) in self.data])
            self.y = np.array([label for (_, label) in self.data])
            del self.data
    def __len__(self) -> int:
        return sum(self.lengths.values())
    
    def __getitem__(self, idx):
        assert idx < self.__len__() # make sure we're not out of bounds
        if not self.in_memory:
            running_count = 0
            for file, length in self.lengths.items():
                if idx < running_count + length:
                    npzfile = np.load(self.data_dir + file)
                    hb = npzfile["heartbeats"][idx - running_count]
                    label = npzfile["labels"][idx - running_count]
                    if self.item_transform:
                        hb = self.item_transform(hb)
                    if self.target_transform:
                        label = self.target_transform(label)
                    return hb, label
                else:
                    running_count += length
        else:
            hb = self.X[idx]
            label = self.y[idx]
            hb = self.item_transform(hb) if self.item_transform else hb
            label = self.target_transform(label) if self.target_transform else label
            return hb, label
        # should never get here
        raise IndexError("Index out of bounds")

def one_hot_encode(label, classes=["N", "S", "V", "F", "Q"]):
    return torch.eye(len(classes))[classes.index(label)]

def one_hot_decode(label, classes=["N", "S", "V", "F", "Q"]):
    return classes[label.argmax()]

def noise_at_frequencies(
    hb,
    frequencies_distribution={
        "breathing": ([1/18, 1/12], [1/5, 1/3])
    },
    fs=360
):
    noise = np.zeros(hb.shape)
    for (source, (freq_range, amp_range)) in frequencies_distribution.items():
        freq = np.random.uniform(*freq_range)
        amp = np.random.uniform(*amp_range)
        phase = np.random.uniform(0, 2 * np.pi)
        noise += amp * np.sin(2 * np.pi * freq * np.arange(hb.shape[0]) / fs + phase)
    return hb + noise

def noise_at_frequencies_tensor(
    hb: torch.Tensor,
    frequencies_distribution={
        "breathing": ([1/18, 1/12], [1/5, 1/3])
    },
    fs=360
):
    noise = torch.zeros(hb.shape)
    for (source, (freq_range, amp_range)) in frequencies_distribution.items():
        freq = torch.distributions.uniform.Uniform(*freq_range).sample()
        amp = torch.distributions.uniform.Uniform(*amp_range).sample()
        phase = torch.distributions.uniform.Uniform(0, 2 * np.pi).sample()
        noise += amp * torch.sin(2 * np.pi * freq * torch.arange(hb.shape[0]) / fs + phase)
    return hb + noise

def z_normalise(hb):
    return (hb - hb.mean()) / hb.std()

def hb_transform(hb):
    hb = torch.from_numpy(hb)
    hb = noise_at_frequencies_tensor(hb)
    hb = z_normalise(hb)
    return hb

def label_encode(label, classes=["N", "S", "V", "F", "Q"]):
    return classes.index(label)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, use_skip_connections=False):
        super(TCN, self).__init__()
        self.tcn = TCN_DimensionalityReduced(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_skip_connections=use_skip_connections
        )
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])
        return o

class LogInterruptable(Callback):
    def __init__(self, fn:str, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        if os.path.exists(self.fn):
            with open(self.fn, "r") as f:
                self.interrupt_info = json.load(f)
        else:
            self.interrupt_info = {
                "fold": 0,
                "epoch": 0
            }
            with open(self.fn, "w") as f:
                json.dump(self.interrupt_info, f, indent=4)

    def after_epoch(self):
        self.interrupt_info["epoch"] = self.epoch
        with open(self.fn, "w") as f:
            json.dump(self.interrupt_info, f, indent=4)

    def after_fit(self):
        self.interrupt_info["fold"] += 1
        self.interrupt_info["epoch"] = 0
        with open(self.fn, "w") as f:
            json.dump(self.interrupt_info, f, indent=4)

def k_fold_cross_val_predictions(skf:StratifiedKFold, model, model_dir, model_fn, train_set, batch_size=4096, shuffle=True, num_workers=0, pin_memory=True):
    model_outputs = []
    classification_reports = []
    train_ys = np.array([y for _, y in train_set])
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_ys)), train_ys)):
        print(f"--- Fold {fold + 1} ---")
        val_fold_set = Subset(train_set, val_idx)
        val_fold_loader = DataLoader(val_fold_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        preds = []
        y = []
        learner = Learner(model.cuda(), val_fold_loader, loss_func=nn.CrossEntropyLoss(), cbs=[MixedPrecision()])
        learner.load(model_dir + f"{fold}" + "/" + model_fn)
        preds, y = learner.get_preds(dl=val_fold_loader)
        classification_reports.append(
            classification_report(
                y.cpu().numpy(),
                preds.argmax(dim=1).cpu().numpy(),
                output_dict=True,
                labels=["N", "S", "V", "F", "Q"]
            )
        )
        model_outputs.append({
            "preds": preds,
            "y": y,
        })
    return model_outputs, classification_reports

def k_fold_inference(model, test_dl, model_dir="./prototyping/tcn_fold_", weights_fn="best", k=10):
    model_outputs = []
    classification_reports = []
    for fold in range(k):
        # model.load_state_dict(torch.load(model_dir + f"{fold + 1}" + "/" + weights_fn))
        # model = model.cuda()
        learner = Learner(model=model.cuda(), dls=test_dl, loss_func=nn.CrossEntropyLoss(), cbs=[MixedPrecision()])
        print(model_dir + f"{fold + 1}" + "/" + weights_fn)
        learner.load(model_dir + f"{fold + 1}" + "/" + weights_fn)
        proba, y = learner.get_preds(dl=test_dl)
        preds = proba.argmax(dim=1)
        classification_reports.append(
            classification_report(
                y.cpu().numpy(),
                preds,
                output_dict=True,
                target_names=["N", "S", "V", "F", "Q"]
            )
        )
        model_outputs.append({
            "preds": preds,
            "proba": proba,
            "y": y,
        })
    return model_outputs, classification_reports

def k_fold_prc_curve(model_outputs, model_name:str):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=100)
    # tprs, aurocs, tpr_threshes = [], [], []
    # fpr_mean = np.linspace(0, 1, 1000)
    precisions, auprcs, recall_threshes = [], [], []
    recall_mean = np.linspace(0, 1, 1000)
    for fold_idx, fold in enumerate(tqdm(model_outputs)):
        # ROC Curve
        # fpr, tpr, tpr_thresh = sk_roc_curve(fold['y'], fold['proba'].detach().cpu().numpy()[:, 1])
        # tpr_threshes.append(tpr_thresh[np.abs(tpr-0.85).argmin()])
        # tpr_interp = np.interp(fpr_mean, fpr, tpr)
        # tpr_interp[0] = 0.0
        # tprs.append(tpr_interp)
        # aurocs.append(auc(fpr, tpr))
        # ax[0].plot(fpr, tpr, label=f"ROC Fold {fold_idx + 1} (AUC = {aurocs[fold_idx]:0.2f})")
        # Precision-Recall Curve
        precision, recall, prc_thresh = precision_recall_curve(fold["y"], fold["proba"].detach().cpu().numpy()[:, 1])
        prec_interp = np.interp(recall_mean, recall[::-1], precision[::-1])
        precisions.append(prec_interp)
        recall_threshes.append(prc_thresh[np.abs(recall-0.85).argmin()])
        auprcs.append(auc(recall, precision))
        ax.plot(recall, precision, label=f"PRC Fold {fold_idx + 1} (AUC = {auprcs[fold_idx]:0.2f})")
    
    # ax[0].set_xlabel("False Positive Rate")
    # ax[0].set_ylabel("True Positive Rate")
    # ax[0].set_title(f"Receiver Operating Characteristic (ROC) Curve for {model_name}")
    # ax[0].set_ylim(-0.1, 1.1)
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve for {model_name}")
    ax.set_ylim(-0.1, 1.1)
    
    # ROC
    # tpr_mean = np.mean(tprs, axis=0)
    # tpr_mean[-1] = 1.0
    # auroc_mean = auc(fpr_mean, tpr_mean)
    # auroc_std = np.std(aurocs)
    # ax[0].plot(
    #     fpr_mean, tpr_mean,
    #     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (auroc_mean, auroc_std),
    #     lw=2, alpha=.8
    # )
    # tpr_std = np.std(tprs, axis=0)
    # ax[0].fill_between(
    #     fpr_mean, np.maximum(tpr_mean - tpr_std, 0),
    #     np.minimum(tpr_mean + tpr_std, 1), alpha=.2,
    #     label=r"$\pm$ 1 std. dev.", color='grey'
    # )
    
    # PRC
    prec_mean = np.mean(precisions, axis=0)
    auprc_mean = auc(recall_mean, prec_mean)
    auprc_std = np.std(auprcs)
    ax.plot(
        recall_mean, prec_mean,
        label=r"Mean PRC (AUC = %0.2f $\pm$ %0.2f)" % (auprc_mean, auprc_std),
        lw=2, alpha=.8
    )
    prec_std = np.std(precisions, axis=0)
    ax.fill_between(
        recall_mean, np.maximum(prec_mean - prec_std, 0),
        np.minimum(prec_mean + prec_std, 1), alpha=.2,
        label=r"$\pm$ 1 std. dev.", color='grey'
    )
    
    fig.suptitle(f"ROC and PRC Curves for {model_name}")
    # ax[0].legend()
    ax.legend()
    plt.show()