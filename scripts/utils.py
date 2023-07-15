from TCN.TCN.tcn import TCN_DimensionalityReduced
import sys
import os

import json
import signal
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, precision_recall_curve, roc_auc_score
from sklearn.metrics import roc_curve as sk_roc_curve
from sklearn.preprocessing import LabelBinarizer
from scipy.interpolate import interp1d

from fastai.learner import Learner
from fastai.metrics import accuracy, Precision, Recall, F1Score, RocAuc
from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
from fastai.metrics import skm_to_fastai

sys.path.append("../" + os.path.dirname(os.path.realpath(__file__)))


class ECGDataset(Dataset):
    """ECG Dataset class.

    Args:
        data_dir (str): path to data directory
        item_transform (Callable, optional): item transform. Defaults to None.
        target_transform (Callable, optional): target transform. Defaults to None.
        in_memory (bool, optional): whether to load all data into memory. Defaults to True.
    """
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
        """Returns length of dataset.

        Returns:
            int: dataset length
        """
        return sum(self.lengths.values())

    def __getitem__(self, idx):
        """Gets item at index idx.

        Args:
            idx (int): index

        Raises:
            IndexError: if index is out of bounds

        Returns:
            (np.ndarray, str): heartbeat signal and label
        """
        assert idx < self.__len__()  # make sure we're not out of bounds
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
            label = self.target_transform(
                label) if self.target_transform else label
            return hb, label
        # should never get here
        raise IndexError("Index out of bounds")

    def get_labels(self):
        return self.y


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
        noise += amp * np.sin(2 * np.pi * freq *
                              np.arange(hb.shape[0]) / fs + phase)
    return hb + noise


def noise_at_frequencies_tensor(
    hb: torch.Tensor,
    frequencies_distribution={
        "breathing": ([1/18, 1/12], [1/5, 1/3])
    },
    fs=360
):
    """Generates noise at specified frequencies and adds it to a heartbeat signal.

    Args:
        hb (torch.Tensor): heartbeat signal
        frequencies_distribution (dict, optional): distributions of noise frequencies and amplitudes. Defaults to { "breathing": ([1/18, 1/12], [1/5, 1/3]) }.
        fs (int, optional): sampling rate. Defaults to 360.

    Returns:
        torch.tensor: augmented heartbeat signal
    """
    noise = torch.zeros(hb.shape)
    for (source, (freq_range, amp_range)) in frequencies_distribution.items():
        freq = torch.distributions.uniform.Uniform(*freq_range).sample()
        amp = torch.distributions.uniform.Uniform(*amp_range).sample()
        phase = torch.distributions.uniform.Uniform(0, 2 * np.pi).sample()
        noise += amp * torch.sin(2 * np.pi * freq *
                                 torch.arange(hb.shape[0]) / fs + phase)
    return hb + noise


def z_normalise(hb):
    """Normalises a heartbeat signal to zero mean and unit variance.

    Args:
        hb (np.ndarray): heartbeat signal

    Returns:
        np.ndarray: normalised heartbeat signal
    """
    return (hb - hb.mean()) / hb.std()


def hb_transform(hb):
    """Transforms a heartbeat signal.

    Args:
        hb (np.ndarray): heartbeat signal

    Returns:
        torch.tensor: transformed heartbeat signal
    """
    hb = torch.from_numpy(hb)
    hb = noise_at_frequencies_tensor(hb)
    hb = z_normalise(hb)
    return hb


def label_encode(label, classes=["N", "S", "V", "F", "Q"]):
    """Encodes a label as an integer.

    Args:
        label (str): label to encode
        classes (list, optional): class list. Defaults to ["N", "S", "V", "F", "Q"].

    Returns:
        int: encoded label
    """
    return classes.index(label)


class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) with dimensionality reduction.

    Args:
        input_size (int): The length of the input vector.
        output_size (int): The length of the output vector.
        num_channels (list): A list of integers, where each integer is the number of channels in a convolutional layer.
        kernel_size (int): The size of the kernel in each convolutional layer.
        dropout (float): The dropout rate to use in each convolutional layer.
        use_skip_connections (bool): Whether to use skip connections between convolutional layers.
    """
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
    """Callback for logging model training progress to a file, for use when resuming training from an interrupted state.
    Args:
        fn (str): The filename to save the interrupt information to.
    """
    def __init__(self, fn: str, **kwargs):
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


def k_fold_inference(model, test_dl, model_dir="./prototyping/tcn_fold_", weights_fn="best", k=10):
    """Performs inference on a k-fold model

    Args:
        model (nn.Module): the model to perform inference on
        test_dl (torch.utils.data.DataLoader): dataloader for the test set
        model_dir (str, optional): directory for all the model weights for each fold. Defaults to "./prototyping/tcn_fold_".
        weights_fn (str, optional): filename for weights file, .pth file suffix unnecessary. Defaults to "best".
        k (int, optional): number of folds. Defaults to 10.

    Returns:
        (list, list): model_outputs, classification_reports for each fold
        model_outputs: list of dicts with keys "preds", "proba", and "y"
        classification_reports: list of classification reports for each fold
    """
    model_outputs = []
    classification_reports = []
    for fold in range(k):
        learner = Learner(model=model.cuda(), dls=test_dl,
                          loss_func=nn.CrossEntropyLoss(), cbs=[MixedPrecision()])
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


def k_fold_roc_curve(model_outputs, model_name: str, num_classes=5, average="macro"):
    fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69), dpi=100)
    tprs, aurocs, tpr_threshes = [], [], []
    fpr_mean = np.linspace(0, 1, 1000)
    precisions, auprcs, recall_threshes = [], [], []
    recall_mean = np.linspace(0, 1, 1000)
    for fold_idx, fold in enumerate(tqdm(model_outputs)):
        roc_label_binarizer = LabelBinarizer().fit(fold["y"])
        y_onehot_test = roc_label_binarizer.transform(fold["y"])
        intermediate_tprs, intermediate_tpr_threshes = [], []
        intermediate_precisions, intermediate_recall_threshes, intermediate_auprcs = [], [], []
        for i in range(num_classes):
            if i not in fold["y"]:
                continue
            # ROC Curve
            fpr, tpr, tpr_thresh = sk_roc_curve(
                y_onehot_test[:, i], fold['proba'].detach().cpu().numpy()[:, i])
            intermediate_tpr_threshes.append(
                tpr_thresh[np.abs(tpr-0.85).argmin()])
            tpr_interp = np.interp(fpr_mean, fpr, tpr)
            tpr_interp[0] = 0.0
            intermediate_tprs.append(tpr_interp)
            # PRC
            precision, recall, prc_thresh = precision_recall_curve(
                y_onehot_test[:, i], fold['proba'].detach().cpu().numpy()[:, i])
            prec_interp = np.interp(recall_mean, recall[::-1], precision[::-1])
            intermediate_precisions.append(prec_interp)
            intermediate_recall_threshes.append(
                prc_thresh[np.abs(recall-0.85).argmin()])
            intermediate_auprcs.append(auc(recall, precision))
        if average == "macro":
            tprs.append(np.mean(intermediate_tprs, axis=0))
            aurocs.append(roc_auc_score(y_onehot_test, fold["proba"].detach().cpu().numpy(
            ), average="macro", multi_class="ovr", labels=list(set(fold["y"].numpy().tolist()))))
            tpr_threshes.append(np.mean(intermediate_tpr_threshes))
            precisions.append(np.mean(intermediate_precisions, axis=0))
            auprcs.append(auc(recall_mean, precisions[-1]))
            recall_threshes.append(np.mean(intermediate_recall_threshes))
        elif average == "weighted":
            class_distributions = np.bincount(fold["y"])
            aurocs.append(roc_auc_score(
                y_onehot_test,
                fold["proba"].detach().cpu().numpy(),
                average="weighted",
                multi_class="ovr"
            ))
            if len(class_distributions) < num_classes:
                class_distributions = np.append(class_distributions, np.zeros(
                    num_classes - len(class_distributions)))
            # normalize the class distributions
            class_distributions = class_distributions / \
                np.sum(class_distributions)
            tprs.append(np.array(intermediate_tprs).T @
                        class_distributions.reshape(-1, 1))
            precisions.append(np.array(intermediate_precisions).T @
                              class_distributions.reshape(-1, 1))
            auprcs.append(np.array(intermediate_auprcs).T @
                          class_distributions.reshape(-1, 1))
            tprs[-1] = tprs[-1].reshape(-1)
            precisions[-1] = precisions[-1].reshape(-1)
            auprcs[-1] = auprcs[-1][0]
        else:
            raise NotImplementedError
        ax[0].plot(fpr_mean, tprs[-1],
                   label=f"ROC Fold {fold_idx + 1} (AUC = {aurocs[fold_idx]:0.2f})", alpha=.3)
        # Precision-Recall Curve
        ax[1].plot(recall_mean, precisions[-1],
                   label=f"PRC Fold {fold_idx + 1} (AUC = {auprcs[fold_idx]:0.2f})", alpha=.3)

    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(
        f"Receiver Operating Characteristic (ROC) Curve for {model_name}")
    ax[0].set_ylim(-0.1, 1.1)

    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Precision-Recall Curve for {model_name}")
    ax[1].set_ylim(-0.1, 1.1)

    # ROC
    tpr_mean = np.mean(tprs, axis=0)
    tpr_mean[-1] = 1.0
    auroc_mean = auc(fpr_mean, tpr_mean)
    auroc_std = np.std(aurocs)
    ax[0].plot(
        fpr_mean, tpr_mean,
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (auroc_mean, auroc_std),
        lw=2, alpha=.8
    )
    tpr_std = np.std(tprs, axis=0)
    ax[0].fill_between(
        fpr_mean, np.maximum(tpr_mean - tpr_std, 0),
        np.minimum(tpr_mean + tpr_std, 1), alpha=.2,
        label=r"$\pm$ 1 std. dev.", color='grey'
    )

    # PRC
    prec_mean = np.mean(precisions, axis=0)
    auprc_mean = auc(recall_mean, prec_mean)
    auprc_std = np.std(auprcs)
    ax[1].plot(
        recall_mean, prec_mean,
        label=r"Mean PRC (AUC = %0.2f $\pm$ %0.2f)" % (auprc_mean, auprc_std),
        lw=2, alpha=.8
    )
    prec_std = np.std(precisions, axis=0)
    ax[1].fill_between(
        recall_mean, np.maximum(prec_mean - prec_std, 0),
        np.minimum(prec_mean + prec_std, 1), alpha=.2,
        label=r"$\pm$ 1 std. dev.", color='grey'
    )

    fig.suptitle(f"ROC and PRC Curves for {model_name}")
    ax[0].legend()
    ax[1].legend()
    plt.show()
