import sys
import os

import json
from typing import Callable, Tuple, Union, Optional, List, Dict

from TCN.TCN.tcn import TCN_DimensionalityReduced
from scripts.GradCAM1D import GradCAM as GradCAM1D

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, precision_recall_curve, roc_auc_score, precision_score
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
            self.X = torch.from_numpy(self.X)
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
            [IndexError | AssertionError]: if index is out of bounds

        Returns:
            ([numpy.ndarray | torch.Tensor], [str | torch.Tensor]): heartbeat signal and label
        """
        assert idx < self.__len__(), \
            f"index {idx} out of bounds for dataset of length {self.__len__()}"
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


class ECGInferenceSet(Dataset):
    """ Dataset class for inference.

    Args:
        data (numpy.ndarray): array of heartbeats with shape (num_samples, num_channels, signal_length)
        item_transform (Callable, optional): item transform. Defaults to None.
    """

    def __init__(self, data, item_transform) -> None:
        super().__init__()
        self.data = data
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)
        self.item_transform = item_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < self.__len__(), \
            f"index {idx} out of bounds for dataset of length {self.__len__()}"
        heartbeat = self.data[idx]
        heartbeat = self.item_transform(
            heartbeat) if self.item_transform else heartbeat
        return heartbeat


def one_hot_encode(label, classes=None):
    """Performs one-hot encoding on a label.

    Args:
        label (str): label to encode
        classes (list, optional): list of classes ordered by encoding index. Defaults to ["N", "S", "V", "F", "Q"].

    Returns:
        torch.Tensor: one-hot encoded label
    """
    if classes is none:
        classes = ["N", "S", "V", "F", "Q"]
    return torch.eye(len(classes))[classes.index(label)]


def one_hot_decode(label, classes=None):
    """Decodes a one-hot encoded label.

    Args:
        label (np.ndarray): one-hot encoded label with shape (num_classes, )
        classes (list, optional): target class names with order corresponding to the encoding index. Defaults to ["N", "S", "V", "F", "Q"].

    Returns:
        str: decoded label
    """
    if classes is None:
        classes = ["N", "S", "V", "F", "Q"]
    return classes[label.argmax()]


def noise_at_frequencies(
    hb: np.ndarray,
    frequencies_distribution=None,
    fs: int = 360
) -> np.ndarray:
    """Generates noise at specified frequencies and adds it to a heartbeat signal.

    Args:
        hb (numpy.ndarray): heartbeat signal
        frequencies_distribution (dict, optional): distributions of noise frequencies and amplitudes. The distribution is assumed to be uniform. Defaults to { "breathing": ([1/18, 1/12], [1/5, 1/3]) }.
        fs (int, optional): sampling rate. Defaults to 360.

    Returns:
        numpy.ndarray: augmented heartbeat signal

    See Also:
        noise_at_frequencies_tensor for the torch.Tensor version
    """
    noise = np.zeros(hb.shape)
    if frequencies_distribution is None:
        frequencies_distribution = {
            "breathing": ([1/18, 1/12], [1/5, 1/3])
        }
    for (_, (freq_range, amp_range)) in frequencies_distribution.items():
        freq = np.random.uniform(*freq_range)
        amp = np.random.uniform(*amp_range)
        phase = np.random.uniform(0, 2 * np.pi)
        noise += amp * np.sin(2 * np.pi * freq *
                              np.arange(hb.shape[0]) / fs + phase)
    return hb + noise


def noise_at_frequencies_tensor(
    hb: torch.Tensor,
    frequencies_distribution=None,
    fs: int = 360
) -> torch.Tensor:
    """Generates noise at specified frequencies and adds it to a heartbeat signal.

    Args:
        hb (torch.Tensor): heartbeat signal
        frequencies_distribution (dict, optional): distributions of noise frequencies and amplitudes.
            The distribution is assumed to be uniform.
            Defaults to { "breathing": ([1/18, 1/12], [1/5, 1/3]) }.
        fs (int, optional): sampling rate. Defaults to 360.

    Returns:
        torch.tensor: augmented heartbeat signal
    """
    noise = torch.zeros(hb.shape)
    if frequencies_distribution is None:
        frequencies_distribution = {
            "breathing": ([1/18, 1/12], [1/5, 1/3])
        }
    for (source, (freq_range, amp_range)) in frequencies_distribution.items():
        freq = torch.distributions.uniform.Uniform(*freq_range).sample()
        amp = torch.distributions.uniform.Uniform(*amp_range).sample()
        phase = torch.distributions.uniform.Uniform(0, 2 * np.pi).sample()
        noise += amp * torch.sin(2 * np.pi * freq *
                                 torch.arange(hb.shape[0]) / fs + phase)
    return hb + noise


def z_normalise(heartbeat: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Normalises a heartbeat signal to zero mean and unit variance.

    Args:
        heartbeat (np.ndarray): heartbeat signal

    Returns:
        np.ndarray: normalised heartbeat signal
    """
    return (heartbeat - heartbeat.mean()) / heartbeat.std()


def hb_transform(
    heartbeat: Union[np.ndarray, torch.Tensor],
    add_noise: bool = True
) -> torch.Tensor:
    """Transforms a heartbeat signal.

    Args:
        heartbeat ([np.ndarray | torch.Tensor]): heartbeat signal
        input_is_tensor (bool, optional): whether the input is a tensor. Defaults to False.

    Returns:
        torch.tensor: transformed heartbeat signal
    """
    if isinstance(heartbeat, np.ndarray):
        heartbeat = torch.from_numpy(heartbeat)
    if add_noise:
        heartbeat = noise_at_frequencies_tensor(heartbeat)
    heartbeat = z_normalise(heartbeat)
    return heartbeat


def label_encode(
    label: str,
    classes: Optional[List] = None
) -> int:
    """Encodes a label as an integer.

    Args:
        label (str): label to encode
        classes (list, optional): class list. Defaults to ["N", "S", "V", "F", "Q"].

    Raises:
        ValueError: if label is not in classes

    Returns:
        int: encoded label
    """
    if classes is None:
        classes = ["N", "S", "V", "F", "Q"]
    if not label in classes:
        raise ValueError(f"label {label} not in classes {classes}")
    return classes.index(label)


class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) with dimensionality reduction.

    Args:
        input_size (int): The length of the input vector.
        output_size (int): The length of the output vector.
        num_channels (List[int]): A list of integers, where each integer
        is the number of channels in a convolutional layer.
        kernel_size (int): The size of the kernel in each convolutional layer.
        dropout (float): The dropout rate to use in each convolutional layer.
        use_skip_connections (bool): Whether to use skip connections between convolutional layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: List[int],
        kernel_size: int,
        dropout: float,
        use_skip_connections: bool = False
    ) -> None:
        super().__init__()
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
        """Initialises the weights of the linear layer.
        """
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TCN.

        Args:
            x (torch.Tensor): input tensor with shape (batch_size, num_channels, signal_length)

        Returns:
            torch.Tensor: output tensor with shape (batch_size, num_classes)
        """
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])
        return o


class LogInterruptable(Callback):
    """Callback for logging model training progress to a file, for use when resuming training from an interrupted state.
    Args:
        filename (str): The filename to save the interrupt information to.
    """

    def __init__(self, filename: str, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.interrupt_info = json.load(f)
        else:
            self.interrupt_info = {
                "fold": 0,
                "epoch": 0
            }
            with open(self.filename, "w") as f:
                json.dump(self.interrupt_info, f, indent=4)

    def after_epoch(self):
        self.interrupt_info["epoch"] = self.epoch
        with open(self.filename, "w") as f:
            json.dump(self.interrupt_info, f, indent=4)

    def after_fit(self):
        self.interrupt_info["fold"] += 1
        self.interrupt_info["epoch"] = 0
        with open(self.filename, "w") as f:
            json.dump(self.interrupt_info, f, indent=4)

def fastai_precision_score(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
    return skm_to_fastai(precision_score, axis=axis, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight, zero_division=zero_division)

def k_fold_inference(
    model: nn.Module,
    test_dl: DataLoader,
    model_dir: str = "./prototyping/tcn_fold_",
    weights_fn: str = "best",
    k: int = 10,
    target_names: Optional[List[str]] = None
) -> Tuple[List[dict], List[dict]]:
    """Performs inference on a k-fold model

    Args:
        model (torch.nn.Module): the model to perform inference on
        test_dl (torch.utils.data.DataLoader): dataloader for the test set
        model_dir (str, optional): directory for all the model weights for each fold. Defaults to "./prototyping/tcn_fold_".
        weights_fn (str, optional): filename for weights file, .pth file suffix unnecessary. Defaults to "best".
        k (int, optional): number of folds. Defaults to 10.
        target_names (List[str], optional): list of target names. Defaults to ["N", "S", "V", "F", "Q"].

    Returns:
        (list, list): model_outputs, classification_reports for each fold
        model_outputs: list of dicts with keys "preds", "proba", and "y"
        classification_reports: list of classification reports for each fold
    """
    if target_names is None:
        target_names = ["N", "S", "V", "F", "Q"]
    model_outputs = []
    classification_reports = []
    for fold in range(k):
        if torch.cuda.is_available():
            model = model.cuda()
            cbs = [MixedPrecision()]
        else:
            cbs = None
        learner = Learner(model=model, dls=test_dl,
                        loss_func=nn.CrossEntropyLoss(), cbs=cbs)
        print(model_dir + f"{fold + 1}" + "/" + weights_fn)
        learner.load(model_dir + f"{fold + 1}" + "/" + weights_fn)
        proba, y = learner.get_preds(dl=test_dl)
        preds = proba.argmax(dim=1)
        classification_reports.append(
            classification_report(
                y.cpu().numpy(),
                preds,
                output_dict=True,
                target_names=target_names
            )
        )
        model_outputs.append({
            "preds": preds.detach().cpu().numpy(),
            "proba": proba.detach().cpu().numpy(),
            "y": y.cpu().numpy(),
        })
    return model_outputs, classification_reports


def k_fold_roc_curve(
    model_outputs: Dict[str, np.ndarray],
    model_name: str,
    num_classes: int = 5,
    average: str = "macro",
    legend_key: str = "Fold",
    show_mean_and_std: bool = True
) -> None:
    """Plots ROC and PRC curves for a k-fold model.

    Args:
        model_outputs (dict): model outputs from k_fold_inference
        model_name (str): name of the model to display on the plot.
        num_classes (int, optional): number of classification targets. Defaults to 5.
        average (str, optional): defines the method for calculating averages for multi-class targets. Can be ("macro" | "weighted") Defaults to "macro".
        legend_key (str, optional): name of the type of iteration to display on the plot. Defaults to "Fold".
        show_mean_and_std (bool, optional): shows the mean and 1 std dev of the ROC curve and PRC. Defaults to True.

    Raises:
        NotImplementedError: if average is not ("macro" | "weighted")
    """
    fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69), dpi=100)
    tprs, aurocs, tpr_threshes = [], [], []
    fpr_mean = np.linspace(0, 1, 1000)
    precisions, auprcs, recall_threshes = [], [], []
    recall_mean = np.linspace(0, 1, 1000)
    for fold_idx, fold in enumerate(tqdm(model_outputs)):
        roc_label_binarizer = LabelBinarizer().fit(fold["y"])
        y_onehot_test = roc_label_binarizer.transform(fold["y"])
        intermediate_tprs, intermediate_tpr_threshes, intermediate_aurocs = [], [], []
        intermediate_precisions, intermediate_recall_threshes, intermediate_auprcs = [], [], []
        for i in range(num_classes):
            if i not in fold["y"]:
                continue
            # ROC Curve
            fpr, tpr, tpr_thresh = sk_roc_curve(
                y_onehot_test[:, i], fold['proba'][:, i])
            intermediate_tpr_threshes.append(
                tpr_thresh[np.abs(tpr-0.85).argmin()])
            tpr_interp = np.interp(fpr_mean, fpr, tpr)
            tpr_interp[0] = 0.0
            intermediate_tprs.append(tpr_interp)
            intermediate_aurocs.append(auc(fpr, tpr))
            # PRC
            precision, recall, prc_thresh = precision_recall_curve(
                y_onehot_test[:, i], fold['proba'][:, i])
            prec_interp = np.interp(recall_mean, recall[::-1], precision[::-1])
            intermediate_precisions.append(prec_interp)
            intermediate_recall_threshes.append(
                prc_thresh[np.abs(recall-0.85).argmin()])
            intermediate_auprcs.append(auc(recall, precision))
        if average == "macro":
            tprs.append(np.mean(intermediate_tprs, axis=0))
            aurocs.append(np.mean(intermediate_aurocs))
            tpr_threshes.append(np.mean(intermediate_tpr_threshes))
            precisions.append(np.mean(intermediate_precisions, axis=0))
            auprcs.append(auc(recall_mean, precisions[-1]))
            recall_threshes.append(np.mean(intermediate_recall_threshes))
        elif average == "weighted":
            class_distributions = np.bincount(fold["y"])
            if len(class_distributions) < num_classes:
                class_distributions = np.append(class_distributions, np.zeros(
                    num_classes - len(class_distributions)))
            # normalize the class distributions
            class_distributions = class_distributions / \
                np.sum(class_distributions)
            aurocs.append(np.array(intermediate_aurocs).T @
                          class_distributions.reshape(-1, 1))
            tprs.append(np.array(intermediate_tprs).T @
                        class_distributions.reshape(-1, 1))
            precisions.append(np.array(intermediate_precisions).T @
                              class_distributions.reshape(-1, 1))
            auprcs.append(np.array(intermediate_auprcs).T @
                          class_distributions.reshape(-1, 1))
            aurocs[-1] = aurocs[-1][0]  # unpack
            auprcs[-1] = auprcs[-1][0]
            tprs[-1] = tprs[-1].reshape(-1)
            precisions[-1] = precisions[-1].reshape(-1)
        else:
            raise NotImplementedError
        ax[0].plot(fpr_mean, tprs[-1],
                   label=f"ROC {legend_key} {fold_idx + 1} (AUC = {aurocs[fold_idx]:0.2f})", alpha=.3)
        # Precision-Recall Curve
        ax[1].plot(recall_mean, precisions[-1],
                   label=f"PRC {legend_key} {fold_idx + 1} (AUC = {auprcs[fold_idx]:0.2f})", alpha=.3)

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
    if show_mean_and_std:
        tpr_mean = np.mean(tprs, axis=0)
        tpr_mean[-1] = 1.0
        auroc_mean = auc(fpr_mean, tpr_mean)
        auroc_std = np.std(aurocs)
        ax[0].plot(
            fpr_mean, tpr_mean,
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (
                auroc_mean, auroc_std),
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
            label=r"Mean PRC (AUC = %0.2f $\pm$ %0.2f)" % (
                auprc_mean, auprc_std),
            lw=2, alpha=.8
        )
        prec_std = np.std(precisions, axis=0)
        ax[1].fill_between(
            recall_mean, np.maximum(prec_mean - prec_std, 0),
            np.minimum(prec_mean + prec_std, 1), alpha=.2,
            label=r"$\pm$ 1 std. dev.", color='grey'
        )

    fig.suptitle(f"ROC and PRC Curves for {model_name}, average={average}")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def perturb_to_mean(
    batched_inputs: torch.Tensor,
    batched_cam: np.ndarray,
    step_size: float = 0.25
) -> np.ndarray:
    """Perturbs the signal to the mean based on the GradCAM saliency map and step size.

    Args:
        batched_inputs (torch.Tensor): batched inputs with shape (batch_size, num_channels, signal_length)
        batched_cam (numpy.ndarray): batched GradCAM saliency map with shape (batch_size, signal_length)
        step_size (float, optional): how much the signal is augmented towards the mean. Defaults to 0.25.

    Returns:
        numpy.ndarray: batched augmented signal
    """
    mean = batched_inputs.mean(dim=1)
    perturbed_signal = (batched_inputs - mean.repeat(360, 1).T).numpy()
    diff = batched_cam * perturbed_signal
    diff *= step_size
    perturbed_signal = perturbed_signal + mean.repeat(360, 1).T.numpy() - diff
    return perturbed_signal


def iterate_perturbations(
    dataset: ECGDataset,
    model: nn.Module,
    target_layer: nn.Module,
    num_iter: int = 10,
    batch_size: int = 2048,
    step_size: float = 0.25,
    save_directory: str = "./models/",
    save_file_str_format: str = "perturbed_data_{idx}",
    num_workers: int = 0,
    use_cuda: bool = True
) -> Dict[str, np.ndarray]:
    """Augments the signal based on the GradCAM saliency map.

    Args:
        dataset (ECGDataset): ECG Dataset with in_memory set to True
        model (torch.nn.Module): Pytorch model
        target_layer (torch.nn.Module): target layers for GradCAM
        num_iter (int, optional): number of perturbation iterations. Defaults to 10.
        batch_size (int, optional): mini batch size for inference. Defaults to 2048.
        step_size (float, optional): how much of a step the augmentation takes towards the signal mean. Defaults to 0.25.
        save_directory (str, optional): directory to save model outputs to. Defaults to "./models/".
        save_file_str_format (str, optional): format for model outputs save file. `idx` represents the iteration number starting from 0. Defaults to "perturbed_data_{idx}".
        num_workers (int, optional): number of workers for the dataloader. Note that persistent_workers is always set to False due to the modification of the dataset object. Defaults to 0.
        use_cuda (bool, optional): whether to use CUDA. Defaults to True.

    Returns:
        dict: model outputs
    """
    model_outputs = []
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=num_workers)
    with GradCAM1D(model=model, target_layer=target_layer, use_cuda=use_cuda) as cam:
        for idx in tqdm(range(num_iter), position=0, leave=True):
            ys, preds, inputs, outputs, cams = [], [], [], [], []
            for i, (in_tensor, target_tensor) in enumerate(tqdm(loader, position=1, leave=False)):
                greyscale_cam = cam(in_tensor, target_tensor)
                perturbed_data = perturb_to_mean(
                    in_tensor.detach(), greyscale_cam, step_size=step_size)
                end_idx = min((i + 1) * batch_size, len(dataset.X))
                dataset.X[i *
                          batch_size:end_idx] = torch.from_numpy(perturbed_data)
                with torch.no_grad():  # get model outputs on perturbed data from the previous iteration
                    output = model(in_tensor.cuda()).cpu().detach()
                    proba = nn.functional.softmax(output, dim=1)
                    pred = proba.argmax(dim=1)
                    inputs.append(in_tensor.detach().numpy())
                    ys.append(target_tensor.numpy())
                    preds.append(pred.numpy())
                    outputs.append(output.numpy())
                    cams.append(greyscale_cam)
                torch.cuda.empty_cache()
            ys = np.concatenate(ys)
            preds = np.concatenate(preds)
            outputs = np.concatenate(outputs)
            inputs = np.concatenate(inputs)
            cams = np.concatenate(cams)
            if save_file_str_format is not None:
                np.savez_compressed(
                    os.path.join(save_directory,
                                 save_file_str_format.format(idx=idx)),
                    inputs=inputs,
                    y=ys,
                    preds=preds,
                    proba=outputs,
                    cams=cams
                )
                model_outputs.append({
                    "proba": outputs,
                    "y": ys
                })
            else:
                model_outputs.append({
                    "preds": preds,
                    "proba": outputs,
                    "y": ys,
                    "inputs": inputs,
                    "cams": cams
                })
    # loader._iterator._shutdown_workers()
    return model_outputs
