import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from fastai.data.core import DataLoaders

from scripts.utils import *
import scripts.pp_cnf_matrix as ppcm


def main(
    model_dir="./models/prototyping6/tcn_fold_10/",
    num_iter=10,
    batch_size=2048,
    step_size=0.25,
    save_directory="./models/prototyping6/",
    use_cuda=True
):
    sns.set("paper", "whitegrid")
    test_set = ECGDataset("./data/heartbeats_evensplit_test/",
                          target_transform=label_encode)
    model = TCN(360, 5, [32]*9, 2, 0.125, use_skip_connections=True)
    model.load_state_dict(torch.load(f"{model_dir}/best.pth"))
    target_layer = model.tcn.network[-1].conv2
    model_outputs = iterate_perturbations(
        test_set, model, target_layer, num_iter, batch_size, step_size, save_directory, use_cuda=use_cuda)

    first_of_each_class = []
    for i in range(num_iter):
        for j in range(5):
            first_of_each_class.append(
                np.where(model_outputs[i]["y"] == j)[0][0]
            )
        break

    plot_classes(save_directory, first_of_each_class)

    plot_perturbation(save_directory)

    k_fold_roc_curve(model_outputs, "TCN with perturbed inputs",
                     legend_key="Perturbation iteration", show_mean_and_std=False)

    k_fold_roc_curve(model_outputs, "TCN with perturbed inputs",
                     legend_key="Perturbation iteration", show_mean_and_std=False, average="weighted")


def plot_classes(save_directory, first_of_each_class):
    x = np.arange(0, 1, 1/360)
    fig, ax = plt.subplots(5, 1, figsize=(8.27, 29.225), dpi=100)
    for perturb_iter in tqdm(range(10)):
        model_output = np.load(
            f"{save_directory}/perturbed_data_{perturb_iter}.npz")
        for i, signal_idx in enumerate(first_of_each_class):
            signal = model_output["inputs"][signal_idx]
            greyscale_cam = model_output["cams"][signal_idx]
            points = np.array([x, signal]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(greyscale_cam.min(), greyscale_cam.max())
            lc = LineCollection(segments, cmap='inferno', norm=norm)
            lc.set_array(greyscale_cam.T)
            line = ax[i].add_collection(lc)
            fig.colorbar(line, ax=ax[i])
            ax[i].set_ylim(
                min(ax[i].get_ylim()[0], min(signal)),
                max(ax[i].get_ylim()[1], max(signal))
            )
        break

    for i, classi in enumerate(["N", "S", "V", "F", "Q"]):
        ax[i].set_title(f"Class {classi}")
        ax[i].set_ylabel("Amplitude")
        ax[i].set_xlabel("Time (s)")


def plot_perturbation(save_directory, signal_idx=0):
    fig, ax = plt.subplots()
    x = np.arange(0, 1, 1/360)
    SIGNAL_IDX = 0
    for i in tqdm(range(10)):
        model_output = np.load(f"{save_directory}/perturbed_data_{i}.npz")
        signal = model_output["inputs"][SIGNAL_IDX]
        ax.plot(x, signal, alpha=.5, label=f"perturbation {i}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (z-normalised)")
    ax.set_title("Perturbed Signals")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainability Metrics")
    parser.add_argument("--model_dir", type=str,
                        default="./models/prototyping6/tcn_fold_10/", help="Directory of model to explain")
    parser.add_argument("--num_iter", type=int, default=10,
                        help="Number of iterations to perturb")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size for perturbation")
    parser.add_argument("--step_size", type=float, default=0.25,
                        help="Step size for perturbation towards mean")
    parser.add_argument("--save_directory", type=str,
                        default="./models/prototyping6/", help="Directory to save results")
    parser.add_argument("--use_cuda", type=bool, default=True,
                        help="Use CUDA for GPU acceleration")
    args = parser.parse_args()
    main(**vars(args))
