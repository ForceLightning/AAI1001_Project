import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix

from scripts.utils import *
import scripts.pp_cnf_matrix as ppcm


def main(
    batch_size=4096,
    pin_memory=True,
    model_dir="./prototyping6/tcn_fold_"
):
    sns.set_style("whitegrid")
    sns.set_context("paper")

    test_set = ECGDataset("./data/heartbeats_evensplit_test/",
                          target_transform=label_encode)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    model = TCN(360, 5, [32]*9, 2, 0.125, use_skip_connections=True)
    model_outputs, classification_reports = k_fold_inference(
        model, test_loader, model_dir=model_dir)

    # ROC and PRC curves
    k_fold_roc_curve(model_outputs, "TCN")
    k_fold_roc_curve(model_outputs, "TCN", average="weighted")

    # Classification reports
    for report in classification_reports:
        print(pd.DataFrame(report))

    # Confusion matrices
    for fold in model_outputs:
        cnf_matrix = confusion_matrix(
            fold["y"],
            fold["preds"]
        )
        cnf_matrix = pd.DataFrame(
            cnf_matrix, columns=["N", "S", "V", "F", "Q"])
        ppcm.pretty_plot_confusion_matrix(cnf_matrix)

    # training graphs
    fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69), dpi=100, sharex=True)
    for i in tqdm(range(10)):
        plot_df = pd.read_csv(f'{model_dir}_{i+1}/log.csv')
        ax[0].plot(plot_df["train_loss"], label=f"Fold {i+1}", alpha=0.5)
        ax[1].plot(plot_df["valid_loss"], label=f"Fold {i+1}", alpha=0.5)
    ax[0].set_title("Training Loss")
    ax[1].set_title("Validation Loss")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Training and Validation Loss vs Epoch")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing metrics")
    parser.add_argument("--batch_size", type=int,
                        default=4096, help="Batch size")
    parser.add_argument("--pin_memory", type=bool,
                        default=True, help="Pin memory")
    parser.add_argument("--model_dir", type=str, default="./prototyping6/tcn_fold_",
                        help=f"Model directory without the parent directory (e.g. './prototyping6/tcn_fold_')")
    args = parser.parse_args()
    main(**vars(args))
