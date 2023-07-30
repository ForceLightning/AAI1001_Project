import os
import shutil
import argparse

import numpy as np
import wfdb
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split


def main(
    mitdb_path="./data/mit-bih-arrhythmia-database-1.0.0",
    split_ratio=0.5
):
    """Preprocesses the MIT-BIH Arrhythmia Database

    Args:
        mitdb_path (str, optional): Path to MIT-BIH Arrhythmia Database. Defaults to "./data/mit-bih-arrhythmia-database-1.0.0".
        split_ratio (float, optional): Test dataset split ratio for train_test_split. Defaults to 0.5.
    """

    ansi_map = {
        ".": "N",
        "N": "N",
        "L": "N",
        "R": "N",
        "e": "N",
        "j": "N",
        "A": "S",
        "J": "S",
        "a": "S",
        "S": "S",
        "E": "V",
        "V": "V",
        "F": "F",
        "/": "Q",
        "f": "Q",
        "Q": "Q"
    }

    with open(f'{mitdb_path}/RECORDS', "r") as f:
        records = f.read().split("\n")

    heartbeat_data, heartbeat_labels = [], []

    for record_id in tqdm(records):
        record = wfdb.rdrecord(
            mitdb_path + record_id)
        annotation = wfdb.rdann(
            mitdb_path + record_id, "atr")
        heartbeats, labels = [], []  # heartbeats and labels for this record
        for (idx, hb_class) in zip(annotation.sample, annotation.symbol):
            # Get +- 180 samples around the heartbeat
            if not hb_class in ansi_map:
                continue
            try:
                # 360 samples, use only 1 channel
                heartbeat = record.p_signal[idx-180:idx+180, 0]
            except Exception:
                continue
            if len(heartbeat) != 360:
                continue
            heartbeats.append(heartbeat)
            labels.append(ansi_map[hb_class])
        heartbeat_data.append(heartbeats)
        heartbeat_labels.append(labels)

    if not os.path.exists("./data/heartbeats/"):
        os.mkdir("./data/heartbeats/")

    for (record_id, heartbeats, labels) in tqdm(zip(records, heartbeat_data, heartbeat_labels)):
        np.savez_compressed("./data/heartbeats/" + record_id,
                            heartbeats=heartbeats, labels=labels)

    train_records, test_records = train_test_split(
        records, test_size=split_ratio, random_state=42)

    if not os.path.exists("./data/heartbeats_evensplit_train/"):
        os.makedirs("./data/heartbeats_evensplit_train/")

    if not os.path.exists("./data/heartbeats_evensplit_test/"):
        os.makedirs("./data/heartbeats_evensplit_test/")

    for record_id in tqdm(records, desc="Copying files"):
        if record_id in test_records:
            # copy to test folder
            shutil.copy("./data/heartbeats/" + record_id + ".npz",
                        "./data/heartbeats_evensplit_test/" + record_id + ".npz")
        else:
            # copy to train folder
            shutil.copy("./data/heartbeats/" + record_id + ".npz",
                        "./data/heartbeats_evensplit_train/" + record_id + ".npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses the MIT-BIH Arrhythmia Database")
    parser.add_argument("--mitdb_path", type=str,
                        default="./data/mit-bih-arrhythmia-database-1.0.0/", help="Path to MIT-BIH Arrhythmia Database")
    parser.add_argument("--split_ratio", type=float, default=0.5,
                        help="Test dataset split ratio for train_test_split")
    args = parser.parse_args()
    main(**vars(args))
