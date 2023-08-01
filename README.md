# AAI1001 Project

This repository serves as an archive for the AAI1001 project.

## Requirements
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- A suitable NVIDIA GPU with CUDA compute capability >= 3.5
- The [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). Folder structure should look like `./data/mit-bih-arrhythmia-database-1.0.0/`


## Installation
1. Start by cloning the repository:  
`git clone -c core.symlinks=true git@github.com:ForceLightning/AAI1001_Project.git --recursive`
2. Ensure that the MIT-BIH Arrhythmia Database is in the `./data/` folder as mentioned above.
3. Install dependencies from `requirements.txt` with `pip install -r requirements.txt`. Modify the `torch` version if necessary.

## Usage
### Preprocessing
1. Ensure that the MIT-BIH Arrhythmia Database is in the `./data/` folder as mentioned above.
2. Available arguments are:
```sh
usage: preprocessing.py [-h] [--mitdb_path MITDB_PATH] [--split_ratio SPLIT_RATIO]

Preprocesses the MIT-BIH Arrhythmia Database

optional arguments:
  -h, --help            show this help message and exit
  --mitdb_path MITDB_PATH
                        Path to MIT-BIH Arrhythmia Database
  --split_ratio SPLIT_RATIO
                        Test dataset split ratio for train_test_split
```

### Training
Available arguments are:
```sh
usage: training.py [-h] [--batch_size BATCH_SIZE] [--shuffle_train SHUFFLE_TRAIN] [--pin_memory PIN_MEMORY]
                   [--num_workers NUM_WORKERS] [--persistent_workers PERSISTENT_WORKERS] [--model_dir MODEL_DIR]
                   [--num_splits NUM_SPLITS]

Train TCN model

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size
  --shuffle_train SHUFFLE_TRAIN
                        Shuffle train set
  --pin_memory PIN_MEMORY
                        Pin memory
  --num_workers NUM_WORKERS
                        Number of workers
  --persistent_workers PERSISTENT_WORKERS
                        Persistent workers
  --model_dir MODEL_DIR
                        Model directory
  --num_splits NUM_SPLITS
                        Number of splits
```

### Testing
Available arguments are:
```sh
usage: testing.py [-h] [--batch_size BATCH_SIZE] [--pin_memory PIN_MEMORY] [--model_dir MODEL_DIR]

Model testing metrics

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size
  --pin_memory PIN_MEMORY
                        Pin memory
  --model_dir MODEL_DIR
                        Model directory without the parent directory (e.g. './prototyping6/tcn_fold_')
```

### Explainability Metrics
Available arguments are:
```sh
usage: explainability.py [-h] [--model_dir MODEL_DIR] [--num_iter NUM_ITER] [--batch_size BATCH_SIZE]
                         [--step_size STEP_SIZE] [--save_directory SAVE_DIRECTORY] [--use_cuda USE_CUDA]

Explainability Metrics

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory of model to explain
  --num_iter NUM_ITER   Number of iterations to perturb
  --batch_size BATCH_SIZE
                        Batch size for perturbation
  --step_size STEP_SIZE
                        Step size for perturbation towards mean
  --save_directory SAVE_DIRECTORY
                        Directory to save results
  --use_cuda USE_CUDA   Use CUDA for GPU acceleration
```

### GUI
```sh
python user_interface.py
```

## Measurements
Measurements are taken from the `testing.py` and `explainability.py` scripts. Firstly, classification metrics like accuracy, positive predictive value, sensitivity, specificity, f1-score, AUROC, and AUPRC are measured. Since ROC plots do not accurately reflect classification performance for unbalanced datasets, AUPRC would be more representative of the model's performance. We include both due to the precedence set by prior works.

For explainability, the model inputs are perturbed towards the mean by multiplying the Grad-CAM score by the distance the signal is from the mean. This is repeated 10 times to show that when the portions of the signal that are important to the prediction are perturbed away from normal values, the classification performance of the model will degrade.

## Contributors
All authors contributed equally.

## License

[MIT](https://choosealicense.com/licenses/mit/)