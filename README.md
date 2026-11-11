# ESdB: Embeddings for Sequential Data Benchmark

Welcome to ESdB, the repository accompanying **"Towards a Universal Sequence Embedding: A Multi-Domain, Multi-Task Benchmark and Evaluation of Self-Supervised Methods."**

ESdB is a benchmark for evaluating self-supervised sequence embeddings across multiple downstream tasks and data domains. It brings 12 source datasets into a common event-sequence representation and provides targets for classification, regression, forecasting, and anomaly detection.

## Code and benchmark annotations

This repository provides the preprocessing, model implementations, and evaluation code for ESdB.

The [Hugging Face release](https://huggingface.co/datasets/On-Point-Rnd/ESdB-Embeddings-for-Sequential-data-Benchmark) contains benchmark annotations for all 12 datasets: source identifiers, downstream targets, evaluation shifts, and train/test assignments. It does not contain the original events or input features.

Obtain the original data from the sources listed below and follow their access and usage terms. The annotations and preprocessing code define how these sources are used in the benchmark.

## Reproducing experiments

The workflow is **source data → preprocessing → SSL pretraining → embedding extraction → downstream evaluation**. Run all commands from the repository root.

### 1. Set up the environment and obtain the source data

Use Python 3.10 or later, PyTorch suitable for your hardware, and Java for PySpark preprocessing. Install the project and its dependencies in your environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pyrallis pyarrow scikit-learn
```

PyTorch must be installed separately for your CPU/CUDA environment. Download the source dataset using the links below. For the AGE example, place `transactions_train.csv` and `train_target.csv` in `data/age/data/`.

### 2. Preprocess the source data

The [dataset preprocessing guide](preprocess/with_shifts/README.md) documents input/output paths, dataset-specific arguments, and target names. For AGE:

```bash
python -m preprocess.with_shifts.age \
    --data-path data/age/data \
    --save-path data/age/preprocessed_full \
    --cat-codes-path data/age/preprocessed_full/cat_codes \
    --which-split train \
    --user-sample-frac 1
```

This creates model-input Parquet files under `data/age/preprocessed_full/train/` and `test/`, including event features, downstream targets, and evaluation shifts. `--which-split train` selects the labeled **source** partition; the script constructs both benchmark partitions from it. `--user-sample-frac 1` retains all eligible users after the script's filtering rules.

Dataset configurations in `configs/datasets/` specify the paths consumed by training. The `full/age` configuration expects the `preprocessed_full` directory above. Hugging Face annotations alone cannot replace these model inputs.

### 3. Pretrain an encoder and evaluate its embeddings

For example, run Mask-InfoNCE on AGE with the regression-selected hyperparameters:

```bash
TASK_NAME=best_regression python main.py \
    --dataset full/age \
    --method SimCLR_masks \
    --experiment train_best \
    --specify best_regression \
    --downstream_validator universal_validator/configs/validator/logreg_3seed.yaml \
    --gpu cuda:0
```

The command runs the following stages sequentially:

1. **SSL pretraining:** train the encoder without downstream labels using the selected method. Mask-InfoNCE creates two masked views of each sequence and optimizes InfoNCE.
2. **Checkpoint loading:** select and load the checkpoint according to the configured training criterion. `best_regression` identifies a hyperparameter configuration selected for that source task; it does not itself change the checkpoint selection rule.
3. **Embedding extraction:** freeze the encoder and generate sequence embeddings and embeddings at the local evaluation shifts.
4. **Downstream evaluation:** fit the configured probes on the training embeddings and evaluate them on the benchmark test data. This example repeats linear-probe fitting with seeds 0, 1, and 2 while keeping the encoder fixed.

The `train_best` experiment currently trains **one encoder**. Its `runner.params.n_runs` and `runner.seed_keys` control independent encoder repetitions; `validator_seeds` controls downstream-probe repetitions. These measure different sources of variability.

Model architecture and optimization defaults are in `configs/methods/`; task-selected hyperparameters are in `configs/specify/<dataset>/<method>/best_*.yaml`. Hyperparameter selection and final encoder training are separate stages. Use `best_classification`, `best_forecasting`, or `best_anomaly` where provided to select another source-task configuration.

Results, resolved configuration, logs, and checkpoints are saved under:

```text
log/full/age/SimCLR_masks/tests/best_regression/
├── results.csv
└── seed_0/
    ├── config.yaml
    ├── log
    └── pretrain/ckpt/
```

A suffix is added if the run directory already exists. Checkpoint filenames include epoch and tracked-metric information. This example runs downstream evaluation automatically after pretraining; no separate extraction command is needed.

### 4. Evaluate an existing checkpoint without pretraining again

Use the `inference` experiment with `PTH` pointing to a saved checkpoint and the same dataset, method, and hyperparameter configuration:

```bash
PTH=/absolute/path/to/saved.ckpt TASK_NAME=reeval_best_regression \
python main.py \
    --dataset full/age \
    --method SimCLR_masks \
    --experiment inference \
    --specify best_regression \
    --downstream_validator universal_validator/configs/validator/logreg_3seed.yaml \
    --gpu cuda:0
```

`inference` sets the training budget to zero, loads the supplied checkpoint, extracts embeddings, and runs the probes. The output is a separate `tests/reeval_best_regression/` run. If a dataset provides an evaluation-only sampling configuration, pass `--extra-config rsample` to this inference command to use it.

For fixed-encoder results, reported mean and standard deviation describe repeated downstream-probe runs. Report independent SSL-training seeds separately when using multiple encoders.

### Method names

Some configuration names retain their original implementation names:

| Paper | Configuration name |
|---|---|
| JEPA | `jepa` |
| MLM (BERT4Rec) | `bert4rec` |
| CoLES | `coles` |
| Slice-InfoNCE | `SimCLR` |
| Mask-InfoNCE | `SimCLR_masks` |
| NTP-GPT | `ntp_gpt` |
| NTP-GRU | `ntp_gru` |

## Original data sources

| Dataset | Domain | Source |
|---|---|---|
| AGE | Transactions | [Age Dataset](https://ods.ai/competitions/sberbank-sirius-lesson/data) |
| AlphaBattle | Transactions | [AlphaBattle 2.0](https://www.kaggle.com/datasets/mrmorj/alfabattle-20) |
| Retail | Transactions | [X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) |
| Taobao | Recommender systems | [User Behavior Data from Taobao](https://tianchi.aliyun.com/dataset/46) |
| Zvuk | Recommender systems | [Zvuk Dataset](https://www.kaggle.com/datasets/alexxl/zvuk-dataset) |
| 30Music | Recommender systems | [30Music Listening and Playlists Dataset](https://remaplab.deib.polimi.it/resources/) |
| Yambda | Recommender systems | [Yambda](https://huggingface.co/datasets/yandex/yambda) |
| Electric Devices | Time series | [ElectricDevices](https://www.timeseriesclassification.com/description.php?Dataset=ElectricDevices) |
| ETT | Time series | [Electricity Transformer Temperature Dataset](https://github.com/zhouhaoyi/ETDataset) |
| Favorita | Time series | [Corporacion Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) |
| Rossmann | Time series | [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) |
| Twitter | Text | [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140/data) |
