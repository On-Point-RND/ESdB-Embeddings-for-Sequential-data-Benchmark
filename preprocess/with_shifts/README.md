# Dataset Preprocessing

Preprocessing converts original source data into model-input Parquet files with sequence features, benchmark targets, evaluation shifts, and split assignments. The [Hugging Face release](https://huggingface.co/datasets/On-Point-Rnd/ESdB-Embeddings-for-Sequential-data-Benchmark) contains annotations only; original events/features must be obtained from the sources in the [repository README](../../README.md#original-data-sources).

## Running Preprocessing

Run commands from the repository root. Scripts have dataset-specific arguments; inspect them before preparing another dataset:

```bash
python -m preprocess.with_shifts.age --help
```

### Example: full AGE preparation

Place the labeled source files at:

```text
data/age/data/transactions_train.csv
data/age/data/train_target.csv
```

Then run:

```bash
python -m preprocess.with_shifts.age \
    --data-path data/age/data \
    --save-path data/age/preprocessed_full \
    --cat-codes-path data/age/preprocessed_full/cat_codes \
    --which-split train \
    --user-sample-frac 1
```

Expected outputs:

```text
data/age/preprocessed_full/train/
data/age/preprocessed_full/test/
data/age/preprocessed_full/cat_codes/
```

`--which-split train` selects the labeled source partition, from which the script builds both benchmark train and test data. It does not mean that only benchmark training data is written. `--user-sample-frac 1` disables user subsampling; eligibility filters and temporal splitting still apply. Omit it to use the dataset's default sampling fraction.

The `full/age` training configuration reads these `preprocessed_full` paths. Other configurations may use `preprocessed`; match `--save-path` to the corresponding `configs/datasets/<dataset>.yaml`. Existing outputs are preserved by default; `--overwrite` explicitly replaces them.

### Other datasets

Available preprocessing modules:

`30music`, `age`, `alpha`, `electric_devices`, `ett`, `favorita`, `rossman`, `taobao`, `twitter`, `x5-retail`, `yambda`, and `zvuk`.

The `--user-sample-frac` option is supported by `30music`, `age`, `alpha`, `twitter`, `x5-retail`, `yambda`, and `zvuk`. Other modules process their full input without this option. Source file layouts and `--which-split` handling differ between modules; use each module's `--help` and input-reading code rather than applying AGE arguments unchanged.

Local tasks are evaluated at the `shifts` positions. Global tasks use `global_train` for their fitting/test assignment. Source identifiers, shifts, assignments, and `target__*` columns can be inspected in the published annotation release; event features remain in the locally prepared data.

## Target Naming Convention

Use the following naming convention when defining target variables:

```
target__<name>__<local/global>__<metric1+metric2+...>
```

### Fields

- **`<name>`** — a short target name (for example, `reg_amount`, `age`, or `anomaly`);
- **`<local/global>`** — whether the task is local (at the individual event or session level) or global (at the entire sequence level);
- **`<metric1+metric2+...>`** — the metrics used to evaluate the model, separated by `+` without spaces.

### Valid Examples

- `target__reg_amount__local__mse+r2`
- `target__age__global__accuracy+f1_macro`
- `target__anomaly__local__roc_auc+f1_macro+accuracy`
