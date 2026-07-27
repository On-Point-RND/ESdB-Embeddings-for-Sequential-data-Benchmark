# Dataset Preprocessing

## Running Preprocessing

Run the commands from the project root.

General command template:

```bash
python -m preprocess.with_shifts.<dataset> \
    --data-path data/<dataset>/data \
    --save-path data/<dataset>/preprocessed/ \
    --cat-codes-path data/<dataset>/preprocessed/cat_codes \
    --overwrite
```

To preprocess the full dataset:

```bash
python -m preprocess.with_shifts.<dataset> \
    --data-path data/<dataset>/data \
    --save-path data/<dataset>/preprocessed_full/ \
    --cat-codes-path data/<dataset>/preprocessed_full/cat_codes \
    --overwrite \
    --user-sample-frac 1
```

Replace `<dataset>` with the name of the corresponding preprocessing module.
The `--user-sample-frac 1` option includes all users in the sample.

Available `<dataset>` values:

- `30music`
- `age`
- `alpha`
- `electric_devices`
- `ett`
- `favorita`
- `rossman`
- `taobao`
- `twitter`
- `x5-retail`
- `yambda`
- `zvuk`

The `--user-sample-frac` option is supported by `30music`, `age`, `alpha`,
`twitter`, `x5-retail`, `yambda`, and `zvuk`. The remaining modules process
the entire input dataset without this option.

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
