# Strict-v2 model-ready datasets

This directory contains the processed tensors used by the formal strict-v2
experiments. It is ready to use with `scripts/train_proposed.py`; no dataset
rebuilding is required.

## Protocol

- Protocol: `nextclose_t1_t6_stride5_v2_20260905`
- Input: 15-session adjusted OHLC windows in `[Close, Open, High, Low]` order
- Normalization: every value is divided by the adjusted close immediately
  preceding the input window
- Information cutoff: adjusted OHLC through `close[t]`
- Decision time: after `close[t]`
- Execution: adjusted `close[t+1]`
- Target: `adjusted_close[t+6] / adjusted_close[t+1]`
- Holding period and stride: five common trading sessions
- Outer validation: 2021-2023
- Frozen final test: 2024-2025

## Included datasets

| Dataset | Assets | Outer train | Outer validation | Final test | Input shape per sample |
|---|---:|---:|---:|---:|---|
| `DJIA30_2026` | 30 | 600 | 149 | 100 | `(30, 15, 4)` |
| `SP500_TOP50_EX_PLTR_2026` | 50 | 398 | 150 | 99 | `(50, 15, 4)` |
| `TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED` | 44 | 585 | 145 | 96 | `(44, 15, 4)` |

## File contents

Each dataset directory contains:

- `DJIA_<split>_asset_states.npy`: normalized model inputs with shape
  `(samples, assets, 15, 4)`;
- `DJIA_<split>_asset_rors.npy`: next-close five-session target relatives with
  shape `(samples, assets)`;
- `DJIA_<split>_sample_dates.csv`: information, execution, and target dates;
- `order.csv`: locked asset order and sector assignment;
- `dataset_summary.json`: protocol, normalization, shapes, and date coverage;
- `excluded_samples.csv`: samples removed by protocol checks;
- `raw_file_checksums.csv`: provenance hashes for the locked source files.

The historical `DJIA_` filename prefix is retained for compatibility with the
audited data loader and does not imply that all directories contain DJIA data.

Raw Yahoo Finance price files and absolute close-price arrays are not
redistributed. The included arrays contain normalized windows and return
relatives required by the model and evaluator.

## Training

From the repository root:

```powershell
python scripts/train_proposed.py `
  --data-root datasets/strict_v2_model_ready `
  --datasets DJIA30_2026 `
  --seed-start 1 --seed-end 1 `
  --smoke
```

For the formal five-learner system over matched seeds 1-10:

```powershell
python scripts/train_proposed.py `
  --data-root datasets/strict_v2_model_ready `
  --seed-start 1 --seed-end 10 `
  --allow-final-test
```

Use `SHA256SUMS.txt` to verify the downloaded files before training.
