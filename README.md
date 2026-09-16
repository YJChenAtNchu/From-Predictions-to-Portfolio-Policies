# From Predictions to Portfolio Policies

Private companion repository for the manuscript:

> **From Predictions to Portfolio Policies: Structured Sparse Signals,
> Adaptive Support, and Decision-Feedback Composition**

## Release status

This repository is being assembled incrementally from the audited strict-v2
paper runtime. The current release contains the canonical Proposed method,
frozen configuration, configurable entry points, source hashes, and lightweight
implementation tests. Compact verified paper artifacts will be added in a
later release step.

The formal timing protocol is:

```text
information through adjusted close[t]
-> decision after close[t]
-> execution at adjusted close[t+1]
-> exit at adjusted close[t+6]
```

Therefore the five-session target is
`adjusted_close[t+6] / adjusted_close[t+1]`.

## Repository layout

- `src/prediction_to_portfolio_policies/runtime/` - byte-preserved audited runtime
- `configs/paper_strict_v2.json` - frozen paper configuration
- `scripts/prepare_data.py` - configurable strict-v2 dataset entry point
- `scripts/train_proposed.py` - canonical full-method training and frozen evaluation
- `tests/` - protocol, shape, HHI, EMA, accounting, and source-integrity tests
- `docs/` - protocol, method-to-code map, scope, and source manifest
- `artifacts/canonical_paper_export/` - reserved for compact verified results

## Installation

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Docker

Build the minimal CUDA environment and run the test suite:

```powershell
docker build -t predictions-to-portfolio-policies:strict-v2 .
docker run --rm --gpus all predictions-to-portfolio-policies:strict-v2
```

The formal experiment image and verified package versions are documented in
`docs/ENVIRONMENT.md`. Dataset mounts and formal training commands are in
`docs/DOCKER.md`; raw data and outputs are never copied into the image.

## Prepare a strict-v2 dataset

Raw adjusted-OHLC files are not redistributed. The builder expects a locked
universe source directory containing `order.csv` and the archived sample-date
files, plus one adjusted-OHLC CSV per asset.

```powershell
python scripts/prepare_data.py `
  --dataset-key DJIA30_2026 `
  --source-root PATH_TO_LOCKED_UNIVERSE `
  --raw-root PATH_TO_ADJUSTED_OHLC `
  --raw-name-style us `
  --output-root data/execution_t1_t6_v2_20260905
```

## Run the Proposed method

Run a one-learner smoke test while keeping final test tensors sealed:

```powershell
python scripts/train_proposed.py `
  --data-root data/execution_t1_t6_v2_20260905 `
  --datasets DJIA30_2026 `
  --seed-start 1 --seed-end 1 `
  --smoke
```

Run the formal five-learner system for seeds 1-10. `--allow-final-test` starts
frozen evaluation only after all requested learner checkpoints and reliability
coefficients have been fixed.

```powershell
python scripts/train_proposed.py `
  --data-root data/execution_t1_t6_v2_20260905 `
  --seed-start 1 --seed-end 10 `
  --allow-final-test
```

## Validate the release

```powershell
python -m pytest
```

See `docs/PROTOCOL.md`, `docs/METHOD_TO_CODE_MAP.md`,
`docs/ENVIRONMENT.md`, `docs/DOCKER.md`, `docs/SOURCE_MANIFEST.csv`, and
`docs/REPRODUCIBILITY_SCOPE.md` before interpreting or extending the code.

Raw market data, checkpoints, full prediction arrays, third-party baseline
repositories, temporary outputs, and historical exploratory experiments are
intentionally excluded.
