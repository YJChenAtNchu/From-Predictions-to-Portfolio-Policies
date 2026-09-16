# Docker Reproduction

The archived paper experiments used the local image
`cscnet-pytorch:cu128` with image ID `ae6241194f61`. The root `Dockerfile`
reconstructs the minimal Proposed-method environment from the same CUDA
12.8.1 base and the verified Python package versions. It does not claim
bit-for-bit identity with the archived local image.

## Build

```powershell
docker build -t predictions-to-portfolio-policies:strict-v2 .
```

## Run the implementation tests

```powershell
docker run --rm --gpus all `
  predictions-to-portfolio-policies:strict-v2
```

## Prepare data

Mount the locked universe and adjusted-OHLC inputs read-only. The output mount
is writable and remains outside the Git repository.

```powershell
docker run --rm --gpus all `
  -v "D:\locked_universe:/inputs/universe:ro" `
  -v "D:\adjusted_ohlc:/inputs/ohlc:ro" `
  -v "D:\strict_v2_data:/data" `
  predictions-to-portfolio-policies:strict-v2 `
  python scripts/prepare_data.py `
    --dataset-key DJIA30_2026 `
    --source-root /inputs/universe `
    --raw-root /inputs/ohlc `
    --raw-name-style us `
    --output-root /data
```

## Train and run frozen evaluation

```powershell
docker run --rm --gpus all `
  -v "D:\strict_v2_data:/data:ro" `
  -v "D:\paper_runs:/outputs" `
  predictions-to-portfolio-policies:strict-v2 `
  python scripts/train_proposed.py `
    --data-root /data `
    --output-root /outputs/proposed `
    --report-root /outputs/reports `
    --seed-start 1 `
    --seed-end 10 `
    --allow-final-test
```

Omit `--allow-final-test` while developing or selecting checkpoints. Raw data,
checkpoints, and generated outputs are intentionally mounted rather than baked
into the image.
