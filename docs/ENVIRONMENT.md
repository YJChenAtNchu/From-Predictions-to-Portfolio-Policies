# Verified Paper Environment

The formal strict-v2 experiments were executed in the archived environment
below. These values come from the KBS revision environment audit rather than
from inferred package metadata.

| Component | Verified value |
|---|---|
| Host OS | Windows 11 Enterprise |
| CPU | AMD Ryzen 7 9800X3D |
| RAM | 128 GiB |
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| NVIDIA driver | 591.86 |
| Docker image | `cscnet-pytorch:cu128` |
| Docker image ID | `ae6241194f61` |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| cuDNN | 9.1.9 |
| NumPy | 2.4.3 |
| pandas | 3.0.1 |

The package requirements intentionally specify portable lower bounds. Exact
paper reproduction should use the archived CUDA environment above; CPU-only
installation is sufficient for the lightweight implementation tests.
