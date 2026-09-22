# Baseline provenance and common protocol

The headline comparison uses the same strict-v2 observation/execution timing,
locked universes, temporal splits, transaction fee, accounting, and final-test
metric implementation as the Proposed method. Final-test data are used only for
frozen evaluation.

## Common evaluation contract

- Information: adjusted OHLC through close `t`.
- Execution: adjusted close `t+1`.
- Exit: adjusted close `t+6`.
- Target: `adjusted_close[t+6] / adjusted_close[t+1]`.
- Portfolio: long-only and fully invested.
- Headline learned-baseline allocation: fixed Top-5 unless noted otherwise.
- Fee: `0.001` under full-L1, drift-aware accounting, including initial entry.
- Annualization: `50.4` non-overlapping five-session decisions per year.
- Stochastic cohort: matched experiment seeds `001-010` on each market.
- Checkpoint selection: chronological inner validation only.
- Outer validation: reference only; never used to tune a baseline checkpoint.
- Final test: frozen evaluation only.

## Learned baselines

| Method | Adaptation and checkpoint rule | Formal training configuration |
|---|---|---|
| FinGAT | Common adjusted-OHLC windows converted to the sector-graph layout; ranking scores mapped to fixed Top-5; checkpoint maximizes inner-validation MRR. | 200 epochs, learning rate `1e-3`, L2 `1e-4`, hidden dimension 16, `week_num=3`. |
| GERU | Common states and return-ratio labels; up probability mapped to fixed Top-5; checkpoint maximizes inner-validation directional accuracy. | 100 epochs, patience 15, learning rate `1e-3`, weight decay `5e-4`, hidden size 64, CE only. |
| StockFormer v2 | Flow/trend, DWT, train-only graph, and ten OHLCV factor proxies; last-step regression score mapped to fixed Top-5; checkpoint minimizes inner-validation MAE. | 100 epochs, batch size 12, learning rate `1e-3`, `T1=15`, `T2=1`. |
| MASTER | 73 stock features and 63 train-only market-gate features; cross-sectional score mapped to fixed Top-5; checkpoint minimizes inner-validation z-score-label MSE. | 40 epochs, learning rate `1e-5`, model width 256, four temporal heads, two spatial heads. |
| DeepTrader | Common states converted to market and asset streams; the actor is constrained to the common long-only contract; checkpoint maximizes inner-validation APV. | At most 500 epochs, patience 80, batch size 16, learning rate `1e-6`, `G=5`. |

DeepAries retains its native dynamic 1/5/20-session horizon and is therefore
reported only in the supplementary export, not ranked in the fixed-five-day
headline table.

## Source-code boundary

The research workspace used archived copies of the authors' implementations.
Those copies did not preserve `.git` metadata, so exact upstream commit hashes
cannot be recovered without inventing provenance. Third-party repositories are
not redistributed here because their licenses and dependency stacks remain
their own. This repository instead publishes the exact adaptation contract,
training settings, canonical metrics, and run-completion audits used in the
paper. Users should obtain each upstream implementation from its original
source and comply with its license.

The strict-v2 completion audit records 30 successful runs for each learned
baseline: three markets times ten matched seeds. DeepAries' native-horizon
results are separated under
`artifacts/canonical_paper_export/results/supplementary/`.
