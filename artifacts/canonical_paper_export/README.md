# Canonical strict-v2 paper export

This directory is a compact, machine-readable export of the verified results
used by the manuscript. It excludes raw market data, model checkpoints, full
prediction tensors, third-party repositories, historical same-close results,
smoke runs, and weighted-ranking experiments.

## Protocol

- Protocol ID: `nextclose_t1_t6_stride5_v2_20260905`
- Split ID: `chronological_90_10_label_availability_purge_v1_20260905`
- Accounting ID: `drift_full_l1_entry_fee_mark_to_market_v1_20260905`
- Fee: `0.001`
- Annualization factor: `50.4`
- Risk-free rate: `0.0`
- Cohort: matched experiment seeds `001-010`; deterministic classical paths
  have no seed interval.

The included markets are `DJIA30_2026`,
`SP500_TOP50_EX_PLTR_2026`, and
`TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED`.

## Contents

- `results/metrics_per_seed.csv`: 375 canonical metric rows.
- `results/metrics_summary.csv`: 51 method-by-market summaries.
- `results/paired_contrasts.csv`: nine matched M3 contrasts with Holm correction.
- `results/portfolio_curves_long.csv`: 36,875 frozen final-test path points.
- `results/results_manifest.json`: metric definitions and formal exclusions.
- `results/supplementary/`: fee sensitivity, market-regime diagnostics, and
  native-horizon DeepAries results.
- `data_protocol/`: universe, sector, preprocessing, split, exclusion, and
  sample-date manifests without raw prices.
- `configuration/`: resolved formal method and accounting settings.
- `audit/`: implementation verification and T07/T08 completion summaries.
- `SHA256SUMS.txt`: checksums generated from the files in this directory.

`DeepAries` is supplementary because its native output uses dynamic 1/5/20-day
horizons and is not directly ranked in the fixed-five-day headline comparison.

The full internal source-path manifest is intentionally omitted because it
contains workstation/container paths and does not improve portable reuse.
