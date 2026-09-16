# Frozen Paper Protocol

The paper release uses the strict next-close protocol
`nextclose_t1_t6_stride5_v2_20260905`.

## Observation and execution

For an information date `t`:

1. The model observes adjusted OHLC data only through `close[t]`.
2. The decision is formed after `close[t]`.
3. The portfolio is executed at adjusted `close[t+1]`.
4. The five-session gross return target is
   `adjusted_close[t+6] / adjusted_close[t+1]`.

The input contains 15 sessions in `[Close, Open, High, Low]` order and is
normalized by the adjusted close immediately preceding the input window.
Samples follow a fixed five-common-session cadence on each locked universe.

## Selection boundary

- The historical architecture, trial020 hyperparameters, and HHI rule were
  frozen before the strict-v2 rerun.
- The training period is split chronologically into inner training and inner
  validation with a label-availability purge.
- Outer validation is reported but is not used for checkpoint selection in the
  strict-v2 migration.
- Final test data remain sealed until every requested learner checkpoint and
  its training-derived reliability coefficient are frozen.

## Portfolio accounting

The formal evaluator uses long-only, fully invested weights, drift-aware full
L1 turnover, an entry transaction fee, and no terminal liquidation fee. The
paper configuration uses a fee of `0.001` per unit of turnover.
