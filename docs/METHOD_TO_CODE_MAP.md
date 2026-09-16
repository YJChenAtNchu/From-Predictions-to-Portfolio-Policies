# Method-to-Code Map

| Paper component | Audited runtime source |
|---|---|
| Strict next-close loader and guards | `torch_data_execution_t1_t6_v2.py` |
| Formal split and tensor construction | `torch_data_formal_split.py`, `torch_data.py` |
| Asset-aligned sparse frontend | `torch_model_paper_sweep_frontend_solver_20260619.py` |
| GRU, temporal attention, and heads | `torch_model.py` |
| ProfitBoost sample reweighting | `torch_train_profitboost.py`, `torch_train_profitboost_formal_val.py` |
| HHI Effective-K and TopKSoftmax | `torch_portfolio.py` |
| Alpha-policy composition | `torch_multiweak_ensemble.py` |
| Canonical strict-v2 orchestration | `paper/teacher/KBS_codex_revision_20260905/runs/run_strict_v2_proposed.py` |
| Public data entry point | `scripts/prepare_data.py` |
| Public training/evaluation entry point | `scripts/train_proposed.py` |

Files below `src/prediction_to_portfolio_policies/runtime/` preserve the
audited experiment bytes and historical import layout. The public scripts wrap
that runtime without altering the model or optimization logic.
