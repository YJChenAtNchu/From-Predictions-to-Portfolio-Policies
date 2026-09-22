# T04 Code-Verified Implementation

This document records the implementation that future KBS reruns must execute. It is derived from the actual forward pass and training loop, not inferred from slides.

## Sparse frontend and backbone

- Formal condition: `F3_a2_5day__lista__hhi`.
- Each sector receives `(B,T,A_g*4)` ordered `[Close, Open, High, Low]` per asset.
- The active implementation is `torch_model_paper_sweep_frontend_solver_20260619.py`, not the similarly named legacy A2 class.
- LISTA kernels are `(4,4)`, `(2,1)`, `(1,1)` with strides `(1,4)`, `(1,1)`, `(1,1)`.
- Same padding is explicitly computed before convolution. Transposed convolution is cropped/padded back to the requested reference shape.
- Every latent layer has 32 channels. The final sparse output is `(B,T,A_g,32)` and preserves the asset axis.
- The nonlinearity is `ReLU(u + b_l)`. It yields an empirically sparse non-negative latent representation, but is not claimed to be an exact classical soft-threshold operator.
- All LISTA B/W parameters use Gaussian initialization with `std = 0.1/sqrt(in_channels*kernel_height*kernel_width)`. The three formal standard deviations are `0.025`, `0.0125`, and `0.0176776695`.
- Each sector flattens its asset codes to `(B,T,A_g*32)` and passes them through a GRU with hidden size 64.
- Sector GRU states are concatenated along the feature axis to `(B,T,G*64)`.
- Three four-head self-attention blocks operate over the time axis `T`. This is temporal self-attention over a concatenated sector-state representation; it is not explicit inter-sector attention.
- The sequence is flattened before the heads. Regression output is `(B,1,A)` and classification logits are `(B,1,A,2)`, in `order.csv` asset order.
- The complete measured tensor trace for all sector sizes and all three markets is in `tensor_trace.csv`.

## Weighted training objective

For each minibatch, CE and MSE are computed per sample. The current ProfitBoost sample weights are normalized to minibatch mean one and multiply only the CE/MSE sample loss. The cross-sectional pairwise ranking term is one batch-level scalar and is not multiplied by sample weights.

With EMA enabled, CE, MSE, and ranking are divided by detached running magnitudes. EMA state resets at the start of each ProfitBoost outer round. The corrected implementation updates EMA only on inner-training minibatches and freezes it during inner-validation loss evaluation.

The optimizer is one `torch.optim.Adam` instance with learning rate `2e-5` and weight decay `1e-4`; its state is retained across ProfitBoost outer rounds. Inner training has a 1,000-epoch cap and patience 5 on inner-validation loss. The outer loop has a 50-round cap and patience 5 on inner-validation net portfolio profit. Outer validation is not used for checkpoint selection.

## Profit feedback and reliability

Initial movement importance is

`w0_n = exp(z_n^2 / sum_m z_m^2)`, where `z_n = sum_a(y_rate[n,a])`.

At each round, scores are converted to the configured HHI policy and the gross sample outcome is

`r_n = sum_a(policy[n,a] * y_ratio[n,a]) - 1`.

For the selected `one_step_sum` source, `R = sum_n r_n`. The training-only reference scale is the sum of the absolute one-step profits obtained when the realized ratios themselves are passed through the same allocator:

`r_max = sum_n abs(r_oracle,n)`.

This quantity is a positive scaling reference for the error-like transform; it
is not claimed to be an optimal-portfolio profit or an upper bound on learner
profit.

The identity is independently checked in
`audit/t04_profit_reference_scale_audit.json` and
`audit/t04_profit_reference_scale_detail.csv` against every formal strict-v2
Full/No-Sparse learner summary.

The implemented mapping is

`tau = clip((r_max - R)/(2*r_max), 1e-4, 1-1e-4)`

and

`alpha = 0.5*log((1-tau)/tau)`, `alpha_plus = max(alpha,0)`.

Thus alpha increases monotonically with training profit. Sample updates are

`a_n = exp(-alpha_plus*tanh(r_n))`

and

`w_next = beta*w0 + (1-beta)*(w*a)`, with `beta=0.8`.

The next round normalizes `w_next` over samples. The selected checkpoint and its alpha are taken from the last outer round that improved inner-validation portfolio profit.

## HHI allocation

- Keep the top five raw scores.
- Standardize those five with population standard deviation (`correction=0`) and epsilon `1e-8`.
- Form `p = softmax(z/eta_k)` with `eta_k=0.5`.
- Compute `HHI = sum_i p_i^2` and `N_eff = 1/HHI`.
- Round by `floor(N_eff+0.5)` and clip to `[1,5]`.
- Select assets by stable descending score order; exact ties prefer the earlier `order.csv` asset.
- Allocate within the selected support by `softmax(score/eta_omega)` with `eta_omega=2.0`.

## Multi-strong composition and seeds

- `J=5` independent strong learners are used per experiment seed.
- Learner seed is `experiment_seed*1000 + zero_based_learner_index + 1`.
- Each learner contributes its executable policy with its non-negative training-derived alpha from the inner-validation-selected outer round.
- Final policy is `sum_j alpha_plus_j*policy_j / sum_j alpha_plus_j`.
- If all non-negative alphas are zero, the corrected fallback is the equal average of all five learner policies.
- Alphas are frozen before outer-validation and final-test evaluation.

## Deterministic execution

Future KBS reruns must set Python, NumPy, Torch, and CUDA seeds; set `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; disable cuDNN benchmark, flash SDP, and memory-efficient SDP; enable cuDNN deterministic and math SDP; and call strict `torch.use_deterministic_algorithms(True)` without `warn_only`.

## Corrections made before rerun

1. Inner-validation evaluation no longer mutates training EMA denominators.
2. Exact Top-K score ties now use stable asset-order sorting.
3. The all-zero-alpha helper fallback now averages all learners instead of returning the last learner.
4. Future code snapshots include the actual paper-sweep model, strict v2 loader, formal trainer, ensemble helper, and canonical accounting module.

All corrections are covered by `tests/test_t04_implementation.py`. Archived numerical results were not overwritten and remain non-canonical under T02.
