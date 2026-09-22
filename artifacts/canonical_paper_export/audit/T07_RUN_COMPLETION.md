# T07 Completion Report

Status: DONE

Audit source: `audit/t07_completion_audit.json`

## Verified audit payload

```json
{
  "status": "PASS",
  "protocol_id": "nextclose_t1_t6_stride5_v2_20260905",
  "split_protocol_id": "chronological_90_10_label_availability_purge_v1_20260905",
  "counts": {
    "full_hhi_learners": 150,
    "full_hhi_compositions": 30,
    "nosparse_hhi_learners": 150,
    "nosparse_hhi_compositions": 30,
    "standard_learners": 150,
    "standard_compositions": 30,
    "baseline_fingat": 30,
    "baseline_geru": 30,
    "baseline_stockformer_v2": 30,
    "baseline_master": 30,
    "baseline_deeptrader": 30,
    "baseline_deeparies": 30,
    "classical_paths": 15
  },
  "expected_counts": {
    "full_hhi_learners": 150,
    "full_hhi_compositions": 30,
    "nosparse_hhi_learners": 150,
    "nosparse_hhi_compositions": 30,
    "standard_learners": 150,
    "standard_compositions": 30,
    "baseline_fingat": 30,
    "baseline_geru": 30,
    "baseline_stockformer_v2": 30,
    "baseline_master": 30,
    "baseline_deeptrader": 30,
    "baseline_deeparies": 30,
    "classical_paths": 15
  },
  "missing": [],
  "mismatches": []
}
```

The task is marked complete only because the machine-readable audit reports PASS.
