from __future__ import annotations

from pathlib import Path

import torch_data_formal_split as formal_data


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "strict_v2_model_ready"

EXPECTED = {
    "DJIA30_2026": (30, 600, 149, 100),
    "SP500_TOP50_EX_PLTR_2026": (50, 398, 150, 99),
    "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED": (44, 585, 145, 96),
}


def test_model_ready_datasets_load_with_frozen_splits() -> None:
    for dataset, (assets, outer_train, outer_val, final_test) in EXPECTED.items():
        bundle = formal_data.load_proposed_djia_formal_data(DATA_ROOT / dataset)
        assert bundle.meta["n_assets"] == assets
        assert bundle.meta["n_outer_train_raw"] == outer_train
        assert bundle.meta["n_outer_val"] == outer_val
        assert bundle.meta["n_final_test"] == final_test
        assert all(bundle.meta["split_rule_checks"].values())
        assert bundle.meta["dataset_summary"]["protocol_id"] == "nextclose_t1_t6_stride5_v2_20260905"
