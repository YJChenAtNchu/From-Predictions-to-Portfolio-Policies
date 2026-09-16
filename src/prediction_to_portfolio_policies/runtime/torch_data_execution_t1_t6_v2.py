from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import torch_data_execution_t1_t6 as base


EXPECTED_TARGET = base.EXPECTED_TARGET
EXPECTED_PROTOCOL_ID = "nextclose_t1_t6_stride5_v2_20260905"
SPLIT_PROTOCOL_ID = "chronological_90_10_label_availability_purge_v1_20260905"


def load_proposed_execution_data(
    root: str | Path,
    inner_train_val_split: float = 0.9,
    device: str | torch.device | None = None,
    use_signature: bool = False,
    signature_level: int = 2,
    allow_final_test: bool = False,
) -> Any:
    root_path = Path(root)
    summary = json.loads((root_path / "dataset_summary.json").read_text(encoding="utf-8"))
    protocol_id = summary.get("protocol_id")
    if protocol_id != EXPECTED_PROTOCOL_ID:
        raise ValueError(
            f"Expected protocol_id={EXPECTED_PROTOCOL_ID}, got {protocol_id!r} at {root_path}"
        )
    bundle = base.load_proposed_execution_data(
        root=root_path,
        inner_train_val_split=inner_train_val_split,
        device=device,
        use_signature=use_signature,
        signature_level=signature_level,
        allow_final_test=allow_final_test,
    )

    # The nominal 90/10 boundary is defined by sample index. Because a five-day
    # label matures after its information date, the final nominal training label
    # can cross the first inner-validation decision cutoff. Purge only those
    # boundary samples whose labels are not yet available at that cutoff.
    train_dates = base._load_dates(root_path, "outer_train")
    nominal_cutoff = int(len(train_dates) * inner_train_val_split)
    first_validation_cutoff = train_dates.iloc[nominal_cutoff]["information_end_date"]
    nominal_train_dates = train_dates.iloc[:nominal_cutoff]
    keep_mask = nominal_train_dates["target_end_date"] < first_validation_cutoff
    kept_train_count = int(keep_mask.sum())
    if not bool(keep_mask.iloc[:kept_train_count].all()) or bool(keep_mask.iloc[kept_train_count:].any()):
        raise ValueError("Inner-boundary availability purge is not a contiguous suffix")
    purged_count = nominal_cutoff - kept_train_count
    if purged_count <= 0:
        raise ValueError("Expected at least one availability-crossing inner-train sample")

    bundle.x_train_list = [tensor[:kept_train_count] for tensor in bundle.x_train_list]
    bundle.y_train_reg = bundle.y_train_reg[:kept_train_count]
    bundle.y_train_rate = bundle.y_train_rate[:kept_train_count]
    bundle.y_train_cls = bundle.y_train_cls[:kept_train_count]
    kept_dates = nominal_train_dates.iloc[:kept_train_count]
    purged_dates = nominal_train_dates.iloc[kept_train_count:nominal_cutoff]
    bundle.meta.update(
        {
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "split_protocol_id": SPLIT_PROTOCOL_ID,
            "nominal_inner_train_count": nominal_cutoff,
            "n_train": kept_train_count,
            "inner_boundary_purged_count": purged_count,
            "inner_boundary_purged_sample_indices": purged_dates["sample_index"].astype(int).tolist(),
            "inner_boundary_purge_rule": (
                "exclude nominal inner-training samples whose target_end_date is not strictly "
                "earlier than the first inner-validation information_end_date"
            ),
        }
    )
    bundle.meta["date_summaries"]["inner_train"] = {
        "n": kept_train_count,
        "information_first": kept_dates["information_end_date"].min().date().isoformat(),
        "information_last": kept_dates["information_end_date"].max().date().isoformat(),
        "execution_first": kept_dates["execution_date"].min().date().isoformat(),
        "execution_last": kept_dates["execution_date"].max().date().isoformat(),
        "target_first": kept_dates["target_end_date"].min().date().isoformat(),
        "target_last": kept_dates["target_end_date"].max().date().isoformat(),
    }
    return bundle
