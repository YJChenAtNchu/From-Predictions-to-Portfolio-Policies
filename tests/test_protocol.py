from __future__ import annotations

import json
from pathlib import Path

import torch_data_execution_t1_t6_v2 as strict_data


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_contract() -> None:
    config = json.loads((ROOT / "configs" / "paper_strict_v2.json").read_text())
    assert config["protocol_id"] == strict_data.EXPECTED_PROTOCOL_ID
    assert config["split_protocol_id"] == strict_data.SPLIT_PROTOCOL_ID
    assert config["target"] == strict_data.EXPECTED_TARGET
    assert config["decision_time"] == "after close[t]"
    assert config["execution"] == "adjusted close[t+1]"
    assert config["final_test"].endswith("checkpoints and alpha are frozen")


def test_paper_configuration_is_full_method() -> None:
    config = json.loads((ROOT / "configs" / "paper_strict_v2.json").read_text())
    assert config["architecture"]["frontend"] == "F3_a2_5day"
    assert config["architecture"]["solver"] == "LISTA"
    assert config["allocation"]["rule"] == "HHI Effective-K"
    assert config["ensemble"]["composition"] == "Alpha-policy Multi-Strong"
    assert config["ensemble"]["learners"] == 5
