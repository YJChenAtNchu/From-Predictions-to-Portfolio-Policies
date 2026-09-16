from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "prediction_to_portfolio_policies" / "runtime"


def test_audited_runtime_hashes() -> None:
    with (ROOT / "docs" / "SOURCE_MANIFEST.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        path = RUNTIME / row["runtime_path"]
        assert path.is_file(), row["runtime_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
