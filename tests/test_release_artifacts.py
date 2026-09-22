import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "artifacts" / "canonical_paper_export"
RESULTS = EXPORT / "results"

DATASETS = {
    "DJIA30_2026",
    "SP500_TOP50_EX_PLTR_2026",
    "TAIWAN50_0050_EX_LATE_HISTORY_2026_PREVCLOSE_ADJUSTED",
}
PROTOCOL = "nextclose_t1_t6_stride5_v2_20260905"
SPLIT_PROTOCOL = "chronological_90_10_label_availability_purge_v1_20260905"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_canonical_result_counts_and_scope() -> None:
    metrics = read_csv(RESULTS / "metrics_per_seed.csv")
    summaries = read_csv(RESULTS / "metrics_summary.csv")
    contrasts = read_csv(RESULTS / "paired_contrasts.csv")
    curves = read_csv(RESULTS / "portfolio_curves_long.csv")

    assert len(metrics) == 375
    assert len(summaries) == 51
    assert len(contrasts) == 9
    assert len(curves) == 36_875
    assert {row["dataset"] for row in metrics} == DATASETS
    assert {row["dataset"] for row in curves} == DATASETS
    assert {row["protocol_id"] for row in metrics} == {PROTOCOL}
    assert {row["split_protocol_id"] for row in metrics} == {SPLIT_PROTOCOL}


def test_stochastic_seed_coverage() -> None:
    metrics = read_csv(RESULTS / "metrics_per_seed.csv")
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in metrics:
        if row["seed"] == "deterministic":
            continue
        grouped.setdefault((row["dataset"], row["method"]), set()).add(row["seed"])

    expected = {str(seed) for seed in range(1, 11)}
    assert grouped
    assert all(seeds == expected for seeds in grouped.values())


def test_manifest_definitions_and_exclusions() -> None:
    manifest = json.loads((RESULTS / "results_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["protocol_id"] == PROTOCOL
    assert manifest["missing_count"] == 0
    assert manifest["metrics"]["Calmar"] == "GARR / MDD"
    assert "weighted_ranking" in manifest["excluded"]
    assert "archived_same_close" in manifest["excluded"]


def test_portable_export_has_no_local_absolute_paths() -> None:
    forbidden = ("/workspace/", "C:\\Users\\", "D:\\CSCNet\\")
    for path in EXPORT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8-sig", errors="strict")
            assert not any(marker in text for marker in forbidden), path


def test_export_checksums() -> None:
    checksum_path = EXPORT / "SHA256SUMS.txt"
    entries = checksum_path.read_text(encoding="ascii").splitlines()
    assert entries

    for entry in entries:
        expected, relative_path = entry.split("  ", maxsplit=1)
        path = EXPORT / Path(relative_path)
        assert path.is_file(), relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, relative_path
