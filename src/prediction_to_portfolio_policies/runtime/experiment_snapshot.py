from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any


CORE_PY_FILES = [
    "torch_data.py",
    "torch_data_execution_t1_t6_v2.py",
    "torch_losses.py",
    "torch_metrics.py",
    "torch_model.py",
    "torch_model_asset_aligned_20260614.py",
    "torch_model_paper_sweep_frontend_solver_20260619.py",
    "torch_multiweak_ensemble.py",
    "torch_portfolio.py",
    "torch_portfolio_accounting_v2.py",
    "torch_train_smoke.py",
    "torch_train_profitboost.py",
    "torch_train_profitboost_formal_val.py",
    "torch_train_profitboost_feeaware.py",
    "torch_train_online.py",
    "torch_train_online_feeaware.py",
    "posthoc_no_online_topk_ablation.py",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst_dir: Path, root: Path) -> dict[str, Any] | None:
    if not src.exists() or not src.is_file():
        return None
    rel_name = src.name
    dst = dst_dir / rel_name
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        dst = dst_dir / f"{stem}_{_sha256(src)[:8]}{suffix}"
    shutil.copy2(src, dst)
    try:
        rel_src = str(src.resolve().relative_to(root.resolve()))
    except ValueError:
        rel_src = str(src.resolve())
    return {
        "source": rel_src,
        "snapshot": str(dst.name),
        "bytes": int(dst.stat().st_size),
        "sha256": _sha256(dst),
    }


def _git_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            info["status_short"] = status.stdout.splitlines()
    except Exception as exc:  # pragma: no cover - snapshot must never block training.
        info["error"] = repr(exc)
    return info


def save_code_snapshot(
    out_dir: str | Path,
    *,
    root: str | Path | None = None,
    extra_files: Iterable[str | Path] | None = None,
    note: str | None = None,
    include_run_scripts: bool = True,
) -> dict[str, Any]:
    """Copy experiment code into `<out_dir>/code_snapshot`.

    The helper is intentionally best-effort: snapshot failures should not stop
    long training runs. It records hashes and git status in a manifest so later
    result summaries can recover which code produced an output folder.
    """
    root_path = Path(root) if root is not None else Path(__file__).resolve().parent
    root_path = root_path.resolve()
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    snapshot_dir = out_path / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    candidate_files: list[Path] = [root_path / name for name in CORE_PY_FILES]
    candidate_files.append(Path(sys.argv[0]).resolve())

    env_run_script = os.environ.get("EXPERIMENT_RUN_SCRIPT")
    if env_run_script:
        candidate_files.append(Path(env_run_script).resolve())
    elif include_run_scripts:
        candidate_files.extend(sorted(root_path.glob("run_*.sh")))

    if extra_files:
        candidate_files.extend(Path(p).resolve() for p in extra_files)

    copied: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for src in candidate_files:
        try:
            src = src.resolve()
        except OSError:
            continue
        if src in seen:
            continue
        seen.add(src)
        item = _copy_file(src, snapshot_dir, root_path)
        if item is not None:
            copied.append(item)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "root": str(root_path),
        "out_dir": str(out_path.resolve()),
        "argv": sys.argv,
        "note": note,
        "experiment_run_script": env_run_script,
        "copied_files": copied,
        "git": _git_info(root_path),
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
