#!/usr/bin/env python3
"""Train REMAP SitToStand for all 5 folds sequentially.

This wrapper:
- makes sure the split CSVs exist
- creates a temporary fold-specific config by editing only `fold`
- launches the existing distributed training entrypoint for each fold

Example:
    python tools/train_remap_5fold.py --gpus 1
    python tools/train_remap_5fold.py --gpus 2 --validate --test-best
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "remap" / "remap.py"
SPLIT_SCRIPT = REPO_ROOT / "tools" / "build_remap_splits.py"
DIST_TRAIN = REPO_ROOT / "tools" / "dist_train.sh"
SPLIT_DIR = REPO_ROOT / "data" / "REMAP" / "SitToStand" / "splits"


def _read_exp_version(config_path: Path) -> str:
    text = config_path.read_text()
    match = re.search(r"^exp_version\s*=\s*['\"]([^'\"]+)['\"]\s*$", text, flags=re.MULTILINE)
    return match.group(1) if match else "remap"


def _has_all_splits() -> bool:
    for fold in range(5):
        for split in ("train", "val", "test"):
            if not (SPLIT_DIR / f"fold_{fold}_{split}.csv").exists():
                return False
    return True


def _ensure_splits() -> None:
    if _has_all_splits():
        return
    cmd = [sys.executable, str(SPLIT_SCRIPT)]
    print(f"[REMAP-5fold] Building splits: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _make_fold_config(base_config: Path, fold: int, tmp_dir: Path) -> Path:
    text = base_config.read_text()
    text = re.sub(r"^fold\s*=\s*\d+\s*$", f"fold = {fold}", text, flags=re.MULTILINE)
    text = re.sub(r"^exp_version\s*=\s*.*$", "exp_version = 'ver0'", text, flags=re.MULTILINE)
    text = re.sub(r"^work_dir\s*=\s*.*$", "work_dir = f'./work_dirs/remap/{exp_version}/fold_{fold}'", text, flags=re.MULTILINE)

    out_path = tmp_dir / f"remap_fold_{fold}.py"
    out_path.write_text(text)
    return out_path


def _run_fold(config_path: Path, gpus: int, extra_args: list[str]) -> None:
    cmd = ["bash", str(DIST_TRAIN), str(config_path), str(gpus)] + extra_args
    print(f"[REMAP-5fold] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train REMAP SitToStand on 5 folds sequentially")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG, help="Base REMAP config")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per fold")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to run")
    parser.add_argument("--start-fold", type=int, default=0, help="First fold index")
    parser.add_argument("--validate", action="store_true", help="Run validation after every epoch")
    parser.add_argument("--test-best", action="store_true", help="Test the best checkpoint after training")
    parser.add_argument("--test-last", action="store_true", help="Test the last checkpoint after training")
    parser.add_argument("--rebuild-splits", action="store_true", help="Force regeneration of split CSVs")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to dist_train.sh")
    args = parser.parse_args()

    if not args.base_config.exists():
        raise FileNotFoundError(args.base_config)

    if args.folds <= 0:
        raise ValueError("--folds must be positive")
    if args.start_fold < 0:
        raise ValueError("--start-fold must be non-negative")
    if args.start_fold >= 5:
        raise ValueError("--start-fold must be in [0, 4]")
    if args.start_fold + args.folds > 5:
        raise ValueError("Invalid fold range")

    if args.rebuild_splits and SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    _ensure_splits()
    exp_version = _read_exp_version(args.base_config)

    extra_args = [arg for arg in args.extra_args if arg != "--"]
    if args.validate:
        extra_args.append("--validate")
    if args.test_best:
        extra_args.append("--test-best")
    if args.test_last:
        extra_args.append("--test-last")

    work_dirs_root = REPO_ROOT / "work_dirs"
    work_dirs_root.mkdir(parents=True, exist_ok=True)
    remap_root = work_dirs_root / "remap"
    remap_root.mkdir(parents=True, exist_ok=True)
    version_root = remap_root / exp_version
    version_root.mkdir(parents=True, exist_ok=True)
    tmp_root = version_root / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="remap_5fold_", dir=str(tmp_root)) as tmp:
        tmp_dir = Path(tmp)
        for fold in range(args.start_fold, args.start_fold + args.folds):
            fold_cfg = _make_fold_config(args.base_config, fold, tmp_dir)
            print(f"[REMAP-5fold] Starting fold {fold}")
            _run_fold(fold_cfg, args.gpus, extra_args)
            print(f"[REMAP-5fold] Finished fold {fold}")


if __name__ == "__main__":
    main()
