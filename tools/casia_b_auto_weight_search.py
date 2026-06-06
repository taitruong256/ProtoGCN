"""Search CASIA-B ensemble weights on valid and cache fused valid predictions."""

import csv
import itertools
import os
import pickle
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from mmcv import Config, load as mmcv_load, dump as mmcv_dump
except ImportError: 
    Config = None

    def mmcv_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def mmcv_dump(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "work_dirs/casia_b/ensemble"
VALID_CSV = ROOT / "data/casia-b/casia-b_pose_valid.csv"
TEST_CSV = ROOT / "data/casia-b/casia-b_pose_test.csv"
VALID_LABEL = ROOT / "data/casia-b/casia-b_labels_valid.pkl"
TEST_LABEL = ROOT / "data/casia-b/casia-b_labels_test.pkl"
SEARCH_WEIGHTS = [0, 1.0, 2.0]
STREAMS = [
    ("J", ROOT / "work_dirs/casia_b/j_3", ROOT / "configs/casia_b/j.py"),
    ("B", ROOT / "work_dirs/casia_b/b", ROOT / "configs/casia_b/b.py"),
    ("K", ROOT / "work_dirs/casia_b/k", ROOT / "configs/casia_b/k.py"),
    ("JM", ROOT / "work_dirs/casia_b/jm", ROOT / "configs/casia_b/jm.py"),
    ("BM", ROOT / "work_dirs/casia_b/bm", ROOT / "configs/casia_b/bm.py"),
    ("KM", ROOT / "work_dirs/casia_b/km", ROOT / "configs/casia_b/km.py"),
]


def seq_name_from_image_name(image_name):
    seq_name = os.path.normpath(image_name).split(os.sep)[0]
    if seq_name == ".":
        seq_name = os.path.normpath(image_name).split(os.sep)[1]
    return seq_name


def build_labels(csv_path, pkl_path):
    labels_by_seq = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seq_name = seq_name_from_image_name(row["image_name"])
            labels_by_seq.setdefault(seq_name, int(seq_name.split("-")[0]) - 1)
    labels = [labels_by_seq[s] for s in sorted(labels_by_seq)]
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    mmcv_dump(labels, str(pkl_path))
    return labels


def load_seq_infos(csv_path):
    def role(condition, sequence):
        if condition == "nm" and sequence in {"01", "02", "03", "04"}:
            return "gallery"
        if condition == "nm" and sequence in {"05", "06"}:
            return "probe"
        if condition in {"bg", "cl"}:
            return "probe"
        return "ignore"

    seq_infos = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seq_name = seq_name_from_image_name(row["image_name"])
            if seq_name in seq_infos:
                continue
            subject, condition, sequence, view = seq_name.split("-")
            seq_infos[seq_name] = dict(
                frame_dir=seq_name,
                label=int(subject) - 1,
                condition=condition,
                sequence=sequence,
                view=view,
                gait_role=role(condition, sequence),
            )
    return [seq_infos[k] for k in sorted(seq_infos)]


def normalize(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def gait_rank1(scores, seq_infos):
    feats = np.asarray([normalize(x) for x in scores], dtype=np.float32)
    labels = np.asarray([x["label"] for x in seq_infos])
    roles = np.asarray([x["gait_role"] for x in seq_infos])
    conds = np.asarray([x["condition"] for x in seq_infos])

    gallery = roles == "gallery"
    probe = roles == "probe"
    gallery_feats = feats[gallery]
    gallery_labels = labels[gallery]
    probe_feats = feats[probe]
    probe_labels = labels[probe]
    probe_conds = conds[probe]

    templates, template_labels = [], []
    for label in sorted(set(gallery_labels.tolist())):
        tpl = normalize(gallery_feats[gallery_labels == label].mean(axis=0))
        templates.append(tpl)
        template_labels.append(label)
    templates = np.stack(templates, axis=0)
    template_labels = np.asarray(template_labels)

    pred = template_labels[np.argmin(1 - np.matmul(probe_feats, templates.T), axis=1)]
    correct = pred == probe_labels
    out = {"gait_rank1": float(correct.mean())}
    for c in ("bg", "cl", "nm"):
        m = probe_conds == c
        if np.any(m):
            out[f"gait_rank1_{c}"] = float(correct[m].mean())
    return out


def load_scores(path):
    return [np.asarray(x, dtype=np.float32) for x in mmcv_load(str(path))]


def fuse(scores_list, weights):
    fused = [s * weights[0] for s in scores_list[0]]
    for scores, w in zip(scores_list[1:], weights[1:]):
        fused = [a + b * w for a, b in zip(fused, scores)]
    return fused


def best_checkpoint(work_dir):
    cands = [p for p in work_dir.glob("*.pth") if p.exists() and p.is_file()]
    if not cands:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}")

    def key(p):
        m = re.search(r"epoch_(\d+)", p.name)
        if p.name.startswith("best_gait_contrastive_loss"):
            priority = 3
        elif p.name.startswith("best"):
            priority = 2
        elif p.name.startswith("latest"):
            priority = 1
        else:
            priority = 0
        return (priority, int(m.group(1)) if m else -1, p.name)

    return max(cands, key=key)


def pred_valid_path(work_dir):
    return work_dir / "pred_valid.pkl"


def pred_test_path(work_dir):
    return work_dir / "best_pred.pkl"


def unique_seq_count(csv_path):
    seqs = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seqs.add(seq_name_from_image_name(row["image_name"]))
    return len(seqs)


def ensure_valid_pred(stream_name, work_dir, cfg_file, expected_len):
    out_path = pred_valid_path(work_dir)
    if out_path.exists():
        try:
            if len(mmcv_load(str(out_path))) == expected_len:
                return load_scores(out_path)
        except Exception:
            pass

    if Config is None:
        raise ImportError("mmcv is required to generate validation predictions.")

    cfg = Config.fromfile(str(cfg_file))
    cfg.data.test = cfg.data.val
    cfg.data.test_dataloader = dict(cfg.data.get("val_dataloader", {}))

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        tmp_cfg = Path(f.name)
    try:
        cfg.dump(str(tmp_cfg))
        cmd = [
            "bash",
            str(ROOT / "tools/dist_test.sh"),
            str(tmp_cfg),
            str(best_checkpoint(work_dir)),
            "1",
            "--eval",
            "gait_rank1",
            "--out",
            str(out_path),
        ]
        print(f"Generating {stream_name} -> {out_path.name}")
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    finally:
        if tmp_cfg.exists():
            tmp_cfg.unlink()
    return load_scores(out_path)


def load_test_scores(stream_name, work_dir):
    out_path = pred_test_path(work_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"Missing test prediction for {stream_name}: {out_path}")
    return load_scores(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not VALID_CSV.exists():
        raise FileNotFoundError(str(VALID_CSV))
    if not TEST_CSV.exists():
        raise FileNotFoundError(str(TEST_CSV))

    if not VALID_LABEL.exists():
        build_labels(VALID_CSV, VALID_LABEL)
    if not TEST_LABEL.exists():
        build_labels(TEST_CSV, TEST_LABEL)
    seq_infos = load_seq_infos(VALID_CSV)
    test_seq_infos = load_seq_infos(TEST_CSV)
    expected_len = unique_seq_count(VALID_CSV)
    expected_test_len = unique_seq_count(TEST_CSV)

    scores_list = []
    for name, work_dir, cfg_file in STREAMS:
        scores_list.append(ensure_valid_pred(name, work_dir, cfg_file, expected_len))

    if len(scores_list[0]) != expected_len:
        raise ValueError(f"pred length mismatch: {len(scores_list[0])} != {expected_len}")

    test_scores_list = []
    for name, work_dir, _ in STREAMS:
        scores = load_test_scores(name, work_dir)
        if len(scores) != expected_test_len:
            raise ValueError(f"{name} test pred length mismatch: {len(scores)} != {expected_test_len}")
        test_scores_list.append(scores)

    best = None
    best_weights = None
    rows = []
    for idx, weights in enumerate(
        tqdm(itertools.product(SEARCH_WEIGHTS, repeat=6), total=len(SEARCH_WEIGHTS) ** 6, desc="search"),
        start=1,
    ):
        fused = fuse(scores_list, weights)
        test_fused = fuse(test_scores_list, weights)
        metrics = gait_rank1(fused, seq_infos)
        test_metrics = gait_rank1(test_fused, test_seq_infos)
        rows.append([
            idx,
            *weights,
            metrics["gait_rank1"],
            metrics.get("gait_rank1_bg", ""),
            metrics.get("gait_rank1_cl", ""),
            metrics.get("gait_rank1_nm", ""),
            test_metrics["gait_rank1"],
            test_metrics.get("gait_rank1_bg", ""),
            test_metrics.get("gait_rank1_cl", ""),
            test_metrics.get("gait_rank1_nm", ""),
        ])
        if best is None or metrics["gait_rank1"] > best["gait_rank1"]:
            best = metrics
            best_weights = weights
            best_fused = fused

    mmcv_dump(best_fused, str(OUT_DIR / "pred_valid.pkl"))
    with open(OUT_DIR / "search_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank",
            "w_j",
            "w_b",
            "w_k",
            "w_jm",
            "w_bm",
            "w_km",
            "valid_gait_rank1",
            "valid_gait_rank1_bg",
            "valid_gait_rank1_cl",
            "valid_gait_rank1_nm",
            "test_gait_rank1",
            "test_gait_rank1_bg",
            "test_gait_rank1_cl",
            "test_gait_rank1_nm",
        ])
        w.writerows(rows)
    test_fused = fuse(test_scores_list, best_weights)
    test_metrics = gait_rank1(test_fused, test_seq_infos)

    print("Best weights:", best_weights)
    print("Valid gait_rank1:", best["gait_rank1"])
    print("Test gait_rank1:", test_metrics["gait_rank1"])
    print("Saved:", OUT_DIR / "pred_valid.pkl")
    print("Saved:", OUT_DIR / "search_results.csv")


if __name__ == "__main__":
    main()
