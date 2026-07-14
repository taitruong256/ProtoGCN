"""Fuse CASIA-B ensemble features and visualize them with t-SNE."""
import argparse
import csv
import os
import os.path as osp
import pickle
import re
import sys
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, load
import torch
from mmcv.runner import load_checkpoint

# Add the project root to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protogcn.smp import comb
from protogcn.models import build_model
from protogcn.utils import cache_checkpoint
from tools.visualize_tsne import (
    _build_split_loader,
    _extract_split_features as _extract_split_features_from_loader,
    _plot_tsne,
    _prepare_feature_mode,
    run_tsne,
)

ROOT = Path(__file__).resolve().parents[1]

joint_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/j_new_full_optimized/last_pred.pkl'
bone_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/b_new_full_optimized/last_pred.pkl'
kbone_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/k_new_full_optimized/last_pred.pkl'
joint_motion_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/jm_new_full_optimized/last_pred.pkl'
bone_motion_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/bm_new_full_optimized/last_pred.pkl'
kbone_motion_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/km_new_full_optimized/last_pred.pkl'
angle_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/a_new_full_optimized/last_pred.pkl'
relative_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/r_new/last_pred.pkl'


def _seq_name_from_image_name(image_name):
    seq_name = os.path.normpath(image_name).split(os.sep)[0]
    if seq_name == ".":
        seq_name = os.path.normpath(image_name).split(os.sep)[1]
    return seq_name


def _load_seq_infos(csv_path):
    seq_infos = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_name = _seq_name_from_image_name(row["image_name"])
            if seq_name in seq_infos:
                continue
            subject, condition, sequence, view = seq_name.split("-")
            if condition == "nm" and sequence in {"01", "02", "03", "04"}:
                gait_role = "gallery"
            elif condition == "nm" and sequence in {"05", "06"}:
                gait_role = "probe"
            elif condition in {"bg", "cl"}:
                gait_role = "probe"
            else:
                gait_role = "ignore"
            seq_infos[seq_name] = dict(
                frame_dir=seq_name,
                label=int(subject) - 1,
                subject=subject,
                condition=condition,
                sequence=sequence,
                view=view,
                gait_role=gait_role,
            )
    return [seq_infos[k] for k in sorted(seq_infos)]


def _load_labels(csv_path, pkl_path):
    if osp.exists(pkl_path):
        label = load(pkl_path)
        if label and isinstance(label[0], dict):
            label = [x["label"] for x in label]
        return label

    labels_by_seq = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_name = _seq_name_from_image_name(row["image_name"])
            labels_by_seq.setdefault(seq_name, int(seq_name.split("-")[0]) - 1)

    labels = [labels_by_seq[seq_name] for seq_name in sorted(labels_by_seq)]
    os.makedirs(osp.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(labels, f)
    return labels


def _normalize_feature(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def gait_rank1(features, seq_infos):
    features = np.asarray([_normalize_feature(x) for x in features], dtype=np.float32)
    labels = np.asarray([ann["label"] for ann in seq_infos])
    roles = np.asarray([ann.get("gait_role", "probe") for ann in seq_infos])
    gallery_mask = roles == "gallery"
    probe_mask = roles == "probe"
    gallery_features = features[gallery_mask]
    gallery_labels = labels[gallery_mask]
    probe_features = features[probe_mask]
    probe_labels = labels[probe_mask]
    probe_conditions = np.asarray([ann.get("condition", "") for ann in seq_infos])[probe_mask]

    templates = []
    template_labels = []
    for label in sorted(set(gallery_labels.tolist())):
        tpl = _normalize_feature(gallery_features[gallery_labels == label].mean(axis=0))
        templates.append(tpl)
        template_labels.append(label)

    templates = np.stack(templates, axis=0)
    template_labels = np.asarray(template_labels)
    pred = template_labels[np.argmin(1 - np.matmul(probe_features, templates.T), axis=1)]
    correct = pred == probe_labels
    out = {"gait_rank1": float(correct.mean())}
    for c in ("bg", "cl", "nm"):
        mask = probe_conditions == c
        if np.any(mask):
            out[f"gait_rank1_{c}"] = float(correct[mask].mean())
    return out


def _best_checkpoint(work_dir):
    work_dir = Path(work_dir)
    ckpts = sorted(work_dir.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}")

    def key(path):
        m = re.search(r"epoch_(\d+)", path.name)
        epoch = int(m.group(1)) if m else -1
        if path.name.startswith("best_gait_contrastive_loss"):
            priority = 3
        elif path.name.startswith("best"):
            priority = 2
        elif path.name.startswith("latest"):
            priority = 1
        else:
            priority = 0
        return (priority, epoch, path.name)

    return max(ckpts, key=key)


def _infer_split_features(cfg_path, checkpoint_path, split):
    split_map = {"train": "train", "valid": "val", "test": "test"}
    if split not in split_map:
        raise KeyError(f"Unsupported split: {split}")
    cfg_split = split_map[split]
    cfg = Config.fromfile(str(cfg_path))
    _prepare_feature_mode(cfg)
    model = build_model(cfg.model)
    checkpoint_path = cache_checkpoint(str(checkpoint_path))
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _, data_loader = _build_split_loader(cfg, cfg_split)
    features, labels = _extract_split_features_from_loader(model, data_loader, device)
    return features, labels


def _load_or_make_scores(stream_name, spec, split):
    if split == "test":
        pred_path = Path(spec["pred_test"])
    elif split == "valid":
        pred_path = Path(spec["pred_valid"])
    else:
        pred_path = Path(spec["pred_train"])

    if pred_path.exists():
        return load(pred_path)

    ckpt_dir = spec.get("ckpt_dir")
    if ckpt_dir is None:
        raise FileNotFoundError(f"[{stream_name}] no checkpoint directory configured.")

    checkpoint_path = _best_checkpoint(ckpt_dir)
    print(f"[{stream_name}] generating {pred_path.name} from {checkpoint_path.name} ...")
    scores, _ = _infer_split_features(spec["cfg"], checkpoint_path, split)
    mmcv.dump(scores, str(pred_path))
    return scores


def _select_csv_and_labels(split):
    csv_path = f"data/casia-b/casia-b_pose_{split}.csv"
    label_path = f"data/casia-b/casia-b_labels_{split}.pkl"
    if not osp.exists(csv_path):
        raise FileNotFoundError(csv_path)
    labels = _load_labels(csv_path, label_path)
    seq_infos = _load_seq_infos(csv_path)
    if len(labels) != len(seq_infos):
        print(f"Warning: label count ({len(labels)}) != sequence count ({len(seq_infos)})")
    return csv_path, labels, seq_infos

STREAM_SPECS = {
    "J": dict(
        pred=joint_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/j_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/j_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/j_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/j_new/j.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/j_new'),
    ),
    "B": dict(
        pred=bone_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/b_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/b_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/b_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/b_new/b.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/b_new'),
    ),
    "K": dict(
        pred=kbone_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/k_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/k_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/k_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/k_new/k.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/k_new'),
    ),
    "JM": dict(
        pred=joint_motion_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/jm_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/jm_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/jm_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/jm_new/jm.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/jm_new'),
    ),
    "BM": dict(
        pred=bone_motion_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/bm_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/bm_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/bm_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/bm_new/bm.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/bm_new'),
    ),
    "KM": dict(
        pred=kbone_motion_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/km_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/km_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/km_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/km_new/km.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/km_new'),
    ),
    "A": dict(
        pred=angle_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/a_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/a_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/a_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/a_new/a.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/a_new'),
    ),
    "R": dict(
        pred=relative_path,
        pred_train=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/r_new/pred_train.pkl'),
        pred_valid=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/r_new/pred_valid.pkl'),
        pred_test=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/r_new/best_pred.pkl'),
        cfg=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/r_new/r.py'),
        ckpt_dir=Path('/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/r_new'),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse CASIA-B ensemble features and visualize with t-SNE")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["J", "B", "K", "JM", "BM", "KM"],
        help="Stream names to fuse, e.g. J B K JM BM KM",
    )
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Fusion weights")
    parser.add_argument("--tsne-output-dir", default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    stream_specs = STREAM_SPECS
    unknown = [s for s in args.streams if s not in stream_specs]
    if unknown:
        raise KeyError(f"Unknown streams: {unknown}. Available: {sorted(stream_specs.keys())}")

    csv_path, labels, seq_infos = _select_csv_and_labels(args.split)
    print(f"Using split: {args.split}")
    print(f"Metadata CSV: {csv_path}")

    scores_list = []
    for stream in args.streams:
        scores = _load_or_make_scores(stream, stream_specs[stream], args.split)
        scores_list.append(scores)

    if args.weights is None:
        weights = [1.0] * len(scores_list)
    else:
        if len(args.weights) != len(scores_list):
            raise ValueError("Number of --weights must match number of --streams")
        weights = args.weights

    fused = comb(scores_list, weights)
    fused = np.asarray(fused, dtype=np.float32)
    print("Fusion done.")
    print(gait_rank1(fused, seq_infos))

    output_dir = args.tsne_output_dir or osp.join("work_dirs", "casia_b", "ensemble_tsne", args.split, "_".join(args.streams))
    mmcv.mkdir_or_exist(output_dir)
    embedding = run_tsne(
        fused,
        perplexity=args.perplexity,
        n_iter=args.iterations,
        learning_rate=args.learning_rate,
        random_state=args.seed,
        pca_dim=args.pca_dim,
    )
    np.save(osp.join(output_dir, f"{args.split}_tsne_embedding.npy"), embedding.astype(np.float32))
    np.save(osp.join(output_dir, f"{args.split}_tsne_labels.npy"), np.asarray(labels, dtype=np.int64))

    if embedding.shape[0] > 0:
        x_min, y_min = np.min(embedding, axis=0)
        x_max, y_max = np.max(embedding, axis=0)
        pad_x = max(1e-6, 0.05 * (x_max - x_min))
        pad_y = max(1e-6, 0.05 * (y_max - y_min))
        xlim = (float(x_min - pad_x), float(x_max + pad_x))
        ylim = (float(y_min - pad_y), float(y_max + pad_y))
    else:
        xlim = None
        ylim = None

    _plot_tsne(
        embedding,
        np.asarray(labels, dtype=np.int64),
        osp.join(output_dir, f"{args.split}_tsne.png"),
        title=f"CASIA-B ensemble t-SNE ({args.split}: {'+'.join(args.streams)})",
        xlim=xlim,
        ylim=ylim,
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
