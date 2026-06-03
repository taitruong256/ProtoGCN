import argparse
import copy
import glob
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from protogcn.datasets import build_dataloader, build_dataset
from protogcn.models import build_model
from protogcn.utils import cache_checkpoint


def _latest_checkpoint(work_dir: str) -> str:
    ckpts = sorted(glob.glob(osp.join(work_dir, "*.pth")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}")
    return ckpts[-1]


def _recursive_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _recursive_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_recursive_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_recursive_to_device(v, device) for v in data)
    return data


def _flatten_labels(label_batch) -> np.ndarray:
    if torch.is_tensor(label_batch):
        return label_batch.detach().cpu().numpy().reshape(-1).astype(np.int64)
    if isinstance(label_batch, np.ndarray):
        return label_batch.reshape(-1).astype(np.int64)
    if isinstance(label_batch, (list, tuple)):
        values = []
        for item in label_batch:
            if torch.is_tensor(item):
                values.append(int(item.detach().cpu().reshape(-1)[0].item()))
            else:
                arr = np.asarray(item).reshape(-1)
                values.append(int(arr[0]))
        return np.asarray(values, dtype=np.int64)
    raise TypeError(f"Unsupported label batch type: {type(label_batch)}")


def _prepare_feature_mode(cfg):
    cfg.model.setdefault("test_cfg", dict())
    cfg.model.test_cfg.feat_ext = True
    cfg.model.test_cfg.setdefault("pool_opt", "all")


def _build_split_dataset(cfg, split: str):
    split_cfg = copy.deepcopy(cfg.data[split])
    if split != "train":
        split_cfg.test_mode = True
        return build_dataset(split_cfg, dict(test_mode=True))
    return build_dataset(split_cfg)


def _build_split_loader(cfg, split: str):
    dataset = _build_split_dataset(cfg, split)
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get("videos_per_gpu", 1),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
        shuffle=False,
    )
    dataloader_setting = dict(dataloader_setting, **cfg.data.get(f"{split}_dataloader", {}))
    loader = build_dataloader(dataset, **dataloader_setting)
    return dataset, loader


def _extract_split_features(model, data_loader, device):
    model.eval()
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for data in tqdm(data_loader, desc="Extract features", leave=False):
        batch = _recursive_to_device(data, device)
        with torch.no_grad():
            batch_features = model(return_loss=False, **batch)

        batch_features = np.asarray(batch_features, dtype=np.float32)
        if batch_features.ndim == 1:
            batch_features = batch_features[None, :]
        features.append(batch_features)

        if "label" not in batch:
            raise KeyError("The batch does not contain `label`; please keep label in the dataset pipeline.")
        labels.append(_flatten_labels(batch["label"]))

    features = np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32)
    labels = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
    return features, labels


def _pca_reduce(x: np.ndarray, n_components: int = 50) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    n_samples, n_features = x.shape
    if n_samples <= 1:
        return x.astype(np.float64, copy=False)

    n_components = max(1, min(n_components, n_samples - 1, n_features))
    x = x.astype(np.float64, copy=False)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:n_components].T


def _pairwise_squared_distances(x: np.ndarray) -> np.ndarray:
    sum_x = np.sum(np.square(x), axis=1)
    dists = sum_x[:, None] + sum_x[None, :] - 2.0 * np.dot(x, x.T)
    np.maximum(dists, 0.0, out=dists)
    return dists


def _hbeta(dist_row: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    p = np.exp(-dist_row * beta)
    sum_p = np.maximum(np.sum(p), 1e-12)
    h = np.log(sum_p) + beta * np.sum(dist_row * p) / sum_p
    p /= sum_p
    return h, p


def _x2p(x: np.ndarray, perplexity: float, tol: float = 1e-5, max_tries: int = 50) -> np.ndarray:
    n = x.shape[0]
    dists = _pairwise_squared_distances(x)
    p = np.zeros((n, n), dtype=np.float64)
    beta = np.ones((n, 1), dtype=np.float64)
    log_u = np.log(perplexity)

    for i in tqdm(range(n), desc="t-SNE entropy search", leave=False):
        betamin = -np.inf
        betamax = np.inf

        dist_row = np.concatenate((dists[i, :i], dists[i, i + 1 :]))
        h, this_p = _hbeta(dist_row, beta[i, 0])
        hdiff = h - log_u
        tries = 0

        while np.abs(hdiff) > tol and tries < max_tries:
            if hdiff > 0:
                betamin = beta[i, 0]
                beta[i, 0] = beta[i, 0] * 2.0 if np.isinf(betamax) else 0.5 * (beta[i, 0] + betamax)
            else:
                betamax = beta[i, 0]
                beta[i, 0] = beta[i, 0] / 2.0 if np.isinf(betamin) else 0.5 * (beta[i, 0] + betamin)

            h, this_p = _hbeta(dist_row, beta[i, 0])
            hdiff = h - log_u
            tries += 1

        p[i, :i] = this_p[:i]
        p[i, i + 1 :] = this_p[i:]

    return p


def run_tsne(
    x: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    learning_rate: float = 200.0,
    random_state: int = 0,
    pca_dim: int = 50,
    early_exaggeration: float = 12.0,
    exaggeration_iters: int = 250,
) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got {x.shape}")
    n_samples = x.shape[0]
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if n_samples == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    x = _pca_reduce(x, n_components=pca_dim)
    max_perplexity = max(1.0, (n_samples - 1) / 3.0)
    perplexity = float(min(perplexity, max_perplexity))
    perplexity = max(perplexity, 1.0)

    p = _x2p(x, perplexity=perplexity)
    p = p + p.T
    p /= np.maximum(np.sum(p), 1e-12)
    p = np.maximum(p, 1e-12)
    p *= early_exaggeration

    rng = np.random.default_rng(random_state)
    y = rng.normal(0.0, 1e-4, size=(n_samples, 2)).astype(np.float64)
    y_inc = np.zeros_like(y)
    gains = np.ones_like(y)
    final_momentum = 0.8

    for it in tqdm(range(n_iter), desc="t-SNE optimize", leave=False):
        sum_y = np.sum(np.square(y), axis=1)
        num = 1.0 / (1.0 + sum_y[:, None] + sum_y[None, :] - 2.0 * np.dot(y, y.T))
        np.fill_diagonal(num, 0.0)
        q = num / np.maximum(np.sum(num), 1e-12)
        q = np.maximum(q, 1e-12)

        l = (p - q) * num
        row_sum = np.sum(l, axis=1, keepdims=True)
        grad = 4.0 * (row_sum * y - np.dot(l, y))

        gains = np.where(np.sign(grad) != np.sign(y_inc), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)

        momentum = 0.5 if it < 250 else final_momentum
        y_inc = momentum * y_inc - learning_rate * gains * grad
        y += y_inc
        y -= np.mean(y, axis=0, keepdims=True)

        if it + 1 == exaggeration_iters:
            p /= early_exaggeration

    return y.astype(np.float32)


def _plot_tsne(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    import matplotlib.pyplot as plt

    mmcv.mkdir_or_exist(osp.dirname(output_path))
    plt.figure(figsize=(8, 7), dpi=180)
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=0.82,
            color=cmap(idx % cmap.N),
            label=str(int(label)),
            linewidths=0,
        )

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if len(unique_labels) <= 20:
        plt.legend(
            loc="best",
            frameon=False,
            fontsize=8,
            markerscale=1.4,
            handletextpad=0.3,
            borderpad=0.2,
            labelspacing=0.3,
        )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract pre-head features and visualize them with t-SNE")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("-C", "--checkpoint", default=None, help="Checkpoint file path; defaults to latest .pth in work_dir")
    parser.add_argument("--feature-dir", default=None, help="Directory to save extracted features and labels")
    parser.add_argument("--output-dir", default=None, help="Directory to save t-SNE figures")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which dataset splits to extract",
    )
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of t-SNE optimization iterations")
    parser.add_argument("--learning-rate", type=float, default=200.0, help="t-SNE learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for t-SNE initialization")
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimension before t-SNE")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--fuse-conv-bn", action="store_true", help="Fuse Conv-BN before inference")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    feature_dir = args.feature_dir or osp.join(cfg.work_dir, "tsne_features")
    output_dir = args.output_dir or osp.join(cfg.work_dir, "tsne_vis")
    mmcv.mkdir_or_exist(feature_dir)
    mmcv.mkdir_or_exist(output_dir)

    checkpoint = args.checkpoint or _latest_checkpoint(cfg.work_dir)
    checkpoint = cache_checkpoint(checkpoint)

    _prepare_feature_mode(cfg)
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    split_features: Dict[str, np.ndarray] = {}
    split_labels: Dict[str, np.ndarray] = {}
    split_sizes: Dict[str, int] = {}

    for split in tqdm(args.splits, desc="Splits"):
        if split not in cfg.data:
            raise KeyError(f"Split `{split}` is not defined in cfg.data")

        _, data_loader = _build_split_loader(cfg, split)
        features, labels = _extract_split_features(model, data_loader, device)
        split_features[split] = features
        split_labels[split] = labels
        split_sizes[split] = len(features)

        feature_path = osp.join(feature_dir, f"{split}_features.npy")
        label_path = osp.join(feature_dir, f"{split}_labels.npy")
        np.save(feature_path, features.astype(np.float32))
        np.save(label_path, labels.astype(np.int64))

    all_features = np.concatenate([split_features[split] for split in args.splits], axis=0)
    all_embedding = run_tsne(
        all_features,
        perplexity=args.perplexity,
        n_iter=args.iterations,
        learning_rate=args.learning_rate,
        random_state=args.seed,
        pca_dim=args.pca_dim,
    )

    embedding_path = osp.join(feature_dir, "all_tsne_embedding.npy")
    np.save(embedding_path, all_embedding.astype(np.float32))

    offset = 0
    x_min, y_min = np.min(all_embedding, axis=0)
    x_max, y_max = np.max(all_embedding, axis=0)
    pad_x = max(1e-6, 0.05 * (x_max - x_min))
    pad_y = max(1e-6, 0.05 * (y_max - y_min))
    xlim = (float(x_min - pad_x), float(x_max + pad_x))
    ylim = (float(y_min - pad_y), float(y_max + pad_y))

    for split in tqdm(args.splits, desc="Write figures"):
        size = split_sizes[split]
        split_embedding = all_embedding[offset : offset + size]
        split_label = split_labels[split]
        split_output = osp.join(output_dir, f"{split}_tsne.png")
        _plot_tsne(
            split_embedding,
            split_label,
            split_output,
            title=f"{split.capitalize()} split t-SNE",
            xlim=xlim,
            ylim=ylim,
        )
        offset += size


if __name__ == "__main__":
    main()
