"""Grad-CAM visualization for ProtoGCN.

This module runs a test-time forward pass, computes Grad-CAM over the final
backbone feature map, and renders the resulting activation over the skeleton
sequence.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .draw_activation import (
    _get_graph,
    _get_point_tensor,
    _layout_from_graph,
    _sequence_name,
    _split_channels,
    _unwrap_meta,
)


logger = logging.getLogger(__name__)


def _minmax_normalize(result: np.ndarray) -> np.ndarray:
    """Scale activation map to ``[0, 1]`` after clipping negatives."""
    result = np.maximum(result, 0)
    min_value = float(np.min(result))
    max_value = float(np.max(result))
    if max_value > min_value:
        result = (result - min_value) / (max_value - min_value)
    else:
        result = np.zeros_like(result, dtype=np.float32)
    return result


def _repeat_temporal_bins(result: np.ndarray, target_t: int) -> np.ndarray:
    """Map a low-rate activation sequence to the original frame count.

    GaitGraph2 uses ``t // 4`` when the backbone reduces 100 frames to 25
    temporal bins. We mirror that behavior here: each activation bin is
    repeated across its corresponding original frames without averaging the
    skeleton motion itself.
    """
    if target_t <= 0:
        raise ValueError(f"target_t must be positive, got {target_t}")
    source_t = result.shape[0]
    if source_t == target_t:
        return result
    if source_t == 1:
        return np.repeat(result, target_t, axis=0)

    indices = np.floor(np.arange(target_t, dtype=np.float32) * source_t / target_t).astype(np.int64)
    indices = np.clip(indices, 0, source_t - 1)
    return result[indices]


def _figure_to_rgb_array(fig) -> np.ndarray:
    """Convert a matplotlib figure into an RGB frame."""
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return frame[..., :3].copy()


def _save_heatmap_gif(result: np.ndarray, output_dir: str, sample_name: str, dpi: int = 120):
    """Save a standalone Grad-CAM heatmap as a one-frame GIF."""
    result = _minmax_normalize(result)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    im = ax.imshow(result.T, aspect="auto", origin="lower", cmap="plasma", interpolation="nearest")
    ax.set_xlabel("time")
    ax.set_ylabel("joint")
    ax.set_title("Grad-CAM heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    frame = _figure_to_rgb_array(fig)
    plt.close(fig)
    imageio.mimsave(os.path.join(output_dir, f"{sample_name}.gif"), [frame], duration=0.2)


def _save_importance_summary(result: np.ndarray, output_dir: str, sample_name: str, dpi: int = 120):
    """Save a summary chart for joint/frame importance."""
    result = _minmax_normalize(result)
    os.makedirs(output_dir, exist_ok=True)

    joint_importance = _minmax_normalize(result.mean(axis=0))
    frame_importance = _minmax_normalize(result.mean(axis=1))
    joint_idx = np.arange(joint_importance.shape[0])
    frame_idx = np.arange(frame_importance.shape[0])

    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1])

    ax_heat = fig.add_subplot(grid[0, :])
    im = ax_heat.imshow(result.T, aspect="auto", origin="lower", cmap="plasma", interpolation="nearest")
    ax_heat.set_xlabel("frame")
    ax_heat.set_ylabel("joint")
    ax_heat.set_title("Grad-CAM joint-frame heatmap")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)

    ax_joint = fig.add_subplot(grid[1, 0])
    ax_joint.bar(joint_idx, joint_importance, color="#f28e2b")
    ax_joint.set_title("Joint importance")
    ax_joint.set_xlabel("joint")
    ax_joint.set_ylabel("importance")
    ax_joint.set_xticks(joint_idx)
    ax_joint.set_xticklabels([str(i) for i in joint_idx], rotation=0)
    ax_joint.set_xlim(-0.5, joint_importance.shape[0] - 0.5)

    ax_frame = fig.add_subplot(grid[1, 1])
    ax_frame.plot(frame_idx, frame_importance, color="#4e79a7", linewidth=2)
    ax_frame.set_title("Frame importance")
    ax_frame.set_xlabel("frame")
    ax_frame.set_ylabel("importance")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{sample_name}_summary.png"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _save_skeleton_gif(
    result: np.ndarray,
    points: np.ndarray,
    layout,
    output_dir: str,
    sample_name: str,
    render_gif: bool = True,
    min_conf: float = 0.025,
    target_t: Optional[int] = None,
    dpi: int = 96,
):
    """Render the skeleton overlay directly into a GIF without PNG intermediates."""
    result = _minmax_normalize(result)
    _, T, V = points.shape
    if target_t is None:
        target_t = T
    result = _repeat_temporal_bins(result, target_t)
    point_x, point_y, point_conf = _split_channels(torch.from_numpy(points))
    point_x = point_x.numpy()
    point_y = point_y.numpy()
    point_conf = point_conf.numpy() if point_conf is not None else np.ones_like(point_x)

    mean_pos = np.mean(np.mean(points[:2], -1), -1)
    all_x = points[0] - mean_pos[0]
    all_y = mean_pos[1] - points[1]
    xmin = np.min(all_x)
    xmax = np.max(all_x)
    ymin = np.min(all_y)
    ymax = np.max(all_y)
    width = xmax - xmin
    height = ymax - ymin
    max_range = max(width, height)
    pad = max_range * 0.25
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    sample_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    scalar_map = plt.get_cmap("plasma")
    norm = colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(plt.cm.ScalarMappable(cmap=scalar_map, norm=norm), ax=ax, fraction=0.045, pad=0.02)

    frames = []
    for t in range(T):
        ax.clear()
        ax.set_xlim(cx - max_range / 2 - pad, cx + max_range / 2 + pad)
        ax.set_ylim(cy - max_range / 2 - pad, cy + max_range / 2 + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"frame: {t}", fontsize=14)

        x = point_x[t, :] - mean_pos[0]
        y = mean_pos[1] - point_y[t, :]
        conf = point_conf[t, :]

        node_colors = []
        activation = []
        for v in range(V):
            k = int(layout.connect_joint[v])
            r = float(result[min(t, result.shape[0] - 1), v])
            activation.append(r)

            if conf[k] < min_conf or conf[v] < min_conf:
                node_colors.append([0, 0, 0, 0])
                continue

            node_colors.append(scalar_map(norm(r)))
            ax.plot(
                [x[v], x[k]],
                [y[v], y[k]],
                "-",
                c=[0.15, 0.15, 0.15],
                alpha=0.8,
                linewidth=5,
                zorder=1,
            )

        for a, b in layout.extra_bones:
            ax.plot(
                [x[a], x[b]],
                [y[a], y[b]],
                "-",
                c=[0.15, 0.15, 0.15],
                alpha=0.8,
                linewidth=5,
                zorder=1,
            )

        node_colors = np.asarray(node_colors, dtype=np.float32)
        node_sizes = np.asarray(activation, dtype=np.float32) * 650.0 + 60.0
        ax.scatter(x, y, marker="o", c=node_colors, s=node_sizes, zorder=3)
        fig.tight_layout()
        frames.append(_figure_to_rgb_array(fig))

    plt.close(fig)
    if render_gif and frames:
        imageio.mimsave(os.path.join(sample_dir, f"{sample_name}.gif"), frames, duration=0.15)


def _video_info_from_dataset(dataset, idx):
    """Fetch raw metadata from the underlying dataset."""
    base = dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    if hasattr(base, "datasets"):
        for sub_dataset in base.datasets:
            try:
                return _video_info_from_dataset(sub_dataset, idx)
            except Exception:
                continue
        raise IndexError(f"Could not resolve sample metadata for idx={idx}")
    if not hasattr(base, "video_infos"):
        raise AttributeError("Dataset does not expose video_infos")
    return base.video_infos[idx]


def _reduce_spatial_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, M, C, T, V)`` or ``(N, C, T, V)`` to ``(C, T, V)``."""
    if tensor.dim() == 5:
        if tensor.size(0) != 1:
            raise ValueError(f"Expected a single-sample batch, got {tuple(tensor.shape)}")
        tensor = tensor.mean(dim=1).squeeze(0)
        return tensor
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            raise ValueError(f"Expected a single-sample batch, got {tuple(tensor.shape)}")
        return tensor.squeeze(0)
    raise ValueError(f"Unexpected feature shape for Grad-CAM: {tuple(tensor.shape)}")


def _prepare_clip_feature(clip_feat: torch.Tensor) -> torch.Tensor:
    """Normalize backbone output to a tensor suitable for Grad-CAM."""
    if isinstance(clip_feat, (tuple, list)):
        clip_feat = clip_feat[0]

    if clip_feat.dim() == 5:
        return clip_feat
    if clip_feat.dim() == 4:
        return clip_feat.unsqueeze(1)

    raise ValueError(f"Unexpected feature shape from backbone: {tuple(clip_feat.shape)}")


def _compute_gradcam_maps(
    model: torch.nn.Module,
    feat: torch.Tensor,
    class_mode: str = "pred",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(activation, cls_score)`` for a single clip.

    ``activation`` has shape ``(K, T, V)`` and ``cls_score`` has shape ``(K,)``.
    """
    if feat.dim() not in {4, 5}:
        raise ValueError(f"Expected feature map with 4 or 5 dims, got {tuple(feat.shape)}")

    cls_score = model.cls_head(feat)
    if cls_score.dim() != 2 or cls_score.size(0) != 1:
        raise ValueError(f"Expected cls_score with shape (1, K), got {tuple(cls_score.shape)}")

    scores = cls_score.squeeze(0)
    feat_map = _reduce_spatial_dims(feat)
    num_classes = scores.size(0)
    maps = []

    if class_mode == "pred":
        target_classes = [int(torch.argmax(scores).item())]
    elif class_mode == "max":
        target_classes = list(range(num_classes))
    else:
        raise ValueError("class_mode must be one of: 'pred', 'max'.")

    for idx, class_idx in enumerate(target_classes):
        retain_graph = idx < len(target_classes) - 1
        score = scores[class_idx]
        grad = torch.autograd.grad(score, feat, retain_graph=retain_graph, create_graph=False)[0]
        grad_map = _reduce_spatial_dims(grad)
        alpha = grad_map.mean(dim=(1, 2))
        cam = torch.relu((alpha[:, None, None] * feat_map).sum(dim=0))
        maps.append(cam)

    activation = torch.stack(maps, dim=0)
    return activation, scores


def visualize_gradcam_batch(
    model: torch.nn.Module,
    batch,
    dataset,
    batch_idx: int,
    output_dir: str,
    sample_name: str,
    meta: Optional[dict] = None,
    class_mode: str = "pred",
    render_gif: bool = True,
    save_heatmap: bool = True,
    min_conf: float = 0.025,
    dpi: int = 96,
):
    """Visualize Grad-CAM maps for one batch from the test loader."""
    if isinstance(batch, dict):
        keypoint = batch["keypoint"]
        if meta is None and "img_metas" in batch:
            meta = _unwrap_meta(batch["img_metas"])
    else:
        keypoint = batch[0]

    if not torch.is_tensor(keypoint):
        keypoint = torch.as_tensor(keypoint)

    device = next(model.parameters()).device
    keypoint = keypoint.to(device)

    if keypoint.dim() != 6:
        raise ValueError(f"Expected keypoint with shape (N, num_clips, M, T, V, C), got {tuple(keypoint.shape)}")

    graph = _get_graph(model)
    layout = _layout_from_graph(graph)

    for i in range(keypoint.size(0)):
        seq = keypoint[i]
        seq = seq[:1]
        clips, persons, frames, _, _ = seq.shape
        seq_full = seq.permute(1, 0, 2, 3, 4).reshape(persons, clips * frames, seq.size(3), seq.size(4)).cpu()

        if not meta:
            try:
                seq_meta = _video_info_from_dataset(dataset, batch_idx + i)
            except Exception:
                seq_meta = {}
        elif isinstance(meta, (list, tuple)):
            seq_meta = meta[i] if i < len(meta) else meta[0]
        else:
            seq_meta = meta
        seq_dir = _sequence_name(seq_meta, sample_name)

        logger.info(
            "Rendering sample %s: extracting features (%d clips, %d persons, %d frames/clip)",
            seq_dir,
            clips,
            persons,
            frames,
        )

        clip_maps = []

        for clip in seq:
            with torch.enable_grad():
                backbone_out = model.extract_feat(clip.unsqueeze(0))
                clip_feat = _prepare_clip_feature(backbone_out)
                clip_feat = clip_feat.requires_grad_(True)
                activation, _scores = _compute_gradcam_maps(model, clip_feat, class_mode=class_mode)
                clip_maps.append(activation.detach())

        activation = torch.cat(clip_maps, dim=1)
        if class_mode == "pred":
            class_map = activation[0]
        else:
            class_map = activation.max(dim=0).values
        class_map = class_map.detach().cpu().numpy()
        class_map = _minmax_normalize(class_map)

        logger.info("Rendering sample %s: drawing activation overlay", seq_dir)
        sample_points = _get_point_tensor(seq_full.unsqueeze(0))
        summary_dir = os.path.join(output_dir, seq_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        _save_importance_summary(class_map, summary_dir, seq_dir, dpi=dpi)
        if save_heatmap:
            heatmap_dir = os.path.join(output_dir, seq_dir, "heatmap")
            os.makedirs(heatmap_dir, exist_ok=True)
            _save_heatmap_gif(class_map, heatmap_dir, seq_dir, dpi=dpi)
        _save_skeleton_gif(
            class_map,
            sample_points.numpy(),
            layout,
            output_dir=output_dir,
            sample_name=seq_dir,
            render_gif=render_gif,
            min_conf=min_conf,
            dpi=dpi,
        )


def visualize_gradcam_test_loader(
    model: torch.nn.Module,
    data_loader,
    output_dir: str,
    class_mode: str = "pred",
    render_gif: bool = True,
    save_heatmap: bool = True,
    min_conf: float = 0.025,
    dpi: int = 96,
):
    """Run inference on the test loader and save Grad-CAM maps."""
    model.eval()
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    logger.info(
        "Start visualize on test loader%s",
        f" ({total_batches} batches)" if total_batches is not None else "",
    )
    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(data_loader),
            total=total_batches,
            desc="visualize",
            dynamic_ncols=True,
            leave=False,
        ):
            tqdm.write(
                f"visualize: batch {batch_idx + 1}"
                if total_batches is None
                else f"visualize: batch {batch_idx + 1}/{total_batches}"
            )
            meta = batch.get("img_metas") if isinstance(batch, dict) else None
            meta = _unwrap_meta(meta)
            visualize_gradcam_batch(
                model,
                batch,
                getattr(data_loader, "dataset", None),
                batch_idx,
                output_dir=output_dir,
                sample_name=f"{batch_idx:06}",
                meta=meta,
                class_mode=class_mode,
                render_gif=render_gif,
                save_heatmap=save_heatmap,
                min_conf=min_conf,
                dpi=dpi,
            )
    logger.info("Visualization completed")
