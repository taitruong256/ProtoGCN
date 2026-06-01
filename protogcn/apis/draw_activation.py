"""Activation visualization for ProtoGCN.

This module runs a test-time forward pass, projects the feature map onto the
classifier weights, and renders the resulting activation over the skeleton
sequence.
"""

from __future__ import annotations

import glob
import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkeletonLayout:
    connect_joint: np.ndarray
    extra_bones: Tuple[Tuple[int, int], ...] = ()


def _get_classifier_weight(model: torch.nn.Module) -> torch.Tensor:
    """Return the classifier weight used for activation projection."""
    if hasattr(model, "cls_head") and model.cls_head is not None and hasattr(model.cls_head, "fc_cls"):
        return model.cls_head.fc_cls.weight.detach().cpu()
    if hasattr(model, "backbone") and hasattr(model.backbone, "fcn"):
        return model.backbone.fcn.weight.detach().cpu()
    if hasattr(model, "fcn"):
        return model.fcn.weight.detach().cpu()
    raise AttributeError("Could not find classifier weights on the model.")


def _get_graph(model: torch.nn.Module):
    """Return a graph-like object from a recognizer or backbone."""
    if hasattr(model, "graph"):
        return model.graph
    if hasattr(model, "backbone") and hasattr(model.backbone, "graph"):
        return model.backbone.graph
    raise AttributeError("Could not find graph information on the model.")


def _build_connect_joint(graph) -> np.ndarray:
    """Infer the parent joint for each node from the graph layout."""
    parent = np.arange(graph.num_node, dtype=np.int64)

    if hasattr(graph, "inward"):
        for child, parent_joint in graph.inward:
            parent[child] = parent_joint
        return parent

    if hasattr(graph, "neighbor"):
        for child, parent_joint in graph.neighbor:
            if child != parent_joint:
                parent[child] = parent_joint
        return parent

    raise AttributeError("Graph does not expose inward or neighbor edges.")


def _layout_from_graph(graph) -> SkeletonLayout:
    """Create the visual layout used by the renderer."""
    connect_joint = _build_connect_joint(graph)
    extra_bones = ()

    if getattr(graph, "layout", "") == "coco":
        extra_bones = ((11, 12),)
    elif getattr(graph, "layout", "") in {"openpose", "openpose_new", "oumvlp"}:
        extra_bones = ((8, 11),)

    return SkeletonLayout(connect_joint=connect_joint, extra_bones=extra_bones)


def _unwrap_meta(meta):
    if meta is None:
        return None
    if hasattr(meta, "data"):
        return _unwrap_meta(meta.data)
    if isinstance(meta, (list, tuple)):
        if len(meta) == 1:
            return _unwrap_meta(meta[0])
        return [_unwrap_meta(item) for item in meta]
    return meta


def _sequence_name(meta=None, sample_name="sequence"):
    meta = meta or {}
    subject = meta.get("subject", meta.get("subject_id", sample_name))
    condition = meta.get("condition", meta.get("cond", "na"))
    sequence = meta.get("sequence", meta.get("seq", meta.get("seq_num", "na")))
    angle = meta.get("angle", meta.get("view", "na"))
    return f"{subject}-{condition}-{sequence}-{angle}"


def _normalize_activation(result: np.ndarray) -> np.ndarray:
    result = np.maximum(result, 0)
    max_value = float(np.max(result))
    if max_value > 0:
        result = result / max_value
    return result


def _select_class_map(
    activation: torch.Tensor,
    cls_score: Optional[torch.Tensor] = None,
    class_mode: str = "pred",
) -> np.ndarray:
    """Select one class map from ``(num_classes, T, V)`` activation maps."""
    if activation.dim() != 3:
        raise ValueError(f"Expected activation with shape (K, T, V), got {tuple(activation.shape)}")

    if class_mode == "max":
        class_map = activation.max(dim=0).values
        return class_map.detach().cpu().numpy()

    if class_mode != "pred":
        raise ValueError("class_mode must be one of: 'pred', 'max'.")

    if cls_score is None:
        raise ValueError("class_mode='pred' requires cls_score.")

    chosen_class = int(torch.argmax(cls_score).item())
    chosen_class = max(0, min(chosen_class, activation.size(0) - 1))
    class_map = activation[chosen_class]
    return class_map.detach().cpu().numpy()


def _upsample_activation(class_map: np.ndarray, target_t: int) -> np.ndarray:
    """Upsample activation on the temporal axis to match the input frames."""
    if class_map.shape[0] == target_t:
        return class_map

    tensor = torch.from_numpy(class_map[None, None].astype(np.float32))
    tensor = F.interpolate(tensor, size=(target_t, class_map.shape[1]), mode="bilinear", align_corners=False)
    return tensor[0, 0].cpu().numpy()


def _get_point_tensor(keypoint: torch.Tensor) -> torch.Tensor:
    """Convert input keypoints to ``(C, T, V)`` for a single person clip."""
    if keypoint.dim() == 6:
        keypoint = keypoint[:, 0]
    if keypoint.dim() != 5:
        raise ValueError(f"Expected keypoint with shape (N, M, T, V, C), got {tuple(keypoint.shape)}")
    if keypoint.size(0) != 1:
        raise ValueError("This helper expects a single-sample batch.")

    points = keypoint[0]
    if points.dim() != 4:
        raise ValueError(f"Expected per-sample keypoints with shape (M, T, V, C), got {tuple(points.shape)}")

    points = points.float().mean(dim=0)  # (T, V, C)
    points = points.permute(2, 0, 1).contiguous()  # (C, T, V)
    return points


def _split_channels(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if points.size(0) == 2:
        return points[0], points[1], None
    if points.size(0) >= 3:
        return points[0], points[1], points[2]
    raise ValueError(f"Unsupported point channels: {points.size(0)}")


def _prepare_clip_feature(clip_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert backbone output into head input and activation input."""
    if isinstance(clip_feat, (tuple, list)):
        clip_feat = clip_feat[0]

    if clip_feat.dim() == 5:
        feat_for_head = clip_feat
        feat_for_act = clip_feat[:, 0] if clip_feat.size(1) == 1 else clip_feat.mean(dim=1)
        feat_for_act = feat_for_act.squeeze(0)
        return feat_for_head, feat_for_act

    if clip_feat.dim() == 4:
        feat_for_head = clip_feat.unsqueeze(1)
        feat_for_act = clip_feat.squeeze(0)
        return feat_for_head, feat_for_act

    raise ValueError(f"Unexpected feature shape from backbone: {tuple(clip_feat.shape)}")


def draw_skeleton(
    result: np.ndarray,
    points: np.ndarray,
    graph,
    output_dir: str,
    sample_name: str,
    render_gif: bool = True,
    min_conf: float = 0.025,
    dpi: int = 96,
):
    layout = _layout_from_graph(graph)

    _, T, V = points.shape
    result = _normalize_activation(result)
    scalar_map = cmx.ScalarMappable(cmap=plt.get_cmap("plasma"), norm=colors.Normalize(vmin=0, vmax=1))

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
    png_dir = os.path.join(sample_dir, "png")
    pdf_dir = os.path.join(sample_dir, "pdf")
    gif_dir = os.path.join(sample_dir, "gif")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    fig.colorbar(scalar_map, ax=ax, fraction=0.045, pad=0.02)

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

            node_colors.append(scalar_map.to_rgba(r))
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
        fig.savefig(os.path.join(pdf_dir, f"{sample_name}-{t:03}.pdf"), bbox_inches="tight", pad_inches=0.1)
        fig.savefig(os.path.join(png_dir, f"{sample_name}-{t:03}.png"), bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)

    if render_gif:
        images = []
        for filename in sorted(glob.glob(os.path.join(png_dir, f"{sample_name}-*.png"))):
            images.append(imageio.imread(filename))
        if images:
            imageio.mimwrite(os.path.join(gif_dir, f"{sample_name}.gif"), images, duration=0.15)


def visualize_activation_batch(
    model: torch.nn.Module,
    batch,
    output_dir: str,
    sample_name: str,
    meta: Optional[dict] = None,
    class_mode: str = "pred",
    render_gif: bool = True,
    min_conf: float = 0.025,
    dpi: int = 96,
):
    """Visualize activation maps for one batch from the test loader."""
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

    with torch.no_grad():
        weight = _get_classifier_weight(model).to(device)
        graph = _get_graph(model)

        for i in range(keypoint.size(0)):
            seq = keypoint[i]
            # Visualize only one clip to avoid concatenating all test-time crops.
            seq = seq[:1]
            clips, persons, frames, _, _ = seq.shape
            seq_full = seq.permute(1, 0, 2, 3, 4).reshape(persons, clips * frames, seq.size(3), seq.size(4)).cpu()

            if isinstance(meta, (list, tuple)):
                seq_meta = meta[i] if i < len(meta) else meta[0]
            else:
                seq_meta = meta if meta is not None else {}
            seq_dir = _sequence_name(seq_meta, sample_name)

            logger.info(
                "Rendering sample %s: extracting features (%d clips, %d persons, %d frames/clip)",
                seq_dir,
                clips,
                persons,
                frames,
            )

            clip_feats = []
            for clip in seq:
                backbone_out = model.extract_feat(clip.unsqueeze(0))
                clip_feat = backbone_out[0] if isinstance(backbone_out, (tuple, list)) else backbone_out
                clip_feat_for_head, clip_feat_for_act = _prepare_clip_feature(clip_feat)
                clip_feats.append((clip_feat_for_head, clip_feat_for_act))

            logger.info("Rendering sample %s: projecting activation map", seq_dir)
            clip_scores = []
            clip_acts = []
            for clip_feat_for_head, clip_feat_for_act in clip_feats:
                clip_score = model.cls_head(clip_feat_for_head)
                clip_scores.append(clip_score.squeeze(0))
                clip_acts.append(torch.einsum("kc,ctv->ktv", weight, clip_feat_for_act))

            cls_score = torch.stack(clip_scores, dim=0)
            if hasattr(model, "average_clip"):
                seq_score = model.average_clip(cls_score.unsqueeze(0)).squeeze(0)
            else:
                seq_score = cls_score.mean(dim=0)

            activation = torch.cat(clip_acts, dim=1)
            class_map = _select_class_map(activation, cls_score=seq_score, class_mode=class_mode)
            class_map = _upsample_activation(class_map, seq_full.shape[1])

            logger.info("Rendering sample %s: drawing activation overlay", seq_dir)
            sample_points = _get_point_tensor(seq_full.unsqueeze(0))
            draw_skeleton(
                class_map,
                sample_points.numpy(),
                graph,
                output_dir=output_dir,
                sample_name=seq_dir,
                render_gif=render_gif,
                min_conf=min_conf,
                dpi=dpi,
            )


def visualize_test_loader(
    model: torch.nn.Module,
    data_loader,
    output_dir: str,
    class_mode: str = "pred",
    render_gif: bool = True,
    min_conf: float = 0.025,
    dpi: int = 96,
):
    """Run inference on the test loader and save activation maps."""
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
            tqdm.write(f"visualize: batch {batch_idx + 1}" if total_batches is None else f"visualize: batch {batch_idx + 1}/{total_batches}")
            meta = batch.get("img_metas") if isinstance(batch, dict) else None
            meta = _unwrap_meta(meta)
            visualize_activation_batch(
                model,
                batch,
                output_dir=output_dir,
                sample_name=f"{batch_idx:06}",
                meta=meta,
                class_mode=class_mode,
                render_gif=render_gif,
                min_conf=min_conf,
                dpi=dpi,
            )
    logger.info("Visualization completed")
