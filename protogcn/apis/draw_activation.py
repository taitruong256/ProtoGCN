"""Activation visualization for ProtoGCN.

This module adapts the idea from GaitGraph2's ``draw_activation.py`` to the
ProtoGCN pipeline:

- use the backbone feature map before pooling
- project features with the classifier weights
- render the activation on top of the skeleton sequence

The implementation is intentionally model-agnostic enough to work with the
current ``RecognizerGCN`` + ``ProtoGCN`` stack, while still supporting the
GaitGraph-style backbone/classifier separation.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
matplotlib.use("Agg")


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

    # Fallback: derive a tree from the neighbor structure if necessary.
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

    # Keep a few anatomy-friendly cross links that help readability.
    if getattr(graph, "layout", "") == "coco":
        extra_bones = ((11, 12),)
    elif getattr(graph, "layout", "") in {"openpose", "openpose_new"}:
        extra_bones = ((8, 11),)
    elif getattr(graph, "layout", "") == "oumvlp":
        extra_bones = ((8, 11),)

    return SkeletonLayout(connect_joint=connect_joint, extra_bones=extra_bones)


def _sequence_name(meta=None, gt_label=None, pred_label=None, sample_name="sequence"):
    meta = meta or {}
    subject = meta.get("subject", meta.get("subject_id", sample_name))
    condition = meta.get("condition", meta.get("cond", "na"))
    angle = meta.get("angle", meta.get("view", meta.get("seq_num", "na")))
    gt_text = "na" if gt_label is None else str(int(gt_label))
    pred_text = "na" if pred_label is None else str(int(pred_label))
    return f"{subject}-{condition}-{angle}-gt{gt_text}-pred{pred_text}"


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


def _normalize_activation(result: np.ndarray) -> np.ndarray:
    result = np.maximum(result, 0)
    max_value = float(np.max(result))
    if max_value > 0:
        result = result / max_value
    return result


def _select_class_map(
    activation: torch.Tensor,
    cls_score: Optional[torch.Tensor] = None,
    label: Optional[int] = None,
    class_mode: str = "pred",
) -> Tuple[np.ndarray, Optional[int]]:
    """Select one class map from ``(num_classes, T, V)`` activation maps."""
    if activation.dim() != 3:
        raise ValueError(f"Expected activation with shape (K, T, V), got {tuple(activation.shape)}")

    chosen_class = None
    if class_mode == "max":
        class_map = activation.max(dim=0).values
    else:
        if class_mode == "label":
            if label is None:
                raise ValueError("class_mode='label' requires label.")
            chosen_class = int(label)
        elif class_mode == "pred":
            if cls_score is None:
                raise ValueError("class_mode='pred' requires cls_score.")
            chosen_class = int(torch.argmax(cls_score).item())
        else:
            raise ValueError("class_mode must be one of: 'pred', 'label', 'max'.")

        chosen_class = max(0, min(chosen_class, activation.size(0) - 1))
        class_map = activation[chosen_class]

    return class_map.detach().cpu().numpy(), chosen_class


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
        # (N, num_clips, M, T, V, C)
        keypoint = keypoint[:, 0]
    if keypoint.dim() != 5:
        raise ValueError(f"Expected keypoint with shape (N, M, T, V, C), got {tuple(keypoint.shape)}")
    if keypoint.size(0) != 1:
        raise ValueError("This helper expects a single-sample batch.")

    points = keypoint[0]
    if points.dim() != 4:
        raise ValueError(f"Expected per-sample keypoints with shape (M, T, V, C), got {tuple(points.shape)}")

    # Average across persons if multiple detections are present.
    points = points.float().mean(dim=0)  # (T, V, C)
    points = points.permute(2, 0, 1).contiguous()  # (C, T, V)
    return points


def _split_channels(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if points.size(0) == 2:
        return points[0], points[1], None
    if points.size(0) >= 3:
        return points[0], points[1], points[2]
    raise ValueError(f"Unsupported point channels: {points.size(0)}")


def draw_skeleton(
    result: np.ndarray,
    points: np.ndarray,
    label: Sequence[int] | int,
    graph,
    output_dir: str,
    sample_name: str,
    pause: float = 0.01,
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

    sample_dir = os.path.join(output_dir, sample_name)
    png_dir = os.path.join(sample_dir, "png")
    pdf_dir = os.path.join(sample_dir, "pdf")
    gif_dir = os.path.join(sample_dir, "gif")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    plt.figure(figsize=(1000 / dpi, 1000 / dpi), dpi=dpi)
    plt.colorbar(scalar_map, shrink=0.25, aspect=5)
    plt.ion()

    if isinstance(label, tuple) and len(label) == 2:
        label_text = f"gt: {label[0]}, pred: {label[1]}"
    elif isinstance(label, int):
        label_text = f"label: {label}"
    else:
        label_text = f"label: {tuple(int(x) for x in label)}"

    for t in range(T):
        plt.cla()
        plt.xlim(-450, 450)
        plt.ylim(-450, 450)
        plt.axis("off")
        plt.title(f"{label_text}, frame: {t}")

        x = point_x[t, :] - mean_pos[0]
        y = mean_pos[1] - point_y[t, :]
        conf = point_conf[t, :]

        c = []
        activation = []
        for v in range(V):
            k = int(layout.connect_joint[v])
            r = float(result[min(t, result.shape[0] - 1), v])
            activation.append(r)

            if conf[k] < min_conf or conf[v] < min_conf:
                c.append([0, 0, 0, 0])
                continue

            c.append(scalar_map.to_rgba(r))
            plt.plot([x[v], x[k]], [y[v], y[k]], "-", c=np.array([0.1, 0.1, 0.1]), alpha=0.5, linewidth=3, markersize=0)

        for a, b in layout.extra_bones:
            plt.plot([x[a], x[b]], [y[a], y[b]], "-", c=np.array([0.1, 0.1, 0.1]), alpha=0.5, linewidth=3, markersize=0)

        c = np.asarray(c, dtype=np.float32)
        s = np.asarray(activation, dtype=np.float32) * 128.0
        plt.scatter(x, y, marker="o", c=c, s=s, zorder=2.5)

        plt.savefig(os.path.join(pdf_dir, f"{sample_name}-{t:03}.pdf"))
        plt.savefig(os.path.join(png_dir, f"{sample_name}-{t:03}.png"))

    plt.ioff()
    plt.close()

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
    """Visualize a sequence of ProtoGCN/RecognizerGCN inputs."""
    if isinstance(batch, dict):
        keypoint = batch["keypoint"]
        label = batch.get("label")
        if meta is None and "img_metas" in batch:
            meta = _unwrap_meta(batch["img_metas"])
    else:
        keypoint, label = batch[:2]

    if not torch.is_tensor(keypoint):
        keypoint = torch.as_tensor(keypoint)
    if torch.is_tensor(label):
        label_values = [int(x) for x in label.flatten().tolist()]
    elif label is None:
        label_values = [None] * keypoint.shape[0]
    else:
        label_values = [int(x) for x in np.asarray(label).flatten().tolist()]

    device = next(model.parameters()).device
    keypoint = keypoint.to(device)

    if keypoint.dim() != 6:
        raise ValueError(f"Expected keypoint with shape (N, num_clips, M, T, V, C), got {tuple(keypoint.shape)}")

    with torch.no_grad():
        weight = _get_classifier_weight(model)
        graph = _get_graph(model)
        results = []
        for i in range(keypoint.size(0)):
            label_value = label_values[i] if i < len(label_values) else label_values[0]
            seq = keypoint[i]
            clips, persons, frames, _, _ = seq.shape
            seq_full = seq.permute(1, 0, 2, 3, 4).reshape(persons, clips * frames, seq.size(3), seq.size(4)).cpu()

            clip_feats = []
            for clip in seq:
                backbone_out = model.extract_feat(clip.unsqueeze(0))
                clip_feat = backbone_out[0] if isinstance(backbone_out, (tuple, list)) else backbone_out
                if clip_feat.dim() == 5:
                    clip_feat = clip_feat[:, 0] if clip_feat.size(1) == 1 else clip_feat.mean(dim=1)
                clip_feats.append(clip_feat.squeeze(0))

            clip_scores = []
            clip_acts = []
            for clip_feat in clip_feats:
                clip_score = model.cls_head(clip_feat.unsqueeze(0))
                clip_scores.append(clip_score.squeeze(0))
                clip_acts.append(torch.einsum("kc,ctv->ktv", weight, clip_feat))

            cls_score = torch.stack(clip_scores, dim=0)
            if hasattr(model, "average_clip"):
                seq_score = model.average_clip(cls_score.unsqueeze(0)).squeeze(0)
            else:
                seq_score = cls_score.mean(dim=0)

            activation = torch.cat(clip_acts, dim=1)
            pred_label = int(torch.argmax(seq_score).item())
            class_map, chosen_class = _select_class_map(
                activation,
                cls_score=seq_score,
                label=label_value,
                class_mode=class_mode,
            )

            class_map = _upsample_activation(class_map, seq_full.shape[1])
            if isinstance(meta, (list, tuple)):
                seq_meta = meta[i] if i < len(meta) else meta[0]
            else:
                seq_meta = meta if meta is not None else {}
            seq_dir = _sequence_name(seq_meta, label_value, pred_label, sample_name)
            sample_points = _get_point_tensor(seq_full.unsqueeze(0))
            draw_skeleton(
                class_map,
                sample_points.numpy(),
                (label_value if label_value is not None else -1, pred_label),
                graph,
                output_dir=output_dir,
                sample_name=seq_dir,
                render_gif=render_gif,
                min_conf=min_conf,
                dpi=dpi,
            )
            results.append({"label": label_value, "pred": pred_label, "chosen_class": chosen_class})

    return results if len(results) > 1 else results[0]


def visualize_test_loader(
    model: torch.nn.Module,
    data_loader,
    output_dir: str,
    class_mode: str = "pred",
    render_gif: bool = True,
    min_conf: float = 0.025,
    dpi: int = 96,
):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            meta = batch.get("img_metas") if isinstance(batch, dict) else None
            meta = _unwrap_meta(meta)
            results.append(
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
            )
    return results
