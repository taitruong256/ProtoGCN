import sys
import types
import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import argparse


def setup_chumpy_shim():
    """Register a lightweight chumpy shim for loading legacy SMPL models."""
    if "chumpy" not in sys.modules:
        chumpy = types.ModuleType("chumpy")
        chumpy.ch = types.ModuleType("chumpy.ch")

        class _ChumpyArray:
            def __setstate__(self, state):
                self.__dict__.update(state if isinstance(state, dict) else {"state": state})

            def __array__(self, dtype=None):
                for key in ("r", "x", "_x", "data", "value"):
                    if key in self.__dict__:
                        return np.asarray(self.__dict__[key], dtype=dtype)
                return np.asarray([])

            @property
            def shape(self):
                return np.asarray(self).shape

            def __getitem__(self, item):
                return np.asarray(self)[item]

        chumpy.ch.Ch = _ChumpyArray
        sys.modules["chumpy"] = chumpy
        sys.modules["chumpy.ch"] = chumpy.ch


def load_pickle_data(file_path):
    """Load a pickle file."""
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = np.core.numeric
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _normalize_participant_id(participant_id):
    """Normalize participant IDs to the zero-padded PD-GaM format."""
    if isinstance(participant_id, str):
        return participant_id.zfill(3)
    return str(participant_id).zfill(3)


def _iter_split_records(split_data, source_data=None):
    """Yield records from either nested participant data or participant lists."""
    if isinstance(split_data, dict):
        participant_to_sequences = split_data
    else:
        if source_data is None:
            raise ValueError("source_data is required when fold split stores participant lists.")
        selected = {_normalize_participant_id(pid) for pid in split_data}
        participant_to_sequences = {
            _normalize_participant_id(pid): seqs
            for pid, seqs in source_data.items()
            if _normalize_participant_id(pid) in selected
        }

    for participant_id in sorted(participant_to_sequences.keys()):
        sequences = participant_to_sequences[participant_id]
        if not isinstance(sequences, dict):
            continue
        for sequence_id in sorted(sequences.keys()):
            record = sequences[sequence_id]
            if isinstance(record, dict):
                yield record


def count_fold_labels(fold_path, data_path=None, label_key="UPDRS_GAIT", num_classes=4):
    """Count labels for each train/eval split in every fold."""
    folds = load_pickle_data(fold_path)
    source_data = load_pickle_data(data_path) if data_path is not None else None
    class_ids = list(range(int(num_classes)))

    fold_counts = {}
    for fold_id in sorted(folds.keys()):
        fold_counts[fold_id] = {}
        for split_name in ("train", "eval"):
            split_data = folds[fold_id].get(split_name, {})
            labels = []
            missing = 0
            for record in _iter_split_records(split_data, source_data=source_data):
                if label_key in record:
                    labels.append(int(record[label_key]))
                else:
                    missing += 1

            counts = Counter(labels)
            fold_counts[fold_id][split_name] = {
                "counts": {class_id: counts.get(class_id, 0) for class_id in class_ids},
                "total": len(labels),
                "missing": missing,
            }
    return fold_counts


def plot_fold_label_distribution(
    fold_counts,
    output_dir,
    num_classes=4,
    dataset_name="PD-GaM",
    label_name="UPDRS_GAIT",
):
    """Save one train/eval UPDRS-label count chart for each fold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_ids = list(range(int(num_classes)))
    saved_paths = []

    for fold_id in sorted(fold_counts.keys()):
        train_counts = [fold_counts[fold_id]["train"]["counts"][class_id] for class_id in class_ids]
        eval_counts = [fold_counts[fold_id]["eval"]["counts"][class_id] for class_id in class_ids]

        x = np.arange(len(class_ids))
        width = 0.36
        max_count = max(train_counts + eval_counts + [1])

        fig, ax = plt.subplots(figsize=(8, 5))
        train_bars = ax.bar(x - width / 2, train_counts, width, label="train", color="#4C78A8")
        eval_bars = ax.bar(x + width / 2, eval_counts, width, label="eval", color="#F58518")

        ax.set_title(f"{dataset_name} fold {fold_id} {label_name} distribution")
        ax.set_xlabel(f"{label_name} label")
        ax.set_ylabel("Number of sequences")
        ax.set_xticks(x)
        ax.set_xticklabels([str(class_id) for class_id in class_ids])
        ax.set_ylim(0, max_count * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend()

        ax.bar_label(train_bars, padding=3, fontsize=9)
        ax.bar_label(eval_bars, padding=3, fontsize=9)
        fig.tight_layout()

        save_path = output_dir / f"fold_{fold_id}_label_distribution.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

        train_total = fold_counts[fold_id]["train"]["total"]
        eval_total = fold_counts[fold_id]["eval"]["total"]
        train_detail = fold_counts[fold_id]["train"]["counts"]
        eval_detail = fold_counts[fold_id]["eval"]["counts"]
        print(f"Fold {fold_id}:")
        print(f"  train total={train_total}, {label_name} counts={train_detail}")
        print(f"  eval  total={eval_total}, {label_name} counts={eval_detail}")
        print(f"  saved={save_path}")

    return saved_paths


def save_fold_label_distribution_plots(
    fold_path,
    output_dir,
    data_path=None,
    label_key="UPDRS_GAIT",
    num_classes=4,
):
    """Count labels and save six train/eval distribution charts."""
    fold_counts = count_fold_labels(
        fold_path,
        data_path=data_path,
        label_key=label_key,
        num_classes=num_classes,
    )
    return plot_fold_label_distribution(
        fold_counts,
        output_dir,
        num_classes=num_classes,
        label_name=label_key,
    )


@lru_cache(maxsize=1)
def load_smpl_body_model(model_path, preprocessing_path):
    """Load and cache the SMPL body model."""
    preprocessing_path = Path(preprocessing_path).resolve()
    if str(preprocessing_path) not in sys.path:
        sys.path.insert(0, str(preprocessing_path))

    try:
        from human_body_prior.body_model.body_model import BodyModel
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        fallback_path = repo_root / "docs" / "CARE-PD" / "data" / "preprocessing"
        if str(fallback_path) not in sys.path:
            sys.path.insert(0, str(fallback_path))
        from human_body_prior.body_model.body_model import BodyModel

    return BodyModel(bm_fname=str(model_path), num_betas=10)


def _prepare_smpl_inputs(record):
    """Prepare SMPL inputs for a full sequence."""
    pose = np.asarray(record['pose'], dtype=np.float32)
    trans = np.asarray(record['trans'], dtype=np.float32)
    beta = np.asarray(record['beta'], dtype=np.float32)

    if beta.ndim == 1:
        beta = beta[None, :]
    if beta.shape[0] != pose.shape[0]:
        beta = np.tile(beta, (pose.shape[0], 1))

    return (
        torch.from_numpy(pose[:, :3]).float(),
        torch.from_numpy(pose[:, 3:]).float(),
        torch.from_numpy(beta).float(),
        torch.from_numpy(trans).float(),
    )


def _forward_body_model(record, body_model):
    """Run SMPL forward pass for a full sequence."""
    root_orient, body_pose, betas, trans_t = _prepare_smpl_inputs(record)
    with torch.no_grad():
        return body_model(
            root_orient=root_orient,
            pose_body=body_pose,
            betas=betas,
            trans=trans_t,
        )


def extract_mesh_and_skeleton(record, body_model, frame_idx=0):
    """Extract vertices, faces, and joints for a single frame."""
    body = _forward_body_model(record, body_model)

    verts = getattr(body, 'v', getattr(body, 'verts', None))
    faces = getattr(body_model, 'f', getattr(body_model, 'faces', None))
    if verts is None or faces is None:
        raise ValueError("Unable to extract vertices or faces from the body model.")

    joints = getattr(body, 'Jtr', getattr(body, 'joints', None))
    if joints is None:
        raise ValueError("Unable to find joints in the body model.")

    return (
        verts.detach().cpu().numpy()[frame_idx],
        np.asarray(faces, dtype=np.int64),
        joints.detach().cpu().numpy()[frame_idx],
    )


def _flip_vertical_axis(points):
    """Flip all three axes to match the current render orientation."""
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 0] *= -1.0
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def _plot_gaitgen_floor(ax, points, span_scale=0.35, floor_color='#A8A8A8', alpha=0.6):
    """Plot a GAITGen-style XZ floor plane."""
    min_v = points.min(axis=0)
    max_v = points.max(axis=0)
    floor_y = min_v[1]
    xx, zz = np.meshgrid(
        [min_v[0] - (max_v[0] - min_v[0]) * span_scale, max_v[0] + (max_v[0] - min_v[0]) * span_scale],
        [min_v[2] - (max_v[2] - min_v[2]) * span_scale, max_v[2] + (max_v[2] - min_v[2]) * span_scale],
    )
    yy = np.full_like(xx, floor_y)
    ax.plot_surface(xx, yy, zz, color=floor_color, alpha=alpha, shade=False)
    return min_v, max_v, floor_y


def _set_gaitgen_view(ax, center, span, floor_y, z_center):
    """Apply the GAITGen camera and bounds."""
    half = span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(floor_y, floor_y + span)
    ax.set_zlim(z_center - half, z_center + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=110, azim=-90)
    ax.dist = 7.5
    ax.axis('off')
    ax.grid(False)
    return half


def _save_figure_as_gif_frames(fig):
    """Convert the current figure to an RGB frame."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def _build_output_path(base_path, participant_id=None, seq_id=None):
    """Build an output path under participant/sequence folders when IDs are provided."""
    base_path = Path(base_path)
    if participant_id is None or seq_id is None:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        return base_path

    output_dir = base_path.parent / str(participant_id) / str(seq_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / base_path.name


def render_3d_mesh(verts, faces, title="3D Mesh", save_path=None, participant_id=None, seq_id=None):
    """Render a 3D mesh and optionally save it."""
    verts = _flip_vertical_axis(verts)
    mesh = verts[faces]
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    poly = Poly3DCollection(mesh, facecolor='#9ca3af', edgecolor='none', alpha=0.35)
    ax.add_collection3d(poly)

    min_v = verts.min(axis=0)
    max_v = verts.max(axis=0)
    center = (max_v + min_v) / 2.0
    span = float(np.max(max_v - min_v))
    span = span if span > 1e-6 else 1.0

    _, _, floor_y = _plot_gaitgen_floor(ax, verts)
    _set_gaitgen_view(ax, center, span, floor_y, center[2])
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path = _build_output_path(save_path, participant_id=participant_id, seq_id=seq_id)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved mesh image to: {save_path}")
    plt.close(fig)
    


def render_skeleton(joints, title="SMPL Skeleton Only", save_path=None, participant_id=None, seq_id=None):
    """Render a skeleton and optionally save it."""
    joints = _flip_vertical_axis(joints)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    smpl_bones = [
        (0, 1), (0, 2), (0, 3),    # Pelvis -> L/R Hip, Spine1
        (1, 4), (2, 5), (3, 6),    # Hips -> Knees, Spine1 -> Spine2
        (4, 7), (5, 8), (6, 9),    # Knees -> Ankles, Spine2 -> Spine3
        (7, 10), (8, 11),          # Ankles -> Feet
        (9, 12), (9, 13), (9, 14), # Spine3 -> Neck, L/R Collar
        (12, 15),                  # Neck -> Head
        (13, 16), (14, 17),        # Collar -> Shoulders
        (16, 18), (17, 19),        # Shoulders -> Elbows
        (18, 20), (19, 21),        # Elbows -> Wrists
        (20, 22), (21, 23)         # Wrists -> Hands
    ]

    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='#e74c3c', s=40, depthshade=True, zorder=5)

    for bone in smpl_bones:
        if bone[0] < len(joints) and bone[1] < len(joints):
            start_joint = joints[bone[0]]
            end_joint = joints[bone[1]]
            ax.plot(
                [start_joint[0], end_joint[0]],
                [start_joint[1], end_joint[1]],
                [start_joint[2], end_joint[2]],
                color='#2c3e50',
                linewidth=3,
                zorder=4,
            )

    min_v = joints.min(axis=0)
    max_v = joints.max(axis=0)
    center = (max_v + min_v) / 2.0
    span = float(np.max(max_v - min_v))
    span = span if span > 1e-6 else 1.0

    _, _, floor_y = _plot_gaitgen_floor(ax, joints)
    _set_gaitgen_view(ax, center, span, floor_y, center[2])

    ax.set_title(title, pad=20)
    plt.tight_layout()

    if save_path:
        save_path = _build_output_path(save_path, participant_id=participant_id, seq_id=seq_id)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        print(f"Saved skeleton image to: {save_path}")
    plt.close(fig)
    


def _forward_smpl_sequence(record, body_model):
    """Forward SMPL for a full sequence."""
    body = _forward_body_model(record, body_model)
    verts = getattr(body, 'v', getattr(body, 'verts', None))
    faces = getattr(body_model, 'f', getattr(body_model, 'faces', None))
    joints = getattr(body, 'Jtr', getattr(body, 'joints', None))

    if verts is None or faces is None or joints is None:
        raise ValueError("Unable to extract vertices, faces, or joints from the body model.")

    return (
        verts.detach().cpu().numpy(),
        np.asarray(faces, dtype=np.int64),
        joints.detach().cpu().numpy(),
    )


def save_mesh_gif(record, body_model, gif_path, frame_stride=1, fps=12, participant_id=None, seq_id=None):
    """Save a mesh animation as a GIF."""
    verts_all, faces, _ = _forward_smpl_sequence(record, body_model)
    frame_ids = list(range(0, len(verts_all), max(1, int(frame_stride))))
    frames = []

    for frame_idx in frame_ids:
        verts = _flip_vertical_axis(verts_all[frame_idx])
        mesh = verts[faces]
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        poly = Poly3DCollection(mesh, facecolor='#d4a017', edgecolor='none', alpha=0.95)
        ax.add_collection3d(poly)

        min_v = verts.min(axis=0)
        max_v = verts.max(axis=0)
        center = (max_v + min_v) / 2.0
        span = float(np.max(max_v - min_v))
        span = span if span > 1e-6 else 1.0

        _, _, floor_y = _plot_gaitgen_floor(ax, verts, alpha=0.65)
        _set_gaitgen_view(ax, center, span, floor_y, center[2])
        ax.set_title(f"3D mesh | frame {frame_idx}")
        plt.tight_layout()

        frames.append(_save_figure_as_gif_frames(fig))
        plt.close(fig)

    gif_path = _build_output_path(gif_path, participant_id=participant_id, seq_id=seq_id)
    imageio.mimsave(gif_path, frames, duration=1.0 / max(int(fps), 1))
    print(f"Saved mesh GIF to: {gif_path}")


def save_skeleton_gif(record, body_model, gif_path, frame_stride=1, fps=12, participant_id=None, seq_id=None):
    """Save a skeleton animation as a GIF."""
    _, _, joints_all = _forward_smpl_sequence(record, body_model)
    frame_ids = list(range(0, len(joints_all), max(1, int(frame_stride))))
    frames = []

    smpl_bones = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (2, 5), (3, 6),
        (4, 7), (5, 8), (6, 9),
        (7, 10), (8, 11),
        (9, 12), (9, 13), (9, 14),
        (12, 15),
        (13, 16), (14, 17),
        (16, 18), (17, 19),
        (18, 20), (19, 21),
        (20, 22), (21, 23)
    ]

    for frame_idx in frame_ids:
        joints = _flip_vertical_axis(joints_all[frame_idx])
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='#e74c3c', s=40, depthshade=True, zorder=5)
        for bone in smpl_bones:
            if bone[0] < len(joints) and bone[1] < len(joints):
                start_joint = joints[bone[0]]
                end_joint = joints[bone[1]]
                ax.plot(
                    [start_joint[0], end_joint[0]],
                    [start_joint[1], end_joint[1]],
                    [start_joint[2], end_joint[2]],
                    color='#2c3e50', linewidth=3, zorder=4
                )

        min_v = joints.min(axis=0)
        max_v = joints.max(axis=0)
        center = (max_v + min_v) / 2.0
        span = float(np.max(max_v - min_v))
        span = span if span > 1e-6 else 1.0

        _, _, floor_y = _plot_gaitgen_floor(ax, joints, alpha=0.65)
        _set_gaitgen_view(ax, center, span, floor_y, center[2])
        ax.set_title(f"SMPL Skeleton Only | frame {frame_idx}", pad=20)
        plt.tight_layout()

        frames.append(_save_figure_as_gif_frames(fig))
        plt.close(fig)

    gif_path = _build_output_path(gif_path, participant_id=participant_id, seq_id=seq_id)
    imageio.mimsave(gif_path, frames, duration=1.0 / max(int(fps), 1))
    print(f"Saved skeleton GIF to: {gif_path}")


def save_skeleton_sequence_pkl(data_path, fold_path, body_model, pkl_path, frame_stride=1):
    """Save skeleton sequences while preserving the fold/split/participant structure."""
    data = load_pickle_data(data_path)
    folds = load_pickle_data(fold_path)

    enriched_folds = {}
    for fold_id, fold_info in tqdm(folds.items(), desc="Folds", total=len(folds)):
        enriched_folds[fold_id] = {}
        for split_name in ('train', 'eval'):
            participants = fold_info.get(split_name, [])
            enriched_folds[fold_id][split_name] = {}
            for participant_id in tqdm(participants, desc=f"Fold {fold_id} {split_name}", leave=False):
                participant_id = str(participant_id)
                if participant_id not in data:
                    continue
                enriched_folds[fold_id][split_name][participant_id] = {}
                for seq_id, record in tqdm(
                    data[participant_id].items(),
                    desc=f"Participant {participant_id}",
                    leave=False,
                ):
                    _, _, joints_all = _forward_smpl_sequence(record, body_model)
                    frame_ids = list(range(0, len(joints_all), max(1, int(frame_stride))))
                    skeleton_seq = np.asarray(
                        [_flip_vertical_axis(joints_all[i]) for i in frame_ids],
                        dtype=np.float32,
                    )
                    enriched_record = dict(record)
                    enriched_record['skeleton'] = skeleton_seq
                    enriched_folds[fold_id][split_name][participant_id][seq_id] = enriched_record

    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump(enriched_folds, f)
    print(f"Saved skeleton sequence pickle to: {pkl_path}")


def par_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render 3D mesh or skeleton from PD-GaM data.")
    parser.add_argument('--data_path', type=str, default='data/CARE-PD/PD-GaM.pkl', help='Pickle data path.')
    parser.add_argument('--fold_path', type=str, default='data/CARE-PD/folds/UPDRS_Datasets/PD-GaM_6fold_participants_skeleton.pkl', help='6-fold pickle path.')
    parser.add_argument('--label_key', type=str, default='UPDRS_GAIT', help='Label key used in PD-GaM records.')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of label classes to draw.')
    parser.add_argument('--label_plot_dir', type=str, default='data/CARE-PD/figures/label_distribution', help='Directory for fold label plots.')
    parser.add_argument('--plot_fold_labels', action='store_true', help='Only save six train/eval label-distribution charts and exit.')
    parser.add_argument('--preprocessing_path', type=str, default='docs/CARE-PD/data/preprocessing', help='Preprocessing directory.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='SMPL model path. Defaults to the bundled model.')
    parser.add_argument('--save_img_path', type=str, default='data/CARE-PD/figures/pd_gam_frame0_mesh.png', help='Output image path.')
    parser.add_argument('--participant_id', type=str, default='001', help='Participant ID.')
    parser.add_argument('--seq_id', type=str, default='001-12-104704_wid01_0', help='Sequence ID.')
    parser.add_argument('--frame_idx', type=int, default=0, help='Frame index to render.')
    return parser.parse_args()


def main():
    args = par_args()
    DATA_PATH = args.data_path
    PREPROCESSING_PATH = args.preprocessing_path
    SMPL_MODEL_PATH = args.smpl_model_path or f'{PREPROCESSING_PATH}/common/body_models/smpl/SMPL_NEUTRAL.pkl'
    SAVE_IMG_PATH = args.save_img_path

    if args.plot_fold_labels:
        save_fold_label_distribution_plots(
            args.fold_path,
            args.label_plot_dir,
            data_path=args.data_path,
            label_key=args.label_key,
            num_classes=args.num_classes,
        )
        return

    setup_chumpy_shim()

    print("Loading data...")
    data = load_pickle_data(DATA_PATH)
    body_model = load_smpl_body_model(SMPL_MODEL_PATH, PREPROCESSING_PATH)

    participant_id = args.participant_id
    seq_id = args.seq_id
    frame_idx = args.frame_idx
    record = data[participant_id][seq_id]

    print(f"Processing frame {frame_idx}...")
    verts, faces, joints = extract_mesh_and_skeleton(record, body_model, frame_idx=frame_idx)

    title = f"3D mesh - participant {participant_id} | sequence {seq_id} | frame {frame_idx}"
    render_3d_mesh(
        verts,
        faces,
        title=title,
        save_path=SAVE_IMG_PATH,
        participant_id=participant_id,
        seq_id=seq_id,
    )

    title = f"SMPL Skeleton Only - participant {participant_id} | frame {frame_idx}"
    save_skeleton_path = SAVE_IMG_PATH.replace('mesh.png', '_skeleton.png')
    render_skeleton(
        joints,
        title=title,
        save_path=save_skeleton_path,
        participant_id=participant_id,
        seq_id=seq_id,
    )

    save_skeleton_gif(
        record,
        body_model,
        'data/CARE-PD/figures/skeleton.gif',
        frame_stride=2,
        fps=12,
        participant_id=participant_id,
        seq_id=seq_id,
    )

    save_mesh_gif(
        record,
        body_model,
        'data/CARE-PD/figures/mesh.gif',
        frame_stride=2,
        fps=12,
        participant_id=participant_id,
        seq_id=seq_id,
    )

    save_skeleton_sequence_pkl(
        'data/CARE-PD/PD-GaM.pkl',
        'data/CARE-PD/folds/UPDRS_Datasets/PD-GaM_6fold_participants.pkl',
        body_model,
        'data/CARE-PD/folds/UPDRS_Datasets/PD-GaM_6fold_participants_skeleton.pkl',
        frame_stride=1,
    )

if __name__ == "__main__":
    main()
