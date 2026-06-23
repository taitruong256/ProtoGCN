import sys
import types
import pickle
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
    with open(file_path, "rb") as f:
        return pickle.load(f)


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
