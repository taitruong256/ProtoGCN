import sys
import types
import pickle
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import argparse


def setup_chumpy_shim():
    """Tạo mock module cho chumpy để parse model SMPL cũ mà không cần cài đặt package."""
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
    """Đọc dữ liệu từ file pickle."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_smpl_body_model(model_path, preprocessing_path):
    """Load và cache SMPL body model để tránh phải load lại nhiều lần."""
    preprocessing_path = Path(preprocessing_path).resolve()
    if str(preprocessing_path) not in sys.path:
        sys.path.insert(0, str(preprocessing_path))

    try:
        from human_body_prior.body_model.body_model import BodyModel
    except ModuleNotFoundError:
        # Fallback an toàn nếu caller truyền nhầm path hoặc môi trường chưa set PYTHONPATH.
        repo_root = Path(__file__).resolve().parent.parent
        fallback_path = repo_root / "docs" / "CARE-PD" / "data" / "preprocessing"
        if str(fallback_path) not in sys.path:
            sys.path.insert(0, str(fallback_path))
        from human_body_prior.body_model.body_model import BodyModel
    
    return BodyModel(bm_fname=str(model_path), num_betas=10)


def extract_mesh_and_skeleton(record, body_model, frame_idx=0):
    """Tính toán và trích xuất cả vertices, faces VÀ joints (skeleton) từ dữ liệu pose, trans, beta."""
    pose = np.asarray(record['pose'], dtype=np.float32)
    trans = np.asarray(record['trans'], dtype=np.float32)
    beta = np.asarray(record['beta'], dtype=np.float32)
    
    if beta.ndim == 1:
        beta = beta[None, :]
    if beta.shape[0] != pose.shape[0]:
        beta = np.tile(beta, (pose.shape[0], 1))

    # Chuyển đổi sang tensor
    root_orient = torch.from_numpy(pose[:, :3]).float()
    body_pose = torch.from_numpy(pose[:, 3:]).float()
    betas = torch.from_numpy(beta).float()
    trans_t = torch.from_numpy(trans).float()

    # Forward qua body model
    with torch.no_grad():
        body = body_model(
            root_orient=root_orient,
            pose_body=body_pose,
            betas=betas,
            trans=trans_t,
        )

    # 1. Lấy vertices và faces
    verts = getattr(body, 'v', getattr(body, 'verts', None))
    faces = getattr(body_model, 'f', getattr(body_model, 'faces', None))
    
    if verts is None or faces is None:
        raise ValueError("Không thể trích xuất vertices hoặc faces từ body model.")

    verts_frame = verts.detach().cpu().numpy()[frame_idx]
    faces_np = np.asarray(faces, dtype=np.int64)

    # 2. Lấy joints (3D skeleton)
    joints = getattr(body, 'Jtr', getattr(body, 'joints', None))
    if joints is None:
        raise ValueError("Không tìm thấy dữ liệu joints trong body model.")
    joints_frame = joints.detach().cpu().numpy()[frame_idx]

    return verts_frame, faces_np, joints_frame


def render_3d_mesh(verts, faces, title="3D Mesh", save_path=None):
    """Render 3D mesh bằng matplotlib và tùy chọn lưu ra file."""
    mesh = verts[faces]
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    poly = Poly3DCollection(mesh, facecolor='#9ca3af', edgecolor='none', alpha=0.35)
    ax.add_collection3d(poly)

    # Tính toán bounding box để view mesh cân đối
    center = verts.mean(axis=0)
    span = np.max(verts.max(axis=0) - verts.min(axis=0))
    span = float(span if span > 1e-6 else 1.0)
    half = span / 2.0
    
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    
    # Thiết lập giao diện plot
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=-75)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Đã lưu ảnh mesh tại: {save_path}")
        
    plt.show()


def render_skeleton(joints, title="SMPL Skeleton Only", save_path=None):
    """Render chỉ bộ xương (Skeleton) từ dữ liệu Joints 3D."""


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 2. Định nghĩa các xương (Bones)
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

    # Vẽ khớp
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='#e74c3c', s=40, depthshade=True, zorder=5)

    # Vẽ xương
    for bone in smpl_bones:
        if bone[0] < len(joints) and bone[1] < len(joints):
            start_joint = joints[bone[0]]
            end_joint = joints[bone[1]]
            ax.plot([start_joint[0], end_joint[0]], 
                    [start_joint[1], end_joint[1]], 
                    [start_joint[2], end_joint[2]], color='#2c3e50', linewidth=3, zorder=4)

    # 3. Tính toán Bounding Box
    min_v = joints.min(axis=0)
    max_v = joints.max(axis=0)
    center = (max_v + min_v) / 2.0
    span = np.max(max_v - min_v)
    half = span / 2.0
    
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(min_v[2], min_v[2] + span)
    ax.set_box_aspect((1, 1, 1))
    
    # 4. Thêm mặt sàn
    floor_z = min_v[2] - 0.02
    floor_length = half * 1.5
    xx, yy = np.meshgrid(
        [center[0] - floor_length, center[0] + floor_length*2],
        [center[1] - floor_length, center[1] + floor_length]
    )
    zz = np.full_like(xx, floor_z)
    ax.plot_surface(xx, yy, zz, color='#A8A8A8', alpha=0.6, shade=False)

    # 5. Tùy chỉnh
    ax.view_init(elev=25, azim=-45)
    ax.axis('off')
    ax.set_title(title, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        print(f"Đã lưu ảnh Skeleton tại: {save_path}")
        
    plt.show()


def par_args():
    """Hàm để parse các tham số từ command line nếu cần."""
    
    parser = argparse.ArgumentParser(description="Render 3D mesh hoặc skeleton từ dữ liệu PD-GaM.")
    parser.add_argument('--data_path', type=str, default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/data/CARE-PD/PD-GaM.pkl', help='Đường dẫn tới file dữ liệu pickle.')
    parser.add_argument('--preprocessing_path', type=str, default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/docs/CARE-PD/data/preprocessing', help='Đường dẫn tới thư mục preprocessing.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Đường dẫn tới file SMPL model. Nếu không cung cấp, sẽ sử dụng mặc định trong preprocessing.')
    parser.add_argument('--save_img_path', type=str, default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/data/CARE-PD/figures/pd_gam_frame0_mesh.png', help='Đường dẫn để lưu ảnh kết quả.')
    parser.add_argument('--participant_id', type=str, default='001', help='ID của participant.')
    parser.add_argument('--seq_id', type=str, default='001-12-104704_wid01_0', help='ID của sequence.')
    parser.add_argument('--frame_idx', type=int, default=0, help='Chỉ số frame cần render.')
    
    # Cập nhật thêm argument để dễ chọn chế độ vẽ
    parser.add_argument('--render_mode', type=str, choices=['mesh', 'skeleton'], default='skeleton', help='Chọn render mesh hoặc skeleton.')
    
    return parser.parse_args()


def main():
    # 1. Khai báo các đường dẫn cố định (Constants)
    args = par_args()
    DATA_PATH = args.data_path
    PREPROCESSING_PATH = args.preprocessing_path
    SMPL_MODEL_PATH = args.smpl_model_path or f'{PREPROCESSING_PATH}/common/body_models/smpl/SMPL_NEUTRAL.pkl'
    SAVE_IMG_PATH = args.save_img_path

    # 2. Khởi tạo môi trường
    setup_chumpy_shim()
    
    # 3. Load dữ liệu và Model
    print("Đang load dữ liệu...")
    data = load_pickle_data(DATA_PATH)
    body_model = load_smpl_body_model(SMPL_MODEL_PATH, PREPROCESSING_PATH)
    
    # 4. Trích xuất thông tin cụ thể
    participant_id = args.participant_id
    seq_id = args.seq_id
    frame_idx = args.frame_idx
    record = data[participant_id][seq_id]
    
    print(f"Xử lý dữ liệu ({args.render_mode}) cho frame {frame_idx}...")
    verts, faces, joints = extract_mesh_and_skeleton(record, body_model, frame_idx=frame_idx)
    
    # 5. Vẽ và lưu ảnh dựa trên render_mode
    if args.render_mode == 'mesh':
        title = f"3D mesh - participant {participant_id} | sequence {seq_id} | frame {frame_idx}"
        render_3d_mesh(verts, faces, title=title, save_path=SAVE_IMG_PATH)
    elif args.render_mode == 'skeleton':
        title = f"SMPL Skeleton Only - participant {participant_id} | frame {frame_idx}"
        # Đổi tên file lưu nếu chạy skeleton để tránh đè ảnh cũ
        save_skeleton_path = SAVE_IMG_PATH.replace('mesh.png', '_skeleton.png')
        render_skeleton(joints, title=title, save_path=save_skeleton_path)


if __name__ == "__main__":
    main()