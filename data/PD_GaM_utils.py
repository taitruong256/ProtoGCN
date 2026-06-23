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


def extract_mesh_data(record, body_model, frame_idx=0):
    """Tính toán và trích xuất vertices, faces từ dữ liệu pose, trans, beta cho một frame cụ thể."""
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

    # Lấy vertices và faces
    verts = getattr(body, 'v', getattr(body, 'verts', None))
    faces = getattr(body_model, 'f', getattr(body_model, 'faces', None))
    
    if verts is None or faces is None:
        raise ValueError("Không thể trích xuất vertices hoặc faces từ body model.")

    verts_frame = verts.detach().cpu().numpy()[frame_idx]
    faces_np = np.asarray(faces, dtype=np.int64)

    return verts_frame, faces_np


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


def par_args():
    """Hàm để parse các tham số từ command line nếu cần."""
    
    parser = argparse.ArgumentParser(description="Render 3D mesh từ dữ liệu PD-GaM.")
    parser.add_argument('--data_path', type=str, default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/data/CARE-PD/PD-GaM.pkl', help='Đường dẫn tới file dữ liệu pickle.')
    parser.add_argument('--preprocessing_path', type=str, default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/docs/CARE-PD/data/preprocessing', help='Đường dẫn tới thư mục preprocessing.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Đường dẫn tới file SMPL model. Nếu không cung cấp, sẽ sử dụng mặc định trong preprocessing.')
    parser.add_argument('--save_img_path', type=str, default='/home/taitruong256/taitruong/CCU/GaitXplain/docs/ProtoGCN_gait/data/CARE-PD/figures/pd_gam_frame0_mesh.png', help='Đường dẫn để lưu ảnh mesh.')
    parser.add_argument('--participant_id', type=str, default='001', help='ID của participant.')
    parser.add_argument('--seq_id', type=str, default='001-12-104704_wid01_0', help='ID của sequence.')
    parser.add_argument('--frame_idx', type=int, default=0, help='Chỉ số frame cần render.')
    
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
    
    print(f"Xử lý dữ liệu mesh cho frame {frame_idx}...")
    verts, faces = extract_mesh_data(record, body_model, frame_idx=frame_idx)
    
    # 5. Vẽ và lưu ảnh
    title = f"3D mesh - participant {participant_id} | sequence {seq_id} | frame {frame_idx}"
    render_3d_mesh(verts, faces, title=title, save_path=SAVE_IMG_PATH)


if __name__ == "__main__":
    main()
