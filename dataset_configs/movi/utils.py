"""
Utility functions for loading MoVi dataset and converting to WIMUSim parameters.

Data layout:
    amass_root/
        Subject_{N}_F_{seq}_poses.npz   poses at 120 Hz (SMPL+H, T×156)
                                         also contains: betas(16,), trans(T,3)
                                         seq = 1..21 (V3D_MOTION_LIST order)

    v3d_root/
        F_v3d_Subject_{N}.mat            Vicon kinematics at 120 Hz

Both sources are at 120 Hz and frame-count aligned per activity.

SMPL+H poses.npz keys:
    poses:           (T, 156) axis-angle — [:3] global orient, [3:66] body joints 1-21
    trans:           (T, 3)  root translation
    betas:           (16,)   shape params
    mocap_framerate: scalar  (120.0)

v3d .mat structure:
    Subject_{N}_F['move']:
        jointsTranslation_v3d : (T_all, 15, 3)  mm, 120 Hz, all activities concat'd
        jointsAffine_v3d      : (T_all, 15, 4, 4) rotation+translation, 120 Hz
        flags120              : (21, 2)  1-indexed start/end frames per activity
        motions_list          : (21,)   activity names

IMU signals derived from v3d kinematics use the same conventions as WIMUSim:
    gravity vector  = [0, 0, -9.8] m/s²  (Z-up world frame)
    acc = R^T @ (d²p/dt² - g_world)      (specific force in body frame)
    gyro = R^T @ ω_world                  (angular velocity in body frame)
"""

from pathlib import Path

import numpy as np
import torch
import pytorch3d.transforms.rotation_conversions as rc

from dataset_configs.movi.consts import (
    V3D_SAMPLE_RATE, SMPL_SAMPLE_RATE,
    V3D_SEGMENT_NAMES, V3D_SEG_TO_IMU, IMU_TO_SEG_IDX,
    IMU_PARENT_JOINT,
)
from dataset_configs.smpl.utils import _to_rotmat, _rotmat_to_quat_wxyz


# ---------------------------------------------------------------------------
# SMPL parameter loading  (AMASS BMLmovi format)
# ---------------------------------------------------------------------------

def load_smpl_params(amass_root: str, subject_num: int, seq_id: int):
    """
    Load SMPL parameters from an AMASS BMLmovi poses file.

    Args:
        amass_root:  Root of AMASS BMLmovi download.
        subject_num: Subject number (1-90).
        seq_id:      Sequence index 1-21 (matches V3D_MOTION_LIST order).

    Returns:
        betas:         np.ndarray (10,)
        global_orient: np.ndarray (T, 3, 3)  rotation matrices
        body_pose:     np.ndarray (T, 23, 3, 3)  joints 1-23 rotation matrices
                       (joints 22-23 L_HAND/R_HAND set to identity)
        trans:         np.ndarray (T, 3)  root translation in metres
    """
    root = Path(amass_root)

    # Poses file also contains betas — no separate shape.npz needed
    poses_path = root / f"Subject_{subject_num}_F_{seq_id}_poses.npz"
    data  = np.load(str(poses_path))

    betas = data["betas"][:10].astype(np.float32)        # (10,)
    trans = data["trans"].astype(np.float32)             # (T, 3) metres
    poses = data["poses"].astype(np.float32)  # (T, 156)

    # global_orient: poses[:, :3] axis-angle → rotation matrix
    global_aa     = torch.tensor(poses[:, :3])
    global_orient = rc.axis_angle_to_matrix(global_aa).numpy()   # (T, 3, 3)

    # body joints 1-21: poses[:, 3:66]
    T = poses.shape[0]
    body_aa_21  = torch.tensor(poses[:, 3:66]).reshape(T, 21, 3)
    body_rot_21 = rc.axis_angle_to_matrix(
        body_aa_21.reshape(-1, 3)
    ).reshape(T, 21, 3, 3)                                         # (T, 21, 3, 3)

    # Pad joints 22-23 (L_HAND, R_HAND) with identity
    eye       = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 2, -1, -1)
    body_pose = torch.cat([body_rot_21, eye], dim=1).numpy()       # (T, 23, 3, 3)

    return betas, global_orient, body_pose, trans


# ---------------------------------------------------------------------------
# IMU data derived from v3d kinematics
# ---------------------------------------------------------------------------

_GRAVITY = np.array([0.0, 0.0, -9.8], dtype=np.float64)   # WIMUSim default, m/s²

# Kinematic chain: segment_index → parent_segment_index (-1 = root)
# RPV (5) is the root; all local affine matrices must be multiplied up to get
# global pose.  Order matters: parents must appear before children.
_SEG_PARENT = {
    5: -1,   # RPV  → root
    0:  5,   # RTA  → RPV
    1:  0,   # RHE  → RTA
    2:  0,   # LAR  → RTA
    3:  2,   # LFA  → LAR
    4:  3,   # LHA  → LFA
    9:  0,   # RAR  → RTA
   10:  9,   # RFA  → RAR
   11: 10,   # RHA  → RFA
    6:  5,   # LTH  → RPV
    7:  6,   # LSK  → LTH
    8:  7,   # LFT  → LSK
   12:  5,   # RTH  → RPV
   13: 12,   # RSK  → RTH
   14: 13,   # RFT  → RSK
}
# Topological order (parents before children)
_SEG_TOPO = [5, 0, 1, 2, 3, 4, 9, 10, 11, 6, 7, 8, 12, 13, 14]


def _global_affines(local_affine_act: np.ndarray) -> np.ndarray:
    """
    Convert local affine matrices to global (world-frame) affine matrices.

    In v3d data only the root segment (RPV, index 5) is stored in global
    coordinates.  All other segments are stored as local transforms relative
    to their parent.  Global pose is obtained by multiplying up the chain:
        global[child] = global[parent] @ local[child]

    Args:
        local_affine_act: (T, 15, 4, 4) local affine matrices for one activity.

    Returns:
        global_affine: (T, 15, 4, 4) world-frame affine matrices.
    """
    T = local_affine_act.shape[0]
    global_aff = np.zeros_like(local_affine_act)

    for seg_idx in _SEG_TOPO:
        parent = _SEG_PARENT[seg_idx]
        if parent == -1:
            # Root: already in global frame
            global_aff[:, seg_idx] = local_affine_act[:, seg_idx]
        else:
            # Chain: global[child] = global[parent] @ local[child]
            global_aff[:, seg_idx] = (
                global_aff[:, parent] @ local_affine_act[:, seg_idx]
            )

    return global_aff


def _compute_segment_imu(global_affine_seg: np.ndarray, dt: float) -> tuple:
    """
    Compute accelerometer and gyroscope signals from a segment's global affine.

    Matches WIMUSim's simulate_imu convention:
        acc  = R^T @ (d²p/dt² - g_world)   [specific force, body frame, m/s²]
        gyro = R^T @ ω_world                [angular velocity, body frame, rad/s]

    Args:
        global_affine_seg: (T, 4, 4) world-frame affine for one segment.
        dt:                Time step in seconds (1 / V3D_SAMPLE_RATE).

    Returns:
        acc:  (T, 3) float32 m/s²
        gyro: (T, 3) float32 rad/s
    """
    # Global position in metres (from affine translation column)
    p = global_affine_seg[:, :3, 3] / 1000.0     # (T, 3)

    # Global rotation matrix R: body-to-world
    R = global_affine_seg[:, :3, :3]              # (T, 3, 3)

    # ---- Accelerometer --------------------------------------------------
    a_world       = np.zeros_like(p)
    a_world[1:-1] = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / (dt ** 2)
    a_world[0]    = a_world[1]
    a_world[-1]   = a_world[-2]

    diff = a_world - _GRAVITY[None, :]
    acc  = np.einsum("tij,tj->ti", R.transpose(0, 2, 1), diff)   # (T, 3)

    # ---- Gyroscope -------------------------------------------------------
    dR         = np.zeros_like(R)
    dR[1:-1]   = (R[2:] - R[:-2]) / (2.0 * dt)
    dR[0]      = dR[1]
    dR[-1]     = dR[-2]

    Omega       = np.einsum("tij,tkj->tik", dR, R)
    omega_world = np.stack([Omega[:, 2, 1], Omega[:, 0, 2], Omega[:, 1, 0]], axis=-1)
    gyro        = np.einsum("tij,tj->ti", R.transpose(0, 2, 1), omega_world)  # (T, 3)

    return acc.astype(np.float32), gyro.astype(np.float32)


def load_imu_data(v3d_root: str, subject_num: int, activity_idx: int,
                  imu_names=None) -> dict:
    """
    Load IMU signals for one activity by computing them from Vicon kinematics.

    Signals are computed at 120 Hz (V3D_SAMPLE_RATE = SMPL_SAMPLE_RATE),
    matching WIMUSim's simulation output frame-for-frame.

    Args:
        v3d_root:     Root directory containing F_v3d_Subject_N.mat files.
        subject_num:  Subject number (1-90).
        activity_idx: Activity index 0-20 (into V3D_MOTION_LIST).
        imu_names:    Subset of IMU names to return (default: all 15).

    Returns:
        imu_dict: {wimusim_imu_name: (acc_np, gyro_np)}
                  acc/gyro shapes (T, 3), units m/s² and rad/s at 120 Hz.
    """
    import scipy.io

    mat_path = Path(v3d_root) / f"F_v3d_Subject_{subject_num}.mat"
    subject_key = f"Subject_{subject_num}_F"
    mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    move = mat[subject_key]["move"]

    flags  = move["flags120"]                        # (21, 2) 1-indexed
    start  = int(flags[activity_idx, 0]) - 1         # convert to 0-indexed
    end    = int(flags[activity_idx, 1])             # exclusive

    affine_all = move["jointsAffine_v3d"]            # (T_all, 15, 4, 4)
    affine_act = affine_all[start:end]               # (T_act, 15, 4, 4)

    # Convert local affines → global world-frame affines
    global_aff = _global_affines(affine_act)         # (T_act, 15, 4, 4)

    dt_120    = 1.0 / V3D_SAMPLE_RATE
    requested = imu_names if imu_names is not None else list(V3D_SEG_TO_IMU.values())
    imu_dict  = {}

    for imu_name in requested:
        seg_idx = IMU_TO_SEG_IDX.get(imu_name)
        if seg_idx is None:
            continue

        acc, gyro = _compute_segment_imu(global_aff[:, seg_idx], dt_120)
        imu_dict[imu_name] = (acc, gyro)

    return imu_dict


# ---------------------------------------------------------------------------
# Default Placement (P) parameters
# ---------------------------------------------------------------------------

def generate_default_placement_params(B_rp: dict) -> dict:
    """
    Generate default IMU placement parameters for MoVi's 15 v3d-derived IMU positions.

    Args:
        B_rp: bone vector dict {(parent, child): np.ndarray(3,)} from compute_B_from_beta.

    Returns:
        dict with "rp" and "ro" sub-dicts keyed by (parent_joint, imu_name).
    """
    rp, ro = {}, {}

    def _bone(p, c):
        return B_rp.get((p, c), np.zeros(3))

    # Head
    rp[("HEAD",       "HED")]  = np.array([0.0,  0.0,  0.05])
    ro[("HEAD",       "HED")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    # Sternum
    rp[("SPINE3",     "STER")] = np.array([0.0,  0.1,  0.0])
    ro[("SPINE3",     "STER")] = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    # Pelvis
    rp[("BASE",       "PELV")] = np.array([0.0,  0.1,  0.0])
    ro[("BASE",       "PELV")] = np.deg2rad(np.array([-90.0, 0.0, -90.0]))

    # Upper arms
    rp[("L_SHOULDER", "LUA")]  = _bone("L_SHOULDER", "L_ELBOW") * 0.5
    ro[("L_SHOULDER", "LUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    rp[("R_SHOULDER", "RUA")]  = _bone("R_SHOULDER", "R_ELBOW") * 0.5
    ro[("R_SHOULDER", "RUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    # Forearms
    rp[("L_ELBOW",    "LLA")]  = _bone("L_ELBOW", "L_WRIST") * 0.5
    ro[("L_ELBOW",    "LLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    rp[("R_ELBOW",    "RLA")]  = _bone("R_ELBOW", "R_WRIST") * 0.5
    ro[("R_ELBOW",    "RLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    # Hands
    rp[("L_WRIST",    "LHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("L_WRIST",    "LHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("R_WRIST",    "RHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("R_WRIST",    "RHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    # Thighs
    rp[("L_HIP",      "LTH")]  = _bone("L_HIP", "L_KNEE") * 0.5
    ro[("L_HIP",      "LTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    rp[("R_HIP",      "RTH")]  = _bone("R_HIP", "R_KNEE") * 0.5
    ro[("R_HIP",      "RTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    # Shins
    rp[("L_KNEE",     "LSH")]  = _bone("L_KNEE", "L_ANKLE") * 0.5
    ro[("L_KNEE",     "LSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    rp[("R_KNEE",     "RSH")]  = _bone("R_KNEE", "R_ANKLE") * 0.5
    ro[("R_KNEE",     "RSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    # Feet
    rp[("L_ANKLE",    "LFT")]  = np.array([0.1, 0.0, -0.03])
    ro[("L_ANKLE",    "LFT")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("R_ANKLE",    "RFT")]  = np.array([0.1, 0.0, -0.03])
    ro[("R_ANKLE",    "RFT")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    return {"rp": rp, "ro": ro}


# ---------------------------------------------------------------------------
# B range helper
# ---------------------------------------------------------------------------

def generate_B_range(B_rp: dict, min_scale=0.85, max_scale=1.15,
                     near_zero=(-0.01, 0.01)) -> dict:
    """Generate per-axis search ranges for each bone vector entry."""
    return {
        k: np.array([
            sorted([e * min_scale, e * max_scale]) if abs(e) > 1e-4 else list(near_zero)
            for e in v
        ])
        for k, v in B_rp.items()
    }
