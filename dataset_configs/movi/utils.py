"""
Utility functions for loading MoVi dataset and converting to WIMUSim parameters.

MoVi data structure (expected):
    {root}/
        {subject}/                        e.g. F_Subject01/
            {subject}_{activity}_poses.npz   AMASS-format SMPL fits
            {subject}_{activity}_v3d.pkl     IMU + mocap data

SMPL .npz keys (AMASS format):
    betas:          (16,)   shape parameters (use first 10)
    poses:          (T, 156) axis-angle rotations (SMPL-H format)
                    poses[:, :3]   = global_orient
                    poses[:, 3:66] = body joints 1-21 (axis-angle)
    trans:          (T, 3)  root translation
    mocap_framerate: float  (typically 60 Hz)

IMU v3d.pkl keys:
    'IMU': dict mapping sensor_label -> {'acc': (T,3), 'gyro': (T,3)}
           sensor labels match MoVi convention (e.g. 'right_upper_arm')
"""

import pickle
import numpy as np
import torch
import pytorch3d.transforms.rotation_conversions as rc

from dataset_configs.movi.consts import IMU_NAME_PAIRS_DICT, IMU_NAMES, IMU_PARENT_JOINT
from dataset_configs.smpl.utils import _to_rotmat, _rotmat_to_quat_wxyz


# ---------------------------------------------------------------------------
# SMPL parameter loading
# ---------------------------------------------------------------------------

def load_smpl_params(data_path: str, subject: str, activity: str, trial: int = 1):
    """
    Load SMPL parameters from a MoVi AMASS-format .npz file.

    Args:
        data_path:  Root directory of the MoVi dataset.
        subject:    Subject ID, e.g. "F_Subject01".
        activity:   Activity name, e.g. "walk".
        trial:      Trial number (1-5).

    Returns:
        betas:         np.ndarray (10,)
        global_orient: np.ndarray (T, 3, 3)  rotation matrices
        body_pose:     np.ndarray (T, 23, 3, 3)  rotation matrices
                       (joints 22-23 L_HAND/R_HAND set to identity)
    """
    import pathlib
    npz_path = (
        pathlib.Path(data_path)
        / subject
        / f"{subject}_{activity}{trial:02d}_poses.npz"
    )
    data = np.load(str(npz_path))

    betas = data["betas"][:10].astype(np.float32)          # (10,)
    poses = data["poses"].astype(np.float32)                # (T, 156)

    # global_orient: poses[:, :3] axis-angle → rotation matrix
    global_aa = torch.tensor(poses[:, :3])                 # (T, 3)
    global_orient = rc.axis_angle_to_matrix(global_aa).numpy()  # (T, 3, 3)

    # body joints 1-21: poses[:, 3:66]
    T = poses.shape[0]
    body_aa_21 = torch.tensor(poses[:, 3:66]).reshape(T, 21, 3)  # (T, 21, 3)
    body_rot_21 = rc.axis_angle_to_matrix(
        body_aa_21.reshape(-1, 3)
    ).reshape(T, 21, 3, 3)                                       # (T, 21, 3, 3)

    # Pad joints 22-23 (L_HAND, R_HAND) with identity
    eye = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 2, -1, -1)
    body_pose = torch.cat([body_rot_21, eye], dim=1).numpy()     # (T, 23, 3, 3)

    return betas, global_orient, body_pose


# ---------------------------------------------------------------------------
# IMU data loading
# ---------------------------------------------------------------------------

def load_imu_data(data_path: str, subject: str, activity: str, trial: int = 1):
    """
    Load raw IMU data from a MoVi v3d .pkl file.

    Returns:
        imu_dict: {wimusim_imu_name: (acc, gyro)}
                  acc, gyro are np.ndarray of shape (T, 3), units m/s² and rad/s.
    """
    import pathlib
    pkl_path = (
        pathlib.Path(data_path)
        / subject
        / f"{subject}_{activity}{trial:02d}_v3d.pkl"
    )
    with open(pkl_path, "rb") as f:
        v3d = pickle.load(f, encoding="latin1")

    imu_raw = v3d["IMU"]  # dict: movi_label -> {'acc': ..., 'gyro': ...}

    imu_dict = {}
    for ws_name, movi_label in IMU_NAME_PAIRS_DICT.items():
        if movi_label in imu_raw:
            imu_dict[ws_name] = (
                imu_raw[movi_label]["acc"].astype(np.float32),   # (T, 3)
                imu_raw[movi_label]["gyro"].astype(np.float32),  # (T, 3)
            )
    return imu_dict


# ---------------------------------------------------------------------------
# Default Placement (P) parameters
# ---------------------------------------------------------------------------

def generate_default_placement_params(B_rp: dict) -> dict:
    """
    Generate default IMU placement parameters for MoVi's 17 IMU positions.

    Args:
        B_rp: bone vector dict {(parent, child): np.ndarray(3,)} from compute_B_from_beta.

    Returns:
        dict with "rp" and "ro" sub-dicts keyed by (parent_joint, imu_name).
    """
    rp, ro = {}, {}

    def _bone(p, c):
        return B_rp.get((p, c), np.zeros(3))

    # Head — on top of the head joint, slightly forward
    rp[("HEAD",      "HED")]  = np.array([0.0,  0.0,  0.05])
    ro[("HEAD",      "HED")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    # Sternum — mid-chest
    rp[("SPINE3",    "STER")] = np.array([0.0,  0.1,  0.0])
    ro[("SPINE3",    "STER")] = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    # Pelvis
    rp[("BASE",      "PELV")] = np.array([0.0,  0.1,  0.0])
    ro[("BASE",      "PELV")] = np.deg2rad(np.array([-90.0, 0.0, -90.0]))

    # Shoulders (clavicle segments)
    rp[("R_COLLAR",  "RSHO")] = _bone("R_COLLAR", "R_SHOULDER") * 0.5
    ro[("R_COLLAR",  "RSHO")] = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("L_COLLAR",  "LSHO")] = _bone("L_COLLAR", "L_SHOULDER") * 0.5
    ro[("L_COLLAR",  "LSHO")] = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    # Upper arms
    rp[("R_SHOULDER","RUA")]  = _bone("R_SHOULDER", "R_ELBOW") * 0.5
    ro[("R_SHOULDER","RUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    rp[("L_SHOULDER","LUA")]  = _bone("L_SHOULDER", "L_ELBOW") * 0.5
    ro[("L_SHOULDER","LUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    # Forearms
    rp[("R_ELBOW",   "RLA")]  = _bone("R_ELBOW", "R_WRIST") * 0.5
    ro[("R_ELBOW",   "RLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    rp[("L_ELBOW",   "LLA")]  = _bone("L_ELBOW", "L_WRIST") * 0.5
    ro[("L_ELBOW",   "LLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    # Hands
    rp[("R_WRIST",   "RHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("R_WRIST",   "RHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("L_WRIST",   "LHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("L_WRIST",   "LHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    # Thighs
    rp[("R_HIP",     "RTH")]  = _bone("R_HIP", "R_KNEE") * 0.5
    ro[("R_HIP",     "RTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    rp[("L_HIP",     "LTH")]  = _bone("L_HIP", "L_KNEE") * 0.5
    ro[("L_HIP",     "LTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    # Shins
    rp[("R_KNEE",    "RSH")]  = _bone("R_KNEE", "R_ANKLE") * 0.5
    ro[("R_KNEE",    "RSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    rp[("L_KNEE",    "LSH")]  = _bone("L_KNEE", "L_ANKLE") * 0.5
    ro[("L_KNEE",    "LSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    # Feet
    rp[("R_ANKLE",   "RFT")]  = np.array([0.1, 0.0, -0.03])
    ro[("R_ANKLE",   "RFT")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("L_ANKLE",   "LFT")]  = np.array([0.1, 0.0, -0.03])
    ro[("L_ANKLE",   "LFT")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    return {"rp": rp, "ro": ro}


# ---------------------------------------------------------------------------
# B range helper (same logic as before)
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
