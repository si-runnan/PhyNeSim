"""
Utility functions for loading TotalCapture dataset.

TotalCapture data structure (expected):
    {root}/
        {subject}/                         e.g. s1/
            {activity}/                    e.g. walking1/
                imu.pkl                    IMU dict (see below)
                smpl.npz                   SMPL fits (see below)

imu.pkl keys:
    {TotalCapture_sensor_label}: {'acc': (T,3), 'gyro': (T,3)}

smpl.npz keys (standard SMPL):
    betas:         (10,)
    global_orient: (T, 3, 3)   rotation matrices
    body_pose:     (T, 23, 3, 3) rotation matrices

Note: TotalCapture does not officially provide SMPL fits. Obtain them by
running a fitting tool (e.g. SMPLify-X, ROMP) on the MoCap / video data,
or use community-provided fits, and save in the format above.
"""

import pickle
import pathlib
import numpy as np

from dataset_configs.totalcapture.consts import IMU_NAME_PAIRS_DICT


# ---------------------------------------------------------------------------
# SMPL parameter loading
# ---------------------------------------------------------------------------

def load_smpl_params(data_path: str, subject: str, activity: str):
    """
    Load SMPL parameters for a TotalCapture sequence.

    Returns:
        betas:         np.ndarray (10,)
        global_orient: np.ndarray (T, 3, 3)
        body_pose:     np.ndarray (T, 23, 3, 3)
        trans:         np.ndarray (T, 3)  root translation in metres
                       (zeros if not present in smpl.npz)
    """
    npz_path = pathlib.Path(data_path) / subject / activity / "smpl.npz"
    data  = np.load(str(npz_path))
    go    = data["global_orient"].astype(np.float32)
    T     = go.shape[0]
    trans = data["trans"].astype(np.float32) if "trans" in data else np.zeros((T, 3), dtype=np.float32)
    return (
        data["betas"].astype(np.float32),
        go,
        data["body_pose"].astype(np.float32),
        trans,
    )


# ---------------------------------------------------------------------------
# IMU data loading
# ---------------------------------------------------------------------------

def load_imu_data(data_path: str, subject: str, activity: str):
    """
    Load IMU data for a TotalCapture sequence.

    Returns:
        imu_dict: {wimusim_imu_name: (acc, gyro)}
                  acc, gyro: np.ndarray (T, 3), units m/s² and rad/s.
    """
    pkl_path = pathlib.Path(data_path) / subject / activity / "imu.pkl"
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    imu_dict = {}
    for ws_name, tc_label in IMU_NAME_PAIRS_DICT.items():
        if tc_label in raw:
            imu_dict[ws_name] = (
                raw[tc_label]["acc"].astype(np.float32),
                raw[tc_label]["gyro"].astype(np.float32),
            )
    return imu_dict


# ---------------------------------------------------------------------------
# Default Placement (P) parameters
# ---------------------------------------------------------------------------

def generate_default_placement_params(B_rp: dict) -> dict:
    """
    Generate default IMU placement parameters for TotalCapture's 13 IMU positions.
    """
    def _bone(p, c):
        return B_rp.get((p, c), np.zeros(3))

    rp, ro = {}, {}

    rp[("HEAD",      "HED")]  = np.array([0.0,  0.0,  0.05])
    ro[("HEAD",      "HED")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("SPINE3",    "STER")] = np.array([0.0,  0.1,  0.0])
    ro[("SPINE3",    "STER")] = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    rp[("BASE",      "PELV")] = np.array([0.0,  0.1,  0.0])
    ro[("BASE",      "PELV")] = np.deg2rad(np.array([-90.0, 0.0, -90.0]))

    rp[("R_SHOULDER","RUA")]  = _bone("R_SHOULDER", "R_ELBOW") * 0.5
    ro[("R_SHOULDER","RUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    rp[("L_SHOULDER","LUA")]  = _bone("L_SHOULDER", "L_ELBOW") * 0.5
    ro[("L_SHOULDER","LUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    rp[("R_ELBOW",   "RLA")]  = _bone("R_ELBOW", "R_WRIST") * 0.5
    ro[("R_ELBOW",   "RLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    rp[("L_ELBOW",   "LLA")]  = _bone("L_ELBOW", "L_WRIST") * 0.5
    ro[("L_ELBOW",   "LLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    rp[("R_WRIST",   "RHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("R_WRIST",   "RHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("L_WRIST",   "LHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("L_WRIST",   "LHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("R_HIP",     "RTH")]  = _bone("R_HIP", "R_KNEE") * 0.5
    ro[("R_HIP",     "RTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    rp[("L_HIP",     "LTH")]  = _bone("L_HIP", "L_KNEE") * 0.5
    ro[("L_HIP",     "LTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    rp[("R_KNEE",    "RSH")]  = _bone("R_KNEE", "R_ANKLE") * 0.5
    ro[("R_KNEE",    "RSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    rp[("L_KNEE",    "LSH")]  = _bone("L_KNEE", "L_ANKLE") * 0.5
    ro[("L_KNEE",    "LSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    return {"rp": rp, "ro": ro}


def generate_B_range(B_rp: dict, min_scale=0.85, max_scale=1.15,
                     near_zero=(-0.01, 0.01)) -> dict:
    return {
        k: np.array([
            sorted([e * min_scale, e * max_scale]) if abs(e) > 1e-4 else list(near_zero)
            for e in v
        ])
        for k, v in B_rp.items()
    }
