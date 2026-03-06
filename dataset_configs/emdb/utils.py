"""
Utility functions for loading EMDB dataset.

EMDB data structure (expected):
    {root}/
        EMDB_1/                    indoor sequences
            {subject}/             e.g. P1/
                {sequence}.pkl     one file per sequence
        EMDB_2/                    outdoor sequences
            {subject}/
                {sequence}.pkl

Each .pkl file contains a dict with keys:
    'smpl':
        'betas':         (10,)
        'body_pose':     (T, 21, 3, 3)  SMPL body joints 1-21 rotation matrices
                         (EMDB provides SMPL-X body joints; first 21 match SMPL)
        'global_orient': (T, 3, 3)
        'transl':        (T, 3)
    'imu':
        'acc':  (T, N_imu, 3)   m/s², sensor order matches IMU_NAME_PAIRS
        'gyro': (T, N_imu, 3)   rad/s
    'fps': int

Note: EMDB originally provides SMPL-X parameters. The 'body_pose' above
refers to the 21 body joints shared between SMPL and SMPL-X (excluding
jaw, eyes, and hand joints). Joints 22-23 (L_HAND, R_HAND) are padded
with identity rotations.
"""

import pickle
import pathlib
import numpy as np

from dataset_configs.emdb.consts import IMU_NAMES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequence(pkl_path: str) -> dict:
    """Load a single EMDB sequence .pkl file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_smpl_params(pkl_path: str):
    """
    Load SMPL parameters from an EMDB sequence .pkl.

    Returns:
        betas:         np.ndarray (10,)
        global_orient: np.ndarray (T, 3, 3)
        body_pose:     np.ndarray (T, 23, 3, 3)
                       joints 22-23 padded with identity
    """
    data = load_sequence(pkl_path)
    smpl = data["smpl"]

    betas         = smpl["betas"].astype(np.float32)          # (10,)
    global_orient = smpl["global_orient"].astype(np.float32)  # (T, 3, 3)
    body_21       = smpl["body_pose"].astype(np.float32)      # (T, 21, 3, 3)

    T = body_21.shape[0]
    eye = np.eye(3, dtype=np.float32)[None, None].repeat(T, axis=0).repeat(2, axis=1)
    body_pose = np.concatenate([body_21, eye], axis=1)        # (T, 23, 3, 3)

    return betas, global_orient, body_pose


def load_imu_data(pkl_path: str) -> dict:
    """
    Load IMU data from an EMDB sequence .pkl.

    Returns:
        imu_dict: {wimusim_imu_name: (acc, gyro)}
                  acc, gyro: np.ndarray (T, 3)
    """
    data = load_sequence(pkl_path)
    imu  = data["imu"]

    acc_all  = imu["acc"].astype(np.float32)   # (T, N_imu, 3)
    gyro_all = imu["gyro"].astype(np.float32)  # (T, N_imu, 3)

    # Sensor order in EMDB matches IMU_NAMES order (LLA, RLA, LSH, RSH, PELV, BACK)
    return {
        name: (acc_all[:, i], gyro_all[:, i])
        for i, name in enumerate(IMU_NAMES)
    }


def iter_sequences(data_path: str, split: str = "EMDB_1"):
    """
    Iterate over all .pkl sequence files in a split.

    Yields:
        (subject, sequence_name, pkl_path)
    """
    root = pathlib.Path(data_path) / split
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for pkl_path in sorted(subject_dir.glob("*.pkl")):
            yield subject_dir.name, pkl_path.stem, str(pkl_path)


# ---------------------------------------------------------------------------
# Default Placement (P) parameters
# ---------------------------------------------------------------------------

def generate_default_placement_params(B_rp: dict) -> dict:
    """
    Generate default IMU placement parameters for EMDB's 6 IMU positions.
    """
    def _bone(p, c):
        return B_rp.get((p, c), np.zeros(3))

    rp, ro = {}, {}

    rp[("L_ELBOW", "LLA")]  = _bone("L_ELBOW", "L_WRIST") * 0.5
    ro[("L_ELBOW", "LLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    rp[("R_ELBOW", "RLA")]  = _bone("R_ELBOW", "R_WRIST") * 0.5
    ro[("R_ELBOW", "RLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    rp[("L_KNEE",  "LSH")]  = _bone("L_KNEE", "L_ANKLE") * 0.5
    ro[("L_KNEE",  "LSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    rp[("R_KNEE",  "RSH")]  = _bone("R_KNEE", "R_ANKLE") * 0.5
    ro[("R_KNEE",  "RSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    rp[("BASE",    "PELV")] = np.array([0.0, 0.1, 0.0])
    ro[("BASE",    "PELV")] = np.deg2rad(np.array([-90.0, 0.0, -90.0]))

    rp[("SPINE3",  "BACK")] = np.array([0.0, 0.1, 0.0])
    ro[("SPINE3",  "BACK")] = np.deg2rad(np.array([90.0, 180.0, 0.0]))

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
