"""
MoVi dataset constants for WIMUSim.

MoVi: A Large Multi-Purpose Human Motion and Video Dataset
- 90 subjects total (numbered 1-90)
- 21 activities × 1 take each
- 4 synchronized video cameras
- Vicon motion capture at 120 Hz → F_v3d_Subject_N.mat
- SMPL fits via AMASS/BMLmovi at 120 Hz → Subject_N_F_{seq}_poses.npz
  (F = Full body dataset marker set, not gender)

Data layout expected:
    amass_root/
        Subject_{N}_F_{seq}_poses.npz   seq = 1..21 (same order as V3D_MOTION_LIST)
                                         contains: poses(T,156), betas(16,), trans(T,3)

    v3d_root/
        F_v3d_Subject_{N}.mat

Reference: Ghorbani et al., "MoVi: A Large Multipurpose Motion and Video Dataset", 2021.
Dataset: https://www.biomotionlab.ca/movi/
"""

# Both v3d and AMASS BMLmovi are at 120 Hz
V3D_SAMPLE_RATE  = 120
SMPL_SAMPLE_RATE = 120

# -----------------------------------------------------------------------
# Activities (21 activities in v3d motions_list order)
# seq_id in AMASS = activity index + 1
# -----------------------------------------------------------------------
V3D_MOTION_LIST = [
    "kicking",
    "dancing_rm",
    "pointing",
    "hand_clapping",
    "jumping_jack",
    "stretching",
    "crossarms",
    "running_in_spot",
    "crawling",
    "walking",
    "hand_waving",
    "checking_watch",
    "sideways",
    "vertical_jumping",
    "sitting_down",
    "taking_photo",
    "cross_legged_sitting",
    "throw_catch",
    "jogging",
    "scratching_head",
    "phone_talking",
]

# -----------------------------------------------------------------------
# Subjects: integers 1-90 (all use _F_ designator in file names)
# -----------------------------------------------------------------------
ALL_SUBJECTS = list(range(1, 91))   # 90 subjects

# Suggested train/test split (no official split; adjust as needed)
TRAIN_SUBJECTS = list(range(1, 61))
TEST_SUBJECTS  = list(range(61, 91))

# -----------------------------------------------------------------------
# v3d segment names (15 body segments tracked in Vicon) and their
# mapping to WIMUSim IMU names.
# -----------------------------------------------------------------------
# v3d segment index → segment short name
V3D_SEGMENT_NAMES = [
    "RTA",   # 0 — Thorax/Sternum (centre)
    "RHE",   # 1 — Head (centre)
    "LAR",   # 2 — Left upper Arm
    "LFA",   # 3 — Left ForeArm
    "LHA",   # 4 — Left HAnd
    "RPV",   # 5 — Pelvis (centre)
    "LTH",   # 6 — Left THigh
    "LSK",   # 7 — Left ShanK
    "LFT",   # 8 — Left FooT
    "RAR",   # 9 — Right upper Arm
    "RFA",   # 10 — Right ForeArm
    "RHA",   # 11 — Right HAnd
    "RTH",   # 12 — Right THigh
    "RSK",   # 13 — Right ShanK
    "RFT",   # 14 — Right FooT
]

# v3d segment name → WIMUSim IMU name (15 of the 17 MoVi IMUs; RSHO/LSHO excluded)
V3D_SEG_TO_IMU = {
    "RHE": "HED",
    "RTA": "STER",
    "RPV": "PELV",
    "LAR": "LUA",
    "LFA": "LLA",
    "LHA": "LHD",
    "RAR": "RUA",
    "RFA": "RLA",
    "RHA": "RHD",
    "LTH": "LTH",
    "LSK": "LSH",
    "LFT": "LFT",
    "RTH": "RTH",
    "RSK": "RSH",
    "RFT": "RFT",
}
# WIMUSim IMU name → v3d segment index
IMU_TO_SEG_IDX = {
    imu: V3D_SEGMENT_NAMES.index(seg)
    for seg, imu in V3D_SEG_TO_IMU.items()
}

# All WIMUSim IMU names derivable from v3d kinematics
IMU_NAMES = list(V3D_SEG_TO_IMU.values())

# -----------------------------------------------------------------------
# SMPL joint that each IMU is attached to (for default P generation)
# -----------------------------------------------------------------------
IMU_PARENT_JOINT = {
    "HED":  "HEAD",
    "STER": "SPINE3",
    "PELV": "BASE",
    "LUA":  "L_SHOULDER",
    "LLA":  "L_ELBOW",
    "LHD":  "L_WRIST",
    "RUA":  "R_SHOULDER",
    "RLA":  "R_ELBOW",
    "RHD":  "R_WRIST",
    "LTH":  "L_HIP",
    "LSH":  "L_KNEE",
    "LFT":  "L_ANKLE",
    "RTH":  "R_HIP",
    "RSH":  "R_KNEE",
    "RFT":  "R_ANKLE",
}
