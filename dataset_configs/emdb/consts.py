"""
EMDB dataset constants for WIMUSim.

EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild
- 9 subjects, indoor (EMDB_1) + outdoor (EMDB_2) sequences
- 6 EM-tracker / IMU sensors (60 Hz)
- 1 moving GoPro camera
- SMPL-X parameters provided; converted to SMPL for WIMUSim

Reference: Kaufmann et al., "EMDB: The Electromagnetic Database of Global 3D
           Human Pose and Shape in the Wild", CVPR 2023.
Dataset: https://ait.ethz.ch/emdb
"""

# Sample rates (Hz)
IMU_SAMPLE_RATE  = 60
SMPL_SAMPLE_RATE = 60

# -----------------------------------------------------------------------
# IMU placements (WIMUSim name → EMDB sensor label)
# EMDB sensors: left/right forearm, left/right lower leg, pelvis, upper back
# -----------------------------------------------------------------------
IMU_NAME_PAIRS = [
    ("LLA",  "left_forearm"),
    ("RLA",  "right_forearm"),
    ("LSH",  "left_lower_leg"),
    ("RSH",  "right_lower_leg"),
    ("PELV", "pelvis"),
    ("BACK", "upper_back"),
]
IMU_NAME_PAIRS_DICT = dict(IMU_NAME_PAIRS)
IMU_NAMES = list(IMU_NAME_PAIRS_DICT.keys())

# -----------------------------------------------------------------------
# SMPL joint that each IMU is attached to
# -----------------------------------------------------------------------
IMU_PARENT_JOINT = {
    "LLA":  "L_ELBOW",
    "RLA":  "R_ELBOW",
    "LSH":  "L_KNEE",
    "RSH":  "R_KNEE",
    "PELV": "BASE",
    "BACK": "SPINE3",
}

# -----------------------------------------------------------------------
# Subjects and sequences
# -----------------------------------------------------------------------
# EMDB_1: indoor, EMDB_2: outdoor
SUBJECT_LIST = [f"P{i}" for i in range(1, 10)]  # P1 – P9

# Suggested split — use EMDB_2 (outdoor) as test set
TRAIN_SPLIT = "EMDB_1"
TEST_SPLIT  = "EMDB_2"
