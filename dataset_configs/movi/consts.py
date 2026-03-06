"""
MoVi dataset constants for WIMUSim.

MoVi: A Large Multi-Purpose Human Motion and Video Dataset
- 90 subjects × 20 activities × 5 trials
- 17 Xsens MTw Awinda IMUs (100 Hz)
- 4 synchronized video cameras
- SMPL fits in AMASS-compatible format (60 Hz)

Reference: Ghorbani et al., "MoVi: A Large Multipurpose Motion and Video Dataset", 2021.
Dataset: https://www.biomotionlab.ca/movi/
"""

# IMU sample rate (Hz)
IMU_SAMPLE_RATE = 100

# SMPL/mocap sample rate (Hz)
SMPL_SAMPLE_RATE = 60

# -----------------------------------------------------------------------
# IMU placements (WIMUSim name → MoVi sensor label)
# -----------------------------------------------------------------------
# WIMUSim uses short uppercase codes; MoVi uses descriptive strings.
IMU_NAME_PAIRS = [
    ("HED",  "head"),
    ("STER", "sternum"),
    ("PELV", "pelvis"),
    ("RSHO", "right_shoulder"),
    ("LSHO", "left_shoulder"),
    ("RUA",  "right_upper_arm"),
    ("LUA",  "left_upper_arm"),
    ("RLA",  "right_forearm"),
    ("LLA",  "left_forearm"),
    ("RHD",  "right_hand"),
    ("LHD",  "left_hand"),
    ("RTH",  "right_thigh"),
    ("LTH",  "left_thigh"),
    ("RSH",  "right_shin"),
    ("LSH",  "left_shin"),
    ("RFT",  "right_foot"),
    ("LFT",  "left_foot"),
]
IMU_NAME_PAIRS_DICT = dict(IMU_NAME_PAIRS)        # WIMUSim → MoVi
IMU_NAMES = list(IMU_NAME_PAIRS_DICT.keys())      # WIMUSim names

# -----------------------------------------------------------------------
# SMPL joint that each IMU is attached to (for default P generation)
# -----------------------------------------------------------------------
IMU_PARENT_JOINT = {
    "HED":  "HEAD",
    "STER": "SPINE3",
    "PELV": "BASE",
    "RSHO": "R_COLLAR",
    "LSHO": "L_COLLAR",
    "RUA":  "R_SHOULDER",
    "LUA":  "L_SHOULDER",
    "RLA":  "R_ELBOW",
    "LLA":  "L_ELBOW",
    "RHD":  "R_WRIST",
    "LHD":  "L_WRIST",
    "RTH":  "R_HIP",
    "LTH":  "L_HIP",
    "RSH":  "R_KNEE",
    "LSH":  "L_KNEE",
    "RFT":  "R_ANKLE",
    "LFT":  "L_ANKLE",
}

# -----------------------------------------------------------------------
# Subjects
# -----------------------------------------------------------------------
# MoVi subjects are labelled F_SubjectXX (female) or M_SubjectXX (male).
FEMALE_SUBJECTS = [f"F_Subject{i:02d}" for i in range(1, 47)]
MALE_SUBJECTS   = [f"M_Subject{i:02d}" for i in range(1, 45)]
ALL_SUBJECTS    = FEMALE_SUBJECTS + MALE_SUBJECTS

# Suggested train/test split (no official split; adjust as needed)
TRAIN_SUBJECTS = ALL_SUBJECTS[:60]
TEST_SUBJECTS  = ALL_SUBJECTS[60:]

# -----------------------------------------------------------------------
# Activities
# -----------------------------------------------------------------------
ACTIVITY_LIST = [
    "squat",
    "mrope",       # jump rope
    "walk",
    "run",
    "sit",
    "catch",
    "throw",
    "washface",
    "phone",
    "armcircle",
    "cartwheel",
    "punch",
    "kick",
    "spin",
    "clap",
    "bend",
    "wave",
    "reach",
    "yoga",
    "dance",
]
