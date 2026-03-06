"""
TotalCapture dataset constants for WIMUSim.

TotalCapture: A Large Scale Dataset with Multimodal Capture of Human Motion
- 5 subjects × 5 activity types × multiple trials
- 13 Xsens MTx/MTi IMUs (60 Hz)
- 8 HD video cameras
- SMPL fits available from follow-up work (e.g. ROMP, SMPLify-X)

Reference: Trumble et al., "Total Capture: 3D Human Pose Estimation Fusing
           Video and Inertial Sensors", BMVC 2017.
Dataset: https://cvssp.org/data/totalcapture/
"""

# IMU and SMPL sample rate (Hz)
IMU_SAMPLE_RATE  = 60
SMPL_SAMPLE_RATE = 60

# -----------------------------------------------------------------------
# IMU placements (WIMUSim name → TotalCapture sensor label)
# -----------------------------------------------------------------------
IMU_NAME_PAIRS = [
    ("HED",  "Head"),
    ("STER", "Sternum"),
    ("PELV", "Pelvis"),
    ("RUA",  "RightUpperArm"),
    ("LUA",  "LeftUpperArm"),
    ("RLA",  "RightLowerArm"),
    ("LLA",  "LeftLowerArm"),
    ("RHD",  "RightHand"),
    ("LHD",  "LeftHand"),
    ("RTH",  "RightUpperLeg"),
    ("LTH",  "LeftUpperLeg"),
    ("RSH",  "RightLowerLeg"),
    ("LSH",  "LeftLowerLeg"),
]
IMU_NAME_PAIRS_DICT = dict(IMU_NAME_PAIRS)
IMU_NAMES = list(IMU_NAME_PAIRS_DICT.keys())

# -----------------------------------------------------------------------
# SMPL joint that each IMU is attached to
# -----------------------------------------------------------------------
IMU_PARENT_JOINT = {
    "HED":  "HEAD",
    "STER": "SPINE3",
    "PELV": "BASE",
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
}

# -----------------------------------------------------------------------
# Subjects and activities
# -----------------------------------------------------------------------
SUBJECT_LIST = ["s1", "s2", "s3", "s4", "s5"]

# Suggested split (s1-s4 train, s5 test — adjust as needed)
TRAIN_SUBJECTS = ["s1", "s2", "s3", "s4"]
TEST_SUBJECTS  = ["s5"]

ACTIVITY_LIST = [
    "rom1", "rom2", "rom3",
    "walking1", "walking2", "walking3", "walking4",
    "acting1", "acting2", "acting3",
    "freestyle1", "freestyle2", "freestyle3",
    "jumping1",
]
