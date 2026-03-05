# SMPL 24-joint skeleton definitions for WIMUSim
#
# SMPL joint index reference:
#  0: Root/Pelvis (BASE)   1: L_Hip       2: R_Hip      3: Spine1
#  4: L_Knee               5: R_Knee      6: Spine2     7: L_Ankle
#  8: R_Ankle              9: Spine3     10: L_Foot    11: R_Foot
# 12: Neck                13: L_Collar  14: R_Collar  15: Head
# 16: L_Shoulder          17: R_Shoulder 18: L_Elbow  19: R_Elbow
# 20: L_Wrist             21: R_Wrist   22: L_Hand    23: R_Hand

SMPL_JOINT_ID_DICT = {
    "BASE":       0,   # Root / Pelvis
    "L_HIP":      1,
    "R_HIP":      2,
    "SPINE1":     3,
    "L_KNEE":     4,
    "R_KNEE":     5,
    "SPINE2":     6,
    "L_ANKLE":    7,
    "R_ANKLE":    8,
    "SPINE3":     9,
    "L_FOOT":    10,
    "R_FOOT":    11,
    "NECK":      12,
    "L_COLLAR":  13,
    "R_COLLAR":  14,
    "HEAD":      15,
    "L_SHOULDER": 16,
    "R_SHOULDER": 17,
    "L_ELBOW":   18,
    "R_ELBOW":   19,
    "L_WRIST":   20,
    "R_WRIST":   21,
    "L_HAND":    22,
    "R_HAND":    23,
}

# Topological order: each parent must appear before its children.
# BASE (joint 0) is the virtual root in WIMUSim, corresponding to SMPL's global_orient.
SMPL_JOINT_PARENT_CHILD_PAIRS = [
    ("BASE", "L_HIP"),
    ("BASE", "R_HIP"),
    ("BASE", "SPINE1"),
    ("L_HIP",    "L_KNEE"),
    ("R_HIP",    "R_KNEE"),
    ("SPINE1",   "SPINE2"),
    ("L_KNEE",   "L_ANKLE"),
    ("R_KNEE",   "R_ANKLE"),
    ("SPINE2",   "SPINE3"),
    ("L_ANKLE",  "L_FOOT"),
    ("R_ANKLE",  "R_FOOT"),
    ("SPINE3",   "NECK"),
    ("SPINE3",   "L_COLLAR"),
    ("SPINE3",   "R_COLLAR"),
    ("NECK",     "HEAD"),
    ("L_COLLAR", "L_SHOULDER"),
    ("R_COLLAR", "R_SHOULDER"),
    ("L_SHOULDER", "L_ELBOW"),
    ("R_SHOULDER", "R_ELBOW"),
    ("L_ELBOW",  "L_WRIST"),
    ("R_ELBOW",  "R_WRIST"),
    ("L_WRIST",  "L_HAND"),
    ("R_WRIST",  "R_HAND"),
]

# SMPL body_pose contains 23 relative rotations for joints 1–23 (in joint-index order).
# This list maps body_pose[i] to the corresponding WIMUSim joint name.
SMPL_BODY_POSE_JOINT_NAMES = [
    "L_HIP",     # body_pose[0]  → joint 1
    "R_HIP",     # body_pose[1]  → joint 2
    "SPINE1",    # body_pose[2]  → joint 3
    "L_KNEE",    # body_pose[3]  → joint 4
    "R_KNEE",    # body_pose[4]  → joint 5
    "SPINE2",    # body_pose[5]  → joint 6
    "L_ANKLE",   # body_pose[6]  → joint 7
    "R_ANKLE",   # body_pose[7]  → joint 8
    "SPINE3",    # body_pose[8]  → joint 9
    "L_FOOT",    # body_pose[9]  → joint 10
    "R_FOOT",    # body_pose[10] → joint 11
    "NECK",      # body_pose[11] → joint 12
    "L_COLLAR",  # body_pose[12] → joint 13
    "R_COLLAR",  # body_pose[13] → joint 14
    "HEAD",      # body_pose[14] → joint 15
    "L_SHOULDER",# body_pose[15] → joint 16
    "R_SHOULDER",# body_pose[16] → joint 17
    "L_ELBOW",   # body_pose[17] → joint 18
    "R_ELBOW",   # body_pose[18] → joint 19
    "L_WRIST",   # body_pose[19] → joint 20
    "R_WRIST",   # body_pose[20] → joint 21
    "L_HAND",    # body_pose[21] → joint 22
    "R_HAND",    # body_pose[22] → joint 23
]

# Maps SMPL joint names to WIMUSim PyBullet humanoid link names.
# (defined in wimusim/consts.py HUMANOID_PARAMS_DEFAULT)
JOINT_WIMUSIM_LINK_PAIRS = [
    ("BASE",       "pelvis"),
    ("SPINE1",     "torso_1"),
    ("SPINE3",     "torso_2"),
    ("NECK",       "head"),
    ("R_COLLAR",   "right_clavicle"),
    ("R_SHOULDER", "right_upperarm"),
    ("R_ELBOW",    "right_lowerarm"),
    ("R_WRIST",    "right_hand"),
    ("L_COLLAR",   "left_clavicle"),
    ("L_SHOULDER", "left_upperarm"),
    ("L_ELBOW",    "left_lowerarm"),
    ("L_WRIST",    "left_hand"),
    ("R_HIP",      "right_upperleg"),
    ("R_KNEE",     "right_lowerleg"),
    ("R_ANKLE",    "right_foot"),
    ("L_HIP",      "left_upperleg"),
    ("L_KNEE",     "left_lowerleg"),
    ("L_ANKLE",    "left_foot"),
]
JOINT_WIMUSIM_LINK_DICT = {
    joint: link for joint, link in JOINT_WIMUSIM_LINK_PAIRS
}

SMPL_JOINT_PAIR_DICT = {
    (parent, child): (SMPL_JOINT_ID_DICT[parent], SMPL_JOINT_ID_DICT[child])
    for parent, child in SMPL_JOINT_PARENT_CHILD_PAIRS
}

SMPL_JOINT_CHILD_PARENT_DICT = {
    child: parent for parent, child in SMPL_JOINT_PARENT_CHILD_PAIRS
}
