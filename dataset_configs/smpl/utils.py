"""
Utility functions for converting SMPL parameters to WIMUSim B and D parameters.

Requirements:
    pip install smplx torch pytorch3d

Usage example:
    from dataset_configs.smpl.utils import compute_B_from_beta, smpl_pose_to_D_orientation
    from wimusim.wimusim import WIMUSim

    # --- B parameter from SMPL beta ---
    rp_dict = compute_B_from_beta(beta, smpl_model_path="path/to/smpl/models")
    B = WIMUSim.Body(rp=rp_dict)

    # --- D parameter from SMPL pose (rotation matrices, shape (T, 23, 3, 3)) ---
    orientation = smpl_pose_to_D_orientation(global_orient_rotmat, body_pose_rotmat)
    D = WIMUSim.Dynamics(orientation=orientation, sample_rate=30)
"""

import numpy as np
import torch
import pytorch3d.transforms.rotation_conversions as rc

from dataset_configs.smpl.consts import (
    SMPL_JOINT_ID_DICT,
    SMPL_JOINT_PARENT_CHILD_PAIRS,
    SMPL_BODY_POSE_JOINT_NAMES,
)


# ---------------------------------------------------------------------------
# B parameter: bone vectors from SMPL shape (beta)
# ---------------------------------------------------------------------------

def compute_B_from_beta(
    beta,
    smpl_model_path: str,
    gender: str = "neutral",
    device: str = "cpu",
) -> dict:
    """
    Compute WIMUSim Body.rp dict from SMPL shape parameters (beta).

    The returned bone vectors are in the SMPL rest-pose coordinate frame
    (T-pose, zero rotation).

    Args:
        beta: SMPL shape parameters, numpy array or Tensor of shape (10,) or (1, 10).
        smpl_model_path: Path to the directory containing SMPL model files
                         (e.g. "SMPL_NEUTRAL.pkl").
        gender: "neutral", "male", or "female".
        device: Torch device string.

    Returns:
        rp_dict: dict mapping (parent_name, child_name) -> np.ndarray of shape (3,).
                 Bone vector points from parent joint to child joint in the T-pose.
    """
    try:
        import smplx
    except ImportError:
        raise ImportError(
            "smplx is required for compute_B_from_beta. "
            "Install it with: pip install smplx"
        )

    model = smplx.create(smpl_model_path, model_type="smpl", gender=gender).to(device)

    if isinstance(beta, np.ndarray):
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
    if beta.dim() == 1:
        beta = beta.unsqueeze(0)  # (1, 10)

    with torch.no_grad():
        output = model(betas=beta, return_verts=False)

    # joints: (1, 45, 3) for smplx, take first 24 SMPL joints
    joints = output.joints[0, :24].cpu().numpy()  # (24, 3)

    rp_dict = {}
    for parent_name, child_name in SMPL_JOINT_PARENT_CHILD_PAIRS:
        if parent_name == "BASE":
            # BASE is a virtual root at the same position as SMPL joint 0.
            # Bone vector is zero because BASE and joint 0 are co-located.
            rp_dict[("BASE", child_name)] = (
                joints[SMPL_JOINT_ID_DICT[child_name]] - joints[SMPL_JOINT_ID_DICT["BASE"]]
            )
        else:
            parent_id = SMPL_JOINT_ID_DICT[parent_name]
            child_id  = SMPL_JOINT_ID_DICT[child_name]
            rp_dict[(parent_name, child_name)] = joints[child_id] - joints[parent_id]

    return rp_dict


# ---------------------------------------------------------------------------
# D parameter: joint orientations from SMPL pose
# ---------------------------------------------------------------------------

def smpl_pose_to_D_orientation(
    global_orient,
    body_pose,
) -> dict:
    """
    Convert SMPL pose parameters to WIMUSim Dynamics.orientation dict.

    WIMUSim expects each joint's rotation to be **relative to its parent**,
    expressed as a unit quaternion in WXYZ format (PyTorch3D convention).
    SMPL body_pose already stores relative rotations, so the conversion is
    a straightforward format change.

    Args:
        global_orient: Root orientation, shape (T, 3, 3) rotation matrices
                       or (T, 3) axis-angle vectors.
        body_pose: Per-joint relative rotations, shape (T, 23, 3, 3)
                   rotation matrices or (T, 23, 3) axis-angle vectors.

    Returns:
        orientation: dict mapping joint_name -> Tensor of shape (T, 4) in WXYZ.
                     Includes "BASE" (from global_orient) and all 23 body joints.
    """
    global_orient = _to_rotmat(global_orient)   # (T, 1, 3, 3) or (T, 3, 3)
    body_pose      = _to_rotmat(body_pose)       # (T, 23, 3, 3)

    if global_orient.dim() == 3:
        global_orient = global_orient.unsqueeze(1)  # (T, 1, 3, 3)

    # Concatenate: full_pose shape (T, 24, 3, 3)
    full_pose = torch.cat([global_orient, body_pose], dim=1)

    orientation = {}

    # BASE = global_orient (joint 0)
    orientation["BASE"] = _rotmat_to_quat_wxyz(full_pose[:, 0])  # (T, 4)

    # Joints 1–23 → body_pose[0–22]
    for i, joint_name in enumerate(SMPL_BODY_POSE_JOINT_NAMES):
        orientation[joint_name] = _rotmat_to_quat_wxyz(full_pose[:, i + 1])  # (T, 4)

    return orientation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_rotmat(x) -> torch.Tensor:
    """Accept rotation matrix (T, [N,] 3, 3) or axis-angle (T, [N,] 3)."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")
    x = x.float()

    # Axis-angle: last dim is 3 and second-to-last is also 3 only if rotation matrix
    if x.shape[-1] == 3 and x.shape[-2] != 3:
        # axis-angle: (..., 3) → rotation matrix (..., 3, 3)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, 3)
        rotmat = rc.axis_angle_to_matrix(x_flat)  # (N, 3, 3)
        return rotmat.reshape(*orig_shape, 3, 3)

    return x  # already rotation matrix


def _rotmat_to_quat_wxyz(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert (T, 3, 3) rotation matrices to (T, 4) WXYZ quaternions."""
    quat = rc.matrix_to_quaternion(rotmat)          # pytorch3d: WXYZ
    return rc.standardize_quaternion(quat)          # (T, 4)
