"""
Evaluate PhyNeSim on the MoVi test split (subjects 61-90).

Physics-only baseline:
    python scripts/evaluate.py \
        --dataset    movi \
        --amass_root /data/MoVi/amass \
        --xsens_root /data/MoVi/xsens \
        --smpl_model path/to/smpl/models \
        --output_dir results/

With neural residual correction:
    python scripts/evaluate.py \
        --dataset    movi \
        --amass_root /data/MoVi/amass \
        --xsens_root /data/MoVi/xsens \
        --smpl_model path/to/smpl/models \
        --checkpoint output/checkpoints/best.pt \
        --output_dir results/

Outputs:
    results/
        movi_metrics.csv          per-sequence per-IMU metrics
        movi_summary.csv          mean across all sequences
        movi_per_imu_rmse.png     per-IMU RMSE bar chart (acc + gyro)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset_configs.smpl.utils import compute_B_from_beta, smpl_pose_to_D_orientation
from pipeline.evaluate import evaluate, save_metrics
from wimusim import WIMUSim, utils


# ---------------------------------------------------------------------------
# Single-sequence runner
# ---------------------------------------------------------------------------

def _run_sequence(betas, global_orient, body_pose, trans, imu_names,
                  placement_fn, sample_rate, smpl_model, checkpoint, device):
    """Simulate one sequence. Returns {imu_name: (acc, gyro)}."""
    dev = torch.device(device)

    def _t(v):
        if isinstance(v, np.ndarray):
            return torch.tensor(v, dtype=torch.float32, device=dev)
        return v.to(dev) if isinstance(v, torch.Tensor) else v

    B_rp        = compute_B_from_beta(betas, smpl_model_path=smpl_model)
    orientation = smpl_pose_to_D_orientation(global_orient, body_pose)
    P_params    = placement_fn(B_rp)
    H_cfg       = utils.generate_default_H_configs(imu_names)

    B_obj = WIMUSim.Body(
        rp={k: _t(v) for k, v in B_rp.items()},
        device=dev,
    )
    D_obj = WIMUSim.Dynamics(
        orientation={k: _t(v) for k, v in orientation.items()},
        translation={"XYZ": _t(trans)},
        sample_rate=sample_rate,
        device=dev,
    )
    P_obj = WIMUSim.Placement(
        rp={k: _t(v) for k, v in P_params["rp"].items()},
        ro={k: _t(v) for k, v in P_params["ro"].items()},
        device=dev,
    )
    H_obj = WIMUSim.Hardware(
        ba=H_cfg["ba"], bg=H_cfg["bg"], sa=H_cfg["sa"], sg=H_cfg["sg"],
        sa_range_dict=H_cfg["sa_range_dict"], sg_range_dict=H_cfg["sg_range_dict"],
        device=dev,
    )

    if checkpoint is None:
        env = WIMUSim(B=B_obj, D=D_obj, P=P_obj, H=H_obj)
        if isinstance(env.E.g, torch.Tensor):
            env.E.g = env.E.g.to(dev)
        return env.simulate(mode="generate")

    # Neural-corrected path
    from nn.infer import corrected_simulate
    return corrected_simulate(
        checkpoint=checkpoint, B=B_obj, D=D_obj, P=P_obj, H=H_obj,
        device=dev,
    )


# ---------------------------------------------------------------------------
# Per-dataset evaluation loops
# ---------------------------------------------------------------------------


def eval_movi(amass_root, xsens_root, v3d_root, smpl_model,
              checkpoint, device, subjects, activity_indices):
    from dataset_configs.movi.consts import (
        TEST_SUBJECTS, SMPL_SAMPLE_RATE, V3D_MOTION_LIST, IMU_NAMES,
    )
    from dataset_configs.movi.utils import (
        load_smpl_params, load_imu_data, load_xsens_imu,
        generate_default_placement_params as movi_placement,
    )

    subj_list   = subjects        if subjects         is not None else TEST_SUBJECTS
    act_indices = activity_indices if activity_indices is not None else list(range(21))
    rows = []

    for subject_num in subj_list:
        for act_idx in act_indices:
            seq_id        = act_idx + 1
            activity_name = V3D_MOTION_LIST[act_idx]
            print(f"  Subject {subject_num}/{activity_name} ...", end=" ", flush=True)
            try:
                betas, go, bp, trans = load_smpl_params(amass_root, subject_num, seq_id)
                if xsens_root is not None:
                    real_imu = load_xsens_imu(
                        xsens_root, subject_num, act_idx,
                        v3d_root=v3d_root, amass_root=amass_root,
                    )
                else:
                    real_imu = load_imu_data(v3d_root, subject_num, act_idx)
            except FileNotFoundError:
                print("skipped (no file)")
                continue
            except Exception as e:
                print(f"skipped ({e})")
                continue

            try:
                virt_imu = _run_sequence(
                    betas, go, bp, trans, IMU_NAMES, movi_placement,
                    SMPL_SAMPLE_RATE, smpl_model, checkpoint, device,
                )
            except Exception as e:
                print(f"skipped ({e})")
                continue

            df = evaluate(virt_imu, real_imu)
            df["subject"]  = subject_num
            df["activity"] = activity_name
            rows.append(df)
            print("done")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_per_imu_rmse(df: pd.DataFrame, output_dir: Path, prefix: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{prefix} — Per-IMU RMSE", fontsize=13)

    for ax, modality, unit in [
        (axes[0], "acc",  "m/s²"),
        (axes[1], "gyro", "rad/s"),
    ]:
        sub = df[(df["modality"] == modality) & (df["imu"] != "MEAN")]
        per_imu = sub.groupby("imu")["rmse"].mean().sort_values()
        per_imu.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{modality.upper()} RMSE ({unit})")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out = output_dir / f"{prefix}_per_imu_rmse.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PhyNeSim evaluation on MoVi test split")
    parser.add_argument("--amass_root",       required=True,
                        help="Root containing F_amass_Subject_N.mat files")
    parser.add_argument("--smpl_model",       required=True,
                        help="Path to SMPL model directory (contains SMPL_NEUTRAL.pkl)")
    parser.add_argument("--xsens_root",       default=None,
                        help="Root containing imu_Subject_N.mat files "
                             "(real Xsens IMU, recommended)")
    parser.add_argument("--v3d_root",         default=None,
                        help="Root containing F_v3d_Subject_N.mat files "
                             "(used when xsens_root is not set)")
    parser.add_argument("--checkpoint",       default=None,
                        help="Path to trained PhyNeSim checkpoint (.pt). "
                             "Omit to run physics-only baseline.")
    parser.add_argument("--output_dir",       default="results",
                        help="Where to save CSV and plots (default: results/)")
    parser.add_argument("--device",           default="cpu",
                        help="Torch device (cpu or cuda)")
    parser.add_argument("--test_subjects",    nargs="+", type=int, default=None,
                        help="Subject numbers to evaluate (default: 61-90)")
    parser.add_argument("--activity_indices", nargs="+", type=int, default=None,
                        help="Activity indices 0-20 to evaluate (default: all 21)")

    args = parser.parse_args()

    if not args.xsens_root and not args.v3d_root:
        parser.error("--xsens_root or --v3d_root is required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode   = "PhyNeSim" if args.checkpoint else "Physics-only"
    prefix = "movi"

    print(f"\n=== Evaluating MoVi  [{mode}] ===\n")

    df = eval_movi(
        args.amass_root, args.xsens_root, args.v3d_root,
        args.smpl_model, args.checkpoint, args.device,
        args.test_subjects, args.activity_indices,
    )

    if df.empty:
        print("\nNo sequences evaluated. Check data paths and subject numbers.")
        sys.exit(1)

    # Summary: mean over all sequences (rows where imu == "MEAN")
    summary = (
        df[df["imu"] == "MEAN"]
        .groupby("modality")[["rmse", "mae", "pearson"]]
        .mean()
    )

    print(f"\n{'=' * 55}")
    print(f"  MoVi  [{mode}]  — Overall Mean")
    print(f"{'=' * 55}")
    print(summary.round(4).to_string())

    # Save outputs
    metrics_path = out_dir / f"{prefix}_metrics.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    save_metrics(df, str(metrics_path))
    summary.to_csv(summary_path)
    print(f"\n  Summary saved to {summary_path}")

    plot_per_imu_rmse(df, out_dir, prefix)


if __name__ == "__main__":
    main()
