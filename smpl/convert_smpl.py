"""
Convert SMPL pkl model files from chumpy format to pure numpy format.

The official SMPL v1.0.0 / v1.1.0 model files distributed at
https://smpl.is.tue.mpg.de/ store internal arrays as chumpy.Ch objects.
chumpy is a Python 2.7 library that cannot be installed on Python 3.10+.

This script converts the pkl files in-place to plain numpy arrays,
removing the chumpy dependency entirely and making them compatible with
modern Python (3.8+) and numpy (1.x / 2.x).

Usage:
    # Option A: Python 3.8 + numpy 1.x (if files still contain chumpy objects)
    conda create -n smpl_convert python=3.8 numpy=1.23 scipy -y
    conda activate smpl_convert
    pip install chumpy
    python smpl/convert_smpl.py

    # Option B: any Python 3 env (if files are already partially converted)
    python smpl/convert_smpl.py

After running this script, the three pkl files can be loaded by smplx
without any chumpy installation.
"""

import os
import sys
import pickle
import types
import argparse
import numpy as np
from scipy.sparse import issparse


def _register_fake_chumpy():
    """Register a minimal fake chumpy module so pickle can deserialise old files."""
    fake = types.ModuleType("chumpy")

    class Ch:
        def __new__(cls, *a, **kw):
            return object.__new__(cls)
        def __init__(self, *a, **kw):
            pass

    fake.Ch = Ch
    fake.array = np.array
    for name in ["chumpy", "chumpy.ch", "chumpy.utils",
                 "chumpy.reordering", "chumpy.linalg"]:
        sys.modules.setdefault(name, fake)

    # Handle references saved as __main__.Ch by previous failed conversions
    import __main__
    if not hasattr(__main__, "Ch"):
        setattr(__main__, "Ch", Ch)

    return Ch


def _to_numpy(v, Ch):
    """Convert a value to a plain numpy array."""
    if issparse(v):
        return v.toarray()
    if isinstance(v, Ch):
        raw = getattr(v, "r", None) or getattr(v, "x", None)
        return np.array(raw) if raw is not None else np.array([])
    try:
        return np.array(v)
    except Exception:
        return v


def convert_file(path: str) -> None:
    """Convert a single SMPL pkl file in-place to pure numpy format."""
    Ch = _register_fake_chumpy()

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    clean = {k: _to_numpy(v, Ch) for k, v in data.items()}

    with open(path, "wb") as f:
        pickle.dump(clean, f, protocol=2)

    print(f"  {os.path.basename(path)}: OK  "
          f"({', '.join(f'{k}:{np.array(v).shape}' for k, v in clean.items() if hasattr(np.array(v), 'shape') and np.array(v).ndim > 0)})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPL pkl files from chumpy to pure numpy format"
    )
    parser.add_argument(
        "--models_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "smpl"),
        help="Directory containing SMPL_NEUTRAL.pkl / SMPL_MALE.pkl / SMPL_FEMALE.pkl "
             "(default: smpl/models/smpl/ relative to this script)",
    )
    parser.add_argument(
        "--genders", nargs="+", default=["NEUTRAL", "MALE", "FEMALE"],
        help="Which gender files to convert (default: all three)",
    )
    args = parser.parse_args()

    print(f"Converting SMPL model files in: {args.models_dir}")
    missing = []
    for gender in args.genders:
        path = os.path.join(args.models_dir, f"SMPL_{gender}.pkl")
        if not os.path.exists(path):
            print(f"  SMPL_{gender}.pkl not found — skipping")
            missing.append(gender)
            continue
        convert_file(path)

    if len(missing) == len(args.genders):
        print("\nNo files were converted. Check --models_dir.")
        sys.exit(1)

    print("\nDone. You can now use these files with smplx on Python 3.10+.")


if __name__ == "__main__":
    main()
