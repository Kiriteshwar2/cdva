"""
dataset.py — CSV parsing and feature extraction for crystal structures.
Each crystal CIF string is parsed with pymatgen to extract 8 structural features:
  [num_atoms, volume, a, b, c, alpha, beta, gamma]
"""

import numpy as np
import pandas as pd
from pymatgen.core import Structure


FEATURE_NAMES = ["num_atoms", "volume", "a", "b", "c", "alpha", "beta", "gamma"]


def extract_features(df: pd.DataFrame):
    """
    Parse CIF strings in `df` and return:
      X — numpy array of shape (N, 8)
      y — numpy array of energy_per_atom values
      errors — number of rows skipped due to parse failures
    """
    features, targets = [], []
    errors = 0

    # Find the CIF column (could be 'cif', 'CIF', etc.)
    cif_col = next((c for c in df.columns if "cif" in c.lower()), None)
    if cif_col is None:
        raise ValueError("No CIF column found in the uploaded CSV.")

    for i in range(len(df)):
        try:
            struct = Structure.from_str(df[cif_col].iloc[i], fmt="cif")
            a, b, c = struct.lattice.abc
            al, be, ga = struct.lattice.angles
            features.append([len(struct), struct.volume, a, b, c, al, be, ga])
            targets.append(df["energy_per_atom"].iloc[i])
        except Exception:
            errors += 1
            continue

    X = np.array(features, dtype=np.float32)
    y = np.array(targets,  dtype=np.float32)
    return X, y, errors


def get_preview(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Return the first `n` rows as a list of dicts (JSON-serializable)."""
    return df.head(n).fillna("").to_dict(orient="records")
