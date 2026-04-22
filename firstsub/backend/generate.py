"""
generate.py — Crystal generation and CIF export utilities.
"""

import io
import numpy as np
from pymatgen.core import Structure, Lattice


def params_to_cif(params_row: np.ndarray, element: str = "C") -> str:
    """
    Convert a row of 8 generated parameters into a CIF string.
    params_row = [num_atoms, volume, a, b, c, alpha, beta, gamma]
    """
    num_atoms, _, a, b, c, alpha, beta, gamma = params_row

    # Clamp lattice dimensions and angles to physically valid ranges
    a, b, c = max(float(a), 1.0), max(float(b), 1.0), max(float(c), 1.0)
    alpha = float(np.clip(alpha, 20.0, 160.0))
    beta  = float(np.clip(beta,  20.0, 160.0))
    gamma = float(np.clip(gamma, 20.0, 160.0))
    n     = int(max(1, round(float(num_atoms))))

    lattice  = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    coords   = np.random.rand(n, 3)          # random fractional coords
    species  = [element] * n

    structure = Structure(lattice, species, coords)

    # Write CIF to an in-memory string
    buf = io.StringIO()
    structure.to(fmt="cif", filename=None)   # pymatgen >= 2023 supports fmt kwarg

    # Fallback: use to_file via temp path or direct string method
    cif_str = structure.to(fmt="cif")
    return cif_str


def compute_antigravity(density: float, volume: float) -> float:
    """
    antigravity_score = -density + lattice_instability + random_energy_term
    where lattice_instability ~ volume normalised, random_energy_term ~ N(0, 0.3)
    This is illustrative; the user-slider overrides this at generation time.
    """
    lattice_instability = volume / 1000.0
    random_energy_term  = float(np.random.normal(0, 0.3))
    return -density + lattice_instability + random_energy_term
