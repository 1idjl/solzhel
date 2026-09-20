"""
src/glass_system.py — Glass system representation and initialization
=====================================================================
Loads the initial XYZ structure, computes partial charges, and prepares
the system for Reactive Monte Carlo simulation.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional
from .constants import (
    NA_AVOGADRO, MASSES, ELEM_TO_TYPE, TYPE_O
)
from .potential import Potential

logger = logging.getLogger(__name__)


class GlassSystem:
    """
    Represents the atomic system: coordinates, types, charges, box.
    """
    
    def __init__(self, xyz_path: Path, topology_path: Optional[Path] = None,
                 density: Optional[float] = None, box_override: Optional[float] = None):
        self.potential = Potential()
        self.masses = MASSES
        
        self._load_xyz(xyz_path)
        self._setup_box(density, box_override)
        self._compute_charges()
        self._prepare_matrices()
        
        # Load initial bonds from topology JSON (if provided)
        self.initial_bonds = []
        if topology_path is not None and topology_path.exists():
            with open(topology_path) as f:
                topo = json.load(f)
            self.initial_bonds = [tuple(b) for b in topo.get('initial_bonds', [])]
            logger.info(f"Loaded {len(self.initial_bonds)} initial bonds from topology")
            
    def _load_xyz(self, path: Path):
        """Read atomic coordinates and symbols from XYZ file."""
        if not path.exists():
            raise FileNotFoundError(f"XYZ file not found: {path}")
            
        with open(path) as f:
            lines = f.readlines()
            
        n_header = int(lines[0].strip())
        self.symbols = []
        coords_list = []
        
        for line in lines[2:2 + n_header]:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            elem = parts[0]
            self.symbols.append(elem)
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
        self.N_ATOMS = len(self.symbols)
        logger.info(f"Loaded {self.N_ATOMS} atoms from {path.name}")
        
        self.coords = np.array(coords_list, dtype=np.float64)
        self.type_indices = np.array([ELEM_TO_TYPE[s] for s in self.symbols], dtype=np.int32)
        self.total_mass = sum(self.masses[s] for s in self.symbols)
        
    def _setup_box(self, density: Optional[float], box_override: Optional[float]):
        """Determine simulation box size based on density or override."""
        if box_override is not None:
            self.box = float(box_override)
        elif density is not None:
            self.box = ((self.total_mass / NA_AVOGADRO) / density * 1.0e24) ** (1.0 / 3.0)
        else:
            # Default box for low-density sol-gel (rho ~ 1.0 g/cm³)
            self.box = 46.77
            logger.warning(f"No box/density provided, using default box {self.box:.2f} A")
            
        self.effective_density = self.total_mass * 1.0e24 / (NA_AVOGADRO * self.box ** 3)
        self.coords = self.coords % self.box
        logger.info(f"Box: {self.box:.4f} A, Density: {self.effective_density:.4f} g/cm3")
        
    def _compute_charges(self):
        """Assign partial charges and neutralize the system on oxygens."""
        # Standard partial charges for 45S5 bioglass (Tilocca / Pedone models)
        base = {'Si': 2.4, 'Ca': 1.2, 'Na': 0.6, 'P': 3.0, 'O': -1.2, 'Sr': 1.2}
        self.charges = np.array([base.get(s, 0.0) for s in self.symbols], dtype=np.float64)
        
        total_charge = float(np.sum(self.charges))
        o_mask = self.type_indices == TYPE_O
        n_o = int(np.sum(o_mask))
        
        if abs(total_charge) > 1.0e-8 and n_o > 0:
            correction = total_charge / n_o
            self.charges[o_mask] -= correction
            logger.info(f"Charge neutralized: corrected by {-correction:+.6f} on {n_o} oxygens")
            
    def _prepare_matrices(self):
        """Build Numba-compatible matrices from the Potential object."""
        matrices = self.potential.build_matrices()
        self.A_mat = matrices['A_mat']
        self.F_mat = matrices['F_mat']
        self.C_mat = matrices['C_mat']
        self.R_HARD_MAT = matrices['R_HARD_MAT']
        self.type_Z = matrices['type_Z']