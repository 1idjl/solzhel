"""
src/potential.py — Interatomic potential parameters for 45S5 bioglass
======================================================================
Contains Buckingham + ZBL parameters for non-bonded interactions,
and harmonic bond parameters for reactive MC.

Key fix (v5.1): Increased r_hard values to prevent over-compression
during NPT densification. Previous values allowed the system to
compress to unphysical densities (~5.8 g/cm³).

References:
- Buckingham: Tilocca et al., Phys. Rev. B 73, 104209 (2006)
              Pedone et al., J. Phys. Chem. C 113, 15723 (2009)
- ZBL: Ziegler, Biersack, Littmark (1985)
- Bond harmonic: Derived from BKS potential curvature at equilibrium
"""

import numpy as np
from .constants import (
    N_TYPES, TYPE_SI, TYPE_CA, TYPE_NA, TYPE_P, TYPE_O, TYPE_SR,
    SWITCH_DR, KE_COULOMB,
    BOND_HARMONIC_K_SI_O, BOND_HARMONIC_K_P_O,
    BOND_HARMONIC_R0_SI_O, BOND_HARMONIC_R0_P_O,
)


class Potential:
    """
    Interatomic potential parameters for 45S5 bioactive glass.
    
    Non-bonded interactions:
    - Buckingham: E = A·exp(-r/F) - C/r⁶
    - Wolf Coulomb: damped, truncated Coulomb summation
    - ZBL: Universal repulsive potential at very short distances
    
    Bonded interactions (for reactive MC):
    - Harmonic: E = ½·k·(r - r₀)²
    """
    
    def __init__(self):
        # ====================================================================
        # ZBL UNIVERSAL REPULSIVE POTENTIAL
        # ====================================================================
        self.zbl_a0 = 0.46850
        self.zbl_c = np.array([0.1818, 0.5099, 0.2802, 0.02817])
        self.zbl_d = np.array([3.2, 0.9423, 0.4029, 0.2016])
        
        self.zbl_z = {
            'Si': 14.0,
            'Ca': 20.0,
            'Na': 11.0,
            'P':  15.0,
            'O':  8.0,
            'Sr': 38.0,
        }
        
        # ====================================================================
        # BUCKINGHAM POTENTIAL PARAMETERS
        # ====================================================================
        self.buck_params = {
            ('Si', 'O'): {'A': 13702.905,  'F': 0.193817, 'C': 54.681},
            ('Ca', 'O'): {'A': 7747.1834,  'F': 0.252623, 'C': 93.109},
            ('P',  'O'): {'A': 26655.472,  'F': 0.181968, 'C': 86.856},
            ('Na', 'O'): {'A': 4383.7555,  'F': 0.243838, 'C': 30.70},
            ('Sr', 'O'): {'A': 14566.637,  'F': 0.245015, 'C': 81.773},
            ('O',  'O'): {'A': 2029.2204,  'F': 0.343645, 'C': 192.58},
            ('Si', 'Si'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Si', 'P'):  {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('P',  'P'):  {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Ca', 'Ca'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Na', 'Na'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Na', 'Ca'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Sr', 'Sr'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Sr', 'Ca'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Sr', 'Na'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Sr', 'Si'): {'A': 5000.0,   'F': 0.25,     'C': 0.0},
            ('Sr', 'P'):  {'A': 5000.0,   'F': 0.25,     'C': 0.0},
        }
        
        # ====================================================================
        # [FIX v5.1] HARD-CORE RADII — INCREASED TO PREVENT OVER-COMPRESSION
        # ====================================================================
        # PREVIOUS VALUES (caused collapse to 5.8 g/cm³):
        #   r_hard_default = 0.9, Si-O = 0.9 (default), O-O = 1.4
        #
        # NEW VALUES (prevent artificial potential wells):
        #   Si-O: 1.60 (equilibrium is 1.61)
        #   P-O:  1.50 (equilibrium is 1.49)
        #   O-O:  2.00 (van der Waals contact)
        #   Ca-O: 1.70 (equilibrium is 2.40)
        #   Na-O: 1.70 (equilibrium is 2.37)
        # ====================================================================
        self.r_hard_default = 1.3  # [FIX] Increased from 0.9 to 1.3
        
        self.r_hard_by_pair = {
            ('O',  'O'):  2.00,   # [FIX] Increased from 1.8 to 2.0
            ('Si', 'O'):  1.60,   # [FIX] Increased from 1.5 to 1.6
            ('O',  'Si'): 1.60,
            ('P',  'O'):  1.50,   # [FIX] Increased from 1.4 to 1.5
            ('O',  'P'):  1.50,
            ('Ca', 'O'):  1.70,   # [FIX] Increased from 1.6 to 1.7
            ('O',  'Ca'): 1.70,
            ('Na', 'O'):  1.70,   # [FIX] Increased from 1.6 to 1.7
            ('O',  'Na'): 1.70,
            ('Sr', 'O'):  1.80,
            ('O',  'Sr'): 1.80,
            ('Si', 'Si'): 2.20,
            ('Si', 'P'):  2.20,
            ('P',  'Si'): 2.20,
            ('P',  'P'):  2.20,
            ('Ca', 'Ca'): 2.00,
            ('Na', 'Na'): 2.00,
            ('Na', 'Ca'): 2.00,
            ('Ca', 'Na'): 2.00,
            ('Sr', 'Sr'): 2.20,
            ('Sr', 'Ca'): 2.10,
            ('Ca', 'Sr'): 2.10,
        }
        
        # ====================================================================
        # HARMONIC BOND PARAMETERS (for reactive MC)
        # ====================================================================
        self.bond_params = {
            (TYPE_SI, TYPE_O): {
                'k':  BOND_HARMONIC_K_SI_O,
                'r0': BOND_HARMONIC_R0_SI_O,
            },
            (TYPE_P, TYPE_O): {
                'k':  BOND_HARMONIC_K_P_O,
                'r0': BOND_HARMONIC_R0_P_O,
            },
        }
    
    def get_r_hard(self, ti: int, tj: int) -> float:
        """Get hard-core radius for a pair of atom types."""
        from .constants import TYPE_TO_ELEM
        e1 = TYPE_TO_ELEM.get(ti, 'O')
        e2 = TYPE_TO_ELEM.get(tj, 'O')
        return self.r_hard_by_pair.get((e1, e2), self.r_hard_default)
    
    def get_buck_params(self, ti: int, tj: int) -> tuple:
        """Get Buckingham (A, F, C) for a pair of atom types."""
        from .constants import TYPE_TO_ELEM
        e1 = TYPE_TO_ELEM.get(ti, 'O')
        e2 = TYPE_TO_ELEM.get(tj, 'O')
        p = self.buck_params.get((e1, e2), {'A': 5000.0, 'F': 0.25, 'C': 0.0})
        return p['A'], p['F'], p['C']
    
    def get_bond_params(self, ti: int, tj: int) -> tuple:
        """Get harmonic bond (k, r0) for a bonded pair."""
        key = (min(ti, tj), max(ti, tj))
        p = self.bond_params.get(key, {'k': 25.0, 'r0': 1.6})
        return p['k'], p['r0']
    
    def build_matrices(self) -> dict:
        """Build numpy matrices for Numba kernels."""
        nt = N_TYPES
        A_mat = np.zeros((nt, nt), dtype=np.float64)
        F_mat = np.zeros((nt, nt), dtype=np.float64)
        C_mat = np.zeros((nt, nt), dtype=np.float64)
        R_HARD_MAT = np.full((nt, nt), self.r_hard_default, dtype=np.float64)
        
        from .constants import TYPE_TO_ELEM
        
        for i in range(nt):
            for j in range(nt):
                e1 = TYPE_TO_ELEM.get(i, 'O')
                e2 = TYPE_TO_ELEM.get(j, 'O')
                
                p = self.buck_params.get((e1, e2))
                if p is not None:
                    A_mat[i, j] = p['A']
                    F_mat[i, j] = p['F']
                    C_mat[i, j] = p['C']
                
                R_HARD_MAT[i, j] = self.r_hard_by_pair.get(
                    (e1, e2), self.r_hard_default)
        
        type_Z = np.array(
            [self.zbl_z[TYPE_TO_ELEM[i]] for i in range(nt)],
            dtype=np.float64)
        
        return {
            'A_mat': A_mat,
            'F_mat': F_mat,
            'C_mat': C_mat,
            'R_HARD_MAT': R_HARD_MAT,
            'type_Z': type_Z,
        }